"""
Tests for FundMover — 12 scenarios.

Tests are grouped by:
  A. Rail routing (USD, EUR, AED)
  B. Happy path
  C. Idempotency / state machine
  D. BalanceTracker
  E. SLA breach
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest
from freezegun import freeze_time

from agents.operations.fund_mover import (
    AEDCutoffPassed,
    BankAPI,
    BalanceTracker,
    ExecutionAlreadyFailed,
    ExecutionState,
    FundMover,
    FundMoverConfig,
    InMemoryExecutionStore,
    InsufficientFunds,
    MockBankAPI,
    SLABreached,
    SubmitUnknownError,
    TransferExecution,
)


# ── Minimal stub for FundMovementProposal ─────────────────────────────────────

@dataclass
class _Proposal:
    id:                 str
    currency:           str
    amount:             float
    source_account:     str = "OPS-USD-001"
    destination_nostro: str = "NOSTRO-USD-001"
    rail:               str = "fedwire"


def _proposal(
    id: str = "P-001",
    currency: str = "USD",
    amount: float = 1000.0,
) -> _Proposal:
    return _Proposal(id=id, currency=currency, amount=amount)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_mover(
    bank: MockBankAPI | None = None,
    balances: dict | None = None,
    config: FundMoverConfig | None = None,
    calendar_svc=None,
) -> tuple[FundMover, MockBankAPI, InMemoryExecutionStore, BalanceTracker]:
    bank    = bank or MockBankAPI()
    store   = InMemoryExecutionStore()
    tracker = BalanceTracker(
        {k: Decimal(v) for k, v in (balances or {"USD": 100_000, "EUR": 100_000, "GBP": 100_000, "AED": 100_000}).items()}
    )
    cfg = config or FundMoverConfig(poll_interval_sec=0.0)
    mover = FundMover(
        bank_api=bank,
        store=store,
        balance_tracker=tracker,
        calendar_svc=calendar_svc,
        config=cfg,
    )
    return mover, bank, store, tracker


# ── A. Rail Routing ───────────────────────────────────────────────────────────

class TestRailRouting:

    async def test_usd_uses_fedwire(self):
        mover, bank, store, _ = _make_mover()
        p = _proposal(currency="USD", amount=5000.0)
        execution = await mover.execute_proposal(p)

        assert execution.rail == "fedwire"
        assert bank.submit_calls[0]["rail"] == "fedwire"

    async def test_eur_below_threshold_uses_sepa_instant(self):
        """EUR < 100,000 → SEPA Instant."""
        mover, bank, store, _ = _make_mover()
        p = _proposal(currency="EUR", amount=50_000.0)
        execution = await mover.execute_proposal(p)

        assert execution.rail == "sepa_instant"
        assert bank.submit_calls[0]["rail"] == "sepa_instant"

    async def test_eur_at_threshold_uses_sepa(self):
        """EUR >= 100,000 → SEPA."""
        mover, bank, store, _ = _make_mover()
        p = _proposal(currency="EUR", amount=100_000.0)
        execution = await mover.execute_proposal(p)

        assert execution.rail == "sepa"
        assert bank.submit_calls[0]["rail"] == "sepa"

    @freeze_time("2026-02-23 09:00:00+00:00")  # 09:00 UTC = 13:00 GST (before 14:00 cutoff)
    async def test_aed_before_cutoff_uses_bank_desk(self):
        """AED before 2pm GST (Dubai UTC+4) on a weekday → bank_desk rail."""
        mover, bank, store, _ = _make_mover()
        p = _proposal(currency="AED", amount=5000.0)
        execution = await mover.execute_proposal(p)

        assert execution.rail == "bank_desk"

    @freeze_time("2026-02-23 10:30:00+00:00")  # 10:30 UTC = 14:30 GST (after cutoff)
    async def test_aed_after_cutoff_raises(self):
        """AED at or after 2pm GST → AEDCutoffPassed."""
        mover, _, _, _ = _make_mover()
        p = _proposal(currency="AED", amount=5000.0)
        with pytest.raises(AEDCutoffPassed):
            await mover.execute_proposal(p)

    async def test_unknown_currency_raises_value_error(self):
        mover, _, _, _ = _make_mover()
        p = _proposal(currency="JPY", amount=1000.0)
        with pytest.raises(ValueError, match="No rail configured for currency JPY"):
            await mover.execute_proposal(p)


# ── B. Happy Path ─────────────────────────────────────────────────────────────

class TestHappyPath:

    async def test_usd_submit_poll_confirmed(self):
        """Full happy path: submit → poll once → CONFIRMED."""
        mover, bank, store, tracker = _make_mover()
        p = _proposal(currency="USD", amount=1000.0)

        execution = await mover.execute_proposal(p)

        assert execution.state == ExecutionState.CONFIRMED
        assert execution.bank_ref == f"MOCK-INST-{p.id}"
        assert execution.confirmed_at is not None
        assert execution.settled_amount is not None
        assert len(bank.submit_calls) == 1

    async def test_instruction_id_is_stable(self):
        """instruction_id must be f'INST-{proposal_id}' for idempotency."""
        mover, bank, _, _ = _make_mover()
        p = _proposal(id="P-STABLE", currency="USD", amount=500.0)
        execution = await mover.execute_proposal(p)

        assert execution.instruction_id == "INST-P-STABLE"
        assert bank.submit_calls[0]["instruction_id"] == "INST-P-STABLE"

    async def test_amount_stored_as_decimal(self):
        """Float amount from proposal is converted to Decimal in execution."""
        mover, _, store, _ = _make_mover()
        p = _proposal(currency="USD", amount=1234.56)
        execution = await mover.execute_proposal(p)

        assert isinstance(execution.amount, Decimal)
        assert execution.amount == Decimal("1234.56")

    async def test_balance_deducted_on_confirm(self):
        """BalanceTracker deducts balance after confirmation."""
        mover, _, _, tracker = _make_mover(balances={"USD": 10_000})
        p = _proposal(currency="USD", amount=3000.0)

        assert tracker.available("USD") == Decimal("10000")
        await mover.execute_proposal(p)
        # After confirm: balance reduced, reservation cleared
        assert tracker.available("USD") == Decimal("7000")


# ── C. Idempotency / State Machine ────────────────────────────────────────────

class TestIdempotency:

    async def test_already_confirmed_returns_immediately(self):
        """Calling execute_proposal on a CONFIRMED execution skips all bank calls."""
        mover, bank, store, _ = _make_mover()
        p = _proposal(currency="USD", amount=500.0)

        first  = await mover.execute_proposal(p)
        assert first.state == ExecutionState.CONFIRMED
        assert len(bank.submit_calls) == 1

        # Second call — must not call bank again
        second = await mover.execute_proposal(p)
        assert second.state == ExecutionState.CONFIRMED
        assert len(bank.submit_calls) == 1   # still 1

    async def test_already_submitted_polls_without_resubmit(self):
        """
        If execution is SUBMITTED (e.g., caller crashed after submit), a second
        call to execute_proposal must NOT submit again — only poll.
        """
        mover, bank, store, _ = _make_mover()
        p = _proposal(currency="USD", amount=500.0)

        # Manually create execution in SUBMITTED state
        execution = TransferExecution(
            execution_id       = "exec-001",
            proposal_id        = p.id,
            instruction_id     = f"INST-{p.id}",
            currency           = "USD",
            amount             = Decimal("500.00"),
            source_account     = "OPS-USD-001",
            destination_nostro = "NOSTRO-USD-001",
            rail               = "fedwire",
            state              = ExecutionState.SUBMITTED,
            bank_ref           = "BANK-REF-001",
            submitted_at       = datetime.now(timezone.utc),
        )
        store.save(execution)
        # Pre-reserve so confirm can deduct
        await mover._tracker.reserve("USD", Decimal("500.00"))

        result = await mover.execute_proposal(p)

        assert result.state == ExecutionState.CONFIRMED
        assert len(bank.submit_calls) == 0   # no submit — poll only

    async def test_submit_unknown_fence_finds_record(self):
        """
        If previous call ended in SUBMIT_UNKNOWN, and bank has the record,
        execute_proposal must NOT submit again — just advance to CONFIRMED.
        """
        mover, bank, store, _ = _make_mover()
        bank.submit_unknown_once = True   # first submit: raises but bank records it

        p = _proposal(currency="USD", amount=500.0)

        # First call → SubmitUnknownError
        with pytest.raises(SubmitUnknownError):
            await mover.execute_proposal(p)

        assert len(bank.submit_calls) == 1
        execution = store.get_by_proposal_id(p.id)
        assert execution.state == ExecutionState.SUBMIT_UNKNOWN

        # Second call → fence finds record → poll → CONFIRMED (no new submit)
        result = await mover.execute_proposal(p)

        assert result.state == ExecutionState.CONFIRMED
        assert len(bank.submit_calls) == 1   # still only 1 submit

    async def test_submit_unknown_fence_not_found_resubmits(self):
        """
        If previous call ended in SUBMIT_UNKNOWN, and bank has NO record,
        execute_proposal resubmits (one new submit call).
        """
        # First call will raise; bank does NOT record (submit_error path)
        bank = MockBankAPI()
        bank.submit_error = ConnectionError("network down, bank did not record")

        mover, _, store, _ = _make_mover(bank=bank)
        p = _proposal(currency="USD", amount=500.0)

        with pytest.raises(SubmitUnknownError):
            await mover.execute_proposal(p)

        assert store.get_by_proposal_id(p.id).state == ExecutionState.SUBMIT_UNKNOWN

        # Clear error — second call should resubmit and succeed
        bank.submit_error = None
        result = await mover.execute_proposal(p)

        assert result.state == ExecutionState.CONFIRMED
        assert len(bank.submit_calls) == 2   # original failed + new successful submit

    async def test_failed_execution_raises(self):
        """Calling execute_proposal on a FAILED execution raises ExecutionAlreadyFailed."""
        mover, bank, store, _ = _make_mover(
            config=FundMoverConfig(
                poll_interval_sec=0.0,
                rail_sla_overrides={"fedwire": 0},  # instant expiry
            )
        )
        bank.settle_after_polls = 999   # never settles

        p = _proposal(currency="USD", amount=500.0)

        with pytest.raises(SLABreached):
            await mover.execute_proposal(p)

        assert store.get_by_proposal_id(p.id).state == ExecutionState.FAILED

        with pytest.raises(ExecutionAlreadyFailed):
            await mover.execute_proposal(p)


# ── D. BalanceTracker ─────────────────────────────────────────────────────────

class TestBalanceTracker:

    async def test_insufficient_funds_raises(self):
        """execute_proposal raises InsufficientFunds when balance is too low."""
        mover, _, _, _ = _make_mover(balances={"USD": 500})
        p = _proposal(currency="USD", amount=1000.0)

        with pytest.raises(InsufficientFunds) as exc_info:
            await mover.execute_proposal(p)

        assert exc_info.value.currency == "USD"
        assert exc_info.value.available == Decimal("500")
        assert exc_info.value.required  == Decimal("1000.00")

    async def test_concurrent_double_spend_prevention(self):
        """
        Two concurrent proposals for the same currency cannot both exceed the balance.
        One should succeed (CONFIRMED), the other should fail (InsufficientFunds).
        """
        mover, bank, store, tracker = _make_mover(balances={"USD": 1500})
        bank.settle_after_polls = 1

        p1 = _proposal(id="P-A", currency="USD", amount=1000.0)
        p2 = _proposal(id="P-B", currency="USD", amount=1000.0)

        results = await asyncio.gather(
            mover.execute_proposal(p1),
            mover.execute_proposal(p2),
            return_exceptions=True,
        )

        successes  = [r for r in results if isinstance(r, TransferExecution)]
        failures   = [r for r in results if isinstance(r, InsufficientFunds)]

        assert len(successes) == 1
        assert len(failures)  == 1

    async def test_balance_released_on_sla_breach(self):
        """On SLA breach, reserved funds are returned to available balance."""
        mover, bank, store, tracker = _make_mover(
            balances={"USD": 5000},
            config=FundMoverConfig(
                poll_interval_sec=0.0,
                rail_sla_overrides={"fedwire": 0},
            ),
        )
        bank.settle_after_polls = 999

        p = _proposal(currency="USD", amount=1000.0)

        # Before: available = 5000
        assert tracker.available("USD") == Decimal("5000")

        with pytest.raises(SLABreached):
            await mover.execute_proposal(p)

        # After breach: funds released, available back to 5000
        assert tracker.available("USD") == Decimal("5000")


# ── E. SLA Breach ─────────────────────────────────────────────────────────────

class TestSLABreach:

    async def test_sla_breach_raises_and_marks_failed(self):
        """If bank never settles within SLA, SLABreached is raised and state = FAILED."""
        mover, bank, store, _ = _make_mover(
            config=FundMoverConfig(
                poll_interval_sec=0.0,
                rail_sla_overrides={"fedwire": 0},
            )
        )
        bank.settle_after_polls = 999   # never settles in time

        p = _proposal(currency="USD", amount=500.0)

        with pytest.raises(SLABreached) as exc_info:
            await mover.execute_proposal(p)

        assert exc_info.value.execution.state == ExecutionState.FAILED
        assert store.get_by_proposal_id(p.id).state == ExecutionState.FAILED
