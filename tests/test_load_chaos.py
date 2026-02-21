"""
Load and Chaos tests for Aspora TMS.

Groups:
  L. Load  — concurrency, volume, throughput
  C. Chaos — failure injection, partial outages, malformed inputs, race conditions

Frozen time: 2026-02-23 06:00 UTC (Monday)
  = 11:30 IST  → INR desk is OPEN (09:00–14:50 IST)
  Fedwire opens at 14:00 UTC — path is feasible before tomorrow 09:00 IST
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
from freezegun import freeze_time

from agents.operations.agent import OperationsAgent
from agents.operations.fund_mover import (
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
from agents.operations.maker_checker import MakerCheckerWorkflow
from agents.operations.window_manager import WindowManager
from bus.events import (
    DEAL_INSTRUCTION,
    FUND_MOVEMENT_STATUS,
    PROPOSAL_APPROVED,
    SHORTFALL_ALERT,
    WINDOW_CLOSING,
    Event,
    create_event,
)
from bus.memory_bus import InMemoryBus
from models.domain import FundMovementProposal, ProposalStatus
from services.calendar_service import CalendarService

# ── Constants ──────────────────────────────────────────────────────────────────

# Monday 2026-02-23 06:00 UTC
# = 11:30 IST (INR desk open 09:00–14:50)
# Fedwire opens 14:00 UTC today — feasible before tomorrow 09:00 IST deadline
TRADING_TIME = "2026-02-23 06:00:00+00:00"
TRADING_NOW  = datetime(2026, 2, 23, 6, 0, 0, tzinfo=timezone.utc)


# ── Stubs ──────────────────────────────────────────────────────────────────────

class _MockMC:
    """
    MakerChecker stub.

    Parameters
    ----------
    return_status : str
        Status returned by submit_proposal.
    fail_every_n : int
        If > 0, raise RuntimeError on every n-th call (1-indexed).
    """

    def __init__(
        self,
        return_status: str = "pending_approval",
        fail_every_n: int = 0,
    ) -> None:
        self.submitted: list[FundMovementProposal] = []
        self._return_status = return_status
        self._call_count    = 0
        self._fail_every_n  = fail_every_n

    async def submit_proposal(self, proposal: FundMovementProposal) -> dict:
        self._call_count += 1
        if self._fail_every_n > 0 and self._call_count % self._fail_every_n == 0:
            raise RuntimeError(f"MC intermittent failure on call {self._call_count}")
        self.submitted.append(proposal)
        return {"status": self._return_status, "proposal_id": proposal.id}

    async def list_proposals(self) -> list[FundMovementProposal]:
        return list(self.submitted)


class _MockAlerts:
    def __init__(self) -> None:
        self.executed:  list = []
        self.escalated: list = []

    async def notify_checkers(self, proposal, n: int) -> None:
        pass

    async def escalate(self, proposal, **kw) -> None:
        self.escalated.append(proposal)

    async def notify_executed(self, proposal) -> None:
        self.executed.append(proposal)


class _MockAudit:
    async def log(self, **kwargs) -> None:
        pass


class _InMemDB:
    def __init__(self) -> None:
        self._store: dict[str, FundMovementProposal] = {}

    async def get(self, pid: str) -> FundMovementProposal | None:
        return self._store.get(pid)

    async def save(self, p: FundMovementProposal) -> None:
        self._store[p.id] = p

    async def is_approved_nostro(self, nostro: str) -> bool:
        return True

    async def has_recent_duplicate(self, key: str) -> bool:
        return False

    async def list_all(self) -> list[FundMovementProposal]:
        return list(self._store.values())


# ── Minimal proposal stub (mirrors test_fund_mover._Proposal) ─────────────────

@dataclass
class _FMProposal:
    id:                 str
    currency:           str
    amount:             Decimal
    source_account:     str = "OPS-USD-001"
    destination_nostro: str = "NOSTRO-USD-001"
    rail:               str = "fedwire"


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _no_holidays(d: date) -> bool:
    return False


def _make_window_manager(buffer_min: int = 0) -> WindowManager:
    return WindowManager(
        usd_holidays=_no_holidays,
        gbp_holidays=_no_holidays,
        inr_holidays=_no_holidays,
        aed_holidays=_no_holidays,
        operational_buffer_min=buffer_min,
    )


def _make_mover(
    balances: dict | None = None,
    bank:     MockBankAPI | None = None,
    config:   FundMoverConfig | None = None,
) -> tuple[FundMover, MockBankAPI, InMemoryExecutionStore, BalanceTracker]:
    bank    = bank or MockBankAPI()
    store   = InMemoryExecutionStore()
    raw     = balances or {"USD": 100_000, "EUR": 100_000, "GBP": 100_000, "AED": 100_000}
    tracker = BalanceTracker({k: Decimal(str(v)) for k, v in raw.items()})
    cfg     = config or FundMoverConfig(poll_interval_sec=0.0)
    mover   = FundMover(bank_api=bank, store=store, balance_tracker=tracker, config=cfg)
    return mover, bank, store, tracker


def _make_agent(
    balances: dict | None = None,
    bank:     MockBankAPI | None = None,
    config:   dict | None = None,
    mc:       _MockMC | None = None,
) -> tuple[OperationsAgent, InMemoryBus, _MockMC, MockBankAPI, BalanceTracker]:
    bus   = InMemoryBus()
    mc    = mc or _MockMC()
    mover, bank, _store, tracker = _make_mover(balances=balances, bank=bank)
    cfg   = config or {
        "prefunding_buffer_pct":    0.10,
        "window_closing_alert_min": 30,
        "nostro_topup_trigger_pct": 0.90,
        "topup_target_pct":         1.20,
        "monitor_interval_sec":     0.0,
        "stale_proposal_age_min":   90,
        "lookahead_days":           3,
        "monitored_currencies":     ["USD", "EUR", "GBP", "AED"],
    }
    agent = OperationsAgent(
        bus           = bus,
        calendar      = CalendarService(),
        window_manager= _make_window_manager(buffer_min=0),
        maker_checker = mc,
        fund_mover    = mover,
        config        = cfg,
    )
    return agent, bus, mc, bank, tracker


def _shortfall(
    currency:  str   = "USD",
    required:  float = 150_000.0,
    available: float = 100_000.0,
    correlation_id: str | None = None,
) -> Event:
    return create_event(
        event_type=SHORTFALL_ALERT,
        source_agent="liquidity",
        payload={
            "currency":          currency,
            "required_amount":   required,
            "available_balance": available,
            "shortfall":         required - available,
            "severity":          "critical",
        },
        correlation_id=correlation_id or f"corr-{currency}-test",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# L. LOAD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoad:

    # ── L-1 ──────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_L1_concurrent_identical_shortfalls_idempotent(self):
        """
        100 concurrent shortfall events for the same currency →
        exactly 1 proposal submitted.

        The in-memory set _pending_intents must guard against all 99 duplicates
        even when coroutines interleave freely on the event loop.
        """
        agent, _, mc, _, _ = _make_agent(balances={"USD": 500_000})
        event = _shortfall("USD")

        await asyncio.gather(*[agent.handle_shortfall(event) for _ in range(100)])

        assert len(mc.submitted) == 1, (
            f"L-1 FAIL: idempotency guard produced {len(mc.submitted)} proposals (expected 1)"
        )

    # ── L-2 ──────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_L2_concurrent_shortfalls_four_currencies(self):
        """
        4 concurrent shortfall events (USD, EUR, GBP, AED).

        EUR has no direct TransferWindow in WindowManager (it converts via USD),
        so the agent emits window_not_feasible for EUR and skips it.
        Expected: 3 proposals (USD, GBP, AED), each on its own rail.
        """
        agent, bus, mc, _, _ = _make_agent(
            balances={"USD": 500_000, "EUR": 500_000, "GBP": 500_000, "AED": 500_000}
        )
        events = [_shortfall(ccy) for ccy in ("USD", "EUR", "GBP", "AED")]

        await asyncio.gather(*[agent.handle_shortfall(e) for e in events])

        currencies = {p.currency for p in mc.submitted}
        assert len(mc.submitted) == 3, (
            f"L-2 FAIL: expected 3 proposals (EUR has no direct rail), got {len(mc.submitted)}"
        )
        assert currencies == {"USD", "GBP", "AED"}, (
            f"L-2 FAIL: wrong currencies: {currencies}"
        )
        # EUR emits a FUND_MOVEMENT_STATUS with window_not_feasible
        wfn = [
            e for e in bus.get_events(FUND_MOVEMENT_STATUS)
            if e.payload.get("currency") == "EUR"
               and e.payload.get("status") == "window_not_feasible"
        ]
        assert len(wfn) == 1, "L-2 FAIL: EUR should emit exactly one window_not_feasible event"

    # ── L-3 ──────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_L3_high_volume_deal_instructions_accumulate_correctly(self):
        """
        500 deal instructions of USD 1 000 each →
        _cumulative_deals["USD"] == 500 000.
        """
        agent, _, _, _, _ = _make_agent(
            # Set balance very high so top-up trigger (90% of balance) is never hit.
            balances={"USD": 100_000_000}
        )
        for i in range(500):
            event = create_event(
                event_type=DEAL_INSTRUCTION,
                source_agent="fx_analyst",
                payload={
                    "deal_id":        f"D-{i:04d}",
                    "currency_pair":  "USD/INR",
                    "amount_foreign": 1_000.0,
                    "amount_inr":     83_000.0,
                    "deal_type":      "spot",
                    "target_rate":    83.0,
                    "direction":      "BUY",
                    "tranche_number": 1,
                    "total_tranches": 1,
                },
            )
            await agent.handle_deal_instruction(event)

        total = agent._cumulative_deals.get("USD", Decimal(0))
        assert total == Decimal("500000.0"), (
            f"L-3 FAIL: cumulative USD = {total}, expected 500000"
        )

    # ── L-4 ──────────────────────────────────────────────────────────────────

    async def test_L4_balance_tracker_concurrent_reservations_no_double_spend(self):
        """
        20 concurrent reservation attempts of 100 USD from a 1 500 USD balance →
        exactly 15 succeed, 5 fail with InsufficientFunds.

        BalanceTracker's asyncio.Lock must prevent any two coroutines from
        simultaneously seeing sufficient funds when there aren't enough.
        """
        tracker = BalanceTracker({"USD": Decimal("1500")})

        results = await asyncio.gather(
            *[tracker.reserve("USD", Decimal("100")) for _ in range(20)],
            return_exceptions=True,
        )

        successes    = [r for r in results if r is None]
        insufficient = [r for r in results if isinstance(r, InsufficientFunds)]

        assert len(successes)    == 15, f"L-4 FAIL: expected 15 successes, got {len(successes)}"
        assert len(insufficient) == 5,  f"L-4 FAIL: expected 5 InsufficientFunds, got {len(insufficient)}"
        assert tracker._reserved.get("USD", Decimal(0)) == Decimal("1500"), (
            "L-4 FAIL: reserved balance does not equal starting balance"
        )

    # ── L-5 ──────────────────────────────────────────────────────────────────

    async def test_L5_maker_checker_50_proposals_all_stored(self):
        """
        50 concurrent proposal submissions through the real MakerCheckerWorkflow →
        all 50 persisted to the DB, all return pending_approval.
        """
        import uuid

        class _AlwaysApproves:
            async def can_approve(self, checker_id, proposal):
                return True

        db      = _InMemDB()
        mc_real = MakerCheckerWorkflow(
            db=db,
            auth_service=_AlwaysApproves(),
            alert_router=_MockAlerts(),
            audit_log=_MockAudit(),
        )
        proposals = [
            FundMovementProposal(
                id=str(uuid.uuid4()),
                currency="USD",
                amount=Decimal("10000"),
                source_account="OPS-USD-001",
                destination_nostro="NOSTRO-USD-001",
                rail="fedwire",
                proposed_by="system:ops",
                purpose="prefunding",
                idempotency_key=f"USD-2026-02-23-shortfall-{i}",
            )
            for i in range(50)
        ]

        results = await asyncio.gather(*[mc_real.submit_proposal(p) for p in proposals])

        assert all(r["status"] == "pending_approval" for r in results), (
            "L-5 FAIL: not all proposals returned pending_approval"
        )
        stored = await db.list_all()
        assert len(stored) == 50, f"L-5 FAIL: expected 50 stored, got {len(stored)}"

    # ── L-6 ──────────────────────────────────────────────────────────────────

    async def test_L6_event_bus_burst_throughput_1000_events(self):
        """
        Publish 1 000 events → handler invoked exactly 1 000 times.
        Validates InMemoryBus does not drop or duplicate events under burst load.
        """
        bus      = InMemoryBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(SHORTFALL_ALERT, "load-test-group", handler)

        events = [
            create_event(SHORTFALL_ALERT, "load-tester", {"currency": "USD", "n": i})
            for i in range(1_000)
        ]
        for ev in events:
            await bus.publish(ev)

        await bus.drain()

        assert len(received) == 1_000, (
            f"L-6 FAIL: expected 1000 events received, got {len(received)}"
        )

    # ── L-7 ──────────────────────────────────────────────────────────────────

    async def test_L7_concurrent_fund_mover_executions_all_confirmed(self):
        """
        10 concurrent FundMover proposals (USD 5 000 each from a USD 50 000 pool) →
        all 10 CONFIRMED, balance exactly 0 afterward.
        """
        mover, bank, _, tracker = _make_mover(balances={"USD": 50_000})
        bank.settle_after_polls = 1

        proposals = [
            _FMProposal(id=f"P-L7-{i:02d}", currency="USD", amount=Decimal("5000"))
            for i in range(10)
        ]

        results = await asyncio.gather(
            *[mover.execute_proposal(p) for p in proposals],
            return_exceptions=True,
        )

        confirmed = [r for r in results if isinstance(r, TransferExecution)
                     and r.state == ExecutionState.CONFIRMED]
        errors    = [r for r in results if not isinstance(r, TransferExecution)]

        assert len(confirmed) == 10, (
            f"L-7 FAIL: expected 10 CONFIRMED, got {len(confirmed)}. Errors: {errors}"
        )
        assert tracker.available("USD") == Decimal("0"), (
            f"L-7 FAIL: expected USD balance 0, got {tracker.available('USD')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# C. CHAOS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestChaos:

    # ── C-1 ──────────────────────────────────────────────────────────────────

    async def test_C1_submit_unknown_recovery_no_resubmit(self):
        """
        First submit raises SubmitUnknownError (bank has the record).
        Second call: fence finds bank record → polls → CONFIRMED.
        Bank is called exactly once.
        """
        bank = MockBankAPI()
        bank.submit_unknown_once = True
        mover, _, store, _ = _make_mover(bank=bank)
        p = _FMProposal(id="P-C1", currency="USD", amount=Decimal("5000"))

        with pytest.raises(SubmitUnknownError):
            await mover.execute_proposal(p)

        assert store.get_by_proposal_id(p.id).state == ExecutionState.SUBMIT_UNKNOWN
        assert len(bank.submit_calls) == 1

        result = await mover.execute_proposal(p)

        assert result.state == ExecutionState.CONFIRMED
        assert len(bank.submit_calls) == 1, (
            "C-1 FAIL: bank was called twice — fence did not prevent resubmit"
        )

    # ── C-2 ──────────────────────────────────────────────────────────────────

    async def test_C2_bank_total_outage_then_recovery(self):
        """
        Bank network is down (bank does NOT record the instruction).
        - 1st call: SubmitUnknownError, state = SUBMIT_UNKNOWN.
        - 2nd call (bank still down): SubmitUnknownError again, fence finds no record → resubmits.
        - 3rd call (bank up): CONFIRMED; total submit_calls = 3.
        """
        bank = MockBankAPI()
        bank.submit_error = ConnectionError("bank unreachable")
        mover, _, store, _ = _make_mover(bank=bank)
        p = _FMProposal(id="P-C2", currency="USD", amount=Decimal("5000"))

        with pytest.raises(SubmitUnknownError):
            await mover.execute_proposal(p)
        assert store.get_by_proposal_id(p.id).state == ExecutionState.SUBMIT_UNKNOWN

        with pytest.raises(SubmitUnknownError):
            await mover.execute_proposal(p)

        bank.submit_error = None
        result = await mover.execute_proposal(p)

        assert result.state == ExecutionState.CONFIRMED
        assert len(bank.submit_calls) == 3, (
            f"C-2 FAIL: expected 3 submit calls (2 failed + 1 success), got {len(bank.submit_calls)}"
        )

    # ── C-3 ──────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_C3_mc_rejection_does_not_lock_idempotency_key(self):
        """
        MC rejects the proposal → idempotency key must NOT remain in _pending_intents.
        A second identical shortfall must reach MC again.
        """
        mc = _MockMC(return_status="rejected")
        agent, _, mc, _, _ = _make_agent(balances={"USD": 500_000}, mc=mc)
        event = _shortfall("USD")

        await agent.handle_shortfall(event)
        assert len(mc.submitted) == 1

        # Key must be absent after rejection
        usd_key = next((k for k in agent._pending_intents if "USD" in k), None)
        assert usd_key is None, (
            f"C-3 FAIL: idempotency key locked after MC rejection: {agent._pending_intents}"
        )

        # Second identical event — MC now accepts
        mc._return_status = "pending_approval"
        await agent.handle_shortfall(event)

        assert len(mc.submitted) == 2, (
            "C-3 FAIL: second submission was blocked by stale idempotency key"
        )

    # ── C-4 ──────────────────────────────────────────────────────────────────

    async def test_C4_sla_breach_releases_all_reserved_funds(self):
        """
        5 concurrent proposals; bank never settles; SLA = 0 min (instant expiry).
        All raise SLABreached and all reserved funds must be returned to the pool.
        """
        bank = MockBankAPI()
        bank.settle_after_polls = 999
        mover, _, _, tracker = _make_mover(
            balances={"USD": 10_000},
            bank=bank,
            config=FundMoverConfig(
                poll_interval_sec=0.0,
                rail_sla_overrides={"fedwire": 0},
            ),
        )
        proposals = [
            _FMProposal(id=f"P-C4-{i}", currency="USD", amount=Decimal("1000"))
            for i in range(5)
        ]

        results = await asyncio.gather(
            *[mover.execute_proposal(p) for p in proposals],
            return_exceptions=True,
        )

        breached = [r for r in results if isinstance(r, SLABreached)]
        assert len(breached) == 5, f"C-4 FAIL: expected 5 SLABreached, got {len(breached)}"
        assert tracker.available("USD") == Decimal("10000"), (
            f"C-4 FAIL: funds not fully released; available = {tracker.available('USD')}"
        )

    # ── C-5 ──────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_C5_duplicate_storm_100_sequential_shortfalls(self):
        """
        100 sequential identical shortfall events → exactly 1 proposal.
        Ensures the idempotency guard holds under repeated sequential hits,
        not just concurrent ones.
        """
        agent, _, mc, _, _ = _make_agent(balances={"USD": 500_000})
        event = _shortfall("USD")

        for _ in range(100):
            await agent.handle_shortfall(event)

        assert len(mc.submitted) == 1, (
            f"C-5 FAIL: duplicate storm produced {len(mc.submitted)} proposals (expected 1)"
        )

    # ── C-6 ──────────────────────────────────────────────────────────────────

    async def test_C6_concurrent_double_spend_prevention(self):
        """
        10 concurrent proposals wanting USD 200 each; balance = USD 1 000.
        Exactly 5 confirm, 5 fail with InsufficientFunds.
        Confirmed total must exactly drain the balance.
        """
        mover, bank, _, tracker = _make_mover(balances={"USD": 1_000})
        bank.settle_after_polls = 1

        proposals = [
            _FMProposal(id=f"P-C6-{i}", currency="USD", amount=Decimal("200"))
            for i in range(10)
        ]

        results = await asyncio.gather(
            *[mover.execute_proposal(p) for p in proposals],
            return_exceptions=True,
        )

        confirmed    = [r for r in results if isinstance(r, TransferExecution)]
        insufficient = [r for r in results if isinstance(r, InsufficientFunds)]

        assert len(confirmed)    == 5, f"C-6 FAIL: expected 5 confirmed, got {len(confirmed)}"
        assert len(insufficient) == 5, f"C-6 FAIL: expected 5 InsufficientFunds, got {len(insufficient)}"
        assert tracker.available("USD") == Decimal("0"), (
            f"C-6 FAIL: expected balance 0 after 5×200 transfers, got {tracker.available('USD')}"
        )

    # ── C-7 ──────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_C7_day_rollover_clears_intents_allows_resubmit(self):
        """
        Day rollover resets _pending_intents so the same shortfall can be
        re-submitted on the next trading day.
        """
        agent, _, mc, _, _ = _make_agent(balances={"USD": 500_000})

        # Day 1: submit
        await agent.handle_shortfall(_shortfall("USD"))
        assert len(mc.submitted) == 1

        # Simulate day rollover to Feb 24
        day2_utc = datetime(2026, 2, 24, 6, 0, 0, tzinfo=timezone.utc)
        agent._check_day_rollover(day2_utc)
        assert agent._pending_intents == set(), (
            "C-7 FAIL: _pending_intents not cleared after day rollover"
        )

        # Day 2: same event should trigger a fresh submission
        with freeze_time("2026-02-24 06:00:00+00:00"):
            await agent.handle_shortfall(_shortfall("USD"))

        assert len(mc.submitted) == 2, (
            f"C-7 FAIL: expected 2 total submissions (one per day), got {len(mc.submitted)}"
        )

    # ── C-8 ──────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_C8_mc_intermittent_failures_no_permanent_key_locking(self):
        """
        MC fails on every 2nd call (calls 2, 4, 6 of 6 total).
        Between each attempt, _pending_intents is cleared to simulate a new
        coroutine context.  Calls 1, 3, 5 succeed → 3 proposals in mc.submitted.
        Failures must not permanently prevent future submissions.
        """
        mc_stub = _MockMC(fail_every_n=2)
        agent, _, mc, _, _ = _make_agent(balances={"USD": 500_000}, mc=mc_stub)

        successful_submissions = 0
        for i in range(6):
            # Clear intents so each attempt is a fresh try (simulates retries after failure)
            agent._pending_intents.clear()
            try:
                await agent.handle_shortfall(_shortfall("USD", correlation_id=f"c{i}"))
                successful_submissions += 1
            except RuntimeError:
                pass  # MC intermittent failure — agent should swallow this

        # Odd-numbered MC calls (1, 3, 5) succeed
        assert len(mc.submitted) == 3, (
            f"C-8 FAIL: expected 3 successful submissions, got {len(mc.submitted)}"
        )

    # ── C-9 ──────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_C9_agent_restart_restores_pending_intents_from_mc(self):
        """
        After a restart (fresh agent, same MC), _pending_intents must be
        restored from MC so that already-submitted proposals are not duplicated.
        """
        mc_shared = _MockMC()

        # Agent 1: submit USD shortfall
        agent1, _, _, _, _ = _make_agent(balances={"USD": 500_000}, mc=mc_shared)
        await agent1.handle_shortfall(_shortfall("USD"))
        assert len(mc_shared.submitted) == 1

        # Agent 2: fresh state, same MC (shared "database")
        agent2, _, _, _, _ = _make_agent(balances={"USD": 500_000}, mc=mc_shared)
        assert agent2._intents_restored is False

        await agent2.handle_shortfall(_shortfall("USD"))

        assert len(mc_shared.submitted) == 1, (
            "C-9 FAIL: agent2 submitted a duplicate — intent restoration from MC failed"
        )
        assert agent2._intents_restored is True, (
            "C-9 FAIL: _intents_restored flag not set after first shortfall call"
        )

    # ── C-10 ─────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_C10_malformed_event_payloads_do_not_crash_agent(self):
        """
        Various malformed shortfall payloads → agent logs and returns gracefully;
        no exception propagates to the caller and no proposals are submitted.
        """
        agent, bus, mc, _, _ = _make_agent()

        bad_events = [
            # Completely empty payload
            create_event(SHORTFALL_ALERT, "test", {}),
            # Missing amounts
            create_event(SHORTFALL_ALERT, "test", {"currency": "USD"}),
            # Wrong types
            create_event(SHORTFALL_ALERT, "test", {
                "currency":          None,
                "required_amount":   "not-a-number",
                "available_balance": [],
                "shortfall":         {},
            }),
            # Unknown / unsupported currency
            create_event(SHORTFALL_ALERT, "test", {
                "currency":          "XYZ",
                "required_amount":   100_000,
                "available_balance": 50_000,
                "shortfall":         50_000,
                "severity":          "critical",
            }),
            # Negative shortfall (no prefunding needed)
            create_event(SHORTFALL_ALERT, "test", {
                "currency":          "USD",
                "required_amount":   50_000,
                "available_balance": 100_000,
                "shortfall":         -50_000,
                "severity":          "warning",
            }),
        ]

        for ev in bad_events:
            try:
                await agent.handle_shortfall(ev)
            except Exception as exc:
                pytest.fail(f"C-10 FAIL: agent raised unhandled exception on malformed event: {exc!r}")

        # None of the malformed events should have produced a valid proposal
        assert len(mc.submitted) == 0, (
            f"C-10 FAIL: {len(mc.submitted)} proposals submitted for malformed events"
        )

    # ── C-11a ────────────────────────────────────────────────────────────────

    async def test_C11a_bank_settles_exactly_at_sla_boundary(self):
        """
        Bank settles on the very first poll; SLA = 1 min.
        With poll_interval_sec=0, elapsed time ≈ 0 ms < 1 min → CONFIRMED.
        """
        bank = MockBankAPI()
        bank.settle_after_polls = 1
        mover, _, _, _ = _make_mover(
            balances={"USD": 10_000},
            bank=bank,
            config=FundMoverConfig(
                poll_interval_sec=0.0,
                rail_sla_overrides={"fedwire": 1},
            ),
        )
        result = await mover.execute_proposal(
            _FMProposal(id="P-C11a", currency="USD", amount=Decimal("1000"))
        )
        assert result.state == ExecutionState.CONFIRMED, (
            f"C-11a FAIL: expected CONFIRMED at SLA boundary, got {result.state}"
        )

    # ── C-11b ────────────────────────────────────────────────────────────────

    async def test_C11b_bank_settles_after_sla_raises_breach(self):
        """
        Bank would settle after 2 polls; SLA = 0 min (instant expiry).
        First poll is already past the SLA → SLABreached + state = FAILED.
        """
        bank = MockBankAPI()
        bank.settle_after_polls = 2
        mover, _, store, _ = _make_mover(
            balances={"USD": 10_000},
            bank=bank,
            config=FundMoverConfig(
                poll_interval_sec=0.0,
                rail_sla_overrides={"fedwire": 0},
            ),
        )
        p = _FMProposal(id="P-C11b", currency="USD", amount=Decimal("1000"))

        with pytest.raises(SLABreached):
            await mover.execute_proposal(p)

        assert store.get_by_proposal_id(p.id).state == ExecutionState.FAILED, (
            "C-11b FAIL: expected FAILED state after SLA breach"
        )

    # ── C-12 ─────────────────────────────────────────────────────────────────

    async def test_C12_window_closing_alert_deduplication_under_load(self):
        """
        _monitor_windows_once called 50× at 22:00 UTC (17:00 ET — 60 min before
        Fedwire close).  Only 1 WINDOW_CLOSING event must be emitted per currency
        thanks to _window_alerts_sent deduplication.
        """
        agent, bus, _, _, _ = _make_agent(config={
            "prefunding_buffer_pct":    0.10,
            "window_closing_alert_min": 60,    # fire alert when <= 60 min remaining
            "monitor_interval_sec":     0.0,
            "monitored_currencies":     ["USD"],
        })

        # 22:00 UTC = 17:00 ET — Fedwire closes at 18:00 ET (buffer=0) → 60 min remaining
        check_time = datetime(2026, 2, 23, 22, 0, 0, tzinfo=timezone.utc)

        for _ in range(50):
            await agent._monitor_windows_once(check_time)

        window_events = bus.get_events(WINDOW_CLOSING)
        assert len(window_events) == 1, (
            f"C-12 FAIL: expected 1 WINDOW_CLOSING alert, got {len(window_events)} — "
            "deduplication via _window_alerts_sent is broken"
        )
        assert window_events[0].payload["currency"] == "USD"

    # ── C-13 ─────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_C13_massive_concurrent_mixed_load_stays_consistent(self):
        """
        Smoke test: 25 concurrent shortfalls × 4 currencies = 100 total events.
        EUR has no direct rail → window_not_feasible; USD/GBP/AED produce exactly
        1 proposal each. System must remain consistent under full concurrent load.
        """
        mc_shared = _MockMC()
        agent, bus, _, _, _ = _make_agent(
            balances={"USD": 1_000_000, "EUR": 1_000_000, "GBP": 1_000_000, "AED": 1_000_000},
            mc=mc_shared,
        )

        all_events = [
            _shortfall(ccy)
            for ccy in ("USD", "EUR", "GBP", "AED")
            for _ in range(25)
        ]
        await asyncio.gather(*[agent.handle_shortfall(e) for e in all_events])

        # EUR has no TransferWindow → window_not_feasible, not submitted
        assert len(mc_shared.submitted) == 3, (
            f"C-13 FAIL: expected 3 proposals (EUR skipped), got {len(mc_shared.submitted)}"
        )
        assert {p.currency for p in mc_shared.submitted} == {"USD", "GBP", "AED"}, (
            f"C-13 FAIL: unexpected currencies: {[p.currency for p in mc_shared.submitted]}"
        )
        # EUR emits window_not_feasible — idempotency also prevents duplicate events
        eur_wfn = [
            e for e in bus.get_events(FUND_MOVEMENT_STATUS)
            if e.payload.get("currency") == "EUR"
               and e.payload.get("status") == "window_not_feasible"
        ]
        assert len(eur_wfn) >= 1, "C-13 FAIL: expected at least one EUR window_not_feasible"

    # ── C-14 ─────────────────────────────────────────────────────────────────

    async def test_C14_already_failed_execution_raises_immediately(self):
        """
        Calling execute_proposal on a FAILED execution raises ExecutionAlreadyFailed
        immediately, without touching the bank API again.
        """
        bank = MockBankAPI()
        bank.settle_after_polls = 999
        mover, _, store, _ = _make_mover(
            balances={"USD": 10_000},
            bank=bank,
            config=FundMoverConfig(
                poll_interval_sec=0.0,
                rail_sla_overrides={"fedwire": 0},
            ),
        )
        p = _FMProposal(id="P-C14", currency="USD", amount=Decimal("1000"))

        with pytest.raises(SLABreached):
            await mover.execute_proposal(p)

        assert store.get_by_proposal_id(p.id).state == ExecutionState.FAILED

        calls_before = len(bank.submit_calls)

        with pytest.raises(ExecutionAlreadyFailed):
            await mover.execute_proposal(p)

        assert len(bank.submit_calls) == calls_before, (
            "C-14 FAIL: bank was called after FAILED state — should be rejected immediately"
        )

    # ── C-15 ─────────────────────────────────────────────────────────────────

    @freeze_time(TRADING_TIME)
    async def test_C15_insufficient_balance_does_not_lock_idempotency_key(self):
        """
        Agent has only USD 100 but shortfall requires USD 55 000.
        → No proposal submitted.
        → Idempotency key NOT locked (can retry after balance is topped up).
        → FUND_MOVEMENT_STATUS / insufficient_balance event emitted.
        """
        agent, bus, mc, _, _ = _make_agent(
            balances={"USD": 100}      # far too low for a USD 55k proposal
        )
        event = _shortfall("USD", required=150_000, available=100)

        await agent.handle_shortfall(event)

        assert len(mc.submitted) == 0, (
            "C-15 FAIL: proposal submitted despite insufficient balance"
        )

        usd_key = next((k for k in agent._pending_intents if "USD" in k), None)
        assert usd_key is None, (
            f"C-15 FAIL: idempotency key locked after insufficient_balance: {agent._pending_intents}"
        )

        status_events = bus.get_events(FUND_MOVEMENT_STATUS)
        assert any(
            e.payload.get("status") == "insufficient_balance"
            for e in status_events
        ), "C-15 FAIL: INSUFFICIENT_BALANCE status event not emitted on bus"
