"""
Tests for OperationsAgent v2.0 — 12 integration scenarios.

Groups:
  A. run_daily           (2 tests)
  B. handle_shortfall    (4 tests)
  C. handle_proposal_approved (2 tests)
  D. handle_deal_instruction  (2 tests)
  E. _monitor_windows_once    (2 tests)

Time conventions used across tests
-----------------------------------
  MON_14_UTC = 2026-02-23 14:00 UTC  (Mon 09:00 ET  — Fedwire/CHAPS open)
  SAT_14_UTC = 2026-02-21 14:00 UTC  (Sat — all windows closed all week-end)
  MON_2235   = 2026-02-23 22:35 UTC  (Mon 17:35 ET  — 25 min before Fedwire close)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from freezegun import freeze_time

from agents.operations.agent import OperationsAgent
from agents.operations.fund_mover import (
    BalanceTracker,
    FundMover,
    FundMoverConfig,
    InMemoryExecutionStore,
    MockBankAPI,
    SubmitUnknownError,
)
from agents.operations.window_manager import WindowManager
from bus.events import (
    FUND_MOVEMENT_STATUS,
    HOLIDAY_LOOKAHEAD,
    NOSTRO_BALANCE_UPDATE,
    PROPOSAL_APPROVED,
    SHORTFALL_ALERT,
    WINDOW_CLOSING,
    create_event,
)
from bus.memory_bus import InMemoryBus
from models.domain import FundMovementProposal
from services.calendar_service import CalendarService


# ── Stubs ────────────────────────────────────────────────────────────────────


class _MockMC:
    """Minimal MakerChecker stub — records proposals and always approves."""

    def __init__(self) -> None:
        self.submitted: list[FundMovementProposal] = []

    async def submit_proposal(self, proposal: FundMovementProposal) -> dict:
        self.submitted.append(proposal)
        return {"status": "pending_approval", "proposal_id": proposal.id}


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _no_holidays(d):
    return False


def _make_window_manager() -> WindowManager:
    """WindowManager with no holidays and zero operational buffer for determinism."""
    return WindowManager(
        usd_holidays=_no_holidays,
        gbp_holidays=_no_holidays,
        inr_holidays=_no_holidays,
        aed_holidays=_no_holidays,
        operational_buffer_min=0,
    )


def _make_agent(
    balances: dict | None = None,
    bank: MockBankAPI | None = None,
    config: dict | None = None,
) -> tuple[OperationsAgent, InMemoryBus, _MockMC, MockBankAPI, BalanceTracker]:
    bus     = InMemoryBus()
    cal     = CalendarService()
    bank    = bank or MockBankAPI()
    store   = InMemoryExecutionStore()
    tracker = BalanceTracker(
        {k: Decimal(v)
         for k, v in (balances or {"USD": 100_000, "GBP": 100_000, "AED": 100_000}).items()}
    )
    fm  = FundMover(
        bank_api=bank, store=store, balance_tracker=tracker,
        calendar_svc=cal, config=FundMoverConfig(poll_interval_sec=0.0),
    )
    mc  = _MockMC()
    wm  = _make_window_manager()
    cfg = config or {
        "prefunding_buffer_pct":    0.10,
        "window_closing_alert_min": 30,
        "nostro_topup_trigger_pct": 0.90,
        "topup_target_pct":         1.20,
        "monitor_interval_sec":     0.0,
        "stale_proposal_age_min":   90,
        "lookahead_days":           3,
        "monitored_currencies":     ["USD", "GBP", "AED"],
    }
    agent = OperationsAgent(
        bus=bus, calendar=cal, window_manager=wm,
        maker_checker=mc, fund_mover=fm, config=cfg,
    )
    return agent, bus, mc, bank, tracker


def _shortfall_event(
    currency: str = "USD",
    shortfall: float = 10_000.0,
    correlation_id: str | None = None,
) -> object:
    return create_event(
        event_type   = SHORTFALL_ALERT,
        source_agent = "liquidity",
        payload      = {
            "currency":          currency,
            "required_amount":   shortfall * 1.1,
            "available_balance": 0.0,
            "shortfall":         shortfall,
            "severity":          "critical",
        },
        correlation_id=correlation_id,
    )


# ── A. run_daily ──────────────────────────────────────────────────────────────


@freeze_time("2026-02-23 14:00:00+00:00")
async def test_run_daily_publishes_holiday_lookahead():
    """run_daily() emits exactly one HOLIDAY_LOOKAHEAD with a non-empty holidays dict."""
    agent, bus, *_ = _make_agent()
    await agent.run_daily()

    events = bus.get_events(HOLIDAY_LOOKAHEAD)
    assert len(events) == 1
    payload = events[0].payload
    assert "holidays" in payload
    assert isinstance(payload["holidays"], dict)


@freeze_time("2026-02-23 14:00:00+00:00")
async def test_run_daily_emits_nostro_balance_update():
    """run_daily() emits NOSTRO_BALANCE_UPDATE with USD/GBP/AED balances."""
    agent, bus, *_ = _make_agent(
        balances={"USD": 50_000, "GBP": 20_000, "AED": 10_000}
    )
    await agent.run_daily()

    events = bus.get_events(NOSTRO_BALANCE_UPDATE)
    assert len(events) >= 1
    balances = events[-1].payload["balances"]
    assert "USD" in balances
    assert Decimal(balances["USD"]) == Decimal("50000")


# ── B. handle_shortfall ───────────────────────────────────────────────────────


@freeze_time("2026-02-23 14:00:00+00:00")   # Mon 09:00 ET — Fedwire open
async def test_handle_shortfall_submits_proposal():
    """
    Happy path: shortfall on USD with open window and sufficient balance
    → MakerChecker receives exactly one proposal; FUND_MOVEMENT_STATUS emitted.
    """
    agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})
    event = _shortfall_event(currency="USD", shortfall=10_000.0)
    await agent.handle_shortfall(event)

    # One proposal submitted to MC
    assert len(mc.submitted) == 1
    proposal = mc.submitted[0]
    assert proposal.currency == "USD"
    # transfer = 10_000 × 1.10 = 11_000.00
    assert abs(proposal.amount - 11_000.0) < 0.01

    # FUND_MOVEMENT_STATUS emitted with pending_approval
    status_events = bus.get_events(FUND_MOVEMENT_STATUS)
    assert len(status_events) == 1
    assert status_events[0].payload["status"] == "pending_approval"
    assert status_events[0].payload["currency"] == "USD"


@freeze_time("2026-02-21 14:00:00+00:00")   # Saturday — all windows closed
async def test_handle_shortfall_window_not_feasible():
    """
    When the relevant window cannot open before INR market tomorrow,
    no proposal is submitted and FUND_MOVEMENT_STATUS contains window_not_feasible.
    """
    agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})
    event = _shortfall_event(currency="USD", shortfall=10_000.0)
    await agent.handle_shortfall(event)

    assert len(mc.submitted) == 0
    status_events = bus.get_events(FUND_MOVEMENT_STATUS)
    assert len(status_events) == 1
    assert status_events[0].payload["status"] == "window_not_feasible"


@freeze_time("2026-02-23 14:00:00+00:00")
async def test_handle_shortfall_insufficient_balance():
    """
    When available balance < transfer_amount (shortfall × 1.10),
    no proposal is submitted and FUND_MOVEMENT_STATUS contains insufficient_balance.
    """
    agent, bus, mc, *_ = _make_agent(balances={"USD": 500})
    # shortfall=1000 → transfer=1100 > 500 available
    event = _shortfall_event(currency="USD", shortfall=1_000.0)
    await agent.handle_shortfall(event)

    assert len(mc.submitted) == 0
    status_events = bus.get_events(FUND_MOVEMENT_STATUS)
    assert len(status_events) == 1
    assert status_events[0].payload["status"] == "insufficient_balance"
    assert Decimal(status_events[0].payload["available"]) == Decimal("500")


@freeze_time("2026-02-23 14:00:00+00:00")
async def test_handle_shortfall_idempotent():
    """
    Calling handle_shortfall twice for the same currency on the same day
    submits exactly one proposal (second call is silently deduplicated).
    """
    agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})
    event = _shortfall_event(currency="USD", shortfall=5_000.0)

    await agent.handle_shortfall(event)
    await agent.handle_shortfall(event)

    assert len(mc.submitted) == 1
    assert len(bus.get_events(FUND_MOVEMENT_STATUS)) == 1


# ── C. handle_proposal_approved ──────────────────────────────────────────────


@freeze_time("2026-02-23 14:00:00+00:00")
async def test_handle_proposal_approved_executes_and_emits_balance_update():
    """
    PROPOSAL_APPROVED for a known pending proposal triggers FundMover execution.
    On success: FUND_MOVEMENT_STATUS (confirmed) + NOSTRO_BALANCE_UPDATE emitted.
    """
    agent, bus, mc, bank, tracker = _make_agent(balances={"USD": 50_000})
    bank.settle_after_polls = 1

    # Inject a pending proposal directly
    proposal = FundMovementProposal(
        id                 = str(uuid.uuid4()),
        currency           = "USD",
        amount             = 5_000.0,
        source_account     = "OPS-USD-001",
        destination_nostro = "NOSTRO-USD-001",
        rail               = "fedwire",
        proposed_by        = "system:operations_agent",
        purpose            = "test shortfall cover",
        idempotency_key    = "USD-2026-02-23-shortfall",
    )
    agent._pending_proposals[proposal.id] = proposal

    approval_event = create_event(
        event_type   = PROPOSAL_APPROVED,
        source_agent = "maker_checker",
        payload      = {"proposal_id": proposal.id},
    )
    await agent.handle_proposal_approved(approval_event)

    # Proposal removed from pending after execution
    assert proposal.id not in agent._pending_proposals

    # FUND_MOVEMENT_STATUS → confirmed
    status_events = bus.get_events(FUND_MOVEMENT_STATUS)
    assert len(status_events) == 1
    assert status_events[0].payload["status"] == "confirmed"

    # NOSTRO_BALANCE_UPDATE emitted
    balance_events = bus.get_events(NOSTRO_BALANCE_UPDATE)
    assert len(balance_events) == 1
    # Balance should have decreased by 5000
    assert Decimal(balance_events[0].payload["balances"]["USD"]) == Decimal("45000")


@freeze_time("2026-02-23 14:00:00+00:00")
async def test_handle_proposal_approved_submit_unknown_triggers_manual_review():
    """
    When FundMover raises SubmitUnknownError, the agent emits
    FUND_MOVEMENT_STATUS with status=manual_review_required.
    """
    bank = MockBankAPI()
    bank.submit_unknown_once = True   # bank records it but raises network error

    agent, bus, mc, _, tracker = _make_agent(
        balances={"USD": 50_000}, bank=bank
    )

    proposal = FundMovementProposal(
        id                 = str(uuid.uuid4()),
        currency           = "USD",
        amount             = 5_000.0,
        source_account     = "OPS-USD-001",
        destination_nostro = "NOSTRO-USD-001",
        rail               = "fedwire",
        proposed_by        = "system:operations_agent",
        purpose            = "test topup",
        idempotency_key    = "USD-2026-02-23-topup-1",
    )
    agent._pending_proposals[proposal.id] = proposal

    approval_event = create_event(
        event_type   = PROPOSAL_APPROVED,
        source_agent = "maker_checker",
        payload      = {"proposal_id": proposal.id},
    )
    await agent.handle_proposal_approved(approval_event)

    status_events = bus.get_events(FUND_MOVEMENT_STATUS)
    assert len(status_events) == 1
    assert status_events[0].payload["status"] == "manual_review_required"
    assert "proposal_id" in status_events[0].payload


# ── D. handle_deal_instruction ────────────────────────────────────────────────


@freeze_time("2026-02-23 14:00:00+00:00")
async def test_handle_deal_instruction_triggers_topup():
    """
    When cumulative deal amount ≥ nostro_topup_trigger_pct × available_balance,
    a top-up proposal is submitted and FUND_MOVEMENT_STATUS is emitted.
    """
    # balance=10_000, trigger at 90% = 9_000
    agent, bus, mc, *_ = _make_agent(
        balances={"USD": 10_000},
        config={
            "prefunding_buffer_pct":    0.10,
            "window_closing_alert_min": 30,
            "nostro_topup_trigger_pct": 0.90,
            "topup_target_pct":         1.20,
            "monitor_interval_sec":     0.0,
            "stale_proposal_age_min":   90,
            "lookahead_days":           3,
            "monitored_currencies":     ["USD"],
        },
    )

    deal_event = create_event(
        event_type   = "fx.deal.instruction",
        source_agent = "fx_analyst",
        payload      = {"currency": "USD", "amount_foreign": 9_500.0},
    )
    await agent.handle_deal_instruction(deal_event)

    # Top-up proposal submitted
    assert len(mc.submitted) == 1
    proposal = mc.submitted[0]
    assert proposal.currency == "USD"
    # topup_amount = 9500 × 1.20 = 11_400.00
    assert abs(proposal.amount - 11_400.0) < 0.01

    # FUND_MOVEMENT_STATUS emitted
    status_events = bus.get_events(FUND_MOVEMENT_STATUS)
    assert len(status_events) == 1
    assert status_events[0].payload["status"] == "pending_approval"


@freeze_time("2026-02-23 14:00:00+00:00")
async def test_handle_deal_instruction_below_threshold_no_topup():
    """
    When cumulative deal amount is below the trigger threshold, no proposal is
    submitted and no FUND_MOVEMENT_STATUS event is emitted.
    """
    # balance=10_000, trigger at 90% = 9_000; deal=5_000 → no trigger
    agent, bus, mc, *_ = _make_agent(
        balances={"USD": 10_000},
        config={
            "prefunding_buffer_pct":    0.10,
            "window_closing_alert_min": 30,
            "nostro_topup_trigger_pct": 0.90,
            "topup_target_pct":         1.20,
            "monitor_interval_sec":     0.0,
            "stale_proposal_age_min":   90,
            "lookahead_days":           3,
            "monitored_currencies":     ["USD"],
        },
    )

    deal_event = create_event(
        event_type   = "fx.deal.instruction",
        source_agent = "fx_analyst",
        payload      = {"currency": "USD", "amount_foreign": 5_000.0},
    )
    await agent.handle_deal_instruction(deal_event)

    assert len(mc.submitted) == 0
    assert bus.event_count(FUND_MOVEMENT_STATUS) == 0


# ── E. _monitor_windows_once ─────────────────────────────────────────────────


async def test_monitor_windows_once_emits_window_closing_alert():
    """
    At Mon 22:35 UTC (17:35 ET, 25 min before Fedwire closes at 18:00 ET),
    _monitor_windows_once emits a WINDOW_CLOSING alert for USD.
    """
    agent, bus, *_ = _make_agent(
        config={
            "prefunding_buffer_pct":    0.10,
            "window_closing_alert_min": 30,
            "nostro_topup_trigger_pct": 0.90,
            "topup_target_pct":         1.20,
            "monitor_interval_sec":     0.0,
            "stale_proposal_age_min":   90,
            "lookahead_days":           3,
            "monitored_currencies":     ["USD"],
        },
    )
    # Mon 2026-02-23 22:35 UTC = 17:35 ET → 25 min before Fedwire close (18:00 ET)
    now_utc = datetime(2026, 2, 23, 22, 35, tzinfo=timezone.utc)
    await agent._monitor_windows_once(now_utc)

    events = bus.get_events(WINDOW_CLOSING)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["currency"] == "USD"
    assert payload["rail"] == "fedwire"
    assert 0 < payload["minutes_remaining"] <= 30


async def test_monitor_windows_once_deduplicates_alerts():
    """
    Calling _monitor_windows_once twice for the same currency on the same day
    emits exactly one WINDOW_CLOSING alert.
    """
    agent, bus, *_ = _make_agent(
        config={
            "prefunding_buffer_pct":    0.10,
            "window_closing_alert_min": 30,
            "nostro_topup_trigger_pct": 0.90,
            "topup_target_pct":         1.20,
            "monitor_interval_sec":     0.0,
            "stale_proposal_age_min":   90,
            "lookahead_days":           3,
            "monitored_currencies":     ["USD"],
        },
    )
    now_utc = datetime(2026, 2, 23, 22, 35, tzinfo=timezone.utc)

    # Call twice
    await agent._monitor_windows_once(now_utc)
    await agent._monitor_windows_once(now_utc)

    # Only one alert for USD
    events = bus.get_events(WINDOW_CLOSING)
    usd_alerts = [e for e in events if e.payload["currency"] == "USD"]
    assert len(usd_alerts) == 1
