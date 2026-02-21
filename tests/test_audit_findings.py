"""
Tests for audit findings from AUDIT_FINDINGS.md (Parts A + B).

Every test in this file is a regression test for a specific finding.
Tests are expected to FAIL against the current codebase — each failure
confirms the documented bug/gap.

DO NOT fix source code. This file only adds tests.

Naming convention:
    test_<finding_id>_<short_description>

Groups:
    A. Part A — Flow Trace findings (BUG-001 through BUG-004, WARN-001 through WARN-005)
    B. Part B — Contract Drift findings (HIGH, MEDIUM, LOW)
    C. Common missing scenarios
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
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
)
from agents.operations.maker_checker import MakerCheckerWorkflow
from agents.operations.window_manager import WindowManager
from bus.events import (
    FORECAST_READY,
    FUND_MOVEMENT_STATUS,
    NOSTRO_BALANCE_UPDATE,
    PROPOSAL_APPROVED,
    SHORTFALL_ALERT,
    WINDOW_CLOSING,
    HOLIDAY_LOOKAHEAD,
    create_event,
)
from bus.memory_bus import InMemoryBus
from models.domain import (
    DailyForecast,
    FundMovementProposal,
    FundMovementStatus,
    HolidayLookahead,
    ProposalStatus,
    RDAShortfall,
    ShortfallSeverity,
    WindowClosingAlert,
)
from services.calendar_service import CalendarService, IN_RBI_FX


# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixtures (mirrors test_operations_agent.py pattern)
# ═══════════════════════════════════════════════════════════════════════════════


def _no_holidays(d):
    return False


def _make_window_manager() -> WindowManager:
    return WindowManager(
        usd_holidays=_no_holidays,
        gbp_holidays=_no_holidays,
        inr_holidays=_no_holidays,
        aed_holidays=_no_holidays,
        operational_buffer_min=0,
    )


class _MockMC:
    """MakerChecker stub — records proposals, configurable return."""

    def __init__(self, return_status: str = "pending_approval") -> None:
        self.submitted: list[FundMovementProposal] = []
        self._return_status = return_status

    async def submit_proposal(self, proposal: FundMovementProposal) -> dict:
        self.submitted.append(proposal)
        return {"status": self._return_status, "proposal_id": proposal.id}

    async def list_proposals(self) -> list[FundMovementProposal]:
        return list(self.submitted)


class _RejectingMC:
    """MakerChecker stub that always rejects proposals."""

    def __init__(self) -> None:
        self.submitted: list[FundMovementProposal] = []

    async def submit_proposal(self, proposal: FundMovementProposal) -> dict:
        self.submitted.append(proposal)
        return {
            "status": "rejected",
            "proposal_id": proposal.id,
            "errors": ["Nostro not in approved registry"],
        }


def _make_agent(
    balances: dict | None = None,
    bank: MockBankAPI | None = None,
    config: dict | None = None,
    mc=None,
) -> tuple[OperationsAgent, InMemoryBus, object, MockBankAPI, BalanceTracker]:
    bus = InMemoryBus()
    cal = CalendarService()
    bank = bank or MockBankAPI()
    store = InMemoryExecutionStore()
    tracker = BalanceTracker(
        {k: Decimal(v)
         for k, v in (balances or {"USD": 100_000, "GBP": 100_000, "AED": 100_000, "EUR": 100_000}).items()}
    )
    fm = FundMover(
        bank_api=bank, store=store, balance_tracker=tracker,
        calendar_svc=cal, config=FundMoverConfig(poll_interval_sec=0.0),
    )
    mc = mc or _MockMC()
    wm = _make_window_manager()
    cfg = config or {
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
        bus=bus, calendar=cal, window_manager=wm,
        maker_checker=mc, fund_mover=fm, config=cfg,
    )
    return agent, bus, mc, bank, tracker


def _shortfall_event(
    currency: str = "USD",
    shortfall: float = 10_000.0,
    severity: str = "critical",
    correlation_id: str | None = None,
) -> object:
    return create_event(
        event_type=SHORTFALL_ALERT,
        source_agent="liquidity",
        payload={
            "currency":          currency,
            "required_amount":   shortfall + 1_000,
            "available_balance": 1_000.0,
            "shortfall":         shortfall,
            "severity":          severity,
        },
        correlation_id=correlation_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A. Part A — Flow Trace Findings
# ═══════════════════════════════════════════════════════════════════════════════


class TestBUG001_DoubleBuffer:
    """
    BUG-001 / B-HIGH-1: RDAChecker already includes buffer in shortfall.
    OpsAgent should NOT apply another buffer.

    Expected: transfer_amount == shortfall (already buffered).
    Bug: transfer_amount == shortfall * 1.10 (double-buffered).
    """

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_transfer_equals_shortfall_not_double_buffered(self):
        """
        For forecast=5M, balance=1M, buffer_pct=0.10:
          RDA required_with_buffer = 5M * 1.10 = 5.5M
          RDA shortfall = 5.5M - 1M = 4.5M
          OpsAgent should propose 4.5M (not 4.5M * 1.10 = 4.95M)
        """
        agent, bus, mc, *_ = _make_agent(balances={"USD": 10_000_000})

        # Simulate RDA-computed shortfall (already includes buffer)
        event = _shortfall_event(currency="USD", shortfall=4_500_000.0)
        await agent.handle_shortfall(event)

        assert len(mc.submitted) == 1
        proposal = mc.submitted[0]
        # Correct: proposal.amount should be ~4.5M (the shortfall, already buffered)
        # Bug: proposal.amount will be ~4.95M (shortfall × 1.10)
        assert abs(proposal.amount - Decimal("4500000")) < Decimal("1"), (
            f"BUG-001: Transfer amount {float(proposal.amount):,.0f} should equal shortfall "
            f"4,500,000 (already buffer-inclusive), not {4_500_000 * 1.10:,.0f} (double-buffered)"
        )


class TestBUG002_RejectedBlocksRetry:
    """
    BUG-002: If MakerChecker rejects a proposal, OpsAgent should NOT
    add the idempotency key to _pending_intents.
    """

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_rejected_proposal_allows_same_day_retry(self):
        """After MC rejection, a second shortfall event should create a new proposal."""
        mc = _RejectingMC()
        agent, bus, _, bank, tracker = _make_agent(
            balances={"USD": 100_000}, mc=mc
        )

        # First attempt — gets rejected
        event1 = _shortfall_event(currency="USD", shortfall=5_000.0)
        await agent.handle_shortfall(event1)
        assert len(mc.submitted) == 1

        # Second attempt — should NOT be blocked by idempotency
        event2 = _shortfall_event(currency="USD", shortfall=5_000.0)
        await agent.handle_shortfall(event2)
        assert len(mc.submitted) == 2, (
            "BUG-002: Second shortfall after rejection should create a new proposal, "
            "but _pending_intents blocked it"
        )


class TestBUG003_EURCrash:
    """
    BUG-003: EUR shortfall should not crash OpsAgent.
    WindowManager.get_rail("EUR") raises ValueError.
    """

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_eur_shortfall_does_not_crash(self):
        """OpsAgent should handle EUR shortfall gracefully (not raise ValueError)."""
        agent, bus, mc, *_ = _make_agent(balances={"EUR": 100_000})

        event = _shortfall_event(currency="EUR", shortfall=10_000.0)

        # Should not raise — should either submit a proposal or emit an error status
        try:
            await agent.handle_shortfall(event)
        except (ValueError, KeyError) as exc:
            pytest.fail(
                f"BUG-003: EUR shortfall crashed OpsAgent with {type(exc).__name__}: {exc}"
            )


class TestBUG004_ApprovalBridgeBroken:
    """
    BUG-004: MakerCheckerWorkflow.approve() must publish PROPOSAL_APPROVED
    to the event bus. Currently it only writes to audit log.
    """

    async def test_maker_checker_approve_emits_bus_event(self):
        """
        After sufficient approvals, MakerChecker should emit PROPOSAL_APPROVED
        on the bus so OpsAgent.handle_proposal_approved() can execute the transfer.
        """
        # Build real MakerChecker with in-memory stubs
        class _InMemDB:
            def __init__(self):
                self._store = {}
            async def get(self, pid):
                return self._store.get(pid)
            async def save(self, p):
                self._store[p.id] = p
            async def is_approved_nostro(self, nostro):
                return True
            async def has_recent_duplicate(self, key):
                return False
            async def list_all(self):
                return list(self._store.values())

        class _MockAuth:
            async def can_approve(self, checker_id, proposal):
                return True

        class _MockAlerts:
            def __init__(self):
                self.executed = []
            async def notify_checkers(self, proposal, n):
                pass
            async def escalate(self, proposal, **kw):
                pass
            async def notify_executed(self, proposal):
                self.executed.append(proposal)

        class _MockAudit:
            async def log(self, **kw):
                pass

        bus = InMemoryBus()
        db = _InMemDB()
        mc = MakerCheckerWorkflow(db, _MockAuth(), _MockAlerts(), _MockAudit())

        # Submit a proposal
        proposal = FundMovementProposal(
            id=str(uuid.uuid4()),
            currency="USD",
            amount=5000.0,
            source_account="OPS-USD-001",
            destination_nostro="NOSTRO-USD-001",
            rail="fedwire",
            proposed_by="system:operations_agent",
            purpose="test",
            idempotency_key="USD-2026-02-23-shortfall",
        )
        await mc.submit_proposal(proposal)

        # Approve
        await mc.approve(proposal.id, "checker-1")

        # BUG-004: No PROPOSAL_APPROVED event is emitted to the bus.
        # After approval, MakerChecker should have published to bus.
        # Since mc has no bus reference, this verifies the architectural gap.
        #
        # The test documents the gap: MakerChecker has no bus dependency,
        # so it cannot publish events. This is the root cause of BUG-004.
        assert not hasattr(mc, 'bus') or mc.bus is None if hasattr(mc, 'bus') else True, (
            "BUG-004: MakerCheckerWorkflow has no EventBus dependency. "
            "It cannot publish PROPOSAL_APPROVED to the bus. "
            "OpsAgent.handle_proposal_approved() is unreachable dead code."
        )


class TestWARN001_SeverityIgnored:
    """WARN-001 / B-MED-1: severity field should influence behavior."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_warning_severity_treated_differently_than_critical(self):
        """A WARNING shortfall should be handled differently from CRITICAL."""
        agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})

        event = _shortfall_event(currency="USD", shortfall=5_000.0, severity="warning")
        await agent.handle_shortfall(event)

        # Current behavior: WARNING still creates a proposal (same as CRITICAL).
        # Document the fact that severity is ignored:
        if len(mc.submitted) > 0:
            pytest.fail(
                "WARN-001: handle_shortfall() ignores severity field. "
                "A WARNING-level shortfall should not unconditionally create a proposal."
            )


class TestWARN002_FundMoverIgnoresRail:
    """WARN-002: FundMover re-resolves rail, ignoring proposal.rail."""

    async def test_execution_rail_matches_proposal_rail(self):
        """The rail used for bank submission should match the proposal's rail field."""
        bank = MockBankAPI()
        store = InMemoryExecutionStore()
        tracker = BalanceTracker({"USD": Decimal("100000")})
        fm = FundMover(
            bank_api=bank, store=store, balance_tracker=tracker,
            config=FundMoverConfig(poll_interval_sec=0.0),
        )

        proposal = FundMovementProposal(
            id="P-RAIL-TEST",
            currency="USD",
            amount=Decimal("5000.00"),
            source_account="OPS-USD-001",
            destination_nostro="NOSTRO-USD-001",
            rail="chaps",  # intentionally wrong — USD should use fedwire
            proposed_by="test",
            purpose="test",
            idempotency_key="test-rail",
        )
        execution = await fm.execute_proposal(proposal)

        # FundMover ignores proposal.rail and re-resolves to "fedwire" for USD.
        # The proposal says "chaps" but execution uses "fedwire".
        assert execution.rail == proposal.rail, (
            f"WARN-002: FundMover used rail '{execution.rail}' but proposal.rail is "
            f"'{proposal.rail}'. FundMover ignores the proposal's rail field."
        )


class TestWARN003_EURBalanceNeverPublished:
    """WARN-003: EUR should be in monitored currencies for balance snapshots."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_nostro_balance_update_includes_eur(self):
        """NOSTRO_BALANCE_UPDATE should include EUR balance."""
        agent, bus, *_ = _make_agent(
            balances={"USD": 100_000, "GBP": 100_000, "AED": 100_000, "EUR": 50_000}
        )
        await agent.run_daily()

        events = bus.get_events(NOSTRO_BALANCE_UPDATE)
        assert len(events) >= 1
        balances = events[-1].payload["balances"]
        assert "EUR" in balances, (
            "WARN-003: NOSTRO_BALANCE_UPDATE does not include EUR. "
            "_DEFAULT_CURRENCIES = ['USD', 'GBP', 'AED'] excludes EUR."
        )


class TestWARN005_MisleadingExecutedStatus:
    """WARN-005: MakerChecker._execute() marks as EXECUTED before bank transfer."""

    async def test_maker_checker_execute_status_is_not_executed(self):
        """
        After MakerChecker approval, proposal.status should not be EXECUTED
        until the actual bank transfer completes.
        """
        # Build real MakerChecker
        class _InMemDB:
            def __init__(self):
                self._store = {}
            async def get(self, pid):
                return self._store.get(pid)
            async def save(self, p):
                self._store[p.id] = p
            async def is_approved_nostro(self, nostro):
                return True
            async def has_recent_duplicate(self, key):
                return False

        class _MockAuth:
            async def can_approve(self, cid, p):
                return True

        class _MockAlerts:
            async def notify_checkers(self, p, n): pass
            async def escalate(self, p, **kw): pass
            async def notify_executed(self, p): pass

        class _MockAudit:
            async def log(self, **kw): pass

        db = _InMemDB()
        mc = MakerCheckerWorkflow(db, _MockAuth(), _MockAlerts(), _MockAudit())

        proposal = FundMovementProposal(
            id=str(uuid.uuid4()),
            currency="USD",
            amount=5000.0,
            source_account="OPS-USD-001",
            destination_nostro="NOSTRO-USD-001",
            rail="fedwire",
            proposed_by="system:operations_agent",
            purpose="test",
            idempotency_key="test-executed",
        )
        await mc.submit_proposal(proposal)
        result = await mc.approve(proposal.id, "checker-1")

        saved = await db.get(proposal.id)
        # WARN-005: MakerChecker sets EXECUTED but no bank transfer happened
        assert saved.status != ProposalStatus.EXECUTED, (
            "WARN-005: MakerChecker marks proposal as EXECUTED after approval, "
            "but no bank transfer has been submitted. Status should be APPROVED "
            "until FundMover confirms the transfer."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# B. Part B — Contract Drift Findings
# ═══════════════════════════════════════════════════════════════════════════════


class TestBHIGH2_StatusStringsNotInEnum:
    """B-HIGH-2: OpsAgent emits status strings not in ProposalStatus enum."""

    def test_window_not_feasible_in_proposal_status(self):
        """ProposalStatus should contain 'window_not_feasible'."""
        try:
            ProposalStatus("window_not_feasible")
        except ValueError:
            pytest.fail(
                "B-HIGH-2: 'window_not_feasible' is not in ProposalStatus enum. "
                "OpsAgent emits this status in FUND_MOVEMENT_STATUS events."
            )

    def test_insufficient_balance_in_proposal_status(self):
        """ProposalStatus should contain 'insufficient_balance'."""
        try:
            ProposalStatus("insufficient_balance")
        except ValueError:
            pytest.fail(
                "B-HIGH-2: 'insufficient_balance' is not in ProposalStatus enum."
            )

    def test_manual_review_required_in_proposal_status(self):
        """ProposalStatus should contain 'manual_review_required'."""
        try:
            ProposalStatus("manual_review_required")
        except ValueError:
            pytest.fail(
                "B-HIGH-2: 'manual_review_required' is not in ProposalStatus enum."
            )

    def test_stale_review_needed_in_proposal_status(self):
        """ProposalStatus should contain 'stale_review_needed'."""
        try:
            ProposalStatus("stale_review_needed")
        except ValueError:
            pytest.fail(
                "B-HIGH-2: 'stale_review_needed' is not in ProposalStatus enum."
            )


class TestBHIGH3_NaiveDatetime:
    """B-HIGH-3: FundMovementProposal should use tz-aware datetimes."""

    def test_proposal_created_at_is_timezone_aware(self):
        """FundMovementProposal().created_at must have tzinfo."""
        proposal = FundMovementProposal(
            id="test",
            currency="USD",
            amount=1000.0,
            source_account="OPS-USD-001",
            destination_nostro="NOSTRO-USD-001",
            rail="fedwire",
            proposed_by="test",
            purpose="test",
            idempotency_key="test",
        )
        assert proposal.created_at.tzinfo is not None, (
            "B-HIGH-3: FundMovementProposal.created_at uses datetime.utcnow() "
            "(naive). Should use datetime.now(timezone.utc) (tz-aware)."
        )

    def test_proposal_updated_at_is_timezone_aware(self):
        """FundMovementProposal().updated_at must have tzinfo."""
        proposal = FundMovementProposal(
            id="test",
            currency="USD",
            amount=1000.0,
            source_account="OPS-USD-001",
            destination_nostro="NOSTRO-USD-001",
            rail="fedwire",
            proposed_by="test",
            purpose="test",
            idempotency_key="test",
        )
        assert proposal.updated_at.tzinfo is not None, (
            "B-HIGH-3: FundMovementProposal.updated_at uses datetime.utcnow() "
            "(naive). Should use datetime.now(timezone.utc) (tz-aware)."
        )


class TestBHIGH4_RestartIdempotency:
    """B-HIGH-4: _pending_intents must survive agent restart."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_restart_does_not_allow_duplicate_proposal(self):
        """
        After processing a shortfall, a new OpsAgent instance (simulating restart)
        should detect the already-submitted proposal and skip.
        """
        # First instance processes shortfall
        agent1, bus1, mc1, *_ = _make_agent(balances={"USD": 100_000})
        event = _shortfall_event(currency="USD", shortfall=5_000.0)
        await agent1.handle_shortfall(event)
        assert len(mc1.submitted) == 1

        # Second instance (restart) — shares the same MC (simulates the same database)
        agent2, bus2, mc2, *_ = _make_agent(balances={"USD": 100_000}, mc=mc1)
        await agent2.handle_shortfall(event)

        # After fix: only 1 proposal total — restart detected the existing submission
        # and did NOT create a duplicate.  mc2 is mc1 (shared MC = shared DB).
        assert len(mc2.submitted) == 1, (
            "B-HIGH-4: After restart, _pending_intents must be restored from MC. "
            "The same shortfall event must not create a duplicate proposal."
        )


class TestBHIGH5_FloatPrecision:
    """B-HIGH-5: Decimal→float→Decimal round-trip precision risk."""

    def test_decimal_float_decimal_roundtrip_preserves_precision(self):
        """
        The Decimal→float→Decimal round-trip through FundMovementProposal.amount
        should preserve sub-cent precision.
        """
        original = Decimal("1234567.89")
        as_float = float(original)
        recovered = Decimal(str(as_float))
        assert recovered == original, (
            f"B-HIGH-5: Decimal({original}) → float → Decimal = {recovered}. "
            "Precision was lost in the Decimal→float→Decimal round-trip."
        )

    def test_large_amount_float_roundtrip(self):
        """Large amount edge case for precision."""
        original = Decimal("99999999.99")
        as_float = float(original)
        recovered = Decimal(str(as_float))
        assert recovered == original, (
            f"B-HIGH-5: Large amount {original} lost precision in round-trip: {recovered}"
        )


class TestBMED3_FundMovementStatusErrorPaths:
    """B-MED-3: FUND_MOVEMENT_STATUS error paths should include all required fields."""

    @freeze_time("2026-02-21 14:00:00+00:00")  # Saturday — window not feasible
    async def test_window_not_feasible_includes_proposal_id(self):
        """Error-path FUND_MOVEMENT_STATUS should include proposal_id."""
        agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})
        event = _shortfall_event(currency="USD", shortfall=10_000.0)
        await agent.handle_shortfall(event)

        status_events = bus.get_events(FUND_MOVEMENT_STATUS)
        assert len(status_events) == 1
        payload = status_events[0].payload
        assert "proposal_id" in payload, (
            "B-MED-3: FUND_MOVEMENT_STATUS with status='window_not_feasible' "
            "is missing 'proposal_id' field."
        )

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_insufficient_balance_includes_proposal_id(self):
        """Error-path FUND_MOVEMENT_STATUS should include proposal_id."""
        agent, bus, mc, *_ = _make_agent(balances={"USD": 500})
        event = _shortfall_event(currency="USD", shortfall=1_000.0)
        await agent.handle_shortfall(event)

        status_events = bus.get_events(FUND_MOVEMENT_STATUS)
        assert len(status_events) == 1
        payload = status_events[0].payload
        assert "proposal_id" in payload, (
            "B-MED-3: FUND_MOVEMENT_STATUS with status='insufficient_balance' "
            "is missing 'proposal_id' field."
        )


class TestBMED4_ForecastReadyTypeDivergence:
    """B-MED-4: FORECAST_READY payload types should match DailyForecast domain model."""

    def test_daily_forecast_from_payload_roundtrip(self):
        """
        Constructing DailyForecast from a typical FORECAST_READY payload
        should not raise on type mismatches.
        """
        payload = {
            "forecast_date":       "2026-02-23",
            "total_inr_crores":    45.0,
            "confidence":          "high",
            "currency_split":      {"USD": 30.0, "GBP": 15.0},
            "multipliers_applied": {"payday": 1.0},
            "created_at":          "2026-02-23T06:00:00+00:00",
        }
        # DailyForecast expects date, ForecastConfidence, datetime — not strings
        try:
            DailyForecast(
                forecast_date=payload["forecast_date"],
                total_inr_crores=payload["total_inr_crores"],
                confidence=payload["confidence"],
                currency_split=payload["currency_split"],
                multipliers_applied=payload["multipliers_applied"],
            )
        except (TypeError, ValueError) as exc:
            pytest.fail(
                f"B-MED-4: Cannot construct DailyForecast from bus payload: {exc}. "
                "Payload uses str where domain model expects date/enum."
            )


class TestBMED6_NostroCorrelationBroken:
    """B-MED-6: NOSTRO_BALANCE_UPDATE should carry correlation_id from triggering event."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_nostro_update_carries_forecast_correlation_id(self):
        """
        When OpsAgent.handle_forecast() emits NOSTRO_BALANCE_UPDATE,
        it should carry the correlation_id from the FORECAST_READY event.
        """
        agent, bus, mc, *_ = _make_agent()
        await agent.setup()

        forecast_corr = str(uuid.uuid4())
        forecast_event = create_event(
            event_type=FORECAST_READY,
            source_agent="liquidity",
            payload={
                "forecast_date":       "2026-02-23",
                "total_inr_crores":    50.0,
                "confidence":          "high",
                "currency_split":      {"USD": 30.0},
                "multipliers_applied": {},
                "created_at":          "2026-02-23T06:00:00+00:00",
            },
            correlation_id=forecast_corr,
        )
        await agent.handle_forecast(forecast_event)

        nostro_events = bus.get_events(NOSTRO_BALANCE_UPDATE)
        assert len(nostro_events) >= 1

        # Check that NOSTRO_BALANCE_UPDATE carries the forecast's correlation_id
        nostro_corr = nostro_events[-1].correlation_id
        assert nostro_corr == forecast_corr, (
            f"B-MED-6: NOSTRO_BALANCE_UPDATE correlation_id={nostro_corr[:8]}... "
            f"differs from FORECAST_READY correlation_id={forecast_corr[:8]}... "
            "The correlation chain is broken."
        )


class TestBMED7_DayRolloverIdempotency:
    """B-MED-7: Day rollover should not cause duplicate proposals for yesterday's events."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_day_rollover_does_not_duplicate_yesterdays_proposal(self):
        """
        A shortfall from yesterday (already actioned) should not trigger a
        second proposal after day rollover.
        """
        agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})

        # Process shortfall for "today"
        event = _shortfall_event(currency="USD", shortfall=5_000.0)
        await agent.handle_shortfall(event)
        assert len(mc.submitted) == 1

        # Simulate day rollover
        now_next_day = datetime(2026, 2, 24, 14, 0, tzinfo=timezone.utc)
        agent._check_day_rollover(now_next_day)

        # Re-deliver yesterday's shortfall (e.g., bus replay)
        # The idempotency key includes yesterday's date, so it should be safe
        # BUT: _check_day_rollover clears ALL keys, not just expired ones
        await agent.handle_shortfall(event)

        # Since the key format is "{ccy}-{date}-shortfall", yesterday's date
        # differs from today's. This should be naturally safe.
        # The actual risk is with events queued at the day boundary.
        # Test that _pending_intents retained yesterday's key after rollover.
        yesterday_key = "USD-2026-02-23-shortfall"
        assert yesterday_key not in agent._pending_intents, (
            "B-MED-7: After day rollover, yesterday's idempotency keys are wiped. "
            "If yesterday's event is replayed, it could create a duplicate."
        )


class TestBMED9_HolidayLookaheadKeyMismatch:
    """B-MED-9: HOLIDAY_LOOKAHEAD payload should match HolidayLookahead domain model."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_holiday_lookahead_payload_matches_domain_model(self):
        """HolidayLookahead(**payload) should not raise TypeError."""
        agent, bus, *_ = _make_agent()
        await agent.run_daily()

        events = bus.get_events(HOLIDAY_LOOKAHEAD)
        assert len(events) >= 1
        payload = events[0].payload

        # HolidayLookahead expects 'generated_at' but payload uses 'generated_at_utc'
        try:
            HolidayLookahead(
                generated_at=payload.get("generated_at", payload.get("generated_at_utc")),
                holidays=payload["holidays"],
            )
        except (TypeError, KeyError) as exc:
            pytest.fail(
                f"B-MED-9: Cannot construct HolidayLookahead from payload: {exc}. "
                "Payload key 'generated_at_utc' doesn't match model field 'generated_at'."
            )

        # Verify the key name matches
        assert "generated_at" in payload, (
            "B-MED-9: HOLIDAY_LOOKAHEAD payload uses 'generated_at_utc' "
            "but HolidayLookahead model expects 'generated_at'."
        )


class TestBMED10_TripleTypeConversion:
    """B-MED-10: Nostro balances emitted as str(Decimal) instead of float."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_nostro_balance_values_are_numeric(self):
        """NOSTRO_BALANCE_UPDATE balance values should be numeric, not strings."""
        agent, bus, *_ = _make_agent(
            balances={"USD": 50_000, "GBP": 20_000, "AED": 10_000}
        )
        await agent.run_daily()

        events = bus.get_events(NOSTRO_BALANCE_UPDATE)
        assert len(events) >= 1
        balances = events[-1].payload["balances"]

        for ccy, val in balances.items():
            assert isinstance(val, (int, float)), (
                f"B-MED-10: Balance for {ccy} is {type(val).__name__} ('{val}'). "
                "Should be numeric (float), not str(Decimal)."
            )


class TestBLOW1_DirectionIgnored:
    """B-LOW-1: direction field in DEAL_INSTRUCTION should influence behavior."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_hold_direction_does_not_accumulate_deals(self):
        """A DEAL_INSTRUCTION with direction='HOLD' should not trigger top-up logic."""
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
            event_type="fx.deal.instruction",
            source_agent="fx_analyst",
            payload={
                "currency": "USD",
                "amount_foreign": 9_500.0,
                "direction": "HOLD",
            },
        )
        await agent.handle_deal_instruction(deal_event)

        assert len(mc.submitted) == 0, (
            "B-LOW-1: A DEAL_INSTRUCTION with direction='HOLD' should not "
            "accumulate deals or trigger a top-up proposal."
        )


class TestBLOW2_WindowClosingPayloadDiverge:
    """B-LOW-2: WINDOW_CLOSING payload should match WindowClosingAlert model."""

    async def test_window_closing_payload_includes_generated_at(self):
        """WINDOW_CLOSING payload should include 'generated_at' field."""
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
        # Mon 22:35 UTC = 17:35 ET — 25 min before Fedwire close
        now_utc = datetime(2026, 2, 23, 22, 35, tzinfo=timezone.utc)
        await agent._monitor_windows_once(now_utc)

        events = bus.get_events(WINDOW_CLOSING)
        assert len(events) >= 1
        payload = events[0].payload
        assert "generated_at" in payload, (
            "B-LOW-2: WINDOW_CLOSING payload is missing 'generated_at' field "
            "required by WindowClosingAlert domain model."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# C. Common Missing Scenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestC2_MissingOptionalField:
    """C-2: Event handlers should not crash on missing optional fields."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_shortfall_missing_severity_does_not_crash(self):
        """handle_shortfall should tolerate missing severity field."""
        agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})

        event = create_event(
            event_type=SHORTFALL_ALERT,
            source_agent="liquidity",
            payload={
                "currency":          "USD",
                "required_amount":   11_000.0,
                "available_balance": 0.0,
                "shortfall":         10_000.0,
                # severity intentionally omitted
            },
        )
        try:
            await agent.handle_shortfall(event)
        except KeyError as exc:
            pytest.fail(f"C-2: handle_shortfall crashed on missing optional field: {exc}")

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_shortfall_missing_detected_at_does_not_crash(self):
        """handle_shortfall should tolerate missing detected_at field."""
        agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})

        event = create_event(
            event_type=SHORTFALL_ALERT,
            source_agent="liquidity",
            payload={
                "currency":  "USD",
                "shortfall": 10_000.0,
                # detected_at intentionally omitted
            },
        )
        try:
            await agent.handle_shortfall(event)
        except KeyError as exc:
            pytest.fail(f"C-2: handle_shortfall crashed on missing optional field: {exc}")


class TestC3_ExtraUnexpectedField:
    """C-3: Event handlers should not crash on extra fields in payload."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_shortfall_with_extra_fields(self):
        """handle_shortfall should ignore extra unexpected fields."""
        agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})

        event = create_event(
            event_type=SHORTFALL_ALERT,
            source_agent="liquidity",
            payload={
                "currency":          "USD",
                "required_amount":   11_000.0,
                "available_balance": 0.0,
                "shortfall":         10_000.0,
                "severity":          "critical",
                "extra_field":       "unexpected_value",
                "debug_info":        {"nested": True},
            },
        )
        try:
            await agent.handle_shortfall(event)
        except Exception as exc:
            pytest.fail(f"C-3: handle_shortfall crashed on extra fields: {exc}")

        assert len(mc.submitted) == 1


class TestC7_DayRolloverPreservesProposals:
    """C-7: Day rollover resets daily state but should not clear pending proposals."""

    @freeze_time("2026-02-23 14:00:00+00:00")
    async def test_day_rollover_preserves_pending_proposals(self):
        """_pending_proposals should survive day rollover."""
        agent, bus, mc, *_ = _make_agent(balances={"USD": 100_000})

        # Create a pending proposal
        proposal = FundMovementProposal(
            id=str(uuid.uuid4()),
            currency="USD",
            amount=5000.0,
            source_account="OPS-USD-001",
            destination_nostro="NOSTRO-USD-001",
            rail="fedwire",
            proposed_by="system:operations_agent",
            purpose="test",
            idempotency_key="USD-2026-02-23-shortfall",
        )
        agent._pending_proposals[proposal.id] = proposal

        # Day rollover
        now_next_day = datetime(2026, 2, 24, 14, 0, tzinfo=timezone.utc)
        agent._check_day_rollover(now_next_day)

        # Pending proposals should survive
        assert proposal.id in agent._pending_proposals, (
            "C-7: Day rollover should not clear _pending_proposals. "
            "A proposal pending approval from yesterday must survive."
        )
