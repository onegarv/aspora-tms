"""
Iteration 1 — domain model contract tests.

These tests are deliberately written BEFORE the fixes so they fail first,
proving each bug exists, then pass after the minimal fix.

Findings covered:
  Part-B-HIGH-2  ProposalStatus enum gaps (OpsAgent emits statuses not in enum)
  Part-B-HIGH-3  naive datetime (FundMovementProposal.created_at / updated_at)
  Part-B-HIGH-5  float amount (FundMovementProposal.amount should be Decimal)
  Part-B-MEDIUM-4 FORECAST_READY type diverge (DailyForecast missing payload methods)
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from models.domain import (
    DailyForecast,
    ForecastConfidence,
    FundMovementProposal,
    ProposalStatus,
    RDAShortfall,
    ShortfallSeverity,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _minimal_proposal(**overrides) -> FundMovementProposal:
    defaults = dict(
        id="p-test-1",
        currency="USD",
        amount=Decimal("5000.00"),
        source_account="ACC-001",
        destination_nostro="NOSTRO-USD",
        rail="fedwire",
        proposed_by="system:ops",
        purpose="test proposal",
        idempotency_key="idem-001",
    )
    defaults.update(overrides)
    return FundMovementProposal(**defaults)


# ── Part-B-HIGH-2: ProposalStatus enum gaps ───────────────────────────────────

class TestProposalStatusEnumGaps:
    """
    OpsAgent emits these literal strings as 'status' values that are not in the
    ProposalStatus enum.  Calling ProposalStatus(value) raises ValueError on the
    consumer side (dashboard, audit log, any deserialiser).
    """

    @pytest.mark.parametrize("status_value", [
        "window_not_feasible",
        "insufficient_balance",
        "manual_review_required",
        "stale_review_needed",
    ])
    def test_ops_agent_statuses_are_valid_enum_members(self, status_value):
        """ProposalStatus(value) must not raise ValueError."""
        # This will raise ValueError until the enum is extended.
        status = ProposalStatus(status_value)
        assert status.value == status_value


# ── Part-B-HIGH-3: naive datetime ────────────────────────────────────────────

class TestTzAwareDatetimes:
    """
    FundMovementProposal.created_at and updated_at use datetime.utcnow()
    (naive, no tzinfo).  TransferExecution uses datetime.now(timezone.utc)
    (tz-aware).  Comparing the two raises TypeError.
    """

    def test_fund_movement_proposal_created_at_is_tz_aware(self):
        proposal = _minimal_proposal()
        assert proposal.created_at.tzinfo is not None, (
            "created_at must be tz-aware (use datetime.now(timezone.utc))"
        )

    def test_fund_movement_proposal_updated_at_is_tz_aware(self):
        proposal = _minimal_proposal()
        assert proposal.updated_at.tzinfo is not None, (
            "updated_at must be tz-aware (use datetime.now(timezone.utc))"
        )

    def test_daily_forecast_created_at_is_tz_aware(self):
        forecast = DailyForecast(
            forecast_date=date(2026, 2, 21),
            total_inr_crores=100.0,
            confidence=ForecastConfidence.HIGH,
            currency_split={"USD": 100.0},
            multipliers_applied={},
        )
        assert forecast.created_at.tzinfo is not None, (
            "DailyForecast.created_at must be tz-aware"
        )

    def test_rda_shortfall_detected_at_is_tz_aware(self):
        sf = RDAShortfall(
            currency="USD",
            required_amount=6_000_000.0,
            available_balance=1_000_000.0,
            shortfall=5_000_000.0,
            severity=ShortfallSeverity.CRITICAL,
        )
        assert sf.detected_at.tzinfo is not None, (
            "RDAShortfall.detected_at must be tz-aware"
        )


# ── Part-B-HIGH-5: float amount → Decimal ────────────────────────────────────

class TestFundMovementProposalAmountIsDecimal:
    """
    FundMovementProposal.amount is typed as float, causing a
    Decimal→float→Decimal round-trip in fund_mover.py.
    """

    def test_amount_accepts_decimal(self):
        proposal = _minimal_proposal(amount=Decimal("5000.00"))
        assert isinstance(proposal.amount, Decimal), (
            "FundMovementProposal.amount should be Decimal, not float"
        )

    def test_amount_preserves_precision(self):
        """No rounding artefacts from float representation."""
        proposal = _minimal_proposal(amount=Decimal("1234567.89"))
        assert proposal.amount == Decimal("1234567.89")

    def test_amount_field_type_annotation_is_decimal(self):
        """The type annotation on the dataclass field is Decimal."""
        import typing
        # get_type_hints() resolves PEP-563 string annotations back to types
        hints = typing.get_type_hints(FundMovementProposal)
        assert hints["amount"] is Decimal, (
            "FundMovementProposal.amount annotation should be Decimal, not float"
        )


# ── Part-B-MEDIUM-4: DailyForecast payload round-trip ────────────────────────

class TestDailyForecastPayloadMethods:
    """
    LiquidityAgent emits a raw dict payload for FORECAST_READY.
    OperationsAgent re-creates a DailyForecast from that dict.
    Without explicit serialisation methods the two sides drift.

    DailyForecast should have:
      to_event_payload() -> dict    (serialise for bus emission)
      from_event_payload(d) -> DailyForecast  (deserialise on consumer side)
    """

    def _make_forecast(self) -> DailyForecast:
        return DailyForecast(
            forecast_date=date(2026, 2, 21),
            total_inr_crores=258.5,
            confidence=ForecastConfidence.HIGH,
            currency_split={"USD": 172.3, "GBP": 51.2, "AED": 35.0},
            multipliers_applied={"holiday": 1.2},
        )

    def test_to_event_payload_returns_dict(self):
        f = self._make_forecast()
        assert hasattr(f, "to_event_payload"), (
            "DailyForecast must have a to_event_payload() method"
        )
        payload = f.to_event_payload()
        assert isinstance(payload, dict)

    def test_payload_contains_required_keys(self):
        f = self._make_forecast()
        payload = f.to_event_payload()
        for key in ("forecast_date", "total_inr_crores", "confidence",
                    "currency_split", "multipliers_applied", "created_at"):
            assert key in payload, f"Missing key: {key}"

    def test_from_event_payload_class_method_exists(self):
        assert hasattr(DailyForecast, "from_event_payload"), (
            "DailyForecast must have a from_event_payload() class method"
        )

    def test_round_trip_preserves_values(self):
        f = self._make_forecast()
        payload = f.to_event_payload()
        restored = DailyForecast.from_event_payload(payload)
        assert restored.forecast_date == f.forecast_date
        assert restored.total_inr_crores == pytest.approx(f.total_inr_crores)
        assert restored.confidence == f.confidence
        assert restored.currency_split == f.currency_split
        assert restored.multipliers_applied == f.multipliers_applied

    def test_forecast_date_survives_json_string_roundtrip(self):
        """forecast_date is serialised as ISO string and deserialized back to date."""
        f = self._make_forecast()
        payload = f.to_event_payload()
        assert isinstance(payload["forecast_date"], str), (
            "forecast_date must be serialised as an ISO string for JSON transport"
        )
        restored = DailyForecast.from_event_payload(payload)
        assert isinstance(restored.forecast_date, date)
        assert restored.forecast_date == date(2026, 2, 21)
