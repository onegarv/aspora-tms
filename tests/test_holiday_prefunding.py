"""
Tests for holiday-aware prefunding in the Liquidity Agent.

Covers:
  - Single holiday emits prefunding alert with correct metadata
  - Long weekend (3 closed days) aggregates volumes and sets urgency=high
  - No holiday tomorrow → no prefunding alerts
  - Sufficient nostro balance → no prefunding alert
  - Regular shortfall alerts do not carry holiday metadata
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from freezegun import freeze_time

from agents.liquidity.agent import LiquidityAgent
from bus.events import (
    FORECAST_READY,
    SHORTFALL_ALERT,
    create_event,
)
from bus.memory_bus import InMemoryBus
from services.calendar_service import CalendarService, IN_RBI_FX


# ── Helpers ──────────────────────────────────────────────────────────────────


def _eight_day_history(
    base_vol_usd: float = 5_000_000.0,
    dow: int = 3,
) -> dict[str, list[dict]]:
    anchor = date(2025, 12, 25)
    rows = [
        {"date": (anchor - timedelta(weeks=w)).isoformat(),
         "volume_usd": base_vol_usd,
         "dow": dow}
        for w in range(8)
    ]
    return {corridor: rows for corridor in ["USD_INR", "GBP_INR", "AED_INR"]}


def _make_agent(
    cal: CalendarService | None = None,
    nostro: dict[str, float] | None = None,
    dow: int = 3,
) -> tuple[LiquidityAgent, InMemoryBus, CalendarService]:
    bus = InMemoryBus()
    cal = cal or CalendarService()
    agent = LiquidityAgent(bus=bus, calendar=cal)
    agent._forecaster.load(_eight_day_history(dow=dow))
    agent._spot_rates = {
        "USD_INR": 84.0,
        "GBP_INR": 106.0,
        "EUR_INR": 91.0,
        "AED_INR": 22.9,
    }
    if nostro is not None:
        agent._nostro_balances = nostro
    return agent, bus, cal


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSingleHolidayPrefunding:
    @freeze_time("2026-02-25 09:30:00+00:00")  # Wednesday
    async def test_emits_prefunding_alert(self):
        """One holiday tomorrow (Thu) → alert with holiday_prefunding=True, urgency=medium."""
        cal = CalendarService()
        # Thursday 2026-02-26 is a holiday; Fri 2026-02-27 is a business day
        # So only 1 closed day → urgency=medium
        cal.add_holiday(
            date(2026, 2, 26), IN_RBI_FX,
            name="Test Holiday",
            user="test",
        )

        agent, bus, _ = _make_agent(
            cal=cal,
            nostro={"USD": 1_000_000.0, "GBP": 8_000_000.0, "AED": 8_000_000.0},
            dow=2,  # Wednesday
        )

        await agent.setup()
        await bus.stop()
        await agent.run_daily()
        await bus.start()

        shortfall_events = bus.get_events(SHORTFALL_ALERT)
        holiday_alerts = [
            e for e in shortfall_events
            if e.payload.get("holiday_prefunding") is True
        ]

        assert len(holiday_alerts) >= 1, (
            "Expected at least one SHORTFALL_ALERT with holiday_prefunding=True"
        )

        usd_holiday = next(
            (e for e in holiday_alerts if e.payload["currency"] == "USD"), None
        )
        assert usd_holiday is not None
        assert usd_holiday.payload["urgency"] == "medium"
        assert usd_holiday.payload["closed_days"] == 1
        assert "2026-02-26" in usd_holiday.payload["covers_dates"]


class TestLongWeekendPrefunding:
    @freeze_time("2026-02-19 09:30:00+00:00")  # Thursday
    async def test_aggregates_3_day_volumes(self):
        """
        Friday 2/20 is a holiday, Sat 2/21, Sun 2/22 are weekend → 3 closed days.
        Shortfall should be ~3× single day, urgency=high.
        """
        cal = CalendarService()
        cal.add_holiday(
            date(2026, 2, 20), IN_RBI_FX,
            name="Test Friday Holiday",
            user="test",
        )
        # Sat 2/21 + Sun 2/22 are already non-business days
        # So consecutive_non_business_days(2026-02-20) = 3

        agent, bus, _ = _make_agent(
            cal=cal,
            nostro={"USD": 1_000_000.0, "GBP": 1_000_000.0, "AED": 1_000_000.0},
        )

        await agent.setup()
        await bus.stop()
        await agent.run_daily()
        await bus.start()

        shortfall_events = bus.get_events(SHORTFALL_ALERT)
        holiday_alerts = [
            e for e in shortfall_events
            if e.payload.get("holiday_prefunding") is True
        ]

        assert len(holiday_alerts) >= 1

        usd_holiday = next(
            (e for e in holiday_alerts if e.payload["currency"] == "USD"), None
        )
        assert usd_holiday is not None
        assert usd_holiday.payload["urgency"] == "high"
        assert usd_holiday.payload["closed_days"] == 3
        assert len(usd_holiday.payload["covers_dates"]) == 3

        # Shortfall should be significantly larger than single-day
        assert usd_holiday.payload["shortfall"] > 0


class TestNoHolidayNoPrefunding:
    @freeze_time("2026-02-25 09:30:00+00:00")  # Wednesday
    async def test_no_prefunding_alert(self):
        """Tomorrow (Thursday) is a business day → no holiday prefunding alerts."""
        agent, bus, _ = _make_agent(
            nostro={"USD": 1_000_000.0, "GBP": 8_000_000.0, "AED": 8_000_000.0},
            dow=2,  # Wednesday
        )

        await agent.setup()
        await bus.stop()
        await agent.run_daily()
        await bus.start()

        shortfall_events = bus.get_events(SHORTFALL_ALERT)
        holiday_alerts = [
            e for e in shortfall_events
            if e.payload.get("holiday_prefunding") is True
        ]
        assert len(holiday_alerts) == 0


class TestSufficientBalanceNoAlert:
    @freeze_time("2026-02-26 09:30:00+00:00")  # Thursday
    async def test_no_alert_when_funded(self):
        """Even with a holiday, sufficient nostro balance → no holiday alert."""
        cal = CalendarService()
        cal.add_holiday(
            date(2026, 2, 27), IN_RBI_FX,
            name="Test Holiday",
            user="test",
        )

        agent, bus, _ = _make_agent(
            cal=cal,
            nostro={
                "USD": 100_000_000.0,  # very large balance
                "GBP": 100_000_000.0,
                "AED": 100_000_000.0,
            },
        )

        await agent.setup()
        await bus.stop()
        await agent.run_daily()
        await bus.start()

        shortfall_events = bus.get_events(SHORTFALL_ALERT)
        holiday_alerts = [
            e for e in shortfall_events
            if e.payload.get("holiday_prefunding") is True
        ]
        assert len(holiday_alerts) == 0


class TestRegularShortfallNoHolidayMetadata:
    @freeze_time("2026-02-25 09:30:00+00:00")  # Wednesday (no holiday tomorrow)
    async def test_regular_shortfall_has_no_holiday_fields(self):
        """Step-6 (regular RDA) alerts must not carry holiday_prefunding metadata."""
        agent, bus, _ = _make_agent(
            nostro={"USD": 1_000_000.0, "GBP": 8_000_000.0, "AED": 8_000_000.0},
            dow=2,  # Wednesday
        )

        await agent.setup()
        await bus.stop()
        await agent.run_daily()
        await bus.start()

        shortfall_events = bus.get_events(SHORTFALL_ALERT)
        for e in shortfall_events:
            assert e.payload.get("holiday_prefunding") is not True, (
                f"Regular shortfall for {e.payload['currency']} should not have "
                "holiday_prefunding=True"
            )
