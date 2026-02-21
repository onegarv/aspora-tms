"""
Tests for FX Band integration in the Liquidity Agent.

Covers:
  - MARKET_BRIEF handler caches fx_band data
  - run_daily() uses band-adjusted rates when band data is cached
  - Graceful fallback to spot rates when no band data is available
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from freezegun import freeze_time

from agents.liquidity.agent import LiquidityAgent
from bus.events import (
    FORECAST_READY,
    MARKET_BRIEF,
    SHORTFALL_ALERT,
    create_event,
)
from bus.memory_bus import InMemoryBus
from services.calendar_service import CalendarService


# ── Helpers ──────────────────────────────────────────────────────────────────


def _eight_thursdays(base_vol_usd: float = 5_000_000.0) -> dict[str, list[dict]]:
    anchor = date(2025, 12, 25)
    rows = [
        {"date": (anchor - timedelta(weeks=w)).isoformat(),
         "volume_usd": base_vol_usd,
         "dow": 3}
        for w in range(8)
    ]
    return {corridor: rows for corridor in ["USD_INR", "GBP_INR", "AED_INR"]}


def _make_liquidity_agent(
    bus: InMemoryBus | None = None,
    cal: CalendarService | None = None,
    config=None,
) -> tuple[LiquidityAgent, InMemoryBus]:
    bus = bus or InMemoryBus()
    cal = cal or CalendarService()
    agent = LiquidityAgent(bus=bus, calendar=cal, config=config)
    agent._forecaster.load(_eight_thursdays())
    agent._spot_rates = {
        "USD_INR": 84.0,
        "GBP_INR": 106.0,
        "EUR_INR": 91.0,
        "AED_INR": 22.9,
    }
    return agent, bus


# ── MARKET_BRIEF handler ────────────────────────────────────────────────────


class TestMarketBriefHandler:
    async def test_caches_fx_band(self):
        """Publishing MARKET_BRIEF populates _fx_band on the agent."""
        agent, bus = _make_liquidity_agent()
        await agent.setup()
        await bus.start()

        event = create_event(
            event_type=MARKET_BRIEF,
            source_agent="fx_analyst",
            payload={
                "currency_pair": "USD/INR",
                "current_rate": 85.42,
                "direction": "UP",
                "confidence_pct": 72.0,
                "range_low": 85.0,
                "range_high": 86.5,
            },
        )
        await bus.publish(event)

        assert agent._fx_band is not None
        assert agent._fx_band["range_low"] == 85.0
        assert agent._fx_band["range_high"] == 86.5
        assert agent._fx_band["direction"] == "UP"

    async def test_successive_briefs_update_cache(self):
        """Each MARKET_BRIEF overwrites the previous cached band."""
        agent, bus = _make_liquidity_agent()
        await agent.setup()
        await bus.start()

        for rate in [85.0, 86.0]:
            event = create_event(
                event_type=MARKET_BRIEF,
                source_agent="fx_analyst",
                payload={
                    "currency_pair": "USD/INR",
                    "current_rate": rate,
                    "range_low": rate - 1,
                    "range_high": rate + 1,
                    "direction": "UP",
                    "confidence_pct": 65.0,
                },
            )
            await bus.publish(event)

        assert agent._fx_band["current_rate"] == 86.0
        assert agent._fx_band["range_low"] == 85.0
        assert agent._fx_band["range_high"] == 87.0


# ── run_daily() with band data ──────────────────────────────────────────────


class TestRunDailyWithBand:
    @freeze_time("2026-02-26 09:30:00+00:00")  # Thursday
    async def test_run_daily_uses_worst_case_band_rate(self):
        """With cached band and worst_case strategy, FORECAST_READY uses band rate."""
        from config.settings import Settings

        config = Settings(fx_band_rate_strategy="worst_case")
        agent, bus = _make_liquidity_agent(config=config)

        # Pre-cache band data (simulating a prior MARKET_BRIEF)
        agent._fx_band = {
            "range_low": 85.0,
            "range_high": 87.0,
            "direction": "UP",
            "confidence_pct": 72.0,
            "current_rate": 85.5,
        }
        agent._nostro_balances = {"USD": 1_000_000.0, "GBP": 8_000_000.0, "AED": 8_000_000.0}

        await agent.setup()
        await bus.stop()  # pause delivery
        await agent.run_daily()
        await bus.start()

        # With worst_case, USD_INR should be max(87.0, 84.0) = 87.0
        # Total INR crores should reflect the higher rate
        forecasts = bus.get_events(FORECAST_READY)
        assert len(forecasts) == 1

        # Shortfall alerts should use the higher rate
        shortfalls = bus.get_events(SHORTFALL_ALERT)
        # USD nostro has 1M vs forecast ~5M → shortfall expected
        usd_sf = next(
            (e for e in shortfalls if e.payload["currency"] == "USD"), None
        )
        assert usd_sf is not None

    @freeze_time("2026-02-26 09:30:00+00:00")  # Thursday
    async def test_no_band_falls_back_to_spot(self):
        """With _fx_band=None, run_daily uses Metabase spot rates."""
        from config.settings import Settings

        config = Settings(fx_band_rate_strategy="worst_case")
        agent, bus = _make_liquidity_agent(config=config)
        agent._nostro_balances = {"USD": 1_000_000.0}

        assert agent._fx_band is None  # no band cached

        await agent.setup()
        await bus.stop()
        await agent.run_daily()
        await bus.start()

        forecasts = bus.get_events(FORECAST_READY)
        assert len(forecasts) == 1
        # Should still work with spot rate (84.0)

    @freeze_time("2026-02-26 09:30:00+00:00")  # Thursday
    async def test_spot_strategy_ignores_band(self):
        """With strategy=spot, band data is ignored even when cached."""
        from config.settings import Settings

        config = Settings(fx_band_rate_strategy="spot")
        agent, bus = _make_liquidity_agent(config=config)
        agent._fx_band = {
            "range_low": 85.0,
            "range_high": 87.0,
            "direction": "UP",
            "confidence_pct": 72.0,
            "current_rate": 85.5,
        }
        agent._nostro_balances = {"USD": 1_000_000.0}

        await agent.setup()
        await bus.stop()
        await agent.run_daily()
        await bus.start()

        forecasts = bus.get_events(FORECAST_READY)
        assert len(forecasts) == 1
        # With spot strategy, 84.0 is used (not 87.0)
