"""
Tests for FXAnalystAgent — FX Band Predictor orchestration.

Categories:
    A  Happy path — prediction fetched, events emitted
    B  Fallback — API unreachable, fallback event emitted
    C  Rate move detection — REFORECAST_TRIGGER on >1% move
    D  Integration — FXAnalystAgent + LiquidityAgent end-to-end
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.fx_analyst.agent import FXAnalystAgent
from bus.events import (
    FX_PREDICTION_READY,
    MARKET_BRIEF,
    REFORECAST_TRIGGER,
)
from bus.memory_bus import InMemoryBus


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_PREDICTION = {
    "current_rate": 85.45,
    "prediction_48h": {
        "direction": "UP",
        "range_low": 85.12,
        "range_high": 85.78,
        "most_likely": 85.45,
        "confidence": 0.72,
        "prefunding_guidance": {
            "action": "HOLD",
            "recommended_rate": 85.52,
        },
    },
    "act_on_signal": True,
    "signal_strength": "STRONG",
    "risk_flags": ["High volatility detected"],
}


@pytest.fixture()
def bus() -> InMemoryBus:
    return InMemoryBus()


@pytest.fixture()
def agent(bus: InMemoryBus) -> FXAnalystAgent:
    return FXAnalystAgent(bus=bus)


# ═════════════════════════════════════════════════════════════════════════════
# A. Happy path
# ═════════════════════════════════════════════════════════════════════════════


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_emits_fx_prediction_ready(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=SAMPLE_PREDICTION,
        ):
            await agent.setup()
            await agent.run_daily()

        events = bus.get_events(FX_PREDICTION_READY)
        assert len(events) == 1
        payload = events[0].payload
        assert payload["rates"]["USD_INR"] == 85.45
        assert payload["band"]["direction"] == "UP"
        assert payload["band"]["confidence"] == 0.72
        assert payload["band"]["range_low"] == 85.12
        assert payload["band"]["range_high"] == 85.78
        assert payload["source"] == "fx_band_predictor"
        assert events[0].source_agent == "fx_analyst"

    @pytest.mark.asyncio
    async def test_emits_market_brief(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=SAMPLE_PREDICTION,
        ):
            await agent.setup()
            await agent.run_daily()

        events = bus.get_events(MARKET_BRIEF)
        assert len(events) == 1
        payload = events[0].payload
        assert payload["current_rate"] == 85.45
        assert payload["direction"] == "UP"
        assert payload["action"] == "HOLD"
        assert payload["recommended_rate"] == 85.52
        assert payload["risk_flags"] == ["High volatility detected"]

    @pytest.mark.asyncio
    async def test_caches_last_rate(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=SAMPLE_PREDICTION,
        ):
            await agent.setup()
            await agent.run_daily()

        assert agent._last_rate == 85.45
        assert agent._last_prediction is not None

    @pytest.mark.asyncio
    async def test_no_reforecast_on_first_run(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        """First run has no _last_rate — should NOT emit REFORECAST_TRIGGER."""
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=SAMPLE_PREDICTION,
        ):
            await agent.setup()
            await agent.run_daily()

        assert bus.event_count(REFORECAST_TRIGGER) == 0


# ═════════════════════════════════════════════════════════════════════════════
# B. Fallback — API unreachable
# ═════════════════════════════════════════════════════════════════════════════


class TestFallback:
    @pytest.mark.asyncio
    async def test_emits_fallback_when_api_returns_none(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=None,
        ):
            await agent.setup()
            await agent.run_daily()

        events = bus.get_events(FX_PREDICTION_READY)
        assert len(events) == 1
        payload = events[0].payload
        assert payload["source"] == "fallback"
        assert payload["rates"] == {}
        assert payload["band"] is None
        assert payload["act_on_signal"] is False
        assert "error" in payload

    @pytest.mark.asyncio
    async def test_no_market_brief_on_fallback(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=None,
        ):
            await agent.setup()
            await agent.run_daily()

        assert bus.event_count(MARKET_BRIEF) == 0

    @pytest.mark.asyncio
    async def test_fallback_when_current_rate_missing(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        bad_prediction = {"prediction_48h": {"direction": "UP"}}
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=bad_prediction,
        ):
            await agent.setup()
            await agent.run_daily()

        events = bus.get_events(FX_PREDICTION_READY)
        assert len(events) == 1
        assert events[0].payload["source"] == "fallback"


# ═════════════════════════════════════════════════════════════════════════════
# C. Rate move detection
# ═════════════════════════════════════════════════════════════════════════════


class TestRateMoveDetection:
    @pytest.mark.asyncio
    async def test_emits_reforecast_on_large_move(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        agent._last_rate = 84.0  # Previous rate

        # New rate: 85.45 → ~1.73% move (above 1% threshold)
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=SAMPLE_PREDICTION,
        ):
            await agent.setup()
            await agent.run_daily()

        events = bus.get_events(REFORECAST_TRIGGER)
        assert len(events) == 1
        payload = events[0].payload
        assert payload["rates"]["USD_INR"] == 85.45
        assert payload["previous_rate"] == 84.0
        assert payload["current_rate"] == 85.45
        assert "moved" in payload["reason"].lower()

    @pytest.mark.asyncio
    async def test_no_reforecast_on_small_move(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        agent._last_rate = 85.40  # Previous rate

        # New rate: 85.45 → ~0.06% move (below 1% threshold)
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=SAMPLE_PREDICTION,
        ):
            await agent.setup()
            await agent.run_daily()

        assert bus.event_count(REFORECAST_TRIGGER) == 0

    @pytest.mark.asyncio
    async def test_no_reforecast_just_below_threshold(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        # 85.45 vs 84.60 → ~1.005% → just above threshold → triggers
        # 85.45 vs 85.00 → ~0.53% → below threshold → no trigger
        agent._last_rate = 85.00

        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=SAMPLE_PREDICTION,
        ):
            await agent.setup()
            await agent.run_daily()

        assert bus.event_count(REFORECAST_TRIGGER) == 0


# ═════════════════════════════════════════════════════════════════════════════
# D. Correlation ID consistency
# ═════════════════════════════════════════════════════════════════════════════


class TestCorrelationId:
    @pytest.mark.asyncio
    async def test_same_correlation_id_across_events(
        self, bus: InMemoryBus, agent: FXAnalystAgent
    ) -> None:
        with patch(
            "data.fx_predictor_client.fetch_prediction",
            return_value=SAMPLE_PREDICTION,
        ):
            await agent.setup()
            await agent.run_daily()

        pred_event = bus.last_event(FX_PREDICTION_READY)
        brief_event = bus.last_event(MARKET_BRIEF)
        assert pred_event is not None
        assert brief_event is not None
        assert pred_event.correlation_id == brief_event.correlation_id
