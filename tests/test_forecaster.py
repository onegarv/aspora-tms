"""
Tests for agents.liquidity.forecaster.VolumeForecaster.

Covers the floor rule: forecast = max(EMA, last_week_actual),
single-sample edge case, empty data, and confidence thresholds.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents.liquidity.forecaster import VolumeForecaster
from models.domain import ForecastConfidence


def _make_rows(volumes: list[float], dow: int, start_date: str = "2025-12-01") -> list[dict]:
    """Build volume rows for a single corridor, all sharing the same weekday."""
    from datetime import timedelta

    base = date.fromisoformat(start_date)
    rows = []
    for i, vol in enumerate(volumes):
        d = base - timedelta(weeks=i)
        rows.append({"date": d.isoformat(), "volume_usd": vol, "dow": dow})
    return rows


class TestFloorRuleActive:
    """When recent actual > EMA, the floor lifts the forecast."""

    def test_declining_volumes_floor_wins(self):
        # Last week = 8M, earlier weeks declining → EMA < 8M → floor = 8M
        target = date(2025, 12, 1)  # Monday (dow=0)
        volumes = [8_000_000, 6_000_000, 5_000_000, 4_500_000, 4_000_000, 3_500_000]
        rows = _make_rows(volumes, dow=0)

        f = VolumeForecaster(lookback_weeks=8, decay=0.85)
        f.load({"USD_INR": rows})
        result = f.forecast(target)

        assert result["USD_INR"] == 8_000_000.0


class TestFloorRuleInactive:
    """When EMA > recent actual, the EMA wins (floor has no effect)."""

    def test_growing_volumes_ema_wins(self):
        # Last week = 3M, earlier weeks much larger → EMA > 3M → EMA wins
        target = date(2025, 12, 1)  # Monday
        volumes = [3_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000]
        rows = _make_rows(volumes, dow=0)

        f = VolumeForecaster(lookback_weeks=8, decay=0.85)
        f.load({"USD_INR": rows})
        result = f.forecast(target)

        # EMA weighted toward older, larger values → > 3M
        assert result["USD_INR"] > 3_000_000.0


class TestSingleSample:
    """With only one data point, forecast = that value (EMA == actual)."""

    def test_single_data_point(self):
        target = date(2025, 12, 1)
        rows = [{"date": "2025-11-24", "volume_usd": 5_500_000.0, "dow": 0}]

        f = VolumeForecaster()
        f.load({"AED_INR": rows})
        result = f.forecast(target)

        assert result["AED_INR"] == 5_500_000.0


class TestEmptyData:
    """No matching weekday data → forecast = 0.0."""

    def test_no_matching_weekday(self):
        target = date(2025, 12, 1)  # Monday (dow=0)
        # All rows are Tuesday (dow=1) — no Monday data
        rows = [{"date": "2025-11-25", "volume_usd": 1_000_000.0, "dow": 1}]

        f = VolumeForecaster()
        f.load({"GBP_INR": rows})
        result = f.forecast(target)

        assert result["GBP_INR"] == 0.0

    def test_empty_volumes(self):
        target = date(2025, 12, 1)
        f = VolumeForecaster()
        f.load({})
        result = f.forecast(target)

        assert result == {}


class TestConfidenceLevels:
    """Verify HIGH/MEDIUM/LOW thresholds based on sample counts."""

    def _build_forecaster(self, samples_per_corridor: int, target: date) -> VolumeForecaster:
        dow = target.weekday()
        rows = _make_rows([1_000_000.0] * samples_per_corridor, dow=dow)
        f = VolumeForecaster()
        f.load({"USD_INR": rows, "GBP_INR": rows, "EUR_INR": rows, "AED_INR": rows})
        return f

    def test_high_confidence(self):
        target = date(2025, 12, 1)
        f = self._build_forecaster(6, target)
        assert f.confidence(target) == ForecastConfidence.HIGH

    def test_medium_confidence(self):
        target = date(2025, 12, 1)
        f = self._build_forecaster(4, target)
        assert f.confidence(target) == ForecastConfidence.MEDIUM

    def test_low_confidence(self):
        target = date(2025, 12, 1)
        f = self._build_forecaster(2, target)
        assert f.confidence(target) == ForecastConfidence.LOW

    def test_no_data_confidence(self):
        target = date(2025, 12, 1)
        f = VolumeForecaster()
        f.load({})
        assert f.confidence(target) == ForecastConfidence.LOW
