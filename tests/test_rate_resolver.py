"""
Tests for agents.liquidity.rate_resolver.RateResolver.

Verifies the three strategies (worst_case, midpoint, spot) and
graceful fallback when fx_band data is None or incomplete.
"""

from __future__ import annotations

import pytest

from agents.liquidity.rate_resolver import RateResolver

_SPOT = {
    "USD_INR": 84.0,
    "GBP_INR": 106.0,
    "EUR_INR": 91.0,
    "AED_INR": 22.9,
}

_BAND = {
    "range_low": 85.0,
    "range_high": 87.0,
    "direction": "UP",
    "confidence_pct": 72.0,
    "current_rate": 85.5,
}


class TestWorstCase:
    def test_uses_range_high(self):
        resolver = RateResolver()
        result = resolver.resolve(_SPOT, _BAND, "worst_case")
        assert result["USD_INR"] == 87.0

    def test_uses_spot_when_spot_is_higher_than_range_high(self):
        """worst_case = max(range_high, spot)."""
        resolver = RateResolver()
        spot = {**_SPOT, "USD_INR": 90.0}
        result = resolver.resolve(spot, _BAND, "worst_case")
        assert result["USD_INR"] == 90.0


class TestMidpoint:
    def test_averages_band(self):
        resolver = RateResolver()
        result = resolver.resolve(_SPOT, _BAND, "midpoint")
        assert result["USD_INR"] == pytest.approx(86.0)

    def test_different_range(self):
        resolver = RateResolver()
        band = {**_BAND, "range_low": 83.0, "range_high": 85.0}
        result = resolver.resolve(_SPOT, band, "midpoint")
        assert result["USD_INR"] == pytest.approx(84.0)


class TestSpotStrategy:
    def test_ignores_band(self):
        resolver = RateResolver()
        result = resolver.resolve(_SPOT, _BAND, "spot")
        assert result == _SPOT

    def test_returns_copy(self):
        resolver = RateResolver()
        result = resolver.resolve(_SPOT, _BAND, "spot")
        result["USD_INR"] = 999.0
        assert _SPOT["USD_INR"] == 84.0  # original unchanged


class TestNoneBandFallthrough:
    def test_none_band_returns_spot(self):
        resolver = RateResolver()
        result = resolver.resolve(_SPOT, None, "worst_case")
        assert result == _SPOT

    def test_none_band_midpoint_returns_spot(self):
        resolver = RateResolver()
        result = resolver.resolve(_SPOT, None, "midpoint")
        assert result == _SPOT


class TestIncompleteBand:
    def test_missing_range_high_returns_spot(self):
        resolver = RateResolver()
        band = {"range_low": 85.0, "range_high": None}
        result = resolver.resolve(_SPOT, band, "worst_case")
        assert result["USD_INR"] == 84.0

    def test_missing_range_low_returns_spot(self):
        resolver = RateResolver()
        band = {"range_low": None, "range_high": 87.0}
        result = resolver.resolve(_SPOT, band, "midpoint")
        assert result["USD_INR"] == 84.0


class TestNonUsdPairsUnaffected:
    def test_gbp_unchanged(self):
        resolver = RateResolver()
        result = resolver.resolve(_SPOT, _BAND, "worst_case")
        assert result["GBP_INR"] == 106.0

    def test_eur_unchanged(self):
        resolver = RateResolver()
        result = resolver.resolve(_SPOT, _BAND, "midpoint")
        assert result["EUR_INR"] == 91.0

    def test_aed_unchanged(self):
        resolver = RateResolver()
        result = resolver.resolve(_SPOT, _BAND, "worst_case")
        assert result["AED_INR"] == 22.9
