"""
RateResolver — selects the effective FX rate from spot rates + optional FX band data.

The FX Band Predictor produces USD/INR band predictions only.  This module
overrides USD_INR when band data is available and the configured strategy
is not "spot".  All other currency pairs pass through unchanged.

Strategies:
    worst_case — use range_high (most expensive INR buying rate → conservative)
    midpoint   — (range_low + range_high) / 2
    spot       — ignore band, return spot_rates unchanged
"""

from __future__ import annotations

from typing import Any


class RateResolver:
    """Stateless resolver: spot rates + optional FX band → effective rates."""

    def resolve(
        self,
        spot_rates: dict[str, float],
        fx_band: dict[str, Any] | None,
        strategy: str,
    ) -> dict[str, float]:
        """
        Return effective rates for RDA calculations.

        Parameters
        ----------
        spot_rates : Current Metabase spot rates, e.g. {"USD_INR": 84.0, ...}
        fx_band    : Cached MARKET_BRIEF data with keys range_low, range_high.
                     None if no MARKET_BRIEF has been received.
        strategy   : "worst_case" | "midpoint" | "spot"

        Returns
        -------
        A new dict with the same keys as spot_rates.  Only USD_INR may differ.
        """
        effective = dict(spot_rates)

        if fx_band is None or strategy == "spot":
            return effective

        range_low = fx_band.get("range_low")
        range_high = fx_band.get("range_high")

        if range_low is None or range_high is None:
            return effective

        if strategy == "worst_case":
            effective["USD_INR"] = max(
                float(range_high), spot_rates.get("USD_INR", 84.0)
            )
        elif strategy == "midpoint":
            effective["USD_INR"] = round(
                (float(range_low) + float(range_high)) / 2, 4
            )

        return effective
