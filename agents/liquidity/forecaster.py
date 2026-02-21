"""
VolumeForecaster — same-weekday exponential moving average.

Algorithm (per SPEC §3.2.1):
  For each corridor and target weekday, collect the N most-recent data points
  that share that weekday (e.g. all Mondays over the last 8 weeks).  Apply
  exponential decay weights so the most recent observation carries the most
  weight.  Return the weighted average as the forecast USD volume.

  w_k = decay^k   where k=0 is the most-recent observation.

Confidence is based on how many same-weekday samples are available.
"""
from __future__ import annotations

from datetime import date

from models.domain import ForecastConfidence


_DEFAULT_LOOKBACK_WEEKS: int   = 8
_DEFAULT_DECAY:          float = 0.85


class VolumeForecaster:
    """Forecast next-day USD corridor volume using same-weekday EMA."""

    def __init__(
        self,
        lookback_weeks: int = _DEFAULT_LOOKBACK_WEEKS,
        decay: float        = _DEFAULT_DECAY,
    ) -> None:
        self.lookback_weeks = lookback_weeks
        self.decay          = decay
        # {corridor: [{date, volume_usd, dow}]}
        self._volumes: dict[str, list[dict]] = {}

    # ── Data loading ──────────────────────────────────────────────────────────

    def load(self, volumes: dict[str, list[dict]]) -> None:
        """
        Replace internal volume history.

        Expected format (from data.metabase.fetch_corridor_volumes):
            {
              "AED_INR": [{"date": "2025-11-29", "volume_usd": 4_500_000.0, "dow": 5}, ...],
              ...
            }
        """
        self._volumes = volumes

    # ── Forecasting ───────────────────────────────────────────────────────────

    def forecast(self, target_date: date) -> dict[str, float]:
        """
        Return {corridor: forecasted_volume_usd} for target_date.

        Uses the N most-recent observations matching target_date's weekday,
        weighted by exponential decay.  Returns 0.0 for corridors with no data.
        """
        target_dow = target_date.weekday()
        result: dict[str, float] = {}

        for corridor, rows in self._volumes.items():
            # Filter to same day-of-week, most recent first
            same_dow = sorted(
                [r for r in rows if r.get("dow") == target_dow],
                key=lambda r: r["date"],
                reverse=True,
            )[: self.lookback_weeks]

            if not same_dow:
                result[corridor] = 0.0
                continue

            weights   = [self.decay ** i for i in range(len(same_dow))]
            total_w   = sum(weights)
            forecast  = sum(s["volume_usd"] * w for s, w in zip(same_dow, weights)) / total_w
            last_week_actual = same_dow[0]["volume_usd"]
            result[corridor] = round(max(forecast, last_week_actual), 2)

        return result

    def confidence(self, target_date: date) -> ForecastConfidence:
        """
        Derive forecast confidence from data availability.

        HIGH   ≥ 6 same-weekday samples per corridor (avg)
        MEDIUM ≥ 3
        LOW    < 3
        """
        if not self._volumes:
            return ForecastConfidence.LOW

        target_dow = target_date.weekday()
        counts = [
            len([r for r in rows if r.get("dow") == target_dow])
            for rows in self._volumes.values()
        ]
        avg = sum(counts) / len(counts)

        if avg >= 6:
            return ForecastConfidence.HIGH
        if avg >= 3:
            return ForecastConfidence.MEDIUM
        return ForecastConfidence.LOW
