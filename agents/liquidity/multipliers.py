"""
MultiplierEngine — applies day-specific volume multipliers.

Multiplier stack (per SPEC §3.2.2):
  - Payday  : days 25–last of month AND days 1–3 of next month → 1.4×
  - Holiday : day before IN_RBI_FX holiday → 1.2×
              day after  IN_RBI_FX holiday → 0.6×

Total multiplier = product of all active factors, capped at
settings.max_total_multiplier (default 2.5×).

Note: FX-elasticity (bullish/bearish INR signal) is intentionally absent.
That is the responsibility of the FX Analyst Agent (SPEC §3.3.2).
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from services.calendar_service import IN_RBI_FX

if TYPE_CHECKING:
    from services.calendar_service import CalendarService


_PAYDAY_START = 25   # day 25+ of current month
_PAYDAY_END   = 3    # day 1–3 of next month
_PAYDAY_MULT  = 1.4


class MultiplierEngine:
    """Compute per-day volume multipliers from the payday and holiday stack."""

    def compute(
        self,
        target_date: date,
        calendar: "CalendarService",
    ) -> dict[str, float]:
        """
        Return active multipliers for target_date.

        Example return:  {"payday": 1.4, "holiday": 1.2}
        Empty dict means no adjustment (implicitly 1.0×).
        """
        multipliers: dict[str, float] = {}

        # ── Payday multiplier ─────────────────────────────────────────────────
        dom = target_date.day
        if dom >= _PAYDAY_START or dom <= _PAYDAY_END:
            multipliers["payday"] = _PAYDAY_MULT

        # ── Holiday multiplier ────────────────────────────────────────────────
        if calendar.is_day_before_holiday(target_date, IN_RBI_FX):
            multipliers["holiday"] = 1.2
        elif calendar.is_day_after_holiday(target_date, IN_RBI_FX):
            multipliers["holiday"] = 0.6

        return multipliers

    def total(self, multipliers: dict[str, float], cap: float = 2.5) -> float:
        """Product of all multiplier values, capped at `cap`."""
        result = 1.0
        for m in multipliers.values():
            result *= m
        return min(result, cap)

    def apply(
        self,
        base_usd: float,
        multipliers: dict[str, float],
        cap: float = 2.5,
    ) -> float:
        """Scale base_usd by the combined multiplier and return rounded result."""
        return round(base_usd * self.total(multipliers, cap), 2)
