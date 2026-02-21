"""
RDAChecker — nostro pre-funding sufficiency check.

Given the forecasted USD-equivalent volumes per corridor and the current
nostro balances per currency, this module determines whether each RDA
(Respondent / Destination Account) is sufficiently funded.

Severity thresholds:
  CRITICAL  balance < required_native         — will fail to fund without action
  WARNING   balance < required_native × (1 + buffer_pct)
                                              — buffer eroded, action recommended

The prefunding buffer (default 10%) is applied on top of the raw forecast so
the agent recommends moving more than the minimum required.
"""
from __future__ import annotations

from models.domain import RDAShortfall, ShortfallSeverity


# Map corridor key → source currency
_CORRIDOR_TO_CCY: dict[str, str] = {
    "AED_INR": "AED",
    "GBP_INR": "GBP",
    "USD_INR": "USD",
    "EUR_INR": "EUR",
}


class RDAChecker:
    """Check whether nostro balances cover the forecasted corridor volumes."""

    def check(
        self,
        forecast_usd: dict[str, float],       # {corridor: volume in USD equiv}
        nostro_balances: dict[str, float],     # {currency: balance in native ccy}
        rates: dict[str, float],               # {USD_INR, GBP_INR, EUR_INR, AED_INR}
        buffer_pct: float = 0.10,
    ) -> list[RDAShortfall]:
        """
        Return a list of RDAShortfall objects for any underfunded corridor.

        Conversion:  vol_native = vol_usd × (USD_INR / CCY_INR)
        (For USD corridor, vol_native = vol_usd directly.)
        """
        usd_inr   = rates.get("USD_INR", 84.0)
        shortfalls: list[RDAShortfall] = []

        for corridor, vol_usd in forecast_usd.items():
            ccy = _CORRIDOR_TO_CCY.get(corridor)
            if ccy is None:
                continue

            balance = nostro_balances.get(ccy, 0.0)

            # ── Convert USD-equivalent volume → native currency ────────────────
            if ccy == "USD":
                required_native = vol_usd
            else:
                ccy_inr = rates.get(f"{ccy}_INR", usd_inr)
                # vol_usd (USD equiv) × (INR per USD) / (INR per CCY) = vol in CCY
                required_native = vol_usd * usd_inr / ccy_inr

            # ── Apply prefunding buffer ────────────────────────────────────────
            required_with_buffer = required_native * (1.0 + buffer_pct)

            if balance >= required_with_buffer:
                continue  # fully funded

            shortfall_amt = required_with_buffer - balance
            severity = (
                ShortfallSeverity.CRITICAL
                if balance < required_native
                else ShortfallSeverity.WARNING
            )
            shortfalls.append(RDAShortfall(
                currency=ccy,
                required_amount=round(required_with_buffer, 2),
                available_balance=round(balance, 2),
                shortfall=round(shortfall_amt, 2),
                severity=severity,
            ))

        return shortfalls
