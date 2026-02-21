"""
LiquidityAgent — daily volume forecaster and RDA sufficiency monitor.

Daily routine (06:00 IST):
  1. Refresh FX rates and corridor volume history from Metabase.
  2. Run same-weekday EMA forecast per corridor (VolumeForecaster).
  3. Apply payday / holiday multiplier stack (MultiplierEngine).
  4. Aggregate to total INR crores; emit FORECAST_READY (DailyForecast payload).
  5. Compare adjusted volumes against current nostro balances (RDAChecker).
  6. Emit SHORTFALL_ALERT for each underfunded corridor.

Event listeners:
  fx.reforecast.trigger    → re-run forecast immediately (e.g. rate moved >1%)
  ops.nostro.balance.update → update cached nostro balances

Design invariants:
  1. This agent NEVER emits a directional FX signal (bullish/bearish INR).
     That responsibility belongs exclusively to the FX Analyst Agent.
  2. All I/O to Metabase is dispatched to a thread-pool executor so the
     asyncio event loop is never blocked.
  3. Handlers are idempotent; re-triggering is safe.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

from agents.base import BaseAgent
from agents.liquidity.forecaster import VolumeForecaster
from agents.liquidity.multipliers import MultiplierEngine
from agents.liquidity.rda_checker import RDAChecker
from bus.events import (
    FORECAST_READY,
    FX_PREDICTION_READY,
    NOSTRO_BALANCE_UPDATE,
    REFORECAST_TRIGGER,
    SHORTFALL_ALERT,
    Event,
)
from config.settings import settings as default_settings

if TYPE_CHECKING:
    from bus.base import EventBus
    from config.settings import Settings
    from services.calendar_service import CalendarService

# Fallback spot rates used if Metabase is unreachable at startup
_FALLBACK_RATES: dict[str, float] = {
    "USD_INR": 84.0,
    "GBP_INR": 106.0,
    "EUR_INR":  91.0,
    "AED_INR":  22.9,
}


class LiquidityAgent(BaseAgent):
    """
    Forecasts daily INR liquidity demand and alerts on RDA shortfalls.

    Parameters
    ----------
    bus       : EventBus instance (Redis or in-memory)
    calendar  : CalendarService for holiday / payday lookups
    config    : Settings override (defaults to global settings singleton)
    """

    def __init__(
        self,
        bus: "EventBus",
        calendar: "CalendarService",
        config: "Settings | None" = None,
    ) -> None:
        super().__init__("liquidity", bus)
        self.calendar = calendar
        self.config   = config or default_settings

        self._spot_rates:      dict[str, float]  = dict(_FALLBACK_RATES)
        self._nostro_balances: dict[str, float]  = {}

        self._forecaster       = VolumeForecaster(lookback_weeks=self.config.forecast_lookback_weeks)
        self._multiplier_engine = MultiplierEngine()
        self._rda_checker      = RDAChecker()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Register event listeners and do initial Metabase data load."""
        await self.listen(REFORECAST_TRIGGER,    self._handle_reforecast)
        await self.listen(NOSTRO_BALANCE_UPDATE, self._handle_nostro_update)
        await self.listen(FX_PREDICTION_READY,   self._handle_fx_prediction)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_market_data)

    async def run_daily(self) -> None:
        """
        06:00 IST entry-point (called by APScheduler).

        1. Refresh market data.
        2. Compute forecast.
        3. Apply multipliers.
        4. Emit FORECAST_READY.
        5. Emit SHORTFALL_ALERT for each underfunded corridor.
        """
        today = date.today()
        self.logger.info("run_daily start  date=%s", today.isoformat())

        # ── 1. Refresh market data (blocking I/O → thread pool) ───────────────
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_market_data)

        # ── 2. Forecast raw USD volumes per corridor ──────────────────────────
        forecast_usd = self._forecaster.forecast(today)
        confidence   = self._forecaster.confidence(today)

        # ── 3. Apply multiplier stack ─────────────────────────────────────────
        multipliers = self._multiplier_engine.compute(today, self.calendar)
        adjusted_usd = {
            corridor: self._multiplier_engine.apply(
                vol, multipliers, self.config.max_total_multiplier
            )
            for corridor, vol in forecast_usd.items()
        }

        # ── 4. Aggregate to INR crores ────────────────────────────────────────
        usd_inr = self._spot_rates.get("USD_INR", 84.0)
        currency_split: dict[str, float] = {}
        for corridor, vol_usd in adjusted_usd.items():
            ccy              = corridor.split("_")[0]          # "AED_INR" → "AED"
            inr_crores       = round(vol_usd * usd_inr / 1e7, 4)
            currency_split[ccy] = inr_crores

        total_inr_crores = round(sum(currency_split.values()), 4)

        # ── 5. Emit FORECAST_READY ────────────────────────────────────────────
        correlation_id = str(uuid.uuid4())
        await self.emit(
            FORECAST_READY,
            {
                "forecast_date":      today.isoformat(),
                "total_inr_crores":   total_inr_crores,
                "confidence":         confidence.value,
                "currency_split":     currency_split,
                "multipliers_applied": multipliers,
                "created_at":         datetime.now(timezone.utc).isoformat(),
            },
            correlation_id=correlation_id,
        )
        self.logger.info(
            "FORECAST_READY  total_inr_crores=%.2f  confidence=%s  multipliers=%s",
            total_inr_crores, confidence.value, multipliers,
        )

        # ── 6. RDA shortfall check ────────────────────────────────────────────
        if not self._nostro_balances:
            self.logger.info("No nostro balances available — skipping RDA check")
            return

        shortfalls = self._rda_checker.check(
            forecast_usd=adjusted_usd,
            nostro_balances=self._nostro_balances,
            rates=self._spot_rates,
            buffer_pct=self.config.prefunding_buffer_pct,
        )
        for sf in shortfalls:
            await self.emit(
                SHORTFALL_ALERT,
                {
                    "currency":         sf.currency,
                    "required_amount":  sf.required_amount,
                    "available_balance": sf.available_balance,
                    "shortfall":        sf.shortfall,
                    "severity":         sf.severity.value,
                    "detected_at":      datetime.now(timezone.utc).isoformat(),
                },
                correlation_id=correlation_id,
            )
            self.logger.warning(
                "SHORTFALL_ALERT  currency=%s severity=%s shortfall=%.2f",
                sf.currency, sf.severity.value, sf.shortfall,
            )

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _handle_reforecast(self, event: Event) -> None:
        """
        FX Analyst Agent fires REFORECAST_TRIGGER when a rate moves >1%.
        Optionally carries updated rates in the payload.
        """
        self.logger.info(
            "Reforecast requested by %s  correlation_id=%s",
            event.source_agent, event.correlation_id,
        )
        new_rates: dict[str, Any] = event.payload.get("rates", {})
        if new_rates:
            self._spot_rates.update(
                {k: float(v) for k, v in new_rates.items()}
            )
        await self.run_daily()

    async def _handle_nostro_update(self, event: Event) -> None:
        """
        Operations Agent publishes NOSTRO_BALANCE_UPDATE after each
        balance refresh.  Payload may be {balances: {...}} or flat {...}.

        NOTE: OpsAgent serialises Decimal balances as strings (e.g. "1000000").
        We accept int, float, and str so the types always round-trip correctly.
        """
        payload_balances: dict = event.payload.get("balances", event.payload)
        updated: dict[str, float] = {}
        for k, v in payload_balances.items():
            try:
                updated[k] = float(v)
            except (TypeError, ValueError):
                pass
        self._nostro_balances.update(updated)
        self.logger.info(
            "Nostro balances updated: %s",
            {k: round(v, 0) for k, v in self._nostro_balances.items()},
        )

    async def _handle_fx_prediction(self, event: Event) -> None:
        """
        FX Analyst Agent fires FX_PREDICTION_READY after fetching from
        the FX Band Predictor API.  Update spot rates and re-run forecast.
        """
        self.logger.info(
            "FX prediction received from %s  correlation_id=%s",
            event.source_agent, event.correlation_id,
        )
        new_rates: dict[str, Any] = event.payload.get("rates", {})
        if new_rates:
            self._spot_rates.update(
                {k: float(v) for k, v in new_rates.items()}
            )
            self.logger.info(
                "Spot rates updated from FX prediction: %s",
                {k: round(v, 4) for k, v in self._spot_rates.items()},
            )
        await self.run_daily()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_market_data(self) -> None:
        """
        Blocking: pull FX rates and volume history from Metabase.
        Safe to call from a thread-pool executor.
        Falls back to cached values if Metabase is unreachable.
        """
        try:
            from data.metabase import fetch_live_rates, fetch_corridor_volumes

            rates = fetch_live_rates()
            if rates:
                self._spot_rates = rates

            lookback_days = self.config.forecast_lookback_weeks * 7 + 14
            volumes = fetch_corridor_volumes(lookback_days=lookback_days)
            if volumes:
                self._forecaster.load(volumes)

            self.logger.info("Market data refreshed from Metabase")

        except Exception as exc:
            self.logger.warning(
                "Metabase data load failed (%s: %s) — using cached values",
                type(exc).__name__, exc,
            )
