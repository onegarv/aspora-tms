"""
FXAnalystAgent — bridge adapter between FX Band Predictor and the TMS event bus.

Architecture:
  FX Band Predictor (port 8001) → [this agent] → TMS Redis event bus

What it does:
  1. Polls GET http://localhost:8001/predict every 30 s during INR trading hours
     (09:00–17:00 IST, Mon–Fri).
  2. Always emits `fx.market.brief` with the current rate, direction, and
     confidence so downstream consumers (dashboard, ops agent) stay informed.
  3. When act_on_signal is True and direction is actionable (UP or DOWN), emits
     `fx.deal.instruction`.  A 15-minute cooldown prevents duplicate instructions
     for the same direction signal.
  4. Responds to `fx.reforecast.trigger` by immediately polling (bypasses the
     30 s timer and the trading-hours guard).
  5. run_daily() fires at 09:00 IST to publish the opening market brief.

Event payloads emitted
──────────────────────
fx.market.brief
  currency_pair    str          "USD/INR"
  current_rate     float        e.g. 85.42
  direction        str          "UP" | "DOWN" | "NEUTRAL"
  confidence_pct   float        0–100
  range_low        float
  range_high       float
  action           str          "HOLD" | "CONVERT_NOW" | "CONVERT_PARTIAL"
  message          str          human-readable summary
  act_on_signal    bool
  source           str          "fx_band_predictor"
  predictor_url    str

fx.deal.instruction
  id               str          "FXDEAL-<uuid>"
  currency_pair    str          "USD/INR"
  amount_foreign   float        USD amount (capped at max_single_deal_usd)
  amount_inr       float        amount_foreign × current_rate
  deal_type        str          "spot"
  target_rate      float        current rate at signal time
  time_window_start str         ISO-8601 UTC
  time_window_end   str         ISO-8601 UTC (start + 2 h)
  tranche_number   int          1
  total_tranches   int          1
  direction        str          "DOWN" (CONVERT_NOW) or "UP" (HOLD — rare)
  confidence_pct   float
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, time, timedelta, timezone
from typing import Any, TYPE_CHECKING

import httpx
from zoneinfo import ZoneInfo

from agents.base import BaseAgent
from bus.events import (
    DEAL_INSTRUCTION,
    MARKET_BRIEF,
    REFORECAST_TRIGGER,
    FORECAST_READY,
    Event,
)
from config.settings import settings

if TYPE_CHECKING:
    from bus.base import EventBus

logger = logging.getLogger("tms.agent.fx_analyst")

TZ_IST = ZoneInfo("Asia/Kolkata")

# Default daily USD volume mirroring the FX Band Predictor's own assumption,
# but capped to the TMS risk limit so we never exceed a single-deal ceiling.
_DEFAULT_DAILY_VOL_USD: float = 9_500_000.0

# How long (minutes) to suppress a repeat deal instruction for the same direction.
_DEAL_COOLDOWN_MIN: int = 15


def _now_ist() -> datetime:
    return datetime.now(TZ_IST)


def _is_trading_hours() -> bool:
    """True during INR market hours: 09:00–17:00 IST, Monday–Friday."""
    now = _now_ist()
    return now.weekday() < 5 and time(9, 0) <= now.time() <= time(17, 0)


class FXAnalystAgent(BaseAgent):
    """
    Bridge adapter: polls FX Band Predictor REST API and emits TMS bus events.
    """

    def __init__(
        self,
        bus: "EventBus",
        fx_predictor_url: str = "http://localhost:8001",
        poll_interval_sec: int = 30,
    ) -> None:
        super().__init__("fx_analyst", bus)
        self._url = fx_predictor_url.rstrip("/")
        self._poll_interval_sec = poll_interval_sec

        # Deduplication state
        self._last_deal_direction: str | None = None
        self._last_deal_ts: datetime | None = None

        self._poll_task: asyncio.Task | None = None

    # ── BaseAgent interface ───────────────────────────────────────────────────

    async def setup(self) -> None:
        await self.listen(REFORECAST_TRIGGER, self._handle_reforecast)
        await self.listen(FORECAST_READY, self._handle_forecast_ready)

    async def run_daily(self) -> None:
        """Morning brief — poll immediately at 09:00 IST."""
        logger.info("run_daily: emitting opening market brief")
        await self._poll_and_emit(force=True)

    async def start(self) -> None:
        await super().start()
        self._poll_task = asyncio.create_task(
            self._polling_loop(), name="fx_analyst.poll"
        )
        logger.info(
            "FXAnalystAgent started — polling %s every %ds during trading hours",
            self._url,
            self._poll_interval_sec,
        )

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _handle_reforecast(self, event: Event) -> None:
        """Immediately re-poll when another agent requests a reforecast."""
        logger.info(
            "reforecast.trigger received (correlation=%s) — polling now",
            event.correlation_id,
        )
        await self._poll_and_emit(force=True, correlation_id=event.correlation_id)

    async def _handle_forecast_ready(self, event: Event) -> None:
        """On a new liquidity forecast, do a fresh market brief."""
        logger.info("forecast.daily.ready received — refreshing market brief")
        await self._poll_and_emit(force=False, correlation_id=event.correlation_id)

    # ── Polling loop ──────────────────────────────────────────────────────────

    async def _polling_loop(self) -> None:
        while True:
            try:
                if _is_trading_hours():
                    await self._poll_and_emit()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("poll cycle error (will retry): %s", exc)
            await asyncio.sleep(self._poll_interval_sec)

    # ── Core: fetch → translate → emit ───────────────────────────────────────

    async def _poll_and_emit(
        self,
        force: bool = False,
        correlation_id: str | None = None,
    ) -> None:
        """
        Fetch /predict, emit fx.market.brief, and (when actionable) fx.deal.instruction.

        force=True bypasses the trading-hours guard and the deal cooldown.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._url}/predict")
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except Exception as exc:
            logger.error("failed to reach FX Band Predictor at %s: %s", self._url, exc)
            return

        if "error" in data:
            logger.warning("predictor returned error payload: %s", data["error"])
            return

        corr = correlation_id or str(uuid.uuid4())

        await self._emit_market_brief(data, corr)

        if self._should_emit_deal(data, force):
            await self._emit_deal_instruction(data, corr)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _should_emit_deal(self, data: dict, force: bool) -> bool:
        """Return True only when there is a fresh, actionable deal signal."""
        if not data.get("act_on_signal"):
            return False

        pred = data.get("prediction_48h", {})
        direction = pred.get("direction", "NEUTRAL")

        if direction not in ("UP", "DOWN"):
            return False

        # Cooldown: suppress repeat instructions for the same direction
        if not force and self._last_deal_direction == direction and self._last_deal_ts:
            age = datetime.now(timezone.utc) - self._last_deal_ts
            if age < timedelta(minutes=_DEAL_COOLDOWN_MIN):
                logger.debug(
                    "deal cooldown active (direction=%s, age=%ds)",
                    direction, age.total_seconds(),
                )
                return False

        return True

    async def _emit_market_brief(self, data: dict, correlation_id: str) -> None:
        pred = data.get("prediction_48h", {})
        summary = data.get("summary", {})

        payload: dict[str, Any] = {
            "currency_pair": "USD/INR",
            "current_rate": data.get("current_rate"),
            "direction": pred.get("direction", "NEUTRAL"),
            "confidence_pct": round(pred.get("confidence", 0.0) * 100, 1),
            "range_low": pred.get("range_low"),
            "range_high": pred.get("range_high"),
            "action": summary.get("action", "CONVERT_PARTIAL"),
            "message": summary.get("message", ""),
            "act_on_signal": data.get("act_on_signal", False),
            "source": "fx_band_predictor",
            "predictor_url": self._url,
        }

        await self.emit(MARKET_BRIEF, payload, correlation_id=correlation_id)
        logger.debug(
            "market.brief emitted: rate=%.4f direction=%s conf=%.1f%%",
            payload["current_rate"] or 0,
            payload["direction"],
            payload["confidence_pct"],
        )

    async def _emit_deal_instruction(self, data: dict, correlation_id: str) -> None:
        pred = data.get("prediction_48h", {})
        direction = pred.get("direction")
        current_rate: float = data.get("current_rate") or 0.0

        amount_usd = min(_DEFAULT_DAILY_VOL_USD, settings.max_single_deal_usd)
        amount_inr = round(amount_usd * current_rate, 2)

        now_utc = datetime.now(timezone.utc)
        deal_id = f"FXDEAL-{uuid.uuid4().hex[:8].upper()}"

        payload: dict[str, Any] = {
            "id": deal_id,
            "currency_pair": "USD/INR",
            "amount_foreign": amount_usd,
            "amount_inr": amount_inr,
            "deal_type": "spot",
            "target_rate": current_rate,
            "time_window_start": now_utc.isoformat(),
            "time_window_end": (now_utc + timedelta(hours=2)).isoformat(),
            "tranche_number": 1,
            "total_tranches": 1,
            "direction": direction,
            "confidence_pct": round(pred.get("confidence", 0.0) * 100, 1),
        }

        await self.emit(DEAL_INSTRUCTION, payload, correlation_id=correlation_id)

        self._last_deal_direction = direction
        self._last_deal_ts = now_utc

        logger.info(
            "deal.instruction emitted: id=%s direction=%s rate=%.4f amount_usd=%.0f",
            deal_id, direction, current_rate, amount_usd,
        )
