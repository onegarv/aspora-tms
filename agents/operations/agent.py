"""
OperationsAgent v2.0 — Round 6 PRD implementation.

Daily routine (06:00 IST):
  1. Day-rollover check — reset cumulative_deals, window_alerts_sent, pending_intents.
  2. Emit holiday lookahead (CalendarService, next `lookahead_days` days).
  3. Emit nostro balance snapshot (NOSTRO_BALANCE_UPDATE).
  4. Restart window-monitoring background task.
  5. Check stale proposals (> stale_proposal_age_min minutes pending).

Event handlers:
  forecast.daily.ready          → handle_forecast
  forecast.rda.shortfall        → handle_shortfall
  fx.deal.instruction           → handle_deal_instruction
  fx.reforecast.trigger         → handle_reforecast
  maker_checker.proposal.approved → handle_proposal_approved

Design invariants:
  1. All money is Decimal — no float arithmetic in this class.
  2. FundMover owns execution; this agent only submits proposals to MakerChecker.
  3. MakerCheckerWorkflow is the sole approval authority.
  4. Handlers are idempotent (_pending_intents deduplication per calendar day).
  5. Ambiguous bank states (SubmitUnknownError) trigger manual review, not retry.
  6. All window queries pass explicit current_time (no implicit datetime.now inside helpers).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, TYPE_CHECKING

from zoneinfo import ZoneInfo

from agents.base import BaseAgent
from bus.events import (
    FORECAST_READY,
    SHORTFALL_ALERT,
    DEAL_INSTRUCTION,
    REFORECAST_TRIGGER,
    PROPOSAL_APPROVED,
    FUND_MOVEMENT_STATUS,
    NOSTRO_BALANCE_UPDATE,
    WINDOW_CLOSING,
    HOLIDAY_LOOKAHEAD,
    Event,
)
from models.domain import FundMovementProposal
from services.calendar_service import IN_RBI_FX, ALL_CALENDARS

if TYPE_CHECKING:
    from bus.base import EventBus
    from agents.operations.fund_mover import FundMover
    from agents.operations.maker_checker import MakerCheckerWorkflow
    from agents.operations.window_manager import WindowManager
    from services.calendar_service import CalendarService

logger = logging.getLogger("tms.agent.operations")

TZ_IST = ZoneInfo("Asia/Kolkata")

# Currencies monitored by default for window alerts and balance snapshots
_DEFAULT_CURRENCIES: list[str] = ["USD", "EUR", "GBP", "AED"]


def _tomorrow_9am_ist(now_utc: datetime) -> datetime:
    """Return tomorrow 09:00 IST as a timezone-aware datetime (the INR market open deadline)."""
    ist_today = now_utc.astimezone(TZ_IST).date()
    return datetime.combine(ist_today + timedelta(days=1), time(9, 0), tzinfo=TZ_IST)


class OperationsAgent(BaseAgent):
    """
    Orchestrates fund movements, maker-checker approvals, window monitoring,
    and holiday lookahead for the Aspora TMS.
    """

    def __init__(
        self,
        bus: "EventBus",
        calendar: "CalendarService",
        window_manager: "WindowManager",
        maker_checker: "MakerCheckerWorkflow",
        fund_mover: "FundMover",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("operations", bus)
        self._cal     = calendar
        self._windows = window_manager
        self._mc      = maker_checker
        self._fm      = fund_mover
        self._cfg     = config or {}

        # ── Internal state ────────────────────────────────────────────────────
        # Populated from forecast.daily.ready; used for diagnostics / context.
        self.today_forecast: dict | None = None

        # proposal_id → FundMovementProposal for all proposals pending execution.
        self._pending_proposals: dict[str, FundMovementProposal] = {}

        # Idempotency keys submitted this calendar day (reset on day rollover).
        self._pending_intents: set[str] = set()
        # Whether _pending_intents has been restored from MC for today (restart safety).
        self._intents_restored: bool = False

        # Cumulative deal amounts per currency for today (reset on day rollover).
        self._cumulative_deals: dict[str, Decimal] = {}

        # f"{date}-{ccy}" keys for which a WINDOW_CLOSING alert has been sent today.
        self._window_alerts_sent: set[str] = set()

        self._running: bool = False
        self._current_date: date | None = None
        self._monitor_task: asyncio.Task | None = None

    # ── Config helpers ─────────────────────────────────────────────────────────

    def _c(self, key: str, default: Any) -> Any:
        return self._cfg.get(key, default)

    @property
    def _monitored_currencies(self) -> list[str]:
        return self._c("monitored_currencies", _DEFAULT_CURRENCIES)

    @property
    def _prefunding_buffer(self) -> Decimal:
        return Decimal(str(self._c("prefunding_buffer_pct", 0.10)))

    @property
    def _window_closing_alert_min(self) -> int:
        return int(self._c("window_closing_alert_min", 30))

    @property
    def _monitor_interval_sec(self) -> float:
        return float(self._c("monitor_interval_sec", 60.0))

    @property
    def _stale_proposal_age_min(self) -> int:
        return int(self._c("stale_proposal_age_min", 90))

    @property
    def _nostro_topup_trigger_pct(self) -> Decimal:
        return Decimal(str(self._c("nostro_topup_trigger_pct", 0.90)))

    @property
    def _topup_target_pct(self) -> Decimal:
        return Decimal(str(self._c("topup_target_pct", 1.20)))

    @property
    def _lookahead_days(self) -> int:
        return int(self._c("lookahead_days", 3))

    # ── BaseAgent interface ────────────────────────────────────────────────────

    async def setup(self) -> None:
        await self.listen(FORECAST_READY,    self.handle_forecast)
        await self.listen(SHORTFALL_ALERT,   self.handle_shortfall)
        await self.listen(DEAL_INSTRUCTION,  self.handle_deal_instruction)
        await self.listen(REFORECAST_TRIGGER, self.handle_reforecast)
        await self.listen(PROPOSAL_APPROVED, self.handle_proposal_approved)
        self._running = True
        logger.info("operations agent v2.0 event handlers registered")

    async def run_daily(self) -> None:
        """
        Scheduled at 06:00 IST by APScheduler.

        Order:
          1. Day rollover — resets daily counters.
          2. Holiday lookahead → HOLIDAY_LOOKAHEAD.
          3. Nostro balance snapshot → NOSTRO_BALANCE_UPDATE.
          4. Restart window-monitoring loop.
          5. Stale-proposal sweep.
        """
        now_utc = datetime.now(timezone.utc)
        self._check_day_rollover(now_utc)

        # Step 2: holiday lookahead (includes dst_transitions)
        lookahead = self._build_holiday_lookahead(now_utc)
        await self.emit(
            HOLIDAY_LOOKAHEAD,
            payload={
                **lookahead,
                "generated_at": now_utc.isoformat(),
            },
        )

        # Step 3: nostro balances
        await self._emit_nostro_balances()

        # Step 4: (re)start monitor
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        self._monitor_task = asyncio.create_task(self._monitor_windows())

        # Step 5: stale proposals
        await self._check_stale_proposals(now_utc)

        logger.info("operations agent daily routine complete")

    # ── Event Handlers ─────────────────────────────────────────────────────────

    async def handle_forecast(self, event: Event) -> None:
        """
        Triggered by forecast.daily.ready.
        Stores the forecast and emits a fresh nostro balance snapshot.
        """
        self.today_forecast = event.payload
        await self._emit_nostro_balances(correlation_id=event.correlation_id)
        logger.info("daily forecast received, nostro snapshot refreshed")

    async def handle_shortfall(self, event: Event) -> None:
        """
        Triggered by forecast.rda.shortfall.

        Steps:
          1. Extract currency and shortfall amount (Decimal; no float arithmetic).
          2. Compute transfer_amount = shortfall × (1 + prefunding_buffer).
          3. Check window feasibility: window must open before INR market tomorrow.
          4. Check available balance ≥ transfer_amount.
          5. Idempotency: skip if this currency's daily shortfall key was already submitted.
          6. Create FundMovementProposal → mc.submit_proposal → emit FUND_MOVEMENT_STATUS.
        """
        p   = event.payload
        ccy = p.get("currency")
        if not ccy:
            logger.error("shortfall event missing currency field", extra={"payload": p})
            return

        # Step 0: severity gate — WARNING shortfalls are logged but do not trigger a proposal
        severity = p.get("severity", "critical").lower()
        if severity == "warning":
            logger.warning(
                "WARNING-level shortfall — monitoring only, no proposal created",
                extra={"currency": ccy, "shortfall": p.get("shortfall")},
            )
            return

        # Step 1: Decimal conversion
        try:
            shortfall_amount = Decimal(str(p["shortfall"]))
        except (KeyError, Exception) as exc:
            logger.error("malformed shortfall payload", extra={"error": str(exc)})
            return

        # Step 2: shortfall from RDAChecker already includes the prefunding buffer
        # (required_amount = forecast * (1 + buffer_pct) → shortfall is buffer-inclusive)
        # Do NOT apply the buffer again — that would result in 1.10 × 1.10 = 1.21× coverage.
        transfer_amount = shortfall_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Pre-generate a stable request ID used in all FUND_MOVEMENT_STATUS events,
        # including error paths, so callers can correlate status events to proposals.
        proposal_id = str(uuid.uuid4())

        now_utc  = datetime.now(timezone.utc)
        deadline = _tomorrow_9am_ist(now_utc)

        # Stale event guard: drop events whose computed trading date precedes the
        # agent's current date (can happen on day-boundary bus replays).
        computed_date = self._cal.today(IN_RBI_FX, now_utc)
        if self._current_date is not None and computed_date < self._current_date:
            logger.warning(
                "stale shortfall event discarded — event date precedes current trading day",
                extra={
                    "currency":     ccy,
                    "event_date":   str(computed_date),
                    "current_date": str(self._current_date),
                },
            )
            return

        # Step 3: window feasibility
        try:
            window_ok = self._windows.opens_before(ccy, deadline, now_utc)
        except Exception as exc:
            logger.error(
                "window feasibility check failed",
                extra={"currency": ccy, "error": str(exc)},
            )
            window_ok = False

        if not window_ok:
            logger.critical(
                "CRITICAL PREFUNDING FAILURE: window will not open before INR market",
                extra={"currency": ccy, "transfer_amount": str(transfer_amount)},
            )
            await self.emit(
                FUND_MOVEMENT_STATUS,
                payload={
                    "proposal_id": proposal_id,
                    "currency": ccy,
                    "amount":   str(transfer_amount),
                    "status":   "window_not_feasible",
                    "reason":   f"No {ccy} window before INR market open",
                },
                correlation_id=event.correlation_id,
            )
            return

        # Step 4: balance check
        try:
            available = self._fm.available_balance(ccy)
        except Exception:
            available = Decimal("0")

        if available < transfer_amount:
            logger.warning(
                "insufficient balance for shortfall cover",
                extra={
                    "currency":  ccy,
                    "available": str(available),
                    "required":  str(transfer_amount),
                },
            )
            await self.emit(
                FUND_MOVEMENT_STATUS,
                payload={
                    "proposal_id": proposal_id,
                    "currency":  ccy,
                    "amount":    str(transfer_amount),
                    "status":    "insufficient_balance",
                    "available": str(available),
                },
                correlation_id=event.correlation_id,
            )
            return

        # Step 5: idempotency
        today_str       = self._cal.today(IN_RBI_FX, now_utc).isoformat()
        idempotency_key = f"{ccy}-{today_str}-shortfall"
        # Restore from MC on the first shortfall of the day (survives process restarts).
        if not self._intents_restored:
            await self._restore_pending_intents(now_utc)
            self._intents_restored = True
        if idempotency_key in self._pending_intents:
            logger.info(
                "shortfall proposal already submitted today — skipping",
                extra={"key": idempotency_key},
            )
            return

        # Step 6: resolve rail — raises ValueError for unsupported currencies (e.g. EUR)
        try:
            rail = self._windows.get_rail(ccy)
        except (ValueError, KeyError) as exc:
            logger.error(
                "no transfer rail configured for currency — cannot route shortfall",
                extra={"currency": ccy, "error": str(exc)},
            )
            await self.emit(
                FUND_MOVEMENT_STATUS,
                payload={
                    "currency": ccy,
                    "amount":   str(transfer_amount),
                    "status":   "window_not_feasible",
                    "reason":   f"No transfer rail configured for {ccy}",
                },
                correlation_id=event.correlation_id,
            )
            return

        # Step 7: submit proposal
        proposal = FundMovementProposal(
            id                 = proposal_id,
            currency           = ccy,
            amount             = transfer_amount,
            source_account     = self._fm.get_operating_account(ccy),
            destination_nostro = self._fm.get_nostro_account(ccy),
            rail               = rail,
            proposed_by        = "system:operations_agent",
            purpose            = f"RDA shortfall cover {today_str}",
            idempotency_key    = idempotency_key,
        )
        result = await self._mc.submit_proposal(proposal)
        if result.get("status") != "rejected":
            self._pending_proposals[proposal.id] = proposal
            self._pending_intents.add(idempotency_key)

        await self.emit(
            FUND_MOVEMENT_STATUS,
            payload={
                "proposal_id":     proposal.id,
                "currency":        ccy,
                "amount":          str(transfer_amount),
                "status":          result.get("status", "pending_approval"),
                "idempotency_key": idempotency_key,
            },
            correlation_id=event.correlation_id,
        )
        logger.info(
            "shortfall proposal submitted",
            extra={"proposal_id": proposal.id, "currency": ccy, "amount": str(transfer_amount)},
        )

    async def handle_proposal_approved(self, event: Event) -> None:
        """
        Triggered by maker_checker.proposal.approved.

        Delegates execution to FundMover.execute_proposal().
        On success: emits FUND_MOVEMENT_STATUS (confirmed) + NOSTRO_BALANCE_UPDATE.
        On SubmitUnknownError / SLABreached: emits manual_review_required status.
        """
        from agents.operations.fund_mover import (
            SubmitUnknownError, SLABreached, ExecutionAlreadyFailed,
        )

        proposal_id = event.payload.get("proposal_id")
        if not proposal_id:
            logger.error("PROPOSAL_APPROVED event missing proposal_id", extra={"payload": event.payload})
            return

        proposal = self._pending_proposals.get(proposal_id)
        if proposal is None:
            logger.warning(
                "PROPOSAL_APPROVED for unknown proposal — may have been handled already",
                extra={"proposal_id": proposal_id},
            )
            return

        logger.info("executing approved proposal via FundMover", extra={"proposal_id": proposal_id})
        try:
            execution = await self._fm.execute_proposal(proposal)

            del self._pending_proposals[proposal_id]

            await self.emit(
                FUND_MOVEMENT_STATUS,
                payload={
                    "proposal_id": proposal_id,
                    "currency":    execution.currency,
                    "amount":      str(execution.settled_amount or execution.amount),
                    "status":      execution.state.value,
                    "bank_ref":    execution.bank_ref,
                },
                correlation_id=event.correlation_id,
            )
            await self._emit_nostro_balances()

        except (SubmitUnknownError, SLABreached, ExecutionAlreadyFailed) as exc:
            logger.error(
                "execution error — escalating to manual review",
                extra={"proposal_id": proposal_id, "error": str(exc)},
            )
            await self.emit(
                FUND_MOVEMENT_STATUS,
                payload={
                    "proposal_id": proposal_id,
                    "status":      "manual_review_required",
                    "reason":      str(exc),
                },
                correlation_id=event.correlation_id,
            )

    async def handle_deal_instruction(self, event: Event) -> None:
        """
        Triggered by fx.deal.instruction.

        Accumulates deal amounts per currency. When cumulative deals exceed
        `nostro_topup_trigger_pct × available_balance`, submits a top-up proposal.

        HOLD direction: signals a pause in deal flow — not accumulated and does
        not trigger top-up logic.
        """
        p   = event.payload
        direction = (p.get("direction") or "").upper()
        if direction == "HOLD":
            logger.info("HOLD deal instruction received — skipping accumulation",
                        extra={"payload_currency": p.get("currency")})
            return

        # Support both flat `currency` and `currency_pair` ("USD/INR") formats
        ccy = p.get("currency") or (p.get("currency_pair", "/").split("/")[0])
        if not ccy:
            logger.warning("deal instruction missing currency info", extra={"payload": p})
            return

        try:
            deal_amount = Decimal(str(p.get("amount_foreign", p.get("amount", 0))))
        except Exception as exc:
            logger.error("malformed deal amount", extra={"error": str(exc)})
            return

        prev = self._cumulative_deals.get(ccy, Decimal("0"))
        self._cumulative_deals[ccy] = prev + deal_amount
        cumulative = self._cumulative_deals[ccy]

        try:
            available = self._fm.available_balance(ccy)
        except Exception:
            available = Decimal("0")

        if available > Decimal("0"):
            trigger = available * self._nostro_topup_trigger_pct
            if cumulative >= trigger:
                topup_n = self._next_topup_n(ccy)
                await self._submit_topup_proposal(
                    ccy=ccy,
                    amount=cumulative,
                    topup_n=topup_n,
                    correlation_id=event.correlation_id,
                )

    async def handle_reforecast(self, event: Event) -> None:
        """
        Triggered by fx.reforecast.trigger.

        If the reforecast signals additional funding need that exceeds available
        balance, submit a top-up proposal.
        """
        p   = event.payload
        ccy = p.get("currency", "USD")
        try:
            additional_need = Decimal(str(p.get("additional_amount", p.get("shortfall", 0))))
        except Exception as exc:
            logger.error("malformed reforecast payload", extra={"error": str(exc)})
            return

        if additional_need <= Decimal("0"):
            return

        try:
            available = self._fm.available_balance(ccy)
        except Exception:
            available = Decimal("0")

        if additional_need > available:
            topup_n = self._next_topup_n(ccy)
            await self._submit_topup_proposal(
                ccy=ccy,
                amount=additional_need,
                topup_n=topup_n,
                correlation_id=event.correlation_id,
            )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _next_topup_n(self, ccy: str) -> int:
        """Return the next top-up sequence number for a currency today."""
        prefix = f"{ccy}-"
        return sum(1 for k in self._pending_intents if k.startswith(prefix) and "-topup-" in k) + 1

    async def _submit_topup_proposal(
        self,
        ccy: str,
        amount: Decimal,
        topup_n: int,
        correlation_id: str | None = None,
    ) -> None:
        """Build and submit a nostro top-up proposal via MakerChecker."""
        now_utc         = datetime.now(timezone.utc)
        today_str       = self._cal.today(IN_RBI_FX, now_utc).isoformat()
        idempotency_key = f"{ccy}-{today_str}-topup-{topup_n}"

        if idempotency_key in self._pending_intents:
            logger.info("top-up already submitted — skipping", extra={"key": idempotency_key})
            return

        topup_amount = (amount * self._topup_target_pct).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        proposal = FundMovementProposal(
            id                 = str(uuid.uuid4()),
            currency           = ccy,
            amount             = topup_amount,
            source_account     = self._fm.get_operating_account(ccy),
            destination_nostro = self._fm.get_nostro_account(ccy),
            rail               = self._windows.get_rail(ccy),
            proposed_by        = "system:operations_agent",
            purpose            = f"Nostro top-up {today_str} #{topup_n}",
            idempotency_key    = idempotency_key,
        )
        result = await self._mc.submit_proposal(proposal)
        if result.get("status") != "rejected":
            self._pending_proposals[proposal.id] = proposal
            self._pending_intents.add(idempotency_key)

        await self.emit(
            FUND_MOVEMENT_STATUS,
            payload={
                "proposal_id":     proposal.id,
                "currency":        ccy,
                "amount":          str(topup_amount),
                "status":          result.get("status", "pending_approval"),
                "idempotency_key": idempotency_key,
            },
            correlation_id=correlation_id,
        )
        logger.info(
            "top-up proposal submitted",
            extra={"proposal_id": proposal.id, "currency": ccy, "amount": str(topup_amount)},
        )

    async def _monitor_windows(self) -> None:
        """Background task: fire WINDOW_CLOSING alerts at each poll interval."""
        try:
            while self._running:
                now_utc = datetime.now(timezone.utc)
                await self._monitor_windows_once(now_utc)
                await asyncio.sleep(self._monitor_interval_sec)
        except asyncio.CancelledError:
            logger.info("window monitor task cancelled")
        except Exception as exc:
            logger.error("window monitor crashed", extra={"error": str(exc)})

    async def _monitor_windows_once(self, now_utc: datetime) -> None:
        """
        Check each monitored currency; emit WINDOW_CLOSING if it is within
        `window_closing_alert_min` minutes of operational close.

        De-duplicated per currency per calendar day (IST date).
        """
        today_str = self._cal.today(IN_RBI_FX, now_utc).isoformat()
        for ccy in self._monitored_currencies:
            alert_key = f"{today_str}-{ccy}"
            if alert_key in self._window_alerts_sent:
                continue

            try:
                mins = self._windows.minutes_until_close(ccy, now_utc)
            except Exception as exc:
                logger.warning(
                    "minutes_until_close failed",
                    extra={"currency": ccy, "error": str(exc)},
                )
                continue

            if 0 < mins <= self._window_closing_alert_min:
                rail = self._windows.get_rail(ccy)
                payload: dict = {
                    "currency":          ccy,
                    "rail":              rail,
                    "minutes_remaining": mins,
                    "close_time_utc":    now_utc.isoformat(),
                    "generated_at":      now_utc.isoformat(),
                }
                # Add dynamic IST equivalent for rails that have a fixed local close time.
                # AED uses a GST cutoff (not a TransferWindow) so is excluded.
                if ccy in ("USD", "GBP", "INR"):
                    try:
                        window = self._windows.get_window(ccy)
                        payload["window_closes_local"] = window.official_close_time.isoformat()
                        payload["window_closes_ist"]   = window.close_time_ist().isoformat()
                    except Exception as exc:
                        logger.warning(
                            "could not compute IST close time for alert",
                            extra={"currency": ccy, "error": str(exc)},
                        )
                await self.emit(WINDOW_CLOSING, payload=payload)
                self._window_alerts_sent.add(alert_key)
                logger.warning(
                    "window closing alert fired",
                    extra={"currency": ccy, "minutes_remaining": mins},
                )

    def _build_holiday_lookahead(
        self,
        now_utc: datetime,
        days: int | None = None,
    ) -> dict:
        """
        Return a dict with:
          "holidays"        — {ISO-date: [calendar, ...]} for the next `days` days
          "dst_transitions" — list of upcoming DST transitions (next 7 days)
        """
        days      = days if days is not None else self._lookahead_days
        from_date = self._cal.today(IN_RBI_FX, now_utc)
        result:   dict[str, list[str]] = {}

        for calendar in ALL_CALENDARS:
            try:
                for info in self._cal.upcoming_holidays(from_date, calendar, days=days):
                    date_str = info.date.isoformat()
                    if date_str not in result:
                        result[date_str] = []
                    if calendar not in result[date_str]:
                        result[date_str].append(calendar)
            except Exception as exc:
                logger.warning(
                    "holiday lookahead failed for calendar",
                    extra={"calendar": calendar, "error": str(exc)},
                )

        dst_alerts = self._get_dst_transitions(now_utc, lookahead_days=7)
        return {"holidays": result, "dst_transitions": dst_alerts}

    def _get_dst_transitions(
        self, now_utc: datetime, lookahead_days: int = 7
    ) -> list[dict]:
        """
        Return upcoming DST transitions in the next `lookahead_days` days for
        USD (America/New_York), GBP (Europe/London), and EUR (Europe/Berlin).

        Detects a transition by comparing the UTC offset at noon on consecutive days.
        INR and AED are intentionally omitted — neither jurisdiction observes DST.
        """
        _DST_CURRENCIES = [
            ("USD", ZoneInfo("America/New_York")),
            ("GBP", ZoneInfo("Europe/London")),
            ("EUR", ZoneInfo("Europe/Berlin")),
        ]
        transitions: list[dict] = []
        today = now_utc.astimezone(TZ_IST).date()

        for currency, tz in _DST_CURRENCIES:
            for day_offset in range(lookahead_days):
                check_date = today + timedelta(days=day_offset)
                dt_noon = datetime(
                    check_date.year, check_date.month, check_date.day,
                    12, 0, tzinfo=tz,
                )
                dt_next_noon = dt_noon + timedelta(days=1)

                offset_today = dt_noon.utcoffset()
                offset_next  = dt_next_noon.utcoffset()

                if offset_today == offset_next:
                    continue

                shift_min = int((offset_next - offset_today).total_seconds() / 60)  # type: ignore[operator]
                direction = "spring_forward" if shift_min > 0 else "fall_back"

                # Compute the IST-equivalent window close before and after the transition
                try:
                    window = self._windows.get_window(currency)
                    close_before = window.close_time_ist(check_date).strftime("%I:%M %p IST")
                    close_after  = window.close_time_ist(check_date + timedelta(days=1)).strftime("%I:%M %p IST")
                    impact = (
                        f"{currency} window close shifts from "
                        f"{close_before} to {close_after} "
                        f"on {check_date + timedelta(days=1)} ({direction.replace('_', ' ')})"
                    )
                except Exception:
                    impact = f"{currency} window IST close shifts on {check_date + timedelta(days=1)}"

                transitions.append({
                    "currency":      currency,
                    "date":          check_date.isoformat(),
                    "direction":     direction,
                    "shift_minutes": abs(shift_min),
                    "impact":        impact,
                    "severity":      "warning",
                })

        return transitions

    async def _emit_nostro_balances(self, correlation_id: str | None = None) -> None:
        """Publish current available balance for each monitored currency."""
        balances: dict[str, float] = {}
        for ccy in self._monitored_currencies:
            try:
                balances[ccy] = float(self._fm.available_balance(ccy))
            except Exception as exc:
                logger.warning(
                    "balance query failed",
                    extra={"currency": ccy, "error": str(exc)},
                )
        await self.emit(NOSTRO_BALANCE_UPDATE, payload={"balances": balances},
                        correlation_id=correlation_id)

    async def _check_stale_proposals(self, now_utc: datetime) -> None:
        """Emit stale-review alert for proposals pending longer than stale_proposal_age_min."""
        threshold = timedelta(minutes=self._stale_proposal_age_min)
        for proposal_id, proposal in list(self._pending_proposals.items()):
            created = proposal.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age = now_utc - created
            if age >= threshold:
                await self.emit(
                    FUND_MOVEMENT_STATUS,
                    payload={
                        "proposal_id": proposal_id,
                        "currency":    proposal.currency,
                        "amount":      str(proposal.amount),
                        "status":      "stale_review_needed",
                        "age_minutes": int(age.total_seconds() / 60),
                    },
                )
                logger.warning(
                    "stale proposal flagged",
                    extra={"proposal_id": proposal_id, "age_minutes": int(age.total_seconds() / 60)},
                )

    async def _restore_pending_intents(self, now_utc: datetime | None = None) -> None:
        """
        Repopulate _pending_intents from the MC's proposal list.

        Called lazily on the first shortfall of each day so that idempotency
        survives process restarts — a restarted agent won't re-submit proposals
        that a previous instance already submitted today.

        Uses getattr so this is a no-op for MC stubs that don't implement
        list_proposals (e.g. simple unit-test mocks).
        """
        list_proposals = getattr(self._mc, "list_proposals", None)
        if list_proposals is None:
            return
        try:
            proposals = await list_proposals()
        except Exception as exc:
            logger.warning(
                "could not restore pending intents from MC",
                extra={"error": str(exc)},
            )
            return
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        today_str = self._cal.today(IN_RBI_FX, now_utc).isoformat()
        restored = 0
        for p in proposals:
            if p.idempotency_key and today_str in p.idempotency_key:
                self._pending_intents.add(p.idempotency_key)
                restored += 1
        if restored:
            logger.info(
                "pending intents restored from MC on startup",
                extra={"count": restored, "date": today_str},
            )

    def _check_day_rollover(self, now_utc: datetime) -> None:
        """Reset per-day state when the IST calendar date advances."""
        today = self._cal.today(IN_RBI_FX, now_utc)
        if self._current_date is None or today != self._current_date:
            logger.info(
                "day rollover detected",
                extra={"old_date": str(self._current_date), "new_date": str(today)},
            )
            self._current_date        = today
            self._cumulative_deals    = {}
            self._window_alerts_sent  = set()
            self._pending_intents     = set()
            self._intents_restored    = False

    async def shutdown(self) -> None:
        """Gracefully stop the window-monitoring loop."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("operations agent shut down")
