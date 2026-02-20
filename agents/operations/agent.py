"""
OperationsAgent — orchestrates fund movements, maker-checker, and window monitoring.

Daily routine (spec §4.4.1 — 6 AM IST):
  1. Receive daily_inr_forecast + currency_split from Liquidity Agent.
  2. Check nostro balances.
  3. For each shortfall: compute transfer amount, verify window, submit maker-checker proposal.
  4. Emit holiday_lookahead for the next 3 business days.
  5. Start intra-day window monitoring loop.

Event reactions:
  - forecast.rda.shortfall   → handle_shortfall
  - fx.deal.instruction      → handle_deal_instruction (verify nostro sufficiency)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, date, time
from typing import TYPE_CHECKING

from agents.base import BaseAgent
from bus.events import (
    SHORTFALL_ALERT, DEAL_INSTRUCTION,
    FUND_MOVEMENT_STATUS, WINDOW_CLOSING, HOLIDAY_LOOKAHEAD,
    TRANSFER_CONFIRMED,
    Event,
)
from models.domain import (
    FundMovementProposal, FundMovementStatus, ProposalStatus,
    RDAShortfall, WindowClosingAlert, HolidayLookahead,
)
from config.settings import settings

if TYPE_CHECKING:
    from bus.base import EventBus
    from agents.operations.fund_mover import FundMover
    from agents.operations.maker_checker import MakerCheckerWorkflow
    from agents.operations.window_manager import WindowManager

logger = logging.getLogger("tms.agent.operations")

# INR market opens at 09:00 IST — shortfalls must be coverable before this
INR_MARKET_OPEN = time(9, 0)


class OperationsAgent(BaseAgent):
    def __init__(
        self,
        bus: "EventBus",
        fund_mover: "FundMover",
        maker_checker: "MakerCheckerWorkflow",
        window_manager: "WindowManager",
    ) -> None:
        super().__init__("operations", bus)
        self.fund_mover     = fund_mover
        self.mc             = maker_checker
        self.windows        = window_manager
        self._monitor_task: asyncio.Task | None = None

    # ── BaseAgent interface ───────────────────────────────────────────────────

    async def setup(self) -> None:
        await self.listen(SHORTFALL_ALERT,  self.handle_shortfall)
        await self.listen(DEAL_INSTRUCTION, self.handle_deal_instruction)
        logger.info("operations agent event handlers registered")

    async def run_daily(self) -> None:
        """
        Scheduled at 06:00 IST by APScheduler.
        Publishes holiday lookahead and starts the window monitoring loop.
        """
        logger.info("operations agent daily routine starting")

        # Publish 3-day holiday lookahead so all agents can plan
        await self._publish_holiday_lookahead()

        # Cancel any stale monitor task from yesterday
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

        self._monitor_task = asyncio.create_task(self._monitor_windows())
        logger.info("window monitoring task started")

    # ── Event Handlers ────────────────────────────────────────────────────────

    async def handle_shortfall(self, event: Event) -> None:
        """
        Triggered by: forecast.rda.shortfall

        spec §4.4.1 step 3:
          - Calculate transfer = shortfall + 10% buffer.
          - Check if the relevant window is open or will open before INR market.
          - If not feasible → escalate (critical prefunding failure).
          - Otherwise → submit maker-checker proposal.
        """
        payload = event.payload
        try:
            shortfall = RDAShortfall(
                currency=payload["currency"],
                required_amount=payload["required_amount"],
                available_balance=payload["available_balance"],
                shortfall=payload["shortfall"],
                severity=payload["severity"],
            )
        except KeyError as exc:
            logger.error("malformed shortfall payload", extra={"missing_key": str(exc)})
            return

        ccy    = shortfall.currency
        amount = shortfall.shortfall * (1 + settings.prefunding_buffer_pct)

        logger.info("handling RDA shortfall", extra={
            "currency": ccy, "shortfall": shortfall.shortfall, "transfer_amount": amount,
        })

        window = self.windows.get_window(ccy)

        # Feasibility check: will the window open before INR market?
        if not window.is_open_now() and not window.opens_before_ist(INR_MARKET_OPEN):
            logger.critical(
                "CRITICAL PREFUNDING FAILURE: window will not open before INR market",
                extra={"currency": ccy, "transfer_amount": amount},
            )
            await self._submit_fund_proposal(
                ccy=ccy,
                amount=amount,
                purpose=f"CRITICAL: RDA shortfall cover {date.today()} — window may miss INR market",
                idempotency_key=f"{ccy}-{date.today()}-shortfall-critical",
                correlation_id=event.correlation_id,
            )
            return

        await self._submit_fund_proposal(
            ccy=ccy,
            amount=amount,
            purpose=f"RDA shortfall cover {date.today()}",
            idempotency_key=f"{ccy}-{date.today()}-shortfall",
            correlation_id=event.correlation_id,
        )

    async def handle_deal_instruction(self, event: Event) -> None:
        """
        Triggered by: fx.deal.instruction

        spec §4.4.3:
          After each FX deal booked by FX Agent, verify that nostro balance
          can support the deal settlement. If cumulative deals exceed nostro
          balance, trigger top-up proposal.
        """
        # TODO: query nostro balance and compare against cumulative deal exposure.
        # If balance < required → call _submit_fund_proposal for a top-up.
        logger.debug(
            "deal instruction received", extra={"deal_id": event.payload.get("id")}
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _submit_fund_proposal(
        self,
        ccy: str,
        amount: float,
        purpose: str,
        idempotency_key: str,
        correlation_id: str | None = None,
    ) -> None:
        proposal = FundMovementProposal(
            id=str(uuid.uuid4()),
            currency=ccy,
            amount=amount,
            source_account=self.fund_mover.get_operating_account(ccy),
            destination_nostro=self.fund_mover.get_nostro_account(ccy),
            rail=self.windows.get_rail(ccy),
            proposed_by="system:operations_agent",
            purpose=purpose,
            idempotency_key=idempotency_key,
        )
        result = await self.mc.submit_proposal(proposal)

        status = FundMovementStatus(
            proposal_id=proposal.id,
            currency=ccy,
            amount=amount,
            status=ProposalStatus(result["status"]) if result["status"] in
                   [s.value for s in ProposalStatus] else ProposalStatus.PENDING_APPROVAL,
            rail=proposal.rail,
        )
        await self.emit(
            FUND_MOVEMENT_STATUS,
            payload=status.__dict__,
            correlation_id=correlation_id,
        )

    async def _monitor_windows(self) -> None:
        """
        Intra-day loop: fires WINDOW_CLOSING alerts 30 min before each cut-off.
        Runs every 60 seconds. Designed to run as a background task.
        """
        alerted: set[str] = set()  # Prevent duplicate alerts per currency per day

        while True:
            try:
                today = date.today().isoformat()
                for ccy in ("USD", "GBP", "EUR"):
                    alert_key = f"{today}-{ccy}"
                    if alert_key in alerted:
                        continue

                    window   = self.windows.get_window(ccy)
                    mins_rem = window.minutes_until_close()

                    if mins_rem is not None and mins_rem <= settings.window_closing_alert_min:
                        alert = WindowClosingAlert(
                            currency=ccy,
                            rail=window.rail,
                            minutes_remaining=mins_rem,
                            close_time_utc=datetime.utcnow(),
                        )
                        await self.emit(WINDOW_CLOSING, payload=alert.__dict__)
                        alerted.add(alert_key)
                        logger.warning("window closing alert fired", extra={
                            "currency": ccy, "minutes_remaining": mins_rem,
                        })

            except Exception as exc:
                logger.error("window monitor error", extra={"error": str(exc)})

            await asyncio.sleep(60)

    async def _publish_holiday_lookahead(self) -> None:
        holidays = await self.windows.holiday_lookahead(days=3)
        lookahead = HolidayLookahead(
            generated_at=datetime.utcnow(),
            holidays=holidays,
        )
        await self.emit(HOLIDAY_LOOKAHEAD, payload=lookahead.__dict__)
        logger.info("holiday lookahead published", extra={"holidays": holidays})
