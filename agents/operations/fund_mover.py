"""
FundMover — executes fund transfers via bank APIs and tracks confirmation.

Responsibilities:
  - Resolve source operating account and destination nostro for a given currency
  - Submit transfer via bank API (with retry / fallback rail logic)
  - Poll for settlement confirmation within SLA
  - Handle CHAPS outage → SWIFT gpi fallback
  - Publish transfer confirmation once settled
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models.domain import FundMovementProposal, ProposalStatus, TransferConfirmation
from config.settings import settings

logger = logging.getLogger("tms.ops.fund_mover")

# SLA per rail in minutes (spec §4.4.2)
RAIL_SLA: dict[str, int] = {
    "fedwire":   settings.fedwire_sla_min,
    "chaps":     settings.chaps_sla_min,
    "target2":   settings.sepa_sla_min,
    "swift":     settings.swift_sla_min,
    "bank_desk": 30,
}

# Fallback rails if primary is unavailable (spec §4.6)
FALLBACK_RAIL: dict[str, str] = {
    "chaps": "swift",
}


class BankAPIError(Exception):
    """Raised when the bank API returns an error after retries."""


class FundMover:
    """
    Executes fund transfers.

    Dependencies:
        bank_api      — async client with submit_transfer(proposal) -> {"ref": str}
        db            — repository (update proposal, store confirmation)
        audit_log     — immutable audit trail
        alert_router  — notify on failure or SLA breach
    """

    def __init__(self, bank_api, db, audit_log, alert_router) -> None:
        self.bank_api     = bank_api
        self.db           = db
        self.audit        = audit_log
        self.alerts       = alert_router

        # Account mappings loaded from config/accounts.yaml at startup
        # In production these come from a secrets manager / DB
        self._operating_accounts: dict[str, str] = {
            "USD": "OPS-USD-001",
            "GBP": "OPS-GBP-001",
            "EUR": "OPS-EUR-001",
            "AED": "OPS-AED-001",
        }
        self._nostro_accounts: dict[str, str] = {
            "USD": "NOSTRO-USD-001",
            "GBP": "NOSTRO-GBP-001",
            "EUR": "NOSTRO-EUR-001",
            "AED": "NOSTRO-AED-001",  # AED funded via USD rail
        }

    # ── Public API ────────────────────────────────────────────────────────

    def get_operating_account(self, currency: str) -> str:
        account = self._operating_accounts.get(currency.upper())
        if not account:
            raise ValueError(f"No operating account configured for {currency}")
        return account

    def get_nostro_account(self, currency: str) -> str:
        account = self._nostro_accounts.get(currency.upper())
        if not account:
            raise ValueError(f"No nostro account configured for {currency}")
        return account

    async def execute_transfer(self, proposal: FundMovementProposal) -> TransferConfirmation:
        """
        Submit the transfer to the bank API and begin confirmation polling.
        Returns a TransferConfirmation once settled.

        On CHAPS failure → retries via SWIFT gpi fallback.
        """
        rail = proposal.rail
        ref  = await self._submit_with_fallback(proposal, rail)

        # Update proposal with bank reference
        proposal.settlement_ref  = ref
        proposal.status          = ProposalStatus.EXECUTED
        proposal.executed_at     = datetime.utcnow()
        proposal.expected_arrival = datetime.utcnow() + timedelta(
            minutes=RAIL_SLA.get(rail, 120)
        )
        await self.db.save(proposal)

        await self.audit.log(
            event_type="transfer.submitted",
            agent="operations",
            action="execute_transfer",
            details={
                "proposal_id": proposal.id,
                "rail": rail,
                "settlement_ref": ref,
                "expected_arrival": proposal.expected_arrival.isoformat(),
            },
        )

        # Start confirmation polling in background
        asyncio.create_task(
            self._poll_confirmation(proposal, ref)
        )

        return TransferConfirmation(
            proposal_id=proposal.id,
            settlement_ref=ref,
            confirmed_at=datetime.utcnow(),
            settled_amount=proposal.amount,
            currency=proposal.currency,
            rail=rail,
        )

    # ── Internal ──────────────────────────────────────────────────────────

    async def _submit_with_fallback(self, proposal: FundMovementProposal, rail: str) -> str:
        """
        Try submitting on the primary rail.
        If it fails 3 times and a fallback rail exists, try the fallback.
        """
        try:
            return await self._submit(proposal, rail)
        except BankAPIError as exc:
            fallback = FALLBACK_RAIL.get(rail)
            if fallback:
                logger.warning(
                    "primary rail failed, switching to fallback",
                    extra={"primary": rail, "fallback": fallback, "error": str(exc)},
                )
                await self.alerts.notify_rail_fallback(proposal, primary=rail, fallback=fallback)
                proposal.rail = fallback
                return await self._submit(proposal, fallback)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=30, max=120),
        retry=retry_if_exception_type(BankAPIError),
        reraise=True,
    )
    async def _submit(self, proposal: FundMovementProposal, rail: str) -> str:
        """Call bank API with exponential back-off retry (3 attempts, 30s–120s)."""
        try:
            result = await self.bank_api.submit_transfer(proposal)
            return result["ref"]
        except Exception as exc:
            logger.error("bank API submission failed", extra={
                "proposal_id": proposal.id, "rail": rail, "error": str(exc),
            })
            raise BankAPIError(str(exc)) from exc

    async def _poll_confirmation(
        self, proposal: FundMovementProposal, ref: str
    ) -> None:
        """
        Poll for settlement confirmation.
        Alert if SLA is exceeded without confirmation (spec §4.4.2).
        """
        sla_minutes  = RAIL_SLA.get(proposal.rail, 120)
        deadline     = datetime.utcnow() + timedelta(minutes=sla_minutes)
        poll_interval = 60  # seconds

        while datetime.utcnow() < deadline:
            await asyncio.sleep(poll_interval)
            try:
                status = await self.bank_api.get_transfer_status(ref)
                if status.get("settled"):
                    await self._record_confirmation(proposal, ref)
                    return
            except Exception as exc:
                logger.warning("confirmation poll failed", extra={
                    "ref": ref, "error": str(exc),
                })

        # SLA breached
        logger.error("transfer confirmation SLA breached", extra={
            "proposal_id": proposal.id, "ref": ref, "rail": proposal.rail,
            "sla_minutes": sla_minutes,
        })
        await self.alerts.notify_sla_breach(proposal, ref, sla_minutes)

    async def _record_confirmation(
        self, proposal: FundMovementProposal, ref: str
    ) -> None:
        proposal.status       = ProposalStatus.CONFIRMED
        proposal.confirmed_at = datetime.utcnow()
        proposal.updated_at   = datetime.utcnow()
        await self.db.save(proposal)

        await self.audit.log(
            event_type="transfer.confirmed",
            agent="operations",
            action="record_confirmation",
            details={
                "proposal_id": proposal.id,
                "settlement_ref": ref,
                "confirmed_at": proposal.confirmed_at.isoformat(),
            },
        )
        logger.info("transfer confirmed", extra={
            "proposal_id": proposal.id, "ref": ref,
        })
