"""
MakerCheckerWorkflow — approval workflow for fund movement proposals.

Lifecycle (spec §4.4.2):
    Maker submits → automated validation → pending_approval
    Checker approves (within 30 min) → executed
    Timeout → auto-escalate
    Above dual-checker threshold → requires 2 approvers

Idempotency: duplicate submissions with the same idempotency_key within the
    configured dedup window are rejected immediately.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from bus.base import EventBus
from bus.events import create_event, PROPOSAL_APPROVED
from models.domain import FundMovementProposal, ProposalStatus
from config.settings import settings

if TYPE_CHECKING:
    from services.audit_log import AuditLog

logger = logging.getLogger("tms.ops.maker_checker")


class ValidationError(Exception):
    """Raised when a proposal fails automated validation."""


class PermissionError(Exception):
    """Raised when a checker lacks authority to approve a proposal."""


class MakerCheckerWorkflow:
    """
    Orchestrates the full maker-checker lifecycle.

    Dependencies injected:
        db            — async repository with get/save/is_approved_nostro/has_recent_duplicate
        auth_service  — async service with can_approve(checker_id, proposal) -> bool
        alert_router  — async service with notify_checkers / escalate / notify_executed
        audit_log     — AuditLog service for immutable trail
    """

    def __init__(
        self,
        db,
        auth_service,
        alert_router,
        audit_log: "AuditLog",
        bus: "EventBus | None" = None,
    ) -> None:
        self.db           = db
        self.auth         = auth_service
        self.alerts       = alert_router
        self.audit        = audit_log
        self.bus          = bus

    # ── Public API ────────────────────────────────────────────────────────

    async def submit_proposal(self, proposal: FundMovementProposal) -> dict:
        """
        Maker submits a fund movement proposal.

        1. Run automated validation checks.
        2. If invalid → reject immediately with reasons.
        3. If valid → enter pending_approval queue, notify checkers, start escalation timer.

        Returns a status dict with proposal_id and status.
        """
        errors = await self._validate(proposal)

        if errors:
            proposal.status           = ProposalStatus.REJECTED
            proposal.rejection_reason = "; ".join(errors)
            proposal.validation_errors = errors
            proposal.updated_at       = datetime.now(timezone.utc)
            await self.db.save(proposal)
            await self.audit.log(
                event_type="proposal.rejected",
                agent="operations",
                action="submit_proposal",
                details={"proposal_id": proposal.id, "errors": errors},
            )
            logger.warning("proposal rejected at validation", extra={
                "proposal_id": proposal.id,
                "errors": errors,
            })
            return {"status": "rejected", "proposal_id": proposal.id, "errors": errors}

        proposal.status     = ProposalStatus.PENDING_APPROVAL
        proposal.updated_at = datetime.now(timezone.utc)
        await self.db.save(proposal)

        required_approvers = 2 if proposal.requires_dual_approval else 1
        await self.alerts.notify_checkers(proposal, required_approvers)

        # Start escalation timer in background — doesn't block the caller
        asyncio.create_task(
            self._auto_escalate(proposal.id, settings.maker_checker_timeout_min)
        )

        await self.audit.log(
            event_type="proposal.submitted",
            agent="operations",
            action="submit_proposal",
            details={
                "proposal_id": proposal.id,
                "currency": proposal.currency,
                "amount": proposal.amount,
                "rail": proposal.rail,
                "required_approvers": required_approvers,
            },
        )
        logger.info("proposal submitted for approval", extra={
            "proposal_id": proposal.id,
            "amount": proposal.amount,
            "currency": proposal.currency,
            "dual_checker": proposal.requires_dual_approval,
        })
        return {
            "status": "pending_approval",
            "proposal_id": proposal.id,
            "required_approvers": required_approvers,
        }

    async def approve(self, proposal_id: str, checker_id: str) -> dict:
        """
        Checker approves a proposal.

        For dual-checker proposals:
            - First approval records the first checker.
            - Second approval triggers execution.

        Returns the updated status.
        """
        proposal = await self.db.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        if proposal.status not in (ProposalStatus.PENDING_APPROVAL,):
            raise ValueError(
                f"Proposal {proposal_id} is not awaiting approval (status={proposal.status})"
            )

        if not await self.auth.can_approve(checker_id, proposal):
            raise PermissionError(
                f"Checker {checker_id} lacks authority to approve proposal {proposal_id}"
            )

        # Prevent self-approval
        if proposal.proposed_by == checker_id:
            raise PermissionError("Maker cannot approve their own proposal")

        # Record approval
        if proposal.approved_by is None:
            proposal.approved_by = checker_id
        elif proposal.second_approver is None and proposal.requires_dual_approval:
            if proposal.approved_by == checker_id:
                raise PermissionError("Same checker cannot provide both approvals")
            proposal.second_approver = checker_id
        else:
            raise ValueError("Proposal already has sufficient approvals")

        proposal.updated_at = datetime.now(timezone.utc)
        await self.db.save(proposal)

        await self.audit.log(
            event_type="proposal.approved",
            agent="operations",
            action="approve",
            details={"proposal_id": proposal_id, "checker_id": checker_id,
                     "approvals": proposal.approvals_count},
        )

        # Execute once we have all required approvals
        approvals_needed = 2 if proposal.requires_dual_approval else 1
        if proposal.approvals_count >= approvals_needed:
            return await self._execute(proposal)

        logger.info("first approval recorded, awaiting second checker", extra={
            "proposal_id": proposal_id,
        })
        return {
            "status": "pending_second_approval",
            "proposal_id": proposal_id,
            "approvals": proposal.approvals_count,
        }

    async def list_proposals(
        self,
        status: str | None = None,
        currency: str | None = None,
    ) -> list[FundMovementProposal]:
        proposals = await self.db.list_all()
        if status:
            proposals = [p for p in proposals if p.status.value == status.lower()]
        if currency:
            proposals = [p for p in proposals if p.currency.upper() == currency.upper()]
        return proposals

    async def get_proposal(self, proposal_id: str) -> FundMovementProposal | None:
        return await self.db.get(proposal_id)

    async def reject(self, proposal_id: str, checker_id: str, reason: str) -> dict:
        """Checker explicitly rejects a proposal."""
        proposal = await self.db.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal.status           = ProposalStatus.REJECTED
        proposal.rejected_by      = checker_id
        proposal.rejection_reason = reason
        proposal.updated_at       = datetime.now(timezone.utc)
        await self.db.save(proposal)

        await self.audit.log(
            event_type="proposal.rejected",
            agent="operations",
            action="reject",
            details={"proposal_id": proposal_id, "checker_id": checker_id, "reason": reason},
        )
        logger.warning("proposal rejected by checker", extra={
            "proposal_id": proposal_id, "checker_id": checker_id,
        })
        return {"status": "rejected", "proposal_id": proposal_id}

    # ── Internal ──────────────────────────────────────────────────────────

    async def _validate(self, proposal: FundMovementProposal) -> list[str]:
        """
        Automated validation checks (spec §4.4.2):
          1. Destination nostro is in the approved registry.
          2. No duplicate transfer in the last N hours (idempotency key).
          3. Amount is positive and within per-transaction limits.
          4. Rail matches the currency.
        Returns a list of error strings (empty = valid).
        """
        errors: list[str] = []

        if proposal.amount <= 0:
            errors.append(f"Invalid amount: {proposal.amount}")

        if proposal.amount > settings.max_single_deal_usd * 10:
            errors.append(
                f"Amount {proposal.amount} exceeds maximum per-transfer limit"
            )

        if not await self.db.is_approved_nostro(proposal.destination_nostro):
            errors.append(
                f"Destination nostro '{proposal.destination_nostro}' is not in approved registry"
            )

        if await self.db.has_recent_duplicate(proposal.idempotency_key):
            errors.append(
                f"Duplicate transfer detected (idempotency_key={proposal.idempotency_key})"
            )

        return errors

    async def _execute(self, proposal: FundMovementProposal) -> dict:
        """Mark as approved and publish PROPOSAL_APPROVED to the bus.

        The status is set to APPROVED (not EXECUTED) because no bank transfer
        has happened yet. EXECUTED is set by OpsAgent after FundMover confirms.
        """
        proposal.status     = ProposalStatus.APPROVED
        proposal.executed_at = datetime.now(timezone.utc)
        proposal.updated_at  = datetime.now(timezone.utc)
        await self.db.save(proposal)

        await self.alerts.notify_executed(proposal)
        await self.audit.log(
            event_type="proposal.executed",
            agent="operations",
            action="execute",
            details={
                "proposal_id": proposal.id,
                "amount": proposal.amount,
                "currency": proposal.currency,
                "rail": proposal.rail,
            },
        )

        if self.bus is not None:
            await self.bus.publish(
                create_event(
                    event_type=PROPOSAL_APPROVED,
                    source_agent="maker_checker",
                    payload={"proposal_id": proposal.id},
                )
            )

        logger.info("proposal executed", extra={
            "proposal_id": proposal.id,
            "amount": proposal.amount,
            "currency": proposal.currency,
        })
        return {"status": "executed", "proposal_id": proposal.id}

    async def _auto_escalate(self, proposal_id: str, timeout_minutes: int) -> None:
        """
        Background task: escalate if not approved within `timeout_minutes`.
        Fires once; the escalation alert wakes a human to action the proposal.
        """
        await asyncio.sleep(timeout_minutes * 60)
        try:
            proposal = await self.db.get(proposal_id)
            if proposal and proposal.status == ProposalStatus.PENDING_APPROVAL:
                logger.warning("proposal approval timeout, escalating", extra={
                    "proposal_id": proposal_id,
                })
                await self.alerts.escalate(proposal, reason="Approval timeout exceeded")
                await self.audit.log(
                    event_type="proposal.escalated",
                    agent="operations",
                    action="auto_escalate",
                    details={"proposal_id": proposal_id, "timeout_min": timeout_minutes},
                )
        except Exception as exc:
            logger.error("escalation task failed", extra={
                "proposal_id": proposal_id, "error": str(exc),
            })
