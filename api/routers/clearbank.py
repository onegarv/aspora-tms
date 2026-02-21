"""
ClearBank dispatch router.

POST /api/v1/proposals/{id}/dispatch
    Send an APPROVED proposal to ClearBank (FPS or CHAPS).
    `confirm` must be "LIVE" (real call) or "DEMO" (dry run — no funds move).

GET /api/v1/proposals/{id}/clearbank-status
    Return the ClearBank payment status for a dispatched proposal.

GET /api/v1/clearbank/guardrails
    Return current rate limit and guardrail config (useful for demo dashboards).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth import require_auth
from api.schemas import _dec, _dt
from models.domain import ProposalStatus
from services.guardrail_service import DispatchContext, GuardrailViolation

logger = logging.getLogger("tms.api.clearbank")

router = APIRouter(tags=["clearbank"])


# ── Request / Response schemas ────────────────────────────────────────────────

class DispatchRequest(BaseModel):
    operator_id: str
    confirm: str          # "LIVE" or "DEMO"
    purpose: str          = "INTC"   # CHAPS 4-char purpose code
    category_purpose: str = "CASH"  # CHAPS 2-4 char category purpose


class DispatchResponse(BaseModel):
    proposal_id:          str
    status:               str
    clearbank_payment_id: str | None = None
    rail:                 str
    amount:               str
    currency:             str
    demo_mode:            bool
    dispatched_at:        str | None = None
    message:              str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/proposals/{proposal_id}/dispatch",
    response_model=DispatchResponse,
    dependencies=[Depends(require_auth)],
)
async def dispatch_to_clearbank(
    proposal_id: str,
    body: DispatchRequest,
    request: Request,
) -> DispatchResponse:
    """
    Dispatch an approved proposal to ClearBank.

    confirm="DEMO": guardrails run, ClearBank HTTP call is skipped,
                    a synthetic paymentId is returned. No funds move.
    confirm="LIVE": real ClearBank FPS/CHAPS call. Requires ASPORA_CLEARBANK_ENABLED=true
                    and valid credentials in settings.
    """
    if body.confirm not in ("LIVE", "DEMO"):
        raise HTTPException(
            status_code=400,
            detail='confirm must be exactly "LIVE" (real call) or "DEMO" (dry run)',
        )

    demo_mode = body.confirm == "DEMO"
    mc        = request.app.state.mc
    fm        = request.app.state.fm
    guardrail = request.app.state.guardrail

    # ── 1. Load proposal ────────────────────────────────────────────────────
    proposal = await mc.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")

    if proposal.status != ProposalStatus.APPROVED:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Proposal must be APPROVED before dispatch "
                f"(current status: {proposal.status.value}). "
                "Complete the maker-checker approval first."
            ),
        )

    # ── 2. Run guardrails ───────────────────────────────────────────────────
    ctx = DispatchContext(
        proposal=proposal,
        operator_id=body.operator_id,
        demo_mode=demo_mode,
    )
    try:
        guardrail.check(ctx)
    except GuardrailViolation as exc:
        logger.warning(
            "guardrail blocked dispatch: proposal=%s operator=%s reason=%s",
            proposal_id, body.operator_id, exc,
        )
        raise HTTPException(status_code=422, detail=str(exc))

    # ── 3a. DEMO mode — dry run ─────────────────────────────────────────────
    if demo_mode:
        fake_payment_id = f"DEMO-{uuid.uuid4().hex[:16].upper()}"
        logger.info(
            "DEMO dispatch: proposal=%s amount=%s GBP rail=%s fake_id=%s operator=%s",
            proposal_id, proposal.amount, proposal.rail, fake_payment_id, body.operator_id,
        )

        proposal.status         = ProposalStatus.DISPATCHED
        proposal.settlement_ref = fake_payment_id
        proposal.executed_at    = datetime.now(timezone.utc)
        proposal.updated_at     = datetime.now(timezone.utc)
        await mc.db.save(proposal)
        guardrail.record_dispatch()

        # Publish event so the live event feed updates
        bus = request.app.state.bus
        if bus is not None:
            from bus.events import create_event, CLEARBANK_DISPATCHED
            await bus.publish(
                create_event(
                    event_type=CLEARBANK_DISPATCHED,
                    source_agent="clearbank",
                    payload={
                        "proposal_id":  proposal_id,
                        "payment_id":   fake_payment_id,
                        "amount":       str(proposal.amount),
                        "currency":     proposal.currency,
                        "rail":         proposal.rail,
                        "demo_mode":    True,
                        "operator_id":  body.operator_id,
                    },
                )
            )

        return DispatchResponse(
            proposal_id          = proposal_id,
            status               = "dispatched",
            clearbank_payment_id = fake_payment_id,
            rail                 = proposal.rail,
            amount               = _dec(proposal.amount),
            currency             = proposal.currency,
            demo_mode            = True,
            dispatched_at        = _dt(proposal.executed_at),
            message              = (
                f"DEMO mode — ClearBank {proposal.rail.upper()} simulated. "
                f"No real funds moved. Synthetic payment ID: {fake_payment_id}"
            ),
        )

    # ── 3b. LIVE mode — real ClearBank call ─────────────────────────────────
    cb_client = getattr(request.app.state, "cb_client", None)
    if cb_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "ClearBank client is not initialised. "
                "Set ASPORA_CLEARBANK_TOKEN, ASPORA_CLEARBANK_PRIVATE_KEY, "
                "ASPORA_CLEARBANK_BASE_URL, and ASPORA_CLEARBANK_ENABLED=true."
            ),
        )

    # Swap the FundMover's bank_api to ClearBankBankAPI for this execution
    from agents.operations.clearbank_bank_api import ClearBankBankAPI
    nostro_map = getattr(request.app.state, "clearbank_nostro_map", None) or {}
    cb_bank_api = ClearBankBankAPI(client=cb_client, nostro_map=nostro_map)
    original_bank = fm._bank
    fm._bank = cb_bank_api

    try:
        execution = await fm.execute_proposal(proposal)
    except Exception as exc:
        fm._bank = original_bank  # restore on failure
        logger.error(
            "clearbank live dispatch failed: proposal=%s error=%s", proposal_id, exc
        )
        raise HTTPException(
            status_code=502,
            detail=f"ClearBank transfer failed: {exc}",
        )
    finally:
        fm._bank = original_bank  # always restore

    bank_ref = execution.bank_ref or ""

    # Reload proposal after FundMover updated it
    proposal = await mc.get_proposal(proposal_id)
    guardrail.record_dispatch()

    # Publish event
    bus = request.app.state.bus
    if bus is not None:
        from bus.events import create_event, CLEARBANK_DISPATCHED
        await bus.publish(
            create_event(
                event_type=CLEARBANK_DISPATCHED,
                source_agent="clearbank",
                payload={
                    "proposal_id":  proposal_id,
                    "payment_id":   bank_ref,
                    "amount":       str(proposal.amount) if proposal else "",
                    "currency":     "GBP",
                    "rail":         proposal.rail if proposal else "",
                    "demo_mode":    False,
                    "operator_id":  body.operator_id,
                },
            )
        )

    return DispatchResponse(
        proposal_id          = proposal_id,
        status               = "dispatched",
        clearbank_payment_id = bank_ref,
        rail                 = proposal.rail if proposal else "",
        amount               = _dec(proposal.amount) if proposal else "0",
        currency             = "GBP",
        demo_mode            = False,
        dispatched_at        = _dt(proposal.executed_at) if proposal else None,
        message              = f"Live transfer submitted via ClearBank. Payment ID: {bank_ref}",
    )


@router.get(
    "/proposals/{proposal_id}/clearbank-status",
    dependencies=[Depends(require_auth)],
)
async def get_clearbank_status(proposal_id: str, request: Request) -> dict:
    """Return ClearBank payment status for a dispatched proposal."""
    mc       = request.app.state.mc
    fm       = request.app.state.fm
    proposal = await mc.get_proposal(proposal_id)

    if proposal is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")

    dispatchable_statuses = (
        ProposalStatus.DISPATCHED,
        ProposalStatus.EXECUTED,
        ProposalStatus.CONFIRMED,
    )
    if proposal.status not in dispatchable_statuses:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Proposal has not been dispatched (status={proposal.status.value}). "
                "Dispatch it first via POST /proposals/{id}/dispatch."
            ),
        )

    execution = fm._store.get_by_proposal_id(proposal_id)
    return {
        "proposal_id":          proposal_id,
        "clearbank_payment_id": proposal.settlement_ref,
        "proposal_status":      proposal.status.value,
        "execution_state":      execution.state.value if execution else "unknown",
        "bank_ref":             execution.bank_ref if execution else None,
        "confirmed_at":         _dt(execution.confirmed_at) if execution else None,
        "settled_amount":       _dec(execution.settled_amount) if execution else None,
    }


@router.get(
    "/clearbank/guardrails",
    dependencies=[Depends(require_auth)],
)
async def get_guardrail_status(request: Request) -> dict:
    """Return guardrail config and current rate limit usage — useful for the demo dashboard."""
    from config.settings import settings
    guardrail = request.app.state.guardrail
    return {
        "clearbank_enabled":          settings.clearbank_enabled,
        "clearbank_kill_switch":      settings.clearbank_kill_switch,
        "demo_max_dispatch_gbp":      settings.demo_max_dispatch_gbp,
        "demo_max_dispatches_per_hour": settings.demo_max_dispatches_per_hour,
        "clearbank_base_url":         settings.clearbank_base_url,
        "rate_limit":                 guardrail.rate_limit_status(),
    }
