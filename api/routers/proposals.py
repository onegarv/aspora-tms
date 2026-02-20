"""
Proposals router — CRUD and approve/reject for FundMovementProposals.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import require_auth
from api.schemas import (
    ApproveRequest,
    ProposalDetail,
    ProposalSummary,
    RejectRequest,
    _dec,
    _dt,
)

router = APIRouter(prefix="/proposals", tags=["proposals"])


def _to_detail(p) -> ProposalDetail:
    return ProposalDetail(
        id=p.id,
        currency=p.currency,
        amount=_dec(p.amount),
        status=p.status.value,
        rail=p.rail,
        proposed_by=p.proposed_by,
        created_at=_dt(p.created_at),
        source_account=p.source_account,
        destination_nostro=p.destination_nostro,
        purpose=p.purpose,
        approved_by=p.approved_by,
        rejected_by=p.rejected_by,
        rejection_reason=p.rejection_reason,
        executed_at=_dt(p.executed_at),
        confirmed_at=_dt(p.confirmed_at),
        validation_errors=list(p.validation_errors) if p.validation_errors else [],
    )


@router.get("", response_model=list[ProposalSummary], dependencies=[Depends(require_auth)])
async def list_proposals(
    request: Request,
    status: str | None = None,
    currency: str | None = None,
) -> list[ProposalSummary]:
    mc = request.app.state.mc
    proposals = await mc.list_proposals(status=status, currency=currency)
    return [
        ProposalSummary(
            id=p.id,
            currency=p.currency,
            amount=_dec(p.amount),
            status=p.status.value,
            rail=p.rail,
            proposed_by=p.proposed_by,
            created_at=_dt(p.created_at),
        )
        for p in proposals
    ]


@router.get("/{proposal_id}", response_model=ProposalDetail, dependencies=[Depends(require_auth)])
async def get_proposal(proposal_id: str, request: Request) -> ProposalDetail:
    mc = request.app.state.mc
    p = await mc.get_proposal(proposal_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    return _to_detail(p)


@router.post("/{proposal_id}/approve", response_model=ProposalDetail, dependencies=[Depends(require_auth)])
async def approve_proposal(
    proposal_id: str,
    body: ApproveRequest,
    request: Request,
) -> ProposalDetail:
    mc = request.app.state.mc
    try:
        await mc.approve(proposal_id, body.checker_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    p = await mc.get_proposal(proposal_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    return _to_detail(p)


@router.post("/{proposal_id}/reject", response_model=ProposalDetail, dependencies=[Depends(require_auth)])
async def reject_proposal(
    proposal_id: str,
    body: RejectRequest,
    request: Request,
) -> ProposalDetail:
    mc = request.app.state.mc
    try:
        await mc.reject(proposal_id, body.checker_id, body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    p = await mc.get_proposal(proposal_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    return _to_detail(p)
