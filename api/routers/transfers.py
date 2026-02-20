"""
Transfers router — GET /transfers and GET /transfers/{id}
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import require_auth
from api.schemas import TransferDetail, TransferSummary, _dec, _dt

router = APIRouter(prefix="/transfers", tags=["transfers"])


def _to_detail(e) -> TransferDetail:
    return TransferDetail(
        execution_id=e.execution_id,
        proposal_id=e.proposal_id,
        currency=e.currency,
        amount=_dec(e.amount),
        state=e.state.value,
        rail=e.rail,
        submitted_at=_dt(e.submitted_at),
        instruction_id=e.instruction_id,
        bank_ref=e.bank_ref,
        settled_amount=_dec(e.settled_amount) if e.settled_amount is not None else None,
        confirmed_at=_dt(e.confirmed_at),
        last_error=e.last_error,
        attempt_count=e.attempt_count,
    )


@router.get("", response_model=list[TransferSummary], dependencies=[Depends(require_auth)])
async def list_transfers(
    request: Request,
    state: str | None = None,
    currency: str | None = None,
) -> list[TransferSummary]:
    fm = request.app.state.fm
    execs = fm.list_executions(currency=currency, state=state)
    return [
        TransferSummary(
            execution_id=e.execution_id,
            proposal_id=e.proposal_id,
            currency=e.currency,
            amount=_dec(e.amount),
            state=e.state.value,
            rail=e.rail,
            submitted_at=_dt(e.submitted_at),
        )
        for e in execs
    ]


@router.get("/{execution_id}", response_model=TransferDetail, dependencies=[Depends(require_auth)])
async def get_transfer(execution_id: str, request: Request) -> TransferDetail:
    fm = request.app.state.fm
    e = fm._store.get(execution_id)
    if e is None:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    return _to_detail(e)
