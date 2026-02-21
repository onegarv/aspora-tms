"""
Balance router — GET /balances and GET /balances/{currency}
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import require_auth
from api.schemas import BalanceDetail, NostroBalanceResponse, _dec

router = APIRouter(prefix="/balances", tags=["balances"])


@router.get("", response_model=list[NostroBalanceResponse], dependencies=[Depends(require_auth)])
async def get_all_balances(request: Request) -> list[NostroBalanceResponse]:
    fm = request.app.state.fm
    raw = fm.all_balances()
    now = datetime.now(timezone.utc).isoformat()
    return [
        NostroBalanceResponse(
            currency=ccy,
            balance=_dec(info["total"]),
            available=_dec(info["available"]),
            reserved=_dec(info["reserved"]),
            last_updated=now,
        )
        for ccy, info in raw.items()
    ]


@router.get("/{currency}", response_model=NostroBalanceResponse, dependencies=[Depends(require_auth)])
async def get_balance(currency: str, request: Request) -> NostroBalanceResponse:
    fm = request.app.state.fm
    raw = fm.all_balances()
    info = raw.get(currency.upper())
    if info is None:
        raise HTTPException(status_code=404, detail=f"No balance tracked for {currency.upper()}")
    now = datetime.now(timezone.utc).isoformat()
    return NostroBalanceResponse(
        currency=currency.upper(),
        balance=_dec(info["total"]),
        available=_dec(info["available"]),
        reserved=_dec(info["reserved"]),
        last_updated=now,
    )
