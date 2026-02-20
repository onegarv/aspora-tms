"""
Balance router — GET /balances and GET /balances/{currency}
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import require_auth
from api.schemas import BalanceDetail, _dec

router = APIRouter(prefix="/balances", tags=["balances"])


@router.get("", response_model=dict[str, BalanceDetail], dependencies=[Depends(require_auth)])
async def get_all_balances(request: Request) -> dict[str, BalanceDetail]:
    fm = request.app.state.fm
    raw = fm.all_balances()
    return {
        ccy: BalanceDetail(
            currency=ccy,
            available=_dec(info["available"]),
            reserved=_dec(info["reserved"]),
            total=_dec(info["total"]),
        )
        for ccy, info in raw.items()
    }


@router.get("/{currency}", response_model=BalanceDetail, dependencies=[Depends(require_auth)])
async def get_balance(currency: str, request: Request) -> BalanceDetail:
    fm = request.app.state.fm
    raw = fm.all_balances()
    info = raw.get(currency.upper())
    if info is None:
        raise HTTPException(status_code=404, detail=f"No balance tracked for {currency.upper()}")
    return BalanceDetail(
        currency=currency.upper(),
        available=_dec(info["available"]),
        reserved=_dec(info["reserved"]),
        total=_dec(info["total"]),
    )
