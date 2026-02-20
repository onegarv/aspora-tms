"""
Exposure router — GET /exposure
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.auth import require_auth
from api.schemas import ExposureResponse, _dec, _dt

router = APIRouter(prefix="/exposure", tags=["exposure"])


@router.get("", response_model=ExposureResponse, dependencies=[Depends(require_auth)])
async def get_exposure(request: Request) -> ExposureResponse:
    bus = request.app.state.bus
    events = bus.get_events("fx.exposure.update")
    if not events:
        return ExposureResponse()

    last = events[-1]
    p = last.payload
    return ExposureResponse(
        as_of=_dt(p.get("as_of")) if p.get("as_of") else str(p.get("as_of", "")),
        total_inr_required=_dec(p.get("total_inr_required")),
        covered_inr=_dec(p.get("covered_inr")),
        open_inr=_dec(p.get("open_inr")),
        blended_rate=_dec(p.get("blended_rate")),
        deal_count=p.get("deal_count"),
    )
