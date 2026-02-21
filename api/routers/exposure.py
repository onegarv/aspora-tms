"""
Exposure router — GET /exposure

Derives FX exposure by aggregating fx.deal.instruction events emitted by
FXAnalystAgent.  No separate fx.exposure.update event is required.

Computation:
  total_inr_required  = sum of all deal amount_inr values
  covered_inr         = sum of amount_inr for deals with direction == DOWN
                        (SELL USD → covered the INR need)
  open_inr            = total - covered
  blended_rate        = weighted average of target_rate across all deals
  deal_count          = number of deal instruction events
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from api.auth import require_auth
from api.schemas import ExposureResponse, _dec

router = APIRouter(prefix="/exposure", tags=["exposure"])


@router.get("", response_model=ExposureResponse, dependencies=[Depends(require_auth)])
async def get_exposure(request: Request) -> ExposureResponse:
    bus = request.app.state.bus
    events = bus.get_events("fx.deal.instruction")

    if not events:
        return ExposureResponse()

    total_inr = 0.0
    covered_inr = 0.0
    weighted_rate_sum = 0.0
    total_foreign = 0.0

    latest_ts: str | None = None

    for e in events:
        p = e.payload
        amount_inr = float(p.get("amount_inr") or 0)
        amount_foreign = float(p.get("amount_foreign") or 0)
        rate = float(p.get("target_rate") or 0)
        direction = (p.get("direction") or "").upper()

        total_inr += amount_inr
        total_foreign += amount_foreign

        # DOWN = sell USD / buy INR now = covering the exposure
        if direction == "DOWN":
            covered_inr += amount_inr

        if amount_foreign > 0 and rate > 0:
            weighted_rate_sum += rate * amount_foreign

        # Track the most recent event timestamp
        ts = p.get("time_window_start") or (
            e.timestamp_utc.isoformat() if e.timestamp_utc else None
        )
        if ts and (latest_ts is None or ts > latest_ts):
            latest_ts = ts

    open_inr = max(0.0, total_inr - covered_inr)
    blended_rate = (weighted_rate_sum / total_foreign) if total_foreign > 0 else 0.0
    as_of = latest_ts or datetime.now(timezone.utc).isoformat()

    return ExposureResponse(
        as_of=as_of,
        total_inr_required=_dec(total_inr),
        covered_inr=_dec(covered_inr),
        open_inr=_dec(open_inr),
        blended_rate=_dec(blended_rate) if blended_rate > 0 else None,
        deal_count=len(events),
    )
