"""
Deals router — GET /deals

Reads fx.deal.instruction events emitted by FXAnalystAgent.
Payload fields: id, currency_pair, amount_foreign, amount_inr, deal_type,
                target_rate, time_window_start, direction, confidence_pct
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.auth import require_auth
from api.schemas import FxDealResponse, _dec, _dt

router = APIRouter(prefix="/deals", tags=["deals"])


def _map_deal(e) -> FxDealResponse:
    p = e.payload
    direction = p.get("direction", "DOWN")
    # DOWN → rate expected to fall → sell USD / buy INR now
    # UP   → rate expected to rise → buy  USD / sell INR now
    side = "SELL" if direction == "DOWN" else "BUY"
    corridor = p.get("currency_pair", "USD/INR").replace("/", "_")

    return FxDealResponse(
        id=p.get("id", e.event_id),
        corridor=corridor,
        side=side,
        amount_usd=_dec(p.get("amount_foreign")),
        rate=_dec(p.get("target_rate")),
        amount_native=_dec(p.get("amount_inr")),
        status="OPEN",
        executed_at=p.get("time_window_start") or _dt(e.timestamp_utc) or "",
        trader="fx_analyst",
    )


@router.get("", response_model=list[FxDealResponse], dependencies=[Depends(require_auth)])
async def list_deals(request: Request) -> list[FxDealResponse]:
    bus = request.app.state.bus
    events = bus.get_events("fx.deal.instruction")
    # Newest-first, cap at 50
    return [_map_deal(e) for e in reversed(events)][:50]
