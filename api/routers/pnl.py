"""
PnL router — GET /pnl

Derives daily unrealised P&L from fx.deal.instruction events vs current market rate.

Methodology:
  - Each deal instruction has a target_rate (rate at signal time) and amount_foreign (USD)
  - Current rate sourced from the latest fx.market.brief event (fallback: 84.0)
  - Unrealised P&L (INR lakh) = Σ amount_usd * (avg_rate - current_rate)
    For SELL-USD deals: profit when INR strengthens (rate falls)
  - Results grouped by date, newest-first, capped at 30 entries
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, Request

from api.auth import require_auth
from api.schemas import PnLEntryResponse, _dec

router = APIRouter(prefix="/pnl", tags=["pnl"])

_LAKH = 1e5  # 1 lakh INR


@router.get("", response_model=list[PnLEntryResponse], dependencies=[Depends(require_auth)])
async def get_pnl(request: Request) -> list[PnLEntryResponse]:
    bus = request.app.state.bus

    # Current rate from latest market brief
    briefs = bus.get_events("fx.market.brief")
    current_rate: float = 84.0
    if briefs:
        current_rate = float(briefs[-1].payload.get("current_rate") or 84.0)

    # Accumulate per-date stats from deal instructions
    daily: dict[str, dict] = defaultdict(
        lambda: {"usd": 0.0, "rate_sum": 0.0, "count": 0}
    )

    for e in bus.get_events("fx.deal.instruction"):
        date_str = e.timestamp_utc.date().isoformat()
        p = e.payload
        amount_usd = float(p.get("amount_foreign") or 0)
        target_rate = float(p.get("target_rate") or current_rate)

        daily[date_str]["usd"] += amount_usd
        daily[date_str]["rate_sum"] += target_rate
        daily[date_str]["count"] += 1

    result: list[PnLEntryResponse] = []
    for date_str, d in sorted(daily.items()):
        count = d["count"] or 1
        avg_rate = d["rate_sum"] / count
        amount_usd = d["usd"]

        # Unrealised: booked at avg_rate, mark-to-market at current_rate
        # SELL-USD: profit = amount_usd * (avg_rate - current_rate) INR
        unrealised_inr_lakh = round(amount_usd * (avg_rate - current_rate) / _LAKH, 4)

        result.append(
            PnLEntryResponse(
                currency="USD",
                realised="0",
                unrealised=_dec(unrealised_inr_lakh),
                total=_dec(unrealised_inr_lakh),
                as_of=date_str,
            )
        )

    # Newest-first, cap at 30 days
    return list(reversed(result))[:30]
