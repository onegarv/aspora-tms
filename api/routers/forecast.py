"""
Forecast router — GET /forecast and GET /forecast/shortfalls

Reads events emitted by LiquidityAgent:
  forecast.daily.ready  → DailyForecast payload
  forecast.rda.shortfall → ShortfallAlert payload
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.auth import require_auth
from api.schemas import DailyForecastResponse, ShortfallAlertResponse, _dec

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("", response_model=DailyForecastResponse | None, dependencies=[Depends(require_auth)])
async def get_forecast(request: Request) -> DailyForecastResponse | None:
    bus = request.app.state.bus
    events = bus.get_events("forecast.daily.ready")
    if not events:
        return None

    last = events[-1]
    p = last.payload
    return DailyForecastResponse(
        forecast_date=p.get("forecast_date", ""),
        total_inr_crores=_dec(p.get("total_inr_crores")),
        confidence=p.get("confidence", "medium"),
        currency_split={k: _dec(v) for k, v in (p.get("currency_split") or {}).items()},
        multipliers_applied={k: _dec(v) for k, v in (p.get("multipliers_applied") or {}).items()},
        created_at=p.get("created_at", ""),
    )


@router.get("/shortfalls", response_model=list[ShortfallAlertResponse], dependencies=[Depends(require_auth)])
async def get_shortfalls(request: Request) -> list[ShortfallAlertResponse]:
    bus = request.app.state.bus
    events = bus.get_events("forecast.rda.shortfall")
    result: list[ShortfallAlertResponse] = []
    for e in events:
        p = e.payload
        result.append(ShortfallAlertResponse(
            currency=p.get("currency", ""),
            required_amount=_dec(p.get("required_amount")),
            available_balance=_dec(p.get("available_balance")),
            shortfall=_dec(p.get("shortfall")),
            severity=(p.get("severity") or "warning").lower(),
            detected_at=p.get("detected_at", ""),
        ))
    # Newest-first
    return list(reversed(result))
