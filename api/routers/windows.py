"""
Windows router — GET /windows and GET /windows/{currency}
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import require_auth
from api.schemas import WindowStatus

router = APIRouter(prefix="/windows", tags=["windows"])

TZ_GST = ZoneInfo("Asia/Dubai")
TZ_UTC = ZoneInfo("UTC")

SUPPORTED = ["USD", "GBP", "EUR", "AED", "INR"]


def _build_window_status(currency: str, wm, now: datetime) -> WindowStatus:
    """Compute WindowStatus for a single currency."""
    if currency == "EUR":
        # SEPA Instant — always open, no close time
        import datetime as dt_mod
        tomorrow = now.astimezone(timezone.utc).date() + dt_mod.timedelta(days=1)
        close_utc = datetime(
            tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0, tzinfo=timezone.utc
        )
        return WindowStatus(
            currency="EUR",
            rail="sepa_instant",
            status="open",
            minutes_until_close=None,
            close_time_utc=close_utc.isoformat(),
        )

    if currency == "AED":
        is_open = wm.is_open_now("AED", now)
        mins = wm.minutes_until_close("AED", now)
        # AED close: today 14:00 GST → UTC
        local_now = now.astimezone(TZ_GST)
        close_gst = datetime.combine(local_now.date(), time(14, 0), tzinfo=TZ_GST)
        close_utc = close_gst.astimezone(timezone.utc)
        status = "open" if is_open else "closed"
        return WindowStatus(
            currency="AED",
            rail="bank_desk",
            status=status,
            minutes_until_close=mins if is_open else None,
            close_time_utc=close_utc.isoformat(),
        )

    # USD, GBP, INR — all have TransferWindow in wm._windows
    window = wm._windows.get(currency)
    if window is None:
        raise HTTPException(status_code=404, detail=f"Window not configured for {currency}")

    is_open = window.is_open_now(now)
    mins = window.minutes_until_close(now)

    # close_time: operational_close_time in window timezone → UTC
    local_now = now.astimezone(window.timezone)
    close_local = datetime.combine(
        local_now.date(), window.operational_close_time, tzinfo=window.timezone
    )
    close_utc = close_local.astimezone(timezone.utc)

    status = "open" if is_open else "closed"

    return WindowStatus(
        currency=currency,
        rail=window.rail_name,
        status=status,
        minutes_until_close=mins if is_open else None,
        close_time_utc=close_utc.isoformat(),
    )


@router.get("", response_model=list[WindowStatus], dependencies=[Depends(require_auth)])
async def list_windows(request: Request) -> list[WindowStatus]:
    wm = request.app.state.wm
    now = datetime.now(timezone.utc)
    return [_build_window_status(ccy, wm, now) for ccy in SUPPORTED]


@router.get("/{currency}", response_model=WindowStatus, dependencies=[Depends(require_auth)])
async def get_window(currency: str, request: Request) -> WindowStatus:
    upper = currency.upper()
    if upper not in SUPPORTED:
        raise HTTPException(status_code=404, detail=f"Currency {upper} not supported")
    wm = request.app.state.wm
    now = datetime.now(timezone.utc)
    return _build_window_status(upper, wm, now)
