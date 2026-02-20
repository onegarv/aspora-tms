"""
Holidays router — GET /holidays?days=7
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, Query, Request

from api.auth import require_auth

router = APIRouter(prefix="/holidays", tags=["holidays"])

CALENDARS = ["US_FEDWIRE", "UK_CHAPS", "IN_RBI_FX", "AE_BANKING"]


@router.get("", dependencies=[Depends(require_auth)])
async def get_holidays(
    request: Request,
    days: int = Query(default=7, ge=1, le=90),
) -> dict[str, list[str]]:
    cal = request.app.state.cal
    today = datetime.now(timezone.utc).date()
    result: dict[str, list[str]] = {}
    for calendar_code in CALENDARS:
        holidays = cal.upcoming_holidays(today, calendar_code, days)
        for h in holidays:
            date_str = h.date.isoformat()
            if date_str not in result:
                result[date_str] = []
            if calendar_code not in result[date_str]:
                result[date_str].append(calendar_code)
    return result
