"""
Events router — GET /events and GET /events/{correlation_id}/trace
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from api.auth import require_auth
from api.schemas import EventSummary, _dt

router = APIRouter(prefix="/events", tags=["events"])

MAX_LIMIT = 200


def _to_summary(e) -> EventSummary:
    return EventSummary(
        event_id=e.event_id,
        event_type=e.event_type,
        source_agent=e.source_agent,
        timestamp_utc=_dt(e.timestamp_utc) or "",
        correlation_id=e.correlation_id,
        payload=dict(e.payload),
    )


@router.get("", response_model=list[EventSummary], dependencies=[Depends(require_auth)])
async def list_events(
    request: Request,
    event_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=MAX_LIMIT),
) -> list[EventSummary]:
    bus = request.app.state.bus
    events = bus.get_events(event_type)
    # Newest-first
    events = list(reversed(events))
    events = events[:limit]
    return [_to_summary(e) for e in events]


@router.get("/{correlation_id}/trace", response_model=list[EventSummary], dependencies=[Depends(require_auth)])
async def trace_events(correlation_id: str, request: Request) -> list[EventSummary]:
    bus = request.app.state.bus
    all_events = bus.get_events()
    matched = [e for e in all_events if e.correlation_id == correlation_id]
    # Chronological order
    matched.sort(key=lambda e: e.timestamp_utc)
    return [_to_summary(e) for e in matched]
