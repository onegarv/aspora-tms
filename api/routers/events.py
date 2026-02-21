"""
Events router — GET /events, GET /events/stream (SSE), GET /events/{correlation_id}/trace
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

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


@router.get("/stream")
async def stream_events_sse(
    request: Request,
    event_type: str | None = None,
) -> StreamingResponse:
    """
    SSE endpoint — pushes events to the client as they arrive.

    Use the browser EventSource API (not WebSocket).
    Auth: no bearer token required — EventSource cannot set Authorization headers.

    Query params:
      event_type  — optional filter (e.g. "fx.deal.instruction")
    """
    bus = request.app.state.bus

    async def generator():
        seen_ids: set[str] = set()

        # Replay existing events first so the client has context on connect
        for e in bus.get_events(event_type):
            seen_ids.add(e.event_id)
            data = json.dumps({
                "id": e.event_id,
                "type": e.event_type,
                "timestamp": _dt(e.timestamp_utc) or "",
                "payload": dict(e.payload),
            })
            yield f"data: {data}\n\n"

        # Poll for new events every 2 s
        while True:
            if await request.is_disconnected():
                break
            for e in bus.get_events(event_type):
                if e.event_id not in seen_ids:
                    seen_ids.add(e.event_id)
                    data = json.dumps({
                        "id": e.event_id,
                        "type": e.event_type,
                        "timestamp": _dt(e.timestamp_utc) or "",
                        "payload": dict(e.payload),
                    })
                    yield f"data: {data}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx response buffering
        },
    )


@router.get("/{correlation_id}/trace", response_model=list[EventSummary], dependencies=[Depends(require_auth)])
async def trace_events(correlation_id: str, request: Request) -> list[EventSummary]:
    bus = request.app.state.bus
    all_events = bus.get_events()
    matched = [e for e in all_events if e.correlation_id == correlation_id]
    # Chronological order
    matched.sort(key=lambda e: e.timestamp_utc)
    return [_to_summary(e) for e in matched]
