"""
Typed event definitions for the Aspora TMS message bus.

All inter-agent communication is modelled as Events.
Each event has a string type constant used as the Redis Stream key suffix.
Stream key format: tms:<event_type>
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# ── Event model ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Event:
    event_id: str               # UUID — unique per event
    event_type: str             # e.g. "forecast.daily.ready"
    source_agent: str           # e.g. "liquidity"
    timestamp_utc: datetime     # timezone-aware UTC datetime
    payload: dict[str, Any]     # event-specific data (JSON-serializable)
    correlation_id: str         # UUID for tracing a chain of related events
    version: str = "1.0"        # schema version for this event type


def create_event(
    event_type: str,
    source_agent: str,
    payload: dict[str, Any],
    correlation_id: str | None = None,
    version: str = "1.0",
) -> Event:
    """
    Factory: create a well-formed Event with auto-generated IDs and UTC timestamp.

    If `correlation_id` is None, a new UUID is generated.
    """
    return Event(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        source_agent=source_agent,
        timestamp_utc=datetime.now(timezone.utc),
        payload=payload,
        correlation_id=correlation_id or str(uuid.uuid4()),
        version=version,
    )


# ── Event type constants ───────────────────────────────────────────────────────

# Liquidity Agent → all consumers
FORECAST_READY      = "forecast.daily.ready"
SHORTFALL_ALERT     = "forecast.rda.shortfall"

# FX Analyst Agent → all consumers
DEAL_INSTRUCTION    = "fx.deal.instruction"
EXPOSURE_UPDATE     = "fx.exposure.update"
MARKET_BRIEF        = "fx.market.brief"
REFORECAST_TRIGGER  = "fx.reforecast.trigger"

# Operations Agent → all consumers
FUND_MOVEMENT_REQ    = "ops.fund.movement.request"
FUND_MOVEMENT_STATUS = "ops.fund.movement.status"
WINDOW_CLOSING       = "ops.window.closing"
HOLIDAY_LOOKAHEAD    = "ops.holiday.lookahead"
TRANSFER_CONFIRMED   = "ops.transfer.confirmed"

# All known event types — used to enumerate streams
ALL_EVENT_TYPES: list[str] = [
    FORECAST_READY,
    SHORTFALL_ALERT,
    DEAL_INSTRUCTION,
    EXPOSURE_UPDATE,
    MARKET_BRIEF,
    REFORECAST_TRIGGER,
    FUND_MOVEMENT_REQ,
    FUND_MOVEMENT_STATUS,
    WINDOW_CLOSING,
    HOLIDAY_LOOKAHEAD,
    TRANSFER_CONFIRMED,
]
