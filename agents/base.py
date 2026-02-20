"""
Abstract base class for all TMS agents.

Every agent (Liquidity, FX Analyst, Operations) inherits from BaseAgent.
It wires up to the EventBus and enforces a common interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from bus.events import Event, create_event

if TYPE_CHECKING:
    from bus.base import EventBus, EventHandler


class BaseAgent(ABC):
    def __init__(self, name: str, bus: "EventBus") -> None:
        self.name   = name
        self.bus    = bus
        self.logger = logging.getLogger(f"tms.agent.{name}")
        self._started = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    async def setup(self) -> None:
        """
        Register event handlers and initialise internal state.
        Called once before start().
        """

    @abstractmethod
    async def run_daily(self) -> None:
        """
        Execute the agent's daily scheduled routine (typically at 6 AM IST).
        Called by APScheduler.
        """

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        correlation_id: str | None = None,
        version: str = "1.0",
    ) -> None:
        """
        Create and publish an event from this agent.
        Uses create_event() to auto-generate event_id and timestamp.
        """
        event = create_event(
            event_type=event_type,
            source_agent=self.name,
            payload=payload,
            correlation_id=correlation_id,
            version=version,
        )
        self.logger.info(
            "emitting event_type=%s correlation_id=%s",
            event_type, event.correlation_id,
        )
        await self.bus.publish(event)

    async def listen(
        self,
        event_type: str,
        handler: "EventHandler",
    ) -> None:
        """
        Subscribe this agent's handler to an event type.
        Consumer group name = agent name.
        Must be called from setup() (before start()).
        """
        await self.bus.subscribe(event_type, group=self.name, handler=handler)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Wire up handlers and begin consuming events."""
        if self._started:
            self.logger.warning("agent already started, ignoring duplicate start()")
            return
        await self.setup()
        self._started = True
        self.logger.info("agent started")
        await self.bus.start()
