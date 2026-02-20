"""
InMemoryBus — deterministic, in-process EventBus for unit tests.

Behavior:
    - Preserves publish order per event_type
    - Supports multiple consumer groups per event_type (independent delivery)
    - Simulates at-least-once delivery: failed handlers keep the event pending
    - start() delivers all pending events; stop() halts delivery
    - Thread-safety: not guaranteed; designed for single-threaded async tests
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from bus.base import EventBus, EventHandler
from bus.events import Event

logger = logging.getLogger("tms.bus.memory")


class InMemoryBus(EventBus):
    """
    In-memory EventBus implementation for unit tests.

    State:
        _streams  — full ordered event log per event_type
        _handlers — registered (group, handler) pairs per event_type
        _pending  — unacknowledged events per (event_type, group)
    """

    def __init__(self) -> None:
        self._streams:  dict[str, list[Event]] = defaultdict(list)
        self._handlers: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)
        self._pending:  dict[tuple[str, str], list[Event]] = defaultdict(list)
        self._running    = False
        self._delivering = False  # re-entrancy guard

    # ── EventBus interface ────────────────────────────────────────────────────

    async def publish(self, event: Event) -> None:
        """
        Append to the event stream and enqueue to each subscribed group's
        pending queue. If the bus is running, attempt immediate delivery.
        """
        self._streams[event.event_type].append(event)

        for group, _ in self._handlers[event.event_type]:
            self._pending[(event.event_type, group)].append(event)

        if self._running and not self._delivering:
            await self._deliver_all()

    async def subscribe(
        self,
        event_type: str,
        group: str,
        handler: EventHandler,
    ) -> None:
        """Register a handler. Ensures the pending queue exists for this pair."""
        self._handlers[event_type].append((group, handler))
        # Ensure pending queue exists even if empty
        if (event_type, group) not in self._pending:
            self._pending[(event_type, group)] = []

    async def start(self) -> None:
        """Mark running and drain all pending queues."""
        self._running = True
        await self._deliver_all()

    async def stop(self) -> None:
        """Halt delivery; pending events remain for inspection."""
        self._running = False

    # ── Test helpers ──────────────────────────────────────────────────────────

    def get_events(self, event_type: str | None = None) -> list[Event]:
        """Return all published events, optionally filtered by type."""
        if event_type is None:
            result: list[Event] = []
            for events in self._streams.values():
                result.extend(events)
            return result
        return list(self._streams[event_type])

    def last_event(self, event_type: str) -> Event | None:
        """Return the most recently published event of this type, or None."""
        events = self._streams[event_type]
        return events[-1] if events else None

    def event_count(self, event_type: str | None = None) -> int:
        """Return total number of published events, optionally filtered by type."""
        if event_type is None:
            return sum(len(v) for v in self._streams.values())
        return len(self._streams[event_type])

    def get_payloads(self, event_type: str) -> list[dict[str, Any]]:
        """Return the payload dicts for all events of this type."""
        return [e.payload for e in self._streams[event_type]]

    def clear(self) -> None:
        """Reset all state (streams, handlers, pending)."""
        self._streams.clear()
        self._handlers.clear()
        self._pending.clear()
        self._running = False

    def pending_count(self, event_type: str, group: str) -> int:
        """Return number of unacknowledged events for a (type, group) pair."""
        return len(self._pending.get((event_type, group), []))

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _deliver_all(self) -> None:
        """
        Deliver pending events to their handlers in FIFO order.

        Uses a loop-based approach: keep iterating until a full pass delivers
        nothing. This naturally handles chains where handlers publish new events
        during delivery (the new events are queued and processed in subsequent
        passes without re-entrant recursion).

        Per-queue FIFO guarantee: a queue's second event is not delivered until
        the first succeeds (failed events stay at the front of the queue).
        """
        if self._delivering:
            return  # guard: no re-entrant delivery from within a handler

        self._delivering = True
        try:
            while True:
                delivered_any = False
                # Snapshot the keys; new queues added by handlers will be
                # picked up in the next pass iteration.
                for key in list(self._pending.keys()):
                    queue = self._pending[key]
                    if not queue:
                        continue
                    event_type, group = key
                    handler = self._find_handler(event_type, group)
                    if handler is None:
                        continue
                    # Take the front-of-queue event (FIFO)
                    event = queue[0]
                    try:
                        await handler(event)
                        queue.pop(0)    # ack: remove delivered event
                        delivered_any = True
                    except Exception as exc:
                        logger.warning(
                            "InMemoryBus: handler failed for %s/%s event_id=%s: %s",
                            event_type, group, event.event_id, exc,
                        )
                        # Leave event at front of queue (pending retry)
                        # Don't attempt subsequent events in this queue this pass
                if not delivered_any:
                    break   # nothing left to deliver
        finally:
            self._delivering = False

    def _find_handler(self, event_type: str, group: str) -> EventHandler | None:
        for grp, handler in self._handlers[event_type]:
            if grp == group:
                return handler
        return None

    # ── drain() convenience ────────────────────────────────────────────────────

    async def drain(self) -> None:
        """
        Manually trigger delivery of all pending events.
        Useful in tests that use stop()/drain() patterns without start().
        """
        await self._deliver_all()
