"""
Abstract EventBus interface for the Aspora TMS message bus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from bus.events import Event

# Type alias for event handler callables
EventHandler = Callable[[Event], Awaitable[None]]


class EventBus(ABC):
    """
    Abstract base class for all EventBus implementations.

    Implementations:
        RedisBus    — Redis Streams, at-least-once, durable (production)
        InMemoryBus — in-process, deterministic (testing)

    Handler contract:
        - Receives the full Event object (not just payload)
        - Must be idempotent; the bus may deliver the same event more than once
        - Should raise an exception to signal failure (will trigger retry / DLQ)
    """

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus.
        Raises on unrecoverable failure (after retries for RedisBus).
        """

    @abstractmethod
    async def subscribe(
        self,
        event_type: str,
        group: str,
        handler: EventHandler,
    ) -> None:
        """
        Register a handler for a given event type and consumer group.
        Must be called before start().
        """

    @abstractmethod
    async def start(self) -> None:
        """
        Start consuming messages.
        Creates consumer groups and begins polling loops.
        """

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop consuming messages and release resources.
        """
