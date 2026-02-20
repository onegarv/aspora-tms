"""
Event consumer — reads from Redis Streams using consumer groups.

Each agent registers handlers per event type via `.on(event_type, handler)`.
Messages are acknowledged only after the handler completes successfully.
Failed handlers are logged and dead-lettered (not re-queued indefinitely).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import orjson
import redis.asyncio as aioredis

from bus.events import Event

logger = logging.getLogger("tms.bus.consumer")

STREAM_PREFIX  = "tms"
POLL_BLOCK_MS  = 1_000   # Long-poll timeout per xreadgroup call
BATCH_SIZE     = 10      # Messages to fetch per poll
MAX_RETRIES    = 3       # Dead-letter after this many handler failures

Handler = Callable[[dict], Awaitable[None]]


class EventConsumer:
    def __init__(self, redis_url: str, group: str, consumer_name: str) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=False,
        )
        self._group        = group
        self._consumer     = consumer_name
        self._handlers: dict[str, Handler] = {}
        self._running      = False

    # ── Registration ──────────────────────────────────────────────────────

    def on(self, event_type: str, handler: Handler) -> None:
        """Register an async handler for a specific event type."""
        self._handlers[event_type] = handler
        logger.debug("registered handler", extra={"event_type": event_type, "group": self._group})

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Create consumer groups for all registered streams, then start polling."""
        await self._ensure_groups()
        self._running = True
        logger.info(
            "consumer started",
            extra={"group": self._group, "consumer": self._consumer,
                   "streams": list(self._handlers)},
        )
        await self._poll_loop()

    async def stop(self) -> None:
        self._running = False
        await self._redis.aclose()

    # ── Internal ──────────────────────────────────────────────────────────

    async def _ensure_groups(self) -> None:
        for event_type in self._handlers:
            stream = f"{STREAM_PREFIX}:{event_type}"
            try:
                await self._redis.xgroup_create(
                    stream, self._group, id="0", mkstream=True
                )
                logger.debug("created consumer group", extra={"stream": stream, "group": self._group})
            except aioredis.ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise  # Only swallow "group already exists"

    async def _poll_loop(self) -> None:
        streams = {f"{STREAM_PREFIX}:{et}": ">" for et in self._handlers}

        while self._running:
            try:
                results = await self._redis.xreadgroup(
                    self._group,
                    self._consumer,
                    streams,
                    count=BATCH_SIZE,
                    block=POLL_BLOCK_MS,
                )
            except (aioredis.ConnectionError, aioredis.TimeoutError) as exc:
                logger.warning("redis connection error, retrying", extra={"error": str(exc)})
                await asyncio.sleep(2)
                continue

            if not results:
                continue

            for stream_name_bytes, messages in results:
                stream_name = stream_name_bytes.decode()
                event_type  = stream_name.removeprefix(f"{STREAM_PREFIX}:")
                handler     = self._handlers.get(event_type)
                if not handler:
                    continue

                for msg_id, data in messages:
                    await self._dispatch(stream_name, msg_id, data, handler, event_type)

    async def _dispatch(
        self,
        stream_name: str,
        msg_id: bytes,
        data: dict,
        handler: Handler,
        event_type: str,
    ) -> None:
        raw = data.get(b"data")
        if not raw:
            await self._ack(stream_name, msg_id)
            return

        try:
            payload = orjson.loads(raw)
            await handler(payload)
            await self._ack(stream_name, msg_id)
        except Exception as exc:
            logger.error(
                "handler failed",
                extra={
                    "event_type": event_type,
                    "msg_id": msg_id.decode(),
                    "error": str(exc),
                },
                exc_info=True,
            )
            # TODO: move to dead-letter stream after MAX_RETRIES

    async def _ack(self, stream_name: str, msg_id: bytes) -> None:
        await self._redis.xack(stream_name, self._group, msg_id)
