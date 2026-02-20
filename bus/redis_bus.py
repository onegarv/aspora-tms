"""
RedisBus — Redis Streams implementation of EventBus.

Design:
    - One stream per event type: tms:{event_type}
    - One consumer group per subscriber group (agent name)
    - Consumer identity: {group}:{hostname}:{pid}
    - At-least-once delivery via XREADGROUP + XACK
    - Retry up to 3 times with exponential backoff (1s → 5s → 30s)
    - Dead-letter queue stream: tms:dead_letter
    - XAUTOCLAIM to recover stuck pending messages on startup
    - publish() retries 3 times before raising
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from bus.base import EventBus, EventHandler
from bus.events import Event, create_event

logger = logging.getLogger("tms.bus.redis")

STREAM_PREFIX   = "tms"
DLQ_STREAM      = "tms:dead_letter"
POLL_BLOCK_MS   = 1_000   # Long-poll block timeout per XREADGROUP call
BATCH_SIZE      = 10      # Messages per poll
MAX_RETRIES     = 3       # Attempts before dead-lettering
PUBLISH_RETRIES = 3       # Publish retries before raising

# Exponential backoff delays (seconds) for handler failure
RETRY_DELAYS = [1, 5, 30]


def _stream_key(event_type: str) -> str:
    return f"{STREAM_PREFIX}:{event_type}"


def _consumer_name(group: str) -> str:
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{group}:{hostname}:{pid}"


def _serialize(event: Event) -> str:
    return json.dumps({
        "event_id":       event.event_id,
        "event_type":     event.event_type,
        "source_agent":   event.source_agent,
        "timestamp_utc":  event.timestamp_utc.isoformat(),
        "payload":        event.payload,
        "correlation_id": event.correlation_id,
        "version":        event.version,
    })


def _deserialize(raw: str) -> Event:
    data = json.loads(raw)
    return Event(
        event_id=data["event_id"],
        event_type=data["event_type"],
        source_agent=data["source_agent"],
        timestamp_utc=datetime.fromisoformat(data["timestamp_utc"]),
        payload=data["payload"],
        correlation_id=data["correlation_id"],
        version=data.get("version", "1.0"),
    )


class RedisBus(EventBus):
    """
    At-least-once event bus backed by Redis Streams.

    Usage:
        bus = RedisBus("redis://localhost:6379")
        await bus.subscribe("forecast.daily.ready", "ops", handler)
        await bus.start()       # blocks until stop() is called
        ...
        await bus.stop()
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: Redis | None = None

        # (event_type, group) → handler
        self._subscriptions: dict[tuple[str, str], EventHandler] = {}

        # Per-message retry counter: (stream_key, message_id) → attempt_count
        self._retry_counts: dict[tuple[str, str], int] = {}

        # Background poll tasks, one per (event_type, group)
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    # ── EventBus interface ────────────────────────────────────────────────────

    async def publish(self, event: Event) -> None:
        """
        Publish an event to its stream via XADD.
        Retries PUBLISH_RETRIES times with 1s sleep before raising.
        """
        r = await self._get_redis()
        stream = _stream_key(event.event_type)
        data = _serialize(event)

        for attempt in range(1, PUBLISH_RETRIES + 1):
            try:
                await r.xadd(stream, {"event_json": data})
                logger.debug(
                    "published event_id=%s type=%s", event.event_id, event.event_type
                )
                return
            except (aioredis.ConnectionError, aioredis.TimeoutError) as exc:
                logger.warning(
                    "publish attempt %d/%d failed: %s", attempt, PUBLISH_RETRIES, exc
                )
                if attempt < PUBLISH_RETRIES:
                    await asyncio.sleep(1)
                else:
                    raise

    async def subscribe(
        self,
        event_type: str,
        group: str,
        handler: EventHandler,
    ) -> None:
        """Register a handler. Must be called before start()."""
        key = (event_type, group)
        if key in self._subscriptions:
            logger.warning(
                "overwriting existing subscription for (%s, %s)", event_type, group
            )
        self._subscriptions[key] = handler

    async def start(self) -> None:
        """
        Create consumer groups for all subscriptions, then launch one poll
        loop per (event_type, group) pair.
        """
        self._running = True
        r = await self._get_redis()

        for (event_type, group) in self._subscriptions:
            stream = _stream_key(event_type)
            await self._ensure_group(r, stream, group)

            task = asyncio.create_task(
                self._poll_loop(event_type, group),
                name=f"redis-bus-{event_type}-{group}",
            )
            self._tasks.append(task)

        logger.info(
            "RedisBus started: %d subscription(s)", len(self._subscriptions)
        )

    async def stop(self) -> None:
        """Cancel all poll loops and close the Redis connection."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        logger.info("RedisBus stopped")

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _get_redis(self) -> Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def _ensure_group(self, r: Redis, stream: str, group: str) -> None:
        try:
            await r.xgroup_create(stream, group, id="$", mkstream=True)
            logger.debug("created consumer group %s on %s", group, stream)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _poll_loop(self, event_type: str, group: str) -> None:
        """Continuously XREADGROUP for new messages, with reconnect backoff."""
        consumer = _consumer_name(group)
        stream   = _stream_key(event_type)
        handler  = self._subscriptions[(event_type, group)]
        backoff  = 1  # seconds

        # --- Phase 1: claim any stuck pending messages from previous consumers ---
        await self._autoclaim_pending(stream, group, consumer, handler)

        # --- Phase 2: normal polling loop ---
        while self._running:
            try:
                r = await self._get_redis()
                results = await r.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={stream: ">"},
                    count=BATCH_SIZE,
                    block=POLL_BLOCK_MS,
                )
                backoff = 1  # reset on success

                if not results:
                    continue

                for _stream, messages in results:
                    for msg_id, fields in messages:
                        await self._dispatch(
                            r, stream, group, msg_id, fields, handler, event_type
                        )

            except asyncio.CancelledError:
                return
            except (aioredis.ConnectionError, aioredis.TimeoutError) as exc:
                logger.warning(
                    "Redis connection error in poll loop (%s/%s): %s — retrying in %ds",
                    event_type, group, exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
                self._redis = None  # force reconnect
            except Exception as exc:
                logger.error(
                    "Unexpected error in poll loop (%s/%s): %s", event_type, group, exc,
                    exc_info=True,
                )
                await asyncio.sleep(1)

    async def _autoclaim_pending(
        self,
        stream: str,
        group: str,
        consumer: str,
        handler: EventHandler,
        min_idle_ms: int = 60_000,
    ) -> None:
        """Claim messages that have been pending > min_idle_ms (stuck from crashed consumers)."""
        try:
            r = await self._get_redis()
            # XAUTOCLAIM is available in Redis >= 6.2
            result = await r.xautoclaim(
                stream, group, consumer, min_idle_ms, start_id="0-0", count=100
            )
            # result format: [next_id, [[msg_id, fields], ...], [deleted_ids]]
            messages = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else []
            for msg_id, fields in messages:
                await self._dispatch(r, stream, group, msg_id, fields, handler, stream)
        except (ResponseError, Exception) as exc:
            # XAUTOCLAIM not available or stream empty — skip
            logger.debug("XAUTOCLAIM skipped for %s/%s: %s", stream, group, exc)

    async def _dispatch(
        self,
        r: Redis,
        stream: str,
        group: str,
        msg_id: str,
        fields: dict[str, str],
        handler: EventHandler,
        event_type: str,
    ) -> None:
        """Deserialize and call handler; retry on failure; DLQ after MAX_RETRIES."""
        raw = fields.get("event_json")
        if not raw:
            await r.xack(stream, group, msg_id)
            return

        retry_key = (stream, msg_id)
        attempt = self._retry_counts.get(retry_key, 0) + 1
        self._retry_counts[retry_key] = attempt

        try:
            event = _deserialize(raw)
            await handler(event)
            await r.xack(stream, group, msg_id)
            self._retry_counts.pop(retry_key, None)

        except Exception as exc:
            logger.error(
                "Handler failed for msg %s on %s (attempt %d/%d): %s",
                msg_id, stream, attempt, MAX_RETRIES, exc,
                exc_info=True,
            )
            if attempt >= MAX_RETRIES:
                await self._dead_letter(r, stream, msg_id, raw, exc, attempt)
                await r.xack(stream, group, msg_id)
                self._retry_counts.pop(retry_key, None)
            else:
                # Exponential backoff before retry (in-process delay)
                delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
                await asyncio.sleep(delay)
                # DO NOT ack — message stays in PEL for next XREADGROUP call

    async def _dead_letter(
        self,
        r: Redis,
        original_stream: str,
        msg_id: str,
        event_json: str,
        exc: Exception,
        attempts: int,
    ) -> None:
        """Publish failed message to DLQ stream."""
        try:
            await r.xadd(DLQ_STREAM, {
                "original_stream":    original_stream,
                "original_message_id": msg_id,
                "event_json":         event_json,
                "error_type":         type(exc).__name__,
                "error_message":      str(exc),
                "failed_at_utc":      datetime.now(timezone.utc).isoformat(),
                "attempts":           str(attempts),
            })
            logger.error(
                "Dead-lettered msg %s from %s after %d attempts",
                msg_id, original_stream, attempts,
            )
        except Exception as dlq_exc:
            logger.critical(
                "Failed to write to DLQ for msg %s: %s", msg_id, dlq_exc
            )
