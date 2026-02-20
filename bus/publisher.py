"""
Event publisher — writes typed Events to Redis Streams.

Each event type gets its own stream: tms:<event_type>
"""

from __future__ import annotations

import logging
import orjson
from typing import TYPE_CHECKING

import redis.asyncio as aioredis

from bus.events import Event

if TYPE_CHECKING:
    pass

logger = logging.getLogger("tms.bus.publisher")

STREAM_PREFIX = "tms"
MAX_STREAM_LEN = 10_000  # Approx 10k events per stream before trimming


class EventPublisher:
    def __init__(self, redis_url: str) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=False,  # We handle serialisation ourselves
        )

    async def publish(self, event: Event) -> str:
        """
        Publish an event to its corresponding Redis stream.
        Returns the stream entry ID assigned by Redis.
        """
        stream_key = f"{STREAM_PREFIX}:{event.event_type}"
        payload_bytes = orjson.dumps(event.to_dict())

        entry_id: bytes = await self._redis.xadd(
            stream_key,
            {"data": payload_bytes},
            maxlen=MAX_STREAM_LEN,
            approximate=True,
        )
        logger.debug(
            "published event",
            extra={
                "event_type": event.event_type,
                "source": event.source_agent,
                "correlation_id": event.correlation_id,
                "stream": stream_key,
                "entry_id": entry_id.decode(),
            },
        )
        return entry_id.decode()

    async def close(self) -> None:
        await self._redis.aclose()
