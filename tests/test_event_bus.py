"""
Tests for EventBus — InMemoryBus implementation.

Categories:
    A  Basic pub/sub
    B  Multiple consumer groups
    C  Isolation by event type
    D  Idempotency-friendly duplication simulation (retry on failure)
    E  Event test helpers
    F  create_event() helper
    G  Integration-style chain (correlation_id continuity)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from bus.events import (
    DEAL_INSTRUCTION,
    FORECAST_READY,
    FUND_MOVEMENT_STATUS,
    Event,
    create_event,
)
from bus.memory_bus import InMemoryBus


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def bus() -> InMemoryBus:
    return InMemoryBus()


def make_event(
    event_type: str = FORECAST_READY,
    source_agent: str = "liquidity",
    payload: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> Event:
    return create_event(
        event_type=event_type,
        source_agent=source_agent,
        payload=payload or {"amount": 1_000_000},
        correlation_id=correlation_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A. Basic pub/sub
# ═══════════════════════════════════════════════════════════════════════════════


class TestBasicPubSub:
    @pytest.mark.asyncio
    async def test_handler_called_once_after_start(self, bus: InMemoryBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(FORECAST_READY, "ops", handler)
        event = make_event(FORECAST_READY)
        await bus.publish(event)
        await bus.start()

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_handler_receives_full_event(self, bus: InMemoryBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(FORECAST_READY, "ops", handler)
        event = make_event(FORECAST_READY, payload={"amount": 42})
        await bus.publish(event)
        await bus.start()

        assert received[0].payload == {"amount": 42}
        assert received[0].event_type == FORECAST_READY
        assert received[0].source_agent == "liquidity"

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(self, bus: InMemoryBus) -> None:
        corr_id = str(uuid.uuid4())
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(FORECAST_READY, "fx", handler)
        event = make_event(FORECAST_READY, correlation_id=corr_id)
        await bus.publish(event)
        await bus.start()

        assert received[0].correlation_id == corr_id

    @pytest.mark.asyncio
    async def test_publish_before_start_delivered_on_start(self, bus: InMemoryBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(FORECAST_READY, "ops", handler)
        # Publish BEFORE start
        await bus.publish(make_event(FORECAST_READY))
        await bus.publish(make_event(FORECAST_READY))
        assert len(received) == 0  # not delivered yet

        await bus.start()
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_publish_after_start_delivers_immediately(self, bus: InMemoryBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(FORECAST_READY, "ops", handler)
        await bus.start()

        await bus.publish(make_event(FORECAST_READY))
        assert len(received) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# B. Multiple consumer groups
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultipleConsumerGroups:
    @pytest.mark.asyncio
    async def test_both_groups_receive_event(self, bus: InMemoryBus) -> None:
        fx_received:  list[Event] = []
        ops_received: list[Event] = []

        async def fx_handler(event: Event) -> None:
            fx_received.append(event)

        async def ops_handler(event: Event) -> None:
            ops_received.append(event)

        await bus.subscribe(FORECAST_READY, "fx",  fx_handler)
        await bus.subscribe(FORECAST_READY, "ops", ops_handler)

        event = make_event(FORECAST_READY, correlation_id="shared-corr")
        await bus.publish(event)
        await bus.start()

        assert len(fx_received)  == 1
        assert len(ops_received) == 1
        assert fx_received[0].correlation_id  == "shared-corr"
        assert ops_received[0].correlation_id == "shared-corr"

    @pytest.mark.asyncio
    async def test_groups_receive_independently(self, bus: InMemoryBus) -> None:
        """One group succeeding does not affect the other group's queue."""
        fx_received:  list[Event] = []
        ops_received: list[Event] = []

        async def fx_handler(event: Event) -> None:
            fx_received.append(event)

        async def ops_handler(event: Event) -> None:
            ops_received.append(event)

        await bus.subscribe(FORECAST_READY, "fx",  fx_handler)
        await bus.subscribe(FORECAST_READY, "ops", ops_handler)

        await bus.publish(make_event(FORECAST_READY))
        await bus.publish(make_event(FORECAST_READY))
        await bus.start()

        assert len(fx_received)  == 2
        assert len(ops_received) == 2

    @pytest.mark.asyncio
    async def test_three_groups_all_receive(self, bus: InMemoryBus) -> None:
        counts: dict[str, int] = {"a": 0, "b": 0, "c": 0}

        for grp in ["a", "b", "c"]:
            async def handler(event: Event, g: str = grp) -> None:
                counts[g] += 1
            await bus.subscribe(FORECAST_READY, grp, handler)

        await bus.publish(make_event(FORECAST_READY))
        await bus.start()

        assert counts == {"a": 1, "b": 1, "c": 1}


# ═══════════════════════════════════════════════════════════════════════════════
# C. Isolation by event type
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventTypeIsolation:
    @pytest.mark.asyncio
    async def test_handler_does_not_receive_other_event_type(
        self, bus: InMemoryBus
    ) -> None:
        forecast_received: list[Event] = []
        deal_received:     list[Event] = []

        async def forecast_handler(event: Event) -> None:
            forecast_received.append(event)

        async def deal_handler(event: Event) -> None:
            deal_received.append(event)

        await bus.subscribe(FORECAST_READY,  "ops", forecast_handler)
        await bus.subscribe(DEAL_INSTRUCTION, "ops", deal_handler)

        await bus.publish(make_event(FORECAST_READY))
        await bus.start()

        assert len(forecast_received) == 1
        assert len(deal_received)     == 0

    @pytest.mark.asyncio
    async def test_two_event_types_delivered_to_correct_handlers(
        self, bus: InMemoryBus
    ) -> None:
        received: dict[str, list[Event]] = {FORECAST_READY: [], DEAL_INSTRUCTION: []}

        async def forecast_handler(event: Event) -> None:
            received[FORECAST_READY].append(event)

        async def deal_handler(event: Event) -> None:
            received[DEAL_INSTRUCTION].append(event)

        await bus.subscribe(FORECAST_READY,  "fx",  forecast_handler)
        await bus.subscribe(DEAL_INSTRUCTION, "ops", deal_handler)

        await bus.publish(make_event(FORECAST_READY))
        await bus.publish(make_event(DEAL_INSTRUCTION, source_agent="fx"))
        await bus.publish(make_event(FORECAST_READY))
        await bus.start()

        assert len(received[FORECAST_READY])  == 2
        assert len(received[DEAL_INSTRUCTION]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# D. Idempotency-friendly duplication simulation
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetryOnFailure:
    @pytest.mark.asyncio
    async def test_event_stays_pending_on_handler_failure(
        self, bus: InMemoryBus
    ) -> None:
        attempt_count = 0

        async def failing_handler(event: Event) -> None:
            nonlocal attempt_count
            attempt_count += 1
            raise RuntimeError("transient failure")

        await bus.subscribe(FORECAST_READY, "ops", failing_handler)
        await bus.publish(make_event(FORECAST_READY))
        await bus.start()

        # Handler was called once but event stayed in pending
        assert attempt_count == 1
        assert bus.pending_count(FORECAST_READY, "ops") == 1

    @pytest.mark.asyncio
    async def test_event_retried_on_drain(self, bus: InMemoryBus) -> None:
        """After failure, calling drain() retries the event."""
        call_log: list[str] = []

        async def handler(event: Event) -> None:
            call_log.append(event.event_id)
            if len(call_log) == 1:
                raise RuntimeError("first attempt fails")
            # Second attempt succeeds

        await bus.subscribe(FORECAST_READY, "ops", handler)
        event = make_event(FORECAST_READY)
        await bus.publish(event)
        await bus.start()   # first attempt: fails → pending
        await bus.drain()   # second attempt: succeeds → acked

        assert len(call_log) == 2
        assert call_log[0] == event.event_id
        assert call_log[1] == event.event_id  # same event_id on retry
        assert bus.pending_count(FORECAST_READY, "ops") == 0

    @pytest.mark.asyncio
    async def test_pending_cleared_after_success(self, bus: InMemoryBus) -> None:
        async def handler(event: Event) -> None:
            pass  # always succeeds

        await bus.subscribe(FORECAST_READY, "ops", handler)
        await bus.publish(make_event(FORECAST_READY))
        await bus.start()

        assert bus.pending_count(FORECAST_READY, "ops") == 0

    @pytest.mark.asyncio
    async def test_fifo_ordering_preserved_across_failures(
        self, bus: InMemoryBus
    ) -> None:
        """First event fails; second event should not be delivered until first succeeds."""
        delivered: list[int] = []
        fail_first = True

        async def handler(event: Event) -> None:
            nonlocal fail_first
            if fail_first:
                fail_first = False
                raise RuntimeError("fail event 1")
            delivered.append(event.payload["seq"])

        await bus.subscribe(FORECAST_READY, "ops", handler)
        await bus.publish(make_event(FORECAST_READY, payload={"seq": 1}))
        await bus.publish(make_event(FORECAST_READY, payload={"seq": 2}))
        await bus.start()  # event 1 fails; event 2 NOT delivered yet (FIFO)

        assert delivered == []  # neither delivered (first failed, blocked second)
        await bus.drain()  # retry: event 1 succeeds, then event 2 delivered
        assert delivered == [1, 2]


# ═══════════════════════════════════════════════════════════════════════════════
# E. Event test helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    @pytest.mark.asyncio
    async def test_get_events_all(self, bus: InMemoryBus) -> None:
        await bus.publish(make_event(FORECAST_READY))
        await bus.publish(make_event(DEAL_INSTRUCTION, source_agent="fx"))

        events = bus.get_events()
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_events_filtered_by_type(self, bus: InMemoryBus) -> None:
        await bus.publish(make_event(FORECAST_READY))
        await bus.publish(make_event(FORECAST_READY))
        await bus.publish(make_event(DEAL_INSTRUCTION, source_agent="fx"))

        assert len(bus.get_events(FORECAST_READY))  == 2
        assert len(bus.get_events(DEAL_INSTRUCTION)) == 1

    @pytest.mark.asyncio
    async def test_last_event(self, bus: InMemoryBus) -> None:
        e1 = make_event(FORECAST_READY, payload={"n": 1})
        e2 = make_event(FORECAST_READY, payload={"n": 2})
        await bus.publish(e1)
        await bus.publish(e2)

        last = bus.last_event(FORECAST_READY)
        assert last is not None
        assert last.payload == {"n": 2}

    @pytest.mark.asyncio
    async def test_last_event_none_when_empty(self, bus: InMemoryBus) -> None:
        assert bus.last_event(FORECAST_READY) is None

    @pytest.mark.asyncio
    async def test_event_count_all(self, bus: InMemoryBus) -> None:
        await bus.publish(make_event(FORECAST_READY))
        await bus.publish(make_event(DEAL_INSTRUCTION, source_agent="fx"))
        await bus.publish(make_event(FORECAST_READY))

        assert bus.event_count() == 3

    @pytest.mark.asyncio
    async def test_event_count_filtered(self, bus: InMemoryBus) -> None:
        await bus.publish(make_event(FORECAST_READY))
        await bus.publish(make_event(FORECAST_READY))
        await bus.publish(make_event(DEAL_INSTRUCTION, source_agent="fx"))

        assert bus.event_count(FORECAST_READY)  == 2
        assert bus.event_count(DEAL_INSTRUCTION) == 1

    @pytest.mark.asyncio
    async def test_get_payloads(self, bus: InMemoryBus) -> None:
        await bus.publish(make_event(FORECAST_READY, payload={"a": 1}))
        await bus.publish(make_event(FORECAST_READY, payload={"a": 2}))

        payloads = bus.get_payloads(FORECAST_READY)
        assert payloads == [{"a": 1}, {"a": 2}]

    @pytest.mark.asyncio
    async def test_clear_resets_state(self, bus: InMemoryBus) -> None:
        await bus.publish(make_event(FORECAST_READY))
        bus.clear()

        assert bus.event_count() == 0
        assert bus.last_event(FORECAST_READY) is None
        assert bus.get_events() == []

    @pytest.mark.asyncio
    async def test_clear_also_resets_handlers(self, bus: InMemoryBus) -> None:
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(FORECAST_READY, "ops", handler)
        await bus.publish(make_event(FORECAST_READY))
        bus.clear()

        # After clear, no handlers registered
        await bus.publish(make_event(FORECAST_READY))
        await bus.start()
        assert len(received) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# F. create_event() helper
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateEvent:
    def test_generates_event_id(self) -> None:
        e = create_event(FORECAST_READY, "liquidity", {})
        assert e.event_id
        # Valid UUID format
        uuid.UUID(e.event_id)

    def test_generates_correlation_id_if_none(self) -> None:
        e = create_event(FORECAST_READY, "liquidity", {})
        assert e.correlation_id
        uuid.UUID(e.correlation_id)

    def test_uses_provided_correlation_id(self) -> None:
        corr = str(uuid.uuid4())
        e = create_event(FORECAST_READY, "liquidity", {}, correlation_id=corr)
        assert e.correlation_id == corr

    def test_timestamp_is_timezone_aware_utc(self) -> None:
        e = create_event(FORECAST_READY, "liquidity", {})
        assert e.timestamp_utc.tzinfo is not None
        assert e.timestamp_utc.tzinfo == timezone.utc

    def test_default_version_is_1_0(self) -> None:
        e = create_event(FORECAST_READY, "liquidity", {})
        assert e.version == "1.0"

    def test_custom_version(self) -> None:
        e = create_event(FORECAST_READY, "liquidity", {}, version="2.0")
        assert e.version == "2.0"

    def test_event_is_frozen(self) -> None:
        e = create_event(FORECAST_READY, "liquidity", {})
        with pytest.raises((AttributeError, TypeError)):
            e.event_type = "mutated"  # type: ignore[misc]

    def test_different_calls_produce_different_event_ids(self) -> None:
        e1 = create_event(FORECAST_READY, "liquidity", {})
        e2 = create_event(FORECAST_READY, "liquidity", {})
        assert e1.event_id != e2.event_id

    def test_payload_preserved(self) -> None:
        payload = {"amount": 123_456, "currency": "USD"}
        e = create_event(FORECAST_READY, "liquidity", payload)
        assert e.payload == payload


# ═══════════════════════════════════════════════════════════════════════════════
# G. Integration-style chain (correlation_id continuity)
# ═══════════════════════════════════════════════════════════════════════════════


class TestChainIntegration:
    @pytest.mark.asyncio
    async def test_correlation_id_continuity_across_chain(
        self, bus: InMemoryBus
    ) -> None:
        """
        Liquidity publishes forecast.daily.ready
        FX handler receives it and publishes fx.deal.instruction with same correlation_id
        Ops handler receives fx.deal.instruction and verifies correlation_id
        """
        all_events: dict[str, list[Event]] = {
            FORECAST_READY:   [],
            DEAL_INSTRUCTION: [],
        }

        # Ops subscribes to fx.deal.instruction
        async def ops_handler(event: Event) -> None:
            all_events[DEAL_INSTRUCTION].append(event)

        await bus.subscribe(DEAL_INSTRUCTION, "ops", ops_handler)

        # FX subscribes to forecast.daily.ready and publishes deal instruction
        async def fx_handler(event: Event) -> None:
            all_events[FORECAST_READY].append(event)
            # FX publishes with the SAME correlation_id from the forecast event
            deal_event = create_event(
                event_type=DEAL_INSTRUCTION,
                source_agent="fx",
                payload={"deal_id": "FX-001", "amount": 500_000},
                correlation_id=event.correlation_id,
            )
            await bus.publish(deal_event)

        await bus.subscribe(FORECAST_READY, "fx", fx_handler)

        # Liquidity publishes initial event
        forecast_event = create_event(
            FORECAST_READY, "liquidity", {"inr_need": 50_000_000}
        )
        await bus.publish(forecast_event)
        await bus.start()

        # Assert the chain worked
        assert len(all_events[FORECAST_READY])   == 1
        assert len(all_events[DEAL_INSTRUCTION]) == 1

        # Correlation ID is preserved end-to-end
        original_corr_id = forecast_event.correlation_id
        assert all_events[FORECAST_READY][0].correlation_id   == original_corr_id
        assert all_events[DEAL_INSTRUCTION][0].correlation_id == original_corr_id

    @pytest.mark.asyncio
    async def test_three_hop_chain(self, bus: InMemoryBus) -> None:
        """Three-hop chain: forecast → deal → fund movement status."""
        ops_received: list[Event] = []

        async def ops_handler(event: Event) -> None:
            ops_received.append(event)

        async def fx_handler(event: Event) -> None:
            deal = create_event(
                DEAL_INSTRUCTION, "fx", {"deal": "x"},
                correlation_id=event.correlation_id,
            )
            await bus.publish(deal)

        async def ops_deal_handler(event: Event) -> None:
            status = create_event(
                FUND_MOVEMENT_STATUS, "ops", {"status": "pending"},
                correlation_id=event.correlation_id,
            )
            await bus.publish(status)

        await bus.subscribe(FORECAST_READY,    "fx",   fx_handler)
        await bus.subscribe(DEAL_INSTRUCTION,  "ops",  ops_deal_handler)
        await bus.subscribe(FUND_MOVEMENT_STATUS, "dashboard", ops_handler)

        seed = create_event(FORECAST_READY, "liquidity", {})
        await bus.publish(seed)
        await bus.start()
        # chain: seed → deal → status (requires multiple delivery passes)
        await bus.drain()   # deliver the deal event that fx_handler published
        await bus.drain()   # deliver the status event that ops_deal_handler published

        assert len(ops_received) == 1
        assert ops_received[0].correlation_id == seed.correlation_id

    @pytest.mark.asyncio
    async def test_event_id_unique_per_hop(self, bus: InMemoryBus) -> None:
        """Each event in the chain must have a unique event_id."""
        event_ids: list[str] = []

        async def collector(event: Event) -> None:
            event_ids.append(event.event_id)
            if event.event_type == FORECAST_READY:
                child = create_event(
                    DEAL_INSTRUCTION, "fx", {},
                    correlation_id=event.correlation_id,
                )
                await bus.publish(child)

        await bus.subscribe(FORECAST_READY,  "fx",  collector)
        await bus.subscribe(DEAL_INSTRUCTION, "ops", collector)

        parent = create_event(FORECAST_READY, "liquidity", {})
        await bus.publish(parent)
        await bus.start()
        await bus.drain()

        assert len(event_ids) == 2
        assert event_ids[0] != event_ids[1]  # different event_ids, same correlation_id
