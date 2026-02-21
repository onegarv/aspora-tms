"""Tests for MakerCheckerWorkflow."""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from agents.operations.maker_checker import MakerCheckerWorkflow
from bus.events import PROPOSAL_APPROVED
from bus.memory_bus import InMemoryBus
from models.domain import FundMovementProposal


# ── Shared stubs ──────────────────────────────────────────────────────────────


class _InMemDB:
    def __init__(self):
        self._store = {}

    async def get(self, pid):
        return self._store.get(pid)

    async def save(self, p):
        self._store[p.id] = p

    async def is_approved_nostro(self, nostro):
        return True

    async def has_recent_duplicate(self, key):
        return False

    async def list_all(self):
        return list(self._store.values())


class _MockAuth:
    async def can_approve(self, checker_id, proposal):
        return True


class _MockAlerts:
    def __init__(self):
        self.executed = []

    async def notify_checkers(self, proposal, n):
        pass

    async def escalate(self, proposal, **kw):
        pass

    async def notify_executed(self, proposal):
        self.executed.append(proposal)


class _MockAudit:
    async def log(self, **kw):
        pass


def _make_proposal(**kwargs) -> FundMovementProposal:
    defaults = dict(
        id=str(uuid.uuid4()),
        currency="USD",
        amount=Decimal("5000.00"),
        source_account="OPS-USD-001",
        destination_nostro="NOSTRO-USD-001",
        rail="fedwire",
        proposed_by="system:operations_agent",
        purpose="test shortfall cover",
        idempotency_key=f"USD-2026-02-23-shortfall-{uuid.uuid4()}",
    )
    defaults.update(kwargs)
    return FundMovementProposal(**defaults)


def _make_mc(bus=None) -> MakerCheckerWorkflow:
    return MakerCheckerWorkflow(
        db=_InMemDB(),
        auth_service=_MockAuth(),
        alert_router=_MockAlerts(),
        audit_log=_MockAudit(),
        bus=bus,
    )


# ── BUG-004: approve() must emit PROPOSAL_APPROVED ────────────────────────────


class TestApprovalEmitsBusEvent:

    async def test_approve_publishes_proposal_approved_event(self):
        """
        After sufficient approvals, _execute() must publish PROPOSAL_APPROVED
        to the bus so OpsAgent.handle_proposal_approved() can trigger FundMover.
        """
        bus = InMemoryBus()
        mc = _make_mc(bus=bus)

        proposal = _make_proposal()
        await mc.submit_proposal(proposal)
        await mc.approve(proposal.id, "checker-1")

        events = bus.get_events(PROPOSAL_APPROVED)
        assert len(events) == 1, (
            "approve() must publish exactly one PROPOSAL_APPROVED event to the bus"
        )
        assert events[0].payload["proposal_id"] == proposal.id

    async def test_approve_without_bus_does_not_raise(self):
        """MC without a bus (legacy / test mode) must not crash on approve."""
        mc = _make_mc(bus=None)
        proposal = _make_proposal()
        await mc.submit_proposal(proposal)
        result = await mc.approve(proposal.id, "checker-1")
        assert result["status"] == "executed"

    async def test_rejected_proposal_does_not_emit_approved_event(self):
        """Validation-rejected proposals must not emit PROPOSAL_APPROVED."""
        bus = InMemoryBus()

        class _BadDB(_InMemDB):
            async def is_approved_nostro(self, nostro):
                return False   # force rejection

        mc = MakerCheckerWorkflow(
            db=_BadDB(),
            auth_service=_MockAuth(),
            alert_router=_MockAlerts(),
            audit_log=_MockAudit(),
            bus=bus,
        )
        proposal = _make_proposal()
        await mc.submit_proposal(proposal)

        assert bus.event_count(PROPOSAL_APPROVED) == 0
