"""
Dashboard API — 15 integration tests.

Groups:
  A. Balances (3)
  B. Proposals (4)
  C. Transfers (2)
  D. Windows (2)
  E. Auth (2)
  F. Events (2)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import jwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agents.operations.fund_mover import (
    BalanceTracker,
    ExecutionState,
    FundMover,
    InMemoryExecutionStore,
    MockBankAPI,
    TransferExecution,
)
from agents.operations.maker_checker import MakerCheckerWorkflow
from agents.operations.window_manager import WindowManager
from api.app import create_app
from bus.events import create_event, EXPOSURE_UPDATE, FUND_MOVEMENT_STATUS
from bus.memory_bus import InMemoryBus
from models.domain import FundMovementProposal, ProposalStatus
from services.calendar_service import CalendarService

# ── JWT helper ────────────────────────────────────────────────────────────────

SECRET = "change-me-in-production"
ALGORITHM = "HS256"
AUTH_HEADERS = {"Authorization": f"Bearer {jwt.encode({}, SECRET, algorithm=ALGORITHM)}"}

# ── In-memory proposal DB stub ────────────────────────────────────────────────


class InMemoryProposalDB:
    """Minimal proposal repository for tests."""

    def __init__(self) -> None:
        self._store: dict[str, FundMovementProposal] = {}

    async def get(self, proposal_id: str) -> FundMovementProposal | None:
        return self._store.get(proposal_id)

    async def save(self, proposal: FundMovementProposal) -> None:
        self._store[proposal.id] = proposal

    async def list_all(self) -> list[FundMovementProposal]:
        return list(self._store.values())

    async def is_approved_nostro(self, nostro: str) -> bool:
        return True

    async def has_recent_duplicate(self, idempotency_key: str) -> bool:
        return False


class _MockAuth:
    async def can_approve(self, checker_id: str, proposal: FundMovementProposal) -> bool:
        return True


class _MockAlerts:
    async def notify_checkers(self, proposal, required_approvers: int) -> None:
        pass

    async def escalate(self, proposal, **kwargs) -> None:
        pass

    async def notify_executed(self, proposal) -> None:
        pass


class _MockAudit:
    async def log(self, **kwargs) -> None:
        pass


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def app_and_ids():
    """
    Build the full app with real service instances populated with test data.
    Returns (app, proposal_pending_id, proposal_executed_id, execution_id, correlation_id).
    """
    # ── BalanceTracker & FundMover ────────────────────────────────────────
    tracker = BalanceTracker({"USD": Decimal("1_000_000"), "GBP": Decimal("500_000")})
    store = InMemoryExecutionStore()
    bank_api = MockBankAPI()
    fm = FundMover(bank_api, store, tracker)

    # Pre-insert one CONFIRMED USD execution
    exec_id = str(uuid.uuid4())
    proposal_exec_ref = str(uuid.uuid4())
    execution = TransferExecution(
        execution_id=exec_id,
        proposal_id=proposal_exec_ref,
        instruction_id=f"INST-{proposal_exec_ref}",
        currency="USD",
        amount=Decimal("50000.00"),
        source_account="OPS-USD-001",
        destination_nostro="NOSTRO-USD-001",
        rail="fedwire",
        state=ExecutionState.CONFIRMED,
        submitted_at=datetime.now(timezone.utc),
        confirmed_at=datetime.now(timezone.utc),
    )
    store.save(execution)

    # ── MakerCheckerWorkflow ──────────────────────────────────────────────
    db = InMemoryProposalDB()

    pending_id = str(uuid.uuid4())
    p_pending = FundMovementProposal(
        id=pending_id,
        currency="USD",
        amount=100_000.0,
        source_account="OPS-USD-001",
        destination_nostro="NOSTRO-USD-001",
        rail="fedwire",
        proposed_by="maker-1",
        purpose="Prefund nostro",
        idempotency_key=f"idem-{pending_id}",
        status=ProposalStatus.PENDING_APPROVAL,
    )
    await db.save(p_pending)

    executed_id = str(uuid.uuid4())
    p_executed = FundMovementProposal(
        id=executed_id,
        currency="GBP",
        amount=200_000.0,
        source_account="OPS-GBP-001",
        destination_nostro="NOSTRO-GBP-001",
        rail="chaps",
        proposed_by="maker-2",
        purpose="Top-up nostro",
        idempotency_key=f"idem-{executed_id}",
        status=ProposalStatus.EXECUTED,
        executed_at=datetime.now(timezone.utc),
    )
    await db.save(p_executed)

    mc = MakerCheckerWorkflow(db, _MockAuth(), _MockAlerts(), _MockAudit())

    # ── WindowManager ─────────────────────────────────────────────────────
    no_holiday = lambda d: False
    wm = WindowManager(no_holiday, no_holiday, no_holiday, no_holiday)

    # ── CalendarService ───────────────────────────────────────────────────
    cal = CalendarService()

    # ── InMemoryBus ───────────────────────────────────────────────────────
    bus = InMemoryBus()
    correlation_id = str(uuid.uuid4())

    exposure_event = create_event(
        event_type=EXPOSURE_UPDATE,
        source_agent="fx_analyst",
        payload={
            "as_of": None,
            "total_inr_required": "10000000",
            "covered_inr": "8000000",
            "open_inr": "2000000",
            "blended_rate": "83.5",
            "deal_count": 3,
        },
        correlation_id=correlation_id,
    )
    await bus.publish(exposure_event)

    status_event = create_event(
        event_type=FUND_MOVEMENT_STATUS,
        source_agent="operations",
        payload={"proposal_id": pending_id, "status": "pending_approval"},
        correlation_id=correlation_id,
    )
    await bus.publish(status_event)

    test_app = create_app(fm, mc, wm, cal, bus)

    yield test_app, pending_id, executed_id, exec_id, correlation_id


@pytest_asyncio.fixture
async def client(app_and_ids):
    test_app, *_ = app_and_ids
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as c:
        yield c


@pytest_asyncio.fixture
async def ids(app_and_ids):
    _, pending_id, executed_id, exec_id, correlation_id = app_and_ids
    return pending_id, executed_id, exec_id, correlation_id


# ── A. Balances ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_all_balances(client):
    resp = await client.get("/api/v1/balances", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "USD" in data
    usd = data["USD"]
    # amounts must be strings
    assert isinstance(usd["available"], str)
    assert isinstance(usd["reserved"], str)
    assert isinstance(usd["total"], str)


@pytest.mark.asyncio
async def test_get_single_balance(client):
    resp = await client.get("/api/v1/balances/USD", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["currency"] == "USD"
    assert Decimal(data["total"]) == Decimal("1000000")
    assert Decimal(data["available"]) + Decimal(data["reserved"]) == Decimal(data["total"])


@pytest.mark.asyncio
async def test_get_unknown_balance(client):
    resp = await client.get("/api/v1/balances/JPY", headers=AUTH_HEADERS)
    assert resp.status_code == 404


# ── B. Proposals ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_proposals_all(client):
    resp = await client.get("/api/v1/proposals", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2


@pytest.mark.asyncio
async def test_list_proposals_filter(client):
    resp = await client.get(
        "/api/v1/proposals?status=pending_approval", headers=AUTH_HEADERS
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["status"] == "pending_approval"


@pytest.mark.asyncio
async def test_approve_proposal(client, ids):
    pending_id, *_ = ids
    resp = await client.post(
        f"/api/v1/proposals/{pending_id}/approve",
        json={"checker_id": "checker-1"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved_by"] == "checker-1"


@pytest.mark.asyncio
async def test_reject_proposal(client, ids):
    pending_id, *_ = ids
    # Re-fetch to check current state; if already approved in prior test this
    # test creates its own via direct DB manipulation — but fixtures are isolated
    # per test function (pytest-asyncio auto creates new fixtures), so this is fresh.
    resp = await client.post(
        f"/api/v1/proposals/{pending_id}/reject",
        json={"checker_id": "checker-2", "reason": "Insufficient documentation"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["rejected_by"] == "checker-2"
    assert data["status"] == "rejected"


# ── C. Transfers ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_transfers(client, ids):
    _, _, exec_id, _ = ids
    resp = await client.get("/api/v1/transfers", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    execution_ids = [e["execution_id"] for e in data]
    assert exec_id in execution_ids


@pytest.mark.asyncio
async def test_get_transfer_detail(client, ids):
    _, _, exec_id, _ = ids
    resp = await client.get(f"/api/v1/transfers/{exec_id}", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_id"] == exec_id
    assert "instruction_id" in data
    assert "attempt_count" in data
    assert "state" in data


# ── D. Windows ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_windows(client):
    resp = await client.get("/api/v1/windows", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5
    for w in data:
        assert "currency" in w
        assert "rail" in w
        assert "status" in w


@pytest.mark.asyncio
async def test_get_window_usd(client):
    resp = await client.get("/api/v1/windows/USD", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["currency"] == "USD"
    assert data["rail"] == "fedwire"


# ── E. Auth ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_token_returns_403(client):
    # HTTPBearer returns 403 (older FastAPI) or 401 (newer FastAPI) with no header
    resp = await client.get("/api/v1/balances")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_valid_token_returns_200(client):
    token = jwt.encode({}, SECRET, algorithm=ALGORITHM)
    resp = await client.get(
        "/api/v1/balances",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200


# ── F. Events ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_events(client):
    resp = await client.get("/api/v1/events", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    # We published 2 events (exposure + fund_movement_status)
    assert len(data) >= 2
    event_types = {e["event_type"] for e in data}
    assert "fx.exposure.update" in event_types
    assert "ops.fund.movement.status" in event_types


@pytest.mark.asyncio
async def test_events_trace(client, ids):
    _, _, _, correlation_id = ids
    resp = await client.get(
        f"/api/v1/events/{correlation_id}/trace", headers=AUTH_HEADERS
    )
    assert resp.status_code == 200
    data = resp.json()
    # Both events share the same correlation_id
    assert len(data) == 2
    # All returned events must have this correlation_id
    for e in data:
        assert e["correlation_id"] == correlation_id
    # Chronological order: earlier timestamp first
    timestamps = [e["timestamp_utc"] for e in data]
    assert timestamps == sorted(timestamps)
