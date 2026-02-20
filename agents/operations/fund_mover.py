"""
FundMover — fintech-safe, idempotent fund transfer execution.

Design principles:
  - Decimal arithmetic throughout; no float money
  - TransferExecution is the source of truth (not FundMovementProposal)
  - instruction_id = f"INST-{proposal_id}" — stable across retries
  - submit_unknown: fence with find_by_instruction_id before any retry
  - No auto-retry after SUBMITTED; poll only until SLA expires
  - BalanceTracker uses asyncio.Lock to prevent double-spend
  - EUR routing: SEPA Instant (<100k) else SEPA
  - AED routing: 2pm GST cutoff on UAE banking days via CalendarService

State machine:
    QUEUED → SUBMITTING → SUBMITTED → SETTLING → CONFIRMED
                       ↘ SUBMIT_UNKNOWN ↗  (fence first on re-entry)
    any → FAILED  (terminal)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, TYPE_CHECKING

from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from models.domain import FundMovementProposal
    from services.calendar_service import CalendarService

logger = logging.getLogger("tms.ops.fund_mover")

# ── Constants ─────────────────────────────────────────────────────────────────

RAIL_SLA_MIN: dict[str, int] = {
    "fedwire":      30,
    "chaps":        120,
    "sepa_instant": 5,
    "sepa":         240,
    "swift":        480,
    "bank_desk":    30,
}

SEPA_INSTANT_THRESHOLD = Decimal("100000")   # EUR
AED_CUTOFF_HOUR_GST    = 14                  # 2pm Dubai time
DUBAI_TZ               = ZoneInfo("Asia/Dubai")
AE_BANKING             = "AE_BANKING"        # calendar key (mirrors CalendarService)

# ── Exceptions ────────────────────────────────────────────────────────────────


class FundMoverError(Exception):
    """Base exception for FundMover errors."""


class InsufficientFunds(FundMoverError):
    def __init__(self, currency: str, available: Decimal, required: Decimal) -> None:
        self.currency  = currency
        self.available = available
        self.required  = required
        super().__init__(
            f"Insufficient {currency}: available={available}, required={required}"
        )


class AEDCutoffPassed(FundMoverError):
    """AED transfer attempted after 2pm GST or on a non-UAE-banking day."""


class SubmitUnknownError(FundMoverError):
    """Submit call failed; unknown whether the bank recorded the transfer."""

    def __init__(self, execution: TransferExecution, cause: Exception) -> None:
        self.execution = execution
        super().__init__(
            f"Submit unknown for {execution.instruction_id}: {cause}"
        )


class SLABreached(FundMoverError):
    """Transfer was not confirmed within the SLA window."""

    def __init__(self, execution: TransferExecution, sla_min: int) -> None:
        self.execution = execution
        super().__init__(
            f"SLA breached: {execution.instruction_id} not settled after {sla_min}m"
        )


class ExecutionAlreadyFailed(FundMoverError):
    """execute_proposal called on a terminal-failed execution."""

    def __init__(self, execution: TransferExecution) -> None:
        self.execution = execution
        super().__init__(f"Execution {execution.execution_id} is in FAILED state")


# ── ExecutionState ────────────────────────────────────────────────────────────


class ExecutionState(str, Enum):
    QUEUED         = "queued"
    SUBMITTING     = "submitting"
    SUBMIT_UNKNOWN = "submit_unknown"
    SUBMITTED      = "submitted"
    SETTLING       = "settling"
    CONFIRMED      = "confirmed"
    FAILED         = "failed"


# ── TransferExecution ─────────────────────────────────────────────────────────


@dataclass
class TransferExecution:
    """
    Source of truth for a single bank transfer attempt.

    One-to-one with FundMovementProposal; persisted independently so that
    proposal approval state and execution state don't collide.
    """
    execution_id:       str
    proposal_id:        str
    instruction_id:     str             # f"INST-{proposal_id}" — idempotency key sent to bank
    currency:           str
    amount:             Decimal         # Always Decimal — never float
    source_account:     str
    destination_nostro: str
    rail:               str
    state:              ExecutionState  = ExecutionState.QUEUED

    bank_ref:           str | None      = None  # Assigned on successful submit
    submitted_at:       datetime | None = None
    confirmed_at:       datetime | None = None
    settled_amount:     Decimal | None  = None
    last_error:         str | None      = None
    attempt_count:      int             = 0
    created_at:         datetime        = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at:         datetime        = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


# ── BankAPI ───────────────────────────────────────────────────────────────────


class BankAPI(ABC):
    """Abstract interface to the bank's transfer API."""

    @abstractmethod
    async def submit_transfer(
        self,
        *,
        instruction_id:    str,
        amount:            Decimal,
        currency:          str,
        rail:              str,
        source_account:    str,
        destination_nostro: str,
    ) -> str:
        """Submit a transfer. Returns bank_ref on success."""

    @abstractmethod
    async def get_transfer_status(self, bank_ref: str) -> dict[str, Any]:
        """
        Return status dict with at least:
          'settled' (bool), 'settled_amount' (Decimal | float | None)
        """

    @abstractmethod
    async def find_by_instruction_id(self, instruction_id: str) -> str | None:
        """
        Idempotency fence: return bank_ref if the bank has seen this
        instruction_id, else None.
        """


# ── MockBankAPI ───────────────────────────────────────────────────────────────


class MockBankAPI(BankAPI):
    """
    Configurable mock for unit tests.

    Attributes
    ----------
    submit_error:         if set, submit_transfer raises this (and bank does NOT record it)
    submit_unknown_once:  if True, first submit raises BUT bank did record it;
                          mimics a network timeout after the bank accepted the request
    settle_after_polls:   bank reports settled after this many get_transfer_status calls
    """

    def __init__(self) -> None:
        self._submissions:  dict[str, str] = {}   # instruction_id → bank_ref
        self._poll_counts:  dict[str, int] = {}   # bank_ref → poll call count
        self.submit_calls:  list[dict]     = []
        self.submit_error:  Exception | None = None
        self.submit_unknown_once: bool = False
        self.settle_after_polls:  int  = 1

    async def submit_transfer(
        self,
        *,
        instruction_id:    str,
        amount:            Decimal,
        currency:          str,
        rail:              str,
        source_account:    str,
        destination_nostro: str,
    ) -> str:
        self.submit_calls.append({
            "instruction_id": instruction_id,
            "amount":         amount,
            "currency":       currency,
            "rail":           rail,
        })

        if self.submit_unknown_once:
            # Bank records it but we get a network error — mimics timeout
            bank_ref = f"MOCK-{instruction_id}"
            self._submissions[instruction_id] = bank_ref
            self._poll_counts[bank_ref] = 0
            self.submit_unknown_once = False
            raise ConnectionError("simulated network timeout (bank recorded transfer)")

        if self.submit_error is not None:
            raise self.submit_error

        bank_ref = f"MOCK-{instruction_id}"
        self._submissions[instruction_id] = bank_ref
        self._poll_counts[bank_ref] = 0
        return bank_ref

    async def get_transfer_status(self, bank_ref: str) -> dict[str, Any]:
        count = self._poll_counts.get(bank_ref, 0) + 1
        self._poll_counts[bank_ref] = count
        settled = count >= self.settle_after_polls
        return {
            "bank_ref":      bank_ref,
            "settled":       settled,
            "settled_amount": Decimal("1000.00") if settled else None,
        }

    async def find_by_instruction_id(self, instruction_id: str) -> str | None:
        return self._submissions.get(instruction_id)


# ── InMemoryExecutionStore ────────────────────────────────────────────────────


class InMemoryExecutionStore:
    """Single-threaded async in-memory store for TransferExecution objects."""

    def __init__(self) -> None:
        self._by_id:       dict[str, TransferExecution] = {}
        self._by_proposal: dict[str, str]               = {}  # proposal_id → execution_id

    def save(self, execution: TransferExecution) -> None:
        execution.touch()
        self._by_id[execution.execution_id]          = execution
        self._by_proposal[execution.proposal_id]     = execution.execution_id

    def get(self, execution_id: str) -> TransferExecution | None:
        return self._by_id.get(execution_id)

    def get_by_proposal_id(self, proposal_id: str) -> TransferExecution | None:
        eid = self._by_proposal.get(proposal_id)
        return self._by_id.get(eid) if eid else None

    def list_by_state(self, state: ExecutionState) -> list[TransferExecution]:
        return [e for e in self._by_id.values() if e.state == state]

    def list_all(self) -> list[TransferExecution]:
        return list(self._by_id.values())


# ── BalanceTracker ────────────────────────────────────────────────────────────


class BalanceTracker:
    """
    Tracks operating account balances and prevents double-spend.

    Workflow:
      reserve()  — lock funds before submission
      confirm()  — deduct locked funds on settlement
      release()  — return locked funds on failure / cancellation
    """

    def __init__(self, initial_balances: dict[str, Decimal]) -> None:
        self._balances: dict[str, Decimal] = dict(initial_balances)
        self._reserved: dict[str, Decimal] = defaultdict(Decimal)
        self._lock = asyncio.Lock()

    def available(self, currency: str) -> Decimal:
        return self._balances.get(currency, Decimal(0)) - self._reserved[currency]

    async def reserve(self, currency: str, amount: Decimal) -> None:
        async with self._lock:
            avail = self.available(currency)
            if avail < amount:
                raise InsufficientFunds(currency, avail, amount)
            self._reserved[currency] += amount

    async def release(self, currency: str, amount: Decimal) -> None:
        async with self._lock:
            self._reserved[currency] = max(Decimal(0), self._reserved[currency] - amount)

    async def confirm(self, currency: str, amount: Decimal) -> None:
        async with self._lock:
            self._balances[currency] = (
                self._balances.get(currency, Decimal(0)) - amount
            )
            self._reserved[currency] = max(Decimal(0), self._reserved[currency] - amount)


# ── FundMoverConfig ───────────────────────────────────────────────────────────


@dataclass
class FundMoverConfig:
    """Runtime configuration knobs for FundMover."""
    poll_interval_sec:      float            = 30.0
    sepa_instant_threshold: Decimal          = field(
        default_factory=lambda: SEPA_INSTANT_THRESHOLD
    )
    aed_cutoff_hour_gst:    int              = AED_CUTOFF_HOUR_GST
    # Per-rail SLA overrides (useful in tests — set to 0 for instant expiry)
    rail_sla_overrides:     dict[str, int]   = field(default_factory=dict)


# ── FundMover ─────────────────────────────────────────────────────────────────


class FundMover:
    """
    Idempotent fund transfer execution engine.

    Call execute_proposal() any number of times for the same proposal;
    it resumes from the current state without duplicating bank submissions.
    """

    def __init__(
        self,
        bank_api:        BankAPI,
        store:           InMemoryExecutionStore,
        balance_tracker: BalanceTracker,
        calendar_svc:    CalendarService | None = None,
        config:          FundMoverConfig | None  = None,
    ) -> None:
        self._bank    = bank_api
        self._store   = store
        self._tracker = balance_tracker
        self._cal     = calendar_svc
        self._cfg     = config or FundMoverConfig()

        self._operating_accounts: dict[str, str] = {
            "USD": "OPS-USD-001",
            "GBP": "OPS-GBP-001",
            "EUR": "OPS-EUR-001",
            "AED": "OPS-AED-001",
        }
        self._nostro_accounts: dict[str, str] = {
            "USD": "NOSTRO-USD-001",
            "GBP": "NOSTRO-GBP-001",
            "EUR": "NOSTRO-EUR-001",
            "AED": "NOSTRO-AED-001",
        }

    # ── Public API ────────────────────────────────────────────────────────

    def get_operating_account(self, currency: str) -> str:
        acc = self._operating_accounts.get(currency.upper())
        if not acc:
            raise ValueError(f"No operating account configured for {currency}")
        return acc

    def get_nostro_account(self, currency: str) -> str:
        acc = self._nostro_accounts.get(currency.upper())
        if not acc:
            raise ValueError(f"No nostro account configured for {currency}")
        return acc

    def available_balance(self, currency: str) -> Decimal:
        """Return the current available (unreserved) balance for a currency."""
        return self._tracker.available(currency)

    def all_balances(self) -> dict[str, dict[str, Decimal]]:
        """Return available, reserved, and total for every tracked currency."""
        result: dict[str, dict[str, Decimal]] = {}
        for ccy in self._tracker._balances:
            avail    = self._tracker.available(ccy)
            reserved = self._tracker._reserved.get(ccy, Decimal(0))
            total    = self._tracker._balances.get(ccy, Decimal(0))
            result[ccy] = {"available": avail, "reserved": reserved, "total": total}
        return result

    def list_executions(
        self,
        currency: str | None = None,
        state: str | None = None,
    ) -> list[TransferExecution]:
        """List executions, optionally filtered by currency and/or state."""
        execs = self._store.list_all()
        if currency:
            execs = [e for e in execs if e.currency.upper() == currency.upper()]
        if state:
            execs = [e for e in execs if e.state.value == state.lower()]
        return execs

    async def execute_proposal(
        self, proposal: FundMovementProposal
    ) -> TransferExecution:
        """
        Execute a fund movement proposal. Idempotent: safe to call multiple times.

        Returns a TransferExecution in CONFIRMED state on success.
        Raises SLABreached, SubmitUnknownError, AEDCutoffPassed, InsufficientFunds,
        or ExecutionAlreadyFailed.
        """
        # ── 1. Look up or create execution record ─────────────────────────
        execution = self._store.get_by_proposal_id(proposal.id)
        if execution is None:
            rail   = self._resolve_rail(proposal)
            amount = Decimal(str(proposal.amount)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            execution = TransferExecution(
                execution_id       = str(uuid.uuid4()),
                proposal_id        = proposal.id,
                instruction_id     = f"INST-{proposal.id}",
                currency           = proposal.currency.upper(),
                amount             = amount,
                source_account     = self.get_operating_account(proposal.currency),
                destination_nostro = self.get_nostro_account(proposal.currency),
                rail               = rail,
            )
            await self._tracker.reserve(execution.currency, execution.amount)
            self._store.save(execution)

        # ── 2. Advance state machine ───────────────────────────────────────
        return await self._advance(execution)

    # ── State machine ─────────────────────────────────────────────────────

    async def _advance(self, execution: TransferExecution) -> TransferExecution:
        state = execution.state

        if state == ExecutionState.CONFIRMED:
            return execution

        if state == ExecutionState.FAILED:
            raise ExecutionAlreadyFailed(execution)

        # Already on the bank's side — poll only, no re-submit
        if state in (ExecutionState.SUBMITTED, ExecutionState.SETTLING):
            return await self._poll_until_settled(execution)

        # In-flight or ambiguous — fence: ask bank if it knows our instruction
        if state in (ExecutionState.SUBMITTING, ExecutionState.SUBMIT_UNKNOWN):
            ref = await self._bank.find_by_instruction_id(execution.instruction_id)
            if ref:
                logger.info(
                    "fence: found existing submission instruction_id=%s bank_ref=%s",
                    execution.instruction_id, ref,
                )
                execution.bank_ref     = ref
                execution.state        = ExecutionState.SUBMITTED
                execution.submitted_at = datetime.now(timezone.utc)
                self._store.save(execution)
                return await self._poll_until_settled(execution)
            # Bank has no record — safe to retry
            logger.info(
                "fence: no record found for instruction_id=%s, reverting to QUEUED",
                execution.instruction_id,
            )
            execution.state = ExecutionState.QUEUED
            self._store.save(execution)

        # QUEUED — fresh submission
        return await self._submit_and_poll(execution)

    async def _submit_and_poll(self, execution: TransferExecution) -> TransferExecution:
        execution.state        = ExecutionState.SUBMITTING
        execution.attempt_count += 1
        self._store.save(execution)

        try:
            bank_ref = await self._bank.submit_transfer(
                instruction_id     = execution.instruction_id,
                amount             = execution.amount,
                currency           = execution.currency,
                rail               = execution.rail,
                source_account     = execution.source_account,
                destination_nostro = execution.destination_nostro,
            )
        except Exception as exc:
            # We don't know whether the bank recorded the transfer
            execution.state      = ExecutionState.SUBMIT_UNKNOWN
            execution.last_error = str(exc)
            self._store.save(execution)
            logger.warning(
                "submit returned unknown state: instruction_id=%s error=%s",
                execution.instruction_id, exc,
            )
            raise SubmitUnknownError(execution, exc) from exc

        execution.bank_ref     = bank_ref
        execution.state        = ExecutionState.SUBMITTED
        execution.submitted_at = datetime.now(timezone.utc)
        self._store.save(execution)
        logger.info(
            "transfer submitted: instruction_id=%s bank_ref=%s rail=%s",
            execution.instruction_id, bank_ref, execution.rail,
        )
        return await self._poll_until_settled(execution)

    async def _poll_until_settled(self, execution: TransferExecution) -> TransferExecution:
        """Poll until settled or SLA deadline passes."""
        execution.state = ExecutionState.SETTLING
        self._store.save(execution)

        _override = self._cfg.rail_sla_overrides.get(execution.rail)
        sla_min   = _override if _override is not None else RAIL_SLA_MIN.get(execution.rail, 120)
        submitted_at = execution.submitted_at or datetime.now(timezone.utc)
        deadline     = submitted_at + timedelta(minutes=sla_min)

        while datetime.now(timezone.utc) < deadline:
            await asyncio.sleep(self._cfg.poll_interval_sec)
            try:
                status = await self._bank.get_transfer_status(execution.bank_ref)
            except Exception as exc:
                logger.warning(
                    "poll failed: bank_ref=%s error=%s", execution.bank_ref, exc
                )
                continue

            if status.get("settled"):
                raw_amount             = status.get("settled_amount", execution.amount)
                execution.settled_amount = (
                    raw_amount if isinstance(raw_amount, Decimal)
                    else Decimal(str(raw_amount))
                )
                execution.state        = ExecutionState.CONFIRMED
                execution.confirmed_at = datetime.now(timezone.utc)
                self._store.save(execution)
                await self._tracker.confirm(execution.currency, execution.amount)
                logger.info(
                    "transfer confirmed: instruction_id=%s bank_ref=%s",
                    execution.instruction_id, execution.bank_ref,
                )
                return execution

        # SLA expired
        execution.state      = ExecutionState.FAILED
        execution.last_error = f"SLA breached after {sla_min}m"
        self._store.save(execution)
        await self._tracker.release(execution.currency, execution.amount)
        raise SLABreached(execution, sla_min)

    # ── Routing ───────────────────────────────────────────────────────────

    def _resolve_rail(self, proposal: FundMovementProposal) -> str:
        """Determine the payment rail from currency and amount."""
        currency = proposal.currency.upper()

        if currency == "USD":
            return "fedwire"

        if currency == "GBP":
            return "chaps"

        if currency == "EUR":
            amount = Decimal(str(proposal.amount))
            if amount < self._cfg.sepa_instant_threshold:
                return "sepa_instant"
            return "sepa"

        if currency == "AED":
            self._check_aed_window()
            return "bank_desk"

        raise ValueError(f"No rail configured for currency {currency}")

    def _check_aed_window(self) -> None:
        """Raise AEDCutoffPassed if outside the AED FX desk window."""
        now_utc   = datetime.now(timezone.utc)
        now_dubai = now_utc.astimezone(DUBAI_TZ)

        if now_dubai.hour >= self._cfg.aed_cutoff_hour_gst:
            raise AEDCutoffPassed(
                f"Current time {now_dubai.strftime('%H:%M')} GST is at or after "
                f"{self._cfg.aed_cutoff_hour_gst}:00 cutoff"
            )

        if self._cal is not None:
            today_dubai = self._cal.today(AE_BANKING, now_utc)
            if not self._cal.is_business_day(today_dubai, AE_BANKING):
                raise AEDCutoffPassed(
                    f"{today_dubai} is not a UAE banking day"
                )
