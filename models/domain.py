"""
Core domain models for Aspora TMS.

These are plain dataclasses used throughout the agent layer.
ORM / DB models live in models/db.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class ProposalStatus(str, Enum):
    PENDING          = "pending"
    PENDING_APPROVAL = "pending_approval"
    APPROVED         = "approved"
    REJECTED         = "rejected"
    EXECUTED         = "executed"
    CONFIRMED        = "confirmed"
    FAILED           = "failed"
    CANCELLED        = "cancelled"


class ShortfallSeverity(str, Enum):
    WARNING  = "warning"   # buffer at risk but not yet critical
    CRITICAL = "critical"  # prefunding will fail without immediate action


class ForecastConfidence(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ── Liquidity / Forecast models ────────────────────────────────────────────────

@dataclass
class DailyForecast:
    """Emitted by the Liquidity Agent each morning."""
    forecast_date: date
    total_inr_crores: float
    confidence: ForecastConfidence
    currency_split: dict[str, float]         # {"USD": 45.2, "GBP": 12.1, ...}
    multipliers_applied: dict[str, float]    # {"payday": 1.4, "holiday": 1.0, ...}
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RDAShortfall:
    """Shortfall alert emitted by the Liquidity Agent when a nostro is underfunded."""
    currency: str
    required_amount: float
    available_balance: float
    shortfall: float
    severity: ShortfallSeverity
    detected_at: datetime = field(default_factory=datetime.utcnow)


# ── FX / Deal models ───────────────────────────────────────────────────────────

@dataclass
class DealInstruction:
    """Instruction from the FX Analyst Agent to execute a currency deal."""
    id: str
    currency_pair: str               # e.g. "USD/INR"
    amount_foreign: float
    amount_inr: float
    deal_type: str                   # "spot" | "tom" | "cash"
    target_rate: float
    time_window_start: datetime
    time_window_end: datetime
    tranche_number: int
    total_tranches: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExposureStatus:
    """Running tally of covered vs open INR exposure."""
    as_of: datetime
    total_inr_required: float
    covered_inr: float
    open_inr: float
    blended_rate: float
    deal_count: int


# ── Operations / Fund Movement models ─────────────────────────────────────────

@dataclass
class FundMovementProposal:
    """
    A proposed inter-account transfer, created by the Maker and approved by the Checker.

    Lifecycle: pending → pending_approval → approved → executed → confirmed
                                         └→ rejected
    """
    id: str
    currency: str                        # ISO 4217 e.g. "USD"
    amount: float                        # Amount in source currency
    source_account: str                  # Operating account ID
    destination_nostro: str              # Destination nostro account ID
    rail: str                            # "fedwire" | "chaps" | "target2" | "swift"
    proposed_by: str                     # User ID or "system:<agent_name>"
    purpose: str                         # Human-readable reason
    idempotency_key: str                 # Prevents duplicate submissions

    status: ProposalStatus = ProposalStatus.PENDING
    validation_errors: list[str] = field(default_factory=list)

    # Approval tracking
    approved_by: Optional[str] = None   # Checker user ID
    second_approver: Optional[str] = None  # For dual-checker transfers
    rejected_by: Optional[str] = None
    rejection_reason: Optional[str] = None

    # Execution tracking
    executed_at: Optional[datetime] = None
    settlement_ref: Optional[str] = None
    confirmed_at: Optional[datetime] = None
    expected_arrival: Optional[datetime] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def requires_dual_approval(self) -> bool:
        """Transfers above the dual-checker threshold need two approvers."""
        from config.settings import settings
        return self.amount > settings.dual_checker_threshold_usd

    @property
    def approvals_count(self) -> int:
        return sum(1 for a in [self.approved_by, self.second_approver] if a is not None)


@dataclass
class TransferConfirmation:
    """Settlement confirmation received from the bank."""
    proposal_id: str
    settlement_ref: str
    confirmed_at: datetime
    settled_amount: float
    currency: str
    rail: str


@dataclass
class WindowClosingAlert:
    """Fired by the Operations Agent when a banking window is about to close."""
    currency: str
    rail: str
    minutes_remaining: int
    close_time_utc: datetime
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HolidayLookahead:
    """3-day lookahead of holidays across all jurisdictions."""
    generated_at: datetime
    # Keys: ISO date strings. Values: list of jurisdiction codes closed on that day.
    holidays: dict[str, list[str]]  # {"2026-02-21": ["IN", "US"], ...}


@dataclass
class FundMovementStatus:
    """Published to the bus whenever a proposal changes state."""
    proposal_id: str
    currency: str
    amount: float
    status: ProposalStatus
    rail: str
    settlement_ref: Optional[str] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
