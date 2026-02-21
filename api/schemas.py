"""
Pydantic request/response schemas for the Dashboard API.

Amounts are serialised as strings to avoid JSON float precision loss.
Timestamps are ISO-8601 UTC strings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dt(dt: datetime | None) -> str | None:
    """Convert datetime | None → ISO-8601 UTC string or None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _dec(d: Decimal | float | None) -> str:
    """Convert Decimal | float | None → str."""
    if d is None:
        return "0"
    return str(d)


# ── Balance schemas ───────────────────────────────────────────────────────────

class BalanceDetail(BaseModel):
    currency: str
    available: str
    reserved: str
    total: str


# ── Proposal schemas ──────────────────────────────────────────────────────────

class ProposalSummary(BaseModel):
    id: str
    currency: str
    amount: str
    status: str
    rail: str
    proposed_by: str
    created_at: str | None


class ProposalDetail(ProposalSummary):
    source_account: str
    destination_nostro: str
    purpose: str
    approved_by: str | None = None
    rejected_by: str | None = None
    rejection_reason: str | None = None
    executed_at: str | None = None
    confirmed_at: str | None = None
    validation_errors: list[str] = []


class ApproveRequest(BaseModel):
    checker_id: str
    note: str = ""


class RejectRequest(BaseModel):
    checker_id: str
    reason: str


# ── Transfer schemas ──────────────────────────────────────────────────────────

class TransferSummary(BaseModel):
    execution_id: str
    proposal_id: str
    currency: str
    amount: str
    state: str
    rail: str
    submitted_at: str | None


class TransferDetail(TransferSummary):
    instruction_id: str
    bank_ref: str | None = None
    settled_amount: str | None = None
    confirmed_at: str | None = None
    last_error: str | None = None
    attempt_count: int


# ── Window schemas ────────────────────────────────────────────────────────────

class WindowStatus(BaseModel):
    currency: str
    rail: str
    status: str  # "open" | "closing" | "closed"
    minutes_until_close: int | None
    close_time_utc: str | None


# ── Exposure schema ───────────────────────────────────────────────────────────

class ExposureResponse(BaseModel):
    as_of: str | None = None
    total_inr_required: str | None = None
    covered_inr: str | None = None
    open_inr: str | None = None
    blended_rate: str | None = None
    deal_count: int | None = None


# ── Event schema ──────────────────────────────────────────────────────────────

class EventSummary(BaseModel):
    event_id: str
    event_type: str
    source_agent: str
    timestamp_utc: str
    correlation_id: str
    payload: dict[str, Any]


# ── FX Deal schema ─────────────────────────────────────────────────────────────

class FxDealResponse(BaseModel):
    id: str
    corridor: str           # "USD_INR", "GBP_INR", etc.
    side: str               # "BUY" | "SELL"
    amount_usd: str
    rate: str
    amount_native: str
    status: str             # "OPEN" | "COVERED" | "SETTLED" | "CANCELLED"
    executed_at: str
    trader: str


# ── PnL schema ────────────────────────────────────────────────────────────────

class PnLEntryResponse(BaseModel):
    currency: str
    realised: str
    unrealised: str
    total: str
    as_of: str


# ── Forecast schema ───────────────────────────────────────────────────────────

class DailyForecastResponse(BaseModel):
    forecast_date: str
    total_inr_crores: str
    confidence: str         # "low" | "medium" | "high"
    currency_split: dict[str, str]
    multipliers_applied: dict[str, str]
    created_at: str


class ShortfallAlertResponse(BaseModel):
    currency: str
    required_amount: str
    available_balance: str
    shortfall: str
    severity: str           # "warning" | "critical"
    detected_at: str


# ── Risk Limits schema ────────────────────────────────────────────────────────

class RiskLimitsResponse(BaseModel):
    max_single_deal_usd: str
    max_open_exposure_pct: str
    stop_loss_paise: str
    dual_checker_threshold_usd: str
    prefunding_buffer_pct: str
    updated_at: str
    updated_by: str


class RiskLimitsUpdate(BaseModel):
    max_single_deal_usd: Optional[str] = None
    max_open_exposure_pct: Optional[str] = None
    stop_loss_paise: Optional[str] = None
    dual_checker_threshold_usd: Optional[str] = None
    prefunding_buffer_pct: Optional[str] = None
