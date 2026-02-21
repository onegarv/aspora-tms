"""
GuardrailService — hackathon demo safety layer for ClearBank dispatches.

All checks run synchronously (no I/O) before any real payment is attempted.

Enforced limits:
  1. Kill switch         — emergency off; blocks everything
  2. Feature flag        — ASPORA_CLEARBANK_ENABLED must be true
  3. Currency allowlist  — GBP only (ClearBank is GBP-only)
  4. Amount cap          — ASPORA_DEMO_MAX_DISPATCH_GBP (default £500)
  5. Destination allowlist — ASPORA_CLEARBANK_ALLOWED_NOSTROS if set
  6. Rate limit          — ASPORA_DEMO_MAX_DISPATCHES_PER_HOUR (default 3)

In demo mode (confirm="DEMO"):
  Guardrails still run so operators see the same validation path.
  The actual HTTP call to ClearBank is skipped; a synthetic paymentId is returned.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from config.settings import settings
from models.domain import FundMovementProposal

logger = logging.getLogger("tms.guardrail")


class GuardrailViolation(Exception):
    """Raised when a dispatch is blocked by a guardrail check."""


@dataclass
class DispatchContext:
    proposal:    FundMovementProposal
    operator_id: str
    demo_mode:   bool = True    # True = dry run; False = real ClearBank call


class GuardrailService:
    """
    Stateful guardrail enforcer.

    Rate limit is tracked with an in-process sliding window — sufficient for
    a hackathon demo where a single process handles all requests.
    """

    def __init__(self) -> None:
        # Sliding window: monotonic timestamps of recent dispatches
        self._dispatch_timestamps: deque[float] = deque()

    # ── Public API ────────────────────────────────────────────────────────

    def check(self, ctx: DispatchContext) -> None:
        """
        Run all guardrails in order. Raises GuardrailViolation on first failure.
        Safe to call from async handlers (pure CPU, no I/O).
        """
        p = ctx.proposal

        # 1. Kill switch — emergency halt
        if settings.clearbank_kill_switch:
            raise GuardrailViolation(
                "ClearBank kill switch is active. All dispatches are blocked. "
                "Set ASPORA_CLEARBANK_KILL_SWITCH=false to resume."
            )

        # 2. Feature flag — must opt in
        if not settings.clearbank_enabled:
            raise GuardrailViolation(
                "ClearBank integration is disabled. "
                "Set ASPORA_CLEARBANK_ENABLED=true to enable live or demo dispatches."
            )

        # 3. Currency — ClearBank handles GBP only
        if p.currency.upper() != "GBP":
            raise GuardrailViolation(
                f"ClearBank dispatch requires GBP. Proposal currency is {p.currency}. "
                "Route non-GBP transfers through their respective payment rails."
            )

        # 4. Amount cap
        max_gbp = Decimal(str(settings.demo_max_dispatch_gbp))
        if p.amount > max_gbp:
            raise GuardrailViolation(
                f"Amount {p.amount} GBP exceeds the demo dispatch cap of {max_gbp} GBP. "
                f"Raise ASPORA_DEMO_MAX_DISPATCH_GBP (currently {max_gbp}) to increase."
            )

        # 5. Destination nostro allowlist
        allowed = settings.clearbank_allowed_nostros
        if allowed and p.destination_nostro not in allowed:
            raise GuardrailViolation(
                f"Destination '{p.destination_nostro}' is not in the ClearBank allowed "
                "nostro list. Add it to ASPORA_CLEARBANK_ALLOWED_NOSTROS."
            )

        # 6. Rate limit
        self._evict_old_timestamps()
        limit = settings.demo_max_dispatches_per_hour
        if len(self._dispatch_timestamps) >= limit:
            raise GuardrailViolation(
                f"Rate limit reached: max {limit} dispatches per hour. "
                "Wait before retrying, or raise ASPORA_DEMO_MAX_DISPATCHES_PER_HOUR."
            )

        logger.info(
            "guardrail check passed: proposal=%s amount=%s GBP demo=%s operator=%s",
            p.id, p.amount, ctx.demo_mode, ctx.operator_id,
        )

    def record_dispatch(self) -> None:
        """Call after a successful dispatch to register it with the rate limiter."""
        self._dispatch_timestamps.append(time.monotonic())

    def rate_limit_status(self) -> dict:
        """Return current rate limit usage (useful for health / debug endpoints)."""
        self._evict_old_timestamps()
        limit = settings.demo_max_dispatches_per_hour
        used  = len(self._dispatch_timestamps)
        return {
            "dispatches_used":      used,
            "dispatches_limit":     limit,
            "dispatches_remaining": max(0, limit - used),
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _evict_old_timestamps(self) -> None:
        cutoff = time.monotonic() - 3600.0
        while self._dispatch_timestamps and self._dispatch_timestamps[0] < cutoff:
            self._dispatch_timestamps.popleft()
