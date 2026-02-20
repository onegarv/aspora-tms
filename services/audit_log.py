"""
AuditLog — immutable, tamper-evident audit trail for all agent actions.

Every state change, decision, and approval is logged with:
  - Timestamp (UTC)
  - Agent name
  - Event type and action
  - Payload details
  - SHA-256 checksum of the payload (for tamper detection)
  - Acting user (defaults to "system")

Retention: 10 years (spec §6).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("tms.services.audit")


class AuditLog:
    """
    Persists audit entries to the database.

    `db` must expose:
        async insert(table: str, record: dict) -> None
    """

    def __init__(self, db) -> None:
        self.db = db

    async def log(
        self,
        event_type: str,
        agent: str,
        action: str,
        details: dict[str, Any],
        user: str = "system",
        correlation_id: str | None = None,
    ) -> None:
        """
        Write an immutable audit entry.

        The checksum covers (event_type + agent + action + details) so any
        post-hoc modification of the record is detectable.
        """
        checksum_payload = json.dumps(
            {
                "event_type": event_type,
                "agent": agent,
                "action": action,
                "details": details,
            },
            sort_keys=True,
        ).encode()
        checksum = hashlib.sha256(checksum_payload).hexdigest()

        record = {
            "timestamp":      datetime.utcnow().isoformat(),
            "event_type":     event_type,
            "agent":          agent,
            "action":         action,
            "details":        json.dumps(details, default=str),
            "user":           user,
            "correlation_id": correlation_id,
            "checksum":       checksum,
        }

        try:
            await self.db.insert("audit_log", record)
        except Exception as exc:
            # Audit failure must never crash the calling agent —
            # but it must be prominently logged for ops investigation.
            logger.critical(
                "AUDIT LOG WRITE FAILED",
                extra={
                    "event_type": event_type,
                    "agent": agent,
                    "action": action,
                    "error": str(exc),
                },
                exc_info=True,
            )

    async def verify_entry(self, entry: dict[str, Any]) -> bool:
        """
        Re-compute the checksum for an existing entry and compare.
        Returns True if the record is unmodified.
        """
        checksum_payload = json.dumps(
            {
                "event_type": entry["event_type"],
                "agent":      entry["agent"],
                "action":     entry["action"],
                "details":    json.loads(entry["details"]),
            },
            sort_keys=True,
        ).encode()
        expected = hashlib.sha256(checksum_payload).hexdigest()
        return expected == entry.get("checksum")
