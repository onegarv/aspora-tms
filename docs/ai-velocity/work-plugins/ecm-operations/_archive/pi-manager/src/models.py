"""Data models for ECM Pi Manager."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class Priority(Enum):
    P1 = "P1"  # Critical (score > 0.7)
    P2 = "P2"  # High (0.5-0.7)
    P3 = "P3"  # Medium (0.3-0.5)
    P4 = "P4"  # Low (< 0.3)

class AssignmentStatus(Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"

@dataclass
class Order:
    order_id: str
    order_date: str
    status: str
    sub_state: str
    currency: str
    send_amount: float
    receive_amount: float
    hours_stuck: float
    category: str  # level_zero, warning, action_required, critical

    # Enriched fields (from detail query)
    stuck_reason: Optional[str] = None
    team_dependency: Optional[str] = None
    payment_status: Optional[str] = None
    lulu_status: Optional[str] = None
    falcon_status: Optional[str] = None

    @property
    def is_high_value(self) -> bool:
        return self.send_amount >= 5000

    @property
    def priority_score(self) -> float:
        """Calculate priority score using knowledge graph formula."""
        age_score = self._age_score()
        amount_score = self._amount_score()
        severity_score = self._severity_score()

        return (
            0.25 * age_score +
            0.20 * amount_score +
            0.25 * severity_score +
            0.15 * 0.5 +  # RFI placeholder
            0.10 * 0.3 +  # Payment risk placeholder
            0.05 * 0.1    # Attempts placeholder
        )

    @property
    def priority(self) -> Priority:
        score = self.priority_score
        if score > 0.7:
            return Priority.P1
        elif score > 0.5:
            return Priority.P2
        elif score > 0.3:
            return Priority.P3
        return Priority.P4

    def _age_score(self) -> float:
        if self.hours_stuck > 72:
            return 1.0
        elif self.hours_stuck > 36:
            return 0.9
        elif self.hours_stuck > 24:
            return 0.7
        elif self.hours_stuck > 12:
            return 0.5
        return 0.2

    def _amount_score(self) -> float:
        if self.send_amount > 15000:
            return 1.0
        elif self.send_amount > 5000:
            return 0.8
        elif self.send_amount > 2000:
            return 0.7
        elif self.send_amount > 500:
            return 0.5
        return 0.2

    def _severity_score(self) -> float:
        severity_map = {
            "refund_pending": 1.0,
            "refund_triggered": 1.0,
            "trigger_refund": 1.0,
            "falcon_failed_order_completed_issue": 0.9,
            "no_rfi_created": 0.8,
            "status_sync_issue": 0.6,
            "brn_issue": 0.5,
            "stuck_at_lulu": 0.5,
        }
        return severity_map.get(self.stuck_reason or "", 0.4)

@dataclass
class Agent:
    email: str
    name: str
    team: str
    slack_handle: str
    active: bool
    max_tickets: int
    current_tickets: int = 0

    @property
    def available_slots(self) -> int:
        return max(0, self.max_tickets - self.current_tickets)

    @property
    def capacity_pct(self) -> float:
        if self.max_tickets == 0:
            return 100.0
        return (self.current_tickets / self.max_tickets) * 100

@dataclass
class Assignment:
    order_id: str
    agent_email: str
    assigned_at: str
    status: AssignmentStatus
    priority: Priority
    currency: str
    amount: float
    hours_stuck: float
    diagnosis: str
    notes: str

@dataclass
class ProgressReport:
    """Progress metrics for reporting."""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Current state
    total_open: int = 0
    total_in_progress: int = 0
    total_by_priority: dict = field(default_factory=dict)
    total_by_agent: dict = field(default_factory=dict)

    # Changes since last run
    new_orders: int = 0
    resolved_orders: int = 0
    self_healed: int = 0  # Resolved without agent action

    # SLA metrics
    sla_breaches: int = 0
    sla_breach_orders: list = field(default_factory=list)

    # Performance
    avg_resolution_hours: float = 0.0
    resolution_rate_24h: float = 0.0

    # High value tracking
    high_value_open: int = 0
    high_value_resolved_today: int = 0
