"""Triage and assignment logic for ECM Pi Manager."""
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Set

from .models import Order, Agent, Assignment, AssignmentStatus, Priority, ProgressReport
from .data_client import DataService
from .config import Config


class TriageEngine:
    """Handles triage, validation, and assignment distribution."""

    def __init__(self, data_service: DataService, config: Config):
        self.data = data_service
        self.config = config

    def run_triage(self) -> Tuple[List[Order], Dict[str, any]]:
        """
        Run full triage workflow.

        Returns:
            Tuple of (actionable_orders, validation_report)
        """
        # Step 1: Get pending orders
        orders = self.data.get_pending_orders()

        # Step 2: Validate data quality
        validation = self._validate_data_quality(orders)
        if validation["status"] == "FAIL":
            return [], validation

        # Step 3: Get current assignments
        assignments = self.data.get_current_assignments()
        assigned_ids = {a.order_id for a in assignments if a.status in
                       (AssignmentStatus.OPEN, AssignmentStatus.IN_PROGRESS, AssignmentStatus.ESCALATED)}

        # Step 4: Filter out already assigned
        unassigned = [o for o in orders if o.order_id not in assigned_ids]

        # Step 5: Sort by priority
        unassigned.sort(key=lambda o: (-o.priority_score, -o.send_amount))

        validation["unassigned_count"] = len(unassigned)
        validation["already_assigned"] = len(assigned_ids)

        return unassigned, validation

    def _validate_data_quality(self, orders: List[Order]) -> Dict[str, any]:
        """Validate order data before assignment."""
        count = len(orders)
        currency_counts = defaultdict(int)
        for o in orders:
            currency_counts[o.currency] += 1

        # Check count sanity
        status = "OK"
        warnings = []

        if count > self.config.triage.fail_count_threshold:
            status = "FAIL"
            warnings.append(f"Order count {count} exceeds fail threshold {self.config.triage.fail_count_threshold}")
        elif count > self.config.triage.expected_count_max:
            status = "WARNING"
            warnings.append(f"Order count {count} exceeds expected max {self.config.triage.expected_count_max}")

        # Check currency distribution
        total = sum(currency_counts.values()) or 1
        aed_pct = (currency_counts.get("AED", 0) / total) * 100
        gbp_pct = (currency_counts.get("GBP", 0) / total) * 100

        if aed_pct == 100 and count > 50:
            warnings.append("AED = 100% - GBP/EUR orders may be filtered incorrectly")
        if gbp_pct > 80:
            warnings.append(f"GBP = {gbp_pct:.1f}% - may include dead orders")

        return {
            "status": status,
            "order_count": count,
            "currency_distribution": dict(currency_counts),
            "warnings": warnings,
        }

    def distribute_orders(
        self,
        orders: List[Order],
        agents: List[Agent],
        current_assignments: List[Assignment]
    ) -> List[Assignment]:
        """
        Distribute orders to agents using value-weighted round-robin.

        High-value orders (>=5K) are distributed first, round-robin.
        Then regular orders are distributed round-robin.
        """
        # Calculate current tickets per agent
        agent_tickets = defaultdict(int)
        for a in current_assignments:
            if a.status in (AssignmentStatus.OPEN, AssignmentStatus.IN_PROGRESS):
                agent_tickets[a.agent_email] += 1

        for agent in agents:
            agent.current_tickets = agent_tickets.get(agent.email, 0)

        # Filter agents with capacity
        available_agents = [a for a in agents if a.available_slots > 0]
        if not available_agents:
            return []

        # Separate high-value and regular orders
        high_value = [o for o in orders if o.is_high_value]
        regular = [o for o in orders if not o.is_high_value]

        # Sort: high-value by amount DESC, regular by priority score DESC
        high_value.sort(key=lambda o: -o.send_amount)
        regular.sort(key=lambda o: -o.priority_score)

        # Round-robin assignment
        new_assignments = []
        agent_idx = 0
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Assign high-value first
        for order in high_value:
            if not available_agents:
                break

            agent = available_agents[agent_idx % len(available_agents)]
            if agent.available_slots > 0:
                assignment = self._create_assignment(order, agent, now)
                new_assignments.append(assignment)
                agent.current_tickets += 1

                if agent.available_slots <= 0:
                    available_agents = [a for a in available_agents if a.available_slots > 0]

            agent_idx += 1

        # Assign regular orders
        for order in regular:
            if not available_agents:
                break

            agent = available_agents[agent_idx % len(available_agents)]
            if agent.available_slots > 0:
                assignment = self._create_assignment(order, agent, now)
                new_assignments.append(assignment)
                agent.current_tickets += 1

                if agent.available_slots <= 0:
                    available_agents = [a for a in available_agents if a.available_slots > 0]

            agent_idx += 1

        return new_assignments

    def _create_assignment(self, order: Order, agent: Agent, timestamp: str) -> Assignment:
        """Create assignment record."""
        notes = self._generate_action_note(order)

        return Assignment(
            order_id=order.order_id,
            agent_email=agent.email,
            assigned_at=timestamp,
            status=AssignmentStatus.OPEN,
            priority=order.priority,
            currency=order.currency,
            amount=order.send_amount,
            hours_stuck=order.hours_stuck,
            diagnosis=order.stuck_reason or order.sub_state.lower(),
            notes=notes,
        )

    def _generate_action_note(self, order: Order) -> str:
        """Generate one-liner action note based on stuck_reason."""
        action_map = {
            "refund_triggered": f"REFUND {order.send_amount:,.0f} {order.currency}: Check acquirer refund queue",
            "trigger_refund": f"REFUND {order.send_amount:,.0f} {order.currency}: Initiate refund",
            "status_sync_issue": "Status sync: Force-sync via AlphaDesk, verify LULU=CREDITED",
            "falcon_failed_order_completed_issue": "Falcon sync: Verify payout at partner, force GOMS update",
            "brn_issue": "BRN push: Get ref from acquirer, push to Lulu",
            "stuck_at_lulu": "Lulu stuck: Check Lulu dashboard, escalate to Binoy if >48h",
            "stuck_at_lean_recon": "Lean recon: Check Lean Admin queue, escalate if stuck",
            "manual_review": "Manual review: Check AlphaDesk for pending action",
            "fulfillment_pending": "Check downstream status",
            "fulfillment_trigger": "About to fulfill - verify beneficiary details",
        }

        reason = order.stuck_reason or order.sub_state.lower()
        return action_map.get(reason, f"Investigate: {reason}")


class ProgressEngine:
    """Handles progress tracking and reporting."""

    def __init__(self, data_service: DataService, config: Config):
        self.data = data_service
        self.config = config

    def generate_report(self) -> ProgressReport:
        """Generate progress report with metrics."""
        report = ProgressReport()

        # Get current state
        assignments = self.data.get_current_assignments()

        # Count by status
        for a in assignments:
            if a.status == AssignmentStatus.OPEN:
                report.total_open += 1
            elif a.status == AssignmentStatus.IN_PROGRESS:
                report.total_in_progress += 1

            # Count by priority
            p = a.priority.value
            report.total_by_priority[p] = report.total_by_priority.get(p, 0) + 1

            # Count by agent (open only)
            if a.status in (AssignmentStatus.OPEN, AssignmentStatus.IN_PROGRESS):
                agent = a.agent_email.split("@")[0]
                report.total_by_agent[agent] = report.total_by_agent.get(agent, 0) + 1

            # High value tracking
            if a.amount >= self.config.triage.high_value_threshold:
                if a.status in (AssignmentStatus.OPEN, AssignmentStatus.IN_PROGRESS):
                    report.high_value_open += 1

        # Check for SLA breaches
        report.sla_breaches, report.sla_breach_orders = self._check_sla_breaches(assignments)

        # Check for self-healed orders (resolved in Redshift but still OPEN in sheet)
        open_order_ids = [a.order_id for a in assignments
                        if a.status in (AssignmentStatus.OPEN, AssignmentStatus.IN_PROGRESS)]
        resolved_ids = self.data.get_resolved_in_redshift(open_order_ids)
        report.self_healed = len(resolved_ids)

        return report

    def _check_sla_breaches(self, assignments: List[Assignment]) -> Tuple[int, List[Dict]]:
        """Check for SLA breaches."""
        breaches = []

        sla_map = {
            "refund_triggered": self.config.sla.refund_triggered,
            "trigger_refund": self.config.sla.refund_triggered,
            "manual_review": self.config.sla.manual_review,
            "status_sync_issue": self.config.sla.status_sync_issue,
            "brn_issue": self.config.sla.brn_pending,
            "rfi": self.config.sla.rfi_pending,
        }

        for a in assignments:
            if a.status not in (AssignmentStatus.OPEN, AssignmentStatus.IN_PROGRESS):
                continue

            sla_hours = sla_map.get(a.diagnosis, self.config.sla.default)
            if a.hours_stuck > sla_hours:
                breaches.append({
                    "order_id": a.order_id,
                    "agent": a.agent_email,
                    "hours": a.hours_stuck,
                    "sla": sla_hours,
                    "diagnosis": a.diagnosis,
                })

        return len(breaches), breaches[:10]  # Limit to top 10

    def get_agent_summary(self, assignments: List[Assignment], agents: List[Agent]) -> List[Dict]:
        """Get per-agent summary for reporting."""
        agent_stats = defaultdict(lambda: {
            "total": 0, "high_value": 0, "p1": 0, "orders": []
        })

        for a in assignments:
            if a.status not in (AssignmentStatus.OPEN, AssignmentStatus.IN_PROGRESS):
                continue

            email = a.agent_email
            agent_stats[email]["total"] += 1

            if a.amount >= self.config.triage.high_value_threshold:
                agent_stats[email]["high_value"] += 1
                agent_stats[email]["orders"].append({
                    "order_id": a.order_id,
                    "currency": a.currency,
                    "amount": a.amount,
                    "hours": a.hours_stuck,
                })

            if a.priority == Priority.P1:
                agent_stats[email]["p1"] += 1

        # Sort orders by amount DESC and limit to top 3
        for email in agent_stats:
            agent_stats[email]["orders"].sort(key=lambda x: -x["amount"])
            agent_stats[email]["orders"] = agent_stats[email]["orders"][:3]

        # Map to agent names
        agent_names = {a.email: a.name for a in agents}
        result = []
        for email, stats in agent_stats.items():
            result.append({
                "email": email,
                "name": agent_names.get(email, email.split("@")[0]),
                **stats
            })

        return sorted(result, key=lambda x: -x["total"])
