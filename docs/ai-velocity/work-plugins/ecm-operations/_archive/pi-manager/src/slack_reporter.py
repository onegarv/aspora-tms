"""Slack reporting for ECM Pi Manager."""
from datetime import datetime
from typing import List, Dict, Optional

from .data_client import SlackClient
from .models import ProgressReport
from .config import Config


class SlackReporter:
    """Handles all Slack messaging."""

    SHEET_URL = "https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
    GITHUB_URL = "https://github.com/Vance-Club/ai-velocity/tree/stage-env/work-plugins/ecm-operations"

    def __init__(self, client: SlackClient, config: Config):
        self.client = client
        self.config = config
        self.sheet_url = self.SHEET_URL.format(spreadsheet_id=config.sheet.spreadsheet_id)

    def post_daily_briefing(
        self,
        agent_summary: List[Dict],
        currency_counts: Dict[str, int],
        total_orders: int,
        p1_count: int,
        high_value_count: int,
        total_amount: float,
    ) -> Optional[str]:
        """Post daily briefing to Slack. Returns thread timestamp."""
        date_str = datetime.utcnow().strftime("%b %d, %Y")

        currency_line = " | ".join(f"{k}: {v}" for k, v in sorted(currency_counts.items()))
        agent_lines = "\n".join(
            f"👤 *{a['name']}* — {a['total']} tickets ({a['high_value']} high-value)"
            for a in agent_summary
        )

        message = f"""📋 *ECM Daily Briefing* — {date_str}

Hey team! Here's your updated queue for today.

*Queue Summary:*
• {total_orders} orders across {len(agent_summary)} agents
• 💰 {high_value_count} high-value orders (≥5K)
• 🔴 {p1_count} P1 critical tickets
• 💵 ~{total_amount:,.0f} total value

*Currency Mix:* {currency_line}

*Your Assignments:*
{agent_lines}

📊 *Dashboard:* <{self.sheet_url}|ECM Assignments Sheet>

---
*Getting Started:*
1. Run `/my-tickets` to see your queue
2. Run `/order {{id}}` for diagnosis + runbook steps
3. Run `/resolve {{id}}` when done

📚 *Skills & Runbooks:* <{self.GITHUB_URL}|GitHub Repo>"""

        return self.client.post_message(self.config.slack.channel_id, message)

    def post_agent_threads(self, thread_ts: str, agent_summary: List[Dict]) -> int:
        """Post per-agent threads. Returns count of threads posted."""
        posted = 0

        for agent in agent_summary:
            if not agent.get("orders"):
                continue

            hv_lines = "\n".join(
                f"• `{o['order_id']}` — {o['amount']:,.0f} {o['currency']} ({o['hours']:.0f}h)"
                for o in agent["orders"]
            )

            first_order = agent["orders"][0]["order_id"] if agent["orders"] else ""

            message = f"""👤 *{agent['name']}* — {agent['total']} tickets

💰 *Top Priority (High Value):*
{hv_lines}

Run: `/order {first_order}`"""

            if self.client.post_message(self.config.slack.channel_id, message, thread_ts):
                posted += 1

        return posted

    def post_progress_report(self, report: ProgressReport, agent_summary: List[Dict]) -> Optional[str]:
        """Post progress report to Slack."""
        date_str = datetime.utcnow().strftime("%b %d, %Y %H:%M UTC")

        # Priority breakdown
        priority_line = " | ".join(
            f"{k}: {v}" for k, v in sorted(report.total_by_priority.items())
        )

        # Self-healed section
        self_healed_section = ""
        if report.self_healed > 0:
            self_healed_section = f"\n🔄 *Self-healed:* {report.self_healed} orders resolved automatically"

        # SLA breach section
        sla_section = ""
        if report.sla_breaches > 0:
            breach_lines = "\n".join(
                f"• `{b['order_id']}` — {b['agent'].split('@')[0]} — {b['hours']:.0f}h (SLA: {b['sla']}h)"
                for b in report.sla_breach_orders[:5]
            )
            sla_section = f"""

⚠️ *SLA Breaches:* {report.sla_breaches} tickets
{breach_lines}"""

        message = f"""📈 *ECM Progress Report* — {date_str}

*Current Queue:*
• Open: {report.total_open}
• In Progress: {report.total_in_progress}
• High Value Open: {report.high_value_open}

*Priority Breakdown:* {priority_line}
{self_healed_section}{sla_section}

📊 *Dashboard:* <{self.sheet_url}|View Full Dashboard>"""

        return self.client.post_message(self.config.slack.channel_id, message)

    def post_sla_breach_alert(self, breaches: List[Dict]) -> Optional[str]:
        """Post SLA breach alert."""
        if not breaches:
            return None

        breach_lines = "\n".join(
            f"🔴 `{b['order_id']}` — {b['agent'].split('@')[0]} — {b['hours']:.0f}h (SLA: {b['sla']}h) — {b['diagnosis']}"
            for b in breaches[:10]
        )

        message = f"""⚠️ *SLA BREACH ALERT*

{len(breaches)} tickets have exceeded their SLA:

{breach_lines}

Agents: Please prioritize these immediately!"""

        return self.client.post_message(self.config.slack.channel_id, message)

    def post_validation_failure(self, validation: Dict) -> Optional[str]:
        """Post validation failure alert."""
        warnings = "\n".join(f"• {w}" for w in validation.get("warnings", []))
        currency_dist = validation.get("currency_distribution", {})
        currency_line = " | ".join(f"{k}: {v}" for k, v in currency_dist.items())

        message = f"""⚠️ *ECM Triage Validation Failed*

*Status:* {validation.get('status')}
*Order Count:* {validation.get('order_count')}
*Currency Mix:* {currency_line}

*Issues:*
{warnings}

Manual review required. @dinesh"""

        return self.client.post_message(self.config.slack.channel_id, message)
