#!/usr/bin/env python3
"""
ECM Pi Manager — Main Entry Point

Scalable, cost-efficient scheduled triage and reporting for ECM operations.

Usage:
    # Full triage + assignment + Slack posting
    python -m pi_manager.src.main triage

    # Progress report only
    python -m pi_manager.src.main progress

    # Dry run (no writes)
    python -m pi_manager.src.main triage --dry-run

Environment:
    Requires .env file with SLACK_BOT_TOKEN, SPREADSHEET_ID, etc.
    See pi-skill/.env.template for full list.
"""
import sys
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import Config
from .models import Priority
from .data_client import (
    MCPRedshiftClient, MCPSheetsClient, SlackClient, DataService
)
from .triage import TriageEngine, ProgressEngine
from .slack_reporter import SlackReporter


class PiManager:
    """Main orchestrator for ECM Pi Manager."""

    def __init__(self, config: Config, mcp_executor: Callable = None):
        """
        Initialize Pi Manager.

        Args:
            config: Configuration object
            mcp_executor: MCP tool executor function (for Pi.dev/Claude Code)
                         If None, will use mock data for testing.
        """
        self.config = config

        if mcp_executor:
            redshift = MCPRedshiftClient(mcp_executor)
            sheets = MCPSheetsClient(mcp_executor)
        else:
            # For standalone testing, would need direct API clients
            raise ValueError("MCP executor required. Run via Pi.dev or Claude Code.")

        self.data_service = DataService(redshift, sheets, config)
        self.triage_engine = TriageEngine(self.data_service, config)
        self.progress_engine = ProgressEngine(self.data_service, config)

        self.slack_client = SlackClient(config.slack.bot_token)
        self.slack_reporter = SlackReporter(self.slack_client, config)

    def run_triage(self, dry_run: bool = False) -> dict:
        """
        Run full triage workflow.

        1. Query pending orders from Redshift
        2. Validate data quality
        3. Get current assignments from Sheet
        4. Distribute unassigned orders to agents
        5. Write new assignments to Sheet
        6. Post daily briefing to Slack

        Args:
            dry_run: If True, don't write to Sheet or post to Slack

        Returns:
            Summary dict with counts and status
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "dry_run": dry_run,
        }

        # Step 1-4: Run triage
        print("🔍 Running triage...")
        unassigned, validation = self.triage_engine.run_triage()

        result["validation"] = validation

        if validation["status"] == "FAIL":
            result["status"] = "validation_failed"
            if not dry_run:
                self.slack_reporter.post_validation_failure(validation)
            return result

        # Step 5: Get agents and distribute
        print(f"📋 Found {len(unassigned)} unassigned orders")
        agents = self.data_service.get_active_agents()
        current_assignments = self.data_service.get_current_assignments()

        new_assignments = self.triage_engine.distribute_orders(
            unassigned, agents, current_assignments
        )

        result["new_assignments"] = len(new_assignments)
        result["agents"] = [a.name for a in agents]

        # Step 6: Write to sheet
        if new_assignments and not dry_run:
            start_row = len(current_assignments) + 2  # +1 for header, +1 for next row
            print(f"📝 Writing {len(new_assignments)} assignments to sheet...")
            self.data_service.write_assignments(new_assignments, start_row)

        # Step 7: Calculate summary stats
        all_assignments = current_assignments + new_assignments
        agent_summary = self.progress_engine.get_agent_summary(all_assignments, agents)

        # Calculate totals
        total_orders = sum(a["total"] for a in agent_summary)
        p1_count = sum(
            1 for a in all_assignments
            if a.priority == Priority.P1 and a.status.value in ("OPEN", "IN_PROGRESS")
        )
        high_value_count = sum(a["high_value"] for a in agent_summary)
        total_amount = sum(
            a.amount for a in all_assignments
            if a.status.value in ("OPEN", "IN_PROGRESS")
        )

        result["summary"] = {
            "total_orders": total_orders,
            "p1_count": p1_count,
            "high_value_count": high_value_count,
            "total_amount": total_amount,
        }

        # Step 8: Post to Slack
        if not dry_run:
            print("📤 Posting to Slack...")
            thread_ts = self.slack_reporter.post_daily_briefing(
                agent_summary=agent_summary,
                currency_counts=validation.get("currency_distribution", {}),
                total_orders=total_orders,
                p1_count=p1_count,
                high_value_count=high_value_count,
                total_amount=total_amount,
            )

            if thread_ts:
                result["slack_thread_ts"] = thread_ts
                posted = self.slack_reporter.post_agent_threads(thread_ts, agent_summary)
                result["agent_threads_posted"] = posted

        print(f"✅ Triage complete: {len(new_assignments)} new assignments")
        return result

    def run_progress_report(self) -> dict:
        """
        Generate and post progress report.

        Includes:
        - Current queue status
        - SLA breaches
        - Self-healed orders
        - Agent workload

        Returns:
            Summary dict with report data
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
        }

        print("📊 Generating progress report...")

        # Get report data
        report = self.progress_engine.generate_report()
        agents = self.data_service.get_active_agents()
        assignments = self.data_service.get_current_assignments()
        agent_summary = self.progress_engine.get_agent_summary(assignments, agents)

        result["report"] = {
            "total_open": report.total_open,
            "total_in_progress": report.total_in_progress,
            "high_value_open": report.high_value_open,
            "sla_breaches": report.sla_breaches,
            "self_healed": report.self_healed,
            "priority_breakdown": report.total_by_priority,
        }

        # Post to Slack
        print("📤 Posting progress report to Slack...")
        self.slack_reporter.post_progress_report(report, agent_summary)

        # Post SLA breach alert if needed
        if report.sla_breaches >= 3:  # Only alert if 3+ breaches
            print(f"⚠️ Posting SLA breach alert ({report.sla_breaches} breaches)...")
            self.slack_reporter.post_sla_breach_alert(report.sla_breach_orders)

        print("✅ Progress report complete")
        return result

    def run_cleanup(self) -> dict:
        """
        Cleanup stale assignments.

        Marks orders as RESOLVED if they're COMPLETED in Redshift
        but still OPEN/IN_PROGRESS in the Sheet.

        Returns:
            Summary of cleaned up orders
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
        }

        print("🧹 Running cleanup...")

        # Get current assignments
        assignments = self.data_service.get_current_assignments()
        open_order_ids = [
            a.order_id for a in assignments
            if a.status.value in ("OPEN", "IN_PROGRESS")
        ]

        # Check which are resolved in Redshift
        resolved_ids = set(self.data_service.get_resolved_in_redshift(open_order_ids))

        result["checked"] = len(open_order_ids)
        result["self_healed"] = len(resolved_ids)
        result["resolved_ids"] = list(resolved_ids)[:20]  # Limit for output

        # TODO: Update sheet to mark as RESOLVED
        # This would require finding the row numbers and updating status column

        print(f"✅ Cleanup complete: {len(resolved_ids)} self-healed orders found")
        return result


def create_mcp_executor_for_claude_code():
    """
    Create MCP executor for Claude Code environment.

    This is a placeholder - in actual Claude Code, the MCP tools
    are called directly via the tool interface.
    """
    def executor(tool_name: str, params: dict):
        # This would be replaced by actual MCP tool calls in Claude Code
        raise NotImplementedError(
            "MCP executor must be provided by Claude Code environment. "
            "Use this module via Claude Code or Pi.dev."
        )
    return executor


# Entry point for command-line usage
def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    # Load config from env file
    env_file = Path(__file__).parent.parent.parent / "pi-skill" / ".env"
    config = Config.from_env(str(env_file))

    print(f"🚀 ECM Pi Manager — {command}")
    print(f"   Spreadsheet: {config.sheet.spreadsheet_id}")
    print(f"   Slack Channel: {config.slack.channel_name}")
    print()

    # Note: This requires MCP executor to be provided
    # In practice, run via Claude Code or Pi.dev
    try:
        mcp_executor = create_mcp_executor_for_claude_code()
        manager = PiManager(config, mcp_executor)

        if command == "triage":
            result = manager.run_triage(dry_run=dry_run)
        elif command == "progress":
            result = manager.run_progress_report()
        elif command == "cleanup":
            result = manager.run_cleanup()
        else:
            print(f"Unknown command: {command}")
            print("Available: triage, progress, cleanup")
            sys.exit(1)

        print()
        print(json.dumps(result, indent=2, default=str))

    except NotImplementedError as e:
        print(f"⚠️  {e}")
        print()
        print("To run this module:")
        print("  1. Use Claude Code with ecm-gateway MCP configured")
        print("  2. Or deploy to Pi.dev with MCP bridge")
        sys.exit(1)


if __name__ == "__main__":
    main()
