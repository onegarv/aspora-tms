#!/usr/bin/env python3
"""
ECM Pi Manager Runner — Execute via Claude Code

This script provides the bridge between the Pi Manager module
and the MCP tools available in Claude Code.

Usage (in Claude Code):
    python run.py triage      # Full triage + assignment + Slack
    python run.py progress    # Progress report only
    python run.py cleanup     # Mark self-healed orders as resolved
    python run.py --dry-run triage  # Dry run (no writes)
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.models import Priority, AssignmentStatus


def load_env():
    """Load environment from .env file."""
    env_file = Path(__file__).parent.parent / "pi-skill" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def main():
    """Main entry point for Claude Code execution."""
    load_env()

    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if not a.startswith("--")]

    if not args:
        print(__doc__)
        return

    command = args[0]
    config = Config.from_env()

    print(f"🚀 ECM Pi Manager — {command}")
    print(f"   Dry run: {dry_run}")
    print()

    if command == "triage":
        run_triage(config, dry_run)
    elif command == "progress":
        run_progress(config)
    elif command == "cleanup":
        run_cleanup(config)
    elif command == "test":
        run_test(config)
    else:
        print(f"Unknown command: {command}")
        print("Available: triage, progress, cleanup, test")


def run_triage(config: Config, dry_run: bool = False):
    """
    Run triage workflow.

    This function is designed to be executed step-by-step in Claude Code,
    with each MCP call made explicitly by the agent.
    """
    print("="*60)
    print("TRIAGE WORKFLOW")
    print("="*60)
    print()
    print("Execute these steps in Claude Code:")
    print()
    print("Step 1: Query pending orders")
    print("   mcp__ecm-gateway__redshift_execute_sql_tool")
    print("   SQL: queries/ecm-pending-list.sql")
    print()
    print("Step 2: Validate data quality")
    print("   - Expected count: 200-600")
    print("   - Currency mix: AED 50-70%, GBP 25-40%, EUR 1-10%")
    print("   - Fail if count > 2000")
    print()
    print("Step 3: Get current assignments")
    print("   mcp__ecm-gateway__sheets_get_sheet_data")
    print(f"   spreadsheet_id: {config.sheet.spreadsheet_id}")
    print("   sheet: Assignments")
    print()
    print("Step 4: Get active agents")
    print("   mcp__ecm-gateway__sheets_get_sheet_data")
    print(f"   spreadsheet_id: {config.sheet.spreadsheet_id}")
    print("   sheet: Agents")
    print()
    print("Step 5: Distribute orders (round-robin)")
    print("   - High-value (≥5K) first, sorted by amount DESC")
    print("   - Then regular orders by priority score")
    print()
    print("Step 6: Write new assignments to sheet")
    print("   mcp__ecm-gateway__sheets_update_cells")
    print()
    print("Step 7: Post to Slack")
    print(f"   Channel: {config.slack.channel_id}")
    print(f"   Token: {config.slack.bot_token[:20]}...")
    print()
    print("="*60)


def run_progress(config: Config):
    """Run progress report workflow."""
    print("="*60)
    print("PROGRESS REPORT WORKFLOW")
    print("="*60)
    print()
    print("Execute these steps in Claude Code:")
    print()
    print("Step 1: Get current assignments")
    print("   mcp__ecm-gateway__sheets_get_sheet_data")
    print(f"   spreadsheet_id: {config.sheet.spreadsheet_id}")
    print("   sheet: Assignments")
    print()
    print("Step 2: Check for self-healed orders")
    print("   Query orders_goms for COMPLETED status")
    print("   Compare with OPEN assignments in sheet")
    print()
    print("Step 3: Check for SLA breaches")
    print("   SLA thresholds:")
    print(f"   - Refund: {config.sla.refund_triggered}h")
    print(f"   - Manual Review: {config.sla.manual_review}h")
    print(f"   - Default: {config.sla.default}h")
    print()
    print("Step 4: Post progress report to Slack")
    print(f"   Channel: {config.slack.channel_id}")
    print()
    print("="*60)


def run_cleanup(config: Config):
    """Run cleanup workflow."""
    print("="*60)
    print("CLEANUP WORKFLOW")
    print("="*60)
    print()
    print("Step 1: Get OPEN/IN_PROGRESS assignments")
    print()
    print("Step 2: Check Redshift for COMPLETED orders")
    print("""
SELECT order_id FROM orders_goms
WHERE order_id IN ({order_ids})
  AND status = 'COMPLETED'
""")
    print()
    print("Step 3: Mark self-healed orders as RESOLVED in sheet")
    print()
    print("="*60)


def run_test(config: Config):
    """Test configuration."""
    print("Configuration Test")
    print("="*60)
    print()
    print("Slack:")
    print(f"  Channel: {config.slack.channel_name} ({config.slack.channel_id})")
    print(f"  Token: {'✓ Set' if config.slack.bot_token else '✗ Missing'}")
    print()
    print("Google Sheet:")
    print(f"  ID: {config.sheet.spreadsheet_id}")
    print()
    print("Triage:")
    print(f"  High-value threshold: {config.triage.high_value_threshold}")
    print(f"  Time range: {config.triage.time_range_days} days")
    print(f"  Expected count: {config.triage.expected_count_min}-{config.triage.expected_count_max}")
    print()
    print("SLA Hours:")
    print(f"  Refund: {config.sla.refund_triggered}h")
    print(f"  Manual Review: {config.sla.manual_review}h")
    print(f"  Default: {config.sla.default}h")
    print()
    print("Excluded Agents:")
    for agent in config.triage.excluded_agents:
        print(f"  - {agent}")


if __name__ == "__main__":
    main()
