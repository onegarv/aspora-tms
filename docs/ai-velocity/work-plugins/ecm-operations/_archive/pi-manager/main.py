#!/usr/bin/env python3
"""
ECM Pi Manager - Main Entry Point

Auto-detects environment:
- K8s/VPS: Uses direct Redshift + Sheets connections (from env vars)
- Claude Code: Would use MCP (not directly runnable from CLI)

Usage:
    python main.py triage      # Full triage + assign + Slack post
    python main.py progress    # Progress report only
    python main.py test        # Test connections

Environment Variables (for K8s/VPS):
    REDSHIFT_HOST, REDSHIFT_PORT, REDSHIFT_DATABASE, REDSHIFT_USER, REDSHIFT_PASSWORD
    GOOGLE_SERVICE_ACCOUNT_JSON
    SLACK_BOT_TOKEN, SLACK_CHANNEL_ID
    SPREADSHEET_ID
"""
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config
from data_client import DirectRedshiftClient, DirectSheetsClient, SlackClient, DataService
from triage import TriageEngine, ProgressEngine
from slack_reporter import SlackReporter


def load_env():
    """Load .env file if present."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def create_services():
    """Create data services with direct connections."""
    config = Config.from_env()

    redshift = DirectRedshiftClient()
    sheets = DirectSheetsClient()
    slack = SlackClient(config.slack.bot_token)

    data_service = DataService(redshift, sheets, config)
    triage_engine = TriageEngine(config)
    progress_engine = ProgressEngine(config)
    slack_reporter = SlackReporter(slack, config)

    return config, data_service, triage_engine, progress_engine, slack_reporter


def cmd_triage():
    """Run full triage workflow."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting ECM Triage...")

    config, data_svc, triage, progress, slack = create_services()

    # Step 1: Get pending orders from Redshift
    print("  Querying pending orders...")
    orders = data_svc.get_pending_orders()
    print(f"  Found {len(orders)} actionable orders")

    # Step 2: Validate
    validation = triage.validate_data_quality(orders)
    if validation["status"] == "FAIL":
        print(f"  VALIDATION FAILED: {validation['warnings']}")
        slack.post_validation_failure(validation)
        return 1

    # Step 3: Get current assignments
    print("  Loading current assignments...")
    current = data_svc.get_current_assignments()
    assigned_ids = {a.order_id for a in current if a.status.value in ("OPEN", "IN_PROGRESS")}
    print(f"  {len(assigned_ids)} orders already assigned")

    # Step 4: Filter to new orders
    new_orders = [o for o in orders if o.order_id not in assigned_ids]
    print(f"  {len(new_orders)} new orders to assign")

    if not new_orders:
        print("  No new orders to assign")
        # Still post progress report
        report = progress.generate_report(current, [])
        slack.post_progress_report(report, [])
        return 0

    # Step 5: Get agents
    agents = data_svc.get_active_agents()
    print(f"  {len(agents)} active agents")

    # Step 6: Distribute
    assignments = triage.distribute_orders(new_orders, agents, current)
    print(f"  Created {len(assignments)} new assignments")

    # Step 7: Write to sheet
    start_row = len(current) + 2  # +1 for header, +1 for next row
    data_svc.write_assignments(assignments, start_row)
    print(f"  Written to sheet starting at row {start_row}")

    # Step 8: Post to Slack
    agent_summary = triage.summarize_by_agent(assignments, agents)
    currency_counts = triage.count_by_currency(new_orders)
    total_amount = sum(o.send_amount for o in new_orders)
    p1_count = sum(1 for a in assignments if a.priority.value == "P1")
    hv_count = sum(1 for o in new_orders if o.is_high_value)

    thread_ts = slack.post_daily_briefing(
        agent_summary, currency_counts, len(new_orders), p1_count, hv_count, total_amount
    )
    if thread_ts:
        slack.post_agent_threads(thread_ts, agent_summary)
        print(f"  Posted to Slack (thread: {thread_ts})")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Triage complete!")
    return 0


def cmd_progress():
    """Run progress report only."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating progress report...")

    config, data_svc, triage, progress, slack = create_services()

    # Get current state
    current = data_svc.get_current_assignments()
    open_ids = [a.order_id for a in current if a.status.value in ("OPEN", "IN_PROGRESS")]

    # Check for self-healed orders
    resolved_in_db = data_svc.get_resolved_in_redshift(open_ids)

    # Generate report
    report = progress.generate_report(current, resolved_in_db)
    agent_summary = triage.summarize_by_agent(
        [a for a in current if a.status.value in ("OPEN", "IN_PROGRESS")],
        data_svc.get_active_agents()
    )

    # Post
    slack.post_progress_report(report, agent_summary)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress report posted!")
    return 0


def cmd_test():
    """Test connections."""
    print("Testing connections...")

    load_env()

    # Test Redshift
    print("\n1. Redshift:")
    try:
        redshift = DirectRedshiftClient()
        result = redshift.execute("SELECT 1 as test")
        print(f"   OK: {result}")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Test Sheets
    print("\n2. Google Sheets:")
    try:
        sheets = DirectSheetsClient()
        spreadsheet_id = os.environ.get("SPREADSHEET_ID", "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks")
        result = sheets.get_data(spreadsheet_id, "Agents", "A1:A2")
        print(f"   OK: {result}")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Test Slack
    print("\n3. Slack:")
    try:
        token = os.environ.get("SLACK_BOT_TOKEN")
        if token:
            slack = SlackClient(token)
            print(f"   Token present: {token[:20]}...")
        else:
            print("   FAILED: No SLACK_BOT_TOKEN")
    except Exception as e:
        print(f"   FAILED: {e}")

    return 0


def main():
    load_env()

    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]

    if cmd == "triage":
        return cmd_triage()
    elif cmd == "progress":
        return cmd_progress()
    elif cmd == "test":
        return cmd_test()
    else:
        print(f"Unknown command: {cmd}")
        print("Available: triage, progress, test")
        return 1


if __name__ == "__main__":
    sys.exit(main())
