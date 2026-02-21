#!/usr/bin/env python3
"""
ECM Scheduled Reporter - Posts CEO-style briefings.

Two modes:
1. WITH Claude Code (--save): Saves live stats from MCP data to cache
2. SCHEDULED (default): Posts briefing from cached stats

Usage:
    # During Claude Code session - save current stats
    python3 scheduled-reporter.py --save '{"open": 397, "resolved": 95, ...}'

    # Scheduled run - post from cache
    python3 scheduled-reporter.py              # Morning briefing
    python3 scheduled-reporter.py --progress   # Progress report

Environment:
    SLACK_BOT_TOKEN - Slack bot token (or in .env file)
"""
import json
import os
import sys
import urllib.request
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Configuration
SLACK_CHANNEL = "C0AD6C36LVC"
SPREADSHEET_ID = "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}"
CACHE_FILE = Path(__file__).parent / ".ecm-stats-cache.json"


def get_slack_token():
    """Get Slack token from environment or fallback."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        # Fallback to .env file
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("SLACK_BOT_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break
    return token


def save_stats(stats_json):
    """Save stats from Claude Code session to cache."""
    stats = json.loads(stats_json)
    stats["cached_at"] = datetime.now().isoformat()
    with open(CACHE_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats cached at {CACHE_FILE}")
    return stats


def load_cached_stats():
    """Load stats from cache file."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE) as f:
            stats = json.load(f)
        # Check staleness (max 24 hours)
        cached_at = datetime.fromisoformat(stats.get("cached_at", "2000-01-01"))
        age_hours = (datetime.now() - cached_at).total_seconds() / 3600
        if age_hours > 24:
            print(f"Warning: Cache is {age_hours:.1f} hours old")
        return stats
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def analyze_assignments(assignments):
    """Calculate metrics from assignments."""
    if not assignments:
        return None

    # Status counts
    status_counts = defaultdict(int)
    agent_counts = defaultdict(lambda: {"open": 0, "resolved": 0, "high_value": 0, "value": 0})
    currency_counts = defaultdict(int)
    priority_counts = defaultdict(int)
    high_value_count = 0
    total_value = 0

    for a in assignments:
        status = a["status"]
        agent = a["agent"].split("@")[0] if a["agent"] else "unassigned"
        currency = a["currency"]
        amount = a["amount"]
        priority = a["priority"]

        status_counts[status] += 1
        currency_counts[currency] += 1
        priority_counts[priority] += 1

        if status in ("OPEN", "IN_PROGRESS"):
            agent_counts[agent]["open"] += 1
            agent_counts[agent]["value"] += amount
            total_value += amount
            if amount >= 5000:
                agent_counts[agent]["high_value"] += 1
                high_value_count += 1
        elif status == "RESOLVED":
            agent_counts[agent]["resolved"] += 1

    return {
        "total": len(assignments),
        "status_counts": dict(status_counts),
        "agent_counts": dict(agent_counts),
        "currency_counts": dict(currency_counts),
        "priority_counts": dict(priority_counts),
        "high_value_count": high_value_count,
        "total_value": total_value,
        "open_count": status_counts.get("OPEN", 0) + status_counts.get("IN_PROGRESS", 0),
        "resolved_count": status_counts.get("RESOLVED", 0),
    }


def format_morning_briefing(metrics):
    """Format CEO-style morning briefing."""
    date_str = datetime.now().strftime("%b %d, %Y %I:%M %p")

    # Agent summary
    agent_lines = []
    for agent, stats in sorted(metrics["agent_counts"].items(), key=lambda x: -x[1]["open"]):
        if stats["open"] > 0:
            hv = f" ({stats['high_value']} high-value)" if stats["high_value"] else ""
            agent_lines.append(f"• *{agent}* — {stats['open']} tickets{hv}")

    # Priority breakdown
    p1 = metrics["priority_counts"].get("P1", 0)
    p2 = metrics["priority_counts"].get("P2", 0)

    return f"""📊 *ECM Morning Briefing* — {date_str}

Good morning team! Here's your queue for today.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*📈 Queue Status*
• *{metrics['open_count']}* orders need attention
• *{metrics['resolved_count']}* resolved (keep it going!)
• *{metrics['high_value_count']}* high-value tickets (≥5K)
• *~{metrics['total_value']:,.0f}* total value

*🔴 Critical Focus*
• *{p1}* P1 tickets — clear these FIRST
• *{p2}* P2 tickets — next priority

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*👥 Team Assignments*
{chr(10).join(agent_lines)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*🛠️ Your Tools*
1. `/my-tickets` — see your queue
2. `/order {{id}}` — diagnosis + runbook
3. `/resolve {{id}}` — mark done

📊 <{SHEET_URL}|Dashboard> | 📚 <https://github.com/Vance-Club/ai-velocity/tree/stage-env/work-plugins/ecm-operations|Runbooks>

Let's clear the queue! 🚀"""


def format_progress_report(metrics):
    """Format CEO-style progress report."""
    date_str = datetime.now().strftime("%b %d, %Y %I:%M %p")

    resolved = metrics["resolved_count"]
    total = metrics["total"]
    open_count = metrics["open_count"]

    # Progress percentage
    if total > 0:
        progress_pct = (resolved / total) * 100
    else:
        progress_pct = 0

    # Agent progress
    agent_lines = []
    for agent, stats in sorted(metrics["agent_counts"].items(), key=lambda x: -x[1]["resolved"]):
        if stats["resolved"] > 0 or stats["open"] > 0:
            agent_lines.append(f"• *{agent}* — {stats['resolved']} resolved, {stats['open']} remaining")

    return f"""📈 *ECM Progress Report* — {date_str}

Here's where we stand this afternoon.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*📊 Progress*
• *{resolved}* tickets resolved ({progress_pct:.0f}% of total)
• *{open_count}* remaining
• *{metrics['high_value_count']}* high-value still open

*👥 Team Progress*
{chr(10).join(agent_lines[:6])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{"✅ *Great progress today!*" if resolved > 10 else "⚠️ *Let's push harder — update your tickets!*"}

📊 <{SHEET_URL}|Dashboard>"""


def post_to_slack(message, token):
    """Post message to Slack."""
    data = json.dumps({
        "channel": SLACK_CHANNEL,
        "text": message,
        "unfurl_links": False
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result.get("ok"), result.get("error")


def main():
    # Parse args
    is_progress = "--progress" in sys.argv

    # Get token
    token = get_slack_token()
    if not token:
        print("❌ SLACK_BOT_TOKEN not found")
        sys.exit(1)

    # Fetch data
    print("Fetching sheet data...")
    assignments = fetch_sheet_data()

    if not assignments:
        print("❌ Could not fetch sheet data")
        sys.exit(1)

    print(f"Found {len(assignments)} assignments")

    # Analyze
    metrics = analyze_assignments(assignments)

    # Format message
    if is_progress:
        message = format_progress_report(metrics)
        msg_type = "Progress Report"
    else:
        message = format_morning_briefing(metrics)
        msg_type = "Morning Briefing"

    # Post to Slack
    print(f"Posting {msg_type}...")
    ok, error = post_to_slack(message, token)

    if ok:
        print(f"✅ {msg_type} posted at {datetime.now().strftime('%I:%M %p')}")
    else:
        print(f"❌ Failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
