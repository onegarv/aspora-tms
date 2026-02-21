#!/bin/bash
# ECM Scheduled Triage - Posts daily briefing to Slack
# Runs without MCP - uses cached/static data + Slack API

SLACK_TOKEN="xoxb-2973233574978-10503818816130-eD4lIp8QwJ1f7A7TdarIyTt4"
SLACK_CHANNEL="C0AD6C36LVC"

python3 << EOF
import urllib.request
import json
from datetime import datetime

token = "$SLACK_TOKEN"
channel = "$SLACK_CHANNEL"
date_str = datetime.now().strftime("%b %d, %Y %I:%M %p")

message = f"""📋 *ECM Daily Briefing* — {date_str}

Good morning team! Time to clear the queue.

*Reminder - Your Actions:*
1. Run \`/my-tickets\` in Claude Code to see your queue
2. Run \`/order {{id}}\` for diagnosis + runbook
3. Run \`/resolve {{id}}\` when done

*Priority Focus:*
• REFUND_TRIGGERED orders — customer funds at risk
• Orders >36h old — clear the backlog
• High-value orders (≥5K) — top priority

📊 <https://docs.google.com/spreadsheets/d/1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks|Dashboard>
📚 <https://github.com/Vance-Club/ai-velocity/tree/stage-env/work-plugins/ecm-operations|Runbooks>

Let's clear this queue! 🚀"""

data = json.dumps({"channel": channel, "text": message, "unfurl_links": False}).encode('utf-8')
req = urllib.request.Request("https://slack.com/api/chat.postMessage", data=data,
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode('utf-8'))
    print(f"Posted: {result.get('ok')} at {date_str}")
EOF
