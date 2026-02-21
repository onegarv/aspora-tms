# Post Slack Script

This sub-skill posts the ECM triage report to Slack. Pi reads this and executes.

## Prerequisites

- Environment variable: `SLACK_WEBHOOK_URL` (Slack incoming webhook for #ops-ecm)
- Input: Triage result JSON from `run-triage.md`

## Input Format

Expects JSON with this structure:

```json
{
  "timestamp": "2026-02-12T07:00:00Z",
  "summary": {
    "total_stuck": 45,
    "already_assigned": 30,
    "new_to_assign": 8,
    "critical_count": 3,
    "high_value_count": 5,
    "disqualified_count": 7
  },
  "assignments": [
    {
      "order_id": "AE136ABC00",
      "agent_email": "dinesh@aspora.com",
      "priority": "P1",
      "stuck_reason": "refund_pending",
      "action_note": "REFUND 5000 AED: Check acquirer refund queue"
    }
  ],
  "agent_capacity": [
    {
      "email": "akshay@aspora.com",
      "name": "Akshay",
      "slack_handle": "@akshay",
      "current_tickets": 102
    }
  ],
  "high_value_tickets": [
    {
      "order_id": "AE12Y0K4BU00",
      "amount": 60100,
      "currency": "AED",
      "agent_email": "raj.kumar@aspora.com",
      "agent_name": "Raj Kumar",
      "priority": "P1",
      "stuck_reason": "stuck_at_lulu"
    }
  ]
}
```

## Execution Steps

### Step 1: Determine Time of Day

Based on current hour:
- 0-11: "Morning"
- 12-16: "Afternoon"
- 17-23: "Evening"

### Step 2: Build Slack Message

Use Slack Block Kit format:

```json
{
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "{emoji} ECM {time_of_day} Triage — {date}",
        "emoji": true
      }
    },
    {
      "type": "section",
      "fields": [
        { "type": "mrkdwn", "text": "*Total Stuck:*\n{total_stuck}" },
        { "type": "mrkdwn", "text": "*New Assigned:*\n{new_to_assign}" },
        { "type": "mrkdwn", "text": "*Critical (>36h):*\n{critical_count} :red_circle:" },
        { "type": "mrkdwn", "text": "*High Value (>5K):*\n{high_value_count} :moneybag:" }
      ]
    },
    { "type": "divider" },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*:clipboard: New Assignments:*\n{assignment_rows}"
      }
    },
    { "type": "divider" },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*:busts_in_silhouette: Agent Distribution:*\n{capacity_rows}"
      }
    },
    { "type": "divider" },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*:moneybag: High-Value Tickets (>5K):*\n{high_value_rows}"
      }
    },
    {
      "type": "context",
      "elements": [
        {
          "type": "mrkdwn",
          "text": "<https://docs.google.com/spreadsheets/d/1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks/edit|:link: View Dashboard> | Run `/my-tickets` to see your queue"
        }
      ]
    }
  ]
}
```

### Step 3: Format Header Emoji

```
IF critical_count > 5:
  emoji = ":rotating_light:"
ELSE IF critical_count > 0:
  emoji = ":warning:"
ELSE:
  emoji = ":dart:"
```

### Step 4: Format Assignment Rows

For each assignment (max 10):

```
{priority_emoji} `{order_id}` → @{agent_name} | {stuck_reason}
```

Priority emojis:
- P1: `:red_circle:`
- P2: `:large_orange_circle:`
- P3: `:white_circle:`
- P4: `:black_circle:`

**Example:**
```
:red_circle: `AE136ABC00` → @dinesh | refund_pending
:large_orange_circle: `AE137DEF00` → @akshay | stuck_at_lulu
```

If no assignments:
```
_No new assignments this cycle_
```

### Step 5: Format Capacity Rows

For each agent:
```
• @{slack_handle}: {current_tickets} tickets
```

**Example:**
```
• @akshay: 102 tickets
• @vishnu: 84 tickets
• @abhijith: 84 tickets
```

### Step 5b: Format High-Value Tickets Section

Query for tickets with amount > 5000 (in local currency):

```sql
SELECT order_id, amount, currency, assigned_agent
FROM assignments
WHERE amount > 5000
ORDER BY amount DESC
LIMIT 10
```

Format each high-value ticket:
```
{priority_emoji} `{order_id}` {amount:,} {currency} → @{agent_name} | {stuck_reason}
```

**Example:**
```
:red_circle: `AE12Y0K4BU00` 60,100 AED → @raj.kumar | stuck_at_lulu
:red_circle: `AE132F6U6F00` 30,000 AED → @raj.kumar | stuck_at_lulu
:large_orange_circle: `AE130O9QK900` 10,000 AED → @raj.kumar | stuck_at_lulu
```

Add this as a new section block after Agent Distribution:
```json
{
  "type": "section",
  "text": {
    "type": "mrkdwn",
    "text": "*:moneybag: High-Value Tickets (>5K):*\n{high_value_rows}"
  }
}
```

### Step 6: Post to Slack (with Threading)

Use Slack Web API with bot token for threading support:

**Step 6a: Post main message**
```bash
curl -X POST "https://slack.com/api/chat.postMessage" \
  -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{
    "channel": "'"$SLACK_CHANNEL_ID"'",
    "text": "ECM Triage Complete",
    "blocks": [{main_message_blocks}]
  }'
```

Capture the `ts` from response: `response.ts`

**Step 6b: Post high-value tickets as thread reply**
```bash
curl -X POST "https://slack.com/api/chat.postMessage" \
  -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{
    "channel": "'"$SLACK_CHANNEL_ID"'",
    "thread_ts": "'"$MAIN_TS"'",
    "text": "High-Value Tickets",
    "blocks": [{high_value_blocks}]
  }'
```

**Environment variables required:**
- `SLACK_BOT_TOKEN`: Bot token starting with `xoxb-`
- `SLACK_CHANNEL_ID`: Channel ID (e.g., `C0AD6C36LVC`)

### Step 7: Handle Escalations

If `critical_count > 5`, add an @mention block:

```json
{
  "type": "section",
  "text": {
    "type": "mrkdwn",
    "text": ":rotating_light: *ESCALATION:* {critical_count} critical orders (>36h stuck). <@ops-lead> please review."
  }
}
```

### Step 8: Confirm Success

After posting, output:

```
✅ Posted ECM triage to #ops-ecm
   • {new_to_assign} new assignments
   • {critical_count} critical flagged
   • Timestamp: {timestamp}
```

---

## Error Handling

If `SLACK_WEBHOOK_URL` is not set:
```
⚠️ SLACK_WEBHOOK_URL not configured. Triage completed but not posted.
Set environment variable to enable Slack notifications.
```

If HTTP POST fails:
```
❌ Failed to post to Slack: {error_message}
Triage data saved to Google Sheet. Manual Slack post may be needed.
```

---

## Example Full Message

### Main Message (Engaging, motivational)
```
🚀 ECM War Room — Let's Clear the Queue!

*{count} customers* are waiting for their money to move. You're the heroes who make it happen. 💪

━━━━━━━━━━━━━━━━━━━━━━━━

🔴 *P1 Critical:* {p1_count} orders (>36h stuck)
💰 *High Value:* {hv_count} orders (>5K each)
⏱️ *Oldest:* {max_hours}h waiting
💵 *Total Value:* ~{total_value} at stake

━━━━━━━━━━━━━━━━━━━━━━━━

*🎯 Your Mission Today:*

👤 *akshay* — {count} tickets ({hv} high-value)
👤 *vishnu.r* — {count} tickets ({hv} high-value)
👤 *abhijith.balan* — {count} tickets ({hv} high-value)
👤 *aakash.raj* — {count} tickets ({hv} high-value)
👤 *raj.kumar* — {count} tickets ({hv} high-value)

📋 Open Dashboard • 🧵 Check thread for your orders
```

### Thread 1: Per-Agent Orders (one thread per agent)
```
*@{agent_name} — Your High-Value Orders:*

`{order_id}` {amount} {currency} ({hours}h old)
`{order_id}` {amount} {currency}
...

_Total: {sum} — These customers are counting on you!_
```

### Thread 2: Getting Started Guide
```
🛠️ *Getting Started*

*Install Claude:*
<https://claude.ai/download|claude.ai/download>

*Commands:*
`/my-tickets` - See your queue
`order ORDER_ID` - Get runbook
`resolve ORDER_ID notes` - Mark done

*Links:*
<https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit|📊 ECM Dashboard>
<https://github.com/Vance-Club/ai-velocity|📖 Skill Docs>

*🏆 Goal: Clear 50% of P1s by EOD!*

_Questions? Reply in this thread._
```
