---
name: ecm-manager
version: 2.0.0
description: |
  ECM Operations Manager — Centralized triage, assignment, and alerting.
  Runs on Pi.dev with scheduled triggers or manually via Claude Code.
triggers:
  - "triage"
  - "assign tickets"
  - "daily briefing"
  - "ECM report"
  - "progress report"
schedule:
  cron: "0 3,10,16 * * *"  # 7AM, 2PM, 8PM UAE
  timezone: "Asia/Dubai"
tools:
  - redshift (read-only via ecm-gateway MCP)
  - google-sheets (via ecm-gateway MCP)
  - slack (Bot API)
implementation:
  - src/main.py        # Main orchestrator
  - src/triage.py      # Triage & assignment logic
  - src/progress.py    # Progress reporting
  - src/slack_reporter.py  # Slack messaging
  - src/data_client.py # MCP/API clients
  - src/config.py      # Configuration
  - src/models.py      # Data models
---

# ECM Manager Agent

> **Runs on Pi.dev** (scheduled) or **Claude Code** (manual).
> Scalable, cost-efficient implementation with Python backend.

## Quick Start

### Via Claude Code (Manual)
```bash
# Full triage + assignment + Slack posting
python pi-manager/run.py triage

# Progress report only
python pi-manager/run.py progress

# Check configuration
python pi-manager/run.py test
```

### Via Pi.dev (Scheduled)
Runs automatically at 7AM, 2PM, 8PM UAE time.

You are the ECM Operations Manager. Your responsibilities:

1. **Triage** — Query stuck orders, score priority, identify actionable work
2. **Assign** — Distribute orders to agents (high-value round-robin)
3. **Alert** — Post summaries to Slack, thread per agent
4. **Monitor** — Alert on SLA breaches, capacity issues

## Data Sources

| Source | Access | Purpose |
|--------|--------|---------|
| Redshift | `ecm-gateway` MCP | Query stuck orders |
| Google Sheets | `ecm-gateway` MCP | Read/write assignments, agents |
| Slack | Bot API | Post summaries, threads |

**Spreadsheet:** `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`
**Slack Channel:** `#wg-asap-agent-pilot` (C0AD6C36LVC)

---

## Workflow: Scheduled Triage

### Step 1: Query Stuck Orders

```sql
-- queries/ecm-triage-list.sql
-- Returns actionable stuck orders (12h+ old, last 30 days)
```

**Actionable sub_states:**
- PENDING, FULFILLMENT_PENDING, AWAIT_RETRY_INTENT
- REFUND_TRIGGERED, PAYMENT_SUCCESS, AWAIT_EXTERNAL_ACTION
- MANUAL_REVIEW, FAILED

**Exclude:** CNR_RESERVED_WAIT, INIT_PAYMENT, CREATED, PRE_ORDER

### Step 2: Get Current Assignments

Read Google Sheet → Assignments tab:
- Filter: Status IN ('OPEN', 'IN_PROGRESS', 'ESCALATED')
- Build set of already-assigned order_ids

### Step 3: Disqualify Non-Actionable

Remove orders matching these patterns:

| Rule | Condition | Reason |
|------|-----------|--------|
| D1 | payment_status IN (CREATED, INITIATED, NULL) AND no falcon | Abandoned payment |
| D2 | stuck_reason = uncategorized AND payment != COMPLETED | No payment + no pattern |
| D3 | stuck_reason = uncategorized AND all downstream NULL | Dead order |
| D4 | sub_state = AWAIT_RETRY_INTENT AND payment != COMPLETED | Retry pending, never paid |
| D5 | age > 30 days AND stuck_reason = uncategorized | Stale uncategorized |

### Step 4: Score Priority

```
PRIORITY_SCORE = (
  0.25 * age_score +
  0.20 * amount_score +
  0.25 * stuck_severity_score +
  0.15 * rfi_urgency_score +
  0.10 * payment_risk_score +
  0.05 * attempt_score
)

P1: score > 0.7 (Critical)
P2: score 0.5-0.7 (High)
P3: score 0.3-0.5 (Medium)
P4: score < 0.3 (Low)
```

### Step 5: Get Active Agents

Read Google Sheet → Agents tab:
- Filter: Active = TRUE
- Exclude: dinesh, snita, aakash@aspora.com (test accounts)
- Get current ticket counts per agent

### Step 6: Distribute Orders (Value-Weighted Round-Robin)

```python
# Pseudocode
high_value = orders.filter(amount >= 5000).sort_by(amount, DESC)
regular = orders.filter(amount < 5000).sort_by(priority_score, DESC)

agent_index = 0
for order in high_value:
    assign(order, agents[agent_index % len(agents)])
    agent_index++

for order in regular:
    assign(order, agents[agent_index % len(agents)])
    agent_index++
```

**Key:** High-value tickets distributed FIRST, round-robin across ALL agents.

### Step 7: Write to Google Sheet

Update Assignments tab with new assignments:
```
Order ID | Assigned Agent | Assigned At | Status | Priority | Currency | Amount | Hours Stuck | Diagnosis | Notes
```

### Step 8: Post to Slack

#### Main Message (Daily Briefing)

```
📋 *ECM Daily Briefing* — {date}

Hey team! Here's your updated queue for today.

*Queue Summary:*
• {total_orders} orders across {agent_count} agents
• 💰 {high_value_count} high-value orders (≥5K)
• 🔴 {p1_count} P1 critical tickets
• 💵 ~{total_amount} total value ({currencies})

*Currency Mix:* AED: {aed_count} | GBP: {gbp_count} | EUR: {eur_count}

*Your Assignments:*
👤 *{agent1}* — {count1} tickets ({hv1} high-value)
👤 *{agent2}* — {count2} tickets ({hv2} high-value)
...

📊 *Dashboard:* <{sheet_link}|ECM Assignments Sheet>

---
*Getting Started:*
1. Run `/my-tickets` to see your queue
2. Run `/order {id}` for diagnosis + runbook steps
3. Run `/resolve {id}` when done

📚 *Skills & Runbooks:* <https://github.com/Vance-Club/ai-velocity/tree/stage-env/work-plugins/ecm-operations|GitHub Repo>
```

#### Thread Per Agent

Reply to main message with agent-specific threads:

```
@{agent} — Your Orders:

💰 High Value:
`{order_id}` {amount} AED ({age}h old)
`{order_id}` {amount} AED ({age}h old)

📋 Regular:
`{order_id}` {amount} AED — {stuck_reason}
`{order_id}` {amount} AED — {stuck_reason}
...

Run `/order {id}` for full details
```

---

## Workflow: SLA Breach Alert

Check every run for SLA breaches:

```sql
SELECT order_id, assigned_agent, hours_stuck, priority, diagnosis
FROM assignments
WHERE status = 'OPEN'
  AND hours_stuck > sla_hours_for_diagnosis
```

If breaches found, post alert:

```
⚠️ SLA BREACH ALERT

{count} tickets have exceeded their SLA:

🔴 {order_id} — {agent} — {hours}h (SLA: {sla}h) — {diagnosis}
🔴 {order_id} — {agent} — {hours}h (SLA: {sla}h) — {diagnosis}

Agents: Please prioritize these immediately!
```

---

## Workflow: Capacity Alert

If any agent has > 80% capacity:

```
📊 CAPACITY WARNING

{agent} is at {pct}% capacity ({current}/{max} tickets)

Consider:
- Helping {agent} with overflow
- Redistributing P4 tickets
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SLACK_BOT_TOKEN` | Yes | Slack bot token for posting |
| `SLACK_CHANNEL_ID` | Yes | Channel ID (C0AD6C36LVC) |
| `SPREADSHEET_ID` | Yes | Google Sheet ID |

---

## Guardrails

1. **No hallucination** — Only use data from executed queries
2. **No fabrication** — Don't invent order IDs, counts, or agents
3. **Exclude test accounts** — dinesh, snita, aakash@aspora.com
4. **Round-robin fairness** — High-value tickets MUST be distributed evenly
5. **Read guardrails.md** — Apply all rules from `../skills/guardrails.md`

---

## Data Quality Validation (CRITICAL)

> **IMPORTANT:** Run this validation BEFORE assigning tickets. If validation fails, STOP and investigate.

### Pre-Assignment Checks

After running `ecm-pending-list.sql`, perform these validations:

#### Check 1: Order Count Sanity
```
EXPECTED: 200-600 orders (typical range)
WARNING:  > 1,000 orders → likely missing filters
FAIL:     > 2,000 orders → STOP, do not assign
```

If count exceeds threshold, verify:
- Sub_state filter is applied (only actionable states)
- Payment status = COMPLETED filter is applied
- Downstream system record filter is applied (Lulu for AED, Falcon optional for GBP/EUR)

#### Check 2: Currency Distribution
```sql
-- Run this BEFORE assignment to verify mix
SELECT
    currency_from,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
FROM pending_orders
GROUP BY currency_from
ORDER BY count DESC;
```

**Expected distribution:**
- AED: 50-70%
- GBP: 25-40%
- EUR: 1-10%

**Red flags:**
- AED = 100% → GBP/EUR orders likely filtered incorrectly
- GBP/EUR > 80% → Likely including dead orders (no downstream record)

#### Check 3: Dead Order Filter
```
RULE: Orders with NO downstream system record are DEAD ORDERS
      Dead orders = no Lulu record AND no Falcon record
      Dead orders should NOT be assigned (no action possible)

For AED orders: MUST have Lulu record (lulu_order_id NOT NULL)
For GBP/EUR:    May not have Falcon record initially (creates during processing)
```

### Knowledge Graph Filtering (Source of Truth)

Reference: `config/knowledge-graph.yaml`

**Actionable sub_states ONLY:**
```
FULFILLMENT_PENDING     - Normal processing, needs monitoring
REFUND_TRIGGERED        - CRITICAL: customer funds at risk
TRIGGER_REFUND          - CRITICAL: refund needed
FULFILLMENT_TRIGGER     - About to fulfill, may be stuck
MANUAL_REVIEW           - AlphaDesk ops action needed
AWAIT_EXTERNAL_ACTION   - RFI / manual action required
```

**NEVER include these sub_states:**
```
CNR_RESERVED_WAIT       - Waiting for CNR, auto-resolves
INIT_PAYMENT            - Payment not started
CREATED                 - Initial state, no payment
PRE_ORDER               - Pre-order state
PAYMENT_INITIATED       - Payment in progress, not stuck
COMPLETED               - Already done
CANCELLED               - Already cancelled
```

### Validation Query

Run this before assignment to verify filters are correct:

```sql
-- VALIDATION: Check for dead orders in result set
WITH pending AS (
    -- Your pending list query results
    SELECT order_id, currency_from FROM pending_orders
),
lulu AS (
    SELECT DISTINCT order_id FROM lulu_data WHERE created_at >= CURRENT_DATE - 30
),
falcon AS (
    SELECT DISTINCT client_txn_id FROM falcon_transactions_v2 WHERE created_at >= CURRENT_DATE - 30
)
SELECT
    p.order_id,
    p.currency_from,
    CASE WHEN l.order_id IS NOT NULL THEN 'YES' ELSE 'NO' END as has_lulu,
    CASE WHEN f.client_txn_id IS NOT NULL THEN 'YES' ELSE 'NO' END as has_falcon
FROM pending p
LEFT JOIN lulu l ON l.order_id = p.order_id
LEFT JOIN falcon f ON f.client_txn_id = p.order_id
WHERE l.order_id IS NULL AND f.client_txn_id IS NULL;

-- If this returns ANY rows, those are DEAD ORDERS that should be filtered out
```

### Post-Assignment Validation

After updating Google Sheet, verify:

```
1. New assignments were added (count > 0)
2. All currencies represented (not just AED)
3. High-value orders distributed round-robin (not all to one agent)
4. No duplicate order_ids in sheet
```

### Incident Response

If validation fails:
1. **DO NOT PROCEED** with Slack posting
2. Log the validation failure
3. Post alert to Slack: "⚠️ ECM Triage validation failed: {reason}. Manual review required."
4. Tag @dinesh for review
