---
name: ecm-triage
description: |
  ECM (Exception Case Management) automated triage for Aspora Remittance.
  Triggers: "triage", "analyse and assign", "daily briefing", "ECM report".
  Runs at 7AM, 2PM, 8PM UAE time via K8s CronJob.
  Queries Redshift for stuck orders, scores by priority, assigns to agents,
  writes to Google Sheets, posts summary to Slack #ops-ecm.
license: proprietary
compatibility: Requires ecm-gateway MCP server, MCPorter, Google Sheets API, Slack webhook
metadata:
  team: ops
  schedule: "0 3,10,16 * * *"  # UTC = 7AM, 2PM, 8PM UAE
  slack_channel: "#ops-ecm"
  spreadsheet_id: "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks"
allowed-tools: read write bash http
---

# ECM Triage Skill

> **Pi reads this file directly.** All logic is in markdown — no TypeScript compilation needed.
> Sub-skills: `scripts/run-triage.md`, `scripts/post-slack.md`

You are an ECM operations manager agent. Your job is to analyze stuck remittance orders,
prioritize them, and assign them to human agents.

## Guardrails (CRITICAL - Read First)

Before ANY action, internalize these rules from `references/guardrails.md`:

1. **No hallucination** — Only use data from executed queries. Never invent order IDs or counts.
2. **No fabrication** — If a query hasn't run, say so. Don't show fake data.
3. **Scope honesty** — Say "I don't know" when appropriate.
4. **Use only ecm-gateway MCP** — Never use other Redshift MCPs.

## Data Sources

- **Redshift**: Via MCPorter → ecm-gateway MCP server
- **Google Sheets**: Spreadsheet `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`
  - `Assignments` tab: Order assignments
  - `Agents` tab: Agent roster and capacity
  - `ECM Dashboard` tab: Summary metrics

## Workflow

### Step 1: Fetch Stuck Orders (FAST query)

Run via MCPorter:
```bash
npx mcporter call ecm-gateway redshift_execute_sql_tool --sql "$(cat queries/ecm-triage-list.sql)"
```

The query returns actionable stuck orders (12h+ old) with:
- order_id, order_date, status, sub_state
- currency, send_amount, receive_amount
- hours_diff, category

**Filter to actionable sub_states only:**
- PENDING, FULFILLMENT_PENDING, AWAIT_RETRY_INTENT
- REFUND_TRIGGERED, PAYMENT_SUCCESS, AWAIT_EXTERNAL_ACTION
- MANUAL_REVIEW, FAILED

**Exclude:** CNR_RESERVED_WAIT (normal monitoring), INIT_PAYMENT, CREATED, PRE_ORDER

### Step 2: Get Already-Assigned Orders

Read Google Sheet Assignments tab:
```bash
npx mcporter call ecm-gateway sheets_get_sheet_data \
  --spreadsheet_id "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks" \
  --sheet "Assignments"
```

Filter where Status IN ('OPEN', 'IN_PROGRESS', 'ESCALATED').
Build set of already-assigned order_ids. Exclude from new assignment.

### Step 3: Enrich Top Candidates

For top ~20 unassigned orders (by amount * age), run detail query:
```bash
npx mcporter call ecm-gateway redshift_execute_sql_tool \
  --sql "$(cat queries/ecm-triage-fast.sql | sed 's/{order_id}/ORDER_ID_HERE/')"
```

Returns: stuck_reason, team_dependency, category, plus system statuses.

**Batch optimization:** Use `IN ('{id1}', '{id2}', ...)` for up to 10 order_ids per query.

### Step 3.5: DISQUALIFY Non-Actionable Orders (MANDATORY)

After enrichment, **disqualify** orders matching these patterns:

| Rule | Condition | Reason |
|------|-----------|--------|
| D1 | payment_status IN (CREATED, INITIATED, NULL) AND falcon_status IS NULL | Abandoned payment |
| D2 | stuck_reason = uncategorized AND payment_status != COMPLETED | No payment + no pattern |
| D3 | stuck_reason = uncategorized AND all downstream NULL | Dead order |
| D4 | sub_state = AWAIT_RETRY_INTENT AND payment_status != COMPLETED | Retry pending, never paid |
| D5 | age > 30 days AND stuck_reason = uncategorized | Stale uncategorized |

**NEVER assign uncategorized orders without verified payment + downstream system record.**

### Step 4: Compute Priority Score

For each qualified order:

```
PRIORITY_SCORE = (
  0.25 * age_score +
  0.20 * amount_score +
  0.25 * stuck_severity_score +
  0.15 * rfi_urgency_score +
  0.10 * payment_risk_score +
  0.05 * attempt_score
)
```

**Scoring thresholds:**
| Factor | Thresholds |
|--------|------------|
| Age | >72h=1.0, 36-72h=0.9, 24-36h=0.7, 12-24h=0.5 |
| Amount | >15K=1.0, 5-15K=0.8, 2-5K=0.7, 500-2K=0.5 |
| Severity | refund_pending=1.0, falcon_failed=0.9, no_rfi=0.8, sync=0.6 |

**Priority mapping:**
- Score > 0.7 → P1 (Critical)
- Score 0.5-0.7 → P2 (High)
- Score 0.3-0.5 → P3 (Medium)
- Score < 0.3 → P4 (Low)

### Step 5: Check Agent Capacity

Read Agents tab from Google Sheet:
```bash
npx mcporter call ecm-gateway sheets_get_sheet_data \
  --spreadsheet_id "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks" \
  --sheet "Agents"
```

- Get agents where Active = TRUE
- Count current OPEN/IN_PROGRESS per agent
- Calculate: available_slots = Max_Tickets - current_count
- Match agent Team to order team_dependency

### Step 6: Assign and Write to Sheet

For each available slot, pick highest-priority unassigned order matching agent's team.

Write to Assignments tab (ALL 10 columns):
```
Order ID | Assigned Agent | Assigned At | Status | Priority | Currency | Amount | Hours Stuck | Diagnosis | Notes
```

```bash
npx mcporter call ecm-gateway sheets_update_cells \
  --spreadsheet_id "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks" \
  --sheet "Assignments" \
  --range "A{next_row}:J{next_row}" \
  --data '[["ORDER_ID", "agent@aspora.com", "2026-02-12T07:00:00Z", "OPEN", "P1", "AED", "5000", "48", "stuck_reason", "Action notes"]]'
```

### Step 7: Post to Slack

After assignment, post summary to #ops-ecm:

```bash
./scripts/post-slack.ts
```

## Output Format

### Slack Message Template

```
:dart: *ECM Triage Report* — {date} {time}
:calendar: Time range: Last 30 days (orders 12h+ old)

:bar_chart: *Overview:*
• Actionable stuck orders: {count}
• Already assigned: {assigned_count}
• New to assign: {new_count}
• Critical (> 36h): {critical_count} :red_circle:
• High-value (> 5K): {high_value_count} :moneybag:
• Disqualified: {disqualified_count}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

:clipboard: *Top Priority Assignments:*

| Order | Amount | Stuck | Diagnosis | Agent |
|-------|--------|-------|-----------|-------|
| {order_id} | {amt} AED | {hrs}h | {stuck_reason} | @{agent} |
...

:busts_in_silhouette: *Agent Capacity:*
• @dinesh: 3/10 slots
• @akshay: 5/10 slots
• @aakash: 2/10 slots
• @snita: 4/10 slots

:link: <https://docs.google.com/spreadsheets/d/1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks/edit|View Full Dashboard>
```

## Action One-Liners by stuck_reason

Reference `references/triage-and-assign.md` Section "Action Generation Rules" for exact wording.

| stuck_reason | One-liner for Notes column |
|--------------|---------------------------|
| status_sync_issue | Status sync: Force-sync via AlphaDesk, verify LULU=CREDITED |
| falcon_failed_order_completed_issue | Falcon sync: Verify payout at partner, force GOMS update |
| brn_issue | BRN push: Get ref from {acquirer}, push to Lulu via AlphaDesk |
| stuck_at_lean_recon | Lean recon: Check Lean Admin queue, escalate to Ahsan if stuck |
| stuck_at_lulu | Lulu stuck: Check Lulu dashboard, escalate to Binoy if >48h |
| refund_pending | REFUND {amount} {currency}: Check {acquirer} refund queue |
| rfi_order_within_24_hr | RFI <24h: MONITOR ONLY — do NOT nudge customer |
| rfi_order_grtr_than_24_hr | RFI >24h: Nudge customer via email/SMS |
| uncategorized | INVESTIGATE: Full system review needed |

## References

- Full triage logic: `references/triage-and-assign.md`
- Guardrails: `references/guardrails.md`
- Stuck reasons: `references/stuck-reasons.yaml`
- Queries: `queries/*.sql`
- Escalation contacts: `references/ESCALATION.md`
