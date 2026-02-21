# My Tickets Skill (Agent Persona)

## CRITICAL: MCP Configuration

**ONLY use the `ecm-gateway` MCP server for all queries.**

| Tool Name | Purpose |
|-----------|---------|
| `mcp__ecm-gateway__redshift_execute_sql_tool` | Execute Redshift SQL |
| `mcp__ecm-gateway__sheets_get_sheet_data` | Read Google Sheets |
| `mcp__ecm-gateway__sheets_update_cells` | Update Google Sheets |

**DO NOT use:**
- `awslabs.redshift-mcp-server` — NOT configured
- Any standalone Redshift MCP — NOT available
- Hallucinated data — NEVER make up results

---

## Trigger
- "my tickets"
- "my queue"
- "show my tickets"
- "what should I work on"
- "next task"

## Persona: Agent
This skill is for the **agent persona** (individual contributor).
Shows their assigned tickets with **clear actionables** — not just data.

For the **manager persona** (triage & assign), see `../../manager/skills/triage-and-assign.md`.

---

## Google Sheet
**Spreadsheet ID:** `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`

### Assignments Tab Schema (10 columns)
```
A: Order ID | B: Assigned Agent | C: Assigned At | D: Status | E: Priority
F: Currency | G: Amount         | H: Hours Stuck | I: Diagnosis | J: Notes
```

---

## Description
Shows the agent's assigned ticket queue from Google Sheets, enriched with live Redshift data and **specific actionable instructions** per ticket.

The agent sees:
1. Their tickets sorted by priority (P1 first)
2. Live hours_stuck from Redshift (not stale Sheet value)
3. The **Diagnosis** (stuck_reason) from the Sheet
4. The **Notes** column with the one-liner action from the manager
5. A full action block per ticket from `triage-and-assign.md` Action Generation Rules

---

## Data Flow

### Step 1: Get agent identity
Get agent email from:
1. Ask once: "What's your email?" → cache for session
2. Or use Claude's user context if available

### Step 2: Get assignments from Google Sheet
Read **Assignments** tab from spreadsheet `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`:
- Filter: `Assigned Agent = {agent_email}` AND `Status IN ('OPEN', 'IN_PROGRESS')`
- Extract: Order ID (col A), Priority (col E), Currency (col F), Amount (col G), Diagnosis (col I), Notes (col J)

### Step 3: Refresh order data from Redshift
For each order_id from Sheet, get **live** status:
```sql
SELECT
    order_id, status AS order_status, sub_state,
    meta_postscript_pricing_info_send_currency AS currency_from,
    meta_postscript_pricing_info_send_amount AS send_amount,
    ROUND(EXTRACT(EPOCH FROM (GETDATE() - created_at)) / 3600, 1) AS hours_diff
FROM orders_goms
WHERE order_id IN ({order_ids_from_sheet})
```

> **DO NOT use `analytics_orders_master_data`** — it is a slow view. Use `orders_goms` directly.

**Auto-resolve check:** If Redshift shows `status = 'COMPLETED'` for any assigned order, flag it as resolved:
- Tell the agent: "Order {id} has been completed — you can resolve it"
- Do NOT auto-update the Sheet (agent should confirm with resolution notes)

### Step 4: Enrich with stuck_reason (for tickets without Diagnosis)
If the Diagnosis column (I) is empty, run `../shared/queries/ecm-triage-fast.sql` per order to get `stuck_reason`.

### Step 5: Generate actionables
For each ticket, look up the `Diagnosis` value (stuck_reason) in the Action Generation Rules from `../../manager/skills/triage-and-assign.md`.

Present the **full action block** (not just the one-liner from Notes).

### Step 6: Calculate SLA
For each order:
- SLA hours from `../shared/config/diagnosis-mapping.yaml` based on Diagnosis
- SLA remaining = SLA hours - live hours_diff from Redshift
- SLA status:
  - `🔴 BREACHED` — past deadline
  - `⚠️ CRITICAL` — < 25% remaining
  - `🟡 WARNING` — 25-50% remaining
  - `🟢 OK` — > 50% remaining

### Step 7: Load agent accuracy from Sentinel Learnings (if available)

Check if `sentinel/learnings.yaml` exists. If it does, load the `agent_performance` section for this agent's email.

Extract:
- `resolved` — total resolved (last 30 days)
- `accuracy` — % where Diagnosis Match = CORRECT
- `avg_resolution_min` — average resolution time
- `sla_met_rate` — % of tickets resolved within SLA

If learnings file doesn't exist or has no data for this agent, skip — show "No Sentinel data yet" in stats.

### Step 8: Sort and display
Sort by:
1. Priority (P1 first)
2. SLA breached first
3. SLA remaining ascending

---

## Output Format

### Agent Queue with Actionables

```
🎫 @{agent}'s Queue — {count} tickets

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 🔴 P1 | {order_id} | {send_amount} {currency} | {hours_stuck}h | SLA: BREACHED
   Diagnosis: {diagnosis_display}
   ┌─────────────────────────────────────────────┐
   │ WHAT TO DO:                                 │
   │ {specific_action_from_stuck_reason}         │
   │                                             │
   │ CHECK: {what_to_verify}                     │
   │ DONE? → resolve {order_id} "{expected_notes}"│
   │ STUCK? → stuck {order_id} "{escalation_hint}"│
   └─────────────────────────────────────────────┘

2. 🟠 P2 | {order_id} | {send_amount} {currency} | {hours_stuck}h | SLA: ⚠️ 2h
   Diagnosis: {diagnosis_display}
   → {one_line_action_from_notes}
   → Type `order {order_id}` for full details

3. 🟡 P3 | {order_id} | {send_amount} {currency} | {hours_stuck}h | SLA: 🟢 8h
   Diagnosis: {diagnosis_display}
   → {one_line_action_from_notes}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ Start with #1 — {brief_reason_why_most_urgent}

📊 Your Stats Today:
│ Queue: {current} / {max} tickets
│ Resolved today: {today_count} (from Resolutions tab)
│ Avg resolution: {avg_time} min
│ SLA met: {sla_pct}%

🎯 Sentinel Accuracy (last 30 days):
│ Diagnosis accuracy: {accuracy}%
│ Total resolved: {resolved_30d}
│ Avg resolution: {avg_resolution_min} min
│ SLA met rate: {sla_met_rate}%
│ (from sentinel/learnings.yaml — omit if no data yet)
```

**Key UX decisions:**
- **P1 tickets** get the full action block with CHECK/DONE/STUCK
- **P2-P4 tickets** get the one-liner from Notes + hint to drill down
- **Auto-resolved orders** are flagged at the top

### Completed Orders Detected

If any assigned orders are now COMPLETED in Redshift:

```
✅ These orders have been completed since last check:

  {order_id} — now COMPLETED in Redshift
  → Run: resolve {order_id} "Completed — auto-detected"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Empty Queue

```
🎫 @{agent}'s Queue — 0 tickets

✨ All clear! No tickets assigned to you.

📊 Today: {resolved_today} resolved | {avg_time} avg | {sla_pct}% SLA met
🎯 30-day accuracy: {accuracy}% | {resolved_30d} resolved

Ask your manager to run `triage and assign` for new tickets.
```

---

## SLA Configuration (from diagnosis-mapping.yaml)

| Diagnosis | SLA (hours) |
|-----------|------------|
| PAYMENT_FAILED | 2 |
| CNR_RESERVED_WAIT | 4 |
| BRN_PENDING | 4 |
| STATUS_SYNC_ISSUE | 1 |
| RFI_PENDING | 24 |
| DOCS_REQUIRED | 24 |
| DEFAULT | 8 |

---

## Guardrails
- Only show tickets from Google Sheet (Assignments tab), spreadsheet `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`
- Order data from Redshift (read-only via `orders_goms`)
- Do not invent tickets, SLA times, or resolution steps
- Action text from `triage-and-assign.md` Action Generation Rules only
- If Diagnosis is empty or `uncategorized`, say so — do not guess
- RFI < 24h: NEVER suggest nudging the customer
- Do NOT auto-resolve tickets in Sheet — agent must confirm with `resolve {id} "{notes}"`
