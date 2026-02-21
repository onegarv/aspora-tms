# Run Triage Script

This is a sub-skill executed by the ECM Triage skill. Pi reads this and executes each step.

## Prerequisites

- MCPorter configured with `ecm-gateway` MCP server
- Environment variable: `SPREADSHEET_ID=1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`

## Execution Steps

### Step 1: Fetch Stuck Orders

Execute this SQL via MCPorter:

```bash
npx mcporter call ecm-gateway redshift_execute_sql_tool --sql "
SELECT
    og.order_id,
    og.created_at::date AS order_date,
    og.status AS goms_order_status,
    og.sub_state,
    og.meta_postscript_pricing_info_send_currency AS currency,
    og.meta_postscript_pricing_info_send_amount AS send_amount,
    og.meta_postscript_pricing_info_receive_amount AS receive_amount,
    ROUND(EXTRACT(EPOCH FROM (GETDATE() - og.created_at)) / 3600, 1) AS hours_diff
FROM orders_goms og
WHERE og.status IN ('PROCESSING_DEAL_IN', 'PENDING', 'FAILED')
  AND og.meta_postscript_pricing_info_send_currency IN ('AED', 'GBP', 'EUR')
  AND og.created_at >= CURRENT_DATE - 30
  AND og.created_at < GETDATE() - INTERVAL '12 hours'
  AND og.sub_state IN (
      'PENDING', 'FULFILLMENT_PENDING', 'AWAIT_RETRY_INTENT',
      'REFUND_TRIGGERED', 'PAYMENT_SUCCESS', 'AWAIT_EXTERNAL_ACTION',
      'MANUAL_REVIEW', 'FAILED'
  )
ORDER BY og.meta_postscript_pricing_info_send_amount DESC, og.created_at ASC
LIMIT 100
"
```

**Store result as:** `stuck_orders`

---

### Step 2: Fetch Already-Assigned Orders

```bash
npx mcporter call ecm-gateway sheets_get_sheet_data \
  --spreadsheet_id "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks" \
  --sheet "Assignments"
```

**Parse result:**
- Skip header row (row 0)
- Extract `order_id` from column A where column D (Status) is `OPEN`, `IN_PROGRESS`, or `ESCALATED`
- Store as set: `assigned_order_ids`

**Filter stuck_orders:**
```
unassigned_orders = stuck_orders.filter(o => !assigned_order_ids.has(o.order_id))
```

---

### Step 3: Enrich Top 20 Orders

For the top 20 unassigned orders by `send_amount * hours_diff`, run detail query:

```bash
npx mcporter call ecm-gateway redshift_execute_sql_tool --sql "
SELECT
    og.order_id,
    -- ... (use queries/ecm-triage-fast.sql content)
FROM orders_goms og
WHERE og.order_id IN ('{order_id_1}', '{order_id_2}', ...)
"
```

**Batch:** Query up to 10 order_ids at a time (MCP timeout = 30s).

**Store enriched data as:** `enriched_orders` with `stuck_reason`, `team_dependency`

---

### Step 3.5: Disqualify Non-Actionable Orders

For each enriched order, check disqualification rules:

| Rule | Check | Action |
|------|-------|--------|
| D1 | `payment_status IN (CREATED, INITIATED, NULL) AND falcon_status IS NULL` | SKIP — abandoned payment |
| D2 | `stuck_reason = 'uncategorized' AND payment_status != 'COMPLETED'` | SKIP — no payment |
| D3 | `stuck_reason = 'uncategorized' AND falcon/lulu/payout all NULL` | SKIP — dead order |
| D4 | `sub_state = 'AWAIT_RETRY_INTENT' AND payment_status != 'COMPLETED'` | SKIP — never paid |
| D5 | `hours_diff > 720 AND stuck_reason = 'uncategorized'` | SKIP — stale (>30 days) |

**Store:** `qualified_orders` (those passing all checks)
**Store:** `disqualified_count`

---

### Step 4: Compute Priority Scores

For each qualified order, compute:

```
priority_score = (
  0.25 * age_score(hours_diff) +
  0.20 * amount_score(send_amount) +
  0.25 * severity_score(stuck_reason) +
  0.15 * 0.5 +  // RFI placeholder
  0.10 * 0.5 +  // Payment risk placeholder
  0.05 * 0.3    // Attempt placeholder
)
```

**Age score:**
| hours_diff | score |
|------------|-------|
| > 72 | 1.0 |
| 36-72 | 0.9 |
| 24-36 | 0.7 |
| 12-24 | 0.5 |

**Amount score:**
| send_amount | score |
|-------------|-------|
| > 15000 | 1.0 |
| 5000-15000 | 0.8 |
| 2000-5000 | 0.7 |
| 500-2000 | 0.5 |
| < 500 | 0.3 |

**Severity score:**
| stuck_reason | score |
|--------------|-------|
| refund_pending | 1.0 |
| falcon_failed_* | 0.9 |
| no_rfi_created | 0.8 |
| *sync* | 0.6 |
| other | 0.5 |

**Priority level:**
| score | level |
|-------|-------|
| > 0.7 | P1 |
| 0.5-0.7 | P2 |
| 0.3-0.5 | P3 |
| < 0.3 | P4 |

**Sort:** `qualified_orders` by `priority_score` descending

---

### Step 5: Fetch Agent Capacity

```bash
npx mcporter call ecm-gateway sheets_get_sheet_data \
  --spreadsheet_id "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks" \
  --sheet "Agents"
```

**Parse:**
- Column A: email
- Column B: name
- Column C: team (Ops, KYC_ops, VDA_ops)
- Column D: slack_handle
- Column E: active (TRUE/FALSE)
- Column F: max_tickets

**Filter:** Only `active = TRUE`

**Calculate current load:** Count assignments from Step 2 where `assigned_agent = email` and `status IN (OPEN, IN_PROGRESS)`

**Store:** `agents` with `available_slots = max_tickets - current_count`

---

### Step 6: Assign Orders to Agents (Value-Weighted Distribution)

**IMPORTANT:** High-value tickets must be distributed evenly across all agents.

#### Step 6a: Separate by Value

```
high_value_orders = qualified_orders.filter(o => o.send_amount >= 5000)
regular_orders = qualified_orders.filter(o => o.send_amount < 5000)
```

#### Step 6b: Assign High-Value Orders (Round-Robin)

High-value orders are assigned round-robin to ensure even distribution:

```
active_agents = agents.filter(a => a.active === true)
agent_index = 0

for each order in high_value_orders (sorted by amount DESC):
    agent = active_agents[agent_index % active_agents.length]
    assign(order, agent)
    agent_index++
```

This ensures each agent gets roughly equal high-value exposure.

#### Step 6c: Assign Regular Orders (Round-Robin)

Continue round-robin for remaining orders:

```
for each order in regular_orders (sorted by priority_score DESC):
    agent = active_agents[agent_index % active_agents.length]
    assign(order, agent)
    agent_index++
```

**Store:** `new_assignments[]`

**Note:** No capacity limits - all actionable orders are assigned. AI handles the workload distribution.

---

### Step 7: Write Assignments to Sheet

For each new assignment, append row to Assignments tab:

```bash
npx mcporter call ecm-gateway sheets_update_cells \
  --spreadsheet_id "1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks" \
  --sheet "Assignments" \
  --range "A{next_row}:J{next_row}" \
  --data '[
    [
      "{order_id}",
      "{agent_email}",
      "{timestamp_iso}",
      "OPEN",
      "{priority_level}",
      "{currency}",
      "{send_amount}",
      "{hours_diff}",
      "{stuck_reason}",
      "{action_note}"
    ]
  ]'
```

**Action notes by stuck_reason:**

| stuck_reason | action_note |
|--------------|-------------|
| status_sync_issue | Status sync: Force-sync via AlphaDesk, verify LULU=CREDITED |
| falcon_failed_order_completed_issue | Falcon sync: Verify payout at partner, force GOMS update |
| brn_issue | BRN push: Get ref from acquirer, push to Lulu via AlphaDesk |
| stuck_at_lean_recon | Lean recon: Check Lean Admin queue, escalate to Ahsan if stuck |
| stuck_at_lulu | Lulu stuck: Check Lulu dashboard, escalate to Binoy if >48h |
| refund_pending | REFUND {amount} {currency}: Check acquirer refund queue |
| rfi_order_within_24_hr | RFI <24h: MONITOR ONLY — do NOT nudge customer |
| rfi_order_grtr_than_24_hr | RFI >24h: Nudge customer via email/SMS |
| uncategorized | INVESTIGATE: Full system review needed |

---

### Step 8: Build Summary

Compile triage result:

```json
{
  "timestamp": "{current_iso_timestamp}",
  "summary": {
    "total_stuck": {stuck_orders.length},
    "already_assigned": {assigned_order_ids.size},
    "new_to_assign": {new_assignments.length},
    "critical_count": {orders where hours_diff > 36},
    "high_value_count": {orders where send_amount > 5000},
    "disqualified_count": {disqualified_count}
  },
  "assignments": {new_assignments},
  "agent_capacity": {agents}
}
```

**Output:** Pass this JSON to `scripts/post-slack.md`
