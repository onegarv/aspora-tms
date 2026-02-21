# Resolve Ticket Skill

## Trigger
- "resolve {order_id} {notes}"
- "fixed {order_id} {notes}"
- "done {order_id} {notes}"
- "close {order_id} {notes}"

## Google Sheet
**Spreadsheet ID:** `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`

## Description
Marks a ticket as resolved by updating **Google Sheets** (Assignments + Resolutions tabs). Redshift is read-only.

## Input
- `order_id` - The order to resolve
- `notes` - Resolution notes (what was done to fix it) — **REQUIRED**

## Data Flow

### Step 1: Find assignment in Sheet
Read **Assignments** tab:
- Find row where `Order ID = {order_id}` and `Status IN ('OPEN', 'IN_PROGRESS')`
- Get `Assigned At` timestamp

### Step 2: Get order details from Redshift (read-only)
```sql
SELECT order_id,
       meta_postscript_pricing_info_send_currency AS currency_from,
       meta_postscript_pricing_info_send_amount AS send_amount,
       status AS order_status
FROM orders_goms
WHERE order_id = '{order_id}'
```

> **DO NOT use `analytics_orders_master_data`** — it is a slow view. Use `orders_goms` directly.

### Step 3: Calculate metrics
- Resolution Time = Now - Assigned At
- SLA Target = from `../shared/config/diagnosis-mapping.yaml`
- SLA Status = MET if Resolution Time < SLA Target, else MISSED

### Step 4: Collect Sentinel Feedback (3 quick questions)

After confirming the resolution, ask the agent 3 quick feedback questions. These feed Sentinel's learning loop.

Present as single-selection options — agent picks one per question:

```
Quick feedback (helps Sentinel improve):

1. Was the diagnosis correct?
   [CORRECT] — the stuck_reason matched reality
   [PARTIAL] — partially right, but needed adjustment
   [WRONG] — completely different issue than diagnosed

2. Did you follow the prescribed action?
   [YES] — followed the action in Notes exactly
   [MODIFIED] — adapted the steps (explain in resolution notes)
   [IGNORED] — used a completely different approach

3. Resolution type?
   [AGENT_RESOLVED] — you fixed it manually
   [SELF_HEALED] — order resolved itself before you acted
   [ESCALATED_RESOLVED] — escalated, then resolved by L2/partner
   [FALSE_POSITIVE] — not actually stuck / no action needed
```

**Default values** (if agent skips or says "just resolve it"):
- Diagnosis Match: `CORRECT`
- Action Followed: `YES`
- Resolution Type: `AGENT_RESOLVED`

> ⚠️ Do NOT make this blocking. If agent just wants to resolve quickly, accept defaults. The feedback is valuable but should never slow down resolution.

### Step 5: Write to Resolutions tab
Append row to **Resolutions** tab with ALL 13 columns:
```
Timestamp | Order ID | Agent | Notes | Assigned At | Time (min) | SLA Target | SLA Status | Stuck Reason | Amount | Currency | Diagnosis Match | Action Followed | Resolution Type
```

Column details:
- `Stuck Reason`: Copy from Assignments tab Diagnosis column (col I)
- `Diagnosis Match`: From feedback Q1 — `CORRECT`, `PARTIAL`, or `WRONG`
- `Action Followed`: From feedback Q2 — `YES`, `MODIFIED`, or `IGNORED`
- `Resolution Type`: From feedback Q3 — `AGENT_RESOLVED`, `SELF_HEALED`, `ESCALATED_RESOLVED`, or `FALSE_POSITIVE`

### Step 6: Update Assignments tab
Update the row:
- `Status` = "RESOLVED"

### Step 7: Get remaining queue
Read **Assignments** tab for remaining OPEN/IN_PROGRESS tickets.

### Step 8: Calculate today's stats
Count from **Resolutions** tab where `Agent = {agent}` and `Timestamp = today`.

## Output Format

```
✅ Ticket Resolved: {order_id}

┌─────────────────────────────────────────┐
│ Metric          │ Value                 │
├─────────────────┼───────────────────────┤
│ Resolution Time │ 6 minutes             │
│ SLA Target      │ 2 hours               │
│ SLA Status      │ ✅ MET                │
└─────────────────────────────────────────┘

📝 Logged to ECM Operations Sheet:
   Resolution: "{notes}"
   Agent: @{agent}
   Time: {resolution_minutes} min
   Feedback: {diagnosis_match} | {action_followed} | {resolution_type}

---

🎫 @{agent}'s Queue: {remaining_count} remaining

📊 Your Stats Today:
┌─────────────────────────────────────────┐
│ Metric          │ Value                 │
├─────────────────┼───────────────────────┤
│ Resolved        │ {today_resolved}      │
│ Avg Time        │ {avg_resolution_time} │
│ SLA Met         │ {sla_met_percent}%    │
└─────────────────────────────────────────┘

Next: `order {next_urgent_order_id}` (SLA in {sla_remaining} ⚠️)
```

## Error Cases

### Order not found in Assignments
```
❌ Order {order_id} not found in your queue.

Check:
- Is this order assigned to you?
- Is it already resolved or escalated?

Run `my tickets` to see your current queue.
```

### Empty notes
```
❌ Resolution notes are required.

Example: resolve {order_id} "Replayed webhook, LULU confirmed"
```

## Guardrails
- Only resolve orders that exist in Assignments tab
- Notes are required (reject empty)
- Never write to Redshift (read-only)
- All writes go to Google Sheets
