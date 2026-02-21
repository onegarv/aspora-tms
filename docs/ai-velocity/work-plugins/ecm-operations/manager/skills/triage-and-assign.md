# Triage & Assign Skill (Manager Persona — Ops Agent Clone)

## Guardrails
- **Always apply** the rules in `../shared/guardrails.md`. Use only data from executed queries and runbooks; never invent order IDs, counts, or resolution steps.

## Trigger
- "triage"
- "triage and assign"
- "analyse and assign"
- "analyze and assign"
- "prioritize tickets"
- "daily briefing"

## Persona: Manager
This skill is for the **manager persona** who coordinates the entire ECM operation.
The manager analyses all stuck orders, scores them by priority, and assigns them to agents.

For the **agent persona** (individual contributor), see `../../field/skills/my-tickets.md`.

---

## Google Sheet
**Spreadsheet ID:** `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`

### Assignments Tab Schema (10 columns)
```
A: Order ID      | B: Assigned Agent | C: Assigned At | D: Status | E: Priority
F: Currency      | G: Amount         | H: Hours Stuck | I: Diagnosis | J: Notes
```
Status values: `OPEN`, `IN_PROGRESS`, `RESOLVED`, `ESCALATED`

### Agents Tab Schema (6 columns)
```
A: Email | B: Name | C: Team | D: Slack Handle | E: Active | F: Max Tickets
```
Teams: `Ops`, `KYC_ops`, `VDA_ops`

---

## Philosophy
> "Agents should never scout for data or problems. The manager-agent analyses, prioritizes, and assigns. Human agents execute with clear instructions."

---

## Data Flow

### Step 1: Get actionable stuck orders (FAST — < 5 seconds)

The fast list returns ~200K+ orders including CNR_RESERVED_WAIT and simple FAILED.
**Filter to actionable sub_states only:**

**Time range is configurable:**
- Default: `{time_range_days}` from `plugin.yaml` config (default **30 days**)
- Override: Manager can say `triage last 14 days` or `triage last 60 days`
- Minimum age filter: **12 hours** (orders younger than 12h are still in normal processing)

```sql
SELECT
    og.order_id,
    og.created_at::date AS order_date,
    og.status AS goms_order_status,
    og.sub_state,
    og.meta_postscript_pricing_info_send_currency AS currency_from,
    og.meta_postscript_pricing_info_send_amount AS send_amount,
    og.meta_postscript_pricing_info_receive_amount AS receive_amount,
    og.meta_postscript_acquirer AS payment_acquirer,
    ROUND(EXTRACT(EPOCH FROM (GETDATE() - og.created_at)) / 3600, 1) AS hours_diff
FROM orders_goms og
WHERE og.status IN ('PROCESSING_DEAL_IN', 'PENDING', 'FAILED')
  AND og.meta_postscript_pricing_info_send_currency IN ('AED', 'GBP', 'EUR')
  AND og.created_at >= CURRENT_DATE - {time_range_days}   -- DEFAULT 30, configurable
  AND og.created_at < GETDATE() - INTERVAL '12 hours'
  AND og.sub_state IN (
      'PENDING', 'FULFILLMENT_PENDING', 'AWAIT_RETRY_INTENT',
      'REFUND_TRIGGERED', 'PAYMENT_SUCCESS', 'AWAIT_EXTERNAL_ACTION',
      'MANUAL_REVIEW', 'FAILED'
  )
ORDER BY
    CASE og.sub_state
        WHEN 'REFUND_TRIGGERED' THEN 1
        WHEN 'AWAIT_RETRY_INTENT' THEN 2
        WHEN 'MANUAL_REVIEW' THEN 3
        WHEN 'AWAIT_EXTERNAL_ACTION' THEN 4
        WHEN 'FULFILLMENT_PENDING' THEN 5
        WHEN 'PENDING' THEN 6
        WHEN 'FAILED' THEN 7
        ELSE 8
    END,
    og.meta_postscript_pricing_info_send_amount DESC,
    og.created_at ASC
LIMIT 100;
```

> **Exclude** `CNR_RESERVED_WAIT` (normal 10-day monitoring) and `INIT_PAYMENT` / `CREATED` / `PRE_ORDER` (transient states).
> **Exclude** `FAILED/FAILED` unless enrichment reveals they need refunds — these are mostly declined payments that need no action.

**Time range override examples:**
- `triage` → uses default (30 days)
- `triage last 7 days` → only recent orders
- `triage last 60 days` → catch very old stuck orders
- `triage all` → no time filter (use with caution — slow on large datasets)

### Step 2: Get already-assigned orders from Google Sheet

Read **Assignments** tab from spreadsheet `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`:
- Filter where `Status IN ('OPEN', 'IN_PROGRESS', 'ESCALATED')`
- Build a set of already-assigned `order_id`s
- Exclude these from new assignment

### Step 3: Enrich top candidates (sequential queries — each < 10s)

For the top ~20 unassigned orders (by amount * age), run `../shared/queries/ecm-triage-fast.sql` with `{order_id}` replaced.

This returns: `stuck_reason`, `team_dependency`, `category`, plus all system statuses (goms, falcon, lulu, payout, rfi, checkout, rda).

**Batch optimization:** Use `IN ('{id1}', '{id2}', ...)` for up to 10 order_ids per query.

**⚠️ MCP Timeout:** Each query must complete within 30s. Run in batches of 10.

### Step 3.5: DISQUALIFY non-actionable orders (MANDATORY — before scoring)

**⚠️ CRITICAL: A wrong assignment means a real stuck customer gets skipped. Every enriched order MUST pass ALL of these checks before it can be scored or assigned.**

After enrichment, **disqualify** any order that matches these patterns. Disqualified orders are **dropped silently** — they are NOT assigned, NOT scored, NOT shown to agents.

#### Disqualification Rules

| # | Condition | Reason | What it actually is |
|---|-----------|--------|---------------------|
| D1 | `payment_status_goms` IN (`CREATED`, `INITIATED`, NULL) AND `falcon_status` IS NULL | **Abandoned payment** | Customer never completed payment (e.g., didn't finish bank auth). No money was taken. Not an ops issue. |
| D2 | `stuck_reason` = `uncategorized` AND `payment_status_goms` != `COMPLETED` | **No completed payment + no known pattern** | Order never progressed past payment. Nothing for ops to act on. |
| D3 | `stuck_reason` = `uncategorized` AND all downstream systems are NULL (no falcon, no lulu, no payout, no rfi) | **Dead order** | Order exists in GOMS only, never moved to any processing system. Customer abandoned or system rejected silently. |
| D4 | `goms_sub_state` = `AWAIT_RETRY_INTENT` AND `payment_status_goms` != `COMPLETED` | **Payment retry pending but never paid** | System waiting for retry but customer never completed initial payment. Auto-expires. |
| D5 | Order age > 30 days AND `stuck_reason` = `uncategorized` | **Stale uncategorized** | If it's been 30+ days and doesn't match any pattern, it's not a real edge case. Flag for engineering review, don't assign to ops. |

#### How to apply

```
FOR EACH enriched_order:
    IF matches ANY D1-D5 rule:
        → SKIP (do not score, do not assign)
        → Add to "Disqualified" summary in briefing (show count + reasons)
    ELSE:
        → Proceed to Step 4 (Priority Scoring)
```

#### What to do with disqualified orders

- **D1/D2/D4 (abandoned payments):** These should auto-expire. If there are many (>20), flag to engineering — the auto-cancel mechanism may be broken.
- **D3 (dead orders):** Flag to engineering for cleanup.
- **D5 (stale uncategorized):** Report as potential gap in stuck_reason logic. Add to `../shared/config/knowledge-graph.yaml` gaps section.

**⚠️ NEVER assign an `uncategorized` order without first verifying that payment was completed AND at least one downstream system (falcon/lulu/payout) has a record. An uncategorized order with no payment = abandoned, not stuck.**

### Step 4: Compute Priority Score (with Sentinel Learnings)

#### Step 4a: Load Sentinel Learnings (if available)

Before scoring, check if `sentinel/learnings.yaml` exists. If it does, load the adjustments:

```
FILE: sentinel/learnings.yaml (auto-generated by Sentinel — DO NOT EDIT MANUALLY)
```

**How learnings modify scoring:**

| Adjustment Type | Condition | Effect on Score |
|-----------------|-----------|-----------------|
| `deprioritize` | `self_heal_rate > 0.7` for a stuck_reason | Multiply `stuck_severity_score` by `severity_multiplier` (usually 0.5) |
| `flag_for_review` | `accuracy < 0.6` for a stuck_reason | Keep normal score, but add ⚠️ "low confidence" flag to briefing |
| `likely_false_positive` | `avg_resolution_min < 5` for a stuck_reason | Multiply `stuck_severity_score` by 0.3 (near-zero priority) |
| `needs_triage_fix` | `false_positive_rate > 0.2` for a stuck_reason | Keep normal score, but add 🔴 flag — triage logic may need fixing |

If `sentinel/learnings.yaml` doesn't exist or is empty, skip this step and use default scoring.

#### Step 4b: Compute Score

For each enriched order, compute:

```
PRIORITY_SCORE = (
  0.25 * age_score +
  0.20 * amount_score +
  0.25 * stuck_severity_score * severity_multiplier +   ← adjusted by learnings
  0.15 * rfi_urgency_score +
  0.10 * payment_risk_score +
  0.05 * attempt_score
)
```

Where `severity_multiplier` = value from learnings for this stuck_reason, default **1.0** if no adjustment exists.

**Factor scoring tables** — see `../shared/config/knowledge-graph.yaml` Section 4 for full definitions.

Quick reference:
| Factor | Weight | Key thresholds |
|--------|--------|----------------|
| Age | 0.25 | >72h=1.0, 36-72h=0.9, 24-36h=0.7, 12-24h=0.5 |
| Amount | 0.20 | >15K=1.0, 5-15K=0.8, 2-5K=0.7, 500-2K=0.5 |
| Severity | 0.25 | refund_pending=1.0, falcon_failed=0.9, no_rfi=0.8, sync=0.6, brn=0.5 |
| RFI | 0.15 | EXPIRED=0.9, REJECTED>24h=0.8, REQUESTED>24h=0.7 |
| Payment | 0.10 | FAILED no refund=0.9, PENDING=0.3 |
| Attempts | 0.05 | 3+=0.9, 2=0.5, 1=0.1 |

**Priority mapping:**
- Score > 0.7 → **P1** (Critical)
- Score 0.5-0.7 → **P2** (High)
- Score 0.3-0.5 → **P3** (Medium)
- Score < 0.3 → **P4** (Low/Monitor)

### Step 5: Check agent capacity

Read **Agents** tab:
- Get all agents where `Active = TRUE`
- Count current OPEN/IN_PROGRESS tickets per agent from Assignments tab
- Calculate available slots = `Max Tickets` - current count
- Match agent `Team` to order `team_dependency`

**Agent roster is read from the Agents tab** — do not hardcode. Current roster:
| Agent | Team | Max | Role |
|-------|------|-----|------|
| Dinesh (dinesh@aspora.com) | Manager | 10 | Manager + Ops (triages AND takes tickets) |
| Akshay (akshay@aspora.com) | Ops | 10 | Ops |
| Aakash (aakash@aspora.com) | Ops | 10 | Ops |
| Snita (snita@aspora.com) | Ops | 10 | Ops |

### Step 6: Assign and write to Sheet

For each available agent slot:
1. Pick highest-priority unassigned order matching agent's team
2. Generate actionable from Action Generation Rules below
3. **Append row to Assignments tab** with ALL 10 columns:

```
Order ID | Assigned Agent | Assigned At | Status | Priority | Currency | Amount | Hours Stuck | Diagnosis | Notes
{order_id} | {agent_email} | {timestamp} | OPEN | P{1-4} | {currency} | {amount} | {hours} | {stuck_reason} | {one_line_actionable}
```

Use `sheets_update_cells` tool:
- spreadsheet_id: `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`
- sheet: `Assignments`
- range: next available row (e.g., `A36:J36` if 35 rows exist)

### Step 7: Update ECM Dashboard tab

After assignment, update the **ECM Dashboard** tab summary:
```
Total Open Tickets | {count of OPEN + IN_PROGRESS in Assignments}
In Progress        | {count of IN_PROGRESS}
Resolved Today     | {count from Resolutions tab where date = today}
Escalated Today    | {count from Escalations tab where date = today}
Avg Resolution Time| {avg from Resolutions tab}
SLA Met Rate       | {from Resolutions tab}
```

---

## Action Generation Rules

For each `stuck_reason`, generate the **exact next step**. These are baked into the Notes column.

### Ops Team Actions

#### `status_sync_issue`
```
🔄 STATUS SYNC — Order delivered but system not updated
ACTION: Force webhook replay. Verify LULU shows CREDITED, then trigger /ops force-sync {order_id}
CHECK: AlphaDesk → Order → Confirm LULU=CREDITED
IF LULU ≠ CREDITED: Escalate — order may not be delivered
```
**One-liner for Notes:** `Status sync: Force-sync via AlphaDesk, verify LULU=CREDITED`

#### `falcon_failed_order_completed_issue`
```
⚠️ FALCON FAILED but ORDER COMPLETED — Payout was successful at partner
ACTION: Verify payout on partner dashboard (LULU/VDA). Get UTR/ref number as proof.
THEN: Force GOMS status update via AlphaDesk ManualAction
IF CUSTOMER COMPLAINED: Respond with "Transfer successful. Reference: {UTR}"
```
**One-liner:** `Falcon sync: Verify payout at partner, force GOMS update via AlphaDesk`

#### `brn_issue`
```
🔄 BRN NOT PUSHED — Payment reconciled but Lulu waiting for BRN
ACTION: Check payment acquirer for transaction reference (BRN)
  - CHECKOUT: Checkout Dashboard → get payment reference
  - LEANTECH: Lean Admin → get transaction ref
  - UAE_MANUAL: Check uae_manual_payments table for UTR
THEN: Push BRN to Lulu via AlphaDesk or /ops push-brn {order_id}
VERIFY: Lulu status should change from PAYMENT_PENDING → TXN_TRANSMITTED
```
**One-liner:** `BRN push: Get ref from {acquirer}, push to Lulu via AlphaDesk`

#### `stuck_at_lean_recon`
```
⏳ LEAN RECONCILIATION STUCK — LEANTECH payment with PAYMENT_PENDING at Lulu
ACTION: Check Lean reconciliation queue in Lean Admin
IF RECONCILED in Lean but not in system: Trigger manual reconciliation
IF NOT RECONCILED: Escalate to Ahsan (Lean Slack channel)
```
**One-liner:** `Lean recon: Check Lean Admin queue, escalate to Ahsan if stuck`

#### `stuck_at_lulu`
```
⏳ STUCK AT LULU — No Falcon transaction but Lulu status exists
ACTION: Check Lulu Dashboard for current sub_status
  - If TXN_TRANSMITTED (< 24h): Wait — normal processing
  - If TXN_TRANSMITTED (> 24h): Verify beneficiary details, check for holds
  - If TXN_TRANSMITTED (> 48h): Escalate to Lulu support (Binoy: binoy.francis@ae.luluexchange.com)
  - If EXECUTED: Check status sync — may need webhook replay
```
**One-liner:** `Lulu stuck: Check Lulu dashboard, escalate to Binoy if >48h`

#### `refund_pending`
```
💰 REFUND NEEDED — Order failed/cancelled, customer funds captured
ACTION: Check if auto-refund was triggered in payment acquirer dashboard
  - CHECKOUT: Check for refund transaction in Checkout Dashboard
  - LEANTECH: Check Lean refund queue
IF NO REFUND EXISTS: Initiate manually via acquirer dashboard
AMOUNT: {send_amount} {currency_from}
LOG: Track refund reference in resolution notes
⚠️ CUSTOMER IMPACT: Funds are held — treat as urgent
```
**One-liner:** `REFUND {amount} {currency}: Check {acquirer} refund queue, initiate if missing`

#### `cancellation_pending`
```
🚫 CANCELLATION NOT COMPLETED
ACTION: Check current Lulu status:
  - If TXN_TRANSMITTED: Can still cancel → Lulu Admin → Request Cancellation
  - If CREDITED: Too late → Submit recall to Lulu support (~60% success, 3-5 days)
THEN: Process refund once cancellation confirmed
```
**One-liner:** `Cancel pending: Check Lulu status, request cancellation or recall`

#### `stuck_at_rda_partner`
**One-liner:** `RDA stuck: Check partner dashboard, escalate if >24h`

#### `stuck_due_to_payment_issue_goms`
**One-liner:** `GOMS payment: Check acquirer for capture status, manual recon if needed`

#### `uncategorized`
**One-liner:** `INVESTIGATE: Full system review needed — GOMS/Falcon/Lulu/Payout`

### KYC Ops Team Actions

#### `rfi_order_within_24_hr`
**One-liner:** `RFI <24h: MONITOR ONLY — do NOT nudge customer`

#### `rfi_order_grtr_than_24_hr`
**One-liner:** `RFI >24h: Nudge customer via email/SMS, check AlphaDesk if data >4h stale`

#### `stuck_at_rda_partner_rfi_within_24_hrs`
**One-liner:** `RDA+RFI <24h: Monitor, no nudge`

#### `stuck_at_rda_partner_rfi_grtr_than_24_hrs`
**One-liner:** `RDA+RFI >24h: Nudge customer urgently, partner blocked`

#### `no_rfi_created`
**One-liner:** `BUG: Create RFI manually in AlphaDesk, flag for engineering`

#### `stuck_due_trm`
**One-liner:** `TRM hold: Check ComplianceReview in AlphaDesk, release or create RFI`

#### `stuck_due_trm_rfi_within_24_hrs`
**One-liner:** `TRM+RFI <24h: Monitor, TRM release pending docs`

#### `stuck_due_trm_rfi_grtr_than_24_hrs`
**One-liner:** `TRM+RFI >24h: Nudge customer for docs, TRM waiting`

### VDA Ops Team Actions

#### `stuck_at_vda_partner`
**One-liner:** `VDA stuck: Check partner dashboard, force sync if completed`

#### `bulk_vda_order_within_16_hrs` / `bulk_vda_order_grtr_than_16_hrs`
**One-liner:** `Bulk VDA: Monitor (<16h) or escalate (>16h)`

---

## Output Format

### Manager Briefing (analyse and assign)

```
🎯 Sentinel Triage Report — {date}
📅 Time range: Last {time_range_days} days (orders 12h+ old)
🎯 Precision: {precision}% | ⏱ Avg MTTR: {avg_mttr} min   ← from learnings (omit if no data yet)

📊 Overview:
│ Actionable stuck orders: {actionable_count}
│ Already assigned: {assigned_count} (across {agent_count} agents)
│ New to assign: {new_count}
│ Critical (> 36h): {critical_count} 🔴
│ High-value (> 5K): {high_value_count} 💰
│ Disqualified (not real edge cases): {disqualified_count} ⛔
│ Self-healed since last triage: {self_healed_count} 🔄  ← from assessment (omit if 0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Top Priority Orders (unassigned):

# | Order ID      | Amount       | Stuck  | Diagnosis                | Priority | Action
--|---------------|-------------|--------|--------------------------|----------|--------
1 | {order_id}    | {amt} {cur} | {hrs}h | {stuck_reason}           | P1 🔴    | {one_liner}
2 | {order_id}    | {amt} {cur} | {hrs}h | {stuck_reason}           | P1 🔴    | {one_liner}
...

⚠️ Sentinel Learnings Applied (if any):
│ {stuck_reason_1}: deprioritized (severity × 0.5) — {reason}
│ {stuck_reason_2}: low confidence ⚠️ ({accuracy}% historical accuracy)
│ (omit this section if no learnings exist yet)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👥 Agent Capacity:

Agent          | Team    | Current | Max | Available | Status
---------------|---------|---------|-----|-----------|--------
{dynamically read from Agents tab — do not hardcode}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 Recommended Assignments:

→ Assign {order_id} (P1, {stuck_reason}) → @{agent} ({team})
→ Assign {order_id} (P1, {stuck_reason}) → @{agent} ({team})
...

Shall I proceed with these assignments? (confirm to write to Sheet)
```

### After Manager Confirms

Write all assignments to the **Assignments** tab and display:

```
✅ Assigned {count} tickets

# | Order ID      | Agent        | Priority | Action
--|---------------|-------------|----------|--------
1 | {order_id}    | @{agent}    | P1 🔴    | {one_liner}
2 | {order_id}    | @{agent}    | P2 🟠    | {one_liner}
...

📊 Updated Capacity:
{dynamically from Agents tab}

ECM Dashboard updated. Agents can now run `my tickets` to see their queue.
```

---

## Re-Triage (refresh)

When manager says "triage" again:
1. Re-run Step 1-4
2. Check if any assigned orders are now resolved in Redshift (status = COMPLETED)
   - Auto-update Assignments tab: Status → RESOLVED
3. Re-rank remaining orders
4. Show updated briefing with delta from last triage

---

## Edge Cases

### No unassigned stuck orders
```
✨ All actionable orders are assigned!

Currently assigned: {count} across {agent_count} agents
Agents can run `my tickets` to see their queue.
```

### All agents at capacity
```
⚠️ All agents at capacity!

{dynamically from Agents tab}

{overflow_count} P1/P2 orders remain unassigned.
Options:
1. Increase capacity in Agents tab
2. Resolve existing tickets first
3. Escalate overflow to L2
```

---

## Guardrails

### Assignment Safety (HIGHEST PRIORITY)
- **NEVER assign an order without running Step 3.5 disqualification first**
- **NEVER assign `uncategorized` orders** unless payment_status = COMPLETED AND at least one downstream system has a record
- **NEVER assign abandoned payments** (payment_status IN CREATED/INITIATED/NULL with no falcon) — these are NOT edge cases
- **A wrong assignment = a real stuck customer gets deprioritized.** Treat every assignment as consequential.
- Only assign orders that match a known `stuck_reason` from the runbooks OR have verified payment completion + system progression
- If in doubt, **do not assign** — add to disqualified summary and flag for manual review

### Data Integrity
- Only assign orders verified in Redshift
- Respect agent capacity limits from Agents tab
- Never write to Redshift (read-only)
- All writes go to Google Sheets (Assignments tab)
- Write ALL 10 columns when appending (Order ID through Notes)
- Priority scores from `../shared/config/knowledge-graph.yaml` — do not invent scores
- Action text from this skill's Action Generation Rules — do not invent steps
- If stuck_reason is `uncategorized`, say so — do not guess
- Team-based assignment: Ops→Ops agents, KYC_ops→KYC agents, VDA_ops→VDA agents
- RFI < 24h: NEVER suggest nudging customer
- Always check data freshness: if transfer_rfi.modified_at > 4h, recommend AlphaDesk
- **Manager must confirm** before writing assignments to Sheet
