# Pattern Intelligence Skill (Manager Persona)

## Guardrails
- **Always apply** the rules in `../shared/guardrails.md`. Use only data from executed queries; never invent patterns, counts, or recommendations.

## Trigger
- "patterns"
- "failure patterns"
- "systemic issues"
- "pattern analysis"
- "what's broken"
- "root cause analysis"

## Purpose
Automatically cluster stuck orders by failure signature, quantify impact, detect trends, and surface systemic issues — so they're fixed at the root instead of resolved one-by-one.

## Persona: Manager
This skill is for the **manager persona**. It does NOT triage, assign, or resolve orders.
It produces an analytical report for engineering and ops leads.

---

## Google Sheet
**Spreadsheet ID:** `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`

### Pattern Intelligence Tab Schema (12 columns)
```
A: Date (IST) | B: Pattern_ID | C: Stuck_Reason | D: Currency | E: Signature
F: Order_Count | G: Total_Amount | H: Avg_Hours | I: High_Value_Count | J: Critical_Count
K: Delta_vs_Previous | L: Recommendation
```

---

## Three Cognitive Layers (SAGE)

### Layer 1: WHAT (Domain Model)

- **Pattern** = unique combination of (currency_from, sub_state, lulu_status, has_falcon, falcon_status, payout_status, payout_partner, rfi_status)
- **Cluster** = group of orders sharing the same pattern signature
- **Impact Score** = `order_count * (total_amount / 10000) * (avg_hours_stuck / 24)`
  - Captures volume, financial exposure, and urgency in one number
- **Trend** = delta in order_count vs previous day's run (from Sheet)
- **Minimum cluster size** = 3 orders (smaller clusters are noise, not patterns)

### Layer 2: WHY (Reasoning Framework)

- **Why cluster by system states?** Stuck orders fail at specific integration points. Grouping by failure point reveals systemic issues (fix once for N orders) vs one-offs (fix individually).
- **Why impact score?** 56 orders at 297K AED is more urgent than 3 orders at 500 AED — even with the same stuck_reason. Impact score captures this.
- **Why trends?** A growing cluster = active systemic issue. A shrinking one = fix is working. NEW clusters = novel failure mode.
- **When is a pattern "novel"?** A cluster that doesn't map to any known stuck_reason in `../shared/stuck-reasons.yaml` with size >= 5 = new failure mode not in knowledge graph.

### Layer 3: HOW (Execution Protocol)

---

## Phase 1: Run Clustering Query

Read and execute `../shared/queries/ecm-pattern-clusters.sql` via `mcp__ecm-gateway__redshift_execute_sql_tool`.

**Validation (DEC-010):**
- Sum all `order_count` values across returned rows
- Expected total: 200-600 orders
- If > 1,000: WARN — dead order filter may be broken
- If > 2,000: STOP — investigate filters before proceeding
- If query fails or times out: fallback to `../shared/queries/ecm-pending-list.sql` and aggregate results in-agent

---

## Phase 2: Classify and Score Patterns

For each cluster row returned by the query:

### 2a. Map to stuck_reason

Use the state combination columns to map each pattern to a `stuck_reason` from `../shared/stuck-reasons.yaml`:

| State Combination | Likely stuck_reason |
|-------------------|---------------------|
| sub_state=REFUND_TRIGGERED, lulu_status=CANCELLATION_COMPLETED, has_falcon=no | `refund_pending` |
| sub_state=FULFILLMENT_PENDING, has_falcon=no, lulu_status has value | `stuck_at_lulu` |
| sub_state=FULFILLMENT_PENDING, has_falcon=yes, falcon_status=FAILED | `falcon_failed_order_completed_issue` |
| sub_state=FULFILLMENT_PENDING, has_falcon=yes, payout_status=COMPLETED | `status_sync_issue` |
| sub_state=FULFILLMENT_PENDING, lulu_status=PAYMENT_PENDING | `stuck_at_lean_recon` or `brn_issue` |
| sub_state=AWAIT_EXTERNAL_ACTION, rfi_status != none | `rfi_order_*` (check hours for 24h threshold) |
| sub_state=MANUAL_REVIEW | Check falcon_status and rfi_status for TRM patterns |
| No known match, cluster size >= 5 | Flag as **novel pattern** |
| No known match, cluster size < 5 | `uncategorized` |

**Do NOT hardcode** — use the state dimensions returned by the query + stuck-reasons.yaml definitions to reason about each pattern. The table above is guidance, not exhaustive.

### 2b. Compute Impact Score

```
impact_score = order_count * (total_amount / 10000) * (avg_hours_stuck / 24)
```

### 2c. Flag Severity Indicators

- `high_value_count > 0` → flag with "contains high-value orders (>5K)"
- `critical_count > order_count / 2` → flag with "majority critical (>36h stuck)"
- `max_hours_stuck > 72` → flag with "contains orders stuck >72h"

### 2d. Rank by Impact Score (descending)

---

## Phase 3: Interpret Top Patterns

For each of the top 10 patterns (by impact_score):

### Known stuck_reasons
Pull from `../shared/stuck-reasons.yaml`:
- `team` — which team owns this
- `sla_hours` — expected resolution time
- `action` — recommended action
- `escalation` — who to escalate to

Generate:
- **What's happening**: Plain-English description of the failure pattern
- **Recommended action**: From stuck-reasons.yaml action field
- **Systemic fix**: If this pattern has > 10 orders, suggest a systemic fix (e.g., "Investigate webhook delivery pipeline" for status_sync_issue)

### Unrecognized state combinations
If a pattern doesn't map to any stuck_reason:

1. **LOAD** `../shared/config/knowledge-graph.yaml` (lazy — only load when needed)
2. Match state combination against `partners.*.failure_modes` (detect → fix, sla)
3. If match found → suggest adding as new stuck_reason to stuck-reasons.yaml
4. If no match → flag as **"novel pattern — manual investigation needed"**

---

## Phase 4: Compare to Previous Run (Trends)

### 4a. Read Previous Data

Read "Pattern Intelligence" tab from Sheet (`1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`).

Use `mcp__ecm-gateway__sheets_get_sheet_data`:
- spreadsheet_id: `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`
- sheet: `Pattern Intelligence`

### 4b. Filter to Most Recent Date

Find rows with the most recent date in column A. These represent the previous run's patterns.

### 4c. Compute Deltas

For each current pattern, compare to previous run by matching on `Signature` (column E):

| Delta | Meaning |
|-------|---------|
| **UP (+N)** | order_count increased — pattern is growing |
| **DOWN (-N)** | order_count decreased — fix may be working |
| **NEW** | Pattern exists now but not in previous run — new failure mode |
| **GONE** | Pattern was in previous run but not now — resolved |

### 4d. First Run

If no previous data exists (empty tab or only headers):
- Skip trend comparison
- Note: "First run — baseline established"

---

## Phase 5: Generate Report (Tiered — SAGE: Gradual)

### Quick Mode (user says "quick patterns")
Top 5 patterns by impact. No trends. No novel detection.

### Standard Mode (default)
Top 10 patterns by impact + trends + novel patterns.

### Deep Mode (user says "deep patterns" or "all patterns")
All patterns + per-pattern recommendations + detailed trend analysis.

---

## Phase 6: Output

### 6a. Write to Google Sheet

Write current run data to "Pattern Intelligence" tab.

Use `mcp__ecm-gateway__sheets_update_cells`:
- spreadsheet_id: `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`
- sheet: `Pattern Intelligence`
- Append below existing data (find next empty row)

**Row format (per pattern):**
```
A: {date in IST, YYYY-MM-DD HH:MM}
B: P{row_number} (e.g., P1, P2, P3)
C: {stuck_reason or "novel" or "uncategorized"}
D: {currency_from}
E: {sub_state}|{lulu_status}|{has_falcon}|{falcon_status}|{payout_status}|{rfi_status}
F: {order_count}
G: {total_amount}
H: {avg_hours_stuck}
I: {high_value_count}
J: {critical_count}
K: {delta string: "UP +5", "DOWN -3", "NEW", "GONE", or "baseline"}
L: {one-line recommendation from Phase 3}
```

**Timestamps**: All dates/times in IST (UTC+5:30). To convert from UTC: add 5 hours 30 minutes.

### 6b. Post to Slack

Post as a **separate message** (not in triage thread). Use Slack Bot API via `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` env vars.

---

## Output Format

### Slack / Terminal Report

```
🔬 Pattern Intelligence Report — {date IST}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Summary:
│ Total actionable orders: {sum of order_count across all clusters}
│ Distinct failure patterns: {number of clusters}
│ Novel patterns (new failure modes): {count of unrecognized clusters >= 5}
│ Patterns trending UP: {count} | DOWN: {count} | NEW: {count}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 Top Patterns by Impact:

#  | Reason              | Currency | Orders | Amount     | Avg Hours | Impact | Trend
---|---------------------|----------|--------|------------|-----------|--------|------
1  | {stuck_reason}      | {cur}    | {n}    | {amount}   | {hrs}     | {score}| {delta}
2  | {stuck_reason}      | {cur}    | {n}    | {amount}   | {hrs}     | {score}| {delta}
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Pattern Details:

Pattern #1: {stuck_reason} ({currency})
  Signature: {sub_state} | Lulu: {lulu_status} | Falcon: {has_falcon}/{falcon_status} | Payout: {payout_status}
  Orders: {n} | Total: {amount} | Stuck avg {hrs}h (max {max_hrs}h)
  Flags: {high_value_count} high-value | {critical_count} critical
  What's happening: {plain English description}
  Team: {team} | SLA: {sla_hours}h
  Recommended action: {action from stuck-reasons.yaml}
  Systemic fix: {if applicable}
  Trend: {UP/DOWN/NEW/GONE with delta}

{repeat for top N patterns}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Novel Patterns (not in stuck-reasons.yaml):

{If any unrecognized clusters >= 5 orders:}
  Signature: {state combination}
  Orders: {n} | Amount: {total}
  Closest known pattern: {best guess from knowledge-graph.yaml}
  Action: Manual investigation needed — consider adding to stuck-reasons.yaml

{If none: "No novel patterns detected."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 Trend Summary (vs previous run):
  Growing: {list patterns with UP trend}
  Shrinking: {list patterns with DOWN trend}
  New today: {list NEW patterns}
  Resolved: {list GONE patterns}

{If first run: "First run — baseline established. Trends available from next run."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Written to Google Sheet "Pattern Intelligence" tab.
```

---

## Guardrails

### Pattern-Specific
- **Never invent patterns** — use only clustering query results
- **Never suggest fixes not grounded in** `../shared/stuck-reasons.yaml` or `../shared/config/knowledge-graph.yaml`
- **Minimum cluster size = 3 orders** — smaller clusters are noise, not patterns (enforced by SQL HAVING clause)
- **Timestamps always in IST** (UTC+5:30)
- **All ECM guardrails** from `../shared/guardrails.md` apply

### Scope
- This skill **detects and reports** failure patterns — it does NOT triage, assign, or resolve
- Recommendations are for engineering/ops leads to action — not for field agents
- If a pattern maps to a known stuck_reason, reference the existing runbook — do NOT invent new resolution steps

### Data Integrity
- All numbers must trace back to query results — no arithmetic on assumed values
- Impact scores are computed from query columns only (order_count, total_amount, avg_hours_stuck)
- Trends compare to Sheet data only — never compare to "expected" or "typical" values
