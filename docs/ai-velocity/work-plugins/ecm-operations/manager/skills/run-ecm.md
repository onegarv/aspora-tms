# Run ECM Skill

## CRITICAL: MCP Configuration

**ONLY use the `ecm-gateway` MCP server for all Redshift queries.**

| Tool Name | Purpose |
|-----------|---------|
| `mcp__ecm-gateway__redshift_execute_sql_tool` | Execute SQL queries |
| `mcp__ecm-gateway__redshift_list_databases` | List databases |

**DO NOT use:**
- `awslabs.redshift-mcp-server` — NOT configured
- Any standalone Redshift MCP — NOT available
- Hallucinated data — NEVER make up results

If a query fails, report the error. Do NOT invent data.

---

## Guardrails
- **Always apply** the rules in `../shared/guardrails.md`. Use only data from the executed query and from runbooks; never invent order IDs, counts, or resolution steps. If the query has not been run, say so — do not show fake dashboard data.

## Trigger
- "run ECM"
- "show ECM dashboard"
- "ECM overview"
- "stuck orders"

## Description
Displays Exception Case Management data for stuck remittance orders. **Choose the right query based on speed vs detail needs.**

---

## Query Options (Choose One)

### Option A: FAST List (< 5 seconds) ⚡
**File:** `../shared/queries/ecm-pending-list.sql`

**Use when:** User wants quick count or list of stuck orders, no stuck_reason needed yet.

```sql
-- NO JOINS - hits orders_goms only (NOT the view)
SELECT order_id, order_date, order_status, currency_from, send_amount, receive_amount, hours_diff, category
FROM orders_goms
WHERE status IN ('PROCESSING_DEAL_IN', 'PENDING', 'FAILED', 'IN_PROGRESS') ...
```

**Returns:** order_id, order_date, order_status, currency_from, send_amount, receive_amount, hours_diff, category

**Does NOT return:** stuck_reason, team_dependency, lulu_status, payout_status

---

### Option B: Dashboard Summary ⚠️ TIMES OUT via MCP
**File:** `../shared/queries/ecm-dashboard-summary.sql`

**⚠️ DO NOT run via MCP** — exceeds 30-second timeout due to JOINs.

**Instead:** Run in Metabase and export CSV, OR use the two-step workflow below.

**Two-Step Workflow for Dashboard:**
1. Run **Option A** (fast list) to get order_ids
2. Run **Option C** (detail) for batches of ~20 order_ids at a time
3. Aggregate results for dashboard display

---

### Option C: Order Detail (Sequential Queries) 🔍
**File:** `../shared/queries/ecm-order-detail.sql`

**Use when:** User asks about specific order(s) - run queries SEPARATELY (JOINs timeout).

**Workflow:**
1. **Query 1** - Base order data from `orders_goms` (NOT the view - it times out)
2. **Query 2** - Get `ftv_transaction_id` from `falcon_transactions_v2` using `client_source_transaction_id`
3. **Query 3** - Lulu status from `lulu_data` (for AED orders)
4. **Query 4** - Payout status from `transaction_payout` (use `ftv_transaction_id` from Query 2)
5. **Query 5** - RFI status from `transfer_rfi`
6. **Combine results** and apply stuck_reason logic (see file for rules)

**Each query is fast (<5s) when filtered by order_id or transaction_id!**

---

## Recommended Workflow

1. **"Run ECM" / "Show dashboard"** → Use **Option A** (Fast List) + **Option C** (sequential) for top 20 orders
2. **"How many stuck orders?"** → Use **Option A** (Fast List) - returns count + basic info
3. **"Tell me about order XYZ"** → Use **Option C** queries 1-6 sequentially
4. **"What's wrong with these orders?"** → Use **Option C** sequential queries, combine results, apply stuck_reason logic

**⚠️ MCP Timeout:** 30 seconds. JOINs on large tables will timeout. Always use sequential queries.

---

## ⚠️ Do NOT Run via MCP
**File:** `../shared/queries/ecm-active-tickets.sql`

This is the full Metabase query with all CTEs. **It will timeout.** Use the options above instead.

---

## Runbook Integration
Once you have `stuck_reason`, point users to the appropriate runbook:

### Ops Team
- `status_sync_issue` → `../shared/runbooks/status-sync-issue.md`
- `falcon_failed_order_completed_issue` → `../shared/runbooks/falcon-failed.md`
- `stuck_at_lean_recon` → `../shared/runbooks/lean-recon.md`
- `brn_issue` → `../shared/runbooks/brn-issue.md`
- `stuck_at_lulu` → `../shared/runbooks/stuck-at-lulu.md`
- `refund_pending` → `../shared/runbooks/refund-pending.md`
- `cancellation_pending` → `../shared/runbooks/cancellation-pending.md`
- `stuck_at_rda_partner` → `../shared/runbooks/stuck-at-rda-partner.md`
- `stuck_due_to_payment_issue_goms` → `../shared/runbooks/goms-payment-issue.md`
- `uncategorized` → `../shared/runbooks/uncategorized.md`

### KYC Ops Team
- `rfi_pending` / `rfi_order_*` → `../shared/runbooks/rfi-pending.md`
- `stuck_at_rda_partner_rfi*` → `../shared/runbooks/rda-rfi.md`
- `no_rfi_created` → `../shared/runbooks/no-rfi-created.md`
- `stuck_due_trm*` → `../shared/runbooks/stuck-due-trm.md`

### VDA Ops Team
- `stuck_at_vda_partner` → `../shared/runbooks/stuck-at-vda-partner.md`
- `bulk_vda_order_*` → `../shared/runbooks/bulk-vda.md`

---

## Output
- **Type**: Artifact (optional)
- **Component**: `ecm-dashboard.jsx`
- **Data Injection**: Query results injected as `ECM_DATA` constant

Or present results as a table/summary in chat.

## Actions Available
- Filter by team_dependency, category, currency, stuck_reason
- View order details (run Option C query)
- Export to CSV
- Open runbook for stuck_reason
