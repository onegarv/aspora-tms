# ECM Query Optimizations

Summary of query optimizations that can be done: **query-level** (SQL changes in this repo) and **DBA-level** (Redshift schema, keys, materialized views). The latter require DBA/DevOps; the former you can apply directly.

---

## 1. Query-level optimizations (apply in this repo)

### 1.1 `ecm-order-detail-v2.sql`

| Change | Why |
|--------|-----|
| **Replace `SELECT *` in CTEs** | In `latest_payments_goms` and `latest_payout`, use explicit columns instead of `SELECT *`. Redshift sends fewer columns through the plan and can reduce broadcast size. |
| **Restrict CTEs by order_id where possible** | `latest_payments_goms` is built from all of `payments_goms` then joined to one order. If you can filter `payments_goms` by `reference_id = '{order_id}'` in that CTE, the CTE does less work. Same idea for `latest_payout`: filter by `transaction_id IN (SELECT transaction_id FROM falcon_transactions_v2 WHERE client_txn_id = '{order_id}')` so the window runs over a small set. |
| **Single-order guarantee** | Query is for one `order_id`; ensure every CTE that can be filtered by that order_id has the filter (you already do in `order_base`). |

Example for `latest_payments_goms` (only need rows for this order):

```sql
latest_payments_goms AS (
    SELECT reference_id, created_at, updated_at, payment_status
    FROM (
        SELECT reference_id, created_at, updated_at, payment_status,
               ROW_NUMBER() OVER (PARTITION BY reference_id ORDER BY COALESCE(updated_at, created_at) DESC) AS rn
        FROM payments_goms
        WHERE payment_status = 'COMPLETED'
          AND reference_id = '{order_id}'   -- restrict early
    ) pg
    WHERE rn = 1
),
```

Apply the same idea to `latest_payout` by filtering on `transaction_id` that comes from falcon for this `order_id` (e.g. in a two-step: first CTE gets `transaction_id` for `order_id`, then `latest_payout` only for that set).

---

### 1.2 `ecm-pending-list.sql`

Already in good shape: single table (`orders_goms`), explicit columns, narrow time window (7 days), `LIMIT 200`. Optional tweaks:

- Ensure **filter columns** (`status`, `meta_postscript_pricing_info_send_currency`, `created_at`) align with **sort key** on `orders_goms` (e.g. `(order_id, created_at)` or `created_at`) so Redshift can use range scans. If DBA adds `SORTKEY (created_at, status)` or similar, this query will benefit.
- Keep **explicit column list**; do not switch to `SELECT *`.

---

### 1.3 `my-tickets.sql`

| Change | Why |
|--------|-----|
| **Stop using `analytics_orders_master_data`** | That view joins 10+ tables. Using it as the driver makes the query heavy and prone to timeouts. |
| **Use `orders_goms` as base** | Same pattern as `ecm-triage-fast.sql`: base CTE from `orders_goms` filtered by `assigned_agent = '{agent_email}'`, then LEFT JOIN only the tables you need (lulu_data, payments_goms, falcon_transactions_v2, transaction_payout) with filters by `order_id` or `transaction_id`. |
| **Filter by assigned_agent and date first** | Restrict to `orders_goms.assigned_agent = '{agent_email}'` and `created_at::date >= CURRENT_DATE - 7` in the base CTE so all downstream joins see a small row set. |

Refactoring `my-tickets.sql` to mirror the structure of `ecm-triage-fast.sql` (order_base from `orders_goms` with filters, then small CTEs per dimension) will align it with the “no heavy view” approach and reduce timeouts.

---

### 1.4 `ecm-triage-fast.sql`

- **Redshift compatibility**: Uses `UNIX_TIMESTAMP()`. Redshift does not support `UNIX_TIMESTAMP()`; use `EXTRACT(EPOCH FROM timestamp_expression)` or `DATEDIFF`/date parts. Replace all `UNIX_TIMESTAMP(...)` with the Redshift-equivalent so the query runs correctly and can use native optimizations.
- **Per-order CTEs**: Already filters each CTE by `order_id` or `transaction_id`; that’s good. Keep explicit column lists in CTEs.

---

### 1.5 `order-details.sql`

- **Deprecate in favor of `ecm-order-detail-v2.sql`** | `order-details.sql` is built on `analytics_orders_master_data` and is not optimized. All call paths should use `ecm-order-detail-v2.sql` (and optionally `ecm-triage-fast.sql` for triage). Remove or clearly mark as deprecated.

---

### 1.6 `lulu-pending-non-terminal.sql`

- Use **explicit columns** instead of `lulu_data.*` in the SELECT so Redshift only reads needed columns.
- Ensure **date filter** (`o.created_at BETWEEN ...`) is on the driving table; if `orders` is large, having a sort key on `created_at` helps (DBA).

---

## 2. DBA-level optimizations (Redshift schema)

These require DBA or DevOps; scripts exist under `_backup/scripts/`. Summary:

### 2.1 Sort keys (non-destructive)

From `_backup/scripts/dba-sortkeys-only.sql`:

- **orders_goms**: `SORTKEY (order_id, created_at)` — helps single-order lookups and date filters.
- **lulu_data**: `SORTKEY (order_id)`.
- **transaction_payout**: `SORTKEY (transaction_id)` (and ideally `updated_at` or `created_at` for “latest” per transaction).
- **falcon_transactions_v2**: `SORTKEY (transaction_id)` or `(client_txn_id, transaction_id)`.
- **transfer_rfi**: `SORTKEY (reference_id)`.

Then run `VACUUM` and `ANALYZE` on those tables so the sort key is effective.

### 2.2 Distribution keys (requires table recreation / deep copy)

If join columns are not the dist key, Redshift broadcasts large tables (DS_BCAST_INNER), which is expensive. Ideal layout:

- **orders_goms**: `DISTKEY (order_id)`.
- **lulu_data**: `DISTKEY (order_id)`.
- **transaction_payout**: `DISTKEY (transaction_id)`.
- **falcon_transactions_v2**: `DISTKEY (client_txn_id)` or `transaction_id` (depending on how you join).
- **transfer_rfi**: `DISTKEY (reference_id)`.

Changing dist key requires creating a new table with the desired `DISTKEY` and migrating data (e.g. `CREATE TABLE ... AS SELECT`), then swapping. See `_backup/scripts/dba-optimization.sql` for the described approach.

### 2.3 Materialized view (recommended in DBA script)

- Create a **materialized view** (e.g. `ecm_orders_mv`) that pre-joins the needed tables and pre-computes `stuck_reason`, `team_dependency`, and key status columns, with `DISTKEY (order_id)` and `SORTKEY (order_id, created_at)`.
- **Refresh** on a schedule (e.g. hourly or daily).
- ECM dashboard and list-style queries then read from the MV instead of joining many tables live; target **&lt; 5 seconds** for list/dashboard.

Details and example DDL are in `_backup/scripts/dba-optimization.sql`.

### 2.4 Flat ETL table (optional)

- Maintain a flat table (e.g. `ecm_stuck_orders`) populated by ETL with only the columns needed for dashboard/triage.
- Queries become single-table reads; latency is minimal. Trade-off is ETL complexity and freshness (e.g. 15–60 min delay).

### 2.5 Statistics

- Run **ANALYZE** on all tables used by ECM queries (orders_goms, lulu_data, payments_goms, falcon_transactions_v2, transaction_payout, transfer_rfi, checkout_payment_data, uae_manual_payments, fulfillments_goms, rda_fulfillments) so the planner has up-to-date stats and can choose better plans.

---

## 3. Quick reference

| Area | Action |
|------|--------|
| **ecm-order-detail-v2** | Explicit columns in CTEs; restrict `latest_payments_goms` / `latest_payout` by order_id/transaction_id where possible. |
| **ecm-pending-list** | Keep as-is; align filters with sort key when DBA adds it. |
| **my-tickets** | Stop using `analytics_orders_master_data`; base on `orders_goms` + targeted JOINs (like ecm-triage-fast). |
| **ecm-triage-fast** | Replace `UNIX_TIMESTAMP()` with Redshift-compatible EXTRACT/DATEDIFF. |
| **order-details** | Deprecate; use ecm-order-detail-v2 only. |
| **lulu-pending-non-terminal** | Explicit column list instead of `lulu_data.*`. |
| **DBA** | Sort keys → VACUUM/ANALYZE; dist keys (recreate tables); materialized view for ECM; optional flat ETL table. |

---

## 4. Where the scripts live

- **Query files**: `queries/*.sql`
- **DBA scripts** (sort keys, MV, dist key notes, flat table): `_backup/scripts/dba-optimization.sql`, `_backup/scripts/dba-sortkeys-only.sql`

Applying the query-level changes in this repo will improve performance without schema changes; the DBA-level changes give the largest gains for dashboard and multi-order workloads.

---

## 5. Verification (test before and after changes)

To ensure user info and output are not impacted:

1. **Before changing a query**: Run it via ecm-gateway MCP (or in Metabase/Redshift) with a known parameter. Capture:
   - Column list (order and names).
   - Row count and one sample row (or full result for single-order queries).

2. **After changing**: Run the same query with the same parameter. Compare:
   - Column list must be identical (same names and order).
   - Row count must match (or be unchanged for the same filters).
   - Sample row: key fields (order_id, status, stuck_reason, amounts, timestamps) must match.

3. **Suggested test cases**:
   - **ecm-order-detail-v2.sql**: Pick one `order_id` (e.g. from ecm-pending-list). Run before/after; compare all columns and the single row.
   - **my-tickets.sql**: Pick one `{agent_email}`. Run before/after; compare column list and row set (order and content of first few rows).
   - **ecm-triage-fast.sql**: Same as order-detail; one `order_id`. Confirm Redshift runs without `UNIX_TIMESTAMP` errors and result matches.
   - **ecm-pending-list.sql**: No structural change; run once and keep result as baseline if you change sort/filters later.
   - **lulu-pending-non-terminal.sql**: Run with a date range; confirm column set matches previous (explicit columns replace `lulu_data.*`; add any missing columns if downstream needs them).

4. **Regression**: If any column is missing or value differs for the same input, treat as regression and fix before deploying.
