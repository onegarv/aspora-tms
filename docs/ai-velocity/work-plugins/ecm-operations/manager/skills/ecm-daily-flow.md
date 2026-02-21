# ECM Daily Flow Analysis

## Guardrails
- **Always apply** the rules in `../shared/guardrails.md`. Use only data from executed queries and runbooks; never invent order IDs, counts, or resolution steps.

## Trigger
- "ECM dashboard"
- "backlog analysis"
- "daily flow"
- "yesterday vs today"
- "how is the backlog"
- "backlog chart"

## Purpose
Analyst-style view of ECM backlog movement: what carried over, what came in yesterday, what came in today, and the 7-day trend. Provides severity breakdown, currency split, and sub-state analysis.

---

## Phase 1: Query Backlog Segments

Run this query to segment the current actionable backlog by cohort (carryover / yesterday / today):

```sql
WITH paid_orders AS (
    SELECT DISTINCT reference_id
    FROM payments_goms
    WHERE payment_status = 'COMPLETED'
      AND created_at >= CURRENT_DATE - 30
),
lulu_orders AS (
    SELECT DISTINCT order_id FROM lulu_data WHERE created_at >= CURRENT_DATE - 30
)
SELECT
    CASE
        WHEN o.created_at::date < CURRENT_DATE - 1 THEN 'carryover'
        WHEN o.created_at::date = CURRENT_DATE - 1 THEN 'yesterday_new'
        WHEN o.created_at::date = CURRENT_DATE THEN 'today_new'
    END AS cohort,
    CASE
        WHEN EXTRACT(EPOCH FROM (GETDATE() - o.created_at)) / 3600 > 36 THEN 'critical'
        WHEN EXTRACT(EPOCH FROM (GETDATE() - o.created_at)) / 3600 > 24 THEN 'action_required'
        WHEN EXTRACT(EPOCH FROM (GETDATE() - o.created_at)) / 3600 > 12 THEN 'warning'
        ELSE 'level_zero'
    END AS category,
    o.meta_postscript_pricing_info_send_currency AS currency,
    COUNT(*) AS order_count,
    ROUND(SUM(o.meta_postscript_pricing_info_send_amount::decimal), 2) AS total_send_amount
FROM orders_goms o
INNER JOIN paid_orders p ON p.reference_id = o.order_id
WHERE
    o.status IN ('PROCESSING_DEAL_IN', 'PENDING', 'FAILED', 'IN_PROGRESS')
    AND o.sub_state IN (
        'FULFILLMENT_PENDING', 'REFUND_TRIGGERED', 'TRIGGER_REFUND',
        'FULFILLMENT_TRIGGER', 'MANUAL_REVIEW', 'AWAIT_EXTERNAL_ACTION'
    )
    AND o.meta_postscript_pricing_info_send_currency IN ('AED', 'GBP', 'EUR')
    AND o.created_at >= CURRENT_DATE - 30
    AND o.created_at < GETDATE() - INTERVAL '12 hours'
    AND (
        o.meta_postscript_pricing_info_send_currency IN ('GBP', 'EUR')
        OR o.order_id IN (SELECT order_id FROM lulu_orders)
    )
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
```

---

## Phase 2: Query 7-Day Inflow Trend

```sql
WITH paid_orders AS (
    SELECT DISTINCT reference_id
    FROM payments_goms
    WHERE payment_status = 'COMPLETED'
      AND created_at >= CURRENT_DATE - 30
),
lulu_orders AS (
    SELECT DISTINCT order_id FROM lulu_data WHERE created_at >= CURRENT_DATE - 30
)
SELECT
    o.created_at::date AS order_date,
    o.meta_postscript_pricing_info_send_currency AS currency,
    COUNT(*) AS new_orders,
    ROUND(SUM(o.meta_postscript_pricing_info_send_amount::decimal), 2) AS total_amount
FROM orders_goms o
INNER JOIN paid_orders p ON p.reference_id = o.order_id
WHERE
    o.status IN ('PROCESSING_DEAL_IN', 'PENDING', 'FAILED', 'IN_PROGRESS')
    AND o.sub_state IN (
        'FULFILLMENT_PENDING', 'REFUND_TRIGGERED', 'TRIGGER_REFUND',
        'FULFILLMENT_TRIGGER', 'MANUAL_REVIEW', 'AWAIT_EXTERNAL_ACTION'
    )
    AND o.meta_postscript_pricing_info_send_currency IN ('AED', 'GBP', 'EUR')
    AND o.created_at >= CURRENT_DATE - 7
    AND (
        o.meta_postscript_pricing_info_send_currency IN ('GBP', 'EUR')
        OR o.order_id IN (SELECT order_id FROM lulu_orders)
    )
GROUP BY 1, 2
ORDER BY 1, 2
```

---

## Phase 3: Query Sub-State Breakdown

```sql
WITH paid_orders AS (
    SELECT DISTINCT reference_id
    FROM payments_goms
    WHERE payment_status = 'COMPLETED'
      AND created_at >= CURRENT_DATE - 30
),
lulu_orders AS (
    SELECT DISTINCT order_id FROM lulu_data WHERE created_at >= CURRENT_DATE - 30
)
SELECT
    o.sub_state,
    o.meta_postscript_pricing_info_send_currency AS currency,
    COUNT(*) AS order_count,
    ROUND(AVG(EXTRACT(EPOCH FROM (GETDATE() - o.created_at)) / 3600), 1) AS avg_hours_stuck,
    ROUND(MAX(EXTRACT(EPOCH FROM (GETDATE() - o.created_at)) / 3600), 1) AS max_hours_stuck
FROM orders_goms o
INNER JOIN paid_orders p ON p.reference_id = o.order_id
WHERE
    o.status IN ('PROCESSING_DEAL_IN', 'PENDING', 'FAILED', 'IN_PROGRESS')
    AND o.sub_state IN (
        'FULFILLMENT_PENDING', 'REFUND_TRIGGERED', 'TRIGGER_REFUND',
        'FULFILLMENT_TRIGGER', 'MANUAL_REVIEW', 'AWAIT_EXTERNAL_ACTION'
    )
    AND o.meta_postscript_pricing_info_send_currency IN ('AED', 'GBP', 'EUR')
    AND o.created_at >= CURRENT_DATE - 30
    AND o.created_at < GETDATE() - INTERVAL '12 hours'
    AND (
        o.meta_postscript_pricing_info_send_currency IN ('GBP', 'EUR')
        OR o.order_id IN (SELECT order_id FROM lulu_orders)
    )
GROUP BY 1, 2
ORDER BY order_count DESC
```

---

## Phase 4: Build Slack Message

Format the results into a Slack Block Kit message with these sections:

### Header
```
📊 ECM Backlog Flow — {date}
```

### Section 1: Waterfall
```
*Backlog Waterfall:*
  Carryover (pre-yesterday):  {carryover_count} orders
+ New Yesterday:             +{yesterday_count} orders
─────────────────────────────────
= Actionable Backlog:        {total} orders
+ Today Pipeline:            +{today_count} (not yet 12h old)
```

### Section 2: Severity Heat Map
Use a table showing cohort × severity with order counts.

### Section 3: 7-Day Inflow Chart
Text bar chart showing daily inflow, highlight spikes (>2x previous day).

### Section 4: Currency Split
Table with currency × cohort breakdown and total value.

### Section 5: Analyst Insights
3-5 bullet points identifying:
- Spikes or anomalies in inflow
- Severity distribution (% critical)
- Bottleneck sub-states
- Currency corridor differences
- Throughput concerns

### Posting
Post via Slack Bot API using `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` env vars.
Use `chat.postMessage` with mrkdwn formatting.

---

## Output Format

Present findings in Minto Pyramid style:
1. **Answer first**: Total backlog number and direction (growing/shrinking)
2. **Supporting data**: Waterfall, severity, trend
3. **Deep dive**: Sub-state analysis, currency breakdown
4. **Action items**: What needs attention

---

## Phase 5: Pattern Intelligence

After completing the daily flow analysis, run pattern intelligence to surface systemic issues.
Say `patterns` to trigger `skills/pattern-intelligence.md`.
