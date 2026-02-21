---
name: datadog-dashboard-creation
description: "Create production-grade Datadog dashboards following Google SRE principles, RED method, and observability best practices. Dashboards enable engineers to detect issues in 30 seconds and reduce MTTR."
version: 1.0.0
author: Eventbus Team
---

# Creating Datadog Dashboards

## Prerequisites

**Required:** Datadog API credentials in `.env` file:
```bash
# .env (NEVER commit this file)
DATADOG_API_KEY=your-api-key
DATADOG_APP_KEY=your-app-key
DATADOG_SITE=datadoghq.eu  # or datadoghq.com for US
```

Load credentials before running commands:
```bash
source .env
```

## Domain Model

A Datadog dashboard is a **single pane of glass** for service health, not a data dump.
Its value comes from enabling engineers to detect issues in 30 seconds and identify
root causes in under 5 minutes.

**Key invariants:**
- Every dashboard follows Four Golden Signals structure (Traffic, Errors, Latency, Saturation)
- Every metric query uses verified metrics — never fabricated
- Every error/latency widget has SLO threshold markers
- Template variables enable filtering without dashboard modification

**What "good" looks like:**
- Engineer opens dashboard → sees service health in 30 seconds
- Anomaly visible → drill down to signal section → identify root cause
- No "No data" widgets — every query verified before adding

## Philosophy

We create dashboards that **inform judgment**, not just display numbers.

**Why Four Golden Signals?** Google SRE proved these four signals catch 95% of production
issues. Traffic shows demand. Errors show failures. Latency shows user experience.
Saturation shows capacity.

**Why verify metrics first?** Dashboards with "No data" widgets erode trust. Engineers
stop checking them. A smaller dashboard with working widgets beats a comprehensive one
with gaps.

## Quick Reference

| Task | Approach | When |
|------|----------|------|
| New service dashboard | Full workflow (Steps 1-5) | Service has metrics but no dashboard |
| Add widgets | Step 3 only | Dashboard exists, need more visibility |
| Verify metrics exist | Query Datadog API or Prometheus endpoint | Before any widget creation |
| List dashboards | `curl` to `/api/v1/dashboard` | Check existing dashboards |

## Workflow

### Step 1: Discover Metrics (NEVER SKIP)

**Determine metric source and verify existence.**

For Prometheus-instrumented services (local):
```bash
curl http://localhost:8080/actuator/prometheus | grep "^[a-z]" | head -50
```

For Datadog-native metrics (query API):
```bash
# Verify a metric exists in Datadog
curl -s -G "https://api.${DATADOG_SITE}/api/v1/query" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}" \
  --data-urlencode "query=sum:metric_name{*}" \
  --data-urlencode "from=$(date -v-1H +%s)" \
  --data-urlencode "to=$(date +%s)"
```

**Establish facts (reference throughout):**
- Service name: [from codebase or user]
- Available metrics: [list with tags]
- Prometheus → Datadog name conversion needed: [yes/no]

**If metrics don't exist:** Stop. Suggest `prometheus-instrumentation` skill first.

### Step 2: Design Structure

**Apply Four Golden Signals layout:**

```
1. Documentation Note (full width)
   - Service name, architecture summary
   - SLO thresholds, debug flow

2. Health Summary Group
   - HTTP Request Rate | HTTP Errors | HTTP Latency p95
   - SQS Rate | SQS Errors | SQS Latency (if applicable)

3. Traffic Section
   - Requests/events per second by operation

4. Errors Section
   - Error rate with SLO markers (1% warning, 5% critical)
   - Errors by type/category

5. Latency Section
   - p50, p95, p99 with SLO markers (100ms warning, 500ms critical)
   - Latency by operation

6. Saturation Section
   - CPU, memory utilization
   - Queue depth, connection pools

7. Infrastructure (optional)
   - JVM heap, GC pause, threads

8. Deep Dives (optional)
   - Per-subscription, per-feature breakdown
```

### Step 3: Build Queries

**Apply Datadog syntax rules (see `skills/guardrails.md` if needed):**

| Rule | Correct | Wrong |
|------|---------|-------|
| Filter order | `{tag:value,$env,$service}` | `{$env,$service,tag:value}` |
| Counter names | `metric.count` | `metric_total` |
| Timer max | `metric_max` | `metric.max` |
| Required field | `"data_source": "metrics"` | (omitted) |

**Example query structure:**
```json
{
  "name": "errors",
  "data_source": "metrics",
  "query": "sum:service_operations.count{outcome:error,$env,$service}.as_rate()"
}
```

### Step 4: Create Dashboard

**Generate JSON and deploy via API:**

1. Generate complete dashboard JSON → save to `{service}-dashboard.json`

2. Create dashboard via Datadog API:
```bash
curl -s -X POST "https://api.${DATADOG_SITE}/api/v1/dashboard" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}" \
  -H "Content-Type: application/json" \
  -d @{service}-dashboard.json
```

3. Response contains dashboard URL:
```json
{
  "id": "abc-123-xyz",
  "url": "/dashboard/abc-123-xyz/service-name",
  ...
}
```

**Full URL:** `https://app.${DATADOG_SITE}/dashboard/{id}`

### Step 5: Verify

**Validation loop:**

1. Get dashboard to verify creation:
```bash
curl -s "https://api.${DATADOG_SITE}/api/v1/dashboard/{dashboard_id}" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}"
```

2. Open dashboard URL in browser and check for "No data" widgets

3. If any widget shows "No data":
   - Verify metric name conversion (Prometheus `_total` → Datadog `.count`)
   - Check tag existence
   - Fix query in JSON and update:
```bash
curl -s -X PUT "https://api.${DATADOG_SITE}/api/v1/dashboard/{dashboard_id}" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}" \
  -H "Content-Type: application/json" \
  -d @{service}-dashboard.json
```

4. Repeat until all widgets display data

## Reasoning Guides

### Why Prometheus names differ from Datadog names

Datadog's Prometheus scraper converts metric suffixes:
- `_total` → `.count` (counters track cumulative totals)
- `_max` stays as `_max` (timer maximums are special)
- `_count`/`_sum` → `.count`/`.sum` (histogram components)

**Consequence of wrong names:** Query returns no data. Widget appears broken.

### Why filter order matters

Datadog's query parser expects tags before template variables.
`{outcome:error,$env}` works. `{$env,outcome:error}` may fail silently.

### Why legends are required for grouped queries

Queries with `by {tag}` produce multiple series. Without legend:
- User can't identify which line is which
- Dashboard becomes useless for debugging

## Mechanical Rules

- **ALWAYS** include `"data_source": "metrics"` in every query — queries fail without it
- **ALWAYS** verify metrics exist before creating widgets — prevents "No data" widgets
- **ALWAYS** add SLO markers to error rate (1%, 5%) and latency (100ms, 500ms)
- **NEVER** use Prometheus regex `=~` in Datadog queries — not supported
- **NEVER** mix OR with commas in tag filters — causes parse errors

## Common Failure Modes

| Failure | Symptom | Cause | Fix |
|---------|---------|-------|-----|
| No data | Widget empty | Wrong metric name | Verify Prometheus→Datadog conversion |
| Parse error | Dashboard import fails | Missing `data_source` | Add field to all queries |
| Silent failure | Some widgets empty | Tag doesn't exist | Verify tags from Prometheus output |
| Legend missing | Can't identify series | `by {}` without legend config | Add `show_legend: true` |

## Dependencies & Resources

**Datadog API Endpoints:**
| Operation | Method | Endpoint |
|-----------|--------|----------|
| Create dashboard | POST | `/api/v1/dashboard` |
| Get dashboard | GET | `/api/v1/dashboard/{id}` |
| Update dashboard | PUT | `/api/v1/dashboard/{id}` |
| Delete dashboard | DELETE | `/api/v1/dashboard/{id}` |
| List dashboards | GET | `/api/v1/dashboard` |
| Query metrics | GET | `/api/v1/query` |

**Base URLs by region:**
- US: `https://api.datadoghq.com`
- EU: `https://api.datadoghq.eu`

**Required credentials (from `.env`):**
- `DATADOG_API_KEY` — API key from Organization Settings
- `DATADOG_APP_KEY` — Application key from Organization Settings
- `DATADOG_SITE` — `datadoghq.com` (US) or `datadoghq.eu` (EU)

**Reference files (load only when needed):**
- `skills/guardrails.md` — detailed syntax rules, valid values
- `examples/war-room-system-health-dashboard.json` — working multi-service example
- `examples/beneficiary-service-dashboard.json` — single-service example

**Related skills:**
- `prometheus-instrumentation` — if metrics don't exist yet
