---
name: discovering-metrics
description: >
  Discovers metrics from codebase, Prometheus endpoints, or Datadog API for dashboard
  creation. Use when asked to "find metrics", "what metrics exist", "discover observability",
  or before creating dashboards. Do NOT use for creating metrics (use prometheus-instrumentation).
---

# Discovering Metrics

## Prerequisites

For Datadog API queries:
```bash
source .env  # Contains DATADOG_API_KEY, DATADOG_APP_KEY, DATADOG_SITE
```

## Domain Model

Metrics are the foundation of observability. Before creating any dashboard widget,
the metric must be **verified to exist** with its exact name and tags.

**Key invariants:**
- A metric exists only if it returns data from Prometheus endpoint or Datadog API
- Prometheus names ≠ Datadog names (conversion required)
- Tags must be verified — don't assume `env`, `region`, etc. exist

**The discovery hierarchy:**
1. Prometheus endpoint (source of truth for instrumented metrics)
2. Datadog API (source of truth for Datadog names)
3. Codebase analysis (understand what SHOULD exist)

## Philosophy

**Why codebase analysis isn't enough:** Code shows intent, not reality. A metric
defined in code may not be scraped, may have wrong labels, or may use different
naming in Datadog.

**Why we verify before creating widgets:** A dashboard with "No data" widgets is
worse than no dashboard. Engineers lose trust and stop checking.

## Workflow

### Step 1: Identify Discovery Method

| Source Available | Method |
|-----------------|--------|
| Local Prometheus endpoint | `curl` the metrics endpoint |
| Datadog access | Query Datadog API |
| Codebase only | Analyze code, then verify in Datadog |

### Step 2: Query Prometheus (if available)

```bash
curl http://localhost:8080/actuator/prometheus | grep "^[a-z]" | head -100
```

**Extract from output:**
- Metric names (lines starting with lowercase)
- Tags (values in `{...}`)
- Metric types (from `# TYPE` comments: counter, gauge, histogram, summary)

### Step 3: Map to Datadog Names

| Prometheus Pattern | Datadog Name |
|--------------------|--------------|
| `service_requests_total{...}` | `service_requests.count` |
| `service_duration_seconds_max{...}` | `service_duration_seconds_max` |
| `service_duration_seconds_count{...}` | `service_duration_seconds.count` |
| `jvm_memory_used_bytes{...}` | `jvm_memory_used_bytes` (unchanged) |

### Step 4: Verify in Datadog

```bash
# Query to verify a metric exists
curl -s -G "https://api.${DATADOG_SITE}/api/v1/query" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}" \
  --data-urlencode "query=sum:service_requests.count{*}" \
  --data-urlencode "from=$(date -v-1H +%s)" \
  --data-urlencode "to=$(date +%s)"
```

**Result interpretation:**
- Returns data → metric exists with correct name
- Returns empty → check name conversion or metric not scraped

```bash
# List available metrics matching a pattern
curl -s "https://api.${DATADOG_SITE}/api/v1/metrics?q=beneficiary" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}"
```

### Step 5: Document Results

**Output format:**
```yaml
service: {service_name}
discovered_at: {timestamp}

counters:
  - prometheus: service_requests_total
    datadog: service_requests.count
    tags: [service, env, operation]

timers:
  - prometheus: service_duration_seconds
    datadog: service_duration_seconds
    suffixes: [_max, .count, .sum]
    tags: [service, env]

gauges:
  - prometheus: jvm_memory_used_bytes
    datadog: jvm_memory_used_bytes
    tags: [area, id]

trace_metrics:  # APM auto-generated
  - trace.http.request.hits
  - trace.http.request.hits.by_http_status
  - trace.aws.http.hits (SQS)
```

## Reasoning Guides

### Why Prometheus and Datadog names differ

Datadog's OpenMetrics integration converts suffixes for consistency:
- `_total` suffix → `.count` (counters are cumulative)
- `_max` suffix → stays `_max` (timer maximums)
- `_bucket` suffix → `.bucket` (histograms)

This is by design — Datadog normalizes metrics from multiple sources.

### Why trace metrics exist without instrumentation

Datadog APM auto-generates trace metrics for HTTP and messaging:
- `trace.http.request.*` — from HTTP instrumentation
- `trace.aws.http.*` — from AWS SDK instrumentation

These don't appear in Prometheus — they're Datadog-native.

## Common Failure Modes

| Failure | Symptom | Cause | Fix |
|---------|---------|-------|-----|
| Wrong name | No data in Datadog | Used Prometheus name directly | Apply conversion rules |
| Missing tags | Query too broad/narrow | Assumed tags exist | Verify from Prometheus output |
| No metrics | Empty Prometheus output | Service not instrumented | Use prometheus-instrumentation skill |

## API Reference

| Operation | Method | Endpoint |
|-----------|--------|----------|
| Query metrics | GET | `/api/v1/query?query=...&from=...&to=...` |
| List metrics | GET | `/api/v1/metrics?q={pattern}` |
| Get metric metadata | GET | `/api/v1/metrics/{metric_name}` |

## Related Skills

- `prometheus-instrumentation` — add metrics if none exist
- `creating-datadog-dashboards` — use discovered metrics
