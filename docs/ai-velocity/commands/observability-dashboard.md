You are creating a Datadog dashboard for a service. Follow these phases IN ORDER.
Do not skip phases. Each phase has a GATE — complete it before proceeding.

## Pre-Flight: Decision Context

Before starting, check for these files in the current project root (read them if they exist, skip if not):
1. `DECISIONS.md` — prior decisions for this project. Do NOT re-decide what's already settled.
2. `GUARDRAILS.md` — anti-patterns from past incidents. These are hard constraints.

Also check: $HOME/code/aspora/ai-velocity/datadog-dashboard-creation/ for DECISIONS.md and GUARDRAILS.md

## Phase 0: Skill Guardrails (MANDATORY — Read First)

Read the guardrails file at $HOME/code/aspora/ai-velocity/datadog-dashboard-creation/skills/guardrails.md

Key rules to internalize:
- NEVER add a metric to a dashboard without verifying it exists first
- Every query MUST include `"data_source": "metrics"`
- Filter order: tags FIRST, then template variables
- Widget IDs must be unique, starting from 1000

## Phase 1: Discover Metrics (NEVER SKIP)

Before designing any dashboard:
- Query the Prometheus endpoint (`/actuator/prometheus` or `/metrics`) to get real metric names
- OR use the Datadog API to list available metrics for the service
- Read MetricsUtil.java / metrics package to understand what's instrumented
- Classify each metric by Golden Signal: Latency, Traffic, Errors, Saturation

GATE: Present a list of verified, real metric names grouped by signal. Do NOT proceed with assumed or fabricated metric names.

## Phase 2: Design Dashboard Structure

Follow the Four Golden Signals layout:

```
┌─────────────────────────────────────────────────┐
│ Row 1: TRAFFIC — Request rate, throughput        │
├─────────────────────────────────────────────────┤
│ Row 2: ERRORS — Error rate, error breakdown      │
├─────────────────────────────────────────────────┤
│ Row 3: LATENCY — p50, p95, p99 percentiles       │
├─────────────────────────────────────────────────┤
│ Row 4: SATURATION — CPU, memory, connection pools │
└─────────────────────────────────────────────────┘
```

For each section, decide:
- Widget type (timeseries, query_value, toplist, heatmap)
- Time window
- SLO threshold markers (error rate, latency targets)

GATE: Present the dashboard layout with widget descriptions. Confirm with user.

## Phase 3: Build Queries

Read the skill reference for Datadog query syntax:
$HOME/code/aspora/ai-velocity/datadog-dashboard-creation/SKILL.md

For each widget, build the query following these rules:
- [ ] Prometheus metric names converted to Datadog format (dots not underscores, check actual names)
- [ ] `"data_source": "metrics"` in every query
- [ ] Filters: tags first (`service:myservice`), then template variables (`$env`)
- [ ] Aggregation appropriate to metric type (avg for latency, sum for counters, max for saturation)
- [ ] SLO markers on error and latency widgets

## Phase 4: Generate Dashboard JSON

Create the complete dashboard JSON with:
- [ ] Unique widget IDs starting from 1000
- [ ] Template variables for environment, service, region
- [ ] Proper layout (widget positions and dimensions)
- [ ] Legend configuration on all timeseries widgets
- [ ] Title and description

Push to Datadog via API or output the JSON file.

## Phase 5: Verification

| Check | Status |
|-------|--------|
| All metrics verified to exist before adding? | [ ] |
| `data_source: metrics` in every query? | [ ] |
| Filter order correct (tags first)? | [ ] |
| All 4 Golden Signals have widgets? | [ ] |
| SLO threshold markers on error/latency? | [ ] |
| Widget IDs unique? | [ ] |
| Template variables working? | [ ] |

GATE: All checks must pass. If any metric was not verified, remove it.
