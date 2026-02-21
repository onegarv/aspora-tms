# Datadog Dashboard Creation Plugin

Production-grade Datadog dashboards following Google SRE Four Golden Signals and RED method.

## SAGE Compliance

| Principle | Implementation |
|-----------|----------------|
| **Scoped** | One domain: Datadog dashboards. Clear triggers and exclusions. |
| **Adaptive** | Explains WHY (reasoning guides), not just HOW (steps). |
| **Gradual** | Main SKILL.md <200 lines. Guardrails loaded on demand. |
| **Evaluated** | Tested with real dashboards. Query patterns verified. |

## Skills

| Skill | Triggers | Purpose |
|-------|----------|---------|
| `creating-datadog-dashboards` | "create dashboard", "build observability" | Create new dashboards |
| `discovering-metrics` | "find metrics", "what metrics exist" | Discover metrics before dashboard creation |
| `managing-dashboards` | "list dashboards", "validate dashboard" | List, update, validate dashboards |

## Setup

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Fill in your Datadog credentials:
```bash
# .env
DATADOG_API_KEY=your-api-key      # Organization Settings → API Keys
DATADOG_APP_KEY=your-app-key      # Organization Settings → Application Keys
DATADOG_SITE=datadoghq.eu         # or datadoghq.com for US
```

3. Load credentials:
```bash
source .env
```

## API Operations

All operations use curl with the Datadog REST API:

```bash
# Create dashboard
curl -s -X POST "https://api.${DATADOG_SITE}/api/v1/dashboard" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}" \
  -H "Content-Type: application/json" \
  -d @{service}-dashboard.json

# Get dashboard
curl -s "https://api.${DATADOG_SITE}/api/v1/dashboard/{id}" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}"

# List dashboards
curl -s "https://api.${DATADOG_SITE}/api/v1/dashboard" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}"

# Query metrics
curl -s -G "https://api.${DATADOG_SITE}/api/v1/query" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}" \
  --data-urlencode "query=sum:metric_name{*}" \
  --data-urlencode "from=$(date -v-1H +%s)" \
  --data-urlencode "to=$(date +%s)"
```

## Directory Structure

```
datadog-dashboard-creation/
├── SKILL.md                      ← Main skill (entry point)
├── plugin.yaml                   ← Plugin configuration
├── .env.example                  ← Credentials template
├── skills/
│   ├── discovering-metrics.md    ← Metric discovery
│   ├── managing-dashboards.md    ← Dashboard operations
│   └── guardrails.md             ← Reference: syntax rules
├── commands/
│   ├── create-dashboard.md
│   ├── list-dashboards.md
│   └── validate-dashboard.md
└── examples/
    ├── war-room-system-health-dashboard.json
    └── beneficiary-service-dashboard.json
```

## Three Cognitive Layers

Following SKILL_DESIGN_PRINCIPLES.md:

### WHAT (Domain Model)
A dashboard is a single pane of glass for service health. Key invariants:
- Four Golden Signals structure
- Verified metrics only
- SLO markers on error/latency

### WHY (Philosophy)
- Four Golden Signals catch 95% of issues
- "No data" widgets erode trust
- Direct API calls are simple and portable

### HOW (Workflow)
1. Discover metrics (verify before use)
2. Design structure (Four Golden Signals)
3. Build queries (Datadog syntax)
4. Create dashboard (curl to API)
5. Verify (validation loop)

## Quick Start

```
User: Create a dashboard for beneficiary-service

Agent:
1. Reads service code → finds MetricsUtil.java
2. Extracts metric names and tags
3. Designs Four Golden Signals layout
4. Generates {service}-dashboard.json
5. Creates via: curl -X POST .../api/v1/dashboard -d @dashboard.json
6. Returns dashboard URL
```

## References

- [Aspora Cortex](https://github.com/Vance-Club/aspora-cortex) — SAGE principles & skill design guide
- [Google SRE Book](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Datadog API Docs](https://docs.datadoghq.com/api/latest/dashboards/)
