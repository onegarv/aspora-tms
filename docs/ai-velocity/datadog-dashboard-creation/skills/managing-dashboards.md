---
name: managing-dashboards
description: >
  Lists, retrieves, updates, and validates Datadog dashboards via API. Use when asked to
  "list dashboards", "show my dashboards", "update dashboard", "validate dashboard JSON".
  Do NOT use for creating new dashboards (use creating-datadog-dashboards).
---

# Managing Dashboards

## Prerequisites

Ensure credentials are loaded:
```bash
source .env  # Contains DATADOG_API_KEY, DATADOG_APP_KEY, DATADOG_SITE
```

## Domain Model

Dashboards are living artifacts that evolve with services. Management operations
include listing, retrieving for modification, updating, and validating.

**Key invariants:**
- Dashboard IDs are immutable references
- Updates replace the entire dashboard (not partial patches)
- Validation must check syntax AND semantic correctness

## Workflow

### List Dashboards

```bash
curl -s "https://api.${DATADOG_SITE}/api/v1/dashboard" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}" \
  | jq '.dashboards[] | {id, title, url}'
```

**Format output as:**
```
| # | Title | ID |
|---|-------|----|
| 1 | Service A - Performance | abc-123 |
| 2 | War Room - Health | xyz-456 |
```

### Get Dashboard

```bash
curl -s "https://api.${DATADOG_SITE}/api/v1/dashboard/{dashboard_id}" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}"
```

**Use cases:**
- Retrieve for modification
- Export for backup
- Analyze widget structure

### Update Dashboard

**Workflow:**
1. Get current dashboard:
```bash
curl -s "https://api.${DATADOG_SITE}/api/v1/dashboard/{id}" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}" > dashboard-backup.json
```

2. Modify widgets/variables as needed in the JSON

3. Push update:
```bash
curl -s -X PUT "https://api.${DATADOG_SITE}/api/v1/dashboard/{id}" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}" \
  -H "Content-Type: application/json" \
  -d @{service}-dashboard.json
```

4. Verify the update was applied

**Warning:** Update replaces entire dashboard. Always get current state first.

### Delete Dashboard

```bash
curl -s -X DELETE "https://api.${DATADOG_SITE}/api/v1/dashboard/{id}" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DATADOG_APP_KEY}"
```

**Warning:** This is irreversible. Always backup before deleting.

### Validate Dashboard JSON

**Check these rules in order:**

1. **Schema validation**
   - Required: `title`, `widgets`, `layout_type`
   - Template variables have `name` and `prefix`

2. **Query validation**
   - Every query has `data_source: "metrics"`
   - No Prometheus regex (`=~`)
   - Filter order: tags before template variables

3. **Widget validation**
   - All IDs unique
   - Legends enabled for `by {}` queries
   - Height increased for widgets with legends

4. **Best practice validation**
   - Documentation note present
   - SLO markers on error/latency widgets
   - Four Golden Signals structure

**Output validation results:**
```
Validation: dashboard.json

Schema: PASSED
Queries: 2 issues
  - Widget 1005: Missing data_source
  - Widget 1012: Wrong filter order
Widgets: PASSED
Best Practices: 1 warning
  - Missing Saturation section

Summary: 2 errors, 1 warning
```

## Mechanical Rules

- **ALWAYS** get current dashboard before updating — prevents data loss
- **ALWAYS** verify after update — confirms changes applied
- **NEVER** delete without explicit user confirmation

## API Reference

| Operation | Method | Endpoint |
|-----------|--------|----------|
| List | GET | `/api/v1/dashboard` |
| Get | GET | `/api/v1/dashboard/{id}` |
| Update | PUT | `/api/v1/dashboard/{id}` |
| Delete | DELETE | `/api/v1/dashboard/{id}` |
