---
name: list-dashboards
description: List existing Datadog dashboards
usage: /list-dashboards [filter]
---

# List Dashboards Command

## Usage

```
/list-dashboards                  # List all dashboards
/list-dashboards goblin           # Filter by name containing "goblin"
/list-dashboards --service=*      # List service-specific dashboards
```

## Workflow

### Step 1: Fetch Dashboards
```
Tool: datadog.list_dashboards
```

### Step 2: Filter (if provided)
Apply filter to title field.

### Step 3: Format Output

```
Found 5 dashboards:

| # | Title | ID | Modified |
|---|-------|----|---------|
| 1 | Goblin Service - Performance | abc-123 | 2024-02-14 |
| 2 | War Room - System Health | xyz-456 | 2024-02-13 |
| 3 | Falcon API - Performance | def-789 | 2024-02-10 |
| 4 | App Server - Performance | ghi-012 | 2024-02-08 |
| 5 | FX API - Performance | jkl-345 | 2024-02-05 |

To view a dashboard: /get-dashboard abc-123
```

## Fallback Mode

If MCP unavailable:
```
MCP connection failed. Cannot list dashboards.

Manual alternative:
1. Go to https://app.datadoghq.com/dashboard/lists
2. Use search to filter dashboards
```
