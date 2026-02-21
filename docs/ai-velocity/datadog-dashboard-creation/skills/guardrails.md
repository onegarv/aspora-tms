---
name: dashboard-guardrails
description: >
  Safety guardrails and syntax rules for Datadog dashboard operations. Reference when
  building queries or debugging dashboard issues. This is a reference file, not a
  standalone skill.
---

# Dashboard Guardrails

**Load this reference when you need detailed syntax rules or valid values.**

## MCP Tools

| Tool | Purpose |
|------|---------|
| `datadog.list_dashboards` | List all dashboards |
| `datadog.get_dashboard` | Get by ID |
| `datadog.create_dashboard` | Create new |
| `datadog.update_dashboard` | Update existing |
| `datadog.query_metrics` | Verify metrics |

**Forbidden:**
- Hallucinated tool names
- Direct HTTP calls to Datadog API
- Creating widgets for unverified metrics

## Query Syntax Rules

### Required Field

Every query MUST include:
```json
{"data_source": "metrics"}
```

### Filter Order

Tags FIRST, then template variables:
```
CORRECT: {outcome:error,$env,$service}
WRONG:   {$env,$service,outcome:error}
```

### Prometheus → Datadog Conversion

| Prometheus | Datadog |
|------------|---------|
| `metric_total` | `metric.count` |
| `metric_max` | `metric_max` |
| `metric_count` | `metric.count` |
| `metric_sum` | `metric.sum` |

### Forbidden Syntax

```
NO: status=~"5.."     (Prometheus regex)
NO: {status:(500 OR 501),$env}  (OR with commas)
```

## Widget Structure

### Unique IDs
- Start from 1000
- Increment for each widget
- Nested widgets need unique IDs too

### Legend Configuration

For queries with `by {tag}`:
```json
{
  "show_legend": true,
  "legend_layout": "auto",
  "legend_columns": ["avg", "min", "max", "value", "sum"],
  "time": {}
}
```

NOT nested in `legend` object — top-level only.

### Height Guidelines

| Widget Type | Standard | With Legend |
|-------------|----------|-------------|
| Timeseries | 2 | 3 |
| Note | 2-4 | N/A |
| Group | Sum of children + 1 | N/A |

## Valid Values

### Layout Types
- `ordered` (recommended)
- `free`

### Widget Types
- `timeseries`, `note`, `group`, `query_value`, `toplist`, `heatmap`

### Color Palettes
- `green` (success), `red` (errors), `blue` (latency), `orange` (warnings)

### Marker Types
- `warning dashed` — warning threshold
- `error solid` — critical threshold

## SLO Thresholds (Defaults)

| Metric | Warning | Critical |
|--------|---------|----------|
| Error Rate | 1% | 5% |
| Latency p95 | 100ms | 500ms |
| CPU | 70% | 90% |
| Memory | 80% | 90% |

## Validation Checklist

Before creating dashboard:
- [ ] All metrics verified
- [ ] All tags verified
- [ ] `data_source` in all queries
- [ ] Filter order correct
- [ ] Name conversion applied
- [ ] Unique widget IDs
- [ ] Legends for `by {}` queries
- [ ] SLO markers included
