---
name: validate-dashboard
description: Validate a dashboard JSON against Datadog schema and best practices
usage: /validate-dashboard [file.json]
---

# Validate Dashboard Command

## Usage

```
/validate-dashboard                      # Validate dashboard.json in current dir
/validate-dashboard my-dashboard.json    # Validate specific file
```

## Validation Checks

### 1. Schema Validation
- Valid JSON structure
- Required fields present (title, widgets, layout_type)
- Widget definitions valid

### 2. Query Validation
- `data_source: "metrics"` present in all queries
- No Prometheus regex syntax (`=~`)
- Correct filter order (tags before template variables)
- No OR/AND mixed with commas

### 3. Metric Name Validation
- Prometheus → Datadog conversion applied
- `_total` → `.count`
- `_max` stays as `_max`

### 4. Widget Validation
- Unique widget IDs
- Legends enabled for `by {}` queries
- Appropriate height for widgets with legends

### 5. Best Practice Validation
- Template variables defined
- SLO markers present
- Documentation note at top
- Four Golden Signals structure

## Output

```
Validating: goblin-service-dashboard.json

Schema Validation: PASSED
Query Validation: 2 issues found
  - Widget ID 1005: Missing data_source field
  - Widget ID 1012: Wrong filter order in query

Metric Names: PASSED
Widget Structure: 1 issue found
  - Widget ID 1008: Has "by {operation}" but no legend

Best Practices: 1 warning
  - Missing Saturation section

Summary: 3 issues, 1 warning

To auto-fix issues: /validate-dashboard --fix
```

## Auto-Fix Mode

```
/validate-dashboard my-dashboard.json --fix
```

Automatically fixes:
- Adds missing `data_source: "metrics"`
- Reorders filters (tags before template vars)
- Adds legends to `by {}` widgets
- Generates unique widget IDs

Does NOT auto-fix:
- Metric name conversion (requires verification)
- Missing SLO markers (requires threshold values)
- Structure changes (requires user decision)
