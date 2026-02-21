---
name: create-dashboard
description: Create a Datadog dashboard for a service following SRE principles
usage: /create-dashboard [service-name]
---

# Create Dashboard Command

## Usage

```
/create-dashboard                    # Interactive mode
/create-dashboard goblin-service     # Create dashboard for specific service
```

## Workflow

### Step 1: Service Identification
If service name not provided, prompt user or detect from current directory.

### Step 2: Metric Discovery
Load `skills/metric-discovery.md` and:
1. Analyze codebase for metrics
2. Query Prometheus endpoint (if available)
3. Verify metrics in Datadog via MCP

### Step 3: Dashboard Design
Load `skills/dashboard-creation.md` and:
1. Design Four Golden Signals layout
2. Create widget definitions
3. Configure template variables

### Step 4: Dashboard Creation
Using Datadog MCP:
```
Tool: datadog.create_dashboard
Parameters:
  - title: "{Service Name} - Application Performance"
  - description: "SRE dashboard following Four Golden Signals and RED method"
  - layout_type: "ordered"
  - template_variables: [...]
  - widgets: [...]
```

### Step 5: Verification
```
Tool: datadog.get_dashboard
Parameters:
  - dashboard_id: {returned_id}
```

## Output

```
Dashboard created successfully!

Title: Goblin Service - Application Performance
ID: abc-123-def
URL: https://app.datadoghq.com/dashboard/abc-123-def

Widgets: 15
- Health Summary: 6 widgets
- Traffic: 2 widgets
- Errors: 3 widgets
- Latency: 2 widgets
- Infrastructure: 2 widgets
```

## Fallback Mode

If MCP is unavailable:
1. Generate dashboard JSON
2. Save to `{service}-dashboard.json`
3. Provide import instructions

```
MCP unavailable. Dashboard JSON saved to goblin-service-dashboard.json

To import manually:
1. Go to Datadog → Dashboards → New Dashboard
2. Click gear icon → Import JSON
3. Paste contents of goblin-service-dashboard.json
```
