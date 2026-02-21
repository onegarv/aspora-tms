---
name: alert-generation
description: "Generate production-grade alerts following SRE best practices. Analyzes metrics from codebase and dashboard, determines static vs anomaly-based alerts, calculates thresholds, assigns priorities, and creates Datadog monitors."
version: 1.0.0
author: Eventbus Team
---

# Intelligent Alert Generation with SRE Best Practices

## Purpose
This skill enables AI agents to generate production-grade alerts that follow Google SRE principles. It builds on top of the prometheus-instrumentation and datadog-dashboard-creation skills to:
- Analyze metrics from MetricsUtil and dashboard JSON
- Determine static vs anomaly-based alert types
- Calculate appropriate thresholds using SRE best practices
- Assign priorities (P0-P3) based on impact
- Generate Datadog monitor configurations
- Push alerts to Datadog via API

## When to Trigger This Skill
- When creating alerts for a new service
- When user requests: "Generate alerts for this service", "Create monitoring alerts", "Set up alerting"
- After metrics instrumentation is complete (use @prometheus-instrumentation skill first)
- After dashboard is created (use @datadog-dashboard-creation skill first)
- When improving existing alert configurations

## Critical: Automatic Codebase Analysis

**ALWAYS analyze the codebase and metrics automatically before generating alerts.**

1. **Read the codebase automatically**:
   - Read service classes to understand architecture
   - Check MetricsUtil to discover all available metrics
   - Read dashboard JSON to understand metric structure
   - Understand service flows and dependencies

2. **Discover metrics automatically**:
   - Read MetricsUtil.java to identify all metrics
   - Check dashboard JSON for metric names and tags
   - Understand metric types (counters, timers, gauges)
   - Map metrics to SRE Four Golden Signals

3. **Do NOT ask the user to list metrics**:
   - All metrics are discoverable from codebase
   - Metric names are in MetricsUtil
   - Tags are defined in the code
   - Service architecture is evident from code structure

## Core Principles

### 1. SRE Alerting Philosophy
Following Google SRE Book principles:
- **Alert on Symptoms, Not Causes**: Alert on user-visible symptoms (high latency, errors) not internal causes
- **Alert Fatigue Prevention**: Only alert on actionable issues
- **Multi-Window Strategy**: Use multiple time windows to reduce noise
- **Burn Rate**: Alert based on error budget burn rate, not just error rate

### 2. Alert Types

#### Static Alerts (Threshold-Based)
**Use when**: Metric has predictable baseline, clear SLO targets, or known thresholds

**Examples**:
- Error rate > 1% for 5 minutes
- Latency p95 > 500ms for 10 minutes
- Queue depth > 1000

**Threshold Calculation**:
- **Baseline Analysis**: Calculate p50, p75, p95, p99 from historical data
- **SLO-Based**: Use SLO targets (e.g., 99.9% availability = 0.1% error rate)
- **Multiplier Approach**: Threshold = baseline * multiplier (e.g., 2x for warning, 5x for critical)
- **Percentile-Based**: Use p95 or p99 as baseline, alert at 2-3x

#### Anomaly-Based Alerts
**Use when**: Metric has high variance, no clear baseline, or seasonal patterns

**Examples**:
- Sudden drop in throughput (50% decrease)
- Unusual error pattern (new error types)
- Latency spike (3 standard deviations above mean)

**Anomaly Detection Methods**:
- **Statistical**: Z-score, moving average with standard deviation
- **Time Series**: ARIMA, exponential smoothing
- **Machine Learning**: Isolation Forest, LSTM for time series
- **Change Point Detection**: Detect sudden shifts in behavior

### 3. Alert Priority Levels

#### P0 - Critical (Page Immediately)
**Criteria**:
- Service completely down (error rate > 50%)
- User-facing functionality broken
- Data loss or corruption risk
- Security breach

**Response Time**: < 5 minutes
**Notification**: PagerDuty/Slack @channel

#### P1 - High (Page During Business Hours)
**Criteria**:
- Significant degradation (error rate 5-50%)
- High latency (> 2x normal)
- Partial service outage
- SLO violation imminent

**Response Time**: < 15 minutes
**Notification**: Slack channel

#### P2 - Medium (Ticket/Email)
**Criteria**:
- Minor degradation (error rate 1-5%)
- Latency increase (1.5-2x normal)
- Non-critical feature issues
- Warning signs of future problems

**Response Time**: < 4 hours
**Notification**: Email/Slack thread

#### P3 - Low (Log for Review)
**Criteria**:
- Minor fluctuations
- Non-user-facing issues
- Informational alerts
- Trend analysis

**Response Time**: Next business day
**Notification**: Dashboard/logs only

### 4. Alert Design Best Practices

#### Reduce False Positives at Design Time
- **Multi-Window**: Require condition to be true for multiple time windows (2-3 consecutive windows)
- **Rate of Change**: Alert on rate of change, not absolute values when appropriate
- **Relative Thresholds**: Use relative changes (e.g., 2x increase) instead of absolute for high-variance metrics
- **Exclude Known Patterns**: Consider time-of-day restrictions for non-critical alerts
- **Cooldown Periods**: Set renotify_interval to prevent re-alerting within X minutes

## Implementation Steps

### Step 0: Analyze Codebase and Metrics (Automatic)

1. **Read MetricsUtil.java**:
   - Extract all metric names
   - Understand metric types (counter, timer, gauge)
   - Map to SRE signals (Traffic, Errors, Latency, Saturation)

2. **Read Dashboard JSON**:
   - Understand metric structure and organization
   - Follow the same Four Golden Signals organization as dashboard
   - Note SLO thresholds already defined in dashboard markers
   - Understand which metrics are L0 (service-level) vs L1 (feature-level)

3. **Understand Service Architecture**:
   - Identify critical paths
   - Map dependencies
   - Understand failure modes

### Critical: Alert Deduplication and Grouping

**ALWAYS prevent duplicate alerts for the same problem. Follow these rules:**

1. **One Alert Per Signal Per Level**:
   - Create ONE alert for L0 (service-level) metrics
   - Create ONE alert for L1 (feature-level) metrics only if it provides unique value
   - DO NOT create separate alerts for each event type if they all indicate the same problem

2. **Alert Hierarchy**:
   ```
   L0 Service-Level Alert (P0/P1) → Covers all events
   ↓
   L1 Feature-Level Alert (P1/P2) → Only if needed for debugging specific events
   ↓
   L2 Sub-Event Alert → SKIP (too granular, use parent alert)
   ```

3. **Group Related Metrics**:
   - If multiple metrics indicate the same problem, create ONE composite alert
   - Example: Don't create separate alerts for `service.errors.total` and `service.events.failed.total` if they're correlated
   - Use formulas to combine related metrics into one alert

4. **Skip Redundant Alerts**:
   - If L0 alert already covers the issue, skip L1 alerts for the same signal
   - If aggregate metric exists, skip individual component alerts
   - If parent metric covers children, skip child metrics

5. **Priority-Based Filtering**:
   - Focus on P0 and P1 alerts (critical and high priority)
   - Only create P2 alerts if they provide unique debugging value
   - Skip P3 alerts entirely (informational only)

**Decision Tree for Alert Creation**:
```
FOR each metric:
  IF metric is L0 (service-level):
    IF signal already has L0 alert:
      → SKIP (duplicate)
    ELSE:
      → CREATE P0/P1 alert
      
  ELSE IF metric is L1 (feature-level):
    IF L0 alert already covers this signal:
      → SKIP (redundant with L0)
    ELSE IF metric provides unique debugging value:
      → CREATE P1/P2 alert
    ELSE:
      → SKIP (not actionable)
      
  ELSE IF metric is L2 (sub-event level):
    → SKIP (too granular, use parent)
```

### Alert Grouping Examples

**Example 1: Error Metrics**
- ✅ CREATE: `service.errors.total` (L0) - covers all errors
- ✅ CREATE: `service.events.failed.total` (L1) - only if needed for event-specific debugging
- ❌ SKIP: `service.sub_events.failed.total` (L2) - too granular

**Example 2: Latency Metrics**
- ✅ CREATE: `service.event.duration` (L0) - overall latency
- ❌ SKIP: `service.event.duration{event_name:user}` - redundant, use L0 + filter in dashboard
- ✅ CREATE: `service.external.duration` (L1) - unique signal for external dependencies

**Example 3: Throughput Metrics**
- ✅ CREATE: `service.events.processed.total` (L0) - overall throughput
- ❌ SKIP: `service.events.processed.total{event_name:*}` - redundant, use L0
- ❌ SKIP: `service.sub_events.processed.total` - too granular

### Step 1: Organize by Four Golden Signals (Same as Dashboard)

**CRITICAL: Follow the exact same organization as the dashboard.**

Organize alerts by the Four Golden Signals, matching the dashboard structure:

1. **Traffic (Rate)** - Throughput metrics
2. **Errors** - Error rates and failures
3. **Latency (Duration)** - Performance metrics
4. **Saturation** - Resource utilization

**Map each metric to a signal and level (L0 or L1):**
- L0: Service-level aggregates (e.g., `service.events.processed.total{service}`)
- L1: Feature-level with tags (e.g., `service.events.processed.total{service,event_name}`)
- L2: Sub-event level (SKIP - too granular)

### Step 2: Apply Alert Deduplication Rules

**CRITICAL: Prevent duplicate alerts for the same problem.**

**Rule 1: One Alert Per Signal Per Level**
- Maximum ONE alert per signal (Traffic/Errors/Latency/Saturation) per level (L0/L1)
- If multiple metrics map to same signal+level, choose the most representative one

**Rule 2: L0 Takes Precedence**
- If L0 alert exists for a signal, L1 alerts for same signal are OPTIONAL
- Only create L1 if it provides unique debugging value (e.g., specific event type failures)

**Rule 3: Aggregate Over Individual**
- Prefer aggregate metrics over individual components
- Example: `service.errors.total` over `service.errors.total{error_type:deserialization}`

**Rule 4: Composite Alerts for Related Metrics**
- If multiple metrics indicate same problem, combine them in one alert
- Use formulas: `(errors / (success + errors)) * 100` instead of separate alerts

**Rule 5: Skip Low-Value Alerts**
- Skip P2/P3 alerts unless they provide unique debugging value
- Skip L2 (sub-event) alerts entirely
- Skip alerts that are covered by parent metrics

### Step 3: Categorize Metrics by Alert Type

For each metric (after deduplication), determine:
1. **Alert Type**: Static or Anomaly
2. **Priority**: P0, P1, or SKIP

**Decision Matrix**:
```
IF metric has SLO target:
    → Static alert at SLO threshold
ELSE IF metric has stable baseline (< 20% variance):
    → Static alert at 2-3x baseline
ELSE IF metric has high variance (> 50%):
    → Anomaly alert (statistical or ML-based)
ELSE:
    → SKIP (no clear pattern)
```

### Step 4: Calculate Static Alert Thresholds

#### For Error Rate Metrics
```yaml
Warning Threshold: SLO target * 0.5 (e.g., 0.5% if SLO is 1%)
Critical Threshold: SLO target (e.g., 1%)
Evaluation Window: 5 minutes
Required Duration: 2 consecutive windows
```

#### For Latency Metrics
```yaml
Warning Threshold: p95 baseline * 1.5
Critical Threshold: p95 baseline * 2.0
Evaluation Window: 10 minutes
Required Duration: 2 consecutive windows
```

#### For Throughput Metrics
```yaml
Warning Threshold: baseline * 0.5 (50% drop)
Critical Threshold: baseline * 0.2 (80% drop)
Evaluation Window: 5 minutes
Required Duration: 1 window
```

#### For Saturation Metrics
```yaml
Warning Threshold: baseline * 2
Critical Threshold: baseline * 5
Evaluation Window: 5 minutes
Required Duration: 2 consecutive windows
```

### Step 5: Design Anomaly Detection Alerts

#### Statistical Anomaly Detection
```yaml
Method: Z-score
Threshold: |z-score| > 3
Window: 30 minutes rolling
Baseline: 7-day moving average
```

#### Change Point Detection
```yaml
Method: CUSUM or PELT
Sensitivity: Medium
Window: 1 hour
Min Change: 20% relative change
```

### Step 6: Final Alert Selection and Prioritization

**Before generating alerts, apply final filtering:**

1. **Remove Duplicates**: Ensure only one alert per signal+level combination
2. **Prioritize**: Focus on P0 and P1 alerts (8-10 alerts total)
3. **Verify Uniqueness**: Each alert should detect a unique problem
4. **Check Coverage**: Ensure all Four Golden Signals are covered

**Final Alert Count Guidelines**:
- **P0 Alerts**: 2-3 (service-level critical issues)
- **P1 Alerts**: 5-7 (feature-level high-priority issues)
- **P2 Alerts**: 0 (skip unless absolutely necessary)
- **Total**: 8-10 alerts maximum (one per unique problem)

**Alert Selection Priority**:
1. L0 service-level alerts for each signal (Traffic, Errors, Latency, Saturation) - REQUIRED
2. L1 feature-level alerts only if they detect unique problems not covered by L0
3. Anomaly detection for high-variance metrics without clear baseline
4. Composite alerts for related metrics (combine instead of separate)

**Final Priority Assignment**:

1. **P0**: L0 service-level metrics with critical thresholds (error rate > 1%, service down)
2. **P1**: L0 service-level with warning thresholds OR L1 feature-level with critical thresholds
3. **SKIP**: P2/P3 alerts, L2 sub-event metrics, redundant alerts, low-priority metrics

### Step 7: Generate Datadog Monitor Configurations

Generate Datadog monitor JSON configurations for each alert. Datadog monitors support both static thresholds and anomaly detection.

**CRITICAL: Datadog API Format Requirements**

The Datadog API has strict format requirements that differ from dashboard queries. Follow these rules exactly:

#### 1. Monitor Type

- **Regular alerts**: Use `"type": "query alert"` (NOT `"metric alert"`)
- **Anomaly detection**: Use `"type": "query alert"` with `anomalies()` function

#### 2. Query Format - CRITICAL RULES

**Rule 1: Query MUST include comparison operator**
- ✅ CORRECT: `"query": "avg(last_5m):sum:metric{tags} >= 10"`
- ❌ WRONG: `"query": "avg(last_5m):sum:metric{tags}"` (no comparison operator)

**Rule 2: Query threshold MUST match critical threshold**
- If `options.thresholds.critical = 10`, query must be `>= 10` (not `>= 5`)
- If `options.thresholds.critical = 0.5`, query must be `>= 0.5` (not `>= 0.1`)
- The query threshold is validated against the **critical** threshold, not warning

**Rule 3: Use simple metric queries**
- ✅ CORRECT: `avg(last_5m):sum:metric{tags} >= 10`
- ✅ CORRECT: `avg(last_5m):max:metric{tags} >= 0.5`
- ❌ WRONG: Complex formulas in query (use dashboard for formulas)
- ❌ WRONG: `.as_rate()` in query (Datadog API may reject it)

**Rule 4: Anomaly detection format**
- ✅ CORRECT: `anomalies(avg:metric{tags}.as_rate(), 'agile', 2, direction='both', interval=60, alert_window='last_15m', count_default_zero='true', seasonality='hourly') >= 1`
- Must include all parameters shown above
- Query threshold must match `critical` threshold (usually `>= 1`)

#### 3. Options Configuration

**CRITICAL: threshold_windows is ONLY for anomaly monitors**
- ✅ Anomaly monitors: Include `threshold_windows`
- ❌ Regular alerts: DO NOT include `threshold_windows` (will cause error)

**CRITICAL: on_missing_data values**
- Valid values: `"show_no_data"`, `"show_and_notify_no_data"`, `"resolve"`, `"default"`
- ❌ INVALID: `"notify"` (does not exist)

**CRITICAL: notify_no_data is deprecated**
- ❌ DO NOT use `"notify_no_data": true` with `on_missing_data`
- ✅ Use `"on_missing_data": "show_and_notify_no_data"` instead
- Remove `notify_no_data` field entirely when using `on_missing_data`

**Other options:**
- `require_full_window`: Use `false` (not `true`) for most alerts
- `include_tags`: Use `false` (default)
- `evaluation_delay`: Use `0` (no delay)

#### 4. Common Pitfalls and Solutions

**Pitfall 1: "Alert threshold does not match that used in the query"**
- **Cause**: Query threshold doesn't match `options.thresholds.critical`
- **Fix**: Ensure query `>= X` matches `critical: X` in thresholds
- **Example**: If `critical: 10`, query must be `>= 10` (not `>= 5`)

**Pitfall 2: "threshold_windows is only used with anomalies monitors"**
- **Cause**: Added `threshold_windows` to regular query alert
- **Fix**: Remove `threshold_windows` from all non-anomaly monitors
- **Keep**: `threshold_windows` only for monitors using `anomalies()` function

**Pitfall 3: "Invalid on_missing_data value"**
- **Cause**: Used invalid value like `"notify"`
- **Fix**: Use one of: `"show_no_data"`, `"show_and_notify_no_data"`, `"resolve"`, `"default"`

**Pitfall 4: "notify_no_data option is deprecated"**
- **Cause**: Used both `notify_no_data` and `on_missing_data`
- **Fix**: Remove `notify_no_data` field, use only `on_missing_data`

**Pitfall 5: "The value provided for parameter 'query' is invalid"**
- **Cause**: Query format doesn't match Datadog API requirements
- **Fix**: 
  - Ensure query includes comparison operator (`>=`, `<=`, `>`)
  - Remove complex formulas (Datadog API doesn't support them)
  - Use simple metric queries: `avg(last_5m):sum:metric{tags} >= threshold`

**Pitfall 6: "Duplicate of an existing monitor"**
- **Cause**: Monitor with same query/name already exists
- **Fix**: Script should check for existing monitors by name and update instead of create
- **Note**: This is expected behavior - update existing monitors

#### Datadog Monitor Structure (Correct Format)

```json
{
  "type": "query alert",
  "query": "avg(last_5m):sum:metric{tags} >= 10",
  "name": "[P0] Service - Alert Name",
  "message": "Alert message with {{value}} placeholder",
  "tags": ["service:service-name", "priority:P0", "team:platform"],
  "options": {
    "thresholds": {
      "critical": 10,
      "warning": 5,
      "critical_recovery": 8,
      "warning_recovery": 3
    },
    "require_full_window": false,
    "notify_audit": false,
    "on_missing_data": "default",
    "evaluation_delay": 0,
    "renotify_interval": 60,
    "new_host_delay": 300,
    "include_tags": false
  },
  "priority": 1
}
```

#### Static Threshold Monitor Example (Correct)

```json
{
  "type": "query alert",
  "query": "avg(last_5m):sum:service_events_failed.count{service:my-service,env:stage} >= 10",
  "name": "[P0] Service - High Error Rate",
  "message": "Error rate exceeded threshold. Current: {{value}} errors\n\n@slack-alerts-critical",
  "tags": ["service:my-service", "priority:P0", "team:platform", "signal:errors"],
  "options": {
    "thresholds": {
      "critical": 10,
      "warning": 5,
      "critical_recovery": 8,
      "warning_recovery": 3
    },
    "require_full_window": false,
    "notify_audit": false,
    "on_missing_data": "default",
    "evaluation_delay": 0,
    "renotify_interval": 60,
    "new_host_delay": 300,
    "include_tags": false
  },
  "priority": 1
}
```

#### Anomaly Detection Monitor Example (Correct)

```json
{
  "type": "query alert",
  "query": "avg(last_15m):anomalies(sum:service_events_processed.count{service:my-service,env:stage}.as_rate(), 'agile', 2, direction='both', interval=60, alert_window='last_15m', count_default_zero='true', seasonality='hourly') >= 1",
  "name": "[P0] Service - Anomalous Event Processing Rate",
  "message": "Anomaly detected in event processing rate. Score: {{value}}\n\n@slack-alerts-critical",
  "tags": ["service:my-service", "priority:P0", "team:platform", "type:anomaly"],
  "options": {
    "thresholds": {
      "critical": 1,
      "critical_recovery": 0
    },
    "threshold_windows": {
      "trigger_window": "last_15m",
      "recovery_window": "last_15m"
    },
    "require_full_window": false,
    "notify_audit": false,
    "on_missing_data": "default",
    "evaluation_delay": 0,
    "renotify_interval": 60,
    "new_host_delay": 300,
    "include_tags": false
  },
  "priority": 1
}
```

#### Service Down Monitor Example (Correct)

```json
{
  "type": "query alert",
  "query": "avg(last_5m):sum:service_events_processed.count{service:my-service,env:stage} <= 1",
  "name": "[P0] Service - No Events Processed (Service Down)",
  "message": "Service appears down - no events processed. Current: {{value}}\n\n@slack-alerts-critical",
  "tags": ["service:my-service", "priority:P0", "team:platform", "signal:traffic"],
  "options": {
    "thresholds": {
      "critical": 1
    },
    "require_full_window": false,
    "notify_audit": false,
    "on_missing_data": "show_and_notify_no_data",
    "evaluation_delay": 0,
    "renotify_interval": 60,
    "new_host_delay": 300,
    "include_tags": false
  },
  "priority": 1
}
```

#### Validation Checklist Before Creating Monitors

Before generating monitor JSON, verify:

- [ ] Query includes comparison operator (`>=`, `<=`, `>`)
- [ ] Query threshold matches `options.thresholds.critical` exactly
- [ ] Type is `"query alert"` (not `"metric alert"`)
- [ ] `threshold_windows` only present for anomaly monitors
- [ ] `on_missing_data` uses valid value (not `"notify"`)
- [ ] `notify_no_data` removed if using `on_missing_data`
- [ ] `require_full_window` is `false` (not `true`)
- [ ] No complex formulas in query (use simple metric queries)
- [ ] Recovery thresholds included if using warning/critical

### Step 7.5: Alert Routing and Notification Channels

**CRITICAL: Route alerts to the right team and channel based on metric type and priority.**

#### Alert Classification

**Classify each alert by two dimensions**:

1. **Metric Type**: Service Metrics vs Infrastructure Metrics
2. **Priority**: P0 vs P1

#### Service Metrics (Application-Level)

**Definition**: Metrics that measure application behavior, business logic, and user experience.

**Examples**:
- Error rates (`service.errors.total`, `service.events.failed.total`)
- Latency (`service.request.duration`, `service.event.processing.duration`)
- Throughput (`service.events.processed.total`, `service.requests.total`)
- Business metrics (`service.orders.processed.total`)

**Identification**:
```
IF metric_name contains "service." OR 
   metric_name starts with service-specific prefix (e.g., "eventbus_", "payment_"):
    → Classify as Service Metric
    → Route to Service Team
```

#### Infrastructure Metrics (System-Level)

**Definition**: Metrics that measure underlying infrastructure health.

**Examples**:
- CPU metrics (`container.cpu.user`, `container.cpu.system`, `jvm.cpu.recent_utilization`)
- Memory metrics (`system.mem.used`, `system.mem.total`, `jvm.memory.used`)
- JVM metrics (`jvm.gc.pause`, `jvm.threads.live`)
- Container metrics (`docker.cpu.*`, `docker.mem.*`)

**Identification**:
```
IF metric_name contains "container." OR 
   metric_name contains "system." OR 
   metric_name contains "jvm." OR 
   metric_name contains "docker.":
    → Classify as Infrastructure Metric
    → Route to Infrastructure Team
```

**Exception**: If infrastructure metric has `service` tag and indicates service-specific issue, route to **Service Team** first.

#### Service Team Mapping

**Read service team configuration** from `alerts/service-teams.yaml`:

```yaml
service_teams:
  eventbus:
    team: platform
    slack_channel: "#backend-signals"
    pagerduty_service: "eventbus-service"
    
  payment-service:
    team: payments
    slack_channel: "#payments-alerts"
    pagerduty_service: "payment-service"

infrastructure_team:
  team: platform
  slack_channel: "#infrastructure-alerts"
  pagerduty_service: "infrastructure"
```

**Extract service name** from metric tags or service name:
- From `service` tag: `service:eventbus` → `eventbus`
- From metric name: `eventbus_events_processed` → `eventbus`
- From container name: `ecs_container_name:eventbus-service` → `eventbus`

#### Notification Channel Selection

**P0 Alerts (Critical)**:
- **Notification**: PagerDuty (page on-call engineer) + Slack `#war-room`
- **Routing**:
  - Service metrics → PagerDuty service for service team
  - Infrastructure metrics → PagerDuty service for infrastructure team
- **Message**: Include `@pagerduty-{{service_name}}` and `@slack-war-room`

**P1 Alerts (High Priority)**:
- **Notification**: Slack channel only
- **Routing**:
  - Service metrics → Service team Slack channel (from `service-teams.yaml`)
  - Infrastructure metrics → Infrastructure team Slack channel
  - Fallback → `#watchtower` (company-wide P1 alerts)
- **Message**: Include `@slack-{{service_channel}}`

#### Monitor Configuration with Routing

**P0 Alert Example**:
```json
{
  "name": "[P0] {{service_name}} - High Error Rate",
  "message": "🚨 [P0] {{service_name}} - High Error Rate\n\nCurrent: {{value}} errors/sec\nThreshold: {{threshold}} errors/sec\nEnvironment: {{env}}\n\nRunbook: {{runbook_url}}\nDashboard: {{dashboard_url}}\n\n@pagerduty-{{service_name}} @slack-war-room",
  "tags": [
    "service:{{service_name}}",
    "priority:P0",
    "team:{{team_name}}",
    "signal:errors",
    "metric_type:service"  // or "infrastructure"
  ],
  "options": {
    "notify_audit": true,
    "renotify_interval": 15
  }
}
```

**P1 Alert Example**:
```json
{
  "name": "[P1] {{service_name}} - High Latency",
  "message": "⚠️ [P1] {{service_name}} - High Latency\n\nCurrent: {{value}} ms\nThreshold: {{threshold}} ms\nEnvironment: {{env}}\n\nRunbook: {{runbook_url}}\nDashboard: {{dashboard_url}}\n\n@slack-{{service_channel}}",
  "tags": [
    "service:{{service_name}}",
    "priority:P1",
    "team:{{team_name}}",
    "signal:latency",
    "metric_type:service"  // or "infrastructure"
  ],
  "options": {
    "notify_audit": false,
    "renotify_interval": 60
  }
}
```

#### Routing Decision Tree

```
FOR each alert:
    // Classify metric type
    IF metric_name contains "container." OR "system." OR "jvm." OR "docker.":
        metric_type = "infrastructure"
        team = infrastructure_team
    ELSE:
        metric_type = "service"
        team = lookup_service_team(service_name)
    
    // Select notification channel
    IF priority == P0:
        notification = "pagerduty + slack-war-room"
        pagerduty_service = team.pagerduty_service
        slack_channel = "#war-room"
    ELSE IF priority == P1:
        notification = "slack"
        slack_channel = team.slack_channel OR "#watchtower"
    
    // Configure monitor
    monitor.message = format_message(priority, team, runbook_url)
    monitor.tags = [service, priority, team, metric_type]
```

### Step 7.6: Generate Runbooks

**CRITICAL: Every alert MUST have a runbook for investigation and resolution.**

#### Runbook Generation

**Auto-generate runbook** from alert configuration:

1. **Read alert metadata**:
   - Alert name and description
   - Metric name and query
   - Threshold values
   - Priority and team

2. **Generate runbook** using template:
   - Alert overview
   - Investigation steps
   - Resolution steps
   - Prevention measures

3. **Store runbook**:
   - Location: `alerts/{{service-name}}/runbooks/{{alert-name}}.md`
   - Format: Markdown
   - Link from monitor message

#### Runbook Template

```markdown
# Runbook: [Alert Name]

## Alert Overview

**Metric**: `{{metric_name}}{tags}`
**Threshold**: `{{threshold_value}}`
**Priority**: {{priority}}
**Team**: {{team_name}}

**What it measures**: [Auto-generated from metric description]
**Why it matters**: [Auto-generated from priority and signal]
**When it fires**: [Auto-generated from threshold and evaluation window]

## Investigation Steps

### 1. Verify the Issue
- [ ] Check Datadog dashboard: {{dashboard_url}}
- [ ] Verify metric value: `{{metric_query}}`
- [ ] Check related metrics: [Auto-generated related metrics]
- [ ] Review recent deployments

### 2. Check Common Causes
- [ ] Recent code deployment
- [ ] Infrastructure changes
- [ ] Dependency service issues
- [ ] Traffic spike
- [ ] Configuration changes

### 3. Gather Context
- [ ] Check logs: `kubectl logs -f {{service_name}}`
- [ ] Check metrics: [Metrics Explorer URL]
- [ ] Check traces: [APM URL]
- [ ] Review recent changes

## Resolution Steps

### Immediate Actions
1. [Step 1 - Auto-generated based on metric type]
2. [Step 2 - Auto-generated based on signal]
3. [Step 3 - Auto-generated based on common causes]

### Rollback Procedure
If issue persists:
1. [Rollback step 1]
2. [Rollback step 2]

### Verification
- [ ] Metric returns to normal: `{{metric_query}}`
- [ ] Related metrics normalized
- [ ] User impact resolved

## Prevention

### Short-term
- [Action 1]
- [Action 2]

### Long-term
- [Action 1]
- [Action 2]

## Related Alerts
- [Related alert 1]
- [Related alert 2]

## References
- Dashboard: {{dashboard_url}}
- Documentation: [Doc URL]
- Team: {{team_slack_channel}}
```

#### Runbook Auto-Generation Rules

**For Service Metrics**:
- Investigation: Check service logs, recent deployments, dependency services
- Resolution: Rollback deployment, scale service, fix code issue
- Prevention: Add monitoring, improve error handling, add circuit breakers

**For Infrastructure Metrics**:
- Investigation: Check container resources, JVM settings, system load
- Resolution: Scale containers, adjust JVM settings, restart service
- Prevention: Right-size containers, optimize JVM, add resource limits

**For Error Rate Alerts**:
- Investigation: Check error logs, recent code changes, dependency failures
- Resolution: Rollback deployment, fix code bug, restart service
- Prevention: Improve error handling, add retries, add circuit breakers

**For Latency Alerts**:
- Investigation: Check slow queries, external dependencies, resource constraints
- Resolution: Optimize queries, scale service, add caching
- Prevention: Performance testing, query optimization, caching strategy

**For Throughput Alerts**:
- Investigation: Check consumer lag, queue depth, service health
- Resolution: Scale consumers, fix processing bottlenecks, restart service
- Prevention: Auto-scaling, queue monitoring, capacity planning

### Step 8: Push Alerts to Datadog

Use Datadog API to create monitors:

**API Endpoint**: `POST https://api.datadoghq.eu/api/v1/monitor` (EU) or `https://api.datadoghq.com/api/v1/monitor` (US)

**Headers**:
```
DD-API-KEY: <DATADOG_API_KEY>
DD-APPLICATION-KEY: <DATADOG_APP_KEY>
Content-Type: application/json
```

**Request Body**: Monitor JSON configuration from Step 7

**Response**: Monitor ID and status

**Error Handling**:
- If monitor already exists (by name), update it using `PUT /api/v1/monitor/{monitor_id}`
- Check for existing monitors by name before creating
- Validate query syntax matches Datadog API requirements
- Common errors and fixes:
  - `400 Bad Request`: Check query format, threshold matching, options configuration
  - `409 Conflict`: Monitor exists - update instead of create
  - `401 Unauthorized`: Check API keys and permissions

**Update Existing Monitor**:
- Use `PUT /api/v1/monitor/{monitor_id}` with same JSON structure
- Get monitor ID by querying monitors by name first

## Best Practices from Scale Companies

### Google SRE Approach
- **Error Budget**: Alert when error budget burn rate exceeds target
- **Multi-Window**: Use multiple time windows (1m, 5m, 15m)
- **Alert on Symptoms**: User-visible issues, not internal metrics
- **Runbook Required**: Every alert must have a runbook

### Netflix Approach
- **Atlas**: Time series database with anomaly detection
- **Alert Fatigue Prevention**: Aggressive threshold tuning
- **Canary Analysis**: Compare canary vs baseline metrics
- **Automated Remediation**: Auto-scale, circuit breakers

### Amazon Approach
- **CloudWatch Alarms**: Multi-dimensional alarms
- **Composite Alarms**: Combine multiple metrics
- **Anomaly Detection**: ML-based anomaly detection
- **Cost Optimization**: Alert on cost anomalies

## Step 9: Complete Alert Generation Workflow

### Workflow Summary

1. **Analyze MetricsUtil.java**: Extract all metrics and their types, identify L0 vs L1
2. **Read Dashboard JSON**: Follow Four Golden Signals organization, note SLO thresholds
3. **Read Service Teams Config**: Load `alerts/service-teams.yaml` for routing
4. **Organize by Signals**: Group metrics by Traffic, Errors, Latency, Saturation (same as dashboard)
5. **Apply Deduplication**: Remove redundant alerts, keep only unique problems (one per signal+level)
6. **Determine Alert Types**: Static vs Anomaly for each unique alert
7. **Calculate Thresholds**: Use SRE best practices to set thresholds
8. **Final Filtering**: Keep only P0/P1 alerts (8-10 total maximum)
9. **Classify Metrics**: Service metrics vs Infrastructure metrics
10. **Determine Routing**: Service team vs Infrastructure team
11. **Select Notification Channels**: PagerDuty (P0) vs Slack (P1)
12. **Generate Runbooks**: Auto-generate runbooks for each alert
13. **Generate Datadog Monitors**: Create complete monitor JSON configurations with routing
14. **Push to Datadog**: Use API to create monitors

### Alert Organization (Matching Dashboard)

**Signal 1: Traffic (Rate)**
- L0: `service.events.processed.total` - Overall throughput (P0 if service down)
- L0 Anomaly: Anomalous event processing rate (P0)

**Signal 2: Errors**
- L0: `service.errors.total` - Overall error rate (P0)
- L1: `service.events.failed.total` - Event failures (P1, only if unique value)

**Signal 3: Latency (Duration)**
- L0: `service.event.duration` - Overall latency (P1)
- L1: `service.external.duration` - External dependency latency (P1, unique signal)

**Signal 4: Saturation**
- L0: `service.pending.events.count` - Queue depth (P1)
- L0: `service.jobs.alert.stuck_events.count` - Stuck events (P1)

**Total**: 8-9 alerts (one per unique problem, no duplicates)

### Example: Complete Alert Set for Notification Service

See `alert-generation/COMPLETE_EXAMPLE.md` for full monitor configurations.

**Summary** (After Deduplication):
- **P0 Alerts (2-3)**: Error rate, service down, anomalous throughput
- **P1 Alerts (5-7)**: Latency, saturation, external dependencies
- **Total**: 8-10 alerts maximum (one per unique problem, no duplicates)

**Alerts Skipped (Deduplication)**:
- ❌ `service.events.processed.total{event_name:*}` - redundant with L0
- ❌ `service.sub_events.processed.total` - too granular (L2)
- ❌ `service.event.duration{event_name:user}` - redundant, use L0 + dashboard filter
- ❌ Individual dependency metrics - use aggregate `service.external.*`

### Usage

```bash
# Generate and push alerts to Datadog
python scripts/generate_alerts.py \
  --service notification-service \
  --datadog-api-key $DATADOG_API_KEY \
  --datadog-app-key $DATADOG_APP_KEY

# Dry-run (generate without pushing)
python scripts/generate_alerts.py \
  --service notification-service \
  --dry-run
```

## Quick Reference: Datadog API Format

### Query Format Rules
1. **MUST include comparison operator**: `>=`, `<=`, `>`, `<`
2. **Query threshold MUST match critical threshold**: If `critical: 10`, query must be `>= 10`
3. **Type is always `"query alert"`**: Never use `"metric alert"`
4. **Simple queries only**: No complex formulas, use basic metric aggregations

### Options Rules
1. **threshold_windows**: ONLY for anomaly monitors (using `anomalies()` function)
2. **on_missing_data**: Valid values: `"show_no_data"`, `"show_and_notify_no_data"`, `"resolve"`, `"default"`
3. **notify_no_data**: Deprecated - remove if using `on_missing_data`
4. **require_full_window**: Use `false` (not `true`)

### Common Error Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "Alert threshold does not match query" | Query threshold ≠ critical threshold | Match query `>= X` to `critical: X` |
| "threshold_windows is only used with anomalies" | Added to regular alert | Remove from non-anomaly monitors |
| "Invalid on_missing_data value" | Used invalid value | Use: `show_no_data`, `show_and_notify_no_data`, `resolve`, `default` |
| "notify_no_data option is deprecated" | Used with `on_missing_data` | Remove `notify_no_data` field |
| "The value provided for parameter 'query' is invalid" | Query format wrong | Add comparison operator, remove formulas |

### Template: Regular Query Alert

```json
{
  "type": "query alert",
  "query": "avg(last_5m):sum:metric{tags} >= 10",
  "name": "[P0] Service - Alert Name",
  "message": "Alert message",
  "tags": ["service:service-name", "priority:P0"],
  "options": {
    "thresholds": {
      "critical": 10,
      "warning": 5
    },
    "require_full_window": false,
    "notify_audit": false,
    "on_missing_data": "default",
    "evaluation_delay": 0,
    "renotify_interval": 60,
    "new_host_delay": 300,
    "include_tags": false
  },
  "priority": 1
}
```

### Template: Anomaly Detection Alert

```json
{
  "type": "query alert",
  "query": "avg(last_15m):anomalies(sum:metric{tags}.as_rate(), 'agile', 2, direction='both', interval=60, alert_window='last_15m', count_default_zero='true', seasonality='hourly') >= 1",
  "name": "[P0] Service - Anomaly Alert",
  "message": "Anomaly detected",
  "tags": ["service:service-name", "priority:P0", "type:anomaly"],
  "options": {
    "thresholds": {
      "critical": 1,
      "critical_recovery": 0
    },
    "threshold_windows": {
      "trigger_window": "last_15m",
      "recovery_window": "last_15m"
    },
    "require_full_window": false,
    "notify_audit": false,
    "on_missing_data": "default",
    "evaluation_delay": 0,
    "renotify_interval": 60,
    "new_host_delay": 300,
    "include_tags": false
  },
  "priority": 1
}
```

## References
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/)
- [Netflix Atlas](https://github.com/Netflix/atlas)
- [Datadog Alerting Best Practices](https://docs.datadoghq.com/monitors/guide/)
- [Datadog Monitor API Documentation](https://docs.datadoghq.com/api/latest/monitors/)

