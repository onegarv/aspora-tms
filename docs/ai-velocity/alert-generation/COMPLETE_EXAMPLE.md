# Complete Alert Generation Example for Notification Service

This document shows the complete output of the alert-generation skill for notification-service.

**Note**: This example follows the deduplication rules - only unique alerts are generated (one per signal+level combination). Alerts are organized by Four Golden Signals, matching the dashboard structure.

## Alert Summary

**Total Alerts**: 9 (after deduplication)
- **P0 Alerts**: 3 (critical service-level issues)
- **P1 Alerts**: 6 (high-priority feature-level issues)

**Organization**: Four Golden Signals (same as dashboard)
- **Traffic (Rate)**: 2 alerts
- **Errors**: 2 alerts
- **Latency (Duration)**: 2 alerts
- **Saturation**: 2 alerts
- **Anomaly Detection**: 1 alert

**Alerts Skipped** (Deduplication):
- `service.events.processed.total{event_name:*}` - redundant with L0 aggregate
- `service.sub_events.processed.total` - too granular (L2)
- `service.event.duration{event_name:user}` - redundant, use L0 + dashboard filter
- Individual dependency metrics - use aggregate `service.external.*`

## Generated Alerts

### P0 - Critical Alerts (Page Immediately)

#### 1. High Error Rate
```json
{
  "type": "metric alert",
  "query": "avg(last_5m):(sum:service.events.failed.total{service:notification-service}.as_rate() / (sum:service.events.processed.total{service:notification-service}.as_rate() + sum:service.events.failed.total{service:notification-service}.as_rate())) * 100 > 1",
  "name": "[P0] Notification Service - Error Rate Exceeds 1%",
  "message": "🚨 CRITICAL: Error rate exceeded 1% SLO threshold.\n\nCurrent error rate: {{value}}%\nThreshold: 1%\n\n@slack-alerts-critical @pagerduty-notification-service\n\nRunbook: https://wiki.company.com/runbooks/notification-service-error-rate",
  "tags": ["service:notification-service", "priority:P0", "team:platform", "signal:errors"],
  "options": {
    "thresholds": {
      "critical": 1.0,
      "warning": 0.5
    },
    "require_full_window": true,
    "notify_audit": false,
    "notify_no_data": false,
    "evaluation_delay": 0,
    "renotify_interval": 60
  },
  "priority": 1
}
```

#### 2. Service Down (No Events Processed)
```json
{
  "type": "metric alert",
  "query": "avg(last_5m):sum:service.events.processed.total{service:notification-service}.as_rate() < 0.1",
  "name": "[P0] Notification Service - No Events Processed",
  "message": "🚨 CRITICAL: Service appears down - no events processed in last 5 minutes.\n\n@slack-alerts-critical @pagerduty-notification-service",
  "tags": ["service:notification-service", "priority:P0", "team:platform", "signal:traffic"],
  "options": {
    "thresholds": {
      "critical": 0.1
    },
    "require_full_window": true,
    "notify_audit": false,
    "notify_no_data": true,
    "no_data_timeframe": 5
  },
  "priority": 1
}
```

#### 3. Anomalous Event Processing Rate (Anomaly Detection)
```json
{
  "type": "query alert",
  "query": "anomalies(avg:service.events.processed.total{service:notification-service}.as_rate(), 'basic', 2)",
  "name": "[P0] Notification Service - Anomalous Event Processing Rate",
  "message": "🚨 CRITICAL: Anomaly detected in event processing rate.\n\nAnomaly score: {{value}}\n\n@slack-alerts-critical",
  "tags": ["service:notification-service", "priority:P0", "team:platform", "type:anomaly", "signal:traffic"],
  "options": {
    "thresholds": {
      "critical": 2
    },
    "require_full_window": true,
    "notify_audit": false,
    "notify_no_data": false
  },
  "priority": 1
}
```

### P1 - High Priority Alerts (Page During Business Hours)

#### 4. High Latency
```json
{
  "type": "metric alert",
  "query": "avg(last_10m):max:service.event.duration.max{service:notification-service} > 0.5",
  "name": "[P1] Notification Service - High Latency (p95 > 500ms)",
  "message": "⚠️ HIGH: Event processing latency p95 exceeded 500ms.\n\nCurrent latency: {{value}}ms\nThreshold: 500ms\n\n@slack-alerts-high",
  "tags": ["service:notification-service", "priority:P1", "team:platform", "signal:latency"],
  "options": {
    "thresholds": {
      "critical": 0.5,
      "warning": 0.1
    },
    "require_full_window": true,
    "notify_audit": false,
    "notify_no_data": false
  },
  "priority": 2
}
```

#### 5. High Pending Events (Saturation)
```json
{
  "type": "metric alert",
  "query": "avg(last_5m):avg:service.pending.events.count{service:notification-service} > 1000",
  "name": "[P1] Notification Service - High Pending Events Count",
  "message": "⚠️ HIGH: Pending events count exceeded 1000.\n\nCurrent count: {{value}}\nThreshold: 1000\n\n@slack-alerts-high",
  "tags": ["service:notification-service", "priority:P1", "team:platform", "signal:saturation"],
  "options": {
    "thresholds": {
      "critical": 1000,
      "warning": 100
    },
    "require_full_window": true,
    "notify_audit": false,
    "notify_no_data": false
  },
  "priority": 2
}
```

#### 6. Stuck Events Detected
```json
{
  "type": "metric alert",
  "query": "avg(last_5m):avg:service.jobs.alert.stuck_events.count{service:notification-service} > 0",
  "name": "[P1] Notification Service - Stuck Events Detected",
  "message": "⚠️ HIGH: Stuck events detected (> 24 hours old).\n\nCurrent count: {{value}}\n\n@slack-alerts-high",
  "tags": ["service:notification-service", "priority:P1", "team:platform", "signal:saturation"],
  "options": {
    "thresholds": {
      "critical": 0
    },
    "require_full_window": true,
    "notify_audit": false,
    "notify_no_data": false
  },
  "priority": 2
}
```

#### 7. External Dependency Failure Rate
```json
{
  "type": "metric alert",
  "query": "avg(last_5m):(sum:service.external.requests.total{service:notification-service,status:error}.as_rate() / sum:service.external.requests.total{service:notification-service}.as_rate()) * 100 > 5",
  "name": "[P1] Notification Service - High External Dependency Failure Rate",
  "message": "⚠️ HIGH: External dependency failure rate exceeded 5%.\n\nCurrent failure rate: {{value}}%\nThreshold: 5%\n\n@slack-alerts-high",
  "tags": ["service:notification-service", "priority:P1", "team:platform", "signal:errors"],
  "options": {
    "thresholds": {
      "critical": 5.0,
      "warning": 2.0
    },
    "require_full_window": true,
    "notify_audit": false,
    "notify_no_data": false
  },
  "priority": 2
}
```

#### 8. Anomalous Error Pattern (Anomaly Detection)
```json
{
  "type": "query alert",
  "query": "anomalies(sum:service.errors.total{service:notification-service}.as_rate(), 'basic', 2)",
  "name": "[P1] Notification Service - Anomalous Error Pattern",
  "message": "⚠️ HIGH: Anomaly detected in error pattern.\n\nAnomaly score: {{value}}\n\n@slack-alerts-high",
  "tags": ["service:notification-service", "priority:P1", "team:platform", "type:anomaly", "signal:errors"],
  "options": {
    "thresholds": {
      "critical": 2
    },
    "require_full_window": true,
    "notify_audit": false,
    "notify_no_data": false
  },
  "priority": 2
}
```

#### 9. Anomalous External Dependency Latency (Anomaly Detection)
```json
{
  "type": "query alert",
  "query": "anomalies(avg:service.external.duration{service:notification-service}, 'basic', 2)",
  "name": "[P1] Notification Service - Anomalous External Dependency Latency",
  "message": "⚠️ HIGH: Anomaly detected in external dependency latency.\n\nAnomaly score: {{value}}\n\n@slack-alerts-high",
  "tags": ["service:notification-service", "priority:P1", "team:platform", "type:anomaly", "signal:latency"],
  "options": {
    "thresholds": {
      "critical": 2
    },
    "require_full_window": true,
    "notify_audit": false,
    "notify_no_data": false
  },
  "priority": 2
}
```

## Usage

1. **Generate alerts**:
   ```bash
   python scripts/generate_alerts.py \
     --service notification-service \
     --datadog-api-key $DATADOG_API_KEY \
     --datadog-app-key $DATADOG_APP_KEY
   ```

2. **Dry-run (generate without pushing)**:
   ```bash
   python scripts/generate_alerts.py \
     --service notification-service \
     --dry-run
   ```

3. **Output**: Alerts saved to `alerts/notification-service/monitors.json`

