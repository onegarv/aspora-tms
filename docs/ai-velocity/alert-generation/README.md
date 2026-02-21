# Alert Generation System

LLM-driven alert generation system that builds on top of prometheus-instrumentation and datadog-dashboard-creation skills.

## Architecture

### Skill-Based Approach
- **alert-generation/SKILL.md**: Comprehensive skill that guides LLM to generate alerts
- **LLM does the work**: Analyzes metrics, determines alert types, calculates thresholds
- **No hardcoded logic**: All intelligence is in the skill

### Builds on Previous Skills
1. **@prometheus-instrumentation**: Provides metrics via MetricsUtil.java
2. **@datadog-dashboard-creation**: Provides dashboard JSON with SLO thresholds
3. **@alert-generation**: Analyzes both to generate alerts

## Features

### Static Threshold Alerts
- Calculates thresholds using SRE best practices
- Uses SLO targets from dashboard
- Multi-window evaluation to reduce noise
- Warning and critical thresholds

### Anomaly Detection Alerts
- Datadog's built-in anomaly detection
- Statistical methods (z-score, moving average)
- ML-based anomaly detection
- Change point detection

### Priority Assignment
- **P0**: Service-level critical alerts (page immediately)
- **P1**: Feature-level high-priority alerts (page during business hours)
- **P2**: Medium-priority alerts (ticket/email)
- **P3**: Low-priority alerts (log only)

## Usage

### Generate Alerts

```bash
# Generate and push to Datadog
python scripts/generate_alerts.py \
  --service notification-service \
  --datadog-api-key $DATADOG_API_KEY \
  --datadog-app-key $DATADOG_APP_KEY

# Dry-run (generate without pushing)
python scripts/generate_alerts.py \
  --service notification-service \
  --dry-run
```

### What It Does

1. **Reads MetricsUtil.java**: Extracts all metrics (counters, timers, gauges)
2. **Reads Dashboard JSON**: Understands SLO thresholds and metric structure
3. **Calls LLM with Skill**: LLM analyzes and generates alert configurations
4. **Generates Datadog Monitors**: Creates complete monitor JSON
5. **Pushes to Datadog**: Creates monitors via API

## Generated Alerts

See `COMPLETE_EXAMPLE.md` for full examples.

**P0 Alerts (Critical)**:
- High Error Rate (> 1%)
- Service Down (no events processed)
- Anomalous Event Processing Rate

**P1 Alerts (High)**:
- High Latency (p95 > 500ms)
- High Pending Events (> 1000)
- Stuck Events Detected
- External Dependency Failure Rate
- Anomalous Error Pattern
- Anomalous External Dependency Latency

## Alert Types

### Static Thresholds
- Error rate metrics → SLO-based thresholds
- Latency metrics → p95 baseline * multiplier
- Throughput metrics → Relative drop thresholds
- Saturation metrics → Baseline * multiplier

### Anomaly Detection
- High-variance metrics → Statistical anomaly detection
- Unpredictable patterns → ML-based anomaly detection
- Change point detection → Sudden shifts

## Datadog Integration

### Monitor Creation
- Uses Datadog API: `POST /api/v1/monitor`
- Supports both static and anomaly detection
- Handles existing monitors (updates if exists)

### Query Format
- Static: `avg(last_5m):metric{tags} > threshold`
- Anomaly: `anomalies(metric{tags}, 'basic', 2)`

## Files

- `SKILL.md`: Complete skill documentation
- `COMPLETE_EXAMPLE.md`: Full alert examples
- `../scripts/generate_alerts.py`: Generation script
- `../alerts/{service}/monitors.json`: Generated monitor configurations

## Next Steps

1. Run alert generation for your service
2. Review generated alerts
3. Adjust thresholds if needed
4. Monitor false positive rates
5. Iterate based on production feedback

