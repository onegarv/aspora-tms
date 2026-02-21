You are setting up production-grade alerts for a service. Follow these phases IN ORDER.
Do not skip phases. Each phase has a GATE — complete it before proceeding.

## Pre-Flight: Decision Context

Before starting, check for these files in the current project root (read them if they exist, skip if not):
1. `DECISIONS.md` — prior decisions for this project. Do NOT re-decide what's already settled.
2. `GUARDRAILS.md` — anti-patterns from past incidents. These are hard constraints.

Also check: $HOME/code/aspora/ai-velocity/alert-generation/ for DECISIONS.md and GUARDRAILS.md

## Phase 0: Understand Alert Philosophy

Before creating any alert, internalize these principles:
- Alert on SYMPTOMS (user impact), not CAUSES (internal state)
- Anomaly-based alerting > static thresholds (baselines adapt, hardcoded numbers go stale)
- 5-7 alerts per service is the sweet spot — enough coverage without alert fatigue
- Every alert MUST have a runbook link — if you can't describe the response, don't alert
- Multi-window strategy reduces false positives (short window detects, long window confirms)

## Phase 1: Discover Metrics

Before writing any alert configuration:
- Read the service's MetricsUtil.java / metrics package to find all available metrics
- Check existing dashboards (Datadog JSON files or Grafana) for metric names
- Query the Prometheus/metrics endpoint if available
- Map each metric to a Golden Signal: Latency, Traffic, Errors, Saturation

GATE: Present a table of discovered metrics and their signal classification. Wait for user confirmation.

## Phase 2: Design Alert Strategy

For each metric, determine:

| Metric | Signal | Alert Type | Priority | Threshold Approach |
|--------|--------|-----------|----------|-------------------|
| [fill] | [L/T/E/S] | static / anomaly / rate-of-change | P0-P3 | [approach] |

Priority levels:
- P0 (Critical): Revenue impact, data loss, full outage → page immediately
- P1 (High): Degraded experience, partial outage → page during business hours
- P2 (Medium): Performance degradation, elevated errors → ticket
- P3 (Low): Informational, capacity planning → weekly review

Threshold approaches:
- SLO-based: derive from SLA (99.9% → alert at 99.5%)
- Baseline: mean + 3σ from historical data
- Rate-of-change: alert on sudden shifts, not absolute values

Read the skill reference for threshold calculation:
$HOME/code/aspora/ai-velocity/alert-generation/SKILL.md

GATE: Present the alert strategy table. Confirm with user before generating configs.

## Phase 3: Generate Alert Configurations

For each alert, generate the Datadog monitor config with:
- [ ] Descriptive name: `[P{priority}] {service} — {signal}: {description}`
- [ ] Query with appropriate evaluation window
- [ ] Warning AND critical thresholds (not just critical)
- [ ] Multi-window for anomaly alerts (short: 5m detect, long: 1h confirm)
- [ ] Tags: service, team, priority, signal
- [ ] Runbook link in message body
- [ ] Notification targets (PagerDuty for P0-P1, Slack for P2-P3)

## Phase 4: Coverage Verification

Fill this table — every signal must have at least one alert:

| Signal     | Alert(s) Created    | Priority | Type          | Runbook? |
|------------|---------------------|----------|---------------|----------|
| Latency    | [fill]              | [P0-P3]  | [type]        | [ ]      |
| Traffic    | [fill]              | [P0-P3]  | [type]        | [ ]      |
| Errors     | [fill]              | [P0-P3]  | [type]        | [ ]      |
| Saturation | [fill]              | [P0-P3]  | [type]        | [ ]      |

GATE: All 4 signals covered. Every alert has a runbook reference. Total alerts between 5-7.

## Phase 5: Self-Check

1. "Does every alert have a clear runbook or response action?" → Must be YES
2. "Are thresholds based on data (SLOs/baselines), not guesses?" → Must be YES
3. "Would these alerts fire during normal operation?" → Must be NO (tune if yes)
4. "Is there alert coverage for all Four Golden Signals?" → Must be YES
5. "Are notification targets appropriate for each priority?" → Must be YES
