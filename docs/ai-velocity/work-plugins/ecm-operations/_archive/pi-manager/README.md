# ECM Pi Manager

Scalable, cost-efficient scheduled triage and reporting for ECM operations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          PI MANAGER                                 │
│            K8s CronJob (scheduled) / Claude Code (manual)           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
    ┌───────────┐           ┌───────────┐           ┌───────────┐
    │  Redshift │           │  Google   │           │   Slack   │
    │  (Direct) │           │  Sheets   │           │  Bot API  │
    │           │           │  (Direct) │           │           │
    └───────────┘           └───────────┘           └───────────┘

Two Modes:
- K8s/VPS: Direct connections via environment variables
- Claude Code: MCP tools (interactive)
```

## Deployment

### Kubernetes (Recommended for Scheduled Runs)

```bash
# Build and push image
docker build -t ecm-manager:latest -f k8s/Dockerfile .
docker push your-registry/ecm-manager:latest

# Create secrets (edit k8s/deployment.yaml first!)
kubectl apply -f k8s/deployment.yaml

# Check CronJob
kubectl get cronjobs -n ecm-operations
```

### Local Testing

```bash
# Set environment variables
export REDSHIFT_HOST="your-cluster.redshift.amazonaws.com"
export REDSHIFT_USER="user"
export REDSHIFT_PASSWORD="pass"
export REDSHIFT_DATABASE="db"
export GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
export SLACK_BOT_TOKEN="xoxb-..."

# Run
python main.py test      # Test connections
python main.py triage    # Full triage
python main.py progress  # Progress report
```

## Features

| Feature | Description |
|---------|-------------|
| **Triage** | Query stuck orders, validate data quality, score priority |
| **Assignment** | Round-robin distribution (high-value first) |
| **Daily Briefing** | Post to Slack with agent threads |
| **Progress Report** | Current queue, SLA breaches, self-healed orders |
| **Guardrails** | Dead order filtering, count sanity, currency validation |

## Quick Start

### Configuration

Environment variables (in `../pi-skill/.env`):

```env
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_ID=C0AD6C36LVC
SPREADSHEET_ID=1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks
```

### Run via Claude Code

```bash
# Full triage workflow
python run.py triage

# Progress report
python run.py progress

# Test config
python run.py test
```

### Scheduled (Pi.dev)

Cron: `0 3,10,16 * * *` (7AM, 2PM, 8PM UAE)

## File Structure

```
pi-manager/
├── SKILL.md              # Skill definition for Pi.dev
├── README.md             # This file
├── run.py                # CLI entry point
├── requirements.txt      # Dependencies (stdlib only)
├── config/
│   └── schedule.yaml     # Schedule configuration
├── queries -> ../queries # Symlink to shared queries
└── src/
    ├── __init__.py
    ├── main.py           # Main orchestrator
    ├── config.py         # Configuration management
    ├── models.py         # Data models (Order, Agent, Assignment)
    ├── data_client.py    # MCP/API clients
    ├── triage.py         # Triage & assignment logic
    └── slack_reporter.py # Slack messaging
```

## Workflows

### 1. Triage (Daily Briefing)

```
Step 1: Query ecm-pending-list.sql   → Get actionable orders
Step 2: Validate data quality        → Fail if >2000 or bad currency mix
Step 3: Read Assignments sheet       → Get already assigned
Step 4: Read Agents sheet            → Get active agents
Step 5: Distribute (round-robin)     → High-value first
Step 6: Write to Assignments sheet   → Add new rows
Step 7: Post Daily Briefing          → Main Slack message
Step 8: Post agent threads           → Per-agent breakdown
```

### 2. Progress Report

```
Step 1: Read Assignments sheet       → Current state
Step 2: Query Redshift for resolved  → Find self-healed orders
Step 3: Check SLA breaches           → By diagnosis type
Step 4: Post Progress Report         → Summary + alerts
```

### 3. Cleanup

```
Step 1: Get OPEN/IN_PROGRESS orders  → From sheet
Step 2: Check Redshift for COMPLETED → Self-healed detection
Step 3: Mark as RESOLVED in sheet    → Update status
```

## Data Quality Validation

| Check | Expected | Warning | Fail |
|-------|----------|---------|------|
| Order count | 200-600 | >600 | >2000 |
| AED % | 50-70% | =100% | - |
| GBP % | 25-40% | >80% | - |

## SLA Configuration

| Diagnosis | SLA Hours |
|-----------|-----------|
| refund_triggered | 2 |
| manual_review | 0.5 |
| status_sync_issue | 1 |
| brn_pending | 4 |
| rfi_pending | 24 |
| default | 8 |

## Cost Efficiency

- **Zero external dependencies** — Pure Python 3.9+ stdlib
- **Batched queries** — Single SQL per step
- **Minimal API calls** — One Sheet read/write per step
- **Efficient Slack** — Threading for agent messages

## Links

- [ECM Dashboard](https://docs.google.com/spreadsheets/d/1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks)
- [GitHub Repo](https://github.com/Vance-Club/ai-velocity/tree/stage-env/work-plugins/ecm-operations)
- [Slack Channel](https://aspora.slack.com/archives/C0AD6C36LVC)
