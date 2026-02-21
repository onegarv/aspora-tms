# ECM VoltAgent Manager

AI-powered ECM Operations Manager with learning capabilities. Built on VoltAgent framework.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ECM MANAGER (Supervisor)                         │
│         "Orchestrate triage, learn from outcomes, report"            │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌─────▼─────┐         ┌────▼────┐
   │ Triage  │          │  Slack    │         │ Learner │
   │  Agent  │          │  Tools    │         │  Agent  │
   └────┬────┘          └───────────┘         └────┬────┘
        │                                          │
   ┌────▼────┐                               ┌─────▼─────┐
   │   MCP   │                               │  LibSQL   │
   │ Gateway │                               │  (Turso)  │
   ├─────────┤                               ├───────────┤
   │Redshift │                               │ outcomes  │
   │ Sheets  │                               │ patterns  │
   └─────────┘                               │ feedback  │
                                             └───────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **Triage** | Query stuck orders, validate, score priority |
| **Assignment** | Round-robin distribution (high-value first) |
| **Learning** | Track outcomes, identify auto-resolve patterns |
| **Feedback** | Record agent feedback, improve diagnoses |
| **CEO-style Reports** | Progress, direction, what's working |

## Quick Start

### Prerequisites

- Node.js 20+
- Turso account (free tier works)
- Anthropic API key
- Slack bot token

### Setup

```bash
cd voltagent-manager

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run triage
npm run triage

# Run progress report
npm run progress

# Update learning patterns
npm run learn
```

### Local Development

```bash
# Watch mode
npm run dev

# Run tests
npm test
```

## Commands

| Command | Schedule | Description |
|---------|----------|-------------|
| `triage` | 7AM, 2PM, 8PM UAE | Full triage + assignment + briefing |
| `progress` | 11AM, 5PM UAE | Progress report + outcome tracking |
| `learn` | Midnight UAE | Aggregate patterns from outcomes |

## Learning System

The agent learns from outcomes:

```sql
-- What it tracks
outcomes: order_id, diagnosis, outcome, resolution_hours, feedback
patterns: diagnosis, auto_resolve_rate, avg_resolution_hours

-- What it learns
"REFUND_TRIGGERED auto-resolves 15% of the time → lower priority"
"stuck_at_lulu needs manual intervention 85% → higher priority"
```

### How Learning Affects Priority

1. High auto-resolve rate (>50%) → Lower priority (system handles it)
2. Low auto-resolve rate (<10%) → Higher priority (needs human)
3. Long resolution time → Higher priority (needs attention)

## K8s Deployment

```bash
# Build image
docker build -t ecm-manager:latest -f k8s/Dockerfile .

# Create secrets
kubectl create secret generic ecm-manager-secrets --from-env-file=.env -n ecm-operations

# Deploy CronJobs
kubectl apply -f k8s/cronjob.yaml
```

### CronJob Schedule

| Job | Cron (UTC) | UAE Time |
|-----|------------|----------|
| triage | `0 3,10,16 * * *` | 7AM, 2PM, 8PM |
| progress | `0 7,13 * * *` | 11AM, 5PM |
| learn | `0 20 * * *` | Midnight |

## Project Structure

```
voltagent-manager/
├── src/
│   ├── agents/
│   │   ├── manager.ts      # Supervisor agent
│   │   ├── triage.ts       # Triage sub-agent
│   │   └── learner.ts      # Learning sub-agent
│   ├── tools/
│   │   ├── slack.ts        # Slack integration
│   │   └── learning.ts     # Memory operations
│   ├── memory/
│   │   ├── schema.ts       # LibSQL schema
│   │   └── client.ts       # Memory client
│   ├── config.ts           # Configuration
│   ├── cli.ts              # CLI entry point
│   └── index.ts            # HTTP server
├── k8s/
│   ├── cronjob.yaml        # K8s CronJobs
│   ├── secrets.yaml        # Secrets template
│   └── Dockerfile          # Container build
└── package.json
```

## Integration with Existing Skills

The agents load existing SKILL.md files as instructions:

- `skills/guardrails.md` → Triage Agent guardrails
- `skills/triage-and-assign.md` → Triage workflow
- `pi-manager/SKILL.md` → Manager knowledge

**Your skills are the knowledge. VoltAgent is the runtime.**

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `TURSO_DATABASE_URL` | Yes | LibSQL connection URL |
| `TURSO_AUTH_TOKEN` | Yes | Turso auth token |
| `SLACK_BOT_TOKEN` | Yes | Slack bot token |
| `SLACK_CHANNEL_ID` | No | Default: C0AD6C36LVC |
| `SPREADSHEET_ID` | No | Default: ECM sheet |

## Links

- [ECM Dashboard](https://docs.google.com/spreadsheets/d/1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks)
- [VoltAgent Docs](https://voltagent.dev/docs)
- [Turso](https://turso.tech)
