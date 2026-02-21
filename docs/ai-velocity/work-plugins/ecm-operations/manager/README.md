# ECM Manager Agent

Automated ECM triage and assignment for Aspora Remittance. Runs on schedule (7AM, 2PM, 8PM UAE), analyses stuck orders, assigns to field agents, posts to Slack.

## Quick Start

### Local (interactive)
```bash
cd manager/
claude    # starts Claude Code with manager CLAUDE.md
> triage
```

### Local (Docker)
```bash
cd manager/deploy && ./deploy.sh local
```

### Kubernetes
```bash
cd manager/deploy && ./deploy.sh k8s
```

### ECS/Fargate
```bash
cd manager/deploy && ./deploy.sh ecs
```

## What It Does

1. **Queries Redshift** — Finds stuck orders (12h+ old, last 30 days)
2. **Filters actionable** — Excludes abandoned payments, dead orders
3. **Scores priority** — P1-P4 based on age, amount, severity
4. **Distributes evenly** — High-value orders (>5K) round-robin across agents
5. **Writes to Sheet** — Assignments with order ID, agent, diagnosis
6. **Posts to Slack** — Summary + per-agent order threads

## Schedule

| Time (UAE) | UTC | Purpose |
|------------|-----|---------|
| 7:00 AM | 3:00 AM | Morning triage |
| 2:00 PM | 10:00 AM | Midday refresh |
| 8:00 PM | 4:00 PM | Evening handoff |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `SLACK_BOT_TOKEN` | Yes | Slack bot token (`xoxb-...`) |
| `SLACK_CHANNEL_ID` | Yes | Channel ID for posting |
| `SPREADSHEET_ID` | No | Google Sheet ID (default: ECM Dashboard) |

## Structure

```
manager/
├── CLAUDE.md             # Manager persona + skill routing
├── README.md             # This file
├── .mcp.json             # MCP gateway config
├── .env.example          # Environment template
├── skills/
│   ├── triage-and-assign.md
│   ├── assign-tickets.md
│   ├── run-ecm.md
│   └── ecm-daily-flow.md
├── config/
│   └── schedule.yaml     # Schedule + thresholds
└── deploy/
    ├── Dockerfile
    ├── entrypoint.sh
    ├── deploy.sh
    ├── k8s/cronjob.yaml
    └── ecs/task-definition.json
```

## Shared Resources

Queries, runbooks, and config live in `../shared/` — not duplicated here.
The Dockerfile copies both `manager/` and `shared/` into the container.
