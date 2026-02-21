# ECM Triage Skill

Automated Exception Case Management triage for Aspora Remittance. Runs on schedule, analyzes stuck orders, assigns to agents, and posts to Slack.

## Quick Start

### Local (Docker)
```bash
# Build and run
./deploy.sh local
```

### Kubernetes
```bash
# Deploy CronJob (runs 7AM, 2PM, 8PM UAE)
./deploy.sh k8s
```

### ECS/Fargate
```bash
# Deploy scheduled task
./deploy.sh ecs
```

## What It Does

1. **Queries Redshift** — Finds stuck orders (12h+ old, last 30 days)
2. **Filters actionable** — Excludes abandoned payments, dead orders
3. **Scores priority** — P1-P4 based on age, amount, severity
4. **Distributes evenly** — High-value orders (>5K) round-robin across agents
5. **Writes to Sheet** — Assignments with order ID, agent, diagnosis
6. **Posts to Slack** — Motivational message + per-agent order threads

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

## Files

```
pi-skill/
├── SKILL.md              # Main skill definition (Claude reads this)
├── Dockerfile            # Container image
├── entrypoint.sh         # Runtime entrypoint
├── deploy.sh             # Deploy helper
├── config/
│   └── schedule.yaml     # Schedule + thresholds
├── queries/
│   └── *.sql             # Redshift queries
├── references/
│   └── *.md              # Runbooks, escalation contacts
├── k8s/
│   └── cronjob.yaml      # Kubernetes deployment
└── ecs/
    ├── task-definition.json
    └── eventbridge-schedule.json
```

## Agent Commands

Once deployed, agents use these in Claude:

```
/my-tickets              # See assigned orders
order AE12Y0K4BU00       # Get diagnosis + runbook
resolve AE12Y0K4BU00 "refund done"  # Mark complete
escalate AE12Y0K4BU00 "Lulu timeout" # Escalate
```

## Slack Output

**Main message:**
```
🚀 ECM War Room — Let's Clear the Queue!
425 customers waiting. You're the heroes! 💪

🔴 P1 Critical: 180 orders
💰 High Value: 25 orders (>5K)
💵 Total: ~2.1M AED at stake

🎯 Your Mission:
👤 akshay — 102 tickets (5 high-value)
👤 vishnu.r — 84 tickets (5 high-value)
...
```

**Thread per agent:**
```
@akshay — Your Orders:
`AE12Y0K4BU00` 60,100 AED (297h old!)
`AE13IZSV2O00` 35,000 AED
...
```

## Development

```bash
# Build image
./deploy.sh build

# Test connections
./deploy.sh test

# Run locally with .env
./deploy.sh local
```

## Links

- [ECM Dashboard](https://docs.google.com/spreadsheets/d/1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks/edit)
- [Slack Channel](https://aspora.slack.com/archives/C0AD6C36LVC)
