# ECM Operations

Exception Case Management for Aspora Remittance.

## What This Solves

Aspora is a remittance company. Customers send money from UAE (AED) to India (via Lulu Exchange), UK and Europe (via Falcon/partner banks). Orders flow through 5-6 systems: GOMS (order management), payment acquirer (Checkout/Leantech), Falcon (transaction router), Lulu Exchange (UAE payout), RDA/VDA partners (international payout), and KYC/compliance checks.

**Orders get stuck.** A customer sends 5,000 AED to their family — the money leaves their account but never arrives. The order sits in a "processing" state somewhere between these systems. On any given day, 200-600 orders are stuck. Total value at risk: ~1M+ AED.

**Before ECM agents:** A human ops team manually ran SQL queries across 5-6 systems, cross-referenced to find where each order was stuck, decided what to do, tracked everything in spreadsheets — 3 times a day for hundreds of orders.

**With ECM agents:** Two AI agents replace that manual loop. Cron fires at 7AM IST, the manager agent queries all systems, scores every stuck order by urgency, assigns them to human agents with exact instructions, posts a summary to Slack, and exits. Zero human intervention to kick it off.

## Two-Agent Architecture

### Manager Agent (autonomous, runs on cron)

Queries Redshift for all stuck orders, filters out dead orders (abandoned payments), scores each by urgency (age x amount x severity), assigns to ops agents via Google Sheet, posts to Slack, and detects systemic failure patterns.

**Skills:** `triage-and-assign`, `ecm-daily-flow`, `pattern-intelligence`, `assign-tickets`, `run-ecm`

### Field Agent (interactive, runs in Claude Code)

Each ops agent sees their assigned orders, gets a root-cause diagnosis across all 6 systems, and follows the exact runbook for that failure type. 25 runbooks cover every known stuck reason.

**Skills:** `my-tickets`, `order-details`, `resolve-ticket`, `escalate-ticket`

### Common Stuck Reasons

| Reason | What Happened | Orders (typical) |
|--------|--------------|-----------------|
| `refund_pending` | Money taken, order cancelled, refund not triggered | 50-60 |
| `stuck_at_lulu` | Money sent to Lulu, recipient not credited | 80-100 |
| `stuck_due_to_payment_issue_goms` | Payment captured but Falcon transaction never created (GBP) | 100+ |
| `status_sync_issue` | Money arrived but systems disagree on status | 10-20 |
| `brn_issue` | Bank reference not pushed to Lulu | 5-10 |
| `rfi_pending` (>24h) | Compliance needs customer documents, waiting too long | 20-30 |

Full list: `shared/stuck-reasons.yaml` (20+ reasons across Ops, KYC, and VDA teams).

## Autonomous Deployment

### Daily Schedule

```
7:00 AM IST    daily flow = Backlog → Triage → Patterns (3 Slack reports)
2:00 PM UAE    triage only (1 Slack report)
8:00 PM UAE    triage only (1 Slack report)
```

### Slack Messages

| Message | Title | Content |
|---------|-------|---------|
| Backlog | `📊 ECM Backlog — {date} IST` | Backlog size, severity split, 7-day trend, value at risk |
| Triage | `🎯 ECM Triage — {date} IST` | Orders assigned, agent capacity, top critical assignments |
| Patterns | `🔬 ECM Patterns — {date} IST` | Systemic failure clusters, trends, novel patterns |
| Error | `🚨 ECM Manager Error` | Which phase failed (only on failure) |

### Run Locally

```bash
# Set up credentials
cp manager/.env.example manager/.env
# Edit manager/.env with your OpenRouter + Slack tokens

# Run full morning flow
source manager/.env && bash manager/deploy/entrypoint.sh daily

# Run individual phases
bash manager/deploy/entrypoint.sh backlog
bash manager/deploy/entrypoint.sh triage
bash manager/deploy/entrypoint.sh patterns
bash manager/deploy/entrypoint.sh test      # validate connections
bash manager/deploy/entrypoint.sh health    # health check
```

### Run via Docker

```bash
# Build
docker build -t ecm-manager -f manager/deploy/Dockerfile .

# Run
docker run --env-file manager/.env ecm-manager daily
docker run --env-file manager/.env ecm-manager triage
docker run --env-file manager/.env ecm-manager patterns
```

### Deploy to VPS (cron)

```bash
# 7AM IST = 1:30 AM UTC
30 1 * * * cd /path/to/ecm-operations/manager/deploy && docker compose run --rm ecm-daily >> /var/log/ecm-manager.log 2>&1
```

### Deploy to Kubernetes

```bash
kubectl apply -f manager/deploy/k8s/cronjob.yaml
```

### Field Agent (interactive)

```bash
cd field/
claude
> my tickets
```

## Directory Structure

```
ecm-operations/
├── manager/                    # Autonomous manager agent
│   ├── CLAUDE.md               # Manager persona + skill routing
│   ├── skills/                 # 5 manager skills (.md)
│   ├── config/schedule.yaml    # Cron schedule config
│   ├── deploy/                 # Dockerfile, entrypoint.sh, k8s/, docker-compose.yml
│   ├── .env.example            # Credential template (OpenRouter + Slack)
│   └── run-patterns.sh         # Standalone pattern runner (crontab)
├── field/                      # Interactive field agent
│   ├── CLAUDE.md               # Field persona + skill routing
│   ├── skills/                 # 4 field skills (.md)
│   └── .claude/commands/       # Slash commands (my-tickets, order, etc.)
├── shared/                     # Shared by both agents
│   ├── queries/                # 8 SQL files (Redshift, read-only)
│   ├── runbooks/               # 25 resolution playbooks
│   ├── config/                 # diagnosis-mapping, knowledge-graph, slack-formatting
│   ├── guardrails.md           # Safety rules for all skills
│   └── stuck-reasons.yaml      # stuck_reason → team, SLA, runbook, action
├── CLAUDE.md                   # Root instructions
├── DECISIONS.md                # Architecture decisions (DEC-001 through DEC-012)
├── ESCALATION.md               # Contact matrix (TechOps, FinOps, Lulu, CNR)
├── .mcp.json                   # MCP gateway config (ecm-gateway)
└── _archive/                   # Superseded: pi-manager, voltagent-manager, etc.
```

## MCP Gateway

All data access goes through `ecm-gateway` MCP server (Redshift read-only + Google Sheets). No direct database credentials needed. Configured in `.mcp.json`.

## Key Decisions

| ID | Decision |
|----|----------|
| DEC-001 | Skills-first — business logic in `.md` skills, not code |
| DEC-002 | ecm-gateway as sole MCP server |
| DEC-004 | Dead order filtering (payment + Lulu join) |
| DEC-007 | MCP for Claude Code, direct connections for K8s batch |
| DEC-011 | Two-agent split (manager + field) |
| DEC-012 | Autonomous manager uses `claude --print` + MCP |

Full list: `DECISIONS.md`

## Links

- [ECM Dashboard (Google Sheet)](https://docs.google.com/spreadsheets/d/1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks/edit)
- [Slack Channel: #wg-asap-agent-pilot](https://slack.com)
- [Escalation Matrix](ESCALATION.md)
