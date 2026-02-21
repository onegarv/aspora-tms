# ECM Operations: Two-Agent Architecture

## Why Two Agents?

The ECM operations evolved from a single monolith (9 skills, 3 manager implementations, 25 runbooks) into a cleanly split two-agent system. The split follows the SAGE principle: **Scoped** — each agent has a bounded context.

### Problem with the Monolith
- **Context bloat**: Loading triage scoring + runbooks + resolution workflow consumed excessive tokens
- **Role confusion**: Same Claude.md routed both manager operations (triage, assign) and field operations (diagnose, resolve)
- **Deployment mismatch**: Manager runs as K8s cron (batch), field runs as interactive Claude Code session

### The Split

| | Manager Agent | Field Agent |
|---|---|---|
| **Purpose** | Analyse, prioritize, assign | Diagnose, resolve, escalate |
| **Persona** | Operations manager | Individual field agent |
| **Runtime** | ECS/K8s scheduled (3x daily) | Interactive Claude Code |
| **Skills** | triage-and-assign, assign-tickets, run-ecm, ecm-daily-flow | my-tickets, order-details, resolve-ticket, escalate-ticket |
| **Reads** | Queries, stuck-reasons, config | Queries, runbooks, config, stuck-reasons |
| **Never loads** | Runbooks (field agent's job) | Triage scoring (manager's job) |

## SAGE Principles Applied

### Scoped
- Manager CLAUDE.md routes only to 4 manager skills. Field CLAUDE.md routes only to 4 field skills.
- Context boundaries prevent leakage: manager never loads runbooks, field never computes priority scores.

### Autonomous
- Manager runs unattended via K8s CronJob at 7AM, 2PM, 8PM UAE.
- Field agent operates interactively — agent says `my tickets`, gets actionable queue.

### Grounded
- Both agents use the same `shared/` resources: queries (7 SQL files), runbooks (25 playbooks), config (diagnosis-mapping, knowledge-graph, stuck-reasons).
- No duplication — single source of truth.

### Efficient
- Two-step query architecture: fast list (<5s) then detail per order.
- Token budget: each agent loads only its 4 skills + shared guardrails (not all 8+).

## Shared Resources

```
shared/
├── guardrails.md           # Universal safety rules
├── stuck-reasons.yaml      # SSOT: stuck_reason → team, SLA, runbook
├── queries/                # 7 SQL files (Redshift read-only)
├── runbooks/               # 25 resolution playbooks
└── config/                 # knowledge-graph, diagnosis-mapping, etc.
```

Both agents reference these via `../shared/` relative paths. The Manager Dockerfile copies both `manager/` and `shared/` into the container.

## Deployment

### Manager (automated)
- **Dockerfile** builds from repo root: `docker build -f manager/deploy/Dockerfile .`
- **K8s CronJob** at `0 3,10,16 * * *` (UTC) = 7AM, 2PM, 8PM UAE
- **ECS** scheduled task with EventBridge rule
- **Entrypoint** calls `claude --print` with triage instructions

### Field (interactive)
- Clone repo → `cd field/` → `claude`
- `.mcp.json` configures ecm-gateway MCP automatically
- Slash commands available: `/my-tickets`, `/order-details`, `/resolve-ticket`, `/escalate-ticket`, `/ecm-dashboard`
- Onboarding: 3 commands to start working

## Decision Records

See `DECISIONS.md` at repo root for the full list (DEC-001 through DEC-010). Key decisions:
- **DEC-001**: Skills-first over code-first
- **DEC-002**: ecm-gateway as sole MCP server
- **DEC-003**: Two-step query architecture
- **DEC-007**: Direct connections for K8s batch, MCP for Claude Code

## Migration from Monolith

The `_archive/` directory contains superseded implementations:
- `pi-manager/` — Python-based triage with direct Redshift
- `pi-skill/` — Shell + Claude CLI approach (entrypoint.sh adapted for manager/deploy/)
- `voltagent-manager/` — VoltAgent framework experiment
- `agent-skills/` — Early skill definitions
