# ECM Operations — Two-Agent Architecture

This repo contains two ECM agents. **Navigate to the correct directory before starting.**

## Agent Routing

| You are... | Directory | Command |
|-----------|-----------|---------|
| **Manager** (triage, assign, dashboard) | `cd manager/` | `claude` or `./deploy/deploy.sh local` |
| **Field agent** (tickets, diagnose, resolve) | `cd field/` | `claude` then `my tickets` |

## Shared Resources

Both agents reference `shared/` for queries, runbooks, config, and guardrails. Do not duplicate.

## Before Any Task
1. Read `DECISIONS.md` — prior architecture decisions.
2. Read `shared/guardrails.md` — hard constraints for all ECM skills.

## Structure

```
ecm-operations/
├── manager/     # Triage, assign, dashboard (ECS/K8s or interactive)
├── field/       # My tickets, order diagnosis, resolve, escalate
├── shared/      # Queries, runbooks, config, guardrails (no duplication)
├── docs/        # Architecture docs
└── _archive/    # Superseded implementations
```
