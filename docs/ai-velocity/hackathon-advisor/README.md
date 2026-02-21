# Hackathon Advisor

**Grounded, honest hackathon project runbooks in 60 seconds.**

Give it a problem statement, get back a realistic plan: what to build, what stack, where you'll get stuck, and an hour-by-hour schedule with specific AI-velocity skills to use at each step.

## What It Does

- Classifies what your problem ACTUALLY needs (service? database? interface? AI? integrations?)
- Recommends a production stack (Go/Java + PostgreSQL + Slack) with honest time estimates
- Maps AI-velocity skills to specific moments in the build (`/standards-go` at hour 0, `/test-go` at hour 16)
- Produces a complete runbook with anti-patterns, a grounded expectations checklist, and the 25% rule
- Validates ideas against the five shipping bars

## What It Doesn't Do

- Production architecture decisions (use `/standards-go` or `/dev-java`)
- Skill authoring (use the skill-creator plugin)
- General project planning outside hackathon context

## Quick Start

```
/hackathon-advisor We want to build a fraud alert triage agent that reads
alerts from monitoring, classifies severity, and routes to the right team on Slack.
```

Output: A full runbook with stack (Go + PostgreSQL + Slack bot), 3 tables, hour-by-hour plan, skills to use at each step, skills to ship, anti-patterns, and a grounded expectations checklist.

## How It Works

1. **Classify** — Determines the five infrastructure needs (service, database, interface, AI, integrations) and flags if scope exceeds 20 hours
2. **Stack** — Recommends backend, database, interface, AI layer, and observability based on team strength and problem type
3. **Map Skills** — Identifies which AI-velocity skills accelerate each phase of the build
4. **Runbook** — Produces a structured plan using the template in `references/runbook-template.md`

## Structure

```
hackathon-advisor/
├── SKILL.md                        # Main skill instructions (~120 lines)
├── DECISIONS.md                    # Why these constraints exist (5 decisions)
├── GUARDRAILS.md                   # Anti-patterns from past hackathons (5 guardrails)
├── references/
│   └── runbook-template.md         # Output template (loaded when producing runbook)
└── README.md                       # This file
```

## Design Principles

This skill follows the [SAGE principles](https://github.com/Vance-Club/aspora-cortex):

| Principle | How It's Applied |
|-----------|-----------------|
| **Scoped** | Hackathon planning only. Not production architecture, not skill authoring. |
| **Adaptive** | Explains WHY constraints exist (why PostgreSQL over Supabase, why Slack over web UI) so the LLM can reason about edge cases. |
| **Gradual** | SKILL.md is the entry point (~120 lines). Runbook template loads only when producing output. |
| **Evaluated** | Tested against real hackathon problem statements. Anti-patterns from observed team failures. |

## Key Constraints

These are documented in `DECISIONS.md` with full reasoning:

- **Go or Java only** — hackathon code that works becomes production code
- **PostgreSQL only** — our production database, no toy alternatives
- **Slack-first interface** — 1-2 hours vs 4-8 hours for web UI
- **SKILL.md first** — write the skill before the service code
- **25% rule** — plan for one flow, not five features
- **Demo Owner from hour 1** — not hour 22

## AI-Velocity Skills Referenced

This skill references these sibling skills by name in its runbooks:

| Skill | When Referenced |
|-------|---------------|
| `/standards-go` | Scaffolding Go services (hours 0-2) |
| `/dev-java` | Scaffolding Java services (hours 0-2) |
| `/test-go` | Writing Go tests (hours 16-20) |
| `/test-java` | Writing Java tests (hours 16-20) |
| `/observability-metrics` | Adding Prometheus metrics (hours 16-20) |
| `/observability-dashboard` | Creating Datadog dashboards (hours 16-20) |
| `/observability-alerts` | Setting up alerts (post-hackathon) |

## The Five Shipping Bars

Every recommended project must be validated against:

1. **Published Skill** with SKILL.md by demo day
2. **Real user, real problem** — not hypothetical
3. **AI-first stack** — Claude Code, Skills, Agent SDK
4. **Context layer contribution** — adds knowledge for the next person
5. **3-minute demo** — if value isn't clear in 3 minutes, scope is wrong

## Contributing

To improve this skill:

1. Run it on a real hackathon problem statement
2. Compare output against what the team actually needed
3. Record new anti-patterns in `GUARDRAILS.md`
4. Record new decisions in `DECISIONS.md`
5. Submit a PR

---

*Part of [AI Velocity](../../README.md) — production-grade AI skills for engineers.*
