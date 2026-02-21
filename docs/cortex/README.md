# Aspora Cortex
> How we build.

A shared engineering operating system for AI-assisted development at Aspora. It standardizes how AI agents (Claude Code, Cursor, Codex, Copilot) write code, design systems, and make decisions across the team.

**Core Philosophy: Velocity = Speed x Direction.**
Build production-grade, reusable systems. Prove by building, not debating.

---

## What's Inside

| File | Purpose | When to Read |
|------|---------|--------------|
| `CLAUDE.md` / `CLAUDE_GLOBAL.md` | Compressed principles for Claude Code | Always (auto-loaded) |
| `CORTEX.md` | Full engineering system — identity, principles, patterns | Deep work, architecture decisions |
| `AGENTIC_SKILLS.md` | Quick reference for skill design (SAGE, structure, checklists) | Starting skill work |
| `SKILL_DESIGN_PRINCIPLES.md` | Deep dive — DDD-to-Skill, SOLID-to-SKILL, Lazy Loading | Designing new skills |
| `CODEX.md` | OpenAI Codex agent instructions | When using Codex |
| `cursorrules_global.txt` | Cursor AI global rules | When using Cursor |

---

## Quick Start (For Team Members)

### 1. Clone This Repo

```bash
git clone git@github.com:Vance-Club/aspora-cortex.git ~/code/aspora-cortex
```

### 2. Set Up for Your AI Tool

#### Claude Code (Recommended)

Claude Code auto-loads `CLAUDE.md` from two locations:

| Location | Scope | What to Do |
|----------|-------|------------|
| `~/CLAUDE.md` | All your projects | Copy `CLAUDE_GLOBAL.md` to `~/CLAUDE.md` |
| `<project>/CLAUDE.md` | One project (shared via git) | Copy `CLAUDE_GLOBAL.md` to your project root as `CLAUDE.md` |

```bash
# Option A: Global (applies to everything you work on)
cp ~/code/aspora-cortex/CLAUDE_GLOBAL.md ~/CLAUDE.md

# Option B: Per-project (team shares via git)
cp ~/code/aspora-cortex/CLAUDE_GLOBAL.md ~/code/my-service/CLAUDE.md
cd ~/code/my-service && git add CLAUDE.md && git commit -m "docs: add Aspora Cortex standards"
```

Run the install script to set up everything:

```bash
./install.sh
```

#### Cursor

```bash
# Copy to your project root as .cursorrules
cp ~/code/aspora-cortex/cursorrules_global.txt ~/code/my-service/.cursorrules
```

#### OpenAI Codex

```bash
# Copy to your project root as AGENTS.md (Codex convention)
cp ~/code/aspora-cortex/CODEX.md ~/code/my-service/AGENTS.md
```

### 3. Verify It Works

In Claude Code, start a new conversation and ask:

```
What engineering principles are you following for this project?
```

Claude should reference SOLID, Clean Architecture, Four Golden Signals, Zero Logic Mutation, and your iterative rhythm.

---

## How Agents Use Skills

Skills are **reusable instructions** that turn AI agents into domain experts on demand. Instead of re-prompting Claude with the same 50-line context every time, you package it once as a skill.

### Discovery

When a user types a prompt, Claude scans all skill descriptions to find the best match:

```
User: "add metrics to all our payment services"
         │
         ▼
Claude reads all skill descriptions
         │
         ▼
Matches: batch-prometheus-instrumentation
  "Instruments multiple services with Prometheus metrics..."
  "Use when asked to 'instrument all services', 'add metrics to the fleet'..."
         │
         ▼
Claude loads the skill's SKILL.md and follows its workflow
```

This is why the `description` field is the most important part of any skill — it's the search index.

### Execution

Once triggered, a skill guides the agent through a structured workflow:

1. **Pre-Flight** — Read `DECISIONS.md` and `GUARDRAILS.md` (don't repeat past mistakes)
2. **Analyze** — Scan the codebase, infer patterns, classify the task
3. **Execute** — Follow the skill's phased workflow with lazy context loading
4. **Verify** — Run validation loops (build, test, check output)
5. **Deliver** — Create the output (code, PR, report, artifact)

### Composition

Skills can delegate to other skills at execution time (lazy, not upfront):

```
monitoring-setup (orchestrator skill)
  │
  ├─ When Java service detected → delegates to observability-metrics skill
  ├─ When Go service detected   → delegates to observability-metrics skill (Go path)
  ├─ When dashboard needed      → delegates to observability-dashboard skill
  └─ When alerts needed         → delegates to observability-alerts skill
```

### User-Invoked vs. Agent-Invoked

| Type | How It's Triggered | YAML Field |
|------|-------------------|------------|
| **User-invoked** | User types `/skill-name` | Default behavior |
| **Agent-invoked** | Claude auto-triggers from prompt | Default behavior |
| **User-only** | Only via `/skill-name`, Claude can't auto-trigger | `disable-model-invocation: true` |
| **Agent-only** | Claude uses it internally, hidden from `/` menu | `user-invocable: false` |

---

## Writing a New Skill

### When to Create a Skill

Create a skill when:
- The workflow will be **repeated** (not a one-off)
- Multiple team members need to do the **same thing consistently**
- You find yourself re-prompting the same context repeatedly
- An operational procedure should be executable by any agent or team member

Do NOT create a skill when:
- It's a one-time task
- The workflow is too simple to benefit from structure (< 5 steps)
- The domain changes too frequently to keep the skill current

### Skill File Structure

```
skills/
└── detecting-fincrime-alerts.md    ← Single-file skill (most common)
```

Or for complex skills with references:

```
skills/detecting-fincrime-alerts/
├── SKILL.md                         ← Entry point (< 500 lines)
├── DECISIONS.md                     ← What we decided and why
├── GUARDRAILS.md                    ← What we must NEVER do
├── references/
│   ├── sar-filing-rules.md          ← Loaded only when SAR path taken
│   └── sanctions-screening.md       ← Loaded only when sanctions path taken
└── queries/
    └── suspicious-transactions.sql  ← Reusable SQL
```

### Step-by-Step: Creating the `detecting-fincrime-alerts` Skill

Here's a real example — a skill that helps agents triage and investigate financial crime alerts.

#### Step 1: Define the Boundary

Ask yourself:
- **One sentence**: "Triage and investigate financial crime alerts from the transaction monitoring system."
- **What's IN**: Alert triage, investigation steps, SAR decision, escalation
- **What's OUT**: Filing SARs (different skill), updating sanctions lists, modifying detection rules

#### Step 2: Write the YAML Frontmatter

The `description` is the most critical field — it determines when Claude triggers this skill.

```yaml
---
name: detecting-fincrime-alerts
description: >
  Triages and investigates financial crime alerts from transaction monitoring
  systems. Use when asked to 'review fincrime alerts', 'investigate suspicious
  transactions', 'triage AML alerts', 'check transaction monitoring queue', or
  'handle compliance alerts'. Do NOT use for filing SARs (use sar-filing skill),
  updating detection rules, or sanctions list management.
---
```

#### Step 3: Write the Domain Model (WHAT)

This activates the LLM's existing knowledge about the domain:

```markdown
# Detecting FinCrime Alerts

## Domain Model

A financial crime alert is a signal from the transaction monitoring system that
a customer's behavior matches a suspicious pattern. Not every alert is a true
positive — the investigator's job is to determine whether the activity is
genuinely suspicious or has a legitimate explanation.

Key entities:
- **Alert**: Generated by rules engine. Has severity (High/Medium/Low), rule ID,
  trigger amount, and customer context.
- **Case**: Created when an alert requires investigation. Links to customer
  profile, transaction history, and prior alerts.
- **SAR Decision**: The outcome — file, escalate, or dismiss with documented
  rationale.

Invariants:
- Every alert gets a disposition (no silent dismissals)
- Investigation rationale is documented before any disposition
- High-severity alerts are reviewed within 24 hours
- Escalation thresholds follow the compliance matrix
```

#### Step 4: Write the Reasoning Framework (WHY)

This is what separates a good skill from a great one:

```markdown
## Philosophy

We investigate alerts to protect the business and comply with regulations, but
we optimize for **investigator efficiency** — not just compliance checkbox
completion. An investigation that takes 2 hours when it should take 20 minutes
is a failure even if the SAR is filed correctly.

We triage before investigating because:
- 60-70% of alerts are false positives from legitimate business patterns
- Spending investigation time on obvious false positives burns out analysts
- Risk-based prioritization ensures high-severity alerts get attention first

We document rationale because:
- Regulators audit investigation quality, not just volume
- "No suspicious activity" without explanation is a regulatory finding
- Future investigators reviewing the same customer need prior context
```

#### Step 5: Write the Workflow (HOW)

Use phased execution with lazy loading:

```markdown
## Workflow

### Phase 1: Triage (Batch)
For each alert in the queue:
1. Read alert details: rule ID, trigger amount, customer segment
2. Check customer's alert history (prior alerts, prior SARs)
3. Classify: **Obvious False Positive** / **Needs Investigation** / **Obvious True Positive**
4. Prioritize investigation queue by severity and age

### Phase 2: Investigate (Per Alert)
For alerts classified as "Needs Investigation":

1. **Customer Profile Review**
   - Account age, segment, expected activity pattern
   - Prior alerts and their dispositions
   - KYC status and last refresh date

2. **Transaction Analysis**
   - Pull transaction history for alert period + 30 days prior
   - Identify the specific transactions that triggered the alert
   - Compare against customer's baseline activity
   - Flag structuring patterns, round amounts, rapid movement

3. **Contextual Assessment**
   - Does the activity match known typologies?
   - Is there a legitimate business explanation?
   - Are counterparties on any watchlists?

   > For sanctions-related alerts, read `references/sanctions-screening.md`
   > For structuring patterns, read `references/structuring-typologies.md`

4. **Disposition Decision**
   - **Dismiss**: Document the legitimate explanation
   - **Escalate**: Flag for senior investigator review
   - **SAR Recommended**: Prepare investigation summary for SAR filing team

### Phase 3: Document
For every alert (including dismissals):
- Record investigation steps taken
- Document rationale for disposition
- Note any follow-up actions required
- Update customer risk profile if warranted

## Verification
- [ ] Every alert has a documented disposition
- [ ] High-severity alerts reviewed within 24 hours
- [ ] Investigation rationale would survive regulatory audit
- [ ] No alerts left in "pending" status without assignment
```

#### Step 6: Add Guardrails

```markdown
## Mechanical Rules

- NEVER dismiss a high-severity alert without senior review — BECAUSE
  regulators specifically audit high-severity dispositions
- NEVER copy-paste rationale between alerts — BECAUSE each investigation
  must reflect the specific facts of that alert
- ALWAYS check customer's prior alert history before investigating — BECAUSE
  repeat alerts on the same customer compound risk
- ALWAYS document what you checked, even if you found nothing — BECAUSE
  "no findings" without methodology is a regulatory gap

## Common Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Rubber-stamping | Same rationale on 50 alerts | Require specific transaction references |
| Investigation tunnel vision | Only looked at triggered transactions | Always review 30-day window |
| Missing context | Dismissed alert on repeat offender | Always check prior alert history |
| Over-escalation | Everything marked "needs senior review" | Use severity matrix, trust triage |
```

### The SAGE Checklist (Before Shipping)

Every skill should pass these checks:

| Principle | Question | Your Skill |
|-----------|----------|------------|
| **S — Scoped** | Can you describe it in one sentence? Does it have clear exclusions? | Yes |
| **A — Adaptive** | Does it explain WHY, not just rules? Can the LLM reason about edge cases? | Yes |
| **G — Gradual** | Is it < 500 lines? Are references loaded lazily? | Yes |
| **E — Evaluated** | Have you tested with 2-3 real prompts? Does removing any section drop quality? | Test it |

### Multi-Model Testing

Test your skill across model tiers:

| Model | What to Check |
|-------|---------------|
| **Haiku** (fast, needs guidance) | Does the skill provide enough detail for Haiku to follow? |
| **Sonnet** (balanced) | Is the workflow clear and efficient? |
| **Opus** (powerful reasoning) | Does the skill avoid over-explaining things Opus already knows? |

---

## Sharing Skills Across Projects

Skills (reusable AI agent workflows) live in [`Vance-Club/ai-velocity`](https://github.com/Vance-Club/ai-velocity), not this repo. This repo provides the **standards and framework** for creating and using skills.

### Where Skills Live

| Location | What Goes There |
|----------|----------------|
| `Vance-Club/ai-velocity` | Shared skills (observability, testing, standards) |
| `<project>/.claude/commands/` | Project-specific skills (deploy, domain workflows) |
| This repo (`Vance-Club/aspora-cortex`) | Skill design framework + engineering standards |

### Per-Project Setup

```
my-service/
├── .claude/
│   └── commands/
│       └── deploy-service.md    ← Project-specific skill
├── CLAUDE.md                    ← Points to Aspora Cortex standards
└── src/
```

---

## Updating Cortex

This is a living system. Update it when:

- A project retro reveals a new pattern or anti-pattern
- A new skill proves valuable and should be shared
- An existing principle is invalidated by experience
- The team agrees on a new standard

```bash
# After making changes
git add -A
git commit -m "docs: add fincrime alerts skill from Q1 investigation automation"
git push
```

Team members pull updates and re-run the install script.

---

## File Reference

### For Deep Dives

| Topic | File | Key Content |
|-------|------|-------------|
| Full engineering principles | `CORTEX.md` | Identity, first principles, SOLID, observability, language patterns, scale design |
| Skill design quick ref | `AGENTIC_SKILLS.md` | SAGE principles, YAML spec, decision records, anti-patterns |
| Skill design deep dive | `SKILL_DESIGN_PRINCIPLES.md` | DDD-to-Skill, SOLID-to-SKILL, Lazy Loading LC1-LC10, cognitive layers |

### Key Concepts

**SAGE Principles** — How to design skills:
- **S**coped: One skill, one domain. Clear boundaries.
- **A**daptive: Explain WHY, let the LLM reason.
- **G**radual: Progressive disclosure. Lazy loading. < 500 lines.
- **E**valuated: Test with real prompts. Prune what doesn't help.

**Three Cognitive Layers** — What makes a great skill:
- **WHAT**: Domain model, entities, invariants
- **WHY**: Reasoning, consequences, tradeoffs
- **HOW**: Steps, tools, validation

Most skills only have HOW. Good skills have WHY + HOW. Great skills have all three.

**Lazy Context (LC1-LC10)** — Don't waste the context window:
- Load references only when the execution branch needs them
- Defer tool calls until output is needed
- Establish facts once, reference throughout
- Summarize intermediate work, don't keep raw details

---

## FAQ

**Q: Do I need all these files in every project?**
No. Only `CLAUDE.md` (or `.cursorrules`) goes in every project. The rest lives in this central repo for reference.

**Q: What if my project needs different principles?**
Override in the project's `CLAUDE.md`. The project-level file takes precedence. Add a `DECISIONS.md` to document why you diverged.

**Q: How do I know if a skill is working?**
Test it: give Claude a prompt that should trigger the skill and check the output. Compare with and without the skill. If quality doesn't improve, the skill needs work.

**Q: Can I use this with non-Anthropic tools?**
Yes. `cursorrules_global.txt` works with Cursor, `CODEX.md` works with OpenAI Codex, and the Cortex principles are tool-agnostic.
