# The Enhanced SKILL.md Template

> **When to load:** When creating a new skill and you need the template structure.
> **Source:** Extracted from SKILL_DESIGN_PRINCIPLES.md Part 7.

```markdown
---
name: [skill-name]
description: >
  [WHAT this skill does — one clear sentence]
  [WHEN to trigger — be generous, list phrases/contexts/intents]
  [WHEN NOT to trigger — explicit exclusions]
---

# [Skill Name]

## Domain Model

[2-4 paragraphs: core entity, what "good" looks like, invariants,
user's ultimate need. Use domain vocabulary.]

## Philosophy

[1-2 paragraphs: approach, tradeoffs, optimization target,
north star quality. Decision framework for ambiguous situations.]

## Quick Reference

| Task | Approach | When to Use |
|------|----------|-------------|
| [Common task 1] | [Method] | [Context] |

## Workflow

### Step 1: Understand the Input
[Classification, fact establishment]

### Step 2: Select the Approach
[Decision logic, lazy reference loading]
"For approach A, read references/approach-a.md"

### Step 3: Execute
[Core steps with WHY for non-obvious ones]

### Step 4: Verify
[Quality gates, invariant checks]

### Step 5: Deliver
[Output format, what to say/not say]

## Reasoning Guides

### Why [Important Decision X]
[Consequence of wrong choice → reasoning → edge cases]

## Mechanical Rules

- [RULE]: [requirement] — BECAUSE [consequence]

[Reserve firm constraints for correctness-critical rules only. Explain reasoning.]

## Common Failure Modes

| Failure | Symptom | Cause | Fix |
|---------|---------|-------|-----|
| [Name] | [What it looks like] | [Why] | [How to fix] |

## Dependencies & Resources

- **Required tools**: [list]
- **Reference files**: [with when-to-read guidance]
- **Related skills**: [with delegation patterns]
```

## Key Principles for the Template

- **Domain Model section** activates the LLM's existing knowledge — use the domain's actual vocabulary.
- **Philosophy section** gives the LLM a decision framework for ambiguous situations.
- **Workflow uses lazy loading** — reference files are loaded per-branch, not upfront.
- **Mechanical Rules include BECAUSE** — reasoning helps the LLM generalize to edge cases the rules don't explicitly cover.
- **Common Failure Modes** prevent recurring mistakes. Seed from GUARDRAILS.md if available.
