Persist decisions and learnings from this session to the project's Decision Record Room.

## What To Do

Review your work in this session and identify:

1. **Decisions made** — architecture choices, technology picks, pattern selections
2. **Mistakes caught** — errors you made and corrected, anti-patterns you encountered
3. **Constraints discovered** — things that must NEVER/ALWAYS be true going forward

## How To Record

### For decisions → append to DECISIONS.md in project root

Find the last DEC number and increment it. Use this exact format:

```
## DEC-NNN: [Short title] ([today's date])
**Chose:** [what was picked]
**Over:** [what was rejected]
**Why:** [one sentence]
**Constraint:** [NEVER/ALWAYS rule]
```

### For mistakes → append to GUARDRAILS.md in project root

```
## From Incident: [title] ([today's date])
**Mistake:** [what happened]
**Impact:** [what broke or would have broken]
**Rule:** [NEVER/ALWAYS rule]
**Detection:** [how to spot this in future]
```

## Rules

- Each entry must be < 5 lines (~50 tokens)
- Constraints must be binary: NEVER or ALWAYS — no ambiguity
- If DECISIONS.md doesn't exist, create it with a `# Decisions` header
- If GUARDRAILS.md doesn't exist, create it with a `# Guardrails` header
- Maximum 20 decisions per project — if near limit, summarize older ones
- Do NOT record trivial choices (variable names, formatting) — only decisions that would affect the next agent
