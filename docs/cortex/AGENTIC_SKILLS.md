# Agentic Skill Design — Quick Reference

> **When to read this file:** Starting work on skill-based projects. For deep design principles, see [SKILL_DESIGN_PRINCIPLES.md](./SKILL_DESIGN_PRINCIPLES.md).
>
> **Last updated:** 2026-02-14

---

## The SAGE Principles (Quick Reference)

| Principle | Meaning | Key Question |
|-----------|---------|--------------|
| **S — Scoped** | One skill, one domain. Clear triggers and boundaries. | Can I describe this skill in one sentence? |
| **A — Adaptive** | Explain WHY, not just rules. Let the LLM reason. | Am I writing ALWAYS/NEVER without explanation? |
| **G — Gradual** | Progressive disclosure. Lazy loading. <500 lines. | Am I loading references I might not need? |
| **E — Evaluated** | Test with real prompts. Measure. Prune ruthlessly. | If I remove this, does quality drop? |

---

## Description Field (Critical for Discovery)

The `description` is how Claude discovers your skill from 100+ options. This is the most important field.

### Must Include:
1. **WHAT** it does (one sentence)
2. **WHEN** to trigger (phrases users might say)
3. **WHEN NOT** to trigger (explicit exclusions)

### Format:
```yaml
description: >
  [What it does in one sentence]. Use when [trigger phrases].
  Do NOT use for [exclusions].
```

### Examples:

| ❌ Bad | ✅ Good |
|--------|---------|
| "Helps with documents" | "Extracts text from PDFs, fills forms, merges documents. Use when working with PDF files or when user mentions 'PDF', 'form filling', or 'document extraction'. Do NOT use for Word docs or spreadsheets." |
| "Processes data" | "Analyzes BigQuery datasets and generates reports. Use when asked about 'revenue metrics', 'sales data', or 'run a query'. Do NOT use for real-time analytics — use Pinot skill instead." |
| "Does instrumentation" | "Instruments multiple services with Prometheus metrics. Use when asked to 'add metrics to all services' or 'batch instrumentation'. Do NOT use for single-service work." |

### Rules:
- Write in **third person** ("Processes files", not "I can process files")
- Max 1024 characters
- Include domain-specific terms users actually say

---

## Skill Structure

```
skill-name/
├── SKILL.md             ← Entry point (<500 lines)
├── DECISIONS.md         ← Compressed ADRs (~50 tokens/decision, max 20)
├── GUARDRAILS.md        ← Anti-patterns from incidents (hard constraints)
├── scripts/             ← Deterministic tasks
├── references/          ← Loaded on demand
└── assets/              ← Templates, fonts
```

### Decision Record Room (Mandatory for Every Skill)

Every skill should have `DECISIONS.md` and `GUARDRAILS.md`. These are the agent's memory — without them, every new session starts cold and repeats mistakes.

**DECISIONS.md** — What we decided and why (~50 tokens per entry):
```markdown
## DEC-001: [Short title] ([date])
**Chose:** [What we picked]
**Over:** [What we rejected]
**Why:** [One sentence — the deciding factor]
**Constraint:** [NEVER/ALWAYS rule]
```

**GUARDRAILS.md** — What we must NEVER do (from past incidents):
```markdown
## From Incident: [title] ([date])
**Mistake:** [what happened]
**Impact:** [what broke]
**Rule:** [NEVER do X / ALWAYS do Y]
**Detection:** [how to spot this mistake]
```

**Rules:**
- Agents read these BEFORE starting work (Pre-Flight phase in every command)
- Max 20 decisions per project (~1,000 tokens total)
- Constraints are binary with reasoning: NEVER X because Y / ALWAYS X because Y
- Record a decision when: architecture choice made, technology chosen, pattern established, or a mistake taught us something
- Use `brain decisions init` to bootstrap, `brain decisions add "title"` to record

### YAML Frontmatter (Official Spec)

```yaml
---
name: skill-name                    # Required: max 64 chars, lowercase + hyphens only
description: >                      # Required: max 1024 chars, essential for discovery
  What it does. When to use. When NOT to use.
---
```

**Official Fields:**
| Field | Required | Notes |
|-------|----------|-------|
| `name` | Yes | Lowercase, hyphens, max 64 chars. No "anthropic" or "claude". |
| `description` | Yes | Max 1024 chars. Include triggers + exclusions. |
| `disable-model-invocation` | No | Set `true` to make user-only (e.g., `/deploy`) |
| `allowed-tools` | No | Tools Claude can use without permission when skill active |
| `context` | No | Set `fork` to run in isolated subagent |

**Non-standard fields are ignored:** `version`, `author`, `tags` don't do anything.

### Naming Convention

Prefer **gerund form** (verb + -ing):
- ✅ `processing-pdfs`, `instrumenting-services`, `analyzing-data`
- ⚠️ Acceptable: `pdf-processor`, `batch-instrumentation` (noun phrases)
- ❌ Avoid: `helper`, `utils`, `tools` (too vague)

**What NOT to include**: README.md, CHANGELOG.md — skills are for agents, not human onboarding.

---

## The Three Cognitive Layers

| Layer | What | In SKILL.md |
|-------|------|-------------|
| **WHAT** | Domain model, entities, invariants | "A financial model is a living tool for scenario analysis" |
| **WHY** | Reasoning, consequences, tradeoffs | "We use formulas because the user will change assumptions" |
| **HOW** | Steps, tools, validation | "Create workbook → define assumptions → build formulas" |

Most skills only have HOW. Good skills have WHY + HOW. Great skills have all three.

---

## Lazy Context Principles

| Pattern | Do This | Not This |
|---------|---------|----------|
| **Lazy References** | Load only when branch requires | Load all references upfront |
| **Deferred Tools** | Call when output needed | Query everything first |
| **Fact Caching** | Establish facts once, reference later | Re-analyze repeatedly |
| **Context GC** | Summarize intermediate work | Keep all details in context |
| **Skill Delegation** | Read dependent skill at delegation point | Load all skills upfront |

---

## Output Format Rule

**In skill-based frameworks, the deliverable is a SKILL, not code.**

| Request | ❌ Wrong | ✅ Correct |
|---------|----------|-----------|
| "Track SLA breaches" | Python script | `track-sla-breaches.md` skill |
| "Daily report" | Java scheduled job | `daily-report.md` skill |
| "Fetch stuck orders" | Go CLI tool | Skill + `queries/stuck-orders.sql` |

---

## Anti-Patterns (Quick Check)

| Pattern | Symptom | Fix |
|---------|---------|-----|
| God Skill | Handles everything | Split by bounded context |
| Rule Swamp | ALWAYS/NEVER without WHY | Explain reasoning |
| Context Hog | 2000+ line SKILL.md | Move to references/ |
| Eager Loader | Loads all refs upfront | Lazy load by branch |
| Overfitter | Works on tests only | Generalize patterns |
| Amnesia | Agents repeat same mistakes | Add DECISIONS.md + GUARDRAILS.md |
| Pointer Skill | Command says "read 4,000-line SKILL.md" | Phased execution protocol with gates |

---

## Skill Creation Checklist

**Step 0 — Decision Record Room (FIRST):**
- [ ] Run `brain decisions init` to create DECISIONS.md + GUARDRAILS.md
- [ ] Review existing decisions across related skills: `brain decisions scan`
- [ ] Record initial architecture decisions in DECISIONS.md
- [ ] If this skill learned from a prior incident, seed GUARDRAILS.md

**Before writing:**
- [ ] One-sentence purpose?
- [ ] Separate skill or variant of existing?
- [ ] Clear triggers defined?
- [ ] Output format specified?

**Writing:**
- [ ] Frontmatter has name + generous description
- [ ] <500 lines, references for details
- [ ] Explains WHY for non-obvious decisions
- [ ] Examples with input → output
- [ ] Command file uses phased execution protocol with Pre-Flight context loading

**Testing:**
- [ ] 2-3 realistic prompts that match description triggers
- [ ] Tested WITH and WITHOUT skill (baseline comparison)
- [ ] Checked transcripts for wasted effort (eager loading, re-analysis)

**Multi-Model Testing (Anthropic Standard):**
- [ ] **Haiku**: Does skill provide enough guidance? (fast, needs more detail)
- [ ] **Sonnet**: Is skill clear and efficient? (balanced)
- [ ] **Opus**: Does skill avoid over-explaining? (powerful, needs less hand-holding)

**Post-Ship:**
- [ ] Record any decisions made during implementation: `brain decisions add "title"`
- [ ] Record any mistakes discovered during testing: `brain decisions mistake "title"`

---

## Claude 4.x Behavior Awareness

Claude Opus 4.6 and Sonnet 4.5 behave differently from earlier models. Skills designed for older models may need adjustment.

### Overtriggering

Claude 4.x is more responsive to system prompts and skill descriptions. If your skill was designed to reduce undertriggering (aggressive language like "You MUST use this skill when..."), dial it back — these models trigger appropriately with normal prompting. Write descriptions in plain third-person language.

### Overengineering

Claude 4.x tends to create extra files, add unnecessary abstractions, and build in flexibility that wasn't requested. Skills should include guidance like: "Keep the solution minimal. Only implement what was requested."

### Excessive Exploration

Claude Opus 4.6 does significantly more upfront exploration than previous models. For skills with an analysis phase, set clear scope boundaries: "Analyze the 3 most relevant files, not the entire codebase."

### Subagent Overuse

Claude 4.6 may spawn subagents for tasks that a direct tool call would handle faster. If your skill involves simple file reads or searches, specify: "Use direct tool calls for file reads and searches. Use subagents only for independent parallel workstreams."

---

## Thinking and Skill Phases

Claude Opus 4.6 uses adaptive thinking — it decides when to reason deeply based on task complexity.

### Match instruction style to phase type:

| Phase Type | Thinking Behavior | Instruction Style |
|------------|------------------|-------------------|
| **Judgment-heavy** (triage, analysis, design) | Let Claude think deeply — don't rush it | Declare intent: "Determine the best approach for..." |
| **Mechanical** (file creation, boilerplate, config) | Claude should execute precisely, not over-think | Be specific: "Create file X with exactly these contents..." |
| **Verification** (self-check, validation) | Encourage reflection between tool results | "After completing analysis, reflect on whether findings are complete before proceeding." |

### After receiving tool results:

Encourage Claude to reason about intermediate results before acting: "Review the tool output and determine the optimal next step before proceeding." This aligns with interleaved thinking — Claude can think between tool calls for better sequential decisions.

### When thinking adds unnecessary cost:

For simple skills (< 5 steps, no branching), add: "This is a straightforward task. Respond directly without extensive reasoning."

---

## Context Rot Prevention

Context degrades in four ways. Skills should be designed to mitigate each:

| Degradation Type | What Happens | How to Prevent in Skills |
|-----------------|--------------|--------------------------|
| **Context Poisoning** | Incorrect/outdated info causes faulty reasoning | GUARDRAILS.md with dated entries. Pre-Flight phase reads these first. |
| **Context Distraction** | Irrelevant info reduces focus on key details | Lazy loading — load references only when the execution branch needs them. Don't dump domain context for a different domain. |
| **Context Confusion** | Similar but distinct info creates association errors | Use precise domain vocabulary. Name things explicitly — "payment order" not "order" when both exist. |
| **Context Clash** | Contradictory info leaves Claude uncertain | Keep DECISIONS.md current. When a decision changes, update (don't append a contradiction). |

### Token budget awareness:

- SKILL.md body: < 500 lines (~2-3K tokens when loaded)
- References: loaded on-demand, not upfront
- CORTEX.md (46K tokens) and SKILL_DESIGN_PRINCIPLES.md (32K tokens) should never be loaded into a system prompt — they are deep reference files
- The compressed CLAUDE.md (~2K tokens) is what belongs in the system prompt

---

## Loading Rules (Enforced)

| File | Allowed Usage | Token Cost |
|------|--------------|------------|
| `CLAUDE_GLOBAL.md` | System prompt / CLAUDE.md | ~2K tokens |
| `AGENTIC_SKILLS.md` | Load when starting skill work | ~3K tokens |
| `CORTEX.md` | Reference only — search for specific sections | ~46K tokens |
| `SKILL_DESIGN_PRINCIPLES.md` | Reference only — read Parts 1-6 for core, Parts 7+ on demand | ~32K tokens |
| `DOMAIN_CONTEXT.md` | Load only for domain-specific work | ~1K tokens |
| `DECISIONS.md` / `GUARDRAILS.md` | Always read in Pre-Flight (small, < 1K tokens) | ~0.5K tokens |

Agents should never load CORTEX.md or SKILL_DESIGN_PRINCIPLES.md in their entirety. Search for the relevant section and read only that.

---

## Deep Dive

For comprehensive coverage of:
- DDD → Skill-Driven Design translation
- SOLID → SKILL principles
- Clean Code → Clean Skill principles
- Full Lazy Loading patterns (LC1-LC10)
- Enhanced SKILL.md template
- Real-world skill anatomy analysis

**See [SKILL_DESIGN_PRINCIPLES.md](./SKILL_DESIGN_PRINCIPLES.md)**
