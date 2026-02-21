# Anthropic Official Standards (2026)

> **When to load:** When you need to verify your skill against Anthropic's canonical spec.
> **Source:** Extracted from SKILL_DESIGN_PRINCIPLES.md Appendix A. Updated 2026-02-16 with Claude 4.x best practices.

## A.1 YAML Frontmatter Specification

```yaml
---
name: skill-name          # Required: max 64 chars
description: >            # Required: max 1024 chars
  What it does. When to use. When NOT to use.
---
```

**Validation Rules:**
| Field | Constraint |
|-------|------------|
| `name` | Max 64 characters |
| `name` | Lowercase letters, numbers, hyphens only |
| `name` | Cannot contain "anthropic" or "claude" |
| `description` | Non-empty, max 1024 characters |
| `description` | Cannot contain XML tags |

**Optional Fields:**
| Field | Purpose |
|-------|---------|
| `disable-model-invocation` | `true` = user-only, Claude cannot auto-trigger |
| `user-invocable` | `false` = Claude-only, hidden from `/` menu |
| `allowed-tools` | Tools Claude can use without permission when active |
| `context` | `fork` = run in isolated subagent |
| `agent` | Subagent type when `context: fork` (e.g., `Explore`, `Plan`) |

Non-standard fields are ignored. Don't use `version`, `author`, `tags`.

---

## A.2 Naming Conventions

Prefer gerund form (verb + -ing):
- `processing-pdfs`, `analyzing-spreadsheets`, `instrumenting-services`

Acceptable: `pdf-processing`, `batch-instrumentation` (noun phrases)

Avoid: `helper`, `utils`, `tools` (too vague), `anthropic-helper`, `claude-tools` (reserved)

---

## A.3 Description Writing

The description is how Claude selects your skill from 100+ options. It must answer:
1. WHAT does this skill do?
2. WHEN should Claude trigger it? (include user phrases)
3. WHEN NOT should Claude use it? (explicit exclusions)

Write in third person. Include domain vocabulary users actually say. Include explicit exclusions.

**Claude 4.x note:** These models are more responsive to descriptions. If your skill was designed for older models with aggressive trigger language ("You MUST use this..."), dial it back. Normal, clear descriptions work well.

---

## A.4 Feedback Loops (Validation Pattern)

For code-based skills: Run validator → fix errors → repeat.
For non-code skills: Check output against requirements → revise → check again.

---

## A.5 Multi-Model Testing

| Model | Characteristics | Test Focus |
|-------|----------------|------------|
| **Haiku** | Fast, economical, needs more guidance | Does skill provide enough detail? |
| **Sonnet** | Balanced | Is skill clear and efficient? |
| **Opus** | Powerful reasoning, may over-think | Does skill avoid over-explaining? Does it keep solutions minimal? |

Evaluation-Driven Development:
1. Create evaluations before writing extensive documentation
2. Build 3+ scenarios that test real gaps
3. Measure baseline performance without skill
4. Write minimal instructions to pass evaluations
5. Iterate based on observed behavior

---

## A.6 Context Window Awareness

The context window is a shared resource:
- Skill descriptions: Always loaded (keep concise)
- SKILL.md body: Loaded when triggered (< 500 lines)
- References: Loaded on-demand only

Default assumption: Claude is already smart. Only add context it doesn't have.
Test each line: "If I remove this, does output quality measurably drop?"

Claude 4.6 tracks its remaining context budget. Context is automatically compacted when approaching limits. Skills should not assume Claude will stop early — encourage task completion.

---

## A.7 Degrees of Freedom

Match instruction specificity to task fragility:

| Freedom Level | When to Use | Example |
|---------------|-------------|---------|
| **High** (general guidance) | Multiple valid approaches, context-dependent | "Analyze code structure and suggest improvements" |
| **Medium** (templates/pseudocode) | Preferred pattern exists, some variation OK | "Use this template, customize as needed" |
| **Low** (exact scripts) | Fragile operations, consistency critical | "Run exactly: `python migrate.py --verify`" |

Analogy: Narrow bridge with cliffs → precise instructions. Open field → general direction.

---

## A.8 Claude 4.x Behavior (Added 2026-02-16)

Key differences from earlier models that affect skill design:

| Behavior | Impact on Skills | Mitigation |
|----------|-----------------|------------|
| **Overtriggering** | Skills activate when they shouldn't | Add explicit "Do NOT use for..." exclusions. Remove aggressive language. |
| **Overengineering** | Claude adds abstractions beyond scope | Add: "Keep solutions minimal. Only implement what was requested." |
| **Excessive exploration** | Burns tokens exploring before acting | Set scope: "Analyze the 3 most relevant files, not the entire codebase." |
| **Subagent overuse** | Spawns agents for simple tasks | "Use direct tool calls for searches. Subagents for parallel workstreams only." |
| **Adaptive thinking** | Thinks deeply on complex tasks, skips for simple ones | Match instruction style to phase type (judgment vs mechanical). |

---

## A.9 Sources

- [Prompting best practices (Claude 4.x)](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
- [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- [Think tool research](https://www.anthropic.com/engineering/claude-think-tool)
- [A complete guide to building skills for Claude](https://claude.com/blog/complete-guide-to-building-skills-for-claude)
- [Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- [Extend Claude with skills (Claude Code)](https://code.claude.com/docs/en/skills)
