# Skill Anti-Patterns Catalog

> **When to load:** When debugging skill behavior or reviewing skill quality.
> **Source:** Extracted from SKILL_DESIGN_PRINCIPLES.md Part 8.

## Anti-Patterns

| Anti-Pattern | Symptom | Fix | Traditional Equivalent |
|--------------|---------|-----|----------------------|
| **God Skill** | One skill handles everything in a domain | Split by bounded context | God Object |
| **Rule Swamp** | Hundreds of ALWAYS/NEVER with no WHY | Group by concern, explain reasoning | Spaghetti code |
| **Leaky Abstraction** | Claims to handle X but fails on subset | Handle full scope or exclude explicitly | Leaky Abstraction |
| **Context Hog** | 2,000+ line SKILL.md | Move details to references/ | Monolithic class |
| **Overfitter** | Works on tests, fails on real inputs | Generalize patterns, don't special-case | ML overfitting |
| **Silent Dependency** | Mysterious failures from missing deps | Explicit dependencies section | "Works on my machine" |
| **Judgment Suppressor** | Every decision micromanaged | Declare intent, trust reasoning | Over-engineering |
| **Eager Loader** | Loads all references upfront | Lazy loading by execution branch | N+1 query problem |
| **Context Amnesiac** | Re-analyzes same facts repeatedly | Explicit fact caching | Redundant computation |
| **Overtrigger** | Skill activates for unrelated requests | Tighten description exclusions; on Claude 4.x, dial back aggressive trigger language | False positive |
| **Context Rot** | Quality degrades over long sessions | Pre-Flight reads GUARDRAILS.md; fact caching; re-verify from files not memory | Memory leak |

## Diagnosis Guide

### "My skill works on test prompts but fails on real user requests"
**Likely anti-pattern:** Overfitter
**Check:** Are your instructions too specific to your test cases? Generalize the patterns.

### "Claude uses my skill when it shouldn't"
**Likely anti-pattern:** Overtrigger
**Check:** Does your description have explicit exclusions ("Do NOT use for...")? On Claude 4.x, remove aggressive trigger language like "You MUST use this when..."

### "Claude ignores important parts of my skill"
**Likely anti-pattern:** Context Hog or Eager Loader
**Check:** Is your SKILL.md over 500 lines? Are you loading all references upfront? Move content to references/ and load lazily.

### "Claude repeats the same mistake across sessions"
**Likely anti-pattern:** Context Amnesiac (no DECISIONS.md/GUARDRAILS.md)
**Check:** Does the skill have GUARDRAILS.md? Does the command file have a Pre-Flight phase that reads it?

### "Claude over-thinks simple steps"
**Likely cause:** Claude 4.x adaptive thinking + vague instructions
**Fix:** Make mechanical steps precise and specific. Add: "This is a straightforward step. Execute directly."
