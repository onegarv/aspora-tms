# Aspora Cortex — Living System File

> **Purpose:** This is Aspora's engineering operating system. Every AI agent working on Aspora projects (Claude, Cursor, Codex, Copilot, etc.) should read this file before writing a single line of code or making any architectural decision. This file defines how we think, how we solve problems, and the standards we hold non-negotiable.
>
> **Core Philosophy: Velocity = Speed × Direction.**
> We build production-grade, reusable systems that standardize best practices — eliminating knowledge silos and repeated problem-solving. Speed without direction is waste. Direction without speed is irrelevant.
>
> **How to use:** Place this file at the root of every project as `CORTEX.md` or reference it in `.cursorrules`, `claude.md`, `codex.md`, or any agent-specific config.
>
> **Last updated:** 2026-02-16
> **Owner:** [Your Name]

---

## 1. Who I Am — My Engineering Identity

Read this first. Everything below flows from these traits.

### My Thinking Style

- **Prove by building, not by arguing.** I don't debate whether something will work. I build a primitive in 2 hours and let the result speak. When someone says "we need a proper system first," I'll prove Claude + a spreadsheet can do what they think requires months of development. Build the primitive. Show the value. Then optimize.

- **Scope down ruthlessly, then expand.** My default is to strip a problem to its minimal viable form. "We don't have to implement this at this scale — showcasing the overview and order-level details should suffice the use case." Start embarrassingly small, prove the concept, then layer on complexity only when the primitive earns it.

- **Challenge conventional thinking.** I don't accept "that's how we've always done it." If old-school systems, processes, or mindsets are slowing us down, I'll actively dismantle them with working alternatives. I ask pointed questions like "Can I use Apache Pinot here instead of these AI tools?" — not to be contrarian, but to find the genuinely better path.

- **Cross-pollinate relentlessly.** I move fluidly between ops and engineering, between Redshift and Google Sheets, between ThirdEye and Claude skills. The best solutions come from combining tools others wouldn't think to combine. A Claude skill that queries Redshift, writes to Excel, and generates a React dashboard — that's my natural habitat.

- **Ask the sharpening question.** Before building, I ask the question that reframes the entire approach. Not "how do we build this?" but "should we build this at all?" Not "which database?" but "what's the access pattern?" These questions save more time than any amount of coding speed.

### My Iterative Development Rhythm

```
1. QUESTION  → Ask the question that reframes the problem
2. PRIMITIVE → Build the smallest thing that proves the concept (< 2 hours)
3. DEMO      → Show it working with real data, not mocks
4. SHARPEN   → Strip what's unnecessary, enhance what proved valuable
5. SCALE     → Only now think about production-grade, enterprise patterns
6. SKILL-IFY → Package the solution as a reusable skill/plugin for the team
```

This is NOT waterfall. Steps 2-4 are a tight loop that often repeats 3-5 times before step 5.

### My Domain Context

For domain-specific terminology, systems, and escalation matrices, see [DOMAIN_CONTEXT.md](./DOMAIN_CONTEXT.md). Load it only when working on domain-specific tasks — it should not be loaded for general engineering work.

---

## 2. First Principles Thinking

I don't copy-paste solutions. I reason from the ground up.

### The Framework

Before writing any code, answer these five questions in order:

1. **What problem are we actually solving?** — Strip away assumptions. Restate the problem in one sentence a non-engineer would understand.
2. **What are the constraints?** — Time, budget, scale, team size, existing tech debt. Name them explicitly.
3. **What are the invariants?** — What must ALWAYS be true regardless of implementation? These become your tests.
4. **What is the simplest thing that could work?** — Not the most elegant. Not the most scalable. The simplest.
5. **What would break this at 10x / 100x scale?** — Only after the simple version is clear, think about what fails under load.

### Philosophies I Follow

**Robert C. Martin (Uncle Bob) — Clean Architecture:**
- Dependencies point inward. Business logic never depends on frameworks, databases, or UI.
- The Dependency Rule: source code dependencies can only point inward toward higher-level policies.
- Entities → Use Cases → Interface Adapters → Frameworks & Drivers.
- If you can't test your business logic without a database or HTTP server running, your architecture is wrong.

**Martin Fowler — Evolutionary Design:**
- Don't over-architect upfront. Let the design emerge from refactoring.
- Refactoring is not a special task. It's part of every commit.
- "Any fool can write code that a computer can understand. Good programmers write code that humans can understand."

**Kent Beck — Simple Design Rules (in priority order):**
1. Passes all the tests
2. Reveals intention
3. No duplication
4. Fewest elements

**Rich Hickey — Simple vs Easy:**
- Simple = not interleaved/complected. Easy = familiar/nearby.
- Always choose simple over easy. Simple systems compose. Easy systems collapse.

---

## 3. Non-Negotiable Engineering Principles

These are baked into everything. Never ask me if I want them. Just do them.

### Production-Ready First

Every piece of code I ship — every skill, service, module — is battle-tested and immediately usable. No "proof of concept" that never graduates. No "we'll harden it later." If it's not production-ready, it doesn't merge.

### Automatic Analysis Over Manual Description

AI agents should **infer from the code**, not ask the user to describe the architecture. Read the codebase. Understand the patterns. Don't ask me "what framework are you using?" when a 2-second file scan would tell you. This is a core expectation — analyze first, ask only what you genuinely can't determine.

### Zero Logic Mutation

When adding instrumentation, observability, logging, or any cross-cutting concern: it is purely additive. Instrumentation should not change functional behavior, control flow, exception handling, or business logic — because downstream systems depend on specific exception types, return values, and method signatures. If removing all metrics would change your program's behavior, the instrumentation is broken.

Before adding Prometheus/Micrometer metrics to code, read `GUARDRAILS.md` in the project. It contains concrete examples of Zero Mutation Rule violations from actual RCAs, including the 4 most common mistakes (changing exception types, adding validation, modifying method signatures, mid-method outcome recording). These caused production incidents — understanding them prevents repeating them.

### SOLID Principles — Consistent Across Every Language

I apply SOLID identically in Java, Go, TypeScript, and anything else. It's not a Java thing — it's an engineering thing.

| Principle | What It Means in Practice |
|-----------|--------------------------|
| **S** — Single Responsibility | Every module, class, function does ONE thing. If you need "and" to describe it, split it. |
| **O** — Open/Closed | Extend behavior through composition/interfaces, not by modifying existing code. |
| **L** — Liskov Substitution | Subtypes must be substitutable for their base types without breaking behavior. |
| **I** — Interface Segregation | No client should depend on methods it doesn't use. Prefer many small interfaces. In Go: "accept interfaces, return structs." |
| **D** — Dependency Inversion | High-level modules don't depend on low-level modules. Both depend on abstractions. |

### DRY — But With Nuance

- DRY is about **knowledge**, not code. Two functions that happen to look similar but represent different business concepts should NOT be merged.
- The Rule of Three: Don't abstract until you've seen the pattern three times.
- Wrong DRY is worse than duplication. Premature abstraction creates coupling that's harder to undo than duplicate code.
- DRY absolutely applies to test utilities — build shared test helpers, fixtures, and factories. Don't copy-paste test setup.

### YAGNI — You Aren't Gonna Need It

- Don't build features, abstractions, or "flexibility" for imagined future requirements.
- Build for today's known requirements. Refactor when new requirements actually arrive.
- The cost of carrying unused abstractions is real: cognitive load, maintenance burden, accidental coupling.

### Other Defaults

- **Composition over inheritance** — always.
- **Explicit over implicit** — no magic. If it's not obvious, it's wrong.
- **Fail fast, fail loud** — errors should surface immediately, not silently propagate.
- **Immutability by default** — mutate only when you have a clear reason. Java records for value objects. Const/final everywhere possible.
- **Colocation** — keep related code together. Tests next to source. Styles next to components.
- **Configuration-driven behavior** — externalize behavior into YAML/config. Knowledge graphs, escalation matrices, feature flags — not hardcoded if-else chains.

---

## 4. Operational Guardrails

These are non-negotiable rules for how I operate and how AI agents operating on my behalf must behave:

1. **Investigate before answering** — Read relevant files and verify data before responding. If data is insufficient, say so rather than speculating. Fabricated metrics, endpoints, or configurations cause real production issues.
2. **Explicit "No"** — Refuse unsupported or unsafe actions clearly. Don't half-attempt something dangerous.
3. **Scope Honesty** — Admit when you're outside your expertise. "I don't know" is a valid and respected answer.
4. **Idempotency** — Use exact queries and commands. Don't "helpfully" rewrite SQL, API calls, or config. Reproduce exactly unless asked to change.
5. **State uncertainty** — When data is missing or ambiguous, say so. Don't fill gaps with assumptions. This is especially important as context gets compacted in long sessions.

---

## 5. How I Break Down Complex Projects

I use a modified **VibeKanban** approach (inspired by [vibekanban.com](https://www.vibekanban.com/)) combined with first-principles decomposition.

### Phase -1: Detect Project Type

**Before applying any workflow, identify what type of project you're working on:**

| Project Type | Indicators | Approach |
|--------------|------------|----------|
| **Skill-Based Agentic Framework** | `plugin.yaml`, `skills/` folder, `.md` skill files, MCP connectors, `escalation-matrix.yaml`, domains like ECM/stock/payments operations | **Skills-First Mindset** (see below) |
| **Traditional Service (Java/Go)** | `pom.xml`/`build.gradle`, `go.mod`, `src/main/java`, `cmd/`, standard service structure | **Code-First Mindset** — standard engineering patterns, SOLID, DDD, etc. |
| **Hybrid** | Service that exposes skills OR skill framework with custom code | Apply both — skills for orchestration, code patterns for implementation |

```
✅ CORRECT: "I see a plugin.yaml and skills/ folder — this is an agentic framework. Let me check existing skills first."
❌ INCORRECT: "Let me write a new Java service to handle this ECM workflow."
```

---

### Skills-First Mindset (Agentic Frameworks Only)

**This section ONLY applies when working on skill-based agentic projects (ECM operations, stock operations, payment plugins, etc.). Skip this entirely for Java/Go services.**

When the project is a skill-based agentic framework:

**1. Inventory First — Before Creating Anything New**
```
BEFORE writing any code or creating new skills:
1. List all existing skills in skills/ folder
2. Read each skill's purpose and triggers
3. Check runbooks/ for existing procedures
4. Review artifacts/ for existing dashboards
5. Scan queries/ for reusable SQL
```

**2. Extend Before Create**
- Can an existing skill be parameterized to handle the new case?
- Can you add a new trigger to an existing skill?
- Can you compose existing skills rather than build from scratch?

**3. Skill Design Principles**
- **One skill = one operational outcome** (triage, resolve, escalate, report)
- **Skills are declarative** — describe WHAT to do, not HOW (the agent figures out HOW)
- **Skills reference runbooks** — don't embed procedures, link to them
- **Skills use shared queries** — SQL lives in `queries/`, skills reference by name

**4. Output Format — Skills, NOT Scripts**

In skill-based frameworks, the deliverable is a SKILL, not code. Scripts are one-off and create knowledge silos; skills are reusable, discoverable, and composable.

| Request Type | ❌ WRONG Output | ✅ CORRECT Output |
|--------------|-----------------|-------------------|
| "Track SLA breaches" | Python script with pandas | `track-sla-breaches.md` skill file |
| "Generate daily report" | Java scheduled job | `daily-report.md` skill + cron trigger |
| "Fetch stuck orders" | Go CLI tool | `fetch-stuck-orders.md` skill referencing `queries/stuck-orders.sql` |
| "Alert on anomalies" | Custom monitoring code | Skill that uses ThirdEye connector |

```
❌ WRONG: Dumping a Python script
"Here's a Python script to track SLA breaches:"
import pandas as pd
def track_breaches():
    ...

✅ CORRECT: Creating a skill file
"I'll create a new skill at skills/track-sla-breaches.md:"
---
name: track-sla-breaches
triggers:
  - "show SLA breaches"
  - "which orders breached SLA"
connectors:
  - redshift
---
# Track SLA Breaches
## Purpose
Identify orders that have exceeded their SLA thresholds...
```

**Why skills over scripts?**
- Skills are reusable by any team member or agent
- Skills are discoverable (they live in a known location with clear triggers)
- Skills compose with other skills
- Scripts are one-off, require execution context, and create knowledge silos

**5. The Skill Creation Checklist**
Only create a new skill when:
- [ ] No existing skill covers this operational outcome
- [ ] The workflow will be repeated (not a one-off)
- [ ] You've defined clear triggers (when does this skill activate?)
- [ ] You've identified the data sources (which connectors/queries?)
- [ ] You've defined the output format (artifact, report, action?)

```
✅ CORRECT (Skills-First):
"The user wants to track SLA breaches for ECM. Let me check:
 - existing skills: run-ecm.md, triage-ecm.md, resolve-ecm.md
 - run-ecm.md already queries SLA data but doesn't filter breaches
 → I'll extend run-ecm.md with a 'breached_only' parameter"

❌ INCORRECT (Code-First in Agentic Context):
"The user wants to track SLA breaches. Let me create a new BreachTracker.java service..."
```

**6. The SAGE Principles + Lazy Context**

For skill design guidance, we have a layered documentation structure:

| Document | When to Load | Content |
|----------|--------------|---------|
| **[AGENTIC_SKILLS.md](./AGENTIC_SKILLS.md)** | Starting skill work | Quick reference: SAGE, structure, checklists |
| **[SKILL_DESIGN_PRINCIPLES.md](./SKILL_DESIGN_PRINCIPLES.md)** | Deep design work | Full principles: DDD→Skill, SOLID→SKILL, Lazy Loading |

**SAGE Principles Summary:**

| Principle | Meaning |
|-----------|---------|
| **S — Scoped** | One skill, one domain. Clear triggers and boundaries. |
| **A — Adaptive** | Explain WHY, not just rules. Let the LLM reason. |
| **G — Gradual** | Progressive disclosure + lazy loading. <500 lines. |
| **E — Evaluated** | Test with real prompts. Prune what doesn't improve output. |

**Lazy Context Principles (LC1-LC10):** Load references only when branch requires. Defer tool calls until output needed. Cache established facts. Delegate to skills at delegation point, not upfront.

**When to read the deep dive (SKILL_DESIGN_PRINCIPLES.md):**
- Translating DDD/SOLID patterns to skill design
- Understanding the three cognitive layers (WHAT/WHY/HOW)
- Applying all 10 lazy loading patterns
- Using the enhanced SKILL.md template
- Debugging anti-patterns

### Phase 0: The Sharpening Question

Before anything else, ask the question that reframes the problem:
- "Do we actually need to build this, or can we prove the concept with existing tools?"
- "What's the minimal version that would change someone's mind?"
- "If I had to demo this in 2 hours, what would I show?"

### Phase 1: Problem Framing (Before ANY Code)

```
INPUT:  Vague idea / requirement / feature request
OUTPUT: One-page problem statement with constraints
```

- Write a **Problem Statement** in plain English (3-5 sentences max)
- List **Success Criteria** — how do we know this is done?
- Identify **Boundaries** — what is explicitly out of scope?
- Define **Users/Actors** — who interacts with this and how?

### Phase 2: The Primitive (Build to Prove)

Don't architecture-spike. Build the smallest working thing with real data:
- Use whatever tools are fastest (Claude + spreadsheet, a single script, a React artifact)
- Connect to real data sources (Redshift, Pinot, production APIs) — never mock data
- Demo it to a stakeholder within hours, not days
- The primitive's job is to **prove value**, not to be production code

### Phase 3: VibeKanban Task Decomposition

Once the primitive proves value, decompose the real build into **vertical slices**, not horizontal layers.

```
❌ BAD:  "Build the database layer" → "Build the API layer" → "Build the UI"
✅ GOOD: "User can see SLA-breached orders" → "User can drill into one order" → "User can resolve and track"
```

**Task sizing rules:**
- Every task should be completable in **< 2 hours of focused work**
- Every task should be **independently demoable** — it produces a visible result
- Every task should have a **clear definition of done**

**Kanban columns:**
| Column | Meaning |
|--------|---------|
| 📋 Backlog | Defined but not prioritized |
| 🎯 Up Next | Prioritized, ready to start (has all context needed) |
| 🔨 In Progress | Actively being worked on (WIP limit: 1-2 per person) |
| 🔍 Review | Code written, needs review or testing |
| ✅ Done | Merged, deployed, verified |

**When working with AI agents (Claude, Cursor, Codex):**
- Feed one task at a time. Don't dump the whole backlog.
- Each task prompt should include: the task description, relevant context from this brain file, any specific files/modules affected, and the definition of done.
- After each task, review the output before moving to the next. AI doesn't "remember" quality standards between prompts.

### Phase 4: Skill-ify

Once a workflow is proven and stabilized, package it as a reusable skill:
- SKILL.md with YAML frontmatter → Purpose → Prerequisites → Instructions → Examples → Troubleshooting
- Runbooks for operational procedures
- Slash commands for quick execution
- Artifacts (React dashboards, detail views) for rich output
- Version with semver, assign maintenance ownership

---

## 6. Observability — SRE-Driven, Non-Negotiable

I treat observability as a first-class architectural concern, not an afterthought. Every service ships with observability from day one.

### The Four Golden Signals (Google SRE)

Every service should expose these — they answer "is my system healthy right now?":

| Signal | What to Measure |
|--------|----------------|
| **Latency** | Duration of requests — distinguish successful from failed |
| **Traffic** | Demand on the system — requests/sec, concurrent users |
| **Errors** | Rate of failed requests — explicit (5xx) and implicit (wrong results) |
| **Saturation** | How full the system is — CPU, memory, queue depth, connection pools |

### Metric Hierarchy

Design metrics in layers. Not everything needs the same granularity:

| Level | Scope | Series Count | Example |
|-------|-------|-------------|---------|
| **L0** | Service-level health | < 10 series | `http_requests_total`, `error_rate` |
| **L1** | Feature-level detail | 10–1,000 series | `payment_processing_duration`, `search_results_count` |
| **L2** | Instance/debug-level | 1,000+ series | Per-endpoint, per-customer, per-shard metrics |

### Cardinality Control — Critical

High-cardinality labels are the #1 way to blow up your metrics backend:

- **Sanitize** — Replace user IDs, request IDs, emails with bucketed categories
- **Bucket** — Use histogram buckets for continuous values, not raw numbers
- **Monitor** — Track series count per metric. Alert on unexpected growth.
- **Avoid unbounded values** (URLs, user agents, error messages) as label values — each unique value creates a new time series

### The Zero Mutation Rule for Instrumentation

```
✅ CORRECT: Metrics are read-only observers of behavior
❌ INCORRECT: Metrics that catch exceptions, change control flow, or alter return values
```

Instrumentation is purely additive. It wraps, it observes, it records — it never modifies. If removing all metrics would change your program's behavior, your instrumentation is broken.

### Alerting Philosophy

- **Anomaly-based over static thresholds.** Use percentage-change, mean-variance, and week-over-week baselines — not hardcoded numbers that become stale.
- **Alert on SLOs**, not raw metrics — alert on user-facing impact.
- **5-7 alerts per service is the sweet spot.** Enough coverage without alert fatigue.
- **Escalation matrix** aligned with SLAs: Warning (16h) → Action Required (24h) → Critical (48h) → Overdue (72h)

### Every Service Ships With

- **Structured logging** (JSON, with correlation IDs)
- **Metrics** (latency percentiles p50/p95/p99, error rates, throughput)
- **Distributed tracing** (trace ID propagates across service boundaries)
- **Health checks** (liveness + readiness, distinct)
- **Data freshness monitoring** (alert if pipeline stops flowing — silent failures are the worst)

---

## 7. Language-Specific Patterns

I'm multi-language but consistent in principles. Here's how I apply my standards in each:

### Java

- **Domain-Driven Design** — Rich domain models, not anemic data bags
- **Value objects via Java `record`** — Immutable, equals/hashCode for free
- **Strict layer separation:** domain → application → infrastructure → API
- **Spring Boot** as the framework — but business logic never imports Spring
- **Micrometer** for metrics — with Prometheus registry
- **JUnit 5 + Mockito** for testing — structured, parameterized tests
- **No field injection.** Constructor injection only.

```
✅ CORRECT: Domain entity with behavior
public record Money(BigDecimal amount, Currency currency) {
    public Money add(Money other) { ... }
}

❌ INCORRECT: Anemic DTO pretending to be a domain object
public class Money {
    private BigDecimal amount;  // just getters/setters, no behavior
}
```

### Go

- **Standard project layout:** `cmd/`, `internal/`, `pkg/`
- **Interface Segregation:** "Accept interfaces, return structs" — define interfaces at the consumer, not the provider
- **Explicit error wrapping:** `fmt.Errorf("fetching user %d: %w", id, err)` — always add context
- **Bounded concurrency:** Use semaphores/worker pools, never unbounded goroutine spawning
- **Table-driven tests:** Every test function with multiple cases uses a test table
- **client_golang** for Prometheus metrics

```
✅ CORRECT: Interface defined by consumer
type UserFetcher interface {
    FetchUser(ctx context.Context, id string) (*User, error)
}

❌ INCORRECT: Giant interface defined by provider
type UserService interface {
    FetchUser(...)
    CreateUser(...)
    DeleteUser(...)
    UpdateUser(...)
    ListUsers(...)
}
```

### General Multi-Language Rule

Same SOLID principles, same SRE patterns, same testing rigor — regardless of language. If you're writing Go, don't suddenly forget about dependency inversion because "Go is simple." If you're writing Java, don't over-engineer because "Java needs patterns."

---

## 8. Designing for Scale — My Playbook

I don't prematurely optimize, but I design with scale *awareness*. Here's how:

### Scalability First Principles

1. **Statelessness** — Services should be stateless wherever possible. State belongs in purpose-built stores (databases, caches, queues).
2. **Horizontal scaling** — Design so adding more instances solves throughput problems. Avoid architectures that require bigger machines.
3. **Async by default** — If a user doesn't need to wait for it, don't make them. Use queues, events, background jobs.
4. **Cache aggressively, invalidate carefully** — Cache at every layer (CDN, application, database). But have a clear invalidation strategy.
5. **Partition and isolate** — Failures in one part of the system should not cascade. Use bulkheads, circuit breakers, timeouts.
6. **Graceful degradation** — When a dependency fails, degrade functionality rather than crash entirely. Serve stale data, disable non-critical features, queue for retry.

### Data Architecture Principles

- **Choose the right database for the access pattern**, not the one you're most familiar with.
- **Schema design follows query patterns**, not entity relationships.
- **Event sourcing where auditability matters** — store facts (events), derive state.
- **CQRS when read and write patterns diverge significantly.**
- **Pinot for real-time analytics, Redshift for historical analysis.** Each has its place — don't force one to be the other.

### API Design

- **REST for CRUD-heavy public APIs** — it's boring and that's good.
- **GraphQL when clients have diverse data needs** — not as a default.
- **gRPC for internal service-to-service** — when performance matters.
- **Always version your APIs.** Always.
- **Pagination from day one.** No endpoint returns unbounded lists.

---

## 9. Code Quality Standards

### Naming

- **Functions:** verb + noun → `calculateTotalPrice()`, `fetchUserProfile()`
- **Booleans:** `is/has/should/can` prefix → `isActive`, `hasPermission`
- **Constants:** SCREAMING_SNAKE → `MAX_RETRY_COUNT`
- **No abbreviations** unless universally understood (`id`, `url`, `api` are fine; `usr`, `mgr`, `btn` are not)

### Functions

- **Max 20 lines.** If longer, it's doing too much.
- **Max 3 parameters.** If more, use an options/config object.
- **Single return type.** Don't return `string | null | undefined | Error`. Pick a pattern (Result type, throw, etc.) and be consistent.
- **Pure functions wherever possible.** Same input → same output, no side effects.

### Error Handling

- **Don't swallow errors silently** — every catch block should log, rethrow, or handle meaningfully. Silent failures are the hardest bugs to diagnose.
- **Use typed errors / error codes.** Not just string messages.
- **Distinguish between recoverable and fatal errors.** Handle the former, crash on the latter.
- **User-facing errors ≠ developer errors.** Users get friendly messages. Logs get stack traces.
- **In Go:** Always wrap errors with context using `%w`. Never return bare `err`.
- **In Java:** Use domain-specific exceptions that extend a common base. Never catch `Exception` generically.

### Testing Strategy

```
         /  E2E Tests  \        ← Few: critical user journeys only
        / Integration    \      ← Some: API contracts, DB queries
       /   Unit Tests     \     ← Many: business logic, pure functions
      ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
```

- **Test behavior, not implementation.** Tests should survive refactors.
- **Arrange-Act-Assert** structure for every test. No exceptions.
- **One assertion per test** (conceptual, not literal).
- **Test the boundaries:** empty inputs, max values, concurrent access, error paths.
- **No mocking what you don't own.** Wrap third-party code, mock the wrapper.
- **Table-driven tests in Go** — every function with multiple cases uses a test table.
- **DRY test utilities** — shared helpers, fixtures, builders. Don't copy-paste test setup across files.
- **Mandatory tests for every code change.** No PR merges without corresponding test coverage.

---

## 10. Documentation Style

I follow the **Minto Pyramid Principle**: main point first, supporting details follow. Never bury the lede.

### How I Structure Documentation

- **Show, don't tell.** Examples before explanation. Code before prose.
- **✅ CORRECT / ❌ INCORRECT patterns.** Every standard includes explicit good and bad code comparisons. AI agents should follow this style when explaining trade-offs.
- **SKILL.md structure** for reusable knowledge: YAML frontmatter → Purpose → Prerequisites → Automatic Codebase Analysis → Step-by-step Instructions → Examples → Troubleshooting
- **Versioned skills** — Semver for reusable docs/skills, deprecation procedures, maintenance ownership

### For AI Agents Writing Docs

When I ask you to document something, follow this format:
1. One-sentence summary of what this is
2. Why it exists (the problem it solves)
3. How to use it (with code examples)
4. Edge cases and gotchas
5. Never lead with history or background — lead with the answer

---

## 11. My Decision-Making Heuristics

When I (or an AI agent working for me) face an ambiguous decision:

### Technology Choices
- **Choose boring technology.** Every new technology has a limited innovation token budget. Spend them only where they create real competitive advantage.
- **Optimize for replaceability.** The best architectural choice is one that's easy to swap out later.
- **Prefer ecosystems over tools.** A mediocre tool with a great ecosystem beats a great tool with no ecosystem.
- **Reference industry standards** — Google SRE, Netflix resilience patterns, Kubernetes conventions. Don't reinvent what's battle-tested.
- **Challenge the default.** Before using the "obvious" tool, ask: is there something better-suited? (This is how I ended up using Pinot+ThirdEye instead of custom AI monitoring.)

### Architecture Choices
- **Monolith first.** Microservices are an optimization for organizational scaling, not technical scaling. Start with a well-structured monolith.
- **Extract when it hurts.** Don't preemptively split services. Split when deployment coupling, scaling needs, or team boundaries demand it.
- **Event-driven ≠ event sourced.** You can publish events without storing them as the source of truth. Start with events for communication, add sourcing when you need it.
- **Polyglot is fine with boundaries.** Java for domain logic, Go for infra tooling — but draw clear lines, document them, and enforce them.

### When Stuck
1. **Ask the reframing question.** Often the problem is the problem statement, not the solution.
2. **Timebox it.** If you can't decide in 15 minutes, the decision probably doesn't matter as much as you think. Pick one and move on.
3. **Reversibility test.** Is this a one-way door or a two-way door? Two-way doors (most decisions) deserve less deliberation.
4. **Build the primitive.** Don't debate. Build a throwaway prototype in 1-2 hours. The learning is worth more than the code.
5. **Write the README first.** If you can't explain the approach in plain English, you don't understand it well enough to build it.

---

## 12. Communication With AI Agents

### What I Expect From You (Claude / Cursor / Codex)

- **Detect project type first.** Determine if this is a skill-based agentic framework or a traditional service. Look for `plugin.yaml`, `skills/` folders, `.md` skill files. This determines your approach.
- **Skills-first for agentic projects.** If it's a skill-based framework, inventory existing skills before creating anything. Extend before create.
- **Code-first for services.** If it's a traditional Java/Go service, apply standard engineering patterns (SOLID, DDD, Clean Architecture).
- **Analyze the codebase first.** Infer architecture, patterns, and conventions from the code rather than asking.
- **Think before you code.** Give a 3-5 sentence plan before implementation. If the task is trivial, skip this.
- **One thing at a time.** Atomic commits, atomic changes.
- **Match existing patterns.** If there's a pattern for error handling, routing, state management, metrics — follow it. Don't introduce a new pattern when one exists.
- **Flag design smells.** If you see a design issue, tech debt, or potential bug — flag it. Don't silently work around it.
- **No placeholder code.** Either implement it or state what's missing and why.
- **Comments explain WHY only.** Code should be self-documenting.
- **Test every change.** Add appropriate tests for features, regression tests for bug fixes.
- **Observe the Zero Mutation Rule.** Instrumentation, logging, and metrics are purely additive — they should not change behavior because downstream systems depend on specific exception types and return values.
- **Use real data in demos.** Connect to actual data sources unless I explicitly ask for mocks.
- **Bias toward shipping.** Build the minimal version, show me, iterate.
- **Keep solutions minimal.** Claude 4.x tends to overengineer — don't add features, abstractions, or flexibility beyond what was requested.
- **If data is insufficient, say so.** State uncertainty explicitly rather than speculating. "I don't know" is a valid answer.

### How to Prompt Me (When I'm Prompting You)

I'll typically give you context in this format:

```
CONTEXT: [what we're building and current state]
TASK: [specific thing to do]
CONSTRAINTS: [any limitations or requirements]
DONE WHEN: [definition of done]
```

If I don't provide all four, ask for what's missing before proceeding.

### Claude 4.x Specific Guidance

Claude Opus 4.6 and Sonnet 4.5 behave differently from earlier models:

- **More responsive to instructions.** Previous models needed aggressive prompting ("You MUST...", "CRITICAL:..."). Claude 4.x follows normal instructions well — over-prompting causes overtriggering and unnecessary work.
- **Tends to overengineer.** Claude 4.x may create extra files, add unnecessary abstractions, or build in flexibility that wasn't requested. Keep solutions focused on exactly what was asked.
- **Does more upfront exploration.** Claude 4.6 explores extensively before acting. For simple tasks, choose an approach and commit to it rather than exploring every option.
- **Spawns subagents aggressively.** Use direct tool calls for file reads and searches. Use subagents only for independent parallel workstreams.
- **Adaptive thinking.** Claude 4.6 decides when to reason deeply. For judgment-heavy work (architecture, design), let it think. For mechanical work (boilerplate, config), keep instructions specific so it executes without over-reasoning.
- **Context window awareness.** Claude 4.6 tracks its remaining context budget. Context will be automatically compacted when approaching limits — don't stop tasks early.

### Anti-Hallucination Protocol

Investigate before answering. If a specific file is referenced, read it before responding. If data is insufficient, say so rather than speculating. Use tool calls to verify rather than relying on memory from earlier in the conversation. This is especially important as context gets compacted — re-verify facts from files rather than trusting earlier context.

---

## 13. Project Structure Preferences

### General (Language Agnostic)

```
project-root/
├── CORTEX.md                   ← This file
├── README.md                   ← What this is, how to run it
├── docs/                       ← ADRs, design docs, diagrams
│   └── adr/
│       └── 001-database-choice.md
├── src/                        ← Source code
│   ├── core/                   ← Business logic (no framework deps)
│   │   ├── entities/
│   │   ├── use-cases/
│   │   └── interfaces/         ← Ports (abstractions)
│   ├── infra/                  ← Adapters (implementations)
│   │   ├── database/
│   │   ├── http/
│   │   └── external-services/
│   └── config/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── scripts/                    ← Build, deploy, utility scripts
```

### Go Projects
```
project-root/
├── cmd/                        ← Entry points
│   └── server/main.go
├── internal/                   ← Private application code
│   ├── domain/
│   ├── service/
│   └── repository/
├── pkg/                        ← Public reusable packages
└── tests/
```

### Java Projects (Spring Boot + DDD)
```
project-root/
├── src/main/java/com/company/service/
│   ├── domain/                 ← Entities, value objects (records), domain services
│   ├── application/            ← Use cases, orchestration
│   ├── infrastructure/         ← DB repos, HTTP clients, external integrations
│   └── api/                    ← Controllers, DTOs, request/response mapping
└── src/test/java/              ← Mirrors main structure
```

### Knowledge-Work Plugin Structure
```
plugins/[domain]-operations/
├── plugin.yaml                 ← Manifest: role, connectors, capabilities
├── skills/
│   ├── run-[domain].md         ← Main skill with triggers + output format
│   ├── triage-[entity].md
│   └── resolve-[entity].md
├── artifacts/                  ← React dashboards, detail views
│   ├── dashboard.jsx
│   └── detail-card.jsx
├── runbooks/                   ← RB-PAY-001, RB-OB-001, etc.
├── queries/                    ← SQL queries (Redshift, Pinot)
├── connectors/                 ← MCP server configs
│   ├── redshift.yaml
│   └── google-sheets.yaml
├── terminology.yaml            ← Domain-specific terms
└── escalation-matrix.yaml      ← SLA thresholds and escalation rules
```

### Key Rules
- **`core/` / `domain/` has ZERO external dependencies.** It depends only on language primitives and its own interfaces.
- **`infra/` / `infrastructure/` implements interfaces defined in `core/`.** All adapters live here.
- **Tests mirror the source structure.** Always.

---

## 14. Tech Stack Preferences

When choosing tools, prefer these unless there's a compelling reason to deviate:

| Area | Preferred Choices |
|------|------------------|
| **Metrics** | Prometheus, Micrometer (Java), client_golang (Go), Datadog |
| **Real-time Analytics** | Apache Pinot, StarTree ThirdEye |
| **Historical Analytics** | Redshift |
| **Java** | Spring Boot, JUnit 5, Mockito, Java records for value objects |
| **Go** | Standard library first, testify, gomock, idiomatic patterns |
| **Infra** | Kubernetes, Docker, Terraform, ECS |
| **Data** | PostgreSQL (default), Redshift (analytics), Redis (caching), Pinot (real-time) |
| **Monitoring** | VictoriaMetrics / Prometheus, Grafana, ThirdEye for anomaly detection |
| **CI/CD** | GitHub Actions, conventional commits, semantic versioning |
| **Docs** | Markdown, ADRs, SKILL.md format for reusable knowledge |
| **AI Operations** | Claude (skills, plugins), MCP servers, Google Sheets for lightweight tracking |

---

## 15. Driving Engineering Culture

I don't just write code — I shape how teams think about engineering.

### Senior Forum Discussions
When opening technical discussions (polyglot tooling, architecture decisions, process changes), I use structured RFCs:
- **State the problem** — what's happening, what's the risk
- **Propose options** — not just one answer, but bounded choices
- **Ask sharpening questions** — force the group to take a stance
- **Goal is alignment, not unanimity** — "not necessarily to unify everything, but to take a deliberate stance"

### Knowledge Sharing
- Package proven solutions as skills/plugins that eliminate knowledge silos
- Build interactive dashboards and artifacts that team members can self-serve
- Reference industry standards (Google SRE, Netflix, Kubernetes) to anchor decisions in proven patterns, not opinions
- ✅ CORRECT / ❌ INCORRECT comparisons teach faster than paragraphs of explanation

---

## 16. Git & Version Control Standards

- **Conventional Commits:** `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- **Small, atomic commits.** Each commit does ONE thing and the codebase compiles + passes tests.
- **Branch naming:** `feat/short-description`, `fix/issue-number-description`, `refactor/what-changed`
- **No force pushes to main/develop** — force pushing rewrites shared history and can lose other developers' work.
- **PRs should be reviewable in < 15 minutes.** If they're bigger, break them up.
- **Semver for reusable skills/libraries.** With deprecation procedures and maintenance ownership.

---

## 17. Security Defaults

- **Secrets belong in environment variables or a secrets manager** — committing secrets to git creates a permanent security exposure that's difficult to remediate.
- **Validate inputs at the boundary** — user input and external APIs are untrusted. Internal code and framework guarantees can be trusted.
- **Principle of Least Privilege** — every service, user, and API key gets the minimum permissions needed.
- **Dependencies are attack surface** — audit them, pin versions, keep them updated.
- **Authentication and authorization are separate concerns** — don't conflate identity verification with permission checking.

---

## 18. References & Influences

These books, talks, and practices shaped my thinking:

| Resource | Key Takeaway |
|----------|-------------|
| *Clean Architecture* — Robert C. Martin | Dependency rule, boundaries, use cases |
| *Refactoring* — Martin Fowler | Design emerges from small improvements |
| *A Philosophy of Software Design* — John Ousterhout | Deep vs shallow modules, complexity management |
| *Designing Data-Intensive Applications* — Martin Kleppmann | Distributed systems, data modeling, trade-offs |
| *Simple Made Easy* — Rich Hickey (talk) | Simple ≠ Easy, complecting is the root of complexity |
| *The Pragmatic Programmer* — Hunt & Thomas | Tracer bullets, DRY, orthogonality |
| *Domain-Driven Design* — Eric Evans | Ubiquitous language, bounded contexts |
| *Release It!* — Michael Nygard | Stability patterns, circuit breakers, bulkheads |
| *Google SRE Book* | Four golden signals, error budgets, SLOs |
| *The Minto Pyramid Principle* — Barbara Minto | Main point first, structured communication |
| *High Output Management* — Andy Grove | Leverage, operational excellence, decision-making |
| *Complete Guide to Building Skills for Claude* — Anthropic (2026) | Official skill standards, description-driven discovery, multi-model testing |

### X / Twitter Accounts Worth Following for This Practice

These people have publicly shared excellent approaches to AI-agent rules files, cursor configs, and engineering operating system documents:

| Handle | Why Follow |
|--------|-----------|
| **@mckaywrigley** | Pioneer of `.cursorrules` and structured AI prompting for codebases. Shares his actual rule files. |
| **@IndyDevDan** | Curated repo of cursor rules, detailed AI-first development workflows. His "Cursor Rules" collection is the canonical reference. |
| **@alexalbert__** | Anthropic's prompt engineer. Patterns for Claude system prompts and project instructions directly applicable to brain files. |
| **@swyx** (Shawn Wang) | "AI-enhanced developer workflows" — coined much of the thinking around context management for AI agents. |
| **@kreaboris** | Detailed Cursor workflows, custom rules files. Strong focus on structuring context for agents. |
| **@NickADobos** | Shares `.cursorrules` examples and AI pair programming techniques. Good for practical, real-world patterns. |
| **@ReuvenCohen** | Writes about "system prompts as engineering culture" — treating your AI config as a team document. |
| **cursor.directory** | Community-contributed `.cursorrules` collection — study the top-rated ones and steal patterns. |
| **@kaboroevich** | Shares thoughtful threads on how to structure engineering knowledge for AI consumption. |
| **@bentoml** | Open-source team sharing how they use AI agents with structured rules for code review and generation. |

### Key Threads & Resources on X

Search for these on X for high-signal content:
- `"cursorrules" best practices` — community sharing their actual config files
- `"claude.md" project instructions` — how people structure Claude Code projects
- `"AI coding rules" SOLID` — intersection of engineering principles with AI agent configuration
- `"vibe coding" rules file` — the VibeKanban community sharing structured approaches
- `"system prompt" engineering culture` — treating AI config as documentation of team standards
- `"engineering brain" document` — people sharing their living engineering operating systems (this exact concept)

### Best Practices From the Community (Distilled)

1. **Be opinionated, not generic.** The more specific your rules file, the less you re-prompt. "Use Java records for value objects" beats "use immutable types."
2. **Include ✅/❌ examples.** AI agents respond dramatically better to concrete good/bad comparisons than abstract principles.
3. **Separate the "what" from the "how."** Your brain file (what/why) is stable. Your `.cursorrules` (how/format) adapts per project.
4. **Version and changelog your brain file.** Treat it like code. It evolves with every project retro.
5. **Layer your config:** Brain file (global) → Project-specific rules → Task-specific context. Don't cram everything into one file.
6. **Include your domain vocabulary.** AI agents hallucinate less when they know your terminology (ECM, TTD, TTM, acquirer, stuck_reason).
7. **Define the rhythm.** Your iterative cycle (question → primitive → demo → sharpen → scale) is a workflow AI agents can follow if you state it explicitly.
8. **Operational guardrails are just as important as coding standards.** "Don't rewrite my SQL" and "say I don't know" are rules most people forget to include.

---

## 19. Living Document Protocol

This file evolves. Here's how:

- **Review monthly.** Does this still reflect how I actually work?
- **Update after every project retro.** What did I learn? What principle was validated or invalidated?
- **Version it.** Keep a changelog at the bottom for major updates.
- **AI agents don't modify this file** unless I explicitly ask them to suggest updates.

### Changelog

| Date | Change |
|------|--------|
| 2025-02-11 | v1: Initial version created |
| 2025-02-11 | v2: Integrated traits from ai-velocity codebase — Observability, Operational Guardrails, Language-Specific Patterns, Zero Logic Mutation, Metric Hierarchy |
| 2025-02-11 | v3: Deep personalization from conversation history — added Section 1 (Engineering Identity), Iterative Development Rhythm, The Primitive phase, Sharpening Questions, Domain Context, Knowledge-Work Plugin structure, Alerting Philosophy, Culture/RFC patterns, X community references, best practices from .cursorrules community |
| 2025-02-14 | v4: Added Skills-First Mindset — project type detection (Phase -1), skills-first approach for agentic frameworks, explicit guidance that output should be skills NOT Python/Java/Go scripts. Only applies to skill-based projects, not traditional services. |
| 2025-02-14 | v5: Added SAGE principles summary + created AGENTIC_SKILLS.md (quick reference) and SKILL_DESIGN_PRINCIPLES.md (deep dive with DDD→Skill, SOLID→SKILL, Lazy Loading LC1-LC10, enhanced template). Three-tier progressive disclosure. |
| 2026-02-14 | v6: Aligned with Anthropic's official "Complete Guide to Building Skills for Claude" (2026). Added: official YAML frontmatter spec, gerund naming conventions, description writing guidance (critical for discovery), multi-model testing (Haiku/Sonnet/Opus), feedback loops, context window awareness. Updated all skill files to match official standards. |
| 2026-02-16 | v7: Audit against Anthropic's Claude 4.x best practices and extended thinking research. Replaced aggressive language (CRITICAL/MUST/NEVER-without-WHY) with reasoned constraints. Added Claude 4.x behavior awareness (overtriggering, overengineering, adaptive thinking). Added anti-hallucination protocol. Extracted domain context to DOMAIN_CONTEXT.md. Added context rot prevention to AGENTIC_SKILLS.md. Restructured CLAUDE_GLOBAL.md with XML tags for system prompt optimization. |

---

> *"Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away."* — Antoine de Saint-Exupéry
