# Decisions

## DEC-001: Four Golden Signals as coverage gate (2026-02-15)
**Chose:** Mandatory Latency/Traffic/Errors/Saturation verification table before PR completion
**Over:** Trusting agents to infer signal coverage from SKILL.md prose
**Why:** RCA PROM-2026-001 — 18 PRs shipped with only Latency (and broken)
**Constraint:** NEVER mark instrumentation complete without filling the Four Golden Signals table

## DEC-002: MetricsAopConfig is Phase 2, not optional (2026-02-15)
**Chose:** MetricsAopConfig.java listed as non-negotiable required file in Phase 2
**Over:** Mentioning it deep in SKILL.md implementation section
**Why:** Without TimedAspect bean, ALL @Timed annotations are silently ignored — zero metrics
**Constraint:** NEVER add @Timed annotations without verifying MetricsAopConfig.java exists

## DEC-003: Tier system for metric placement (2025-02-11)
**Chose:** Tier 1 (@Timed) → Tier 2 (@MeteredOperation) → Tier 3 (inline) hierarchy
**Over:** All-inline manual metrics recording
**Why:** @Timed covers 80% of cases with zero boilerplate; inline only for gauges/queue depth
**Constraint:** NEVER use inline metrics when @Timed or @MeteredOperation would suffice

## DEC-004: Phased execution protocol for commands (2026-02-16)
**Chose:** Commands as sequenced phase protocols with gates
**Over:** Commands as pointers to SKILL.md ("read the 4,454-line file")
**Why:** Agents skim large docs, miss critical sections — phases force sequential completion
**Constraint:** NEVER load full SKILL.md upfront; load sections lazily per phase

## DEC-005: Do NOT instrument auto-provided metrics (2026-02-15)
**Chose:** Explicitly list Spring Boot auto-metrics in Phase 1 analysis
**Over:** Letting agents discover overlap after instrumentation
**Why:** RCA showed agents adding @Timed to controllers already covered by http.server.requests
**Constraint:** NEVER add @Timed to methods covered by http.server.requests, spring.data.*, or hikaricp.*

## DEC-006: Zero Mutation Rule is non-negotiable (2025-02-11)
**Chose:** Separate GUARDRAILS.md enforcing purely additive instrumentation
**Over:** Inline warnings in SKILL.md
**Why:** Agents changed exception types, added validation, modified signatures when instrumenting
**Constraint:** NEVER change behavior, control flow, exception types, or method signatures when adding metrics
