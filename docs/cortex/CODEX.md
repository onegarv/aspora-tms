
# CODEX.md

# OpenAI Codex Agent Instructions

## Primary Directive
Read `CORTEX.md` at the project root before any work.

## Core Philosophy: Velocity = Speed × Direction
Prove by building, not debating. Primitive → Demo → Sharpen → Scale.

## Critical Rules
- **Analyze the codebase first.** Infer architecture and patterns — don't ask me to describe what you can read.
- **Use real data.** Connect to actual sources, never fabricate mocks unless asked.
- Clean Architecture (core/domain/ has zero external deps)
- SOLID across all languages, DRY (rule of three), YAGNI
- Functions: < 20 lines, < 3 params, pure when possible
- Mandatory tests for every change. Arrange-Act-Assert. Table-driven in Go.
- Conventional commits, atomic changes
- Configuration-driven behavior — YAML/config over hardcoded logic

## Observability
- Four Golden Signals on every service
- Zero Logic Mutation: instrumentation is purely additive, never changes behavior
- Anomaly-based alerting over static thresholds
- Cardinality control on all metric labels

## Language-Specific
- **Java:** DDD, `record` value objects, constructor injection, Micrometer, JUnit 5 + Mockito
- **Go:** `cmd/internal/pkg`, accept interfaces return structs, `%w` error wrapping, table-driven tests

## Operational Guardrails
- No hallucination, no assumptions when data is missing
- Idempotency — don't rewrite my queries/commands unless asked
- Explicit "No" for unsafe actions
- Scope honesty — admit limitations

## Workflow
1. Read existing code patterns before writing new code
2. Plan before implementing (brief summary for non-trivial tasks)
3. Implement one thing at a time
4. Add tests alongside implementation
5. Flag design issues — don't silently work around them

## Task Format
```
CONTEXT: ...
TASK: ...
CONSTRAINTS: ...
DONE WHEN: ...
```
Ask for missing pieces before starting.

## Domain Context
Fintech / cross-border payments. ECM operations, SLA tracking, acquirer ops (CHECKOUT, TRUELAYER, LEANTECH). Redshift, Pinot, ThirdEye, Kubernetes, ECS.
