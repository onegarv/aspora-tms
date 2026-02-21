You are developing a Java application with rich domain models and clean architecture. Follow these phases IN ORDER.
Do not skip phases. Each phase has a GATE — complete it before proceeding.

## Pre-Flight: Decision Context

Before starting, check for these files in the current project root (read them if they exist, skip if not):
1. `DECISIONS.md` — prior decisions for this project. Do NOT re-decide what's already settled.
2. `GUARDRAILS.md` — anti-patterns from past incidents. These are hard constraints.

Also check: $HOME/code/aspora/ai-velocity/java-application-development/ for DECISIONS.md and GUARDRAILS.md

## Phase 0: Analyze Existing Codebase

Before writing any code:
- Read existing package structure — identify domain/, application/, infrastructure/, api/ layers
- Read existing domain entities, value objects, and aggregates
- Identify existing patterns: How are errors handled? How is DI done? What naming conventions?
- Check for existing base classes, shared utilities, or domain primitives

GATE: Describe the existing architecture patterns you found. If this is a new project, describe the package structure you will create.

## Phase 1: Domain Model First

Design the domain model BEFORE any framework code:
- [ ] Entities: Objects with identity and lifecycle (have an ID, are mutable)
- [ ] Value Objects: Immutable objects defined by their attributes (use Java `record`)
- [ ] Aggregates: Consistency boundaries — one aggregate per transaction
- [ ] Domain Services: Business logic that doesn't belong to a single entity

Rules:
- Domain layer has ZERO framework dependencies (no Spring imports)
- Rich domain models with behavior — NOT anemic data bags with only getters/setters
- Value objects are always Java `record` types
- Constructor injection only — no field injection

Read the skill reference for domain modeling patterns:
$HOME/code/aspora/ai-velocity/java-application-development/SKILL.md

GATE: Present your domain model design (entities, value objects, aggregates). Confirm with user.

## Phase 2: Layer Architecture

Implement following Clean Architecture — dependencies point INWARD:

```
api/ (Controllers, DTOs, request/response mapping)
  ↓ depends on
application/ (Use cases, orchestration, transaction boundaries)
  ↓ depends on
domain/ (Entities, value objects, domain services, repository interfaces)
  ↑ NO outward dependencies
infrastructure/ (DB repos, HTTP clients, external integrations)
  ↑ implements domain interfaces
```

- [ ] `domain/` — entities, value objects (records), domain services, repository INTERFACES (ports)
- [ ] `application/` — use case classes, one public method per use case
- [ ] `infrastructure/` — repository IMPLEMENTATIONS, external service adapters
- [ ] `api/` — controllers, DTOs, mappers (DTOs ≠ domain objects)

## Phase 3: Implement

For each component:
- Functions: < 20 lines, < 3 parameters, pure when possible
- Error handling: Domain-specific exceptions extending a common base, never catch generic `Exception`
- Validation: In the domain layer (constructor validation for value objects, business rules in entities)
- DI: Constructor injection, program to interfaces

## Phase 4: Verification

| Principle | Check | Status |
|-----------|-------|--------|
| SRP | Each class does ONE thing? | [ ] |
| OCP | Behavior extended via interfaces, not modification? | [ ] |
| LSP | Subtypes substitutable for base types? | [ ] |
| ISP | No client depends on methods it doesn't use? | [ ] |
| DIP | High-level modules depend on abstractions? | [ ] |
| Clean Arch | Domain layer has zero Spring/framework imports? | [ ] |
| Rich Model | Entities have behavior, not just getters/setters? | [ ] |
| Value Objects | Immutable, using Java `record`? | [ ] |
| No Field Injection | All DI via constructor? | [ ] |

## Phase 5: Self-Check

1. "Does the domain layer import any framework classes?" → Must be NO
2. "Are value objects implemented as Java records?" → Must be YES
3. "Can I test domain logic without starting Spring?" → Must be YES
4. "Is every class under 200 lines, every method under 20 lines?" → Should be YES
5. "Do DTOs and domain objects have separate types?" → Must be YES
