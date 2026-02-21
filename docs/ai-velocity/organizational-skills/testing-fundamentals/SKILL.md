---
name: testing-fundamentals
description: "Base layer for testing across all languages. Defines test pyramid, mocking strategy, when to use unit vs integration vs E2E, and how to avoid over-mocking. Use when designing tests, deciding what to mock, writing E2E tests, or when the user asks about testing strategy, mocking, or test types."
---

# Testing Fundamentals

Language-agnostic testing principles. Apply before language-specific skills (e.g., java-testing, golang-unit-testing). Covers test pyramid, mocking strategy, and E2E approaches.

## When to Apply

- Designing test strategy for a feature or service
- Deciding what to mock vs stub vs use real
- Writing or reviewing E2E tests
- When user asks: "unit vs integration vs E2E?", "what should I mock?", "how to test this?"

---

## 1. Test Pyramid

### Layers

| Layer | Count | Speed | Scope | Purpose |
|-------|-------|-------|-------|---------|
| **Unit** | Many | Fast | Single unit, mocked deps | Logic, business rules |
| **Integration** | Fewer | Slower | Real DB, APIs, services | Wiring, contracts |
| **E2E** | Minimal | Slowest | Full system | Critical paths, user journeys |

### When to Use Each

- **Unit**: Fast feedback; test business logic in isolation. Mock external dependencies.
- **Integration**: Verify components work together. Use real DB, mock external APIs when needed.
- **E2E**: Few, critical paths only. Expensive; use for signup, checkout, core flows.

### Inverted Pyramid

- Avoid: Many E2E tests, few unit tests. Leads to slow, flaky test suites.
- Prefer: Broad unit coverage, targeted integration, minimal E2E.

---

## 2. Mocking Strategy

### What to Mock

- **External services**: HTTP clients, third-party APIs, message queues
- **Infrastructure**: Databases (in unit tests), file system, network
- **Unpredictable**: Time, random, UUID generation
- **Expensive**: Slow or paid external calls

### What Not to Mock

- **Domain logic**: Test pure logic with real code
- **Value objects**: Use real instances
- **Simple collaborators**: Use real when fast and deterministic
- **What you own**: Prefer real in unit scope when cheap

### Mock vs Stub vs Spy

| Type | Use |
|------|-----|
| **Mock** | Verify interactions (e.g., "was this called with X?") |
| **Stub** | Return fixed response; no interaction verification |
| **Spy** | Wrap real object; verify calls while delegating behavior |

### Over-Mocking

- **Don't** mock everything. Tests become brittle and couple to implementation.
- **Do** mock at boundaries. Use real for in-memory logic.
- **Test behavior**, not implementation. If you mock internals, you're testing the mock.

---

## 3. E2E Testing

### When to Add E2E

- Critical user flows (signup, login, purchase)
- Cross-service flows (API → DB → external call)
- Contract verification between services

### E2E Patterns

| Pattern | Use |
|---------|-----|
| **API E2E** | Full HTTP request → response; real or mocked downstream |
| **Contract tests** | Verify provider/consumer contract without full integration |
| **Smoke tests** | Minimal checks that system is up and core paths work |

### E2E Trade-offs

- **Slower**: Run less frequently; in CI, maybe on main only
- **Flakier**: Network, timing, shared state can cause failures
- **Maintenance**: Keep E2E minimal; prefer unit and integration for coverage

### Design for E2E

- **Idempotent**: E2E should not depend on prior test state
- **Isolated**: Each test cleans up; no shared mutable state
- **Deterministic**: Avoid sleeps; use polling or event-driven waits

---

## 4. Test Design

### AAA Pattern

- **Arrange**: Set up data, mocks, preconditions
- **Act**: Execute the behavior under test
- **Assert**: Verify outcome

### What to Test

- **Happy path**: Core behavior works
- **Edge cases**: Null, empty, boundaries, invalid input
- **Error paths**: Exceptions, failures, timeouts

### Avoid Flakiness

- No randomness unless explicitly tested
- No time-dependent logic without injection
- No shared mutable state between tests
- No external dependencies in unit tests

---

## Quick Reference

| Question | Answer |
|----------|--------|
| Unit or integration? | Unit for logic; integration for wiring |
| Mock this? | Mock at boundaries; use real for in-memory logic |
| Add E2E? | Only for critical paths; keep minimal |
| Test flaky? | Remove randomness, shared state, external deps |

---

## Additional Resources

- For Java: See [java-testing](../../java-testing/)
- For Go: See [golang-unit-testing](../../golang-unit-testing/)
