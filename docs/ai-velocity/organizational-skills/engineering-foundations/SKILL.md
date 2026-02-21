---
name: engineering-foundations
description: "Base layer for engineering teams. Defines coding standards, architectural principles, testing, edge cases, scalability, security, operational readiness, code review, technical debt, and documentation. Use when designing systems, implementing features, reviewing code, or when the user asks for architectural guidance, best practices, or how to build robust solutions."
---

# Engineering Foundations

Base layer of organizational engineering standards. Apply before task-specific skills (e.g., golang-coding-standards, java-testing). Claude Code should reference this when implementing, designing, or reviewing code.

## When to Apply

- Designing new systems, services, or features
- Implementing any functionality
- Writing or reviewing tests
- Code reviews and architectural discussions
- When user asks: "how should I architect this?", "best practices for...", "how to make this scalable?"

---

## 1. Coding Standards

### General Principles

- **Readability over cleverness**: Code is read 10x more than written. Prefer explicit over implicit.
- **Single Responsibility**: One function/class/module = one reason to change.
- **DRY with judgment**: Extract duplication when it improves clarity; don't abstract prematurely.
- **Fail fast**: Validate inputs early, surface errors clearly.
- **No magic**: Avoid constants, config, or behavior that isn't obvious from context.

### Naming and Structure

- Names reveal intent: `getUserById` not `fetchData`.
- Functions do one thing and do it well; keep them short (< 50 lines typical).
- Prefer small, focused modules over large ones.
- Avoid generic names: `utils`, `helpers`, `misc` — use domain-specific names.

### Error Handling

- Never silently swallow errors. Always handle, log, or propagate.
- Use typed/domain errors when callers need to react differently.
- Include context when wrapping errors (what operation, what failed).

---

## 2. Architectural Principles

### Think Like an Architect

**Before writing code, ask:**

1. **Boundaries**: What are the boundaries of this component? What does it own vs. delegate?
2. **Dependencies**: What does it depend on? Can those be swapped or mocked?
3. **Data flow**: Where does data enter? Where does it leave? Is it unidirectional?
4. **Failure modes**: What happens when dependencies fail? Timeout? Retry? Degrade?
5. **Scale**: What grows (users, requests, data)? Where are bottlenecks?

### Design Principles

| Principle | Apply As |
|-----------|----------|
| **SOLID** | Single responsibility, small interfaces, depend on abstractions |
| **Separation of concerns** | Business logic ≠ infrastructure ≠ presentation |
| **Dependency injection** | Inject dependencies; avoid globals and singletons |
| **Accept interfaces, return concretions** | Enables testability and swapping implementations |
| **Composition over inheritance** | Prefer "has-a" over "is-a" |

### Layering (when applicable)

```
API/Transport → Application/Use Cases → Domain → Infrastructure
```

- Domain: pure business logic, no I/O
- Application: orchestration, use cases
- Infrastructure: DB, HTTP, external APIs

---

## 3. Testing

### Test Pyramid

- **Unit tests**: Fast, many. Test logic in isolation with mocks.
- **Integration tests**: Fewer. Test real interactions (DB, APIs).
- **E2E tests**: Minimal. Critical paths only.

### What to Test

- **Happy path**: Core behavior works.
- **Edge cases**: Empty input, null, zero, max values, invalid formats.
- **Error paths**: Dependency failures, validation failures, timeouts.
- **State changes**: Side effects, transitions.

### Test Design

- Arrange-Act-Assert (AAA) structure.
- One logical assertion per test (or closely related ones).
- Test behavior, not implementation.
- Use table-driven / parameterized tests for multiple scenarios.

### Mocking

- Mock external dependencies (DB, HTTP, file system).
- Don't mock what you own; use real implementations in unit scope when cheap.
- Verify interactions when behavior matters (e.g., "was this called with correct args?").

---

## 4. Edge Cases and Robustness

### Input Validation

- **Null/empty**: How does the system behave with null, empty string, empty array?
- **Bounds**: Zero, negative, max int, overflow.
- **Format**: Invalid JSON, malformed IDs, wrong types.
- **Concurrency**: Race conditions, duplicate requests, out-of-order events.

### Failure Handling

- **Timeouts**: Every external call should have a timeout.
- **Retries**: Define retry policy (backoff, max attempts) for transient failures.
- **Circuit breaker**: Consider for unreliable dependencies.
- **Graceful degradation**: Can the system do something useful when a dependency is down?

### Checklist for New Code

- [ ] Null/empty inputs handled
- [ ] Boundary values (0, -1, max) considered
- [ ] Timeout on external calls
- [ ] Errors logged with context
- [ ] No silent failures

---

## 5. Scalable Solutions

### Horizontal vs. Vertical

- **Horizontal**: Add more instances; design stateless when possible.
- **Vertical**: Add CPU/memory; know limits.

### Patterns for Scale

| Concern | Approach |
|---------|----------|
| **Stateless services** | No local session state; use external store if needed |
| **Async processing** | Queues for non-blocking, background work |
| **Caching** | Cache read-heavy data; define invalidation strategy |
| **Sharding/partitioning** | Split data by key when single node is insufficient |
| **Rate limiting** | Protect APIs and downstream services |
| **Idempotency** | Design operations to be safe when retried |

### Data and Performance

- Index for query patterns; avoid full scans at scale.
- Paginate large result sets.
- Avoid N+1 queries; batch when possible.
- Consider eventual consistency when strong consistency isn't required.

### Observability

- **Metrics**: Latency, throughput, error rate.
- **Logs**: Structured, with correlation IDs.
- **Traces**: For distributed flows.

---

## 6. Security by Default

### Input as Attack Surface

- **Validate all inputs**: Treat user and external input as untrusted. Validate type, format, length, range.
- **Avoid injection**: Use parameterized queries (SQL), escape output (XSS), validate before use.
- **Principle of least privilege**: Services and DB users should have minimal permissions.

### Secrets and Credentials

- **Never** commit secrets, API keys, or passwords to code or config in repo.
- Use environment variables or secret managers (Vault, cloud secrets).
- Rotate credentials; avoid long-lived tokens for production.

### Authentication vs. Authorization

- **Authentication**: Who is this? (identity)
- **Authorization**: What can they do? (permissions)
- Fail closed: deny by default when unsure.

---

## 7. Operational Readiness

### Health Checks

- **Liveness**: Is the process alive? (simple ping)
- **Readiness**: Can it accept traffic? (DB up, dependencies ok)
- Expose endpoints for orchestration (Kubernetes, load balancers).

### Lifecycle

- **Graceful shutdown**: Drain in-flight requests before exit.
- **Startup**: Be ready only when dependencies are reachable.

### Configuration

- 12-factor style: config via environment; avoid hardcoded values.
- Feature flags for safe rollout and kill switches.

---

## 8. Code Review Standards

### What Reviewers Check

- **Correctness**: Logic is right; edge cases handled.
- **Security**: No obvious vulnerabilities; inputs validated.
- **Maintainability**: Readable, testable, follows team patterns.
- **Tests**: Adequate coverage for changes.

### Review Culture

- **Turnaround**: Aim for same-day or next-day for small PRs.
- **Tone**: Constructive; explain *why*, not just *what*.
- **Scope**: Focus on impact. Nitpicks are optional; critical issues are blocking.

### Mentorship

- Use reviews to share context and patterns.
- Leave comments that teach, not just critique.

---

## 9. Technical Debt and Refactoring

### When to Take Debt

- **Acceptable**: Short-term tactical trade-off with a planned payoff.
- **Avoid**: "We'll fix it later" with no plan or ticket.

### Paying Down Debt

- **Boy Scout rule**: Leave code better than you found it.
- **Refactor safely**: Ensure tests exist; change in small, reviewable steps.
- **Dedicate time**: Allocate sprint capacity for debt reduction.

### Refactoring Signals

- Tests are hard to write → design may need change.
- Change touches many files → extract shared logic.
- Comments explain "why" hacks → schedule cleanup.

---

## 10. Documentation Expectations

### When to Document

- **APIs**: OpenAPI/spec, clear request/response, error codes.
- **Architecture**: Components, boundaries, key flows (especially for new services).
- **Decisions**: ADRs for significant choices (see [reference.md](reference.md)).

### What to Document

- **Contracts**: API schemas, event payloads.
- **Assumptions**: What the system assumes about its environment.
- **Failure modes**: How it behaves when dependencies fail.

### Living Docs

- Update docs when code changes; stale docs are worse than none.
- Prefer docs close to code (README, inline comments for non-obvious logic).

---

## Quick Reference Checklist

**Before implementing:**
- [ ] Boundaries and dependencies identified
- [ ] Failure modes considered
- [ ] Input validation strategy defined

**While implementing:**
- [ ] Single responsibility per unit
- [ ] Dependencies injected
- [ ] Errors handled with context
- [ ] No silent failures
- [ ] Inputs validated; no injection vectors
- [ ] Secrets externalized (no hardcoded credentials)

**Before shipping:**
- [ ] Unit tests for logic and edge cases
- [ ] Integration tests for critical paths
- [ ] Timeouts on external calls
- [ ] Observability in place
- [ ] Health checks (liveness/readiness) implemented
- [ ] Graceful shutdown supported
- [ ] API/architecture documented where relevant

**In code review:**
- [ ] Correctness, security, maintainability checked
- [ ] Feedback constructive and timely

---

## Additional Resources

- For language-specific standards: See `golang-coding-standards`, `java-testing`, etc.
- For architecture deep-dives, data privacy, dependencies: See [reference.md](reference.md)
