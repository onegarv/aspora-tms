# Testing Fundamentals — Reference

Deeper reference for the testing-fundamentals skill.

## Contract Testing

Verify API contracts between consumer and provider without full integration:

- **Consumer-driven**: Consumer defines expected contract; provider tests against it
- **Provider-driven**: Provider publishes contract; consumer verifies compatibility
- **Tools**: Pact, Spring Cloud Contract (Java), or OpenAPI-based contract tests

## Test Doubles Summary

| Double | Controls input | Controls output | Verifies calls |
|--------|----------------|-----------------|----------------|
| Stub | No | Yes | No |
| Mock | Yes | Yes | Yes |
| Spy | Partial | Delegates | Yes |
| Fake | No | Real (simplified) | No |

## E2E Test Data

- Use **test fixtures** or **factories** for consistent data
- **Database seeding**: Separate test DB or Testcontainers with known state
- **Cleanup**: Each test should leave no side effects, or use transactions that roll back

## Flakiness Checklist

- [ ] No `Thread.sleep()` — use polling or explicit waits
- [ ] No `new Date()` or `System.currentTimeMillis()` — inject clock
- [ ] No `Math.random()` — inject or seed random
- [ ] No shared static state — each test isolated
- [ ] No order-dependent tests — any order should pass
