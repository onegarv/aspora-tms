You are setting up test data builders and fixtures for a Java project. Follow these phases IN ORDER.
Do not skip phases. Each phase has a GATE — complete it before proceeding.

## Pre-Flight: Decision Context

Before starting, check for these files in the current project root (read them if they exist, skip if not):
1. `DECISIONS.md` — prior decisions for this project. Do NOT re-decide what's already settled.
2. `GUARDRAILS.md` — anti-patterns from past incidents. These are hard constraints.

Also check: $HOME/code/aspora/ai-velocity/java-testing/ for DECISIONS.md and GUARDRAILS.md

## Phase 0: Analyze Domain Model

Before creating any test infrastructure:
- Read the domain entities, value objects, and aggregates in the project
- Identify which objects are complex enough to need builders (3+ fields, nested objects)
- Check for existing test builders, fixtures, or factory methods
- Identify shared test data that repeats across multiple test classes

GATE: List: (1) domain objects needing builders, (2) existing test utilities found, (3) shared test data identified. Present to user.

## Phase 1: Test Data Builders

Create one builder per aggregate or key entity:

```java
public class OrderTestBuilder {
    private String orderId = "ORD-001";
    private OrderStatus status = OrderStatus.CREATED;
    private Money amount = MoneyFixtures.tenUSD();
    private Customer customer = CustomerTestBuilder.aCustomer().build();

    public static OrderTestBuilder anOrder() { return new OrderTestBuilder(); }

    public OrderTestBuilder withStatus(OrderStatus status) { this.status = status; return this; }
    public OrderTestBuilder withAmount(Money amount) { this.amount = amount; return this; }

    public Order build() { return new Order(orderId, status, amount, customer); }
}
```

Rules:
- [ ] Builders provide sensible defaults — a bare `.build()` produces a valid object
- [ ] Builder methods use `with{Field}` naming
- [ ] Static factory: `a{Entity}()` or `an{Entity}()`
- [ ] Builders live in `src/test/java/` mirroring the domain package
- [ ] Use domain types, NOT test-only DTOs

## Phase 2: Fixtures (Static Factory Methods)

For value objects and simple shared data:

```java
public class MoneyFixtures {
    public static Money tenUSD() { return new Money(BigDecimal.TEN, Currency.USD); }
    public static Money zeroCurrency(Currency c) { return new Money(BigDecimal.ZERO, c); }
}
```

Rules:
- [ ] One fixture class per domain concept
- [ ] Method names describe the scenario: `validPayment()`, `expiredCard()`, `overdueOrder()`
- [ ] Constants for reusable primitive values: `VALID_EMAIL`, `TEST_CUSTOMER_ID`

Read the skill reference for builder and fixture patterns:
$HOME/code/aspora/ai-velocity/java-testing/SKILL.md

## Phase 3: Wire Into Existing Tests

- [ ] Replace inline object construction in existing tests with builders
- [ ] Replace repeated setup with fixtures
- [ ] Move common `@BeforeEach` setup into shared builders
- [ ] Ensure no test is constructing complex domain objects inline (use builders instead)

## Phase 4: Verification

| Check | Status |
|-------|--------|
| One builder per aggregate/key entity? | [ ] |
| Bare `.build()` produces valid object? | [ ] |
| Builders use domain types (not test DTOs)? | [ ] |
| Fixture methods have descriptive names? | [ ] |
| Existing tests updated to use builders? | [ ] |
| No duplicate test data construction across test classes? | [ ] |
| Builders in src/test/java mirroring domain package? | [ ] |

## Phase 5: Self-Check

1. "Can I create a valid test object with just `anOrder().build()`?" → Must be YES
2. "Do builders use the actual domain types (records, entities)?" → Must be YES
3. "Did I eliminate duplicate object construction across test files?" → Must be YES
4. "Are fixture method names self-documenting?" → Must be YES

If JUnit/Mockito mechanics are also needed, additionally read:
$HOME/code/aspora/ai-velocity/java-unit-testing/SKILL.md
