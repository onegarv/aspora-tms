You are writing or improving Java unit/integration tests. Follow these phases IN ORDER.
Do not skip phases. Each phase has a GATE — complete it before proceeding.

## Pre-Flight: Decision Context

Before starting, check for these files in the current project root (read them if they exist, skip if not):
1. `DECISIONS.md` — prior decisions for this project. Do NOT re-decide what's already settled.
2. `GUARDRAILS.md` — anti-patterns from past incidents. These are hard constraints.

Also check: $HOME/code/aspora/ai-velocity/java-unit-testing/ for DECISIONS.md and GUARDRAILS.md

## Phase 0: Analyze Target Code

Before writing any tests:
- Read the class/service you are testing
- Identify all dependencies (constructor-injected interfaces)
- Identify edge cases: null inputs, empty collections, invalid state transitions
- Check for existing tests — extend, don't duplicate
- Check for existing test utilities: builders, fixtures, base test classes

GATE: List: (1) classes to test, (2) dependencies to mock, (3) edge cases identified, (4) existing test utilities found. Present to user.

## Phase 1: Test Structure

Set up correctly:
- [ ] Test class: `src/test/java/.../{ClassName}Test.java` (mirrors main structure)
- [ ] Integration test: `{ClassName}IntegrationTest.java`
- [ ] Use JUnit 5 (`@Test`, `@DisplayName`, `@Nested`, `@ParameterizedTest`)
- [ ] Use Mockito for mocking (`@Mock`, `@InjectMocks`, `@ExtendWith(MockitoExtension.class)`)
- [ ] Use AssertJ for assertions (prefer over JUnit assertions)

## Phase 2: Write Tests Following AAA Pattern

Every test MUST follow Arrange-Act-Assert with blank lines between sections:

```java
@Test
@DisplayName("should create order when request is valid")
void shouldCreateOrder_whenRequestIsValid() {
    // Arrange
    var request = CreateOrderRequest.builder()...build();
    when(repository.save(any())).thenReturn(expectedOrder);

    // Act
    var result = service.createOrder(request);

    // Assert
    assertThat(result.status()).isEqualTo(OrderStatus.CREATED);
    verify(repository).save(any(Order.class));
}
```

Read the skill reference for test patterns:
$HOME/code/aspora/ai-velocity/java-unit-testing/SKILL.md

## Phase 3: Use Parameterized Tests for Similar Scenarios

When multiple test cases follow the same pattern, use `@ParameterizedTest`:

```java
@ParameterizedTest
@MethodSource("invalidRequests")
@DisplayName("should reject invalid requests")
void shouldReject_whenRequestInvalid(CreateOrderRequest request, String expectedError) {
    assertThatThrownBy(() -> service.createOrder(request))
        .isInstanceOf(ValidationException.class)
        .hasMessageContaining(expectedError);
}
```

## Phase 4: Coverage Requirements

Every test suite must cover:
- [ ] Happy path — normal successful operation
- [ ] Validation failures — invalid inputs, null values, empty collections
- [ ] Error paths — every exception thrown by the method
- [ ] Edge cases — boundary values, state transitions, concurrent access
- [ ] Mock verification — verify interactions with dependencies

## Phase 5: Test Data Management

Check if the project has test builders/fixtures. If yes, USE THEM. If not, create them:
- [ ] Use test data builders for complex domain objects (see /test:java-fixtures)
- [ ] Use `@BeforeEach` for common setup — DRY principle
- [ ] Constants for reusable test values
- [ ] No hardcoded magic values in test bodies

Read fixtures skill if needed:
$HOME/code/aspora/ai-velocity/java-testing/SKILL.md

## Phase 6: Verification

| Check | Status |
|-------|--------|
| AAA pattern with blank line separation? | [ ] |
| @DisplayName on every test method? | [ ] |
| Happy path covered? | [ ] |
| Error/exception paths covered? | [ ] |
| Edge cases (null, empty, boundary) covered? | [ ] |
| Parameterized tests for similar scenarios? | [ ] |
| Mocks verified (verify() calls)? | [ ] |
| AssertJ used (not JUnit assertions)? | [ ] |
| Existing test builders/fixtures reused? | [ ] |
| Tests pass: `./gradlew test` or `mvn test`? | [ ] |

## Phase 7: Self-Check

1. "Does every test follow AAA with clear section separation?" → Must be YES
2. "Am I testing behavior, not implementation?" → Must be YES (tests survive refactors)
3. "Am I using AssertJ, not JUnit assertions?" → Must be YES
4. "Am I using existing test builders/fixtures if available?" → Must be YES
5. "Would a new team member understand each test from its @DisplayName alone?" → Must be YES
