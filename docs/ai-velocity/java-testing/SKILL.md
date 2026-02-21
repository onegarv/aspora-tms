---
name: java-testing
description: "Generate high-quality unit, integration, and E2E test cases for Java services using JUnit 5, Mockito, Spring Boot Test, RestAssured, and Testcontainers. Covers mocking strategy, test pyramid, and when to use each test type. Use when writing tests, deciding what to mock, or adding E2E tests."
---

# Java Testing Skill

## Purpose
This skill enables AI agents to generate high-quality unit, integration, and E2E test cases for Java services. Covers the full test pyramid: unit tests with Mockito, integration tests with Spring Boot Test, and E2E with RestAssured or Testcontainers. Includes mocking strategy and when to use each approach. Designed for Spring Boot applications with JUnit 5.

**See also:** [testing-fundamentals](organizational-skills/testing-fundamentals/) for language-agnostic testing principles.

## When to Trigger This Skill
- When user requests: "Write test cases for this Java class/service..."
- When deciding what to mock vs use real (Mockito vs WireMock vs Testcontainers)
- When adding E2E or API tests
- After code changes or pull requests to validate new/modified functionality
- When no tests exist for a class or method
- When user asks: "Add tests", "Write unit tests", "Integration tests", "E2E tests", "how to mock this?"

## Critical: Automatic Codebase Analysis

**ALWAYS analyze the codebase first before asking questions or making assumptions.**

1. **Read the codebase automatically**:
   - Read the class/service to be tested
   - Identify all public methods and their behaviors
   - Understand dependencies from constructor injection or field injection
   - Identify edge cases from method logic
   - Check existing test patterns in the codebase

2. **Do NOT ask the user to describe the class**:
   - The codebase contains all necessary information
   - Understand behavior from method implementations
   - Identify dependencies from imports and field declarations
   - Discover edge cases from conditional logic

3. **Only ask clarifying questions if**:
   - Code is unclear or ambiguous
   - Multiple interpretations are possible
   - Critical information is missing from codebase

## Core Principles

### SOLID Principles in Testing

1. **Single Responsibility Principle (SRP)**
   - Each test class should focus on testing one class/component
   - Each test method should test one specific behavior or scenario
   - Avoid testing multiple behaviors in a single test method

2. **Open/Closed Principle**
   - Tests should be extensible without modifying existing tests
   - Use parameterized tests for similar scenarios
   - Add new tests rather than rewriting existing ones

3. **Liskov Substitution Principle**
   - Use interfaces and abstractions in test code
   - Mock dependencies using interfaces, not concrete implementations

4. **Interface Segregation Principle**
   - Mock only what you need, not entire interfaces
   - Use specific mock configurations for each test

5. **Dependency Inversion Principle**
   - Dependencies should be mocked/injected, not hardcoded
   - Test code should depend on abstractions (interfaces), not concrete implementations
   - Use dependency injection frameworks (Mockito, etc.)

### DRY (Don't Repeat Yourself) Principle

- Extract common setup into `@BeforeEach` methods or test fixtures
- Use helper methods and builders for test data creation
- Create reusable test utilities and factories
- Use parameterized tests (`@ParameterizedTest`) for similar scenarios with different inputs
- Avoid duplicating test logic across multiple test methods

### AAA Pattern (Arrange-Act-Assert)

Every test should follow this structure. Also known as "Given-When-Then" pattern:
```java
@Test
@DisplayName("Descriptive test name")
void methodName_condition_expectedBehavior() {
    // Arrange: Set up test data, mocks, and preconditions
    
    // Act: Execute the method under test
    
    // Assert: Verify the expected behavior
}
```

**Best Practice**: Use blank lines between each section (Arrange, Act, Assert) to improve readability without requiring comments. This makes the test structure immediately clear.

## Test Structure & Organization

### Directory Structure
- Tests must be placed in `src/test/java/...` mirroring the package structure of `src/main/java/...`
- Integration tests should be in a separate package (e.g., `...integration`) or clearly annotated
- Test class naming: `<ClassName>Test.java` for unit tests, `<ClassName>IntegrationTest.java` for integration tests

### Test Class Structure for Unit Tests

```java
package tech.vance.goblin.service;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

// Note: Prefer AssertJ over JUnit assertions for better readability
// import static org.junit.jupiter.api.Assertions.*;  // Use only if AssertJ unavailable

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * Unit tests for {@link ServiceClass}.
 * 
 * <p>This test class follows SOLID and DRY principles:
 * <ul>
 *   <li>Single Responsibility: Each test method tests one specific behavior</li>
 *   <li>DRY: Common setup and assertions are extracted to helper methods</li>
 *   <li>Dependency Injection: Dependencies are mocked using Mockito</li>
 * </ul>
 * 
 * <p>Test coverage includes:
 * <ul>
 *   <li>Happy path scenarios</li>
 *   <li>Edge cases (null inputs, empty collections, boundary values)</li>
 *   <li>Error cases (exceptions, failures)</li>
 *   <li>Boundary conditions</li>
 * </ul>
 */
@ExtendWith(MockitoExtension.class)
class ServiceClassTest {
    
    @Mock
    private Dependency dependency;
    
    @Mock
    private AnotherDependency anotherDependency;
    
    @InjectMocks
    private ServiceClass serviceClass;
    
    @BeforeEach
    void setUp() {
        // Common setup for all tests
        // Configure default mock behaviors if needed
    }
    
    @AfterEach
    void tearDown() {
        // Cleanup if needed
        // Reset mocks if necessary
    }
    
    @Nested
    @DisplayName("Method Name Tests")
    class MethodNameTests {
        
        @Test
        @DisplayName("Should process valid input successfully")
        void methodName_withValidInput_returnsSuccess() {
            // Arrange
            
            // Act
            
            // Assert
        }
        
        @Test
        @DisplayName("Should throw exception when input is null")
        void methodName_withNullInput_throwsException() {
            // Arrange
            
            // Act & Assert
        }
    }
    
    // Note: @BeforeAll and @AfterAll don't work in @Nested classes by default
    // due to Java's static member restrictions. Use @BeforeEach/@AfterEach instead.
    
    // Helper methods for test data creation
    private TestData createTestData() {
        return TestData.builder()
            .field1("value1")
            .field2("value2")
            .build();
    }
}
```

### Test Class Structure for Spring Boot Integration Tests

```java
package tech.vance.goblin.service;

import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;

/**
 * Integration tests for {@link ServiceClass}.
 * 
 * <p>These tests verify the complete flow with real Spring context
 * and mocked external dependencies.
 */
@SpringBootTest
@ActiveProfiles("test")
class ServiceClassIntegrationTest {
    
    @Autowired
    private ServiceClass serviceClass;
    
    @MockBean
    private ExternalServiceClient externalServiceClient;
    
    @Test
    @DisplayName("Should process order end-to-end")
    void processOrder_endToEnd_success() {
        // Integration test with real Spring context
    }
}
```

### Test Method Naming Conventions

Use descriptive names that clearly indicate:
- What is being tested
- Under what condition
- What the expected outcome is

**Recommended patterns:**
- `methodName_condition_expectedBehavior` (e.g., `calculateTax_zeroAmount_returnsZero`)
- `shouldDoX_whenY` (e.g., `shouldProcessOrder_whenAllDependenciesSucceed`)
- `givenX_whenY_thenZ` (Given-When-Then pattern, e.g., `givenValidOrder_whenProcessing_thenReturnsOrderId`)
- `testMethodName_WithCondition_EdgeCase` (e.g., `testProcess_WithNullInput_EdgeCase`)

**Best Practice**: Test names should be specific and descriptive. Vague names like "CausesFailure" make tests harder to maintain. The method name should clearly communicate what scenario is being tested and what conditions trigger it.

## Test Coverage Requirements

### Happy Path Scenarios
- Test normal, expected behavior with valid inputs
- Verify successful execution and correct results
- Test typical use cases

### Edge Cases
- **Null inputs**: Test behavior when null is passed
- **Empty collections/strings**: Test with empty lists, strings, maps, etc.
- **Boundary values**: Test minimum, maximum, and boundary values
- **Invalid formats**: Test with malformed data, invalid formats
- **Special characters**: Test with special characters, Unicode, etc.

### Error Cases
- **Exception handling**: Test that appropriate exceptions are thrown
- **Dependency failures**: Test behavior when dependencies fail
- **Invalid state**: Test behavior with invalid object states
- **Resource exhaustion**: Test behavior when resources are unavailable

### Boundary Conditions
- **Minimum values**: Test with smallest valid inputs
- **Maximum values**: Test with largest valid inputs
- **Zero/empty**: Test with zero, empty, or null values
- **Just above/below limits**: Test values just above and below boundaries

## Mocking Strategy: When to Use What

| Tool | Use For | When |
|------|---------|------|
| **Mockito** | Unit tests | Mock dependencies (repos, clients) in isolation |
| **@MockBean** | Integration tests | Mock Spring beans (e.g., external HTTP client) |
| **WireMock** | HTTP mocking | When service calls external APIs; stub responses |
| **Testcontainers** | Real DB/Redis/etc. | Integration tests needing real infrastructure |
| **RestAssured** | API E2E | Full HTTP request → response; test REST contracts |

### Decision Flow

- **Unit test?** → Mockito for all external dependencies
- **Integration test, external API?** → WireMock or @MockBean
- **Integration test, DB?** → Testcontainers (real Postgres, Redis) or H2 for simpler cases
- **E2E API test?** → RestAssured + Spring Boot Test (full context) or Testcontainers for full stack

### Avoid Over-Mocking

- Don't mock domain logic; test it with real code
- Don't mock value objects; use real instances
- Prefer Testcontainers over mocking DB when tests need real SQL behavior

## E2E Testing

### API E2E with RestAssured

```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
class OrderApiE2ETest {
    @LocalServerPort
    int port;

    @BeforeEach
    void setUp() {
        RestAssured.port = port;
    }

    @Test
    @DisplayName("Should create order end-to-end")
    void createOrder_e2e_success() {
        given()
            .contentType(ContentType.JSON)
            .body(Map.of("amount", 100, "currency", "USD"))
        .when()
            .post("/api/v1/orders")
        .then()
            .statusCode(201)
            .body("orderId", notNullValue());
    }
}
```

### Full Stack with Testcontainers

For E2E with real DB, message queue, etc.:

```java
@SpringBootTest
@Testcontainers
@ActiveProfiles("test")
class OrderE2ETest {
    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15");

    @DynamicPropertySource
    static void configure(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
    }

    @Autowired
    private OrderService orderService;

    @Test
    void fullFlow_e2e_success() {
        // Real DB, real service; mock only external APIs
    }
}
```

### E2E Guidelines

- **Minimal**: Only critical paths (checkout, signup, core flows)
- **Idempotent**: Each test independent; clean up data
- **Fast enough**: Consider excluding from default `./gradlew test`; run in CI on main

## Testing Frameworks & Tools

### Required Frameworks
- **JUnit 5** (Jupiter) - Primary testing framework
- **Mockito** - For mocking dependencies
- **Spring Boot Test** - For Spring Boot integration tests
- **AssertJ** (strongly recommended) - For fluent, readable assertions. Prefer AssertJ over JUnit assertions for better readability and more expressive error messages

### Additional Tools (when applicable)
- **Testcontainers** - Real DB, Redis, etc. for integration and E2E tests
- **RestAssured** - API E2E; fluent HTTP request/response testing
- **WireMock** - Mock HTTP services (external APIs)
- **Spring Security Test** - For security testing
- **Parameterized Tests** - For testing multiple inputs with same logic

## Spring Boot Specific Patterns

### Unit Testing with Spring Boot

For unit tests, **DO NOT** use `@SpringBootTest`. Use `@ExtendWith(MockitoExtension.class)` instead:

```java
@ExtendWith(MockitoExtension.class)
class ServiceClassTest {
    @Mock
    private Dependency dependency;
    
    @InjectMocks
    private ServiceClass serviceClass;  // Constructor injection works automatically
}
```

### Integration Testing with Spring Boot

For integration tests, use `@SpringBootTest`:

```java
@SpringBootTest
@ActiveProfiles("test")
class ServiceClassIntegrationTest {
    @Autowired
    private ServiceClass serviceClass;
    
    @MockBean  // Use @MockBean for Spring beans
    private ExternalServiceClient externalServiceClient;
}
```

### Testing Spring Controllers

```java
@WebMvcTest(ControllerClass.class)
class ControllerClassTest {
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private ServiceClass serviceClass;
    
    @Test
    void testEndpoint_returnsSuccess() throws Exception {
        when(serviceClass.method()).thenReturn(result);
        
        mockMvc.perform(get("/api/endpoint"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.field").value("expected"));
    }
}
```

### Testing with Spring Data JPA

```java
@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
class RepositoryTest {
    @Autowired
    private TestEntityManager entityManager;
    
    @Autowired
    private RepositoryClass repository;
    
    @Test
    void findByField_returnsEntity() {
        // Test repository methods
    }
}
```

## Documentation Requirements

### Class-Level Documentation
Each test class should include:
- JavaDoc explaining what class is being tested
- List of principles followed (SOLID, DRY)
- Summary of test coverage (happy path, edge cases, error cases, etc.)

### Method-Level Documentation
Each test method should include:
- `@DisplayName` annotation for human-readable test names
- JavaDoc comment explaining:
  - What scenario is being tested
  - What the expected behavior is
  - Any special preconditions or setup

Example:
```java
/**
 * Tests the happy path scenario where a valid order is processed successfully.
 *
 * <p>Scenarios:
 * <ul>
 *   <li>Valid order data provided</li>
 *   <li>All dependencies are properly initialized</li>
 *   <li>Order is processed without exceptions</li>
 * </ul>
 *
 * <p>Expected Behavior: Order should be processed successfully and return order ID.
 */
@Test
@DisplayName("Should process valid order successfully")
void testProcessOrder_WithValidOrder_HappyPath() {
    // Test implementation
}
```

## Using @Nested Classes for Organization

Use `@Nested` classes to group related tests. This improves readability, enhances maintainability, reduces code duplication, and serves as living documentation:

```java
@ExtendWith(MockitoExtension.class)
class ServiceClassTest {
    
    @Nested
    @DisplayName("Method A Tests")
    class MethodATests {
        @Test
        @DisplayName("Should process valid input successfully")
        void methodA_happyPath_success() { }
        
        @Test
        @DisplayName("Should throw exception when input is null")
        void methodA_nullInput_throwsException() { }
    }
    
    @Nested
    @DisplayName("Method B Tests")
    class MethodBTests {
        @Test
        @DisplayName("Should return expected result")
        void methodB_happyPath_success() { }
    }
}
```

**Best Practices for @Nested:**
- Use meaningful nested class names that clearly indicate functionality (e.g., `UserValidationTests` instead of generic names)
- Apply `@DisplayName` to both nested classes and methods for maximum readability
- Group by scenario: each `@Nested` class represents a specific scenario or aspect of functionality
- Leverage setup/teardown inheritance: `@BeforeEach`/`@AfterEach` execute from parent to nested classes
- **Important**: `@BeforeAll` and `@AfterAll` don't work in `@Nested` classes by default due to Java's static member restrictions. Use `@BeforeEach`/`@AfterEach` instead.

## Quality Checklist

Before delivering test code, ensure:

- [ ] Every public method has at least one happy path test
- [ ] Edge cases are covered (null, empty, boundary values)
- [ ] Error cases are covered (exceptions, failures)
- [ ] Tests follow AAA pattern (Arrange-Act-Assert)
- [ ] Common setup is extracted to `@BeforeEach` or helper methods
- [ ] Test methods are independent (no shared mutable state)
- [ ] Tests are deterministic (no randomness, no external dependencies)
- [ ] Dependencies are properly mocked
- [ ] Test names are descriptive and follow naming conventions
- [ ] Documentation is clear and explains what each test does
- [ ] No test logic duplication (DRY principle followed)
- [ ] Integration tests are separate from unit tests
- [ ] Tests are placed in correct directory structure
- [ ] `@DisplayName` annotations are used for readability
- [ ] `@Nested` classes are used for logical grouping

## Best Practices

### Do's ✅
- **Prioritize critical and high-risk areas** of code first
- **Use meaningful variable names** in tests
- **Keep tests small and focused** - one assertion per test when possible
- **Use builders or factories** for complex test data creation
- **Verify both return values and side effects** (if applicable)
- **Test behavior, not implementation** - focus on what, not how
- **Use parameterized tests** for similar scenarios
- **Mock external dependencies** (databases, APIs, file system)
- **Clean up resources** in `@AfterEach` if needed
- **Use descriptive assertions** with clear failure messages
- **Prefer AssertJ** over JUnit assertions for fluent, readable assertions
- **Use `@Nested` classes** to organize related tests and improve maintainability
- **Use `@DisplayName`** for human-readable test names (both on classes and methods)
- **Extract helper methods** for common test data creation
- **Use blank lines** between AAA sections for better readability
- **Aim for 80% coverage** as the industry standard (focus on critical paths)

### Don'ts ❌
- **Don't test implementation details** - test public behavior
- **Don't use shared mutable state** between tests
- **Don't write tests that depend on test execution order**
- **Don't embed complex logic** in test code (mirrors production logic)
- **Don't use hard-coded values** that are environment-dependent
- **Don't test trivial getters/setters** unless they contain logic
- **Don't create overly broad tests** that test multiple behaviors
- **Don't skip documentation** - explain what and why
- **Don't ignore flaky tests** - fix or remove them
- **Don't test framework code** - test your code
- **Don't use `@SpringBootTest` for unit tests** - use `@ExtendWith(MockitoExtension.class)`
- **Don't use `@Mock` with `@SpringBootTest`** - use `@MockBean` instead
- **Don't use `@BeforeAll`/`@AfterAll` in `@Nested` classes** - use `@BeforeEach`/`@AfterEach` instead
- **Don't use vague test names** - be specific about what scenario is being tested

## Example Test Generation Workflow

When generating tests, follow this workflow:

1. **Analyze the code**
   - Identify public methods and their behaviors
   - Identify dependencies that need mocking
   - Identify edge cases and error conditions
   - Check existing test patterns in the codebase

2. **Plan test scenarios**
   - List happy path scenarios
   - List edge cases (null, empty, boundary)
   - List error cases (exceptions, failures)
   - Determine if integration tests are needed

3. **Create test structure**
   - Set up test class with proper annotations
   - Create mocks for dependencies
   - Set up `@BeforeEach` for common initialization
   - Organize tests using `@Nested` classes

4. **Write test methods**
   - Follow AAA pattern
   - Use descriptive names
   - Add `@DisplayName` annotations
   - Add documentation
   - Cover all identified scenarios

5. **Review and refine**
   - Check for duplication (apply DRY)
   - Verify all principles are followed
   - Ensure documentation is clear
   - Verify test independence

## Running Tests

### Prerequisites Check

**Before running tests, verify Java is set up:**
```bash
# Check Java version (should be 17+)
java -version

# Check JAVA_HOME is set
echo $JAVA_HOME  # macOS/Linux
echo %JAVA_HOME% # Windows

# Verify Gradle can find Java
./gradlew --version
```

**If Java is not found, see [Java Runtime Setup](#java-runtime-setup) section above.**

### Local Test Execution

#### Using Gradle (Recommended)

**Run all tests:**
```bash
./gradlew test
```

**Run tests for a specific class:**
```bash
./gradlew test --tests "tech.vance.goblin.service.ServiceClassTest"
```

**Run tests matching a pattern:**
```bash
./gradlew test --tests "*Test"
./gradlew test --tests "*IntegrationTest"
```

**Run tests with verbose output:**
```bash
./gradlew test --info
```

**Run tests in parallel (faster execution):**
```bash
./gradlew test --parallel
```

**Clean and run tests:**
```bash
./gradlew clean test
```

**Run tests continuously (watch mode):**
```bash
./gradlew test --continuous
```

**Run only unit tests (exclude integration):**
```bash
./gradlew test --tests "*Test" --exclude-tests "*IntegrationTest"
```

**Run only integration tests:**
```bash
./gradlew test --tests "*IntegrationTest"
```

#### Using IDE

**IntelliJ IDEA:**
- Right-click on test class → Run 'TestClassName'
- Right-click on test method → Run 'testMethodName'
- Use keyboard shortcut: `Ctrl+Shift+F10` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Run all tests in package: Right-click package → Run 'Tests in package'

**Eclipse:**
- Right-click on test class → Run As → JUnit Test
- Use keyboard shortcut: `Alt+Shift+X, T`

**VS Code:**
- Use Java Test Runner extension
- Click "Run Test" above test methods

### Test Execution Options

**Skip tests during build:**
```bash
./gradlew build -x test
```

**Run tests with specific JVM options:**
```bash
./gradlew test -Dorg.gradle.jvmargs="-Xmx2048m -XX:MaxMetaspaceSize=512m"
```

**Run tests with specific profile:**
```bash
./gradlew test -Dspring.profiles.active=test
```

## Code Coverage Setup

### JaCoCo Configuration

#### Add JaCoCo Plugin to build.gradle

```gradle
plugins {
    id 'java'
    id 'jacoco'
    // ... other plugins
}

// JaCoCo configuration
jacoco {
    toolVersion = "0.8.10"
    reportsDirectory = layout.buildDirectory.dir('reports/jacoco')
}

// Test task configuration
test {
    finalizedBy jacocoTestReport
    useJUnitPlatform()
}

// Generate coverage report
jacocoTestReport {
    dependsOn test
    reports {
        xml.required = true
        csv.required = false
        html.required = true
        html.outputLocation = layout.buildDirectory.dir('reports/jacoco/test/html')
    }
    
    // Exclude classes/packages from coverage
    afterEvaluate {
        classDirectories.setFrom(files(classDirectories.files.collect {
            fileTree(dir: it, exclude: [
                '**/config/**',
                '**/dto/**',
                '**/enums/**',
                '**/models/**',
                '**/*Application.class',
                '**/*Config.class'
            ])
        }))
    }
}

// Coverage verification
jacocoTestCoverageVerification {
    dependsOn jacocoTestReport
    violationRules {
        rule {
            limit {
                minimum = 0.80  // 80% minimum coverage
            }
        }
        
        rule {
            element = 'CLASS'
            excludes = [
                '*.dto.*',
                '*.enums.*',
                '*.models.*',
                '*.config.*'
            ]
            limit {
                counter = 'LINE'
                minimum = 0.75  // 75% minimum for classes
            }
        }
        
        rule {
            element = 'METHOD'
            limit {
                counter = 'BRANCH'
                minimum = 0.70  // 70% branch coverage
            }
        }
    }
}

// Make coverage verification part of build
check.dependsOn jacocoTestCoverageVerification
```

### Coverage Commands

**Generate coverage report:**
```bash
./gradlew jacocoTestReport
```

**View coverage report:**
```bash
# Open in browser
open build/reports/jacoco/test/html/index.html
```

**Verify coverage thresholds:**
```bash
./gradlew jacocoTestCoverageVerification
```

**Generate and verify in one command:**
```bash
./gradlew test jacocoTestReport jacocoTestCoverageVerification
```

### Coverage Metrics

JaCoCo tracks several coverage metrics:

- **Line Coverage**: Percentage of lines executed
- **Branch Coverage**: Percentage of branches (if/else, switch) executed
- **Method Coverage**: Percentage of methods called
- **Class Coverage**: Percentage of classes with at least one method executed
- **Instruction Coverage**: Percentage of bytecode instructions executed

### Recommended Coverage Thresholds

| Metric | Minimum Threshold | Recommended Threshold |
|--------|----------------------|----------------------|
| Overall Line Coverage | 70% | **80%** (Industry Standard) |
| Overall Branch Coverage | 60% | 70% |
| Critical Classes | 90% | 95% |
| Service Classes | 80% | 85% |
| Utility Classes | 75% | 80% |
| DTOs/Models | Excluded | Excluded |

**Note**: 80% code coverage is the widely accepted industry standard for production code. Focus on critical paths and business logic rather than achieving 100% coverage.

## AssertJ Best Practices

**Why AssertJ?**
- Fluent, readable assertions that read like natural language
- More expressive error messages when tests fail
- Better IDE support with autocomplete
- Chainable assertions for complex validations

**Common AssertJ Patterns:**

```java
import static org.assertj.core.api.Assertions.*;

// Basic assertions
assertThat(result).isNotNull();
assertThat(result).isEqualTo(expected);
assertThat(result).isNotEqualTo(unexpected);

// String assertions
assertThat(str).isNotEmpty();
assertThat(str).contains("substring");
assertThat(str).startsWith("prefix");

// Collection assertions
assertThat(list).hasSize(3);
assertThat(list).contains("item");
assertThat(list).doesNotContain("badItem");
assertThat(list).isEmpty();

// Exception assertions
assertThatThrownBy(() -> service.method())
    .isInstanceOf(IllegalArgumentException.class)
    .hasMessage("Expected error message");

// Optional assertions
assertThat(optional).isPresent();
assertThat(optional).isEmpty();
assertThat(optional).contains(expectedValue);

// Chained assertions
assertThat(order)
    .isNotNull()
    .extracting(Order::getStatus, Order::getAmount)
    .containsExactly(OrderStatus.COMPLETED, BigDecimal.valueOf(100));
```

## Common Patterns & Examples

### Pattern 1: Testing Service Methods with Dependencies

```java
@ExtendWith(MockitoExtension.class)
class OrderServiceTest {
    
    @Mock
    private OrderRepository orderRepository;
    
    @Mock
    private PaymentService paymentService;
    
    @InjectMocks
    private OrderService orderService;
    
    @Test
    @DisplayName("Should create order successfully")
    void createOrder_withValidRequest_returnsOrder() {
        // Arrange
        CreateOrderRequest request = CreateOrderRequest.builder()
            .userId("user-123")
            .amount(BigDecimal.valueOf(100))
            .build();
        
        OrderEntity savedOrder = OrderEntity.builder()
            .orderId("order-456")
            .userId("user-123")
            .build();
        
        when(orderRepository.save(any(OrderEntity.class))).thenReturn(savedOrder);
        when(paymentService.processPayment(any())).thenReturn(PaymentResult.success());
        
        // Act
        OrderEntity result = orderService.createOrder(request);
        
        // Assert
        assertThat(result).isNotNull();
        assertThat(result.getOrderId()).isEqualTo("order-456");
        verify(orderRepository).save(any(OrderEntity.class));
        verify(paymentService).processPayment(any());
    }
}
```

### Pattern 2: Testing Exception Scenarios

```java
@Test
@DisplayName("Should throw exception when order not found")
void getOrder_withInvalidId_throwsException() {
    // Arrange
    String orderId = "invalid-id";
    when(orderRepository.findById(orderId)).thenReturn(Optional.empty());
    
    // Act & Assert
    assertThatThrownBy(() -> orderService.getOrder(orderId))
        .isInstanceOf(OrderNotFoundException.class);
    
    verify(orderRepository).findById(orderId);
}
```

### Pattern 3: Testing with Parameterized Tests

```java
@ParameterizedTest
@ValueSource(strings = {"", " ", "  "})
@DisplayName("Should reject empty or blank input")
void validateInput_withBlankInput_throwsException(String input) {
    // Arrange
    
    // Act & Assert
    assertThatThrownBy(() -> serviceClass.processInput(input))
        .isInstanceOf(ValidationException.class);
}
```

### Pattern 4: Testing Spring Boot Controllers

```java
@WebMvcTest(OrderController.class)
class OrderControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private OrderService orderService;
    
    @Test
    @DisplayName("Should return order when found")
    void getOrder_withValidId_returnsOrder() throws Exception {
        // Arrange
        OrderDto order = OrderDto.builder()
            .orderId("order-123")
            .build();
        
        when(orderService.getOrder("order-123")).thenReturn(order);
        
        // Act & Assert
        mockMvc.perform(get("/api/orders/order-123"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.orderId").value("order-123"));
    }
}
```

### Pattern 5: Testing with Spring Security

```java
@WebMvcTest(OrderController.class)
class OrderControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private OrderService orderService;
    
    @Test
    @WithMockUser(roles = "USER")
    @DisplayName("Should allow authenticated user to access endpoint")
    void getOrder_withAuthenticatedUser_returnsOrder() throws Exception {
        // Test with authentication
    }
    
    @Test
    @DisplayName("Should reject unauthenticated requests")
    void getOrder_withoutAuthentication_returnsUnauthorized() throws Exception {
        mockMvc.perform(get("/api/orders/order-123"))
            .andExpect(status().isUnauthorized());
    }
}
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test and Coverage

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up JDK 21
        uses: actions/setup-java@v4
        with:
          java-version: '21'
          distribution: 'temurin'
          cache: 'gradle'
      
      - name: Grant execute permission for gradlew
        run: chmod +x gradlew
      
      - name: Run tests with coverage
        run: ./gradlew test jacocoTestReport
        env:
          PACKAGES_TOKEN: ${{ secrets.PACKAGES_TOKEN }}
          PACKAGES_USER: ${{ secrets.PACKAGES_USER }}
      
      - name: Verify coverage threshold
        run: ./gradlew jacocoTestCoverageVerification
      
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./build/reports/jacoco/test/jacocoTestReport.xml
          flags: unittests
          name: codecov-umbrella
      
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results
          path: build/test-results/test/
          retention-days: 30
```

## Java Runtime Setup

### Prerequisites

Before running tests, ensure Java is properly installed and configured:

**Required Java Version:**
- Most Spring Boot 3.x projects require **Java 17 or later**
- Check `build.gradle` or `pom.xml` for `sourceCompatibility` or `java.version`

### Setting Up Java Runtime

#### macOS

**1. Install Java (if not installed):**
```bash
# Using Homebrew
brew install openjdk@17

# Or download from Oracle/Adoptium
# https://adoptium.net/
```

**2. Set JAVA_HOME:**
```bash
# Find Java installation
/usr/libexec/java_home -V

# Set JAVA_HOME for current session
export JAVA_HOME=$(/usr/libexec/java_home -v 17)

# Add to ~/.zshrc or ~/.bash_profile for persistence
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
source ~/.zshrc
```

**3. Verify installation:**
```bash
java -version
echo $JAVA_HOME
```

#### Linux

**1. Install Java:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install openjdk-17-jdk

# CentOS/RHEL
sudo yum install java-17-openjdk-devel
```

**2. Set JAVA_HOME:**
```bash
# Find Java installation
sudo update-alternatives --config java

# Set JAVA_HOME (replace path with actual installation)
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Add to ~/.bashrc for persistence
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

#### Windows

**1. Install Java:**
- Download from https://adoptium.net/
- Run installer and follow prompts

**2. Set JAVA_HOME:**
```cmd
# Find Java installation (usually in Program Files)
# Set environment variable
setx JAVA_HOME "C:\Program Files\Java\jdk-17"
setx PATH "%PATH%;%JAVA_HOME%\bin"

# Restart terminal/IDE after setting
```

**3. Verify installation:**
```cmd
java -version
echo %JAVA_HOME%
```

### Using SDKMAN (Recommended for macOS/Linux)

SDKMAN makes Java version management easier:

```bash
# Install SDKMAN
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"

# Install Java 17
sdk install java 17.0.8-tem

# Set as default
sdk default java 17.0.8-tem

# Verify
java -version
```

### Gradle-Specific Java Configuration

**Option 1: Use gradle.properties**
Create or edit `~/.gradle/gradle.properties`:
```properties
org.gradle.java.home=/path/to/java/home
```

**Option 2: Use project gradle.properties**
Create `gradle.properties` in project root:
```properties
org.gradle.java.home=/path/to/java/home
```

**Option 3: Set in build.gradle**
```gradle
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}
```

### Verifying Java Setup for Tests

**Check Java version:**
```bash
java -version
# Should show: openjdk version "17.x.x" or similar
```

**Check JAVA_HOME:**
```bash
echo $JAVA_HOME  # macOS/Linux
echo %JAVA_HOME% # Windows
# Should point to JDK installation directory
```

**Test Gradle can find Java:**
```bash
./gradlew --version
# Should show Java version in output
```

## Troubleshooting

### Common Issues

**Java Runtime Not Found:**
```
ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
```

**Solutions:**
1. **Install Java 17+** (see Java Runtime Setup above)
2. **Set JAVA_HOME environment variable:**
   ```bash
   # macOS
   export JAVA_HOME=$(/usr/libexec/java_home -v 17)
   
   # Linux
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
   
   # Windows
   setx JAVA_HOME "C:\Program Files\Java\jdk-17"
   ```
3. **Add Java to PATH:**
   ```bash
   export PATH=$JAVA_HOME/bin:$PATH
   ```
4. **Use Gradle toolchain** (recommended):
   ```gradle
   java {
       toolchain {
           languageVersion = JavaLanguageVersion.of(17)
       }
   }
   ```
5. **Restart terminal/IDE** after setting environment variables

**Tests fail in CI but pass locally:**
- Check JDK version matches (use `java -version` in both environments)
- Verify environment variables are set
- Check for timezone/time-dependent tests
- Ensure test isolation
- Verify JAVA_HOME is set correctly in CI

**Coverage not generating:**
- Verify JaCoCo plugin is applied
- Check test task runs before coverage report
- Ensure test classes are in correct package
- Verify Java version compatibility

**Coverage threshold failures:**
- Review coverage report to identify low coverage areas
- Adjust thresholds if too strict
- Add tests for uncovered code paths

**Slow test execution:**
- Enable parallel test execution
- Use test filtering for specific tests
- Optimize test setup/teardown
- Consider test containers optimization
- Check Java heap size: `-Xmx2048m`

**Mock not working:**
- Ensure `@ExtendWith(MockitoExtension.class)` is present
- Check that `@Mock` or `@MockBean` is used correctly
- Verify `@InjectMocks` is used for the class under test
- For Spring Boot tests, use `@MockBean` instead of `@Mock`

**Gradle build fails with Java errors:**
- Verify Java version: `./gradlew --version`
- Check `build.gradle` for Java version requirements
- Ensure JAVA_HOME points to JDK (not JRE)
- Try: `./gradlew clean build --refresh-dependencies`

## Example Prompt Template

When a user requests test generation, use this structure:

```
Generate unit tests for [ClassName] located at [path/to/Class.java].

Requirements:
1. Test all public methods
2. Cover happy path, edge cases, and error scenarios
3. Use JUnit 5 and Mockito
4. Prefer AssertJ over JUnit assertions for better readability
5. Follow SOLID and DRY principles
6. Follow AAA pattern with blank lines between sections
7. Include comprehensive documentation
8. Use @Nested classes for organization
9. Use @DisplayName for readable test names (both classes and methods)
10. Place tests in src/test/java/[package]/[ClassName]Test.java
11. Aim for 80% code coverage (industry standard)

The class has the following dependencies: [list dependencies]
The class handles: [brief description of functionality]
```

## References & Best Practices Sources

- **Unit Testing Best Practices**: AAA pattern, independence, determinism
  - Baeldung: Java Unit Testing Best Practices
  - Industry standard: 80% code coverage for production code
- **SOLID Principles in Testing**: Apply SOLID to test code structure
- **DRY Principle in Testing**: Use test fixtures, builders, parameterized tests
- **Spring Boot Testing**: Official Spring Boot testing documentation
  - Use `@ExtendWith(MockitoExtension.class)` for unit tests (no Spring context)
  - Use `@SpringBootTest` only for integration tests
- **JUnit 5 User Guide**: Official JUnit 5 documentation
  - `@Nested` classes for hierarchical test organization
  - `@DisplayName` for human-readable test names
  - `@BeforeAll`/`@AfterAll` limitations in nested classes
- **Mockito Documentation**: Official Mockito documentation
- **AssertJ**: Fluent assertions library (preferred over JUnit assertions)
  - Better readability and more expressive error messages

## Skill Metadata

- **Skill Name**: `java-testing`
- **Category**: Testing / Quality Assurance
- **Triggers**: 
  - `generate-tests`
  - `write test cases`
  - `unit-test`
  - `integration-test`
  - `test coverage`
  - `add tests`
- **Input Format**: Java source file(s) + optional specification/requirements
- **Output Format**: Test class files in `src/test/java/...` with documentation
- **Frameworks**: JUnit 5, Mockito, Spring Boot Test, AssertJ (strongly recommended)

---

## Usage Instructions for AI Agents

When you receive a request to generate test cases:

1. **Read and analyze** the provided Java code
2. **Identify** all public methods, dependencies, and behaviors
3. **Check existing test patterns** in the codebase for consistency
4. **Generate** test classes following the structure and principles above
5. **Include** comprehensive documentation explaining what each test does
6. **Ensure** tests follow SOLID, DRY, and AAA principles
7. **Cover** happy path, edge cases, error cases, and boundary conditions
8. **Place** tests in the correct directory structure (`src/test/java/...`)
9. **Use** `@Nested` classes for logical grouping
10. **Use** `@DisplayName` for human-readable test names
11. **Prefer AssertJ** over JUnit assertions for better readability
12. **Use blank lines** between AAA sections for clarity
13. **Provide** a summary of what scenarios are covered and why

Remember: Quality over quantity. Better to have fewer, well-written, comprehensive tests than many poorly structured ones. Follow industry standards: 80% code coverage is the accepted benchmark for production code.
