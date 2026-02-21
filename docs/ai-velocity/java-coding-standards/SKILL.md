---
name: java-coding-standards
description: "Use when bootstrapping new Java/Spring Boot services, implementing functionality in existing Java repositories, or reviewing Java code. Enforces production-grade Java standards including JDBI over Spring JPA for data access. Includes SOLID, DRY, OpenAPI, and patterns from Spring Boot production projects."
---

# Java Coding Standards and Best Practices

## Purpose
This skill enables AI agents to write production-grade Java code following SOLID, DRY, and clean code principles. Standards align with Spring Boot conventions, industry best practices, and complement [engineering-foundations](organizational-skills/engineering-foundations/) for architecture and design.

**Key Patterns Enforced:**
- JDBI over Spring JPA for new data access (Dao + Repository)
- Constructor injection (no field injection for required dependencies)
- Domain-specific exceptions with proper cause chaining
- Try-with-resources for all Closeable/AutoCloseable
- Bounded concurrency with ExecutorService
- Immutable DTOs and records where appropriate

## When to Trigger This Skill
- When bootstrapping a new Java/Spring Boot service
- When implementing new functionality in existing Java repositories
- When user requests: "Create a new Java service", "Add this feature", "Implement this in Java"
- When creating REST APIs (also create OpenAPI spec)
- During code reviews for Java code quality
- When refactoring or improving existing Java code

## Critical: Automatic Codebase Analysis

**ALWAYS analyze the codebase first before asking questions or making assumptions.**

1. **Read the codebase automatically**: Understand package structure, existing patterns, interfaces, dependencies, and conventions.
2. **Do NOT ask the user to describe their architecture**: The codebase contains the necessary information.
3. **Only ask clarifying questions if**: Code is unclear, multiple valid approaches exist, or critical requirements are missing.

---

## 1. Package and Module Structure

### Standard Layout (Maven/Gradle)

```
src/main/java/
  com.example.service/
    application/          # Use cases, orchestration
    domain/               # Business logic, entities
    infrastructure/      # Repositories, external clients
    api/                  # Controllers, DTOs
    config/               # Configuration classes
```

### Package Principles
- **Single responsibility per package**: Each package has one clear purpose.
- **Avoid generic names**: `utils`, `helpers`, `common` — use domain-specific names.
- **Domain-driven**: Organize by domain/feature, not by technical layer only.

### Visibility
- Prefer package-private for implementation details.
- Expose interfaces; hide implementations behind them.

---

## 2. Interface Design

### Small, Focused Interfaces
- Prefer many small interfaces over few large ones.
- Follow Interface Segregation Principle.
- Use "accept interfaces, return concretions" where practical.

```java
// ✅ GOOD: Small, focused interfaces
public interface OrderReader {
    Optional<Order> findById(String id);
}

public interface OrderWriter {
    Order save(Order order);
}

public interface OrderRepository extends OrderReader, OrderWriter {}

// ❌ BAD: Large interface with unrelated methods
public interface OrderRepository {
    Optional<Order> findById(String id);
    Order save(Order order);
    void sendEmail(String to, String subject);  // Wrong abstraction
}
```

---

## 3. Dependency Injection

### Constructor Injection (Required)
- Use constructor injection for all required dependencies.
- Avoid `@Autowired` on fields for required dependencies.
- Use `@RequiredArgsConstructor` (Lombok) when appropriate.

```java
// ✅ GOOD: Constructor injection
@Service
@RequiredArgsConstructor
public class OrderService {
    private final OrderRepository orderRepository;
    private final PaymentClient paymentClient;

    public Order processOrder(OrderRequest request) {
        // ...
    }
}

// ❌ BAD: Field injection for required dependencies
@Service
public class OrderService {
    @Autowired
    private OrderRepository orderRepository;  // Hard to test, hidden dependency
}
```

### Avoid Static Utilities for Testability
- Inject dependencies instead of calling static methods.
- Use `@Configuration` + `@Bean` for wiring; avoid `@Component` with static state.

---

## 4. Exception Handling

### Domain-Specific Exceptions
- Create domain exceptions: `OrderNotFoundException`, `PaymentFailedException`.
- Extend `RuntimeException` for unrecoverable; use checked exceptions sparingly (e.g., for retryable I/O).
- Always chain cause: `throw new OrderNotFoundException("Order " + id, e)`.

```java
// ✅ GOOD: Domain-specific exception with cause
public class OrderNotFoundException extends RuntimeException {
    public OrderNotFoundException(String message) {
        super(message);
    }
    public OrderNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}

// Usage
throw new OrderNotFoundException("Order not found: " + orderId, e);
```

### Never Swallow Exceptions
- Never catch and ignore without logging or rethrowing.
- If intentionally ignoring (e.g., best-effort cleanup), log at appropriate level and document.

```java
// ✅ GOOD: Log and rethrow or document
try {
    cache.invalidate(key);
} catch (Exception e) {
    log.warn("Cache invalidation failed (best effort): {}", key, e);
}
```

### Global Exception Handling
- Use `@ControllerAdvice` / `@ExceptionHandler` for REST APIs.
- Map domain exceptions to HTTP status codes consistently.
- Return structured error responses (code, message, details).

---

## 5. Resource Management

### Try-With-Resources
- Always use try-with-resources for `Closeable`, `AutoCloseable`.
- Ensure streams, connections, and file handles are closed.

```java
// ✅ GOOD: Try-with-resources
try (InputStream in = Files.newInputStream(path);
     BufferedReader reader = new BufferedReader(new InputStreamReader(in))) {
    return reader.lines().collect(Collectors.toList());
}

// ❌ BAD: Manual close (error-prone)
InputStream in = Files.newInputStream(path);
// ... use in ...
in.close();  // Never reached if exception thrown
```

---

## 6. Concurrency

### Prefer CompletableFuture for Async
- Use `CompletableFuture` for async operations with clear lifecycle.
- Use `ExecutorService` with bounded thread pools; avoid `Executors.newCachedThreadPool()` in production.

```java
// ✅ GOOD: Bounded executor
private final ExecutorService executor = Executors.newFixedThreadPool(
    Runtime.getRuntime().availableProcessors(),
    new ThreadFactoryBuilder().setNameFormat("order-processor-%d").build()
);

CompletableFuture.supplyAsync(() -> processOrder(order), executor)
    .orTimeout(5, TimeUnit.SECONDS)
    .exceptionally(ex -> handleFailure(ex));
```

### Thread Safety
- Use `synchronized` or `ReentrantLock` for shared mutable state.
- Prefer immutable objects (`record`, `@Value` Lombok) when possible.
- Use `ConcurrentHashMap` instead of `synchronized` + `HashMap` for concurrent maps.

---

## 7. Naming and Code Organization

### Naming Conventions
- Classes: PascalCase; methods/variables: camelCase.
- Interfaces: no `I` prefix (e.g., `OrderRepository` not `IOrderRepository`).
- Constants: `UPPER_SNAKE_CASE`.
- Booleans: `isEnabled`, `hasPermission`, `canProcess`.

### Method Design
- Keep methods short (< 50 lines); extract when logic grows.
- One level of abstraction per method.
- Use meaningful names; avoid comments to explain "what" (code should show it).

---

## 8. Immutability and Records

### Prefer Immutable DTOs
- Use Java `record` for DTOs, config, value objects (Java 16+).
- Use Lombok `@Value` for immutable classes when records don't fit.

```java
// ✅ GOOD: Immutable DTO
public record OrderRequest(String orderId, BigDecimal amount, String currency) {}

// ✅ GOOD: Builder for complex objects
@Value
@Builder
public class OrderDetails {
    String orderId;
    OrderStatus status;
    List<OrderItem> items;
}
```

---

## 9. Data Access: Prefer JDBI over Spring JPA

### Rule
- **Do not use Spring Data JPA in new projects.** Prefer JDBI for relational data access.
- **Existing projects**: Respect current stack; migrate to JDBI when adding new domains or during planned refactors.

### Why JDBI
- **Explicit SQL**: What runs is visible; no hidden queries or N+1 surprises.
- **No ORM magic**: No lazy loading, proxy complexity, or unexpected cascades.
- **Simpler testing**: DAOs are interfaces; mock or use in-memory DB without EntityManager.
- **Performance**: Direct SQL, no ORM overhead; better for read-heavy and batch workloads.
- **SQL-first**: Suits services that own their schema and optimize queries.

### Pattern: Dao + Repository

**Dao interface** — SQL operations with JDBI SqlObject (`@SqlQuery`, `@SqlUpdate`, `@BindBean`, `@BindList`):

```java
public interface OrderDao {
    @SqlUpdate("INSERT INTO orders (external_id, amount, currency) VALUES (:externalId, :amount, :currency)")
    @GetGeneratedKeys
    long save(@BindBean Order order);

    @SqlQuery("SELECT * FROM orders WHERE id = :id")
    @RegisterBeanMapper(Order.class)
    Optional<Order> findById(@Bind long id);

    @SqlQuery("SELECT * FROM orders WHERE id IN (<ids>)")
    @RegisterBeanMapper(Order.class)
    List<Order> findByIds(@BindList Set<Long> ids);
}
```

**Repository** — Injects `Jdbi`; uses `withExtension` or `inTransaction`:

```java
@Repository
@RequiredArgsConstructor
public class OrderRepository {
    private final Jdbi jdbi;

    public Order save(Order order) {
        long id = jdbi.withExtension(OrderDao.class, dao -> dao.save(order));
        order.setId(id);
        return order;
    }

    public Optional<Order> findById(long id) {
        return jdbi.withExtension(OrderDao.class, dao -> dao.findById(id));
    }

    public Order saveWithItems(Order order, List<OrderItem> items) {
        return jdbi.inTransaction(handle -> {
            OrderDao orderDao = handle.attach(OrderDao.class);
            OrderItemDao itemDao = handle.attach(OrderItemDao.class);
            long orderId = orderDao.save(order);
            order.setId(orderId);
            items.forEach(item -> item.setOrderId(orderId));
            itemDao.saveAll(items);
            return order;
        });
    }
}
```

### Dependencies
- `jdbi3-core`, `jdbi3-sqlobject`, `jdbi3-jackson2` (for JSON columns)
- HikariCP for connection pooling (via `Jdbi.create(dataSource)`)

### Configuration
- Create `Jdbi` bean from `DataSource`; install `SqlObjectPlugin`, `Jackson2Plugin`.
- Use `@Qualifier` for read-only vs write Jdbi when using replicas.
- Register custom `ColumnMapper` / `Argument` for JSON, UUID, enums.

See [reference.md](reference.md#jdbi-setup) for detailed setup.

---

## 10. Testing (See java-testing)

- Dependencies injected via interfaces — easy to mock.
- Use `@ExtendWith(MockitoExtension.class)` for unit tests; avoid `@SpringBootTest` unless integration.
- Cover happy path, edge cases, error paths.
- See [java-testing](java-testing/) for full testing standards.

---

## 11. REST APIs and OpenAPI

### OpenAPI Specification
- Create OpenAPI 3.0+ for all REST APIs.
- Place in `src/main/resources/openapi.yaml` or `api/` directory.
- Include schemas, examples, error responses.

### API Documentation and Tooling

**Use Springdoc OpenAPI** to auto-generate OpenAPI from Spring annotations. Enables:
- **Swagger UI** — Interactive docs at `/swagger-ui.html`; try endpoints in browser
- **OpenAPI JSON** — Served at `/v3/api-docs` for Postman, client generation
- **Postman import** — Import from `http://localhost:8080/v3/api-docs` or from static `openapi.yaml`
- **Test execution** — Postman collections, Newman (CLI), or Swagger UI for manual testing

```gradle
implementation 'org.springdoc:springdoc-openapi-starter-webmvc-ui:2.3.0'
```

```java
// Optional: customise in configuration (io.swagger.v3.oas.models)
@Bean
public OpenAPI customOpenAPI() {
    return new OpenAPI()
        .info(new Info().title("Order Service API").version("1.0.0"));
}
```

**Annotate controllers** for better docs: `@Operation(summary = "...")`, `@ApiResponse` for error codes.

### Design-First Alternative
- When API contract is critical: write `openapi.yaml` first; use openapi-generator for DTOs.
- Postman imports the spec file directly; keeps docs as source of truth.

### Controller Conventions
- Use `@RestController` + `@RequestMapping` with version prefix (e.g., `/api/v1/orders`).
- Validate input with `@Valid` and Bean Validation.
- Return DTOs; never expose domain entities directly.

```java
// ✅ GOOD: Validated, versioned, DTO response
@RestController
@RequestMapping("/api/v1/orders")
@RequiredArgsConstructor
public class OrderController {
    private final OrderService orderService;

    @PostMapping
    public ResponseEntity<OrderResponse> createOrder(@Valid @RequestBody CreateOrderRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
            .body(orderService.createOrder(request));
    }
}
```

---

## 12. Avoid Overengineering

### YAGNI and KISS
- Don't add abstractions "just in case."
- Start simple; add complexity when there's a concrete need.
- Use standard library and Spring features before introducing new dependencies.

### When to Add Complexity
- Multiple implementations of the same interface.
- Proven need for extensibility.
- Clear reduction in maintenance burden.

---

## Quick Reference Checklist

- [ ] Prefer JDBI over Spring JPA for new data access
- [ ] Constructor injection for dependencies
- [ ] Domain-specific exceptions with cause chaining
- [ ] Try-with-resources for all Closeable
- [ ] Bounded ExecutorService for async
- [ ] Immutable DTOs (record / @Value)
- [ ] Interfaces small and focused
- [ ] No generic package names
- [ ] OpenAPI spec for REST APIs (Springdoc; Swagger UI + Postman import)
- [ ] @Valid on request bodies
- [ ] @ControllerAdvice for exception mapping

---

## Additional Resources

- For testing: See [java-testing](java-testing/)
- For architecture: See [engineering-foundations](organizational-skills/engineering-foundations/)
- For Java patterns and Spring specifics: See [reference.md](reference.md)
