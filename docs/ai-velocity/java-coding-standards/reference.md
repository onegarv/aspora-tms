# Java Coding Standards — Reference

Detailed reference for the java-coding-standards skill.

## Spring Boot Specific Patterns

### Configuration
- Use `@ConfigurationProperties` for grouped config; avoid `@Value` for complex structures.
- Externalize config via `application.yml` / `application-{profile}.yml`.
- Use `spring.profiles.active` for environment-specific behavior.

### Lombok Usage
- `@RequiredArgsConstructor` for constructor injection.
- `@Value` for immutable DTOs; `@Data` only for internal objects (avoid in APIs).
- `@Slf4j` for logging; avoid manual logger creation.

### Transaction Management
- Use `@Transactional` at service layer, not repository.
- Specify `readOnly = true` for read-only operations.
- Avoid long-running transactions; keep units of work small.

## JDBI Setup

### Gradle Dependencies
```gradle
implementation 'org.jdbi:jdbi3-core:3.41.0'
implementation 'org.jdbi:jdbi3-sqlobject:3.41.0'
implementation 'org.jdbi:jdbi3-jackson2:3.41.0'
implementation 'org.jdbi:jdbi3-stringtemplate4:3.41.0'  // Optional, for SQL templates
```

### Configuration (Spring)
```java
@Bean
public Jdbi provideJdbi(DataSource dataSource, ObjectMapper objectMapper) {
    Jdbi jdbi = Jdbi.create(dataSource);
    jdbi.installPlugin(new SqlObjectPlugin());
    jdbi.installPlugin(new Jackson2Plugin());
    jdbi.getConfig(Jackson2Config.class).setMapper(objectMapper);
    jdbi.setSqlLogger(new Slf4jSqlLogger());  // Log SQL for debugging
    return jdbi;
}
```

### Read/Write Split (Replicas)
```java
@Bean("ro-jdbi")
public Jdbi provideReadOnlyJdbi(DataSource readOnlyDataSource) { ... }

@Bean
@Primary
public Jdbi provideJdbi(DataSource writeDataSource) { ... }
```

Inject with `@Qualifier("ro-jdbi")` for read-only repositories.

### Custom Types (JSON, UUID, Enums)
- Implement `ColumnMapper` and `Argument` (or extend `ColumnSerde`) for custom types.
- Register: `jdbi.registerColumnMapper(serde); jdbi.registerArgument(serde);`

### Reference
- Pattern example: `users-service` (Vance) — Dao interfaces, Repository with `jdbi.withExtension` / `inTransaction`, HikariCP, read replicas.

## Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| God class | Split by responsibility; extract services |
| Anemic domain | Move logic into domain objects |
| Primitive obsession | Use value objects, records |
| Service locator | Use constructor injection |
| Null returns | Use `Optional`; avoid `Optional.get()` without `isPresent()` check |

## Optional Best Practices

```java
// ✅ GOOD: Safe Optional handling
return repository.findById(id)
    .orElseThrow(() -> new OrderNotFoundException("Order not found: " + id));

// ✅ GOOD: Default value
return optional.orElse(defaultValue);

// ❌ BAD: Unsafe get
return optional.get();  // Throws if empty
```

## Checked vs Unchecked Exceptions

- **Unchecked (RuntimeException)**: Business rule violations, not found, validation — caller typically can't recover by retrying.
- **Checked**: I/O, network — when caller might retry or handle differently.

## Named Thread Factories

For named threads (easier debugging in logs and thread dumps):

```java
// With Guava (if available)
ThreadFactory factory = new ThreadFactoryBuilder()
    .setNameFormat("order-worker-%d")
    .build();

// Plain Java alternative
ThreadFactory factory = r -> {
    Thread t = new Thread(r, "order-worker-" + System.currentTimeMillis());
    t.setDaemon(false);
    return t;
};
ExecutorService executor = Executors.newFixedThreadPool(10, factory);
```

## OpenAPI / Swagger Setup

### Springdoc Dependencies
```gradle
implementation 'org.springdoc:springdoc-openapi-starter-webmvc-ui:2.3.0'
```

### Postman Import
- **From running app**: Import → Link → `http://localhost:8080/v3/api-docs`
- **From file**: Export spec to `openapi.yaml`; Import → File → select YAML

### Swagger UI
- Default: `http://localhost:8080/swagger-ui.html`
- Config: `springdoc.swagger-ui.path=/swagger-ui.html` (or custom path)
- Disable in prod if needed: `springdoc.api-docs.enabled=false`

### Controller Annotations
```java
@Operation(summary = "Create order", description = "Creates a new order from request")
@ApiResponses({
    @ApiResponse(responseCode = "201", description = "Order created"),
    @ApiResponse(responseCode = "400", description = "Validation error")
})
@PostMapping
public ResponseEntity<OrderResponse> createOrder(@Valid @RequestBody CreateOrderRequest request) { ... }
```

## Bean Validation

Common annotations for request DTOs:
- `@NotNull`, `@NotBlank`, `@NotEmpty`
- `@Size`, `@Min`, `@Max`
- `@Email`, `@Pattern`
- `@Valid` for nested objects

## Reference Repositories

- **Spring Boot**: [spring-projects/spring-boot](https://github.com/spring-projects/spring-boot) — official samples
- **Spring Petclinic**: Domain-driven Spring application
- **Baeldung tutorials**: Common patterns and conventions
