---
name: java-application-development
description: Build production-grade Java applications with rich domain models, layered architecture, SOLID, DRY, and Java best practices. Covers domain-driven design, Spring Boot, validation, and clean separation of concerns.
version: 1.0.0
author: AI Velocity Team
tags: [java, spring-boot, domain-driven-design, solid, dry, domain-models, clean-architecture, ddd, best-practices]
---

# Java Application Development — Domain Models, SOLID, DRY

## Purpose

This skill enables AI agents to design and implement production-grade Java applications using **rich domain models**, **SOLID** and **DRY** principles, and patterns that are central to the Java ecosystem. Code is layered (domain, application, infrastructure, API), testable, and maintainable.

**Emphasis:** Domain models are first-class; avoid anemic domains. Use value objects, entities, and aggregates where they add clarity. Apply SOLID and DRY consistently across packages and classes.

## When to Trigger This Skill

- When designing or implementing a new Java service or module
- When user requests: "Design the domain model for...", "Add a new feature following best practices", "Refactor this to use domain models"
- When reviewing or refactoring Java code for structure and maintainability
- When introducing or aligning with layered/hexagonal architecture
- When implementing business logic that should live in the domain, not in controllers or repositories

## Critical: Automatic Codebase Analysis

**ALWAYS analyze the codebase first before asking questions or making assumptions.**

1. **Read the codebase automatically:**
   - Identify existing package structure (domain, application, infrastructure, api)
   - Find domain entities, value objects, and aggregates
   - Understand current layering and dependencies (who calls whom)
   - Detect anemic vs rich domain usage
   - Note validation and error-handling patterns

2. **Do NOT ask the user to describe the domain:**
   - Infer domain concepts from class names, methods, and existing behavior
   - Propose domain models that fit existing or stated requirements
   - Align with existing conventions (e.g. `*Request`/`*Response` DTOs, exception hierarchy)

3. **Only ask clarifying questions if:**
   - Business rules are ambiguous or missing
   - Multiple valid domain boundaries exist
   - Critical constraints (e.g. consistency, performance) are not evident from code

---

## Core Principles

### 1. Domain Models First (DRY, Single Source of Truth)

**Prefer rich domain models over anemic DTOs for core business logic.**

- **Entities:** Identity and lifecycle; use when the same logical object is tracked over time (e.g. `Order`, `Customer`).
- **Value objects:** Defined by attributes; immutable; no identity (e.g. `Money`, `OrderId`, `Email`, `Address`). Use Java `record` where appropriate.
- **Aggregates:** Cluster of entities and value objects with a single root; consistency boundary. Keep aggregates small.

**✅ GOOD: Value objects and rich domain**

```java
// Value object — immutable, no identity
public record OrderId(UUID value) {
    public OrderId {
        if (value == null) throw new IllegalArgumentException("OrderId cannot be null");
    }
}

public record Money(java.math.BigDecimal amount, Currency currency) {
    public Money {
        if (amount == null || amount.compareTo(BigDecimal.ZERO) < 0)
            throw new IllegalArgumentException("Invalid amount");
        if (currency == null) throw new IllegalArgumentException("Currency required");
    }

    public Money add(Money other) {
        requireSameCurrency(other);
        return new Money(amount.add(other.amount), currency);
    }
}

// Entity — identity, behavior
public class Order {
    private final OrderId id;
    private OrderStatus status;
    private final List<OrderLine> lines;

    public void addLine(OrderLine line) {
        if (status != OrderStatus.DRAFT)
            throw new IllegalStateException("Cannot modify order in " + status);
        lines.add(line);
    }

    public Money total() {
        return lines.stream()
            .map(OrderLine::subtotal)
            .reduce(Money.zero(currency()), Money::add);
    }
}
```

**❌ BAD: Anemic domain (logic in services only)**

```java
// Only getters/setters; no behavior
public class Order {
    private UUID id;
    private String status;
    private List<OrderLineDto> lines;
    // ... getters/setters only
}
// All logic in OrderService — domain is a data bag
```

**DRY with domain types:** Use domain types (e.g. `OrderId`, `Money`) everywhere in the domain and application layer so validation and behavior live in one place. Avoid duplicating validation in DTOs and entities.

### 2. SOLID in Java

**Single Responsibility (SRP)**  
- One reason to change per class. Domain entities handle domain rules; application services orchestrate use cases; repositories handle persistence.  
- Split large services into smaller ones (e.g. `OrderCreationService`, `OrderQueryService`).

**Open/Closed (OCP)**  
- Extend via new implementations (e.g. new `PaymentGateway` impl) or strategy/enum, not by modifying existing core logic.  
- Use interfaces for extension points (e.g. `EventHandler`, `Validator`).

**Liskov Substitution (LSP)**  
- Implementations of interfaces must be substitutable. Don’t throw unexpected exceptions or change semantics in subtypes.

**Interface Segregation (ISP)**  
- Small, focused interfaces. Prefer `OrderRepository.findById(OrderId)` + `OrderRepository.save(Order)` over a single “god” repository interface if clients only need read or write.  
- In tests, mock the narrow interface the class under test needs.

**Dependency Inversion (DIP)**  
- Depend on abstractions (interfaces), not concretions. Domain and application layers define ports (interfaces); infrastructure implements adapters.  
- Use constructor injection; framework (Spring) injects implementations.

```java
// ✅ GOOD: Application depends on port (interface); infrastructure implements it
// application layer
public interface OrderRepository {
    Optional<Order> findBy(OrderId id);
    void save(Order order);
}

public class CreateOrderUseCase {
    private final OrderRepository orderRepository;
    private final EventPublisher eventPublisher;

    public CreateOrderUseCase(OrderRepository orderRepository, EventPublisher eventPublisher) {
        this.orderRepository = orderRepository;
        this.eventPublisher = eventPublisher;
    }
}

// infrastructure layer
@Repository
public class JpaOrderRepository implements OrderRepository { ... }
```

### 3. DRY — Don’t Repeat Yourself

- **Domain types:** One definition of `OrderId`, `Money`, etc.; reuse in domain, application, and (where appropriate) API.
- **Validation:** Centralize in value-object constructors, Bean Validation on DTOs, or small validator classes; avoid copy-paste checks.
- **Exception hierarchy:** Domain-specific exceptions (e.g. `OrderNotFound`, `InvalidOrderState`) in one place; map to HTTP/API in one adapter.
- **Builders:** Use builders for complex entities or test data to avoid repeated construction code (see Java Testing skill).

### 4. Layered / Package Structure

Keep dependencies pointing inward: **API → Application → Domain**; **Infrastructure** implements interfaces defined in Application or Domain.

**Recommended package layout (align with existing codebase when present):**

```
com.example.order/
  domain/                    # Entities, value objects, domain services, domain exceptions
    Order.java
    OrderId.java
    OrderStatus.java
    Money.java
    OrderRepository.java     # Port (interface) — only if domain defines it
  application/               # Use cases, application services, ports
    CreateOrderUseCase.java
    OrderRepository.java     # Port (interface) — often here
    OrderQueryService.java
  infrastructure/            # Adapters: persistence, messaging, external clients
    persistence/
      JpaOrderRepository.java
      OrderEntity.java       # JPA entity if different from domain model
    messaging/
      OrderEventPublisher.java
  api/                       # Controllers, DTOs, exception mappers
    OrderController.java
    OrderRequest.java
    OrderResponse.java
```

- **Domain:** No dependencies on Spring, JPA, or HTTP. Pure Java.  
- **Application:** Orchestrates domain and ports; no framework-specific types in method signatures where avoidable.  
- **Infrastructure:** Implements `OrderRepository`, `EventPublisher`, etc.  
- **API:** Maps HTTP to application (DTOs → use case input); maps exceptions to status codes.

### 5. Java-Specific Practices

**Immutability**  
- Prefer immutable value objects (`record`, final fields, no setters).  
- For entities, keep identity and invariants clear; mutate only through explicit methods.

**Optional**  
- Use `Optional<T>` for return types that may be absent; avoid null for “no result” in public API.  
- Don’t use `Optional` as field type in entities/DTOs; use null or a dedicated “empty” value object if needed.

**Records**  
- Use for value objects and DTOs (Java 16+): `record OrderId(UUID value)`, `record CreateOrderRequest(...)`.

**Bean Validation**  
- Validate at API boundary: `@Valid` on request body; constraints on DTOs.  
- Keep domain validation in domain types (constructors, methods); avoid duplicating the same rule in both DTO and entity.

**Spring Boot**  
- Prefer **constructor injection** for required dependencies.  
- Expose **interfaces** for application/domain ports; inject implementation in infrastructure.  
- Use `@Transactional` at application/service boundary, not on domain types.

**Exceptions**  
- Domain: use domain exceptions (e.g. `OrderNotFound`, `InvalidOrderState`).  
- Wrap infrastructure exceptions in application/domain exceptions at adapter boundary; map to HTTP in one place (e.g. `@ControllerAdvice`).

---

## Domain Layer — Details

- **Entities:** Identity (e.g. `OrderId`); protect invariants in methods; avoid public setters that bypass rules.  
- **Value objects:** Immutable; implement `equals`/`hashCode` (or use `record`); validate in constructor.  
- **Domain services:** Use when an operation doesn’t naturally belong to one entity (e.g. `TransferService` between two accounts).  
- **Repositories (port):** Define interface in domain or application (e.g. `OrderRepository`); return domain types, accept domain types.  
- **Domain events (optional):** If used, define in domain; publish from entity or application service.

---

## Application Layer — Use Cases

- One use case per class or clear group of methods (e.g. `CreateOrderUseCase`, `CancelOrderUseCase`).  
- Input: domain types or simple DTOs; output: domain types or DTOs.  
- Orchestrate: load aggregate via repository, call domain methods, save, publish events.  
- No business rules here that belong in the domain; application = coordination.

---

## API Layer — Controllers and DTOs

- Controllers thin: parse request, call use case or application service, map result to response.  
- Use **DTOs** for request/response; map to/from domain at boundary.  
- Validate with Bean Validation; map domain exceptions to HTTP (e.g. 404, 409, 422).  
- Don’t expose domain entities directly in API; avoid coupling and over-fetching.

---

## Error Handling and Validation

- **Domain:** Throw domain exceptions (e.g. `IllegalStateException`, or custom `OrderNotFound`).  
- **Application:** Let domain exceptions propagate or translate once at boundary.  
- **API:** Single place (e.g. `@ControllerAdvice`) to map `OrderNotFound` → 404, validation → 400/422.  
- **DRY:** One definition per error type; one mapping per exception to HTTP.

---

## Summary Checklist

- [ ] Domain model is rich (value objects, entities with behavior), not anemic.  
- [ ] SOLID: single responsibility, interfaces for ports, constructor injection, dependency inversion.  
- [ ] DRY: shared domain types, centralized validation and exception mapping.  
- [ ] Layered packages: domain ← application ← infrastructure; API depends on application.  
- [ ] Java idioms: immutability, `record`, `Optional`, Bean Validation at boundary.  
- [ ] Spring: constructor injection, interfaces for services/repositories, `@Transactional` at service layer.

When implementing or refactoring, apply these principles so that all engineers get consistent, maintainable Java applications with clear domain models and boundaries.
