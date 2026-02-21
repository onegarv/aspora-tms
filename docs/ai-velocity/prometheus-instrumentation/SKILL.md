---
name: prometheus-instrumentation
description: "Instrument Java and Go services with production-grade Prometheus metrics following Google SRE principles, RED method, and observability best practices. Enables fast issue detection and reduces MTTR in mature organizations."
version: 2.0.0
author: AI Velocity Team
---

# Prometheus Metrics Instrumentation for Java and Go Services

## 🚨 STOP: Read This First

**Before adding ANY metrics instrumentation, you MUST read [`GUARDRAILS.md`](./GUARDRAILS.md).**

This guardrail file contains the **4 most common mistakes** from actual RCAs (changing exception types, adding validation, modifying method signatures, mid-method outcome recording). Violating the Zero Mutation Rule breaks production systems.

**The One-Line Rule:** If removing all metrics changes your program's behavior, your instrumentation is **broken**.

---

## Purpose
This skill enables AI agents to instrument Java and Go services with production-grade Prometheus metrics following Google SRE principles, RED method, and observability best practices. The goal is to create metrics that enable fast issue detection and reduce MTTR (Mean Time To Recovery) in mature organizations like Google, Facebook, and Netflix.

## When to Trigger This Skill
- When implementing observability for a new Java or Go service
- When reviewing or improving existing metrics instrumentation
- When user requests: "Add metrics for this service", "Instrument this code", "What metrics should I track?"
- During code reviews for observability gaps
- When debugging production issues and metrics are missing
- When designing new features that need observability

## ⚠️ CRITICAL: What Prometheus Is (and Isn't) For

**Before adding ANY metric, ask: "Does this answer 'Is my system healthy right now?'"**

### Prometheus IS For (Four Golden Signals)

| Signal | Purpose | Examples |
|--------|---------|----------|
| **Latency** | How long requests take | `http_request_duration_seconds`, `operation_duration_seconds` |
| **Traffic** | Request/event rate | `http_requests_total`, `webhook_ingress_total`, `queue_messages_processed` |
| **Errors** | Failure rate | `http_requests_total{status="5xx"}`, `operation_errors_total` |
| **Saturation** | Resource utilization | `queue_depth`, `connection_pool_used`, `memory_used_bytes` |

### Prometheus is NOT For

| Anti-Pattern | Why It's Wrong | Better Alternative |
|--------------|----------------|-------------------|
| **State transitions** (`order_status_transition{from, to}`) | Business analytics, not operations. High cardinality. | Database + CDC → Data Warehouse |
| **Business events** (`transfer_lifecycle{stage}`) | Business tracking, not health | Database + CDC → Data Warehouse |
| **Business logic outcomes** (mid-method `success=true/false`) | Already captured by API-level metrics | `@Timed` on method (exception = failure) |
| **Entity counts by status** (`orders_by_status{status}`) | Point-in-time business state | Database queries, BI dashboards |
| **Audit trails** (`user_action{user_id, action}`) | Compliance/audit concern | Audit logs, dedicated audit service |

**Key Insight: If state is persisted to database, you already have the data.**
- Database IS the source of truth for business state
- CDC (Change Data Capture) streams changes to data warehouse automatically
- Analytics team queries warehouse directly - zero application code needed
- Don't duplicate what CDC already captures

### Decision Framework: Should This Be a Prometheus Metric?

```
┌─────────────────────────────────────────────────────────────┐
│ Does this metric answer: "Is my system healthy RIGHT NOW?" │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
            YES                          NO
              │                           │
              ▼                           ▼
    ┌─────────────────┐         ┌─────────────────────┐
    │ Is it one of:   │         │ Use instead:        │
    │ • Latency       │         │ • Structured logs   │
    │ • Traffic       │         │ • Event streaming   │
    │ • Errors        │         │ • Analytics/BI      │
    │ • Saturation    │         │ • Database queries  │
    └─────────────────┘         └─────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────┐
    │ Can it be captured at method level? │
    └─────────────────────────────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
   YES                  NO
    │                   │
    ▼                   ▼
  Use @Timed      Keep inline metric
  annotation      (gauges, queue depth)
```

### Examples: Correct vs Incorrect

**❌ WRONG: Business State Transitions in Prometheus**
```java
// This is business analytics, not operational health
metrics.recordOrderStatusTransition(from, COMPLETED);
metrics.recordPaymentStatusTransition(from, AUTHORIZED);
metrics.recordFulfillmentStatusTransition(from, INITIATED);
```

**❌ ALSO WRONG: Redundant Structured Logs (if you have CDC)**
```java
// If state is persisted to DB, this log is redundant - CDC captures it
log.info("Order status transition | orderId={} | from={} | to={}", orderId, from, to);
```

**✅ RIGHT: Just Update Database (CDC handles the rest)**
```java
order.setOrderStatus(COMPLETED);
orderRepository.save(order);
// CDC (Debezium/Kafka Connect) streams change to data warehouse
// Analytics team queries warehouse - zero application code
```

**❌ WRONG: Mid-Method Business Logic Outcomes**
```java
public void processTransaction() {
    if (businessCondition1) {
        metrics.recordTransaction(true);   // Redundant
    } else if (businessCondition2) {
        metrics.recordTransaction(true);   // Redundant
    } else {
        metrics.recordTransaction(false);  // Redundant
    }
}
```

**✅ RIGHT: Method-Level Annotation**
```java
@Timed(value = "transaction.process", description = "Transaction processing")
public void processTransaction() {
    // Business logic only
    // @Timed automatically records: success (normal return), failure (exception)
}
```

**✅ RIGHT: Operational Metrics That Belong in Prometheus**
```java
// Traffic signal - webhook ingress rate
metrics.recordWebhookIngress(provider);

// Saturation signal - queue depth
metrics.recordQueueDepth(queueName, depth);

// Operational health - cron job status
metrics.recordCronRun(jobName, success);

// Rate limiting - security/traffic signal
metrics.recordRateLimiterCheck(type, allowed);
```

### Tier 1: Use @Timed Annotation (Preferred)
For any operation where success = normal return, failure = exception:
```java
@Timed(value = "appserver.payment.create", description = "Payment creation")
public Payment createPayment(Request req) {
    return paymentService.create(req);  // No manual metrics needed
}
```

### Tier 2: Use @MeteredOperation (Dynamic Labels)
For operations needing labels from method parameters:
```java
@MeteredOperation(
    value = "appserver.payment.create",
    labels = {"acquirer=#{#payment.acquirer}", "method=#{#payment.paymentMethod}"}
)
public Payment createPayment(Payment payment) {
    return paymentService.create(payment);
}
```

### Tier 3: Keep Inline (Only When Necessary)
Only for metrics that CANNOT be captured at method boundaries:
- **Gauges**: `recordQueueDepth()`, `recordActiveConnections()`
- **Traffic counters**: `recordWebhookIngress()` (counting events, not method calls)
- **Cron job tracking**: `recordCronRun()` (job success/failure)
- **Rate limiter metrics**: Security/traffic signals

## Critical: Automatic Codebase Analysis

**ALWAYS analyze the codebase first before asking questions or making assumptions.**

1. **Read the codebase automatically**:
   - Read service classes, controllers, and main components
   - Understand request/event flows from code
   - Identify business entities and operations from code analysis
   - Map dependencies and external calls from imports and method calls

2. **Do NOT ask the user to describe their service**:
   - The codebase contains all necessary information
   - Understand architecture from code structure
   - Identify operations from method signatures and implementations
   - Discover entities from class names, fields, and method parameters

3. **Only ask clarifying questions if**:
   - Code is unclear or ambiguous
   - Multiple interpretations are possible
   - Critical information is missing from codebase

**Example approach**:
```
User: "Add metrics to my service"

AI should:
1. Read the codebase (service classes, controllers, etc.)
2. Understand: "This is an order processing service with OrderService, PaymentService, InventoryService"
3. Identify: "Main flow is OrderService.processOrder() → PaymentService.processPayment() → InventoryService.reserveItems()"
4. Implement: Add metrics automatically based on code analysis
5. Explain: "I've added metrics based on your codebase structure..."
```

**NOT**:
```
AI: "What does your service do? What are the main operations? Describe the flow..."
```

## ⚠️ CRITICAL: Do Not Change Functional Logic

**ABSOLUTE REQUIREMENT: Metrics instrumentation must NEVER modify business logic, validation, error handling, or control flow.**

**📖 REQUIRED READING:** See [`GUARDRAILS.md`](./GUARDRAILS.md) for concrete examples of the 4 most common mistakes from actual RCAs. Read it BEFORE implementing any instrumentation.

### Rules for Instrumentation

1. **ONLY Add Metrics Calls**
   - Add `metrics.record*()` calls
   - Add `var timerSample = metrics.start()` and timer stop calls
   - Add try-catch blocks ONLY for metrics recording
   - Do NOT modify existing business logic, validation, error handling, or control flow

2. **Preserve All Existing Behavior**
   - Method signatures must remain identical
   - Return values must be identical
   - Exception types and messages must be identical
   - Side effects must be identical
   - All existing try-catch blocks must be preserved

3. **Wrap, Don't Replace**
   - Wrap existing code in try-catch for metrics
   - Preserve all existing exception handling
   - Do NOT change exception types
   - Do NOT add new validation or early returns
   - Do NOT modify existing control flow

4. **Verification Checklist**
   - ✅ All existing business logic preserved
   - ✅ All existing exception handling preserved
   - ✅ Method signatures unchanged
   - ✅ Return values unchanged
   - ✅ Only metrics calls added
   - ✅ No new validation or early returns
   - ✅ No changes to control flow
   - ✅ No changes to exception types

### Example: Correct vs Incorrect

**✅ CORRECT - No Logic Changes**:
```java
// BEFORE
public Order createOrder(CreateOrderRequest request) {
    Order order = doCreateOrder(request);
    return order;
}

// AFTER - Only metrics added
public Order createOrder(CreateOrderRequest request) {
    var timerSample = metrics.start();
    try {
        Order order = doCreateOrder(request); // UNCHANGED
        metrics.recordOrderCreate(orderType, timerSample, true); // ADDED
        return order; // UNCHANGED
    } catch (Exception e) {
        metrics.recordOrderCreate(orderType, timerSample, false); // ADDED
        throw e; // PRESERVED - same exception type
    }
}
```

**❌ INCORRECT - Logic Changed**:
```java
// ❌ WRONG: Changed exception handling
public Order createOrder(CreateOrderRequest request) {
    try {
        Order order = doCreateOrder(request);
        metrics.recordOrderCreate(orderType, timerSample, true);
        return order;
    } catch (IllegalArgumentException e) {
        // ❌ WRONG: Changed exception type
        throw new AppServerException(e);
    }
}

// ❌ WRONG: Added validation
public Order createOrder(CreateOrderRequest request) {
    if (request == null) {
        metrics.recordOrderCreate(orderType, timerSample, false);
        return null; // ❌ WRONG: Changed behavior
    }
    // ...
}

// ❌ WRONG: Changed method signature
public Order createOrder(CreateOrderRequest request, MeterRegistry registry) {
    // ❌ WRONG: Added parameter
}
```

---

## MANDATORY: Implementation Checklist

**Before any PR is considered complete, ALL of these must be present.**

### Config Files (REQUIRED - @Timed won't work without these!)

| File | Purpose | Required When |
|------|---------|---------------|
| `MetricsAopConfig.java` | Enables @Timed/@Counted AOP support | **ALWAYS** - @Timed annotations do NOTHING without this |
| `MeteredOperation.java` | Tier 2 annotation for dynamic labels | When using dynamic labels from method parameters |
| `MeteredOperationAspect.java` | Aspect that processes @MeteredOperation | When using @MeteredOperation annotation |
| `HikariMetricsConfig.java` | DB connection pool metrics (saturation signal) | When using HikariCP datasource |

### Four Golden Signals Coverage

- [ ] **Latency**: Timer/histogram metrics (`.duration` or `_seconds`) - measures how long operations take
- [ ] **Traffic**: Counter metrics (`.total`) - measures request/event rate
- [ ] **Errors**: Counter with `outcome=error` tag or error-specific counters
- [ ] **Saturation**: Queue depth, connection pool utilization, resource metrics (HikariCP, thread pools)

### What NOT to Instrument

- [ ] **NO** @Timed on REST controllers - `http.server.requests` already covers it
- [ ] **NO** @Timed on simple getters/setters - no business value
- [ ] **NO** @Timed on repository methods - `spring.data.repository` metrics cover it (if enabled)
- [ ] **NO** business state transitions as metrics - use database + CDC instead

---

## CRITICAL: @Timed Alone Does NOT Cover Four Golden Signals

**Many teams add @Timed annotations and assume they have full observability. This is wrong.**

### What @Timed Actually Provides

| Signal | @Timed Only | @MeteredOperation | Complete Setup |
|--------|-------------|-------------------|----------------|
| **Latency** | histogram + max + sum | histogram (`.duration`) | histogram |
| **Traffic** | count (via histogram) | dedicated counter (`.total`) | counter + histogram count |
| **Errors** | `exception` tag only | `outcome=error` + `exception` tag | explicit error tracking |
| **Saturation** | (nothing) | (nothing) | HikariMetricsConfig, thread pool metrics |

### Why @Timed Is Insufficient

1. **No Dedicated Counter**: @Timed gives you a count via the histogram, but no standalone counter for rate queries
2. **No Outcome Tracking**: You can't easily query success vs failure rate - only exception class
3. **No Saturation Metrics**: @Timed cannot measure connection pool usage, queue depth, or resource utilization
4. **No Traffic Signal Without Queries**: Need `increase(metric_count[5m])` rather than direct `metric_total`

### Required for Complete Four Golden Signals

```
Four Golden Signals Coverage:

1. LATENCY  -> @Timed or @MeteredOperation (both provide histograms)

2. TRAFFIC  -> @MeteredOperation provides .total counter
              OR add explicit Counter alongside @Timed

3. ERRORS   -> @MeteredOperation provides outcome=error label
              OR track exceptions separately

4. SATURATION -> HikariMetricsConfig for DB connections
                 Thread pool metrics for executor saturation
                 Custom gauges for queue depth
```

### The MetricsAopConfig Problem

**CRITICAL: @Timed annotations DO NOTHING without MetricsAopConfig!**

Spring Boot does NOT auto-configure `TimedAspect`. You MUST add this config class:

```java
@Configuration
public class MetricsAopConfig {
    @Bean
    public TimedAspect timedAspect(MeterRegistry registry) {
        return new TimedAspect(registry);
    }

    @Bean
    public CountedAspect countedAspect(MeterRegistry registry) {
        return new CountedAspect(registry);
    }
}
```

Without this, your @Timed annotations are **completely ignored** - no errors, no warnings, just silent failure.

---

## Required Config Files (Copy-Paste Templates)

### MetricsAopConfig.java (REQUIRED for @Timed/@Counted)

```java
package tech.vance.yourservice.config;

import io.micrometer.core.aop.CountedAspect;
import io.micrometer.core.aop.TimedAspect;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * REQUIRED: Enables Micrometer AOP annotations (@Timed, @Counted).
 *
 * Without this configuration, @Timed and @Counted annotations are SILENTLY IGNORED.
 * Spring Boot does NOT auto-configure these aspects.
 */
@Configuration
public class MetricsAopConfig {

    /**
     * Enables @Timed annotation support.
     * Methods annotated with @Timed will automatically record:
     * - Execution time (histogram)
     * - Call count (via histogram)
     * - Exception tracking (via 'exception' tag)
     */
    @Bean
    public TimedAspect timedAspect(MeterRegistry registry) {
        return new TimedAspect(registry);
    }

    /**
     * Enables @Counted annotation support.
     * Methods annotated with @Counted will automatically record call counts.
     */
    @Bean
    public CountedAspect countedAspect(MeterRegistry registry) {
        return new CountedAspect(registry);
    }
}
```

### HikariMetricsConfig.java (REQUIRED for Saturation Signal with HikariCP)

```java
package tech.vance.yourservice.config;

import com.zaxxer.hikari.HikariDataSource;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PostConstruct;

/**
 * Configures HikariCP metrics for saturation signal monitoring.
 *
 * Exposes metrics:
 * - hikaricp_connections_active: Currently active connections
 * - hikaricp_connections_idle: Idle connections in pool
 * - hikaricp_connections_pending: Threads waiting for connection
 * - hikaricp_connections_max: Maximum pool size
 * - hikaricp_connections_min: Minimum pool size
 * - hikaricp_connections_usage_seconds: Connection usage time
 * - hikaricp_connections_creation_seconds: Connection creation time
 * - hikaricp_connections_acquire_seconds: Connection acquire time
 */
@Configuration
@ConditionalOnClass(HikariDataSource.class)
public class HikariMetricsConfig {

    private final HikariDataSource dataSource;
    private final MeterRegistry meterRegistry;

    public HikariMetricsConfig(HikariDataSource dataSource, MeterRegistry meterRegistry) {
        this.dataSource = dataSource;
        this.meterRegistry = meterRegistry;
    }

    @PostConstruct
    public void bindMetrics() {
        // HikariCP metrics are auto-bound when micrometer is on classpath
        // and spring.datasource.hikari.register-mbeans=true
        // This explicit binding ensures metrics are registered
        dataSource.setMetricRegistry(meterRegistry);
    }
}
```

**application.yml addition for HikariCP metrics:**

```yaml
spring:
  datasource:
    hikari:
      register-mbeans: true
      pool-name: HikariPool-1
```

### MeteredOperation.java (For Tier 2 - Dynamic Labels)

```java
package tech.vance.yourservice.common.metrics;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a method for complete metrics instrumentation with dynamic labels.
 *
 * Provides ALL Four Golden Signals:
 * - Latency: Timer/histogram (.duration)
 * - Traffic: Counter (.total)
 * - Errors: outcome=error label + exception type
 *
 * Unlike @Timed, this allows dynamic labels extracted from method parameters
 * using SpEL expressions.
 *
 * Example:
 * <pre>
 * {@literal @}MeteredOperation(
 *     value = "payment.process",
 *     labels = {"acquirer=#{#request.acquirer}", "method=#{#request.paymentMethod}"}
 * )
 * public PaymentResult process(PaymentRequest request) { ... }
 * </pre>
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface MeteredOperation {

    /** Operation name for the metric (e.g., "order.create") */
    String value();

    /** Description for the metric */
    String description() default "";

    /**
     * SpEL expressions to extract label values from parameters.
     * Format: "labelName=#{expression}"
     * Examples:
     *   - "order_type=#{#request.orderType}"
     *   - "customer_tier=#{#request.customer.tier}"
     *   - "batch_size=#{#items.size()}"
     */
    String[] labels() default {};

    /** Whether to record outcome (success/error) as a label. Default: true */
    boolean recordOutcome() default true;
}
```

### MeteredOperationAspect.java (Processes @MeteredOperation)

```java
package tech.vance.yourservice.common.metrics;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.expression.MethodBasedEvaluationContext;
import org.springframework.core.DefaultParameterNameDiscoverer;
import org.springframework.core.ParameterNameDiscoverer;
import org.springframework.expression.EvaluationContext;
import org.springframework.expression.ExpressionParser;
import org.springframework.expression.spel.standard.SpelExpressionParser;
import org.springframework.stereotype.Component;

import java.lang.reflect.Method;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * AOP Aspect that handles @MeteredOperation annotations.
 *
 * Provides complete Four Golden Signals coverage:
 * - Latency: Timer records duration histogram
 * - Traffic: Counter tracks call rate (.total)
 * - Errors: Outcome label distinguishes success/failure
 *
 * (Saturation requires separate configuration - see HikariMetricsConfig)
 */
@Aspect
@Component
@RequiredArgsConstructor
@Slf4j
public class MeteredOperationAspect {

    private static final Pattern LABEL_PATTERN = Pattern.compile("([^=]+)=#\\{(.+)}");

    @Value("${spring.application.name:unknown-service}")
    private String serviceName;

    private final MeterRegistry meterRegistry;
    private final ExpressionParser parser = new SpelExpressionParser();
    private final ParameterNameDiscoverer parameterNameDiscoverer = new DefaultParameterNameDiscoverer();

    @Around("@annotation(meteredOperation)")
    public Object measureOperation(ProceedingJoinPoint joinPoint, MeteredOperation meteredOperation) throws Throwable {
        String operationName = sanitizeOperation(meteredOperation.value());
        Tags baseTags = buildTags(joinPoint, meteredOperation);

        Timer.Sample sample = Timer.start(meterRegistry);
        String outcome = "success";
        String exceptionType = null;

        try {
            return joinPoint.proceed();
        } catch (Throwable t) {
            outcome = "error";
            exceptionType = t.getClass().getSimpleName();
            throw t;
        } finally {
            // LATENCY SIGNAL: Record timer/histogram
            Tags timerTags = baseTags
                .and("service", serviceName)
                .and("operation", operationName);

            sample.stop(Timer.builder(serviceName + ".operations.duration")
                .description(meteredOperation.description())
                .tags(timerTags)
                .register(meterRegistry));

            // TRAFFIC + ERROR SIGNALS: Record counter with outcome
            if (meteredOperation.recordOutcome()) {
                Tags counterTags = baseTags
                    .and("service", serviceName)
                    .and("operation", operationName)
                    .and("outcome", outcome);

                if (exceptionType != null) {
                    counterTags = counterTags.and("exception", sanitizeLabel(exceptionType));
                }

                meterRegistry.counter(serviceName + ".operations.total", counterTags).increment();
            }
        }
    }

    private Tags buildTags(ProceedingJoinPoint joinPoint, MeteredOperation meteredOperation) {
        Tags tags = Tags.empty();
        if (meteredOperation.labels().length == 0) {
            return tags;
        }

        try {
            Method method = ((MethodSignature) joinPoint.getSignature()).getMethod();
            EvaluationContext context = new MethodBasedEvaluationContext(
                null, method, joinPoint.getArgs(), parameterNameDiscoverer);

            for (String labelDef : meteredOperation.labels()) {
                Matcher matcher = LABEL_PATTERN.matcher(labelDef);
                if (matcher.matches()) {
                    String labelName = matcher.group(1).trim();
                    String expression = matcher.group(2).trim();

                    try {
                        Object value = parser.parseExpression(expression).getValue(context);
                        String labelValue = value != null ? sanitizeLabel(value.toString()) : "unknown";
                        tags = tags.and(labelName, labelValue);
                    } catch (Exception e) {
                        log.warn("Failed to evaluate SpEL '{}' for label '{}': {}", expression, labelName, e.getMessage());
                        tags = tags.and(labelName, "unknown");
                    }
                }
            }
        } catch (Exception e) {
            log.warn("Failed to build tags for @MeteredOperation: {}", e.getMessage());
        }
        return tags;
    }

    private String sanitizeOperation(String operation) {
        if (operation == null) return "unknown";
        String sanitized = operation.length() > 50 ? operation.substring(0, 50) : operation;
        return sanitized.toLowerCase().replaceAll("[^a-z0-9._-]", "_");
    }

    private String sanitizeLabel(String value) {
        if (value == null) return "unknown";
        String sanitized = value.length() > 30 ? value.substring(0, 30) : value;
        return sanitized.toLowerCase().replaceAll("[^a-z0-9._-]", "_");
    }
}
```

---

## Core Principles

### 1. SRE Four Golden Signals
Every service must track these four signals:
- **Latency**: Time to serve a request/process an event
- **Traffic**: Rate of requests/events (throughput)
- **Errors**: Rate of requests/events that fail
- **Saturation**: How "full" the service is (queue depth, resource utilization)

### 2. RED Method (Request-Driven Services)
For request-driven services, track:
- **Rate**: Requests per second
- **Errors**: Error rate (failures per second)
- **Duration**: Request latency (p50, p95, p99)

### 3. USE Method (Resource-Driven Services)
For resource monitoring, track:
- **Utilization**: Percentage of resource in use
- **Saturation**: Queue depth or resource contention
- **Errors**: Error count

### 4. Code Quality Principles: SOLID, DRY, and Clean Code

**CRITICAL: All instrumentation must follow software engineering best practices.**

#### SOLID Principles

**Single Responsibility Principle (SRP)**:
- ✅ **MetricsUtil/Metrics Package**: Single responsibility - only handles metrics recording
- ✅ **Service Classes**: Business logic remains separate from metrics
- ✅ **Sanitization Functions**: Each function has one clear purpose
- ❌ **DON'T**: Mix metrics logic with business logic in the same function

**Open/Closed Principle (OCP)**:
- ✅ **Extensible Design**: Add new metrics without modifying existing code
- ✅ **Interface-Based**: Use interfaces for metrics (Java: `MeterRegistry`, Go: `prometheus.Collector`)
- ✅ **Composition**: Metrics are composed into services, not inherited

**Liskov Substitution Principle (LSP)**:
- ✅ **Consistent Interfaces**: All metric recording functions follow same patterns
- ✅ **Predictable Behavior**: Metrics never change business logic behavior

**Interface Segregation Principle (ISP)**:
- ✅ **Focused Interfaces**: Metrics package provides specific, focused functions
- ✅ **No Fat Interfaces**: Services only depend on metrics functions they use

**Dependency Inversion Principle (DIP)**:
- ✅ **Java**: Depend on `MeterRegistry` abstraction, not concrete implementations
- ✅ **Go**: Use `promauto` for automatic registration, abstract metric creation
- ✅ **Dependency Injection**: Metrics utilities injected via constructor (Java) or struct fields (Go)

#### DRY (Don't Repeat Yourself)

**Centralized Metrics Package**:
- ✅ **Single Source of Truth**: All metrics defined in one place (`MetricsUtil`/`metrics` package)
- ✅ **Reusable Functions**: Common patterns extracted into helper functions
- ✅ **Consistent Naming**: Metric names follow consistent patterns
- ❌ **DON'T**: Duplicate metric definitions across multiple services
- ❌ **DON'T**: Repeat sanitization logic in every service

**Example - DRY Violation (❌ BAD)**:
```java
// ❌ BAD: Duplicated in every service
public Order createOrder(CreateOrderRequest request) {
    Counter counter = Counter.builder("service.operations.total")
        .tag("service", "order-service")
        .tag("operation", "create_order")
        .register(meterRegistry);
    // ...
}
```

**Example - DRY Compliance (✅ GOOD)**:
```java
// ✅ GOOD: Centralized in MetricsUtil
public Order createOrder(CreateOrderRequest request) {
    metricsUtil.recordOperation("create_order", "success");
    // ...
}
```

#### Clean Code Principles

**Meaningful Names**:
- ✅ **Descriptive Function Names**: `RecordOperation()`, `StartTimer()`, `SanitizeEndpoint()`
- ✅ **Clear Variable Names**: `operationTimer`, `stopTimer`, `sanitizedEndpoint`
- ✅ **Self-Documenting Code**: Code explains intent without excessive comments
- ❌ **DON'T**: Use abbreviations like `recOp()`, `st()`, `sanEp()`

**Small Functions**:
- ✅ **Focused Functions**: Each function does one thing well
- ✅ **Short Functions**: Keep functions under 20-30 lines when possible
- ✅ **Single Level of Abstraction**: Functions operate at consistent abstraction levels

**Error Handling**:
- ✅ **Fail-Safe Metrics**: Metrics never break business logic
- ✅ **Graceful Degradation**: If metrics fail, log and continue
- ✅ **No Side Effects**: Metric recording doesn't affect business outcomes

**Comments for Why, Not What**:
- ✅ **Explain Decisions**: Comment on why specific buckets or tags are chosen
- ✅ **Document Constraints**: Explain cardinality limits and sanitization rules
- ❌ **DON'T**: Comment obvious code like `// Increment counter`

#### Standard Java Practices

**Dependency Injection**:
- ✅ **Constructor Injection**: Use `@RequiredArgsConstructor` (Lombok) or explicit constructors
- ✅ **Spring Boot Auto-Configuration**: Leverage Spring's `MeterRegistry` bean
- ✅ **Interface-Based**: Depend on `MeterRegistry` interface, not implementations

**Exception Handling**:
- ✅ **Preserve Exception Types**: Never change exception types when adding metrics
- ✅ **Re-throw Original**: Always re-throw the original exception
- ✅ **No Exception Swallowing**: Metrics errors don't hide business exceptions

**Code Organization**:
- ✅ **Package Structure**: `com.company.service.common.util.MetricsUtil`
- ✅ **Naming Conventions**: camelCase for methods, UPPER_CASE for constants
- ✅ **Annotations**: Use `@Component`, `@Service` appropriately

**Example - Standard Java Pattern**:
```java
@Service
@RequiredArgsConstructor  // ✅ Constructor injection
@Slf4j
public class OrderService {
    
    private final MetricsUtil metricsUtil;  // ✅ Interface-based dependency
    private final OrderRepository orderRepo;
    
    public Order createOrder(CreateOrderRequest request) {
        var timer = metricsUtil.startOperationTimer();  // ✅ Clear naming
        try {
            Order order = orderRepo.save(convert(request));
            metricsUtil.recordOperation("create_order", "success");
            return order;
        } catch (Exception e) {
            metricsUtil.recordOperation("create_order", "error");
            log.error("Failed to create order", e);  // ✅ Proper logging
            throw e;  // ✅ Preserve exception
        } finally {
            metricsUtil.stopOperationTimer(timer, "create_order");
        }
    }
}
```

#### Standard Go Practices

**Package Organization**:
- ✅ **Internal Packages**: Use `internal/metrics` for service-specific metrics
- ✅ **Package Naming**: Lowercase, no underscores, descriptive (`metrics`, not `mtrcs`)
- ✅ **Exported vs Unexported**: Capitalize exported functions, lowercase for internal

**Error Handling**:
- ✅ **Explicit Error Returns**: Always return errors explicitly, never swallow
- ✅ **Error Wrapping**: Use `fmt.Errorf("...: %w", err)` for context
- ✅ **Defer for Cleanup**: Use `defer` for timer stops and cleanup

**Code Organization**:
- ✅ **Struct-Based Services**: Services as structs with methods
- ✅ **Dependency Injection**: Pass dependencies via struct fields or constructor functions
- ✅ **Context Propagation**: Always accept `context.Context` as first parameter

**Naming Conventions**:
- ✅ **PascalCase for Exported**: `RecordOperation()`, `StartTimer()`
- ✅ **camelCase for Unexported**: `sanitizeOperation()`, `idPattern`
- ✅ **Short Variable Names**: `ctx`, `req`, `resp` in local scope
- ✅ **Descriptive Names**: `operationTimer`, `stopTimer` for clarity

**Idiomatic Go Patterns**:
- ✅ **Defer for Cleanup**: Use `defer` for guaranteed cleanup
- ✅ **Early Returns**: Return early on errors, reduce nesting
- ✅ **Zero Values**: Leverage Go's zero values (nil slices, empty maps)
- ✅ **Interface Satisfaction**: Use interfaces for testability

**Example - Standard Go Pattern**:
```go
package service

import (
    "context"
    "time"
    "github.com/Vance-Club/goms/internal/metrics"
)

type OrderService struct {  // ✅ Struct-based service
    repo OrderRepository    // ✅ Dependency as field
}

// NewOrderService creates a new OrderService with dependencies
func NewOrderService(repo OrderRepository) *OrderService {  // ✅ Constructor function
    return &OrderService{repo: repo}
}

func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {  // ✅ Context first
    stopTimer := metrics.StartTimer("create_order")  // ✅ Clear naming
    defer stopTimer()  // ✅ Defer for cleanup
    
    // Business logic
    order, err := s.repo.Save(ctx, req)
    if err != nil {  // ✅ Early return on error
        metrics.RecordOperation("create_order", "error")
        return nil, err  // ✅ Explicit error return
    }
    
    metrics.RecordOperation("create_order", "success")
    return order, nil
}
```

#### Metrics Package Design (Both Languages)

**Separation of Concerns**:
- ✅ **Metrics Package**: Only handles metrics recording
- ✅ **Service Layer**: Only handles business logic
- ✅ **Clear Boundaries**: Metrics don't know about business entities

**Testability**:
- ✅ **Dependency Injection**: Metrics can be mocked/injected for testing
- ✅ **No Global State**: Avoid global metric registries where possible
- ✅ **Interface-Based**: Use interfaces for metrics in tests

**Performance**:
- ✅ **Pre-registered Metrics**: Register metrics at startup, not per-request
- ✅ **Efficient Sanitization**: Compile regex patterns once, reuse
- ✅ **Minimal Overhead**: Metrics recording should be fast (< 1ms)

### 4. Metric Hierarchy (L0, L1, L2)

#### L0 Metrics: Service-Level (Aggregated)
**Purpose**: Overall service health, SLO tracking, alerting
**Cardinality**: Very low (< 10 unique series)
**Tags**: `service` (infrastructure tags only)
**Examples**:
- `service.requests.total` - Total requests processed
- `service.errors.total` - Total errors
- `service.request.duration` - Request latency
- `service.throughput` - Requests per second

**When to use**: 
- Service-level dashboards
- SLO monitoring
- High-level alerts
- Executive reporting

#### L1 Metrics: Feature/Entity-Level (Business Logic)
**Purpose**: Business logic observability, feature-level debugging
**Cardinality**: Medium (10-1000 unique series)
**Tags**: `service`, `operation`, `outcome`, `entity_type`
**Examples**:
- `service.operations.total{operation, outcome}` - Operations by type and outcome
- `service.operations.duration{operation}` - Operation duration by type
- `service.feature.errors.total{feature, error_type}` - Feature-level errors

**When to use**:
- Feature-level dashboards
- Business metrics
- Identifying which feature/entity is problematic
- Debugging specific workflows

#### L2 Metrics: Instance/Resource-Level (Deep Debugging)
**Purpose**: Deep debugging, resource-level analysis
**Cardinality**: High (1000-10000+ unique series)
**Tags**: `service`, `instance_id`, `resource_id`, `user_id` (sanitized)
**Examples**:
- `service.resource.events.processed.total{resource_id, event_name}` (with sanitization)
- `service.user.operations.duration{user_id, operation}` (with sanitization)

**When to use**:
- Deep debugging sessions
- Per-instance troubleshooting
- Resource-level optimization
- Only when L0/L1 metrics indicate an issue

**⚠️ Warning**: L2 metrics can cause cardinality explosion. Always:
- Sanitize high-cardinality values (user IDs, resource IDs)
- Use bucketing for continuous values
- Monitor metric series count
- Set cardinality limits

### Metric Design Principle: Reuse Metrics with Labels, Don't Create Separate Metric Families

**CRITICAL**: When you have limitations on the number of custom metrics, or when you need to add dimensions to existing metrics, **always extend existing metrics with labels rather than creating new metric families**.

#### The Problem: Metric Family Bloat

**❌ BAD - Creating Separate Metrics**:
```go
// Creates 4 separate metric families
paymentAttemptUpdates = promauto.NewCounterVec(...)  // New metric family
paymentAttemptUpdateDuration = promauto.NewHistogramVec(...)  // New metric family
refundAttemptUpdates = promauto.NewCounterVec(...)  // New metric family
refundAttemptUpdateDuration = promauto.NewHistogramVec(...)  // New metric family
```

**Problems**:
- Increases metric count unnecessarily
- Violates metric count limits
- Creates maintenance overhead
- Duplicates metric structure

#### The Solution: Extend Existing Metrics with Labels

**✅ GOOD - Reusing Existing Metrics with Labels**:
```go
// Extend existing metric with optional label
serviceOperations = promauto.NewCounterVec(
    prometheus.CounterOpts{
        Name: "service_operations_total",
        Help: "Operations by type, outcome, and optional acquirer",
    },
    []string{"service", "operation", "outcome", "acquirer"},  // Added acquirer label
)

serviceOperationDuration = promauto.NewHistogramVec(
    prometheus.HistogramOpts{
        Name: "service_operations_duration_seconds",
        Help: "Operation duration by operation and optional acquirer",
    },
    []string{"service", "operation", "acquirer"},  // Added acquirer label
)

// Helper functions for operations with/without acquirer
func RecordOperation(operation, outcome string) {
    RecordOperationWithAcquirer(operation, outcome, "unknown")  // Default to "unknown"
}

func RecordOperationWithAcquirer(operation, outcome, acquirer string) {
    serviceOperations.WithLabelValues(ServiceName, sanitizeOperation(operation), outcome, sanitizeAcquirer(acquirer)).Inc()
}
```

**Benefits**:
- ✅ **Zero new metric families** - Reuses existing metrics
- ✅ **Respects metric count limits** - No metric bloat
- ✅ **Consistent structure** - All operations use same metrics
- ✅ **Low cardinality impact** - Only relevant operations use real label values
- ✅ **Follows Prometheus best practices** - Labels for dimensions, not separate metrics

#### Implementation Pattern

**1. Identify When to Add Labels**:
- When you need breakdown by a dimension (acquirer, region, feature, etc.)
- When multiple operations share the same metric structure
- When you have metric count limitations

**2. Extend Existing Metrics**:
```go
// Before: service_operations_total{service, operation, outcome}
// After:  service_operations_total{service, operation, outcome, acquirer}

// Add label to existing metric definition
serviceOperations = promauto.NewCounterVec(
    prometheus.CounterOpts{
        Name: "service_operations_total",
        Help: "Operations by type, outcome, and optional acquirer",
    },
    []string{"service", "operation", "outcome", "acquirer"},  // Added dimension
)
```

**3. Provide Backward-Compatible Helpers**:
```go
// Default function (for operations without the dimension)
func RecordOperation(operation, outcome string) {
    RecordOperationWithAcquirer(operation, outcome, "unknown")
}

// Extended function (for operations with the dimension)
func RecordOperationWithAcquirer(operation, outcome, acquirer string) {
    serviceOperations.WithLabelValues(ServiceName, sanitizeOperation(operation), outcome, sanitizeAcquirer(acquirer)).Inc()
}
```

**4. Use in Code**:
```go
// Regular operations (acquirer="unknown")
metrics.RecordOperation("create_order", "success")

// Operations with acquirer breakdown
acquirer := string(payment.Acquirer)
metrics.RecordOperationWithAcquirer("update_payment_attempt", "success", acquirer)
```

#### Cardinality Management

**Key Principle**: Only operations that need the dimension should use real label values. Others use a default value (e.g., "unknown").

**Example**:
```
# Regular operations (acquirer="unknown" - no cardinality increase)
service_operations_total{service="goms", operation="create_order", outcome="success", acquirer="unknown"} 5000

# Payment operations (acquirer breakdown - low cardinality: ~10-20 acquirers)
service_operations_total{service="goms", operation="update_payment_attempt", outcome="success", acquirer="checkout"} 1500
service_operations_total{service="goms", operation="update_payment_attempt", outcome="success", acquirer="truelayer"} 800
```

**Cardinality Impact**:
- Operations without dimension: No increase (all use "unknown")
- Operations with dimension: Low increase (only relevant operations × dimension values)
- Total impact: Minimal (only operations that need breakdown contribute)

#### When NOT to Use This Pattern

**Create separate metrics when**:
- The metric has fundamentally different semantics (e.g., `workflow_executions_total` vs `service_operations_total`)
- The metric has different retention/aggregation needs
- The metric represents a different abstraction level (L0 vs L1 vs L2)

**Use labels when**:
- You need dimensional breakdown of the same metric
- You have metric count limitations
- The operations share the same semantic meaning
- You want consistent querying across operations

#### Decision Framework

**Question**: "Should I create a new metric or add a label to an existing one?"

**Answer**: 
1. **Same semantic meaning?** → Use label
2. **Different abstraction level?** → Create separate metric
3. **Metric count limitation?** → Always prefer labels
4. **Need dimensional breakdown?** → Use label
5. **Fundamentally different metric?** → Create separate metric

**Example Decision**:
- ❌ `payment_attempt_updates_total` vs `service_operations_total` → **Use label** (same semantic: operation counts)
- ✅ `workflow_executions_total` vs `service_operations_total` → **Separate metric** (different abstraction: workflow-level vs operation-level)

## Implementation Strategy

### Step 0: Verify Prerequisites and Setup

**CRITICAL: Check and set up dependencies and configuration BEFORE implementing metrics.**

1. **Check Build Dependencies**:
   - Read `build.gradle` (Gradle) or `pom.xml` (Maven)
   - Verify `io.micrometer:micrometer-registry-prometheus` is present
   - Verify `org.springframework.boot:spring-boot-starter-actuator` is present (if using Spring Boot)
   - If missing, add them automatically

2. **Check Application Configuration**:
   - Read `application.yml` or `application.properties`
   - Verify Prometheus endpoint is exposed:
     ```yaml
     management:
       endpoints:
         web:
           exposure:
             include: health,metrics,prometheus
       metrics:
         export:
           prometheus:
             enabled: true
     ```
   - If missing, add configuration automatically

3. **Verify MeterRegistry Availability**:
   - Check if `MeterRegistry` bean is available (Spring Boot provides this automatically)
   - If using non-Spring Boot, check if Micrometer is configured
   - Verify dependency injection setup

4. **Check Existing Instrumentation**:
   - Search for existing `MeterRegistry` usage in codebase
   - Identify what metrics already exist
   - Understand existing patterns and naming conventions

**Example verification process**:
```
1. Read build.gradle → Check: micrometer-registry-prometheus present? NO → Add it
2. Read application.yml → Check: prometheus endpoint exposed? NO → Add configuration
3. Search codebase → Find: Existing MeterRegistry usage in ServiceA.java
4. Verify: Check if MeterRegistry bean is available (Spring Boot auto-configures)
5. Verify build: ./gradlew build → SUCCESS
6. Result: Dependencies added, configuration updated, ready for instrumentation
```

**Verification Checklist** (complete before proceeding to Step 1):
- ✅ `micrometer-registry-prometheus` dependency present in build file (`build.gradle` or `pom.xml`)
- ✅ `spring-boot-starter-actuator` dependency present (if using Spring Boot)
- ✅ Prometheus endpoint configured in `application.yml` or `application.properties`
- ✅ `MeterRegistry` available (auto-configured by Spring Boot when dependencies present)
- ✅ Build compiles successfully (`./gradlew build` or `mvn compile`)
- ✅ No duplicate dependencies in build file
- ✅ Configuration syntax is correct (YAML indentation, property names)

---

## Go Service Instrumentation

### Step 0 (Go): Verify Prerequisites and Setup

**CRITICAL: Check and set up dependencies and configuration BEFORE implementing metrics.**

1. **Check Go Dependencies**:
   - Read `go.mod`
   - Verify `github.com/prometheus/client_golang` is present
   - If missing, add it automatically:
     ```bash
     go get github.com/prometheus/client_golang/prometheus
     go get github.com/prometheus/client_golang/prometheus/promhttp
     ```

2. **Check HTTP Server Setup**:
   - Read main.go or router setup files
   - Verify HTTP server is configured (Gin, Echo, net/http, etc.)
   - Identify where to add Prometheus metrics endpoint

3. **Check Existing Instrumentation**:
   - Search for existing `prometheus` usage in codebase
   - Identify what metrics already exist
   - Understand existing patterns and naming conventions

**Example verification process**:
```
1. Read go.mod → Check: prometheus/client_golang present? NO → Add it
2. Read cmd/main.go → Check: HTTP server setup found (Gin router)
3. Search codebase → Find: Existing prometheus usage in service.go
4. Verify: Check if prometheus registry is initialized
5. Verify build: go build → SUCCESS
6. Result: Dependencies added, ready for instrumentation
```

**Verification Checklist** (complete before proceeding to Step 1):
- ✅ `github.com/prometheus/client_golang` dependency present in `go.mod`
- ✅ HTTP server/router identified (Gin, Echo, net/http, etc.)
- ✅ Prometheus metrics endpoint can be added to router
- ✅ Build compiles successfully (`go build`)
- ✅ No duplicate dependencies in `go.mod`

### Step 1 (Go): Analyze the Service Architecture (Automatic Codebase Analysis)

**CRITICAL: Read and analyze the codebase first. Do NOT ask the user to describe their service.**

1. **Read the codebase**:
   - Read main service files, handlers, and components
   - Read configuration files (config.json, config.yaml)
   - Understand package structure and dependencies
   - Identify entry points (HTTP handlers, message consumers, workers)

2. **Identify Service Type** (from code analysis):
   - Request-driven: Look for HTTP handlers, Gin routes, Echo handlers
   - Resource-driven: Look for cron jobs, batch processors, workers
   - Message-driven: Look for Kafka/RabbitMQ consumers, event handlers
   - Hybrid: Combination of above patterns

3. **Map Request/Event Flow** (from code analysis):
   - Trace function calls from entry points
   - Identify processing steps from function implementations
   - Map error handling from error returns
   ```
   Input → Processing → Output
   (Rate)  (Duration)  (Success/Error)
   ```

4. **Identify Key Entities** (from code analysis):
   - Business entities: From struct names, domain models (Order, User, Event, etc.)
   - Key operations: From function names (ProcessOrder, CreateUser, HandleEvent, etc.)
   - Critical paths: From code flow analysis (happy path, error paths from error handling)

5. **Identify Dependencies** (from code analysis):
   - External services: From HTTP client usage, service clients
   - Message queues: From Kafka/RabbitMQ consumers and producers
   - Databases: From GORM, database/sql, repository patterns
   - Resources: From connection pool configurations, goroutine pools

### Step 2 (Go): Add Missing Dependencies (If Needed)

**After Step 0 verification, add any missing dependencies automatically.**

**For Go (`go.mod`)**:
```bash
go get github.com/prometheus/client_golang/prometheus
go get github.com/prometheus/client_golang/prometheus/promhttp
go mod tidy
```

**Verify addition**:
- Check that dependencies are in `go.mod`
- Ensure version compatibility
- Verify build compiles: `go build`
- Check no duplicate dependencies

### Step 3 (Go): Configure Prometheus Endpoint

**Add Prometheus metrics endpoint to your HTTP server.**

**For Gin Framework**:
```go
import (
    "github.com/gin-gonic/gin"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

func setupRouter() *gin.Engine {
    router := gin.Default()
    
    // Add Prometheus metrics endpoint
    router.GET("/metrics", gin.WrapH(promhttp.Handler()))
    
    // Your other routes...
    return router
}
```

**For Echo Framework**:
```go
import (
    "github.com/labstack/echo/v4"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

func setupRouter() *echo.Echo {
    e := echo.New()
    
    // Add Prometheus metrics endpoint
    e.GET("/metrics", echo.WrapHandler(promhttp.Handler()))
    
    // Your other routes...
    return e
}
```

**For net/http**:
```go
import (
    "net/http"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
    http.Handle("/metrics", promhttp.Handler())
    // Your other handlers...
    http.ListenAndServe(":8080", nil)
}
```

**Verify configuration**:
- Check that `/metrics` endpoint is added to router
- Ensure `promhttp.Handler()` is imported and used correctly
- Verify server starts successfully
- Test endpoint: `curl http://localhost:8080/metrics`

### Step 3.5 (Go): Configure Infrastructure for Prometheus Scraping (Required for Production)

**CRITICAL: This step is required for metrics to be collected by Datadog Agent in production.**

After code instrumentation, you must configure the Datadog Agent to scrape the Prometheus endpoint. This is typically done via **Docker labels in the ECS task definition** (for ECS Fargate) or **Kubernetes annotations** (for Kubernetes).

#### For ECS Fargate (Recommended: Docker Labels)

**Add Docker labels to your Dockerfile** (for reference, but may need to be in ECS task definition):

```dockerfile
# Datadog Auto-Discovery Labels
# These labels enable Datadog to automatically discover and scrape Prometheus metrics
# NOTE: In ECS Fargate, these labels may not be accessible from Dockerfile.
# You MUST also add them to the ECS task definition's dockerLabels section for auto-discovery to work.
LABEL com.datadoghq.ad.check_names='["openmetrics"]'
LABEL com.datadoghq.ad.init_configs='[{}]'
LABEL com.datadoghq.ad.instances='[{"openmetrics_endpoint":"http://%%host%%:8080/metrics","namespace":"","metrics":[".*"],"tags":["service:your-service-name"]}]'
```

**Add Docker labels to ECS task definition** (REQUIRED for ECS Fargate):

```json
{
  "name": "your-service-name",
  "dockerLabels": {
    "com.datadoghq.ad.check_names": "[\"openmetrics\"]",
    "com.datadoghq.ad.init_configs": "[{}]",
    "com.datadoghq.ad.instances": "[{\"openmetrics_endpoint\":\"http://%%host%%:8080/metrics\",\"namespace\":\"\",\"metrics\":[\".*\"],\"tags\":[\"service:your-service-name\",\"env:${ENV}\"]}]"
  }
}
```

**Important Configuration Details**:
- **Port**: Use the port your Go service listens on (default: 8080)
- **Path**: Use `/metrics` (Prometheus endpoint)
- **Service Tag**: Use your service name (e.g., `goms-service`)
- **Environment Tag**: Use your environment variable (e.g., `env:prod`, `env:stg`)

**Example for goms-service** (with context-path `/goms-service`):
```json
{
  "dockerLabels": {
    "com.datadoghq.ad.check_names": "[\"openmetrics\"]",
    "com.datadoghq.ad.init_configs": "[{}]",
    "com.datadoghq.ad.instances": "[{\"openmetrics_endpoint\":\"http://%%host%%:8080/goms-service/metrics\",\"namespace\":\"\",\"metrics\":[\".*\"],\"tags\":[\"service:goms-service\",\"env:prod\"]}]"
  }
}
```

**Datadog Agent Container Configuration**:
Ensure the Datadog Agent sidecar has Prometheus scraping enabled:
```json
{
  "name": "datadog-agent",
  "environment": [
    {
      "name": "DD_PROMETHEUS_SCRAPE_ENABLED",
      "value": "true"
    }
  ]
}
```

#### For Kubernetes (Annotations)

Add annotations to your Pod/Deployment:

```yaml
metadata:
  annotations:
    ad.datadoghq.com/your-service-name.check_names: '["openmetrics"]'
    ad.datadoghq.com/your-service-name.init_configs: '[{}]'
    ad.datadoghq.com/your-service-name.instances: '[{"openmetrics_endpoint":"http://%%host%%:8080/metrics","namespace":"","metrics":[".*"],"tags":["service:your-service-name","env:prod"]}]'
```

#### Why This Step Was Previously Missing

The skill document focused on **code instrumentation** but didn't explicitly include **infrastructure configuration** for metric collection. This is a critical gap because:

1. **Code instrumentation alone is not sufficient**: Metrics must be scraped by the monitoring agent
2. **Infrastructure configuration is deployment-specific**: ECS vs Kubernetes vs Docker Compose require different approaches
3. **This step is often handled by DevOps/Platform teams**: But developers need to know what to request

**Going Forward**: Always include infrastructure configuration as Step 3.5 when instrumenting services for production deployment.

### Step 4 (Go): Create MetricsUtil Helper Package

**CRITICAL: Create a centralized metrics package to ensure consistency and reduce code duplication. Follow Go best practices: package organization, error handling, and idiomatic patterns.**

Create `internal/metrics/metrics.go`:

```go
package metrics

import (
    "regexp"
    "strings"
    "time"
    
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

// ServiceName is the service identifier used in all metrics.
// Set this at package initialization or via environment variable.
var ServiceName = "your-service-name"

var (
    // L0: Service-level metrics (pre-registered for performance)
    serviceRequests = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "service_requests_total",
            Help: "Total requests processed",
        },
        []string{"service"},
    )
    
    serviceErrors = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "service_errors_total",
            Help: "Total errors",
        },
        []string{"service", "error_type"},
    )
    
    serviceRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "service_request_duration_seconds",
            Help:    "Request processing duration",
            Buckets: prometheus.DefBuckets, // [.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10]
        },
        []string{"service"},
    )
    
    // L1: Operation-level metrics
    serviceOperations = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "service_operations_total",
            Help: "Operations by type and outcome",
        },
        []string{"service", "operation", "outcome"},
    )
    
    serviceOperationDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "service_operations_duration_seconds",
            Help:    "Operation duration",
            Buckets: prometheus.DefBuckets,
        },
        []string{"service", "operation"},
    )
    
    // Pre-compiled regex patterns (compile once, reuse many times)
    idPattern          = regexp.MustCompile(`/(?:\d+|\p{XDigit}{8}-\p{XDigit}{4}-\p{XDigit}{4}-\p{XDigit}{4}-\p{XDigit}{12}|[a-zA-Z0-9]{24,})`)
    sanitizePattern   = regexp.MustCompile(`[^a-zA-Z0-9._-]`)
    endpointPattern   = regexp.MustCompile(`[^a-zA-Z0-9/_{}-]`)
)

// RecordRequest records a service-level request.
// This is thread-safe and can be called concurrently.
func RecordRequest() {
    serviceRequests.WithLabelValues(ServiceName).Inc()
}

// RecordError records a service-level error with sanitized error type.
func RecordError(errorType string) {
    serviceErrors.WithLabelValues(ServiceName, sanitizeErrorType(errorType)).Inc()
}

// RecordRequestDuration records request processing duration in seconds.
func RecordRequestDuration(duration time.Duration) {
    serviceRequestDuration.WithLabelValues(ServiceName).Observe(duration.Seconds())
}

// RecordOperation records an operation with outcome (success/error).
func RecordOperation(operation, outcome string) {
    serviceOperations.WithLabelValues(ServiceName, sanitizeOperation(operation), outcome).Inc()
}

// RecordOperationDuration records operation duration in seconds.
func RecordOperationDuration(operation string, duration time.Duration) {
    serviceOperationDuration.WithLabelValues(ServiceName, sanitizeOperation(operation)).Observe(duration.Seconds())
}

// StartTimer returns a function to stop and record duration.
// Use with defer for guaranteed cleanup:
//   stopTimer := metrics.StartTimer("operation_name")
//   defer stopTimer()
func StartTimer(operation string) func() {
    start := time.Now()
    return func() {
        RecordOperationDuration(operation, time.Since(start))
    }
}

// sanitizeOperation sanitizes operation names for use in metric labels.
// Limits length to 50 chars and normalizes to lowercase with underscores.
func sanitizeOperation(operation string) string {
    if operation == "" {
        return "unknown"
    }
    sanitized := operation
    if len(sanitized) > 50 {
        sanitized = sanitized[:50]
    }
    sanitized = strings.ToLower(sanitized)
    sanitized = sanitizePattern.ReplaceAllString(sanitized, "_")
    return sanitized
}

// sanitizeErrorType sanitizes error type names for use in metric labels.
// Limits length to 50 chars and normalizes to lowercase with underscores.
func sanitizeErrorType(errorType string) string {
    if errorType == "" {
        return "unknown"
    }
    sanitized := errorType
    if len(sanitized) > 50 {
        sanitized = sanitized[:50]
    }
    sanitized = strings.ToLower(sanitized)
    sanitized = sanitizePattern.ReplaceAllString(sanitized, "_")
    return sanitized
}

// SanitizeEndpoint normalizes endpoint paths by replacing ID-like segments with {id}.
// This prevents cardinality explosion from dynamic path segments.
// Preserves short resource names like "list", "search", "status".
func SanitizeEndpoint(endpoint string) string {
    if endpoint == "" {
        return "unknown"
    }
    
    // Replace numeric IDs, UUIDs, and long alphanumeric strings (24+ chars) with {id}
    sanitized := idPattern.ReplaceAllString(endpoint, "/{id}")
    
    // Limit length to prevent excessive cardinality
    if len(sanitized) > 100 {
        sanitized = sanitized[:100]
    }
    
    // Clean up special characters
    sanitized = endpointPattern.ReplaceAllString(sanitized, "_")
    return sanitized
}
```

**Key Features Following Go Best Practices**:
- ✅ **Package Organization**: Clear package structure with exported/unexported functions
- ✅ **DRY Principle**: Centralized metric definitions, no duplication
- ✅ **Performance**: Pre-compiled regex patterns, pre-registered metrics
- ✅ **Idiomatic Go**: Defer pattern for cleanup, explicit error handling
- ✅ **Documentation**: Clear function comments explaining purpose
- ✅ **Thread Safety**: Prometheus metrics are thread-safe by design
- ✅ **Single Responsibility**: Package only handles metrics, no business logic
- ✅ **Naming Conventions**: PascalCase for exported, camelCase for unexported

### Step 5 (Go): Instrument Service Functions

**CRITICAL: Instrument all service functions with L0 and L1 metrics.**

**⚠️ CRITICAL: DO NOT CHANGE FUNCTIONAL LOGIC**

When adding metrics instrumentation:
1. **ONLY add metrics calls** - Do not modify business logic, validation, error handling, or control flow
2. **Wrap existing code** - Add defer statements for metrics recording, preserve all existing error handling
3. **Preserve all existing behavior** - All function signatures, return values, side effects must remain identical
4. **Add metrics non-intrusively** - Metrics should be additive only, never replace or modify existing code paths

**Example of CORRECT instrumentation (no logic changes)**:
```go
package service

import (
    "time"
    "github.com/Vance-Club/goms/internal/metrics"
)

type OrderService struct {
    // ... existing fields
}

// BEFORE (original code):
// func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {
//     order, err := s.doCreateOrder(ctx, req)
//     if err != nil {
//         return nil, err
//     }
//     return order, nil
// }

// AFTER (with metrics - logic unchanged):
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {
    stopTimer := metrics.StartTimer("create_order")
    defer stopTimer()
    
    // Business logic - UNCHANGED
    order, err := s.doCreateOrder(ctx, req)
    
    // Metrics recording - ADDED
    if err != nil {
        metrics.RecordOperation("create_order", "error")
        metrics.RecordError(err.Error())
        return nil, err // PRESERVED - same return value and error
    }
    
    metrics.RecordOperation("create_order", "success")
    return order, nil // PRESERVED - same return value
}
```

**Example of INCORRECT instrumentation (logic changed)**:
```go
// ❌ WRONG: Changed error handling
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {
    order, err := s.doCreateOrder(ctx, req)
    if err != nil {
        metrics.RecordOperation("create_order", "error")
        // ❌ WRONG: Changed error type
        return nil, fmt.Errorf("failed to create order: %w", err)
    }
    // ...
}

// ❌ WRONG: Added early return
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {
    if req == nil {
        metrics.RecordOperation("create_order", "error")
        return nil, nil // ❌ WRONG: Changed behavior (original might return error)
    }
    // ...
}

// ❌ WRONG: Changed function signature
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest, registry *prometheus.Registry) (*Order, error) {
    // ❌ WRONG: Added parameter, breaks existing callers
}
```

#### Pattern: Service Function Instrumentation

```go
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {
    stopTimer := metrics.StartTimer("create_order")
    defer stopTimer()
    
    // Business logic - PRESERVE EXACTLY AS IS
    order, err := s.doCreateOrder(ctx, req)
    
    // Metrics recording - ADD ONLY
    if err != nil {
        metrics.RecordOperation("create_order", "error")
        metrics.RecordError(err.Error())
        return nil, err // PRESERVED - same error
    }
    
    metrics.RecordOperation("create_order", "success")
    return order, nil // PRESERVED - same return value
}
```

#### Pattern: HTTP Handler Instrumentation (Gin)

```go
import (
    "time"
    "github.com/gin-gonic/gin"
    "github.com/Vance-Club/goms/internal/metrics"
)

func (h *OrderHandler) CreateOrder(c *gin.Context) {
    start := time.Now()
    metrics.RecordRequest()
    defer func() {
        metrics.RecordRequestDuration(time.Since(start))
    }()
    
    var req CreateOrderRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        metrics.RecordError("validation_error")
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    
    order, err := h.orderService.CreateOrder(c.Request.Context(), &req)
    if err != nil {
        metrics.RecordError(err.Error())
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(200, order)
}
```

#### Pattern: External Dependency Calls

```go
func (s *PaymentService) CallExternalAPI(ctx context.Context, endpoint string, req *Request) (*Response, error) {
    stopTimer := metrics.StartTimer("external_api_call")
    defer stopTimer()
    
    sanitizedEndpoint := metrics.SanitizeEndpoint(endpoint)
    
    // External call - PRESERVE EXACTLY AS IS
    resp, err := s.httpClient.Post(ctx, endpoint, req)
    
    // Metrics recording - ADD ONLY
    if err != nil {
        metrics.RecordOperation("external_api_call", "error")
        externalErrors.WithLabelValues(ServiceName, "external-service", sanitizedEndpoint, err.Error()).Inc()
        return nil, err // PRESERVED
    }
    
    metrics.RecordOperation("external_api_call", "success")
    externalRequests.WithLabelValues(ServiceName, "external-service", sanitizedEndpoint, "success").Inc()
    return resp, nil // PRESERVED
}
```

### Step 6 (Go): Implement Cardinality Management

**Use the same sanitization functions from Step 4 (Go) MetricsUtil package.**

The `sanitizeOperation`, `sanitizeErrorType`, and `SanitizeEndpoint` functions handle:
- Length limits (50 chars for operations/errors, 100 for endpoints)
- Character sanitization (only alphanumeric, dots, underscores, hyphens)
- ID pattern replacement (numeric IDs, UUIDs, long strings → `{id}`)

**For high-cardinality values, use bucketing**:
```go
func getBatchSizeBucket(size int) string {
    switch {
    case size == 1:
        return "1"
    case size <= 5:
        return "2-5"
    case size <= 10:
        return "6-10"
    case size <= 50:
        return "11-50"
    case size <= 100:
        return "51-100"
    default:
        return "100+"
    }
}
```

### Step 7 (Go): Choose Metric Types

#### Counters
**Use for**: Cumulative counts (requests, errors, events)
**Characteristics**: Monotonically increasing
**Example**:
```go
var eventsProcessed = promauto.NewCounterVec(
    prometheus.CounterOpts{
        Name: "service_events_processed_total",
        Help: "Total events processed",
    },
    []string{"service", "event_type"},
)

eventsProcessed.WithLabelValues("my-service", "order_created").Inc()
```

#### Histograms
**Use for**: Duration measurements (latency, processing time)
**Characteristics**: Automatically tracks count, sum, percentiles
**Example**:
```go
var operationDuration = promauto.NewHistogramVec(
    prometheus.HistogramOpts{
        Name:    "service_operation_duration_seconds",
        Help:    "Operation duration",
        Buckets: prometheus.DefBuckets,
    },
    []string{"service", "operation"},
)

start := time.Now()
// Do work
operationDuration.WithLabelValues("my-service", "create_order").Observe(time.Since(start).Seconds())
```

#### Gauges
**Use for**: Current values (queue size, active connections, cache size)
**Characteristics**: Can go up or down
**Example**:
```go
var queueSize = promauto.NewGaugeVec(
    prometheus.GaugeOpts{
        Name: "service_queue_size",
        Help: "Current queue size",
    },
    []string{"service", "queue_name"},
)

queueSize.WithLabelValues("my-service", "orders").Set(float64(len(queue)))
```

### Step 8 (Go): Naming Conventions

Follow Prometheus naming conventions:
- Use underscores (`_`) as separators: `service_operation_duration_seconds`
- Use lowercase: `service_events_processed_total`
- Use `_total` suffix for counters: `service_requests_total`
- Use `_seconds`, `_bytes`, etc. for units: `service_request_duration_seconds`
- Use descriptive names: `service_events_processed_total` not `service_events_total`
- Be consistent: Use same prefix for related metrics

**Good Examples**:
- `service_requests_total`
- `service_request_duration_seconds`
- `service_errors_total`
- `service_events_processed_total`
- `service_operations_total{operation, outcome, acquirer}` (with optional labels)

**Bad Examples**:
- `requests` (missing service prefix)
- `serviceRequests` (use underscores, not camelCase)
- `service.req.total` (use underscores, not dots)
- `service_events` (not descriptive)
- `payment_attempt_updates_total` (should reuse `service_operations_total` with `acquirer` label)

#### ⚠️ CRITICAL: Prefer Labels Over Separate Metrics

**When adding dimensional breakdowns (acquirer, region, feature, etc.), extend existing metrics with labels rather than creating new metric families.**

**❌ BAD - Creating Separate Metrics**:
```go
// Don't do this - creates unnecessary metric families
paymentAttemptUpdates = promauto.NewCounterVec(...)
refundAttemptUpdates = promauto.NewCounterVec(...)
```

**✅ GOOD - Extending Existing Metrics**:
```go
// Extend existing metric with optional label
serviceOperations = promauto.NewCounterVec(
    prometheus.CounterOpts{
        Name: "service_operations_total",
        Help: "Operations by type, outcome, and optional acquirer",
    },
    []string{"service", "operation", "outcome", "acquirer"},  // Added dimension
)

// Use helper functions
func RecordOperation(operation, outcome string) {
    RecordOperationWithAcquirer(operation, outcome, "unknown")  // Default
}

func RecordOperationWithAcquirer(operation, outcome, acquirer string) {
    serviceOperations.WithLabelValues(ServiceName, sanitizeOperation(operation), outcome, sanitizeAcquirer(acquirer)).Inc()
}
```

**Benefits**:
- Respects metric count limitations
- Maintains consistent metric structure
- Reduces maintenance overhead
- Follows Prometheus best practices (labels for dimensions, not separate metrics)

### Step 9 (Go): Error Handling

Always handle metric recording errors gracefully:
```go
func recordMetricSafely(fn func()) {
    defer func() {
        if r := recover(); r != nil {
            // Log but don't fail the business logic
            log.Printf("Failed to record metric: %v", r)
        }
    }()
    fn()
}

// Usage
recordMetricSafely(func() {
    metrics.RecordOperation("create_order", "success")
})
```

**CRITICAL**: Prometheus client_golang panics are rare, but wrap metric operations in defer/recover if needed to prevent metric failures from breaking business logic.

### Step 10 (Go): Testing Metrics

Test that metrics are recorded correctly:
```go
func TestOrderServiceMetrics(t *testing.T) {
    // Arrange
    registry := prometheus.NewRegistry()
    // Register your metrics to test registry
    service := NewOrderService()
    
    // Act
    _, err := service.CreateOrder(context.Background(), &CreateOrderRequest{})
    
    // Assert
    if err != nil {
        t.Fatalf("Unexpected error: %v", err)
    }
    
    // Verify metrics were recorded
    // Use prometheus/testutil to gather metrics
    metricFamilies, err := registry.Gather()
    if err != nil {
        t.Fatalf("Failed to gather metrics: %v", err)
    }
    
    // Assert metric values
    // ...
}
```

### Step 11 (Go): Verify Metrics Are Exposed

**After implementing metrics, verify they are exposed correctly:**

1. **Start the service**:
   ```bash
   go run cmd/main.go
   ```

2. **Check Prometheus endpoint**:
   ```bash
   curl http://localhost:8080/metrics | grep service_
   ```

3. **Verify metric format**:
   - Metrics should be in Prometheus format: `metric_name{tag1="value1",tag2="value2"} value`
   - Check that labels are present and correct
   - Verify metric values are being recorded

4. **Test metric recording**:
   - Trigger operations that should record metrics
   - Verify metrics appear in `/metrics` endpoint
   - Check metric values increase as expected

### Step 12 (Go): Create Observability Documentation

**CRITICAL: After implementing metrics, create comprehensive documentation for the team. This is mandatory for production services.**

**Create `internal/metrics/README.md`** (or `docs/observability.md` at project root) with the following sections:

#### 1. Overview Section
- Purpose of metrics instrumentation
- Metric hierarchy (L0, L1, L2) explanation
- Cardinality management approach
- Service name and configuration

#### 2. Metrics Catalog
**Document ALL metrics with:**
- **Metric name** (exact Prometheus metric name)
- **Metric type** (Counter, Histogram, Gauge)
- **Labels** (all label names and their possible values)
- **Description** (what the metric measures)
- **Cardinality** (estimated number of unique series)
- **Example values** (sample Prometheus output)

**Format for each metric:**
```markdown
### `service_requests_total`
- **Type**: Counter
- **Labels**: `service` (string, always "goms-service")
- **Description**: Total HTTP requests processed by the service
- **Cardinality**: Very low (< 10 series)
- **Example**: `service_requests_total{service="goms-service"} 15234`
```

#### 3. Label Values Reference
**Document all possible label values:**
- `workflow_type`: List all valid values (e.g., "order", "fulfillment", "refund", "payment")
- `operation`: List common operations (e.g., "create_order", "process_payment")
- `outcome`: Always "success" or "error"
- `error_type`: Common error types (sanitized)
- `state`: Workflow state names (document that these are sanitized, max 50 chars)

#### 4. Usage Guide
**How to use the metrics package:**
- Import statement
- Basic usage patterns
- Code examples for:
  - Recording operations
  - Recording errors
  - Timing operations
  - Recording workflow metrics

**Example code snippets:**
```go
// Recording an operation
stopTimer := metrics.StartTimer("create_order")
defer stopTimer()

result, err := service.DoWork()
if err != nil {
    metrics.RecordOperation("create_order", "error")
    metrics.RecordError(err.Error())
    return err
}
metrics.RecordOperation("create_order", "success")
```

#### 5. Prometheus Query Examples
**Common queries for:**
- Service health (request rate, error rate)
- Operation performance (p95 latency by operation)
- Workflow metrics (execution rate, state transitions)
- Error analysis (error rate by type)

**Example queries:**
```promql
# Request rate
rate(service_requests_total[5m])

# Error rate
rate(service_errors_total[5m])

# P95 latency by operation
histogram_quantile(0.95, rate(service_operations_duration_seconds_bucket[5m]))

# Workflow execution success rate
rate(workflow_executions_total{outcome="success"}[5m]) / rate(workflow_executions_total[5m])
```

#### 6. Cardinality Management
- Explain how cardinality is controlled
- Document sanitization functions
- State assumptions (e.g., "workflow states are bounded by workflow definitions")
- Monitoring guidance (what to watch for)

#### 7. Infrastructure Configuration
- Prometheus endpoint location (`/metrics`)
- How to access metrics (curl example)
- Datadog Agent configuration (if applicable)
- ECS/Kubernetes scraping setup (if applicable)

#### 8. SLO Thresholds & Alerting (Optional)
- Recommended alert thresholds
- SLO targets (e.g., 99.9% success rate)
- Example alert rules

#### 9. Troubleshooting
- Common issues (metrics not appearing, high cardinality)
- How to verify metrics are working
- Debugging tips

**Template Structure:**
```markdown
# Metrics Documentation

## Overview
[Service name, purpose, metric hierarchy]

## Metrics Catalog

### L0: Service-Level Metrics
[Document each L0 metric]

### L1: Operation-Level Metrics
[Document each L1 metric]

### L1: Workflow Metrics
[Document workflow-specific metrics]

## Label Values Reference
[All possible label values]

## Usage Guide
[How to use the package]

## Prometheus Query Examples
[Common queries]

## Cardinality Management
[How cardinality is controlled]

## Infrastructure Configuration
[Endpoint, scraping setup]

## Troubleshooting
[Common issues and solutions]
```

---

### Step 1: Analyze the Service Architecture (Automatic Codebase Analysis)

**CRITICAL: Read and analyze the codebase first. Do NOT ask the user to describe their service.**

1. **Read the codebase**:
   - Read main service classes, controllers, and components
   - Read configuration files (application.yml, build.gradle)
   - Understand package structure and dependencies
   - Identify entry points (REST controllers, message listeners, schedulers)

2. **Identify Service Type** (from code analysis):
   - Request-driven: Look for `@RestController`, `@RequestMapping`, HTTP handlers
   - Resource-driven: Look for `@Scheduled`, batch processors, ETL patterns
   - Message-driven: Look for `@KafkaListener`, `@RabbitListener`, message consumers
   - Hybrid: Combination of above patterns

3. **Map Request/Event Flow** (from code analysis):
   - Trace method calls from entry points
   - Identify processing steps from method implementations
   - Map error handling from try-catch blocks
   ```
   Input → Processing → Output
   (Rate)  (Duration)  (Success/Error)
   ```

4. **Identify Key Entities** (from code analysis):
   - Business entities: From class names, DTOs, domain models (Order, User, Event, etc.)
   - Key operations: From method names (processOrder, createUser, handleEvent, etc.)
   - Critical paths: From code flow analysis (happy path, error paths from exception handling)

5. **Identify Dependencies** (from code analysis):
   - External services: From HTTP client usage, service clients
   - Message queues: From Kafka/RabbitMQ listeners and producers
   - Databases: From repository interfaces, JPA entities
   - Resources: From connection pool configurations, thread pool executors

### Step 2: Add Missing Dependencies (If Needed)

**After Step 0 verification, add any missing dependencies automatically.**

**For Gradle (`build.gradle`)**:
```gradle
dependencies {
    // Micrometer Prometheus registry
    implementation 'io.micrometer:micrometer-registry-prometheus'
    
    // Spring Boot Actuator (if using Spring Boot)
    implementation 'org.springframework.boot:spring-boot-starter-actuator'
}
```

**For Maven (`pom.xml`)**:
```xml
<dependencies>
    <!-- Micrometer Prometheus registry -->
    <dependency>
        <groupId>io.micrometer</groupId>
        <artifactId>micrometer-registry-prometheus</artifactId>
    </dependency>
    
    <!-- Spring Boot Actuator (if using Spring Boot) -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>
</dependencies>
```

**Verify addition**:
- Check that dependencies are in the correct section (`dependencies` block for Gradle, `<dependencies>` for Maven)
- Ensure version compatibility (Spring Boot manages versions automatically via BOM)
- If NOT using Spring Boot, specify version explicitly:
  ```gradle
  implementation 'io.micrometer:micrometer-registry-prometheus:1.11.0'
  ```
- Verify no duplicate dependencies
- Check build file syntax is correct
- Verify build compiles: `./gradlew build` or `mvn compile`

### Step 3: Configure Application (If Needed)

**After Step 0 verification, add missing configuration automatically.**

**For `application.yml`**:
```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,metrics,prometheus
  metrics:
    export:
      prometheus:
        enabled: true
  endpoint:
    prometheus:
      enabled: true
```

**For `application.properties`**:
```properties
management.endpoints.web.exposure.include=health,metrics,prometheus
management.metrics.export.prometheus.enabled=true
management.endpoint.prometheus.enabled=true
```

**Verify configuration**:
- Check that configuration is in the correct file (`application.yml` or `application.properties`)
- Ensure no conflicts with existing `management.*` configuration
- Preserve existing management endpoint configurations (merge, don't replace)
- Check YAML indentation is correct (2 spaces)
- Verify property names match Spring Boot actuator configuration

**After Steps 0-3, verify setup**:
1. **Check build compiles**:
   ```bash
   ./gradlew build  # or mvn compile
   ```

2. **Verify Prometheus endpoint** (after service starts):
   ```bash
   curl http://localhost:8080/actuator/prometheus
   # Should return Prometheus-formatted metrics
   # Note: Adjust port and context-path based on your application configuration
   ```

3. **Check actuator endpoints**:
   ```bash
   curl http://localhost:8080/actuator
   # Should list available endpoints including /prometheus
   ```

4. **Verify MeterRegistry bean** (if Spring Boot):
   - Spring Boot auto-configures `MeterRegistry` when dependencies are present
   - No additional bean configuration needed
   - Can inject `MeterRegistry` directly into services

### Step 3.5: Configure Infrastructure for Prometheus Scraping (Required for Production)

**CRITICAL: This step is required for metrics to be collected by Datadog Agent in production.**

After code instrumentation, you must configure:
1. **Datadog Java Agent** for system metrics (HTTP, Kafka, JVM, traces)
2. **Docker labels** for Prometheus autodiscovery (business metrics)
3. **Remove OpenTelemetry** if present (replaced by Datadog + Micrometer)

---

#### Step 3.5.1: Configure Datadog Java Agent in Dockerfile

**CRITICAL: Use Datadog Java Agent for system metrics and APM traces. Do NOT use OpenTelemetry.**

The Datadog Java Agent provides **zero-code instrumentation** for:
- HTTP request metrics (latency, throughput, errors)
- Kafka consumer/producer metrics
- JVM metrics (memory, GC, threads)
- Distributed tracing (APM)

**Complete Dockerfile Template:**

```dockerfile
# Download Datadog Java Agent (REQUIRED)
ADD https://dtdg.co/latest-java-tracer /app/dd-java-agent.jar

# Set environment variables for Datadog
ENV DD_SERVICE="your-service-name"
ENV DD_VERSION="latest"
ENV DD_AGENT_HOST="localhost"
ENV DD_TRACE_AGENT_PORT="8126"
ENV DD_DOGSTATSD_PORT="8125"
ENV DD_LOGS_INJECTION="true"
ENV DD_TRACE_ENABLED="true"
ENV DD_RUNTIME_METRICS_ENABLED="true"
ENV DD_PROFILING_ENABLED="true"

# Profiling configuration (memory-bounded)
ENV DD_PROFILING_DDPROF_ENABLED="true"
ENV DD_PROFILING_CPU_ENABLED="true"
ENV DD_PROFILING_WALLCLOCK_ENABLED="true"
ENV DD_PROFILING_ALLOCATION_ENABLED="true"
ENV DD_PROFILING_ALLOCATION_SAMPLE_LIMIT="10000"
ENV DD_PROFILING_HEAP_ENABLED="true"
ENV DD_PROFILING_HEAP_SAMPLE_LIMIT="100"
ENV DD_TRACE_AGENT_MAX_QUEUE_SIZE="1000"
ENV DD_PROFILING_UPLOAD_PERIOD="60"
ENV DD_PROFILING_STACKDEPTH="128"

# CRITICAL: Include dd-java-agent in JAVA_TOOL_OPTIONS
ENV JAVA_TOOL_OPTIONS="-javaagent:/app/dd-java-agent.jar -Dlogging.level.root=info"

# Datadog Auto-Discovery Labels for Prometheus scraping
LABEL com.datadoghq.ad.check_names='["openmetrics"]'
LABEL com.datadoghq.ad.init_configs='[{}]'
LABEL com.datadoghq.ad.instances='[{"openmetrics_endpoint":"http://%%host%%:8080/actuator/prometheus","namespace":"","metrics":[".*"],"tags":["service:your-service-name"]}]'

# IMPORTANT: Do NOT include -javaagent in ENTRYPOINT if using JAVA_TOOL_OPTIONS
ENTRYPOINT ["java", "-jar", "your-app.jar", "--spring.profiles.active=${profile}"]
```

---

#### Step 3.5.2: Remove OpenTelemetry (If Present)

**CRITICAL: If the service has OpenTelemetry instrumentation, it MUST be removed.**

OpenTelemetry is replaced by:
- **Datadog Java Agent** → System metrics + APM traces
- **Micrometer + Prometheus** → Business metrics

**Checklist - Remove these from Dockerfile:**

```dockerfile
# ❌ REMOVE: OpenTelemetry agent download
ADD https://github.com/open-telemetry/opentelemetry-java-instrumentation/releases/download/v2.x.x/opentelemetry-javaagent.jar /app/opentelemetry-javaagent.jar

# ❌ REMOVE: All OTEL_* environment variables
ENV OTEL_SERVICE_NAME="..."
ENV OTEL_EXPORTER_OTLP_ENDPOINT="..."
ENV OTEL_EXPORTER_OTLP_PROTOCOL="..."
ENV OTEL_TRACES_EXPORTER="..."
ENV OTEL_LOGS_EXPORTER="..."
ENV OTEL_METRICS_EXPORTER="..."

# ❌ REMOVE: OpenTelemetry agent from JAVA_TOOL_OPTIONS
ENV JAVA_TOOL_OPTIONS="-javaagent:/app/opentelemetry-javaagent.jar ..."

# ❌ REMOVE: OpenTelemetry agent from ENTRYPOINT
ENTRYPOINT ["java", "-javaagent:/otel/opentelemetry-javaagent.jar", "-jar", ...]
```

**Before/After Example:**

```dockerfile
# ❌ BEFORE (OpenTelemetry - WRONG)
ADD https://github.com/open-telemetry/opentelemetry-java-instrumentation/releases/download/v2.13.3/opentelemetry-javaagent.jar /app/opentelemetry-javaagent.jar
ENV OTEL_SERVICE_NAME="my-service"
ENV OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
ENV JAVA_TOOL_OPTIONS="-javaagent:/app/opentelemetry-javaagent.jar"
ENTRYPOINT ["java", "-jar", "app.jar"]

# ✅ AFTER (Datadog - CORRECT)
ADD https://dtdg.co/latest-java-tracer /app/dd-java-agent.jar
ENV DD_SERVICE="my-service"
ENV DD_AGENT_HOST="localhost"
ENV DD_TRACE_AGENT_PORT="8126"
ENV DD_RUNTIME_METRICS_ENABLED="true"
ENV JAVA_TOOL_OPTIONS="-javaagent:/app/dd-java-agent.jar"
LABEL com.datadoghq.ad.check_names='["openmetrics"]'
LABEL com.datadoghq.ad.init_configs='[{}]'
LABEL com.datadoghq.ad.instances='[{"openmetrics_endpoint":"http://%%host%%:8080/actuator/prometheus","namespace":"","metrics":[".*"],"tags":["service:my-service"]}]'
ENTRYPOINT ["java", "-jar", "app.jar"]
```

---

#### Step 3.5.3: Add Docker Labels for Prometheus Autodiscovery

**Add Docker labels to your Dockerfile** (for reference, but may need to be in ECS task definition):

```dockerfile
# Datadog Auto-Discovery Labels
# These labels enable Datadog to automatically discover and scrape Prometheus metrics
# NOTE: In ECS Fargate, these labels may not be accessible from Dockerfile.
# You MUST also add them to the ECS task definition's dockerLabels section for auto-discovery to work.
LABEL com.datadoghq.ad.check_names='["openmetrics"]'
LABEL com.datadoghq.ad.init_configs='[{}]'
LABEL com.datadoghq.ad.instances='[{"openmetrics_endpoint":"http://%%host%%:8080/actuator/prometheus","namespace":"","metrics":[".*"],"tags":["service:your-service-name"]}]'
```

**Add Docker labels to ECS task definition** (REQUIRED for ECS Fargate):

```json
{
  "name": "your-service-name",
  "dockerLabels": {
    "com.datadoghq.ad.check_names": "[\"openmetrics\"]",
    "com.datadoghq.ad.init_configs": "[{}]",
    "com.datadoghq.ad.instances": "[{\"openmetrics_endpoint\":\"http://%%host%%:8080/actuator/prometheus\",\"namespace\":\"\",\"metrics\":[\".*\"],\"tags\":[\"service:your-service-name\",\"env:${ENV}\"]}]"
  }
}
```

**Important Configuration Details**:
- **Port**: Use the port from `server.port` in `application.yaml` (default: 8080)
- **Path**: Use `/actuator/prometheus` (or `/your-context-path/actuator/prometheus` if context-path is set)
- **Service Tag**: Use your service name (e.g., `goblin-service`)
- **Environment Tag**: Use your environment variable (e.g., `env:prod`, `env:stg`)

**Example for goblin-service** (with context-path `/goblin-service`):
```json
{
  "dockerLabels": {
    "com.datadoghq.ad.check_names": "[\"openmetrics\"]",
    "com.datadoghq.ad.init_configs": "[{}]",
    "com.datadoghq.ad.instances": "[{\"openmetrics_endpoint\":\"http://%%host%%:8080/goblin-service/actuator/prometheus\",\"namespace\":\"\",\"metrics\":[\".*\"],\"tags\":[\"service:goblin-service\",\"env:prod\"]}]"
  }
}
```

**Datadog Agent Container Configuration**:
Ensure the Datadog Agent sidecar has Prometheus scraping enabled:
```json
{
  "name": "datadog-agent",
  "environment": [
    {
      "name": "DD_PROMETHEUS_SCRAPE_ENABLED",
      "value": "true"
    }
  ]
}
```

#### For Kubernetes (Annotations)

Add annotations to your Pod/Deployment:

```yaml
metadata:
  annotations:
    ad.datadoghq.com/your-service-name.check_names: '["openmetrics"]'
    ad.datadoghq.com/your-service-name.init_configs: '[{}]'
    ad.datadoghq.com/your-service-name.instances: '[{"openmetrics_endpoint":"http://%%host%%:8080/actuator/prometheus","namespace":"","metrics":[".*"],"tags":["service:your-service-name","env:prod"]}]'
```

#### Why This Step Was Previously Missing

The skill document focused on **code instrumentation** (Steps 1-12) but didn't explicitly include **infrastructure configuration** for metric collection. This is a critical gap because:

1. **Code instrumentation alone is not sufficient**: Metrics must be scraped by the monitoring agent
2. **Infrastructure configuration is deployment-specific**: ECS vs Kubernetes vs Docker Compose require different approaches
3. **This step is often handled by DevOps/Platform teams**: But developers need to know what to request

**Going Forward**: Always include infrastructure configuration as Step 3.5 when instrumenting services for production deployment.

### Step 4: Choose Your Instrumentation Approach

**CRITICAL: Use the tiered approach. See "Java Instrumentation: Tiered Approach (AOP-First)" section below.**

**Quick Reference:**
- **Tier 1 (80% of cases)**: Use `@Timed`/`@Counted` annotations - zero boilerplate
- **Tier 2 (15% of cases)**: Use `@MeteredOperation` for dynamic labels
- **Tier 3 (4% of cases)**: Use `TimedOperation` for complex conditional metrics
- **Tier 4 (1% of cases)**: Manual instrumentation (legacy only)

**Skip to Step 5 if using Tier 1 or Tier 2.** Only create MetricsUtil/MetricsService if using Tier 3.

---

### Step 4a: Create MetricsService (For Tier 3 Only)

**Only needed if you have complex cases requiring `TimedOperation`. Most services can skip this and use `@Timed` annotations directly.**

Create `src/main/java/tech/vance/your-service/common/util/MetricsUtil.java`:

```java
package tech.vance.yourservice.common.util;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.regex.Pattern;

/**
 * Centralized metrics utility following SOLID principles and DRY.
 * 
 * Single Responsibility: Only handles metrics recording
 * Dependency Inversion: Depends on MeterRegistry abstraction
 * Open/Closed: Extensible without modification
 */
@Component
@Slf4j
@RequiredArgsConstructor  // ✅ Constructor injection (SOLID: Dependency Inversion)
public class MetricsUtil {
    
    private final MeterRegistry meterRegistry;  // ✅ Interface-based dependency (SOLID: Dependency Inversion)
    private static final String SERVICE_NAME = "your-service-name";
    
    // Pre-compiled patterns for performance (compile once, reuse many times)
    private static final Pattern ID_PATTERN = Pattern.compile(
        "/(?:\\d+|\\p{XDigit}{8}-\\p{XDigit}{4}-\\p{XDigit}{4}-\\p{XDigit}{4}-\\p{XDigit}{12}|[a-zA-Z0-9]{24,})"
    );
    private static final Pattern SANITIZE_PATTERN = Pattern.compile("[^a-zA-Z0-9._-]");
    private static final Pattern ENDPOINT_PATTERN = Pattern.compile("[^a-zA-Z0-9/_{}-]");
    
    // L0: Service-level metrics (pre-registered for performance)
    private final Counter serviceRequests = Counter.builder("service.requests.total")
        .description("Total requests processed")
        .tag("service", SERVICE_NAME)
        .register(meterRegistry);
    
    private final Counter serviceErrors = Counter.builder("service.errors.total")
        .description("Total errors")
        .tag("service", SERVICE_NAME)
        .register(meterRegistry);
    
    private final Timer serviceRequestDuration = Timer.builder("service.request.duration")
        .description("Request processing duration")
        .tag("service", SERVICE_NAME)
        .register(meterRegistry);
    
    /**
     * Starts a timer for request-level metrics.
     * Use with try-finally to ensure timer is stopped.
     * 
     * @return Timer.Sample to be stopped in finally block
     */
    public Timer.Sample startRequestTimer() {
        return Timer.start(meterRegistry);
    }
    
    /**
     * Records a service-level request.
     * Thread-safe and can be called concurrently.
     */
    public void recordRequest() {
        serviceRequests.increment();
    }
    
    /**
     * Records a service-level error with sanitized error type.
     * 
     * @param errorType Error type to record (will be sanitized)
     */
    public void recordError(String errorType) {
        try {
            serviceErrors.increment(Tags.of("error_type", sanitizeErrorType(errorType)));
        } catch (Exception e) {
            // ✅ Fail-safe: Metrics never break business logic
            log.warn("Failed to record error metric", e);
        }
    }
    
    /**
     * Stops the request timer and records duration.
     * 
     * @param sample Timer sample from startRequestTimer()
     */
    public void stopRequestTimer(Timer.Sample sample) {
        sample.stop(serviceRequestDuration);
    }
    
    /**
     * Starts a timer for operation-level metrics.
     * 
     * @return Timer.Sample to be stopped in finally block
     */
    public Timer.Sample startOperationTimer() {
        return Timer.start(meterRegistry);
    }
    
    /**
     * Records an operation with outcome (success/error).
     * 
     * @param operation Operation name (will be sanitized)
     * @param outcome Operation outcome ("success" or "error")
     */
    public void recordOperation(String operation, String outcome) {
        try {
            meterRegistry.counter("service.operations.total",
                Tags.of("service", SERVICE_NAME,
                        "operation", sanitizeOperation(operation),
                        "outcome", outcome)).increment();
        } catch (Exception e) {
            log.warn("Failed to record operation metric", e);
        }
    }
    
    /**
     * Stops the operation timer and records duration.
     * 
     * @param sample Timer sample from startOperationTimer()
     * @param operation Operation name (will be sanitized)
     */
    public void stopOperationTimer(Timer.Sample sample, String operation) {
        try {
            sample.stop(Timer.builder("service.operations.duration")
                .description("Operation duration")
                .tag("service", SERVICE_NAME)
                .tag("operation", sanitizeOperation(operation))
                .register(meterRegistry));
        } catch (Exception e) {
            log.warn("Failed to record operation duration", e);
        }
    }
    
    /**
     * Sanitizes operation names for use in metric labels.
     * Limits length to 50 chars and normalizes to lowercase with underscores.
     * 
     * @param operation Operation name to sanitize
     * @return Sanitized operation name
     */
    private String sanitizeOperation(String operation) {
        if (operation == null) {
            return "unknown";
        }
        String sanitized = operation.length() > 50 ? operation.substring(0, 50) : operation;
        sanitized = sanitized.toLowerCase();
        return SANITIZE_PATTERN.matcher(sanitized).replaceAll("_");
    }
    
    /**
     * Sanitizes error type names for use in metric labels.
     * Limits length to 50 chars and normalizes to lowercase with underscores.
     * 
     * @param errorType Error type to sanitize
     * @return Sanitized error type
     */
    private String sanitizeErrorType(String errorType) {
        if (errorType == null) {
            return "unknown";
        }
        String sanitized = errorType.length() > 50 ? errorType.substring(0, 50) : errorType;
        sanitized = sanitized.toLowerCase();
        return SANITIZE_PATTERN.matcher(sanitized).replaceAll("_");
    }
    
    /**
     * Normalizes endpoint paths by replacing ID-like segments with {id}.
     * Prevents cardinality explosion from dynamic path segments.
     * Preserves short resource names like "list", "search", "status".
     * 
     * @param endpoint Endpoint path to sanitize
     * @return Sanitized endpoint path
     */
    public String sanitizeEndpoint(String endpoint) {
        if (endpoint == null) {
            return "unknown";
        }
        
        // Replace numeric IDs, UUIDs, and long alphanumeric strings (24+ chars) with {id}
        String sanitized = ID_PATTERN.matcher(endpoint).replaceAll("/{id}");
        
        // Limit length to prevent excessive cardinality
        if (sanitized.length() > 100) {
            sanitized = sanitized.substring(0, 100);
        }
        
        // Clean up special characters
        return ENDPOINT_PATTERN.matcher(sanitized).replaceAll("_");
    }
}
```

**Key Features Following Java Best Practices**:
- ✅ **SOLID Principles**: Single Responsibility (metrics only), Dependency Inversion (MeterRegistry interface), Open/Closed (extensible)
- ✅ **DRY Principle**: Centralized metric definitions, reusable sanitization functions
- ✅ **Dependency Injection**: Constructor injection via `@RequiredArgsConstructor`
- ✅ **Clean Code**: Meaningful method names, clear JavaDoc comments, self-documenting code
- ✅ **Performance**: Pre-compiled regex patterns, pre-registered metrics
- ✅ **Fail-Safe**: Try-catch blocks prevent metrics failures from breaking business logic
- ✅ **Thread Safety**: Micrometer metrics are thread-safe by design
- ✅ **Standard Java**: Follows Spring Boot conventions, Lombok for boilerplate reduction

### Step 5: Instrument Service Classes

**CRITICAL: Use the AOP-first tiered approach. See "Java Instrumentation: Tiered Approach (AOP-First)" section for full details.**

**⚠️ ABSOLUTE REQUIREMENT: Metrics instrumentation must NEVER modify business logic.**

#### Recommended: Tier 1/2 for Business Operations (Zero Boilerplate)

```java
@Service
public class OrderService {

    // ✅ Tier 1: Business operation worth measuring
    @Timed(value = "business.order.fulfill", description = "Order fulfillment workflow")
    public void fulfillOrder(String orderId) {
        Order order = orderRepository.findById(orderId).orElseThrow();
        inventoryService.commit(order.getItems());
        paymentService.capture(order.getPaymentId());
        order.setStatus(OrderStatus.FULFILLED);
        orderRepository.save(order);
    }

    // ✅ Tier 2: Business operation needing context
    @MeteredOperation(
        value = "business.order.create",
        labels = {"order_type=#{#request.type}", "channel=#{#request.channel}"}
    )
    public Order createOrder(CreateOrderRequest request) {
        return doCreateOrder(request);
    }

    // ❌ NOT instrumented: Simple read - http.server.requests covers the API
    public Order getOrder(String orderId) {
        return orderRepository.findById(orderId)
            .orElseThrow(() -> new OrderNotFoundException(orderId));
    }
}
```

**Benefits of `@Timed`:**
- ✅ Zero metrics code in business methods
- ✅ Automatic exception tracking (adds `exception` tag)
- ✅ Consistent metric naming enforced by annotations
- ✅ Easy to audit (just search for `@Timed`)
- ✅ Can be added/removed without changing business logic

#### When Dynamic Labels Are Needed: Tier 2 with `@MeteredOperation`

```java
@Service
public class PaymentService {

    // ✅ CORRECT: Dynamic labels from parameters via SpEL
    @MeteredOperation(
        value = "payment.process",
        labels = {
            "payment_type=#{#request.type}",
            "acquirer=#{#request.acquirer}"
        }
    )
    public PaymentResult processPayment(PaymentRequest request) {
        return paymentGateway.process(request);  // Pure business logic
    }
}
```

#### When Complex Logic Is Needed: Tier 3 with `TimedOperation`

```java
public BatchResult processBatch(List<Order> orders) {
    try (var op = metrics.timed("batch_processing")
            .tag("processor", "order_processor")) {

        op.batchSize(orders.size());  // Mid-execution update

        int failures = 0;
        for (Order order : orders) {
            try {
                processOrder(order);
            } catch (Exception e) {
                failures++;
            }
        }

        if (failures > 0) {
            op.markFailed("partial_failure");  // Conditional marking
        }

        return new BatchResult(orders.size() - failures, failures);
    }
}
```

#### ❌ AVOID: Manual Try-Catch Instrumentation (Tier 4)

```java
// ❌ AVOID: Verbose, error-prone, mixes concerns
public Order createOrder(CreateOrderRequest request) {
    var operationTimer = metricsUtil.startOperationTimer();

    try {
        Order order = doCreateOrder(request);
        metricsUtil.recordOperation("create_order", "success");
        metricsUtil.stopOperationTimer(operationTimer, "create_order");
        return order;
    } catch (Exception e) {
        metricsUtil.recordOperation("create_order", "error");
        metricsUtil.stopOperationTimer(operationTimer, "create_order");
        throw e;
    }
}

// ✅ PREFER: Use @Timed instead (see above)
```

#### Pattern: External Dependency Calls (Business-Critical Integrations)

Instrument external calls when they represent **business-critical integrations** where you need visibility beyond generic HTTP client metrics:

```java
@Service
@RequiredArgsConstructor
public class PaymentGatewayClient {

    private final RestTemplate restTemplate;

    /**
     * ✅ Tier 2: Payment gateway calls with acquirer breakdown.
     * This is a BUSINESS metric - we need to know payment success rates by acquirer.
     */
    @MeteredOperation(
        value = "business.payment.gateway",
        labels = {
            "acquirer=#{#request.acquirer}",
            "payment_method=#{#request.method}"
        }
    )
    public PaymentResponse processPayment(PaymentRequest request) {
        return restTemplate.postForObject(
            getGatewayUrl(request.getAcquirer()),
            request,
            PaymentResponse.class
        );
    }

    /**
     * ✅ Tier 1: Partner API calls where we care about the integration health.
     */
    @Timed(value = "business.partner.inventory_check", description = "Partner inventory lookup")
    public InventoryResponse checkPartnerInventory(String sku) {
        return restTemplate.getForObject("/partner/inventory/{sku}", InventoryResponse.class, sku);
    }

    /**
     * ❌ NOT instrumented: Generic HTTP calls.
     * Spring's RestTemplate/WebClient metrics or http.client.requests cover this.
     */
    public HealthResponse healthCheck(String serviceUrl) {
        return restTemplate.getForObject(serviceUrl + "/health", HealthResponse.class);
    }
}

### Step 6: Implement Cardinality Management

#### Enhanced Sanitization Functions

```java
// Endpoint sanitization - CRITICAL: Preserves resource names, only replaces IDs
private static final Pattern ID_PATTERN = Pattern.compile(
    "/(?:\\d+|\\p{XDigit}{8}-\\p{XDigit}{4}-\\p{XDigit}{4}-\\p{XDigit}{4}-\\p{XDigit}{12}|[a-zA-Z0-9]{24,})"
);

public String sanitizeEndpoint(String endpoint) {
    if (endpoint == null) return "unknown";
    
    // Replace numeric IDs, UUIDs, and long alphanumeric strings (24+ chars) with {id}
    // Preserves short resource names like "list", "search", "status"
    String sanitized = ID_PATTERN.matcher(endpoint).replaceAll("/{id}");
    
    // Limit length
    if (sanitized.length() > 100) {
        sanitized = sanitized.substring(0, 100);
    }
    
    // Clean up special characters
    return sanitized.replaceAll("[^a-zA-Z0-9/_{}-]", "_");
}

// Operation name sanitization
private String sanitizeOperation(String operation) {
    if (operation == null) return "unknown";
    String sanitized = operation.length() > 50 ? operation.substring(0, 50) : operation;
    return sanitized.replaceAll("[^a-zA-Z0-9._-]", "_").toLowerCase();
}

// Error type sanitization
private String sanitizeErrorType(String errorType) {
    if (errorType == null) return "unknown";
    String sanitized = errorType.length() > 50 ? errorType.substring(0, 50) : errorType;
    return sanitized.replaceAll("[^a-zA-Z0-9._-]", "_").toLowerCase();
}

// For high-cardinality values, use bucketing
private String getBatchSizeBucket(int size) {
    if (size == 1) return "1";
    if (size <= 5) return "2-5";
    if (size <= 10) return "6-10";
    if (size <= 50) return "11-50";
    if (size <= 100) return "51-100";
    return "100+";
}
```

**Important**: The endpoint sanitization regex is conservative - it only replaces:
- Numeric IDs: `/123`, `/456789`
- UUIDs: `/550e8400-e29b-41d4-a716-446655440000`
- Long alphanumeric strings (24+ chars): `/abc123def456ghi789jkl012mno345`

It preserves short resource names like `/list`, `/search`, `/status` to maintain meaningful metric tags.

### Step 7: Choose Metric Types

#### Counters
**Use for**: Cumulative counts (requests, errors, events)
**Characteristics**: Monotonically increasing
**Example**:
```java
Counter.builder("service.events.processed.total")
    .register(meterRegistry)
    .increment();
```

#### Timers
**Use for**: Duration measurements (latency, processing time)
**Characteristics**: Automatically tracks count, sum, max, percentiles
**Example**:
```java
Timer.Sample sample = Timer.start(meterRegistry);
try {
    // Do work
} finally {
    sample.stop(Timer.builder("service.operation.duration")
        .register(meterRegistry));
}
```

#### Gauges
**Use for**: Current values (queue size, active connections, cache size)
**Characteristics**: Can go up or down
**Example**:
```java
meterRegistry.gauge("service.subscriptions.count",
    Tags.of("event_name", eventName),
    subscriptionCount);
```

### Step 8: Naming Conventions

Follow Prometheus naming conventions:
- Use dots (`.`) as separators: `service.operation.duration`
- Use lowercase: `service.events.processed.total`
- Use `_total` suffix for counters: `service.requests.total`
- Use descriptive names: `service.events.processed.total` not `service.events.total`
- Be consistent: Use same prefix for related metrics

**Good Examples**:
- `service.requests.total`
- `service.request.duration`
- `service.errors.total`
- `service.events.processed.total`
- `service.operations.total{operation, outcome, acquirer}` (with optional labels)

**Bad Examples**:
- `requests` (missing service prefix)
- `service_requests` (use dots, not underscores)
- `service.req.total` (abbreviations unclear)
- `service.events` (not descriptive)
- `payment.attempt.updates.total` (should reuse `service.operations.total` with `acquirer` label)

#### ⚠️ CRITICAL: Prefer Labels Over Separate Metrics

**When adding dimensional breakdowns (acquirer, region, feature, etc.), extend existing metrics with tags/labels rather than creating new metric families.**

**❌ BAD - Creating Separate Metrics**:
```java
// Don't do this - creates unnecessary metric families
Counter.builder("payment.attempt.updates.total")
    .tag("service", SERVICE_NAME)
    .tag("acquirer", acquirer)
    .tag("outcome", outcome)
    .register(meterRegistry)
    .increment();
```

**✅ GOOD - Extending Existing Metrics**:
```java
// Extend existing metric with optional tag
Counter.builder("service.operations.total")
    .tag("service", SERVICE_NAME)
    .tag("operation", "update_payment_attempt")
    .tag("outcome", outcome)
    .tag("acquirer", acquirer != null ? acquirer : "unknown")  // Optional dimension
    .register(meterRegistry)
    .increment();

// Helper method in MetricsUtil
public void recordOperationWithAcquirer(String operation, String outcome, String acquirer) {
    String finalAcquirer = acquirer != null ? sanitizeAcquirer(acquirer) : "unknown";
    Counter.builder("service.operations.total")
        .tag("service", SERVICE_NAME)
        .tag("operation", sanitizeOperation(operation))
        .tag("outcome", outcome)
        .tag("acquirer", finalAcquirer)
        .register(meterRegistry)
        .increment();
}

// Default method (for operations without acquirer)
public void recordOperation(String operation, String outcome) {
    recordOperationWithAcquirer(operation, outcome, null);  // Defaults to "unknown"
}
```

**Benefits**:
- Respects metric count limitations
- Maintains consistent metric structure
- Reduces maintenance overhead
- Follows Prometheus best practices (labels for dimensions, not separate metrics)

### Step 9: Error Handling

Always handle metric recording errors gracefully:
```java
try {
    meterRegistry.counter("service.events.processed.total",
        Tags.of("service", "myservice")).increment();
} catch (Exception e) {
    // Log but don't fail the business logic
    log.warn("Failed to record metric", e);
}
```

**CRITICAL**: Wrap all metric operations in try-catch to prevent metric failures from breaking business logic.

### Step 10: Testing Metrics

Test that metrics are recorded correctly:
```java
@Test
void testEventProcessingMetrics() {
    // Arrange
    MeterRegistry registry = new SimpleMeterRegistry();
    MetricsUtil metricsUtil = new MetricsUtil(registry);
    Service service = new Service(metricsUtil);
    
    // Act
    service.processEvent(event);
    
    // Assert
    Counter counter = registry.counter("service.events.processed.total",
        "service", "myservice");
    assertEquals(1, counter.count());
    
    Timer timer = registry.timer("service.event.duration",
        "service", "myservice");
    assertEquals(1, timer.count());
    assertTrue(timer.totalTime(TimeUnit.MILLISECONDS) > 0);
}
```

### Step 11: Verify Metrics Are Exposed

**After implementing metrics, verify they are exposed correctly:**

1. **Start the service**:
   ```bash
   ./gradlew bootRun  # or mvn spring-boot:run
   ```

2. **Check Prometheus endpoint**:
   ```bash
   curl http://localhost:8080/actuator/prometheus | grep your_metric_name
   ```

3. **Verify metric format**:
   - Metrics should be in Prometheus format: `metric_name{tag1="value1",tag2="value2"} value`
   - Check that tags are present and correct
   - Verify metric values are being recorded

4. **Check for common issues**:
   - Metric names follow Prometheus naming (dots in code, underscores in output)
   - Tags are properly formatted
   - No special characters in tag values
   - Metrics are being incremented/recorded

5. **Test metric recording**:
   - Trigger operations that should record metrics
   - Verify metrics appear in Prometheus endpoint
   - Check metric values increase as expected

### Step 12: Create Observability Documentation

**After implementing metrics, create documentation for the team:**

1. **Create `observability/README.md`** with:
   - Overview of metrics instrumentation
   - Metric hierarchy (L0, L1, L2)
   - Metric naming conventions
   - Tag descriptions and values
   - Query examples for common use cases
   - SLO thresholds and alerting recommendations
   - Troubleshooting guide

2. **Document all metrics**:
   - List all L0 metrics with descriptions
   - List all L1 metrics with tags and descriptions
   - Include error types tracked
   - Document metric types (Counter, Timer, Gauge)

3. **Include usage examples**:
   - How to use MetricsUtil
   - Code examples for common patterns
   - Prometheus query examples
   - Dashboard creation guidance

4. **Document access**:
   - Prometheus endpoint URL
   - How to access metrics
   - Health check endpoints

**Example structure**:
```markdown
# Observability & Metrics Documentation

## Overview
[Service metrics overview]

## Metric Hierarchy
### L0 Metrics
[Service-level metrics]

### L1 Metrics
[Feature-level metrics]

## Accessing Metrics
[Endpoint URLs and access methods]

## Metric Naming Conventions
[Naming rules and examples]

## Query Examples
[Prometheus queries]

## Troubleshooting
[Common issues and solutions]
```

## CRITICAL: Do Not Change Functional Logic

**⚠️ ABSOLUTE REQUIREMENT: Metrics instrumentation must NEVER modify business logic.**

### Rules for Instrumentation

1. **ONLY Add Metrics Calls**
   - Add `metrics.record*()` calls
   - Add `var timerSample = metrics.start()` and `metrics.recordDuration()` calls
   - Add try-catch blocks ONLY for metrics recording
   - Do NOT modify existing business logic, validation, error handling, or control flow

2. **Preserve All Existing Behavior**
   - Method signatures must remain identical
   - Return values must be identical
   - Exception types and messages must be identical
   - Side effects must be identical
   - All existing try-catch blocks must be preserved

3. **Wrap, Don't Replace**
   - Wrap existing code in try-catch for metrics
   - Preserve all existing exception handling
   - Do NOT change exception types
   - Do NOT add new validation or early returns

4. **Verification Checklist**
   - ✅ All existing business logic preserved
   - ✅ All existing exception handling preserved
   - ✅ Method signatures unchanged
   - ✅ Return values unchanged
   - ✅ Only metrics calls added
   - ✅ No new validation or early returns
   - ✅ No changes to control flow

### Example: Correct vs Incorrect

**✅ CORRECT - No Logic Changes (Java)**:
```java
// BEFORE
public Order createOrder(CreateOrderRequest request) {
    Order order = doCreateOrder(request);
    return order;
}

// AFTER - Only metrics added
public Order createOrder(CreateOrderRequest request) {
    var timerSample = metrics.start();
    try {
        Order order = doCreateOrder(request); // UNCHANGED
        metrics.recordOrderCreate(orderType, timerSample, true); // ADDED
        return order; // UNCHANGED
    } catch (Exception e) {
        metrics.recordOrderCreate(orderType, timerSample, false); // ADDED
        throw e; // PRESERVED - same exception type
    }
}
```

**✅ CORRECT - No Logic Changes (Go)**:
```go
// BEFORE
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {
    order, err := s.doCreateOrder(ctx, req)
    if err != nil {
        return nil, err
    }
    return order, nil
}

// AFTER - Only metrics added
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {
    stopTimer := metrics.StartTimer("create_order")
    defer stopTimer()
    
    // Business logic - UNCHANGED
    order, err := s.doCreateOrder(ctx, req)
    
    // Metrics recording - ADDED
    if err != nil {
        metrics.RecordOperation("create_order", "error") // ADDED
        return nil, err // PRESERVED - same return value and error
    }
    
    metrics.RecordOperation("create_order", "success") // ADDED
    return order, nil // PRESERVED - same return value
}
```

**❌ INCORRECT - Logic Changed (Java)**:
```java
// ❌ WRONG: Changed exception handling
public Order createOrder(CreateOrderRequest request) {
    try {
        Order order = doCreateOrder(request);
        metrics.recordOrderCreate(orderType, timerSample, true);
        return order;
    } catch (IllegalArgumentException e) {
        // ❌ WRONG: Changed exception type
        throw new AppServerException(e);
    }
}

// ❌ WRONG: Added validation
public Order createOrder(CreateOrderRequest request) {
    if (request == null) {
        metrics.recordOrderCreate(orderType, timerSample, false);
        return null; // ❌ WRONG: Changed behavior
    }
    // ...
}

// ❌ WRONG: Changed method signature
public Order createOrder(CreateOrderRequest request, MeterRegistry registry) {
    // ❌ WRONG: Added parameter
}
```

**❌ INCORRECT - Logic Changed (Go)**:
```go
// ❌ WRONG: Changed error handling
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {
    order, err := s.doCreateOrder(ctx, req)
    if err != nil {
        metrics.RecordOperation("create_order", "error")
        // ❌ WRONG: Changed error type
        return nil, fmt.Errorf("failed to create order: %w", err)
    }
    // ...
}

// ❌ WRONG: Added early return
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest) (*Order, error) {
    if req == nil {
        metrics.RecordOperation("create_order", "error")
        return nil, nil // ❌ WRONG: Changed behavior (original might return error)
    }
    // ...
}

// ❌ WRONG: Changed function signature
func (s *OrderService) CreateOrder(ctx context.Context, req *CreateOrderRequest, registry *prometheus.Registry) (*Order, error) {
    // ❌ WRONG: Added parameter, breaks existing callers
}
```

## Best Practices

### 1. Start with L0, Add L1/L2 as Needed
- Begin with service-level metrics (L0)
- Add feature-level metrics (L1) when you need to debug specific features
- Add instance-level metrics (L2) only when debugging specific instances

### 2. Monitor Cardinality
- Set alerts on metric series count
- Use sanitization to limit cardinality
- Bucket continuous values
- Remove high-cardinality tags when not needed

### 3. Use Consistent Tagging
- Always include `service` tag
- Use consistent tag names across all metrics
- Document tag values and their meanings

### 4. Track Business Metrics
- Don't just track technical metrics
- Track business outcomes (orders processed, users created, etc.)
- Align metrics with business goals

### 5. Measure What Matters
- Focus on metrics that help detect issues
- Avoid metrics that are "nice to have" but not actionable
- Every metric should answer a question

### 6. Use Percentiles for Latency
- Track p50, p95, p99 for latency
- p50 (median) shows typical performance
- p95/p99 show tail latency (user experience)

### 7. Track Error Rates, Not Just Counts
- Error rate = errors / total requests
- Error rate helps identify issues faster than absolute counts
- Use formulas in dashboards: `errors / (errors + successes)`

## Common Pitfalls to Avoid

1. **Cardinality Explosion**: Adding high-cardinality tags without sanitization
2. **Too Many Metrics**: Tracking everything instead of what matters
3. **Inconsistent Naming**: Different naming patterns across services
4. **Missing Error Metrics**: Only tracking success, not failures
5. **No Duration Metrics**: Only tracking counts, not latency
6. **Over-Tagging**: Adding tags that don't help debugging
7. **Under-Tagging**: Missing tags that would help identify issues
8. **Unbounded Metric Name Growth**: Never build metric names dynamically (e.g. `metric." + status + ".total`). Use a single metric name with a bounded tag like `{status="completed"}`.
9. **Missing Timer Declarations**: Always declare `Timer.Sample` variables at method start, even if conditionally used
10. **Unused Timer Parameters**: Don't pass parameters to `start*Timer()` methods - only pass them to `stop*Timer()` methods

## Guardrail: Custom Metrics Budget (< 40 metric names)

For production services, keep **custom metric names under 40** (excluding JVM/Spring/DB/client library defaults).

Enforcement rules:
- Prefer **one metric name + bounded tag** instead of multiple metric names.
  - ✅ `service.payment.refund.total{outcome="initiated|completed|failed"}`
  - ❌ `service.payment.refund.initiated.total`, `service.payment.refund.completed.total`, ...
- Avoid "nice-to-have" duplicates:
  - If you already have `*.status_transition.total`, avoid also adding `*.completed.total` unless it's a top-level business outcome.
- If you add a new custom metric, delete or consolidate an existing one to stay within budget.

## Java Instrumentation: Tiered Approach (AOP-First)

**CRITICAL: Focus on BUSINESS LOGIC metrics. Infrastructure metrics (HTTP latency, DB queries) are already handled automatically.**

### What's Already Instrumented (Don't Duplicate)

Spring Boot Actuator + Micrometer automatically provide:

| Metric | What It Covers | You Get For Free |
|--------|----------------|------------------|
| `http.server.requests` | All HTTP endpoints (REST APIs) | Latency, count, status codes, URI |
| `jdbc.*` / `hikaricp.*` | Database connections & queries | Connection pool, query timing |
| `jvm.*` | JVM health | Memory, GC, threads |
| `spring.data.repository.*` | Spring Data repositories | Repository method timing |
| `resilience4j.*` | Circuit breakers (if configured) | State, calls, failures |

**⚠️ DO NOT add `@Timed` to REST controllers just to measure API latency - it's already measured.**

### What You SHOULD Instrument (Business Logic)

Focus your instrumentation on:

| Category | Examples | Why It Matters |
|----------|----------|----------------|
| **Business Operations** | `order.create`, `payment.process`, `refund.initiate` | Business KPIs, SLOs |
| **Workflow Steps** | `workflow.state_transition`, `approval.process` | Process visibility |
| **External Integrations** | `acquirer.call`, `partner.api` | Dependency health |
| **Domain-Specific Outcomes** | `payment.declined`, `inventory.reserved` | Business impact |
| **Batch Processing** | `batch.orders.processed`, `sync.records` | Throughput tracking |

### Decision Framework: Which Tier to Use?

| Question | If YES → | If NO → |
|----------|----------|---------|
| Is this a business operation with domain meaning? | Continue ↓ | Don't instrument (use built-in) |
| Need custom labels based on method parameters? | **Tier 2: `@MeteredOperation`** | **Tier 1: `@Timed`** |
| Need conditional metrics or mid-operation updates? | **Tier 3: `TimedOperation`** | Use Tier 1 or 2 |
| Legacy code or framework limitations? | **Tier 4: Manual** | Use Tier 1-3 |

### Comparison of Approaches

| Aspect | Tier 1: `@Timed` | Tier 2: `@MeteredOperation` | Tier 3: `TimedOperation` | Tier 4: Manual |
|--------|------------------|----------------------------|-------------------------|----------------|
| **Boilerplate** | Zero | Zero | Minimal (3-5 lines) | High (15-20 lines) |
| **Business Logic Separation** | ✅ Complete | ✅ Complete | ⚠️ Partial | ❌ Mixed |
| **Custom Labels** | Static only | ✅ Dynamic | ✅ Dynamic | ✅ Dynamic |
| **Conditional Metrics** | ❌ No | ⚠️ Limited | ✅ Yes | ✅ Yes |
| **Framework Required** | Spring AOP | Spring AOP | None | None |
| **Debuggability** | ⚠️ Implicit | ⚠️ Implicit | ✅ Explicit | ✅ Explicit |
| **Best For** | Simple business ops | Ops needing context | Complex/conditional | Legacy only |

---

## Tier 1: AOP-Based `@Timed` and `@Counted` (Recommended Default)

**Use this for 80% of your instrumentation needs.** Zero boilerplate, complete separation of concerns.

### Step 1: Enable Micrometer AOP Support

Add the `TimedAspect` bean to enable `@Timed` annotation processing:

```java
package tech.vance.yourservice.config;

import io.micrometer.core.aop.CountedAspect;
import io.micrometer.core.aop.TimedAspect;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Enables Micrometer AOP annotations (@Timed, @Counted) for declarative metrics.
 *
 * This follows the principle: "Metrics are a cross-cutting concern that should be
 * separated from business logic using AOP."
 */
@Configuration
public class MetricsAopConfig {

    /**
     * Enables @Timed annotation support.
     * Methods annotated with @Timed will automatically record:
     * - Execution time (histogram)
     * - Call count
     * - Exception count
     */
    @Bean
    public TimedAspect timedAspect(MeterRegistry registry) {
        return new TimedAspect(registry);
    }

    /**
     * Enables @Counted annotation support.
     * Methods annotated with @Counted will automatically record call counts.
     */
    @Bean
    public CountedAspect countedAspect(MeterRegistry registry) {
        return new CountedAspect(registry);
    }
}
```

### Step 2: Annotate Business Logic Methods (NOT Controllers)

**Focus on service layer and domain operations. Controllers are already measured by `http.server.requests`.**

```java
@Service
@RequiredArgsConstructor
public class OrderService {

    private final OrderRepository orderRepository;
    private final InventoryService inventoryService;
    private final PaymentService paymentService;

    /**
     * Business operation: Creating an order involves multiple steps.
     * This metric tells us how long the BUSINESS LOGIC takes,
     * separate from HTTP overhead.
     *
     * @Timed automatically records:
     * - business_order_create_seconds_count (counter)
     * - business_order_create_seconds_sum (total time)
     * - business_order_create_seconds_max (max time)
     * - business_order_create_seconds histogram buckets
     */
    @Timed(value = "business.order.create", description = "Order creation business logic")
    public Order createOrder(CreateOrderRequest request) {
        // Business logic - reserve inventory, validate, persist
        inventoryService.reserve(request.getItems());
        Order order = Order.from(request);
        return orderRepository.save(order);
    }

    /**
     * Business operation: Order fulfillment workflow.
     * NOT the same as GET /orders/{id} latency (that's http.server.requests).
     */
    @Timed(value = "business.order.fulfill", description = "Order fulfillment processing")
    public void fulfillOrder(String orderId) {
        Order order = orderRepository.findById(orderId).orElseThrow();
        inventoryService.commit(order.getItems());
        order.setStatus(OrderStatus.FULFILLED);
        orderRepository.save(order);
    }

    /**
     * Business event: Track cancellations as a business metric.
     * @Counted is appropriate when you only care about count, not duration.
     */
    @Counted(value = "business.order.cancelled", description = "Order cancellation events")
    public void cancelOrder(String orderId) {
        Order order = orderRepository.findById(orderId).orElseThrow();
        inventoryService.release(order.getItems());
        order.setStatus(OrderStatus.CANCELLED);
        orderRepository.save(order);
    }
}
```

### Step 3: What NOT to Instrument

```java
@RestController
@RequestMapping("/api/orders")
@RequiredArgsConstructor
public class OrderController {

    private final OrderService orderService;

    /**
     * ❌ DON'T add @Timed here - http.server.requests already covers this!
     *
     * Spring Boot Actuator automatically records:
     * - http_server_requests_seconds{uri="/api/orders", method="POST", status="201"}
     */
    @PostMapping
    public ResponseEntity<Order> createOrder(@RequestBody CreateOrderRequest request) {
        Order order = orderService.createOrder(request);
        return ResponseEntity.created(URI.create("/api/orders/" + order.getId())).body(order);
    }

    /**
     * ❌ DON'T instrument simple CRUD reads at controller level.
     * The http.server.requests metric already tracks GET /api/orders/{id} latency.
     */
    @GetMapping("/{id}")
    public Order getOrder(@PathVariable String id) {
        return orderService.getOrder(id);
    }
}
```

### When TO Use `@Timed` on Controllers

Only add controller-level `@Timed` if you need **custom tags not available in http.server.requests**:

```java
/**
 * ✅ OK: Adding custom business context tag that http.server.requests doesn't have.
 * But prefer doing this in the service layer instead.
 */
@PostMapping("/bulk")
@Timed(value = "api.orders.bulk", extraTags = {"bulk_operation", "true"})
public List<Order> createBulkOrders(@RequestBody List<CreateOrderRequest> requests) {
    return requests.stream().map(orderService::createOrder).toList();
}
```

### Benefits of Tier 1

1. **Zero Business Logic Pollution**: Methods contain only business logic
2. **Automatic Exception Tracking**: Failed calls are automatically tagged with `exception=<ExceptionClass>`
3. **Consistent Naming**: Enforced by annotation values
4. **Easy to Add/Remove**: Just add/remove annotation - no code changes
5. **IDE Support**: Annotations are visible and searchable
6. **Testability**: Business logic is completely isolated from metrics

### Limitations of Tier 1

- **Static labels only**: Cannot add labels based on method parameters (e.g., `order_type`)
- **No conditional metrics**: Metrics are always recorded
- **No outcome-based labels**: Cannot distinguish success/failure beyond exceptions

---

## Tier 2: Custom `@MeteredOperation` with AOP (For Dynamic Labels)

**Use when you need labels derived from method parameters or return values.**

### Step 1: Create Custom Annotation

```java
package tech.vance.yourservice.common.metrics;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a method for business-level metrics instrumentation.
 *
 * Unlike @Timed, this allows dynamic labels from method parameters.
 * The aspect can extract values from parameters using SpEL expressions.
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface MeteredOperation {

    /** Operation name for the metric (e.g., "order.create") */
    String value();

    /** Description for the metric */
    String description() default "";

    /**
     * SpEL expressions to extract label values from parameters.
     * Format: "labelName=#{expression}"
     * Examples:
     *   - "order_type=#{#request.orderType}"
     *   - "customer_tier=#{#request.customer.tier}"
     *   - "batch_size=#{#items.size()}"
     */
    String[] labels() default {};

    /** Whether to record outcome (success/error) as a label */
    boolean recordOutcome() default true;
}
```

### Step 2: Create the Aspect

```java
package tech.vance.yourservice.common.metrics;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.context.expression.MethodBasedEvaluationContext;
import org.springframework.core.DefaultParameterNameDiscoverer;
import org.springframework.core.ParameterNameDiscoverer;
import org.springframework.expression.EvaluationContext;
import org.springframework.expression.ExpressionParser;
import org.springframework.expression.spel.standard.SpelExpressionParser;
import org.springframework.stereotype.Component;

import java.lang.reflect.Method;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * AOP Aspect that handles @MeteredOperation annotations.
 *
 * Provides:
 * - Automatic timing (histogram)
 * - Counter with outcome (success/error)
 * - Dynamic labels from method parameters via SpEL
 *
 * Following SRE principles:
 * - Latency: Timer records duration
 * - Traffic: Counter tracks call rate
 * - Errors: Outcome label distinguishes success/failure
 */
@Aspect
@Component
@RequiredArgsConstructor
@Slf4j
public class MeteredOperationAspect {

    private static final String SERVICE_NAME = "your-service-name";
    private static final Pattern LABEL_PATTERN = Pattern.compile("([^=]+)=#\\{(.+)}");

    private final MeterRegistry meterRegistry;
    private final ExpressionParser parser = new SpelExpressionParser();
    private final ParameterNameDiscoverer parameterNameDiscoverer = new DefaultParameterNameDiscoverer();

    @Around("@annotation(meteredOperation)")
    public Object measureOperation(ProceedingJoinPoint joinPoint, MeteredOperation meteredOperation) throws Throwable {
        String operationName = sanitizeOperation(meteredOperation.value());
        Tags baseTags = buildTags(joinPoint, meteredOperation);

        Timer.Sample sample = Timer.start(meterRegistry);
        String outcome = "success";
        String exceptionType = null;

        try {
            Object result = joinPoint.proceed();
            return result;
        } catch (Throwable t) {
            outcome = "error";
            exceptionType = t.getClass().getSimpleName();
            throw t;
        } finally {
            // Record timer
            Tags timerTags = baseTags;
            sample.stop(Timer.builder("service.operations.duration")
                .description(meteredOperation.description())
                .tags(timerTags)
                .tag("service", SERVICE_NAME)
                .tag("operation", operationName)
                .register(meterRegistry));

            // Record counter with outcome
            if (meteredOperation.recordOutcome()) {
                Tags counterTags = baseTags.and("outcome", outcome);
                if (exceptionType != null) {
                    counterTags = counterTags.and("exception", sanitizeErrorType(exceptionType));
                }
                meterRegistry.counter("service.operations.total",
                    counterTags
                        .and("service", SERVICE_NAME)
                        .and("operation", operationName))
                    .increment();
            }
        }
    }

    private Tags buildTags(ProceedingJoinPoint joinPoint, MeteredOperation meteredOperation) {
        Tags tags = Tags.empty();

        if (meteredOperation.labels().length == 0) {
            return tags;
        }

        try {
            Method method = ((MethodSignature) joinPoint.getSignature()).getMethod();
            EvaluationContext context = new MethodBasedEvaluationContext(
                null, method, joinPoint.getArgs(), parameterNameDiscoverer);

            for (String labelDef : meteredOperation.labels()) {
                Matcher matcher = LABEL_PATTERN.matcher(labelDef);
                if (matcher.matches()) {
                    String labelName = matcher.group(1).trim();
                    String expression = matcher.group(2).trim();

                    try {
                        Object value = parser.parseExpression(expression).getValue(context);
                        String labelValue = value != null ? sanitizeLabel(value.toString()) : "unknown";
                        tags = tags.and(labelName, labelValue);
                    } catch (Exception e) {
                        log.warn("Failed to evaluate SpEL expression '{}' for label '{}': {}",
                            expression, labelName, e.getMessage());
                        tags = tags.and(labelName, "unknown");
                    }
                }
            }
        } catch (Exception e) {
            log.warn("Failed to build tags for @MeteredOperation: {}", e.getMessage());
        }

        return tags;
    }

    private String sanitizeOperation(String operation) {
        if (operation == null) return "unknown";
        String sanitized = operation.length() > 50 ? operation.substring(0, 50) : operation;
        return sanitized.toLowerCase().replaceAll("[^a-z0-9._-]", "_");
    }

    private String sanitizeLabel(String value) {
        if (value == null) return "unknown";
        String sanitized = value.length() > 30 ? value.substring(0, 30) : value;
        return sanitized.toLowerCase().replaceAll("[^a-z0-9._-]", "_");
    }

    private String sanitizeErrorType(String errorType) {
        if (errorType == null) return "unknown";
        String sanitized = errorType.length() > 50 ? errorType.substring(0, 50) : errorType;
        return sanitized.toLowerCase().replaceAll("[^a-z0-9._-]", "_");
    }
}
```

### Step 3: Use the Custom Annotation

```java
@Service
@RequiredArgsConstructor
public class PaymentService {

    private final PaymentGateway paymentGateway;

    /**
     * Dynamic labels extracted from method parameters:
     * - payment_type from request.getType()
     * - acquirer from request.getAcquirer()
     */
    @MeteredOperation(
        value = "payment.process",
        description = "Process payment request",
        labels = {
            "payment_type=#{#request.type}",
            "acquirer=#{#request.acquirer}"
        }
    )
    public PaymentResult processPayment(PaymentRequest request) {
        // Pure business logic - dynamic labels handled by aspect
        return paymentGateway.process(request);
    }

    @MeteredOperation(
        value = "payment.refund",
        labels = {"acquirer=#{#payment.acquirer}", "reason=#{#reason}"}
    )
    public RefundResult refundPayment(Payment payment, String reason) {
        return paymentGateway.refund(payment, reason);
    }
}
```

### Resulting Metrics

```prometheus
# Counter with dynamic labels
service_operations_total{service="your-service",operation="payment.process",payment_type="card",acquirer="stripe",outcome="success"} 150
service_operations_total{service="your-service",operation="payment.process",payment_type="card",acquirer="stripe",outcome="error",exception="paymentdeclinedexception"} 12

# Timer/Histogram
service_operations_duration_seconds_bucket{service="your-service",operation="payment.process",payment_type="card",acquirer="stripe",le="0.1"} 120
service_operations_duration_seconds_bucket{service="your-service",operation="payment.process",payment_type="card",acquirer="stripe",le="0.5"} 145
```

---

## Tier 3: Fluent `TimedOperation` (For Complex Cases)

**Use when you need conditional metrics, mid-operation tagging, or batch tracking that cannot be expressed declaratively.**

### When to Use Tier 3

- **Conditional metrics**: Record different metrics based on runtime conditions
- **Mid-operation updates**: Add batch size or other values discovered during execution
- **Custom failure reasons**: Distinguish between different failure modes
- **Non-Spring environments**: When AOP is not available
- **Complex control flow**: When annotation placement doesn't match metric boundaries

### TimedOperation Implementation

Add this to your metrics service class:

```java
@Service
@RequiredArgsConstructor
public class MetricsService {
    private static final String PREFIX = "service_";
    private final MeterRegistry registry;

    // Fluent entry points
    public CounterBuilder count(String name) { return new CounterBuilder(name); }
    public TimedOperation timed(String name) { return new TimedOperation(name); }
    public GaugeSetter gauge(String name) { return new GaugeSetter(name); }

    // AutoCloseable TimedOperation - use with try-with-resources
    public class TimedOperation implements AutoCloseable {
        private final String name;
        private final long startTime;
        private Tags tags = Tags.empty();
        private boolean success = true;
        private String failureReason;
        private Integer batchSize;

        public TimedOperation(String name) {
            this.name = name;
            this.startTime = System.currentTimeMillis();
        }

        public TimedOperation tag(String key, String value) {
            if (value != null) {
                this.tags = tags.and(key, sanitize(value));
            }
            return this;
        }

        public TimedOperation tag(String key, Enum<?> value) {
            return tag(key, value != null ? value.name().toLowerCase() : null);
        }

        public TimedOperation batchSize(int size) {
            this.batchSize = size;
            return this;
        }

        public void markFailed(String reason) {
            this.success = false;
            this.failureReason = reason;
        }

        @Override
        public void close() {
            long duration = System.currentTimeMillis() - startTime;
            String outcome = success ? "success" : "failure";

            // Record counter with outcome
            Tags counterTags = tags.and("outcome", outcome);
            if (!success && failureReason != null) {
                counterTags = counterTags.and("failure_reason", sanitize(failureReason));
            }
            registry.counter(PREFIX + name + "_total", counterTags).increment();

            // Record timer
            registry.timer(PREFIX + name + "_duration_ms", tags)
                .record(duration, TimeUnit.MILLISECONDS);

            // Record batch size if set
            if (batchSize != null) {
                registry.gauge(PREFIX + name + "_batch_size", tags, batchSize);
            }
        }

        private String sanitize(String value) {
            if (value == null) return "unknown";
            return value.length() > 50 ? value.substring(0, 50) : value;
        }
    }

    // Fluent counter builder
    public class CounterBuilder {
        private final String name;
        private Tags tags = Tags.empty();

        public CounterBuilder(String name) { this.name = name; }

        public CounterBuilder tag(String key, String value) {
            if (value != null) {
                this.tags = tags.and(key, value);
            }
            return this;
        }

        public CounterBuilder tag(String key, Enum<?> value) {
            return tag(key, value != null ? value.name().toLowerCase() : null);
        }

        public void increment() {
            registry.counter(PREFIX + name + "_total", tags).increment();
        }

        public void increment(double amount) {
            registry.counter(PREFIX + name + "_total", tags).increment(amount);
        }
    }

    // Fluent gauge setter
    public class GaugeSetter {
        private final String name;
        private Tags tags = Tags.empty();

        public GaugeSetter(String name) { this.name = name; }

        public GaugeSetter tag(String key, String value) {
            if (value != null) {
                this.tags = tags.and(key, value);
            }
            return this;
        }

        public void set(Number value) {
            registry.gauge(PREFIX + name, tags, value);
        }
    }
}
```

### Usage Examples

**Basic Timed Operation:**
```java
public void processOrder(Order order) {
    try (TimedOperation op = metrics.timed("order_processing")
            .tag("order_type", order.getType())
            .tag("client", order.getClient())) {

        doProcessOrder(order);

    } // Auto-records: counter, timer, outcome=success
}
```

**With Batch Size Tracking:**
```java
public void processBatch(List<Transaction> transactions) {
    try (TimedOperation op = metrics.timed("batch_processing")
            .tag("processor", getProcessorName())) {

        op.batchSize(transactions.size());  // Record batch size

        for (Transaction txn : transactions) {
            process(txn);
        }

    } // Auto-records: counter, timer, batch_size gauge
}
```

**With Failure Tracking:**
```java
public void processPayment(Payment payment) {
    try (TimedOperation op = metrics.timed("payment_processing")
            .tag("partner", payment.getPartner())) {

        PaymentResult result = doProcess(payment);

        if (!result.isSuccess()) {
            op.markFailed(result.getErrorCode());  // Mark as failed with reason
        }

    } catch (Exception e) {
        // Exception auto-marks as failure
        throw e;
    }
}
```

**Fluent Counter:**
```java
metrics.count("payout_skipped")
    .tag("processor", processorName)
    .tag("partner", partner)
    .tag("reason", "rate_limit")
    .increment();
```

**Fluent Gauge:**
```java
metrics.gauge("queued_transactions")
    .tag("router", getRouterName())
    .set(transactions.size());
```

### Migration Guide: Old Pattern → TimedOperation

**Before (verbose, error-prone):**
```java
public void execute() {
    Date startTime = new Date();
    boolean success = true;
    try {
        List<Transaction> txns = fetchTransactions();
        metricsUtil.recordQueuedTransactions(getRouterName(), txns.size());

        for (Transaction txn : txns) {
            routeTransaction(txn);
        }

        metricsUtil.recordRouterExecution(getRouterName(), txns.size(), startTime, true);
    } catch (Exception e) {
        success = false;
        metricsUtil.recordRouterExecution(getRouterName(), 0, startTime, false);
        metricsUtil.recordException(getRouterName(), "execute", e.getClass().getSimpleName());
        throw e;
    }
}
```

**After (clean, auto-managed):**
```java
public void execute() {
    try (TimedOperation op = metrics.timed("router_execution")
            .tag("router", getRouterName())) {

        List<Transaction> txns = fetchTransactions();
        op.batchSize(txns.size());

        metrics.gauge("queued_transactions")
            .tag("router", getRouterName())
            .set(txns.size());

        for (Transaction txn : txns) {
            routeTransaction(txn);
        }

    } catch (Exception e) {
        metrics.count("exception")
            .tag("component", getRouterName())
            .tag("operation", "execute")
            .tag("exception_type", e.getClass().getSimpleName())
            .increment();
        throw e;
    }
}
```

### Tier 3 Benefits Summary

1. **Cleaner Code**: 60-70% less metrics boilerplate vs manual
2. **Guaranteed Recording**: try-with-resources ensures metrics are always recorded
3. **Consistent Tagging**: Fluent API enforces consistent tag patterns
4. **Type Safety**: Tag methods accept enums directly
5. **Reduced Errors**: No more forgotten timer stops or missed error paths
6. **Self-Documenting**: Fluent chain reads like documentation

---

## Tier 4: Manual Instrumentation (Legacy/Edge Cases Only)

**Use ONLY when Tiers 1-3 are not applicable. This should represent < 1% of your instrumentation.**

### When Tier 4 is Necessary

- **Framework limitations**: Pre-Spring Boot 2.x or non-Spring environments without AOP
- **Third-party library callbacks**: Instrumenting code you don't control
- **Performance-critical hot paths**: When AOP overhead is measurable (rare)
- **Gradual migration**: Intermediate step when refactoring to Tier 1-3

### Manual Pattern (Avoid When Possible)

```java
// ❌ AVOID: Manual instrumentation is verbose and error-prone
public Order createOrder(CreateOrderRequest request) {
    var operationTimer = metricsUtil.startOperationTimer();

    try {
        // Business logic
        Order order = doCreateOrder(request);

        // Metrics recording
        metricsUtil.recordOperation("create_order", "success");
        metricsUtil.stopOperationTimer(operationTimer, "create_order");

        return order;

    } catch (Exception e) {
        metricsUtil.recordOperation("create_order", "error");
        metricsUtil.stopOperationTimer(operationTimer, "create_order");
        throw e;
    }
}

// ✅ PREFER: Tier 1 with @Timed - zero boilerplate
@Timed(value = "service.order.create", extraTags = {"layer", "service"})
public Order createOrder(CreateOrderRequest request) {
    return doCreateOrder(request);
}
```

### Migration Path: Manual → Declarative

If you have existing manual instrumentation, migrate to declarative:

```java
// Step 1: Add @Timed annotation
// Step 2: Remove manual timer code
// Step 3: Test that metrics still work
// Step 4: Remove MetricsUtil injection if no longer needed

// BEFORE (Tier 4)
@Service
@RequiredArgsConstructor
public class OrderService {
    private final MetricsUtil metricsUtil;

    public Order createOrder(CreateOrderRequest request) {
        var timer = metricsUtil.startOperationTimer();
        try {
            Order order = doCreateOrder(request);
            metricsUtil.recordOperation("create_order", "success");
            return order;
        } catch (Exception e) {
            metricsUtil.recordOperation("create_order", "error");
            throw e;
        } finally {
            metricsUtil.stopOperationTimer(timer, "create_order");
        }
    }
}

// AFTER (Tier 1)
@Service
public class OrderService {

    @Timed(value = "service.order.create")
    public Order createOrder(CreateOrderRequest request) {
        return doCreateOrder(request);
    }
}
```

---

## Complete Example: Business Logic Instrumentation

This example shows a realistic service focused on **business metrics** (not duplicating HTTP metrics):

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class OrderProcessingService {

    private final OrderRepository orderRepository;
    private final PaymentService paymentService;
    private final InventoryService inventoryService;
    private final MetricsService metrics; // Only needed for Tier 3

    /**
     * Tier 1: Simple business operation.
     * Measures the business logic of order fulfillment, not the API latency.
     */
    @Timed(value = "business.order.fulfill", description = "Order fulfillment workflow")
    public void fulfillOrder(String orderId) {
        Order order = orderRepository.findById(orderId).orElseThrow();
        inventoryService.commit(order.getItems());
        paymentService.capture(order.getPaymentId());
        order.setStatus(OrderStatus.FULFILLED);
        orderRepository.save(order);
    }

    /**
     * Tier 2: Business operation needing context from parameters.
     * Labels help us slice metrics by order_type and customer_tier for business analysis.
     */
    @MeteredOperation(
        value = "business.order.create",
        description = "Order creation with business context",
        labels = {
            "order_type=#{#request.type}",
            "customer_tier=#{#request.customerTier}"
        }
    )
    public Order createOrder(CreateOrderRequest request) {
        inventoryService.reserve(request.getItems());
        Order order = Order.from(request);
        return orderRepository.save(order);
    }

    /**
     * Tier 2: Payment processing with acquirer breakdown.
     * Critical for understanding payment success rates by provider.
     */
    @MeteredOperation(
        value = "business.payment.process",
        labels = {
            "acquirer=#{#request.acquirer}",
            "payment_method=#{#request.method}"
        }
    )
    public PaymentResult processPayment(PaymentRequest request) {
        return paymentService.process(request);
    }

    /**
     * Tier 3: Batch processing with conditional outcomes.
     * Uses TimedOperation because:
     * - Need to track batch_size discovered at runtime
     * - Need conditional failure marking based on results
     * - Can't express this with annotations alone
     */
    public BatchResult processBatch(List<Order> orders) {
        try (var op = metrics.timed("business.batch.process")
                .tag("processor", "order_processor")) {

            op.batchSize(orders.size());

            int successCount = 0;
            int failureCount = 0;

            for (Order order : orders) {
                try {
                    fulfillOrder(order.getId());
                    successCount++;
                } catch (Exception e) {
                    failureCount++;
                    log.warn("Failed to process order {}: {}", order.getId(), e.getMessage());
                }
            }

            // Business outcome: Distinguish between partial and total failures
            if (failureCount == orders.size()) {
                op.markFailed("all_failed");
            } else if (failureCount > 0) {
                op.markFailed("partial_failure");
            }

            return new BatchResult(successCount, failureCount);
        }
    }

    /**
     * ❌ NOT instrumented: Simple getter that maps to GET /api/orders/{id}
     * The http.server.requests metric already covers API latency.
     * Only add business metrics if there's domain logic worth measuring.
     */
    public Order getOrder(String orderId) {
        return orderRepository.findById(orderId)
            .orElseThrow(() -> new OrderNotFoundException(orderId));
    }
}
```

---

## Example: Complete Service Instrumentation (Using Tiered Approach)

```java
@Service
@RequiredArgsConstructor
public class EventProcessingService {

    private final MetricsService metrics;

    public void processEvent(Event event) {
        try (TimedOperation op = metrics.timed("event_processing")
                .tag("event_type", sanitizeEventName(event.getEventName()))) {

            // Business logic - UNCHANGED
            doProcess(event);

            // L1: Feature-level (optional additional counter)
            metrics.count("events_processed")
                .tag("event_type", sanitizeEventName(event.getEventName()))
                .increment();

        } catch (Exception e) {
            // Exception handling - TimedOperation auto-records failure
            metrics.count("event_error")
                .tag("event_type", sanitizeEventName(event.getEventName()))
                .tag("error_type", e.getClass().getSimpleName())
                .increment();
            throw e;  // PRESERVED - same exception
        }
    }

    private String sanitizeEventName(String eventName) {
        if (eventName == null) return "unknown";
        String sanitized = eventName.length() > 50
            ? eventName.substring(0, 50)
            : eventName;
        return sanitized.replaceAll("[^a-zA-Z0-9._-]", "_").toLowerCase();
    }
}
```

## Example: Complete Go Service Instrumentation

```go
package service

import (
    "context"
    "time"
    "github.com/Vance-Club/goms/internal/metrics"
)

type EventProcessingService struct {
    // ... existing fields
}

func (s *EventProcessingService) ProcessEvent(ctx context.Context, event *Event) error {
    start := time.Now()
    metrics.RecordRequest()
    defer func() {
        metrics.RecordRequestDuration(time.Since(start))
    }()
    
    stopTimer := metrics.StartTimer("process_event")
    defer stopTimer()
    
    // Business logic - PRESERVE EXACTLY AS IS
    err := s.doProcess(ctx, event)
    
    // Metrics recording - ADD ONLY
    if err != nil {
        // L0: Service-level error
        metrics.RecordError(err.Error())
        
        // L1: Feature-level error
        metrics.RecordOperation("process_event", "error")
        return err // PRESERVED - same error
    }
    
    // L0: Service-level
    // (handled by metrics.RecordRequest() and defer)
    
    // L1: Feature-level
    metrics.RecordOperation("process_event", "success")
    return nil // PRESERVED - same return value
}

// HTTP Handler Example (Gin)
package handler

import (
    "time"
    "github.com/gin-gonic/gin"
    "github.com/Vance-Club/goms/internal/metrics"
)

type EventHandler struct {
    eventService *EventProcessingService
}

func (h *EventHandler) ProcessEvent(c *gin.Context) {
    start := time.Now()
    metrics.RecordRequest()
    defer func() {
        metrics.RecordRequestDuration(time.Since(start))
    }()
    
    var req ProcessEventRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        metrics.RecordError("validation_error")
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    
    event := &Event{
        EventName: req.EventName,
        Data:      req.Data,
    }
    
    err := h.eventService.ProcessEvent(c.Request.Context(), event)
    if err != nil {
        metrics.RecordError(err.Error())
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(200, gin.H{"status": "success"})
}
```

## References

### General
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [RED Method](https://www.weave.works/blog/the-red-method-key-metrics-for-microservices-architecture)
- [USE Method](http://www.brendangregg.com/usemethod.html)
- [Cardinality Management](https://prometheus.io/docs/practices/instrumentation/#things-to-watch-out-for)

### Java
- [Micrometer Documentation](https://micrometer.io/docs)
- [Micrometer AOP Annotations](https://micrometer.io/docs/concepts#_the_timed_annotation)
- [Spring Boot Actuator](https://docs.spring.io/spring-boot/docs/current/reference/html/actuator.html)
- [Removing Cross-Cutting Concerns with Micrometer and Spring AOP](https://medium.com/thg-tech-blog/removing-cross-cutting-concerns-with-micrometer-and-spring-aop-916a5602770f)
- [Spring AOP Documentation](https://docs.spring.io/spring-framework/docs/current/reference/html/core.html#aop)

### Go
- [Prometheus Go Client Library](https://github.com/prometheus/client_golang)
- [Prometheus Go Instrumentation Guide](https://prometheus.io/docs/guides/go-application/)
- [Go Prometheus Best Practices](https://prometheus.io/docs/instrumenting/writing_clientlibs/)