# Java Metrics Instrumentation Examples

## Quick Start

### 1. Copy these files to your project

```
your-project/
├── src/main/java/your/package/
│   ├── config/
│   │   └── MetricsAopConfig.java      # Required for @Timed/@Counted
│   ├── metrics/
│   │   ├── MeteredOperation.java      # Custom annotation (optional, for Tier 2)
│   │   └── MeteredOperationAspect.java # Aspect (optional, for Tier 2)
│   └── service/
│       └── YourService.java           # Add annotations to your methods
```

### 2. Ensure dependencies are present

```gradle
// build.gradle
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-actuator'
    implementation 'io.micrometer:micrometer-registry-prometheus'
    implementation 'org.springframework.boot:spring-boot-starter-aop'
}
```

### 3. Create the AOP config (required)

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

### 4. Annotate your business methods

```java
// Tier 1: Simple business operation
@Timed(value = "business.order.fulfill")
public void fulfillOrder(String orderId) { ... }

// Tier 2: With dynamic labels from parameters
@MeteredOperation(
    value = "business.payment.process",
    labels = {"acquirer=#{#request.acquirer}"}
)
public PaymentResult process(PaymentRequest request) { ... }
```

## Testing the Examples

### Start your app
```bash
./gradlew bootRun
```

### Trigger the methods
```bash
# Tier 1: @Timed
curl -X POST localhost:8080/test/orders/123/fulfill

# Tier 1: @Counted
curl -X POST localhost:8080/test/orders/123/cancel

# Tier 2: @MeteredOperation with dynamic labels
curl -X POST localhost:8080/test/payments \
  -H "Content-Type: application/json" \
  -d '{"type":"card","acquirer":"stripe","amount":99.99}'

curl -X POST "localhost:8080/test/orders?orderType=subscription&channel=web"
```

### Check metrics
```bash
curl -s localhost:8080/actuator/prometheus | grep business_
```

### Expected output
```
# Tier 1 metrics
business_order_fulfill_seconds_count{class="ExampleService",method="fulfillOrder"} 1.0
business_order_fulfill_seconds_sum{class="ExampleService",method="fulfillOrder"} 0.105
business_order_cancelled_total{class="ExampleService",method="cancelOrder"} 1.0

# Tier 2 metrics (with dynamic labels!)
business_payment_process_total{acquirer="stripe",outcome="success",payment_type="card"} 1.0
business_payment_process_duration_seconds_count{acquirer="stripe",payment_type="card"} 1.0
business_order_create_total{channel="web",order_type="subscription",outcome="success"} 1.0
```

## Files in This Directory

| File | Purpose |
|------|---------|
| `MetricsAopConfig.java` | Enables @Timed/@Counted (copy to your config package) |
| `MeteredOperation.java` | Custom annotation for Tier 2 (optional) |
| `MeteredOperationAspect.java` | Processes @MeteredOperation (optional) |
| `ExampleService.java` | Shows Tier 1 and Tier 2 patterns |
| `ExampleController.java` | Test endpoints to trigger the methods |

## What NOT to Instrument

Don't add `@Timed` to:
- REST controllers (http.server.requests already covers this)
- Simple CRUD getters
- Repository methods (spring.data.repository.* covers this)

Focus on **business logic** that has domain meaning.
