package tech.vance.example.service;

import io.micrometer.core.annotation.Counted;
import io.micrometer.core.annotation.Timed;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import tech.vance.example.metrics.MeteredOperation;

/**
 * Example service demonstrating the tiered instrumentation approach.
 *
 * Copy the patterns you need to your own services.
 */
@Service
@Slf4j
public class ExampleService {

    // =========================================================================
    // TIER 1: @Timed - Simple business operations with static labels
    // =========================================================================

    /**
     * Use @Timed for simple business operations where you don't need
     * dynamic labels from method parameters.
     *
     * Metrics produced:
     * - business_order_fulfill_seconds_count
     * - business_order_fulfill_seconds_sum
     * - business_order_fulfill_seconds_max
     */
    @Timed(value = "business.order.fulfill", description = "Order fulfillment workflow")
    public void fulfillOrder(String orderId) {
        log.info("Fulfilling order: {}", orderId);
        simulateWork(100);
    }

    /**
     * Use @Counted when you only care about count, not duration.
     *
     * Metrics produced:
     * - business_order_cancelled_total
     */
    @Counted(value = "business.order.cancelled", description = "Order cancellations")
    public void cancelOrder(String orderId) {
        log.info("Cancelling order: {}", orderId);
    }

    // =========================================================================
    // TIER 2: @MeteredOperation - Operations needing dynamic labels
    // =========================================================================

    /**
     * Use @MeteredOperation when you need labels derived from method parameters.
     * SpEL expressions extract values at runtime.
     *
     * Metrics produced:
     * - business_payment_process_total{payment_type="card", acquirer="stripe", outcome="success"}
     * - business_payment_process_duration_seconds{payment_type="card", acquirer="stripe"}
     */
    @MeteredOperation(
        value = "business.payment.process",
        description = "Payment processing with acquirer breakdown",
        labels = {
            "payment_type=#{#request.type}",
            "acquirer=#{#request.acquirer}"
        }
    )
    public PaymentResult processPayment(PaymentRequest request) {
        log.info("Processing {} payment via {}", request.getType(), request.getAcquirer());
        simulateWork(200);
        return new PaymentResult(true, "txn_123");
    }

    /**
     * Another example with different parameter types.
     */
    @MeteredOperation(
        value = "business.order.create",
        labels = {
            "order_type=#{#orderType}",
            "channel=#{#channel}"
        }
    )
    public String createOrder(String orderType, String channel, Object orderData) {
        log.info("Creating {} order from {}", orderType, channel);
        simulateWork(150);
        return "order_" + System.currentTimeMillis();
    }

    // =========================================================================
    // NOT INSTRUMENTED: Simple reads (http.server.requests covers the API)
    // =========================================================================

    /**
     * Don't instrument simple getters - the HTTP layer metrics cover API latency.
     */
    public Order getOrder(String orderId) {
        return new Order(orderId, "ACTIVE");
    }

    // =========================================================================
    // Helper classes and methods
    // =========================================================================

    private void simulateWork(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // Simple DTOs for the example
    public record PaymentRequest(String type, String acquirer, double amount) {}
    public record PaymentResult(boolean success, String transactionId) {}
    public record Order(String id, String status) {}
}
