package tech.vance.example.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import tech.vance.example.service.ExampleService;
import tech.vance.example.service.ExampleService.PaymentRequest;
import tech.vance.example.service.ExampleService.PaymentResult;

/**
 * Test controller to trigger the instrumented methods.
 *
 * After starting your app, run these curl commands:
 *
 * # Test Tier 1: @Timed
 * curl -X POST localhost:8080/test/orders/123/fulfill
 * curl -X POST localhost:8080/test/orders/123/cancel
 *
 * # Test Tier 2: @MeteredOperation with dynamic labels
 * curl -X POST localhost:8080/test/payments \
 *   -H "Content-Type: application/json" \
 *   -d '{"type":"card","acquirer":"stripe","amount":99.99}'
 *
 * curl -X POST "localhost:8080/test/orders?orderType=subscription&channel=web"
 *
 * # Check metrics
 * curl -s localhost:8080/actuator/prometheus | grep business_
 */
@RestController
@RequestMapping("/test")
@RequiredArgsConstructor
public class ExampleController {

    private final ExampleService exampleService;

    @PostMapping("/orders/{orderId}/fulfill")
    public String fulfillOrder(@PathVariable String orderId) {
        exampleService.fulfillOrder(orderId);
        return "Order " + orderId + " fulfilled";
    }

    @PostMapping("/orders/{orderId}/cancel")
    public String cancelOrder(@PathVariable String orderId) {
        exampleService.cancelOrder(orderId);
        return "Order " + orderId + " cancelled";
    }

    @PostMapping("/payments")
    public PaymentResult processPayment(@RequestBody PaymentRequest request) {
        return exampleService.processPayment(request);
    }

    @PostMapping("/orders")
    public String createOrder(
            @RequestParam String orderType,
            @RequestParam String channel) {
        return exampleService.createOrder(orderType, channel, null);
    }

    @GetMapping("/orders/{orderId}")
    public ExampleService.Order getOrder(@PathVariable String orderId) {
        return exampleService.getOrder(orderId);
    }
}
