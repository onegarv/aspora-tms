package tech.vance.example.metrics;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a method for business-level metrics with dynamic labels.
 * Labels are extracted from method parameters using SpEL expressions.
 *
 * Example usage:
 * <pre>
 * {@code
 * @MeteredOperation(
 *     value = "business.payment.process",
 *     labels = {
 *         "acquirer=#{#request.acquirer}",
 *         "payment_method=#{#request.method}"
 *     }
 * )
 * public PaymentResult process(PaymentRequest request) { ... }
 * }
 * </pre>
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface MeteredOperation {

    /** Operation name for the metric (e.g., "business.payment.process") */
    String value();

    /** Description for the metric */
    String description() default "";

    /**
     * SpEL expressions to extract label values from parameters.
     * Format: "labelName=#{expression}"
     */
    String[] labels() default {};

    /** Whether to record outcome (success/error) as a label */
    boolean recordOutcome() default true;
}
