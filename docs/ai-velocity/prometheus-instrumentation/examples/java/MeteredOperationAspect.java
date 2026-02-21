package tech.vance.example.metrics;

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
 * AOP Aspect that processes @MeteredOperation annotations.
 *
 * Provides:
 * - Automatic timing (histogram)
 * - Counter with outcome (success/error)
 * - Dynamic labels from method parameters via SpEL
 */
@Aspect
@Component
@RequiredArgsConstructor
@Slf4j
public class MeteredOperationAspect {

    private static final Pattern LABEL_PATTERN = Pattern.compile("([^=]+)=#\\{(.+)}");
    private static final Pattern SANITIZE_PATTERN = Pattern.compile("[^a-z0-9._-]");

    private final MeterRegistry meterRegistry;
    private final ExpressionParser parser = new SpelExpressionParser();
    private final ParameterNameDiscoverer paramDiscoverer = new DefaultParameterNameDiscoverer();

    @Around("@annotation(meteredOperation)")
    public Object measureOperation(ProceedingJoinPoint joinPoint, MeteredOperation meteredOperation) throws Throwable {
        String operationName = sanitize(meteredOperation.value());
        Tags baseTags = extractTags(joinPoint, meteredOperation);

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
            recordMetrics(operationName, baseTags, sample, outcome, exceptionType, meteredOperation);
        }
    }

    private void recordMetrics(String operation, Tags baseTags, Timer.Sample sample,
                                String outcome, String exceptionType, MeteredOperation annotation) {
        try {
            // Record timer
            sample.stop(Timer.builder(operation + ".duration")
                .description(annotation.description())
                .tags(baseTags)
                .register(meterRegistry));

            // Record counter with outcome
            if (annotation.recordOutcome()) {
                Tags counterTags = baseTags.and("outcome", outcome);
                if (exceptionType != null) {
                    counterTags = counterTags.and("exception", sanitize(exceptionType));
                }
                meterRegistry.counter(operation + ".total", counterTags).increment();
            }
        } catch (Exception e) {
            log.warn("Failed to record metrics for {}: {}", operation, e.getMessage());
        }
    }

    private Tags extractTags(ProceedingJoinPoint joinPoint, MeteredOperation annotation) {
        Tags tags = Tags.empty();

        if (annotation.labels().length == 0) {
            return tags;
        }

        try {
            Method method = ((MethodSignature) joinPoint.getSignature()).getMethod();
            EvaluationContext context = new MethodBasedEvaluationContext(
                null, method, joinPoint.getArgs(), paramDiscoverer);

            for (String labelDef : annotation.labels()) {
                Matcher matcher = LABEL_PATTERN.matcher(labelDef);
                if (matcher.matches()) {
                    String labelName = matcher.group(1).trim();
                    String expression = matcher.group(2).trim();

                    try {
                        Object value = parser.parseExpression(expression).getValue(context);
                        String labelValue = value != null ? sanitize(value.toString()) : "unknown";
                        tags = tags.and(labelName, labelValue);
                    } catch (Exception e) {
                        log.debug("SpEL evaluation failed for '{}': {}", expression, e.getMessage());
                        tags = tags.and(labelName, "unknown");
                    }
                }
            }
        } catch (Exception e) {
            log.warn("Failed to extract tags: {}", e.getMessage());
        }

        return tags;
    }

    private String sanitize(String value) {
        if (value == null) return "unknown";
        String sanitized = value.length() > 50 ? value.substring(0, 50) : value;
        return SANITIZE_PATTERN.matcher(sanitized.toLowerCase()).replaceAll("_");
    }
}
