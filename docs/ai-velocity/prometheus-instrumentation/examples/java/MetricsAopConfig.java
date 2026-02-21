package tech.vance.example.config;

import io.micrometer.core.aop.CountedAspect;
import io.micrometer.core.aop.TimedAspect;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Enables Micrometer AOP annotations (@Timed, @Counted) for declarative metrics.
 *
 * REQUIRED: Without this config, @Timed and @Counted annotations will be ignored!
 *
 * Copy this file to your project's config package:
 *   src/main/java/your/package/config/MetricsAopConfig.java
 */
@Configuration
public class MetricsAopConfig {

    /**
     * Enables @Timed annotation support.
     *
     * Methods annotated with @Timed will automatically record:
     * - {metric}_seconds_count - number of calls
     * - {metric}_seconds_sum - total time spent
     * - {metric}_seconds_max - max time observed
     * - Histogram buckets for percentile calculation
     *
     * Failed calls are automatically tagged with exception={ExceptionClass}
     */
    @Bean
    public TimedAspect timedAspect(MeterRegistry registry) {
        return new TimedAspect(registry);
    }

    /**
     * Enables @Counted annotation support.
     *
     * Methods annotated with @Counted will automatically record:
     * - {metric}_total - number of calls
     *
     * Use @Counted when you only care about count, not duration.
     */
    @Bean
    public CountedAspect countedAspect(MeterRegistry registry) {
        return new CountedAspect(registry);
    }
}
