You are instrumenting a service with Prometheus metrics. Follow these phases IN ORDER.
Do not skip phases. Each phase has a GATE — complete it before proceeding.

## Pre-Flight: Decision Context

Before starting, check for these files in the current project root (read them if they exist, skip if not):
1. `DECISIONS.md` — prior decisions for this project. Do NOT re-decide what's already settled.
2. `GUARDRAILS.md` — anti-patterns from past incidents. These are hard constraints.

Also read: $HOME/code/aspora/ai-velocity/prometheus-instrumentation/DECISIONS.md

## Phase 0: Skill Guardrails (MANDATORY)

Read the guardrails file at $HOME/code/aspora/ai-velocity/prometheus-instrumentation/GUARDRAILS.md

Internalize the Zero Mutation Rule before touching any code:
- Instrumentation is PURELY ADDITIVE — it NEVER changes functional behavior
- If removing all metrics would change program behavior, the instrumentation is BROKEN
- Never change exception types, method signatures, validation, or control flow

## Phase 1: Analyze Codebase

Before writing any code:
- Identify the language (Java or Go)
- List all service classes, controllers, and entry points
- Search for existing metrics: @Timed, MeterRegistry, prometheus, metrics.record
- List what Spring Boot / framework already provides for free:
  - `http.server.requests` — all HTTP endpoints (latency, count, status codes)
  - `jdbc.*` / `hikaricp.*` — database connections and queries
  - `spring.data.repository.*` — repository method timing
- Do NOT instrument what is already covered automatically

GATE: Present your codebase analysis to the user. List: (1) services found, (2) existing metrics, (3) what's auto-provided, (4) what NEEDS instrumentation. Wait for confirmation.

## Phase 2: Required Infrastructure (Non-Negotiable)

These files MUST exist or be created. Without them, ALL annotations are SILENTLY IGNORED:

### Java — Required Files
- [ ] `config/MetricsAopConfig.java` — TimedAspect + CountedAspect beans (enables @Timed/@Counted)
- [ ] Dependencies in build.gradle/pom.xml:
  - `io.micrometer:micrometer-registry-prometheus`
  - `org.springframework.boot:spring-boot-starter-actuator`
  - `org.springframework.boot:spring-boot-starter-aop`
- [ ] `application.yml` — actuator prometheus endpoint exposed
- [ ] `config/HikariMetricsConfig.java` — if service uses a database (Saturation signal)

### Java — Tier 2 Files (if dynamic labels needed)
- [ ] `metrics/MeteredOperation.java` — custom annotation
- [ ] `metrics/MeteredOperationAspect.java` — aspect implementation

### Go — Required Setup
- [ ] `prometheus.MustRegister()` calls for all custom metrics
- [ ] `/metrics` HTTP endpoint exposed
- [ ] Histogram buckets configured for latency metrics

Reference examples: Read $HOME/code/aspora/ai-velocity/prometheus-instrumentation/examples/java/ or examples/go/

GATE: List every file you will create or modify. Confirm with user before proceeding.

## Phase 3: Implement

Read the relevant implementation section from the skill reference:
$HOME/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md
- For Java: search for "Java Implementation" section
- For Go: search for "Go Implementation" section

Apply the Tier system:
- Tier 1: @Timed annotation (preferred — when success = normal return, failure = exception)
- Tier 2: @MeteredOperation (when you need dynamic labels from method parameters)
- Tier 3: Inline metrics (ONLY for gauges, queue depth, cron jobs, rate limiters)

Do NOT add @Timed to methods already covered by auto-instrumentation (controllers with http.server.requests, repositories with spring.data.*).

## Phase 4: Four Golden Signals Verification

Before completing, fill this table for the service:

| Signal     | Metric(s) Added        | Source                     | Verified? |
|------------|------------------------|----------------------------|-----------|
| Latency    | [fill metric names]    | @Timed / @MeteredOperation | [ ]       |
| Traffic    | [fill metric names]    | counter / auto             | [ ]       |
| Errors     | [fill metric names]    | outcome=error label        | [ ]       |
| Saturation | [fill metric names]    | HikariCP / thread pool     | [ ]       |

GATE: All 4 signals MUST have at least one metric. If any row is empty, go back and add coverage.

## Phase 5: Self-Check

Review every file you changed and answer these questions:
1. "If I deleted all metrics.record*() and @Timed lines, would the code behave identically?" → Must be YES
2. "Did I change any method signatures?" → Must be NO
3. "Did I change any exception types?" → Must be NO
4. "Does MetricsAopConfig.java exist with TimedAspect bean?" → Must be YES (Java)
5. "Did I add @Timed to methods already covered by http.server.requests?" → Must be NO

If any answer is wrong, fix it before completing.
