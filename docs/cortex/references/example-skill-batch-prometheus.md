---
name: batch-prometheus-instrumentation
description: >
  Instruments multiple services with Prometheus metrics following SRE best
  practices (Four Golden Signals, RED method). Clones repos, adds
  Micrometer (Java) or client_golang (Go), creates PRs. Use when asked to
  "instrument all services", "add metrics to the fleet", "batch
  instrumentation", or "add observability to multiple repos". Do NOT use
  for single-service instrumentation — use manual instrumentation instead.
---

# Batch Prometheus Instrumentation Skill

## Purpose
Automate the instrumentation of multiple services with production-grade Prometheus metrics following Google SRE principles (Four Golden Signals), RED method, and the organization's observability standards.

## Prerequisites
- GitHub CLI authenticated (`gh auth status`)
- Target directory exists for cloning repos
- Access to source organization repos

## Workflow Per Service

### 1. Clone & Branch
```bash
cd {target_dir}
gh repo clone {org}/{service-name}
cd {service-name}
git checkout -b olly-metrics
```

### 2. Analyze & Route by Language
Determine the service type and follow the appropriate path:

**Java service?** (has `pom.xml` or `build.gradle`)
→ Follow "Java Services" workflow below

**Go service?** (has `go.mod`)
→ Follow "Go Services" workflow below

Also analyze:
- Map request/event flows from code
- Identify key entities and operations
- Find existing metrics patterns to match

### 3. Instrument by Language

#### Java Services
- Add micrometer-registry-prometheus to build.gradle/pom.xml
- Configure actuator endpoints in application.yml
- Create MetricsUtil class (centralized, DRY)
- Add MetricsAopConfig for @Timed/@Counted support
- Instrument services with @Timed annotations (Tier 1)
- Add inline metrics only for gauges/traffic counters (Tier 3)

#### Go Services
- Add prometheus/client_golang to go.mod
- Create internal/metrics package
- Add /metrics endpoint to router
- Instrument handlers and services

### 4. Verification Loop
Validate after each service — fix errors before proceeding:

```bash
# Java
./gradlew build && ./gradlew test
# If fails: fix compilation/test errors, re-run

# Go
go build ./... && go test ./...
# If fails: fix errors, re-run
```

**Checklist per service:**
- [ ] Build compiles without errors
- [ ] All existing tests pass
- [ ] `/metrics` or `/actuator/prometheus` endpoint responds
- [ ] No functional logic changes (Zero Mutation Rule)

Only proceed to PR creation when all checks pass.

### 5. Create PR
```bash
git add -A
git commit -m "feat(observability): add Prometheus metrics instrumentation

- Add Four Golden Signals metrics (latency, traffic, errors, saturation)
- Centralized MetricsUtil following DRY principles
- @Timed annotations for method-level instrumentation
- Zero functional logic changes

Co-Authored-By: Claude <noreply@anthropic.com>"

git push -u origin olly-metrics
gh pr create --base stage-env --title "feat(observability): Prometheus metrics instrumentation" --body "## Summary
- Added production-grade Prometheus metrics
- Four Golden Signals: Latency, Traffic, Errors, Saturation
- Centralized MetricsUtil for DRY compliance
- @Timed annotations for declarative metrics

## Changes
- Added micrometer/prometheus dependencies
- Created MetricsUtil class
- Instrumented key service methods
- Configured /metrics endpoint

## Test Plan
- [ ] Build passes
- [ ] Existing tests pass
- [ ] /metrics endpoint returns Prometheus format
- [ ] No functional behavior changes"
```

## Principles (Non-Negotiable)
1. **Zero Logic Mutation** - Metrics NEVER change business logic
2. **Four Golden Signals** - Latency, Traffic, Errors, Saturation
3. **DRY** - Centralized MetricsUtil, no duplicated metric definitions
4. **SOLID** - Single responsibility, dependency injection
5. **Cardinality Control** - Sanitize labels, avoid high-cardinality explosions

## Reference
See full instrumentation patterns: `~/code/aspora/ai-velocity/prometheus-instrumentation/SKILL.md`
