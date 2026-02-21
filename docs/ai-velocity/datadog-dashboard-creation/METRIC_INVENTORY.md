# Metric Inventory - All Prom Services

Generated: 2026-02-16

## Services with Custom Prometheus Metrics

### beneficiary-service

**Prometheus Names → Datadog Names**

| Golden Signal | Prometheus | Datadog | Tags |
|---------------|-----------|---------|------|
| **TRAFFIC** | | | |
| | `beneficiary.create.total` | `beneficiary.create.count` | service, beneficiary_type, outcome |
| | `beneficiary.verification.total` | `beneficiary.verification.count` | service, verifier, status |
| | `beneficiary.verifier.outcome.total` | `beneficiary.verifier.outcome.count` | service, verifier, account_type, outcome, external_status |
| | `beneficiary.strategy.resolution.total` | `beneficiary.strategy.resolution.count` | service, strategy, resolved_type, is_valid |
| | `external.api.calls.total` | `external.api.calls.count` | service, provider, operation, outcome |
| **ERRORS** | | | |
| | `beneficiary.errors.total` | `beneficiary.errors.count` | service, operation, error_type |
| | `beneficiary.verification.failures.total` | `beneficiary.verification.failures.count` | service, verifier, reason |
| **LATENCY** | | | |
| | `beneficiary.operation.duration` | `beneficiary.operation.duration` | service, operation, outcome |
| | `external.api.duration` | `external.api.duration` | service, provider, operation, outcome |
| **SATURATION** | | | |
| | `beneficiary.lock.events.total` | `beneficiary.lock.events.count` | service, lock_type, outcome |

---

### email-service

**Prometheus Names → Datadog Names**

| Golden Signal | Prometheus | Datadog | Tags |
|---------------|-----------|---------|------|
| **TRAFFIC** | | | |
| | `emailservice.cron.runs.total` | `emailservice.cron.runs.count` | cron_name, success |
| | `emailservice.emails.processed.total` | `emailservice.emails.processed.count` | operation, success |
| | `emailservice.external.calls.total` | `emailservice.external.calls.count` | provider, operation, success |
| | `emailservice.imap.connections.total` | `emailservice.imap.connections.count` | success |
| **ERRORS** | | | |
| | `emailservice.errors.total` | `emailservice.errors.count` | operation, error_type |
| | `emailservice.rfi.automation.failures.total` | `emailservice.rfi.automation.failures.count` | reason |
| | `emailservice.rfi.automation.total` | `emailservice.rfi.automation.count` | success |
| **LATENCY** | | | |
| | `emailservice.operation.duration` | `emailservice.operation.duration` | operation, success |
| | `emailservice.cron.duration` | `emailservice.cron.duration` | cron_name, success |
| | `emailservice.external.call.duration` | `emailservice.external.call.duration` | provider, operation, success |

---

### goblin-service

**Prometheus Names → Datadog Names**

| Golden Signal | Prometheus | Datadog | Tags |
|---------------|-----------|---------|------|
| **TRAFFIC** | | | |
| | `goblin.requests.total` | `goblin.requests.count` | service |
| | `goblin.events.processed.total` | `goblin.events.processed.count` | service |
| **ERRORS** | | | |
| | `goblin.errors.total` | `goblin.errors.count` | service, error_type |
| | `goblin.events.failed.total` | `goblin.events.failed.count` | service, error_type |
| **LATENCY** | | | |
| | `goblin.request.duration` | `goblin.request.duration` | service |
| | `goblin.order.operations.duration` | `goblin.order.operations.duration` | operation, outcome |
| | `goblin.payment.operations.duration` | `goblin.payment.operations.duration` | operation, outcome |
| | `goblin.fulfillment.operations.duration` | `goblin.fulfillment.operations.duration` | operation, outcome |
| | `goblin.pricing.operations.duration` | `goblin.pricing.operations.duration` | operation, outcome |
| | `goblin.quote.operations.duration` | `goblin.quote.operations.duration` | operation, outcome |
| | `goblin.external.requests.duration` | `goblin.external.requests.duration` | provider, operation |
| | `goblin.activity.executions.duration` | `goblin.activity.executions.duration` | activity_type, outcome |
| | `goblin.payment.events.duration` | `goblin.payment.events.duration` | event_type, outcome |
| | `goblin.fulfillment.events.duration` | `goblin.fulfillment.events.duration` | event_type, outcome |

---

### settlement-service

**Prometheus Names → Datadog Names**

| Golden Signal | Prometheus | Datadog | Tags |
|---------------|-----------|---------|------|
| **TRAFFIC** | | | |
| | `settlement_*_total` | `settlement_*.count` | outcome |
| | `settlement_job_executions_total` | `settlement_job_executions.count` | job_name, outcome |
| | `settlement_external_api_calls_total` | `settlement_external_api_calls.count` | client_type, outcome |
| **LATENCY** | | | |
| | `settlement_*_duration_seconds` | `settlement_*_duration_seconds` | operation, outcome |
| | `settlement_database_operation_duration_seconds` | `settlement_database_operation_duration_seconds` | operation, outcome |
| **SATURATION** | | | |
| | `settlement_job_queue_depth` | `settlement_job_queue_depth` | job_name |
| | `settlement_batch_size` | `settlement_batch_size` | job_name |

---

### users-service

**Prometheus Names → Datadog Names**

| Golden Signal | Prometheus | Datadog | Tags |
|---------------|-----------|---------|------|
| **TRAFFIC** | | | |
| | `users_auth_operations_total` | `users_auth_operations.count` | operation, outcome |
| | `users_profile_operations_total` | `users_profile_operations.count` | operation, outcome |
| | `users_otp_operations_total` | `users_otp_operations.count` | provider, operation, outcome |
| **LATENCY** | | | |
| | `users_external_call_duration_seconds` | `users_external_call_duration_seconds` | service, operation, outcome |

---

### verification-service

**Status:** Has MetricsConstants.java but needs deeper analysis for actual metrics.

---

## Services Using Spring Boot Actuator Metrics Only

These services will use standard Spring Boot + JVM metrics:

- alphadesk-api
- backoffice
- bbps-service
- cron-server
- eventbus
- falcon-service
- fx-service
- lulu-fulfillment-service
- notification-service
- partner-dashboard-api
- recon-service
- rewards-api-service
- rewards-service
- workflow-service

**Standard Spring Boot Metrics (available on all):**

| Golden Signal | Metric | Datadog Name |
|---------------|--------|--------------|
| **TRAFFIC** | `http.server.requests` | `http.server.requests.count` |
| **ERRORS** | `http.server.requests{status:5xx}` | `http.server.requests.count{status:5xx}` |
| **LATENCY** | `http.server.requests` (percentiles) | `http.server.requests_max`, `http.server.requests.sum` |
| **SATURATION** | `jvm.memory.used` | `jvm.memory.used` |
| | `jvm.threads.live` | `jvm.threads.live` |
| | `process.cpu.usage` | `process.cpu.usage` |
| | `hikaricp.connections.active` | `hikaricp.connections.active` |

---

## Next Steps

1. Create dashboards for services WITH custom metrics first (6 services)
2. Create template dashboard for Spring Boot services (apply to remaining 14)
3. Verify all metrics exist in Datadog before adding widgets
