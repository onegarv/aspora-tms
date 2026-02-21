# Domain Context

> **When to load this file:** Only when working on domain-specific tasks (fintech, payments, ECM operations). Do not load for general engineering work.
>
> **Last updated:** 2026-02-16

---

## Industry

Fintech / Cross-border payments. Multi-region corridors: UAE, UK, US, EU. Regulatory awareness baked into all operational decisions.

## Core Domains

| Domain | Key Entities | Operations |
|--------|-------------|------------|
| **Order Management** | Orders, payments, refunds | SLA tracking, status transitions, stuck order resolution |
| **Acquirer Operations** | CHECKOUT, TRUELAYER, LEANTECH/LULU | Payment processing, reconciliation, provider routing |
| **ECM Operations** | Tickets, cases, escalations | Triage, assign, resolve, escalate workflows |
| **Compliance / FinCrime** | Alerts, SARs, cases | Transaction monitoring, alert triage, investigation, SAR filing |

## Terminology

| Term | Meaning |
|------|---------|
| ECM | Exception Case Management — managing orders that need manual intervention |
| TTD | Time to Detect — how quickly an issue is identified |
| TTM | Time to Mitigate — how quickly an issue is resolved |
| SLA | Service Level Agreement — contractual time limits for order processing |
| Acquirer | Payment provider that processes transactions (CHECKOUT, TRUELAYER, etc.) |
| Stuck order | An order that has not progressed through its expected lifecycle |
| CDC | Change Data Capture — streaming database changes to data warehouse |

## Escalation Matrix

Standard SLA thresholds for ECM operations:

| Level | Time Threshold | Action |
|-------|---------------|--------|
| Warning | 16 hours | Monitor, prepare investigation |
| Action Required | 24 hours | Active investigation and resolution |
| Critical | 48 hours | Senior escalation, stakeholder notification |
| Overdue | 72 hours | Management escalation, incident response |

## Data Architecture

| System | Purpose | Access Pattern |
|--------|---------|---------------|
| Redshift | Historical analytics, reporting | SQL, batch queries |
| Apache Pinot | Real-time analytics, live dashboards | SQL, sub-second queries |
| ThirdEye | Anomaly detection, alerting | Metrics monitoring, baselines |
| Google Sheets | Lightweight operational tracking | API, shared team access |

## AI-First Operations

Claude is used not just for coding but as an operational tool:
- Monitoring orders and SLA breaches
- Triaging issues and assigning tickets
- Managing operational queues
- Generating dashboards and reports
- Standardizing repeatable work into skills, artifacts, and runbooks
