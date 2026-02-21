# Decisions

## DEC-001: Skills-first over code-first architecture (2025-02-14)
**Chose:** Skill .md files as the primary deliverable — TypeScript/Python is trigger only
**Over:** VoltAgent framework, LangChain, CrewAI, custom agent code
**Why:** Skills are reusable by any agent/team member, code creates knowledge silos
**Constraint:** NEVER add business logic to TypeScript/Python — all workflow logic lives in skills/*.md

## DEC-002: ecm-gateway as sole MCP server (2025-12-01)
**Chose:** Single ecm-gateway MCP for both Redshift and Google Sheets
**Over:** Multiple MCP servers (awslabs.redshift, separate sheets server)
**Why:** Multiple MCP servers caused tool name confusion — agents called wrong tools
**Constraint:** NEVER use awslabs.redshift-mcp-server or any standalone MCP — only ecm-gateway

## DEC-003: Two-step query architecture (2026-01-10)
**Chose:** Fast list query first (ecm-pending-list.sql <5s), then detail per order
**Over:** Single heavy query (ecm-dashboard-summary.sql) that joins everything
**Why:** Heavy queries timeout via MCP gateway (~30s limit) — dashboard query was >60s
**Constraint:** NEVER use ecm-active-tickets.sql or ecm-dashboard-summary.sql — they timeout via MCP

## DEC-004: Dead order filtering via payment + Lulu join (2026-01-20)
**Chose:** INNER JOIN paid_orders + Lulu existence check for AED orders
**Over:** Showing all GOMS orders regardless of payment/downstream state
**Why:** 40% of stuck orders were dead (no payment, no Lulu/Falcon) — agents wasted time on unactionable orders
**Constraint:** NEVER show orders without COMPLETED payment. AED orders MUST have Lulu record.

## DEC-005: stuck-reasons.yaml as single source of truth (2026-01-15)
**Chose:** YAML file mapping stuck_reason → team, SLA, runbook path, severity
**Over:** Hardcoded if-else in skills, diagnosis logic in TypeScript
**Why:** Config-driven behavior — adding a new stuck_reason is a YAML edit, not a code change
**Constraint:** NEVER hardcode stuck_reason logic in skills or code — always reference stuck-reasons.yaml

## DEC-006: orders_goms as base table, not analytics_orders_master_data (2026-01-08)
**Chose:** orders_goms for all ECM queries
**Over:** analytics_orders_master_data (legacy warehouse table)
**Why:** analytics_orders_master_data has stale data (6h lag) and missing fields needed for diagnosis
**Constraint:** NEVER use analytics_orders_master_data as base table in ECM queries

## DEC-007: Pi Manager direct connections for K8s, MCP for Claude Code (2026-02-10)
**Chose:** Dual mode — direct Redshift/Sheets in K8s, MCP tools in Claude Code
**Over:** MCP-only (latency issues in batch), direct-only (no Claude Code support)
**Why:** K8s cron needs reliable batch execution (<30s); Claude Code needs MCP for tool access
**Constraint:** NEVER use MCP in K8s batch mode — use direct connections for reliability

## DEC-008: Actionable sub_states whitelist (2026-01-20)
**Chose:** Explicit whitelist of 6 actionable sub_states
**Over:** Blacklist of non-actionable states
**Why:** Whitelist is safer — new unknown sub_states are excluded by default until reviewed
**Constraint:** ALWAYS use whitelist (FULFILLMENT_PENDING, REFUND_TRIGGERED, TRIGGER_REFUND, FULFILLMENT_TRIGGER, MANUAL_REVIEW, AWAIT_EXTERNAL_ACTION). Never add new sub_states without reviewing with ops team.

## DEC-009: High-value order threshold at 5,000 AED (2026-02-01)
**Chose:** Orders >= 5,000 AED flagged as high-value requiring supervisor approval
**Over:** No threshold (all orders treated equally)
**Why:** Regulatory and risk requirement — large transfers need additional oversight
**Constraint:** NEVER auto-resolve high-value orders (>= 5,000 AED) without supervisor approval

## DEC-010: Order count sanity check (200-600 typical) (2026-01-25)
**Chose:** Validation gates: >1,000 = red flag, >2,000 = STOP and investigate
**Over:** No validation on query result counts
**Why:** Missing dead order filter once returned 4,000+ orders — agents started working unactionable queue
**Constraint:** ALWAYS validate order count after query. If >1,000, check dead order filter. If >2,000, STOP.

## DEC-011: Two-agent split — Manager + Field (2026-02-16)
**Chose:** Two segregated agents in one repo: Manager (triage/assign, K8s batch) and Field (diagnose/resolve, interactive Claude Code)
**Over:** Monolith (all 9 skills in one CLAUDE.md, one persona)
**Why:** Context bloat (loading triage scoring + runbooks consumed excessive tokens), role confusion (same CLAUDE.md routed both manager and field operations), deployment mismatch (batch vs interactive)
**Constraint:** Manager NEVER loads runbooks. Field NEVER computes priority scores. Shared resources live in `shared/` — no duplication.

## DEC-012: Autonomous manager uses claude --print + MCP (not direct DB) (2026-02-17)
**Chose:** claude --print with MCP tools for autonomous batch execution
**Over:** Direct Redshift/Sheets connections via Python (per DEC-007)
**Why:** DEC-001 (skills-first) takes precedence — business logic lives in skills/*.md, not Python code. claude --print IS Claude Code, so MCP is the correct interface. DEC-007's "direct connections" was for the archived Pi Manager (Python).
**Constraint:** If MCP latency exceeds 30s in batch, revisit with direct connections.

## DEC-013: Field Slack bot uses Socket Mode + claude --print (2026-02-17)
**Chose:** Slack Bolt (Socket Mode) + claude --print subprocess
**Over:** Claude API + custom MCP client, or Slack Events API + webhook server
**Why:** Socket Mode needs no public URL (VPS/Docker friendly). claude --print reuses existing MCP config and skills — zero duplication of business logic (DEC-001).
**Constraint:** If latency >30s becomes unacceptable for real-time use, consider Claude API with direct MCP client.
