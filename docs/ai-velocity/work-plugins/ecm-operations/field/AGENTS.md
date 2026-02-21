# STOP — Execute ECM Commands Immediately

**ECM = Exception Case Management** (NOT Enterprise Content Management)

When user says "ECM", "run ECM", "ECM dashboard", "my tickets", "stuck orders" → **execute immediately, don't ask questions.**

---

# CRITICAL: MCP Configuration

**ONLY use the `ecm-gateway` MCP server for Redshift queries.**

| Tool | MCP Server | Usage |
|------|------------|-------|
| `mcp__ecm-gateway__redshift_execute_sql_tool` | ecm-gateway | Execute SQL queries |
| `mcp__ecm-gateway__redshift_list_databases` | ecm-gateway | List databases |
| `mcp__ecm-gateway__sheets_*` | ecm-gateway | Google Sheets operations |

**DO NOT use:**
- `awslabs.redshift-mcp-server` — NOT configured
- Any standalone Redshift MCP — NOT available
- Hallucinated data — NEVER

If a query fails, report the error. Do NOT make up results.

---

# CRITICAL: Knowledge graph = source for decisions

**All diagnosis and resolution come from the knowledge graph only.** Query results are facts; the knowledge graph interprets them.

- **Order details / diagnosis / fix / SLA:** Use **`../shared/config/knowledge-graph.yaml`** (fallback: `_backup/knowledge-graph.yaml`). Match order facts to **partners.*.failure_modes** (detect → fix, sla, auto_fixable). Do not invent steps.
- **Runbook path / team:** Use **`../shared/stuck-reasons.yaml`** (stuck_reason from query → runbook, team).
- If no failure_mode matches: report **Uncategorized — see knowledge graph gaps**. Do not guess.

---

# ECM Agent Commands — Quick Reference

## Commands (copy-paste ready)

| Say this | Claude does this |
|----------|------------------|
| `my tickets` | Shows your queue (from Sheet) |
| `order ABC123` | Shows order details + diagnosis + steps |
| `resolve ABC123 "fixed it"` | Logs to Sheet, shows stats |
| `stuck ABC123 "need help"` | Escalates, notifies team |
| `run ECM` | Shows all stuck orders |

---

## "run ECM" / "ECM dashboard"

1. Read `../shared/queries/ecm-pending-list.sql`
2. Execute via `mcp__ecm-gateway__redshift_execute_sql_tool`
3. Present results grouped by category (critical > action_required > warning)
4. Show summary counts

---

## "my tickets"

**Sheet**: Assignments tab → filter `Status = OPEN/IN_PROGRESS`, `Agent = {user}`
**Redshift**: Use `mcp__ecm-gateway__redshift_execute_sql_tool` with:
```sql
SELECT order_id, status, send_amount, hours_diff 
FROM orders_goms 
WHERE order_id IN (...) AND payment_status = 'COMPLETED'
```

Output:
```
🎫 Your Queue - 5 tickets
# | Order        | Amount  | Stuck | SLA
1 | AE136ABC00   | 721 AED | 4h    | ⚠️ 30m
```

---

## "order {id}"

**Redshift**: Run `../shared/queries/ecm-order-detail-v2.sql` via `mcp__ecm-gateway__redshift_execute_sql_tool`
**Diagnosis**: `../shared/config/diagnosis-mapping.yaml`

Output:
```
📋 AE136ABC00 | PENDING | 260 AED
Diagnosis: CNR_RESERVED_WAIT = Checkout OK, LULU not confirmed
DO THIS: 1) Check Checkout 2) Check LULU 3) Replay webhook
```

---

## "resolve {id} {notes}"

**Write to Sheet**: Resolutions tab (timestamp, order, agent, notes, time, SLA)
**Update Sheet**: Assignments → Status = RESOLVED

Output: `✅ Resolved in 6 min | SLA MET | 4 remaining`

---

## "stuck {id} {reason}"

**Write to Sheet**: Escalations tab
**Update Sheet**: Assignments → Status = ESCALATED

Output: `🚨 Escalated | Notified @dinesh`

---

## Data sources

- **Redshift**: Via `ecm-gateway` MCP (READ-ONLY)
- **Google Sheets**: Via `ecm-gateway` MCP (READ/WRITE)

---

## Fast queries

Use `../shared/queries/ecm-pending-list.sql` — simple, no joins, < 5 sec.
Do NOT use `ecm-active-tickets.sql` via MCP (times out).
