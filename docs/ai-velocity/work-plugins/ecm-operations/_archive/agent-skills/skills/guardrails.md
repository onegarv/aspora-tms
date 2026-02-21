# Agent Guardrails

**Apply these rules to every command you run.**

---

## MCP Configuration

**ONLY use `ecm-gateway` MCP for all queries.**

| Tool | Purpose |
|------|---------|
| `mcp__ecm-gateway__redshift_execute_sql_tool` | Query Redshift |
| `mcp__ecm-gateway__sheets_get_sheet_data` | Read Google Sheets |
| `mcp__ecm-gateway__sheets_update_cells` | Update Google Sheets |

**NEVER use:**
- `awslabs.redshift-mcp-server` — NOT configured
- Hallucinated tool names

---

## Data Rules

1. **No hallucination** — Only use data from query results
2. **No fabrication** — Don't invent order IDs, amounts, or statuses
3. **No guessing** — If data is missing, say so
4. **Follow runbooks** — Don't add steps that aren't documented

---

## Action Rules

1. **Only YOUR tickets** — Don't view/modify others' assignments
2. **RFI < 24h** — NEVER nudge the customer
3. **High-value (>50K)** — Requires supervisor approval
4. **Mask PII** — Show partial email (a***@), last 4 of phone

---

## Resolution Rules

1. **Notes required** — Explain what you did
2. **Don't auto-resolve** — Confirm before marking done
3. **Escalate if stuck** — Don't guess solutions

---

## Valid Stuck Reasons

Only these values are valid — don't invent new ones:

**Ops Team:**
- `status_sync_issue`
- `falcon_failed_order_completed_issue`
- `stuck_at_lean_recon`
- `brn_issue`
- `stuck_at_lulu`
- `refund_pending`
- `cancellation_pending`
- `uncategorized`

**KYC Ops:**
- `rfi_order_within_24_hr`
- `rfi_order_grtr_than_24_hr`
- `stuck_due_trm`
- `stuck_due_trm_rfi_within_24_hrs`
- `stuck_due_trm_rfi_grtr_than_24_hrs`
- `no_rfi_created`

**VDA Ops:**
- `stuck_at_vda_partner`
