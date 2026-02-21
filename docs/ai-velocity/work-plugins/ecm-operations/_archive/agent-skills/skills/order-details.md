# Order Details

Get full diagnosis and resolution steps for an order.

## Trigger
- `/order {order_id}`
- "lookup {order_id}"

---

## Flow

### 1. Query Order Details
Run `queries/ecm-triage-fast.sql` with `{order_id}` replaced.

### 2. Check Actionability
If `is_actionable = false`:
```
⛔ Order not actionable: {disqualification_reason}
```
Stop here.

### 3. Get Stuck Reason
The query returns `stuck_reason` — use this to find the runbook.

### 4. Calculate Priority
```
score = 0.25 × age + 0.20 × amount + 0.25 × severity + 0.15 × rfi + 0.10 × payment

P1 (🔴): score ≥ 0.7
P2 (🟠): score 0.5-0.7
P3 (🟡): score 0.3-0.5
P4 (🟢): score < 0.3
```

### 5. Load Runbook
From `stuck-reasons.yaml`, get the runbook path for this `stuck_reason`.
Read the runbook and present the steps.

---

## Output Format

```
## AE12Y0K4BU00 | P1 🔴 | Ops Team

### What's Wrong
AED order completed at Lulu (CREDITED) but GOMS still shows
PROCESSING_DEAL_IN after 297 hours. The status webhook was
likely missed or failed to process.

### What To Do
1. Open **AlphaDesk** → Search `AE12Y0K4BU00`
2. Go to Order Details → Verify Lulu shows CREDITED
3. Click **Trigger Webhook Replay** or **Force Sync**
4. Wait 30 seconds, refresh
5. Verify GOMS status updates to COMPLETED

### Order Facts
| Field | Value |
|-------|-------|
| Status | PROCESSING_DEAL_IN / FULFILLMENT_PENDING |
| Amount | 60,100 AED → 1,320,000 INR |
| Age | 297h (12 days) |
| Payment | ✅ COMPLETED via Checkout |
| Falcon | ✅ 1125612612 (CREDITED) |
| Lulu | ✅ CREDITED |
| Payout | ✅ Completed |
| RFI | None |

### Customer
| Field | Value |
|-------|-------|
| User ID | usr_abc123 |
| Email | a***@gmail.com |
| Phone | ***1234 |

### Resolution
**SLA:** 1h | **Escalation:** TechOps | **Runbook:** `status-sync-issue.md`

DONE? → `/resolve AE12Y0K4BU00 "Status synced via AlphaDesk"`
STUCK? → `/escalate AE12Y0K4BU00 "Webhook replay failed"`
```

---

## What's Wrong Templates

| stuck_reason | Explanation |
|--------------|-------------|
| `status_sync_issue` | Order completed at partner but GOMS not updated. Webhook missed. |
| `brn_issue` | Payment reconciled but BRN not pushed to Lulu. Lulu waiting for confirmation. |
| `refund_pending` | Order failed/cancelled but customer funds not refunded. |
| `stuck_at_lulu` | Sent to Lulu but stuck in processing. No Falcon transaction. |
| `rfi_order_grtr_than_24_hr` | RFI pending over 24h. Customer hasn't responded. |
| `stuck_due_trm` | Blocked at TRM compliance check. Payment complete but order frozen. |

---

## Guardrails
- Use `ecm-gateway` MCP ONLY for queries
- Do NOT invent order data — use query results only
- Do NOT suggest actions not in the runbook
- RFI < 24h: Do NOT suggest nudging customer
- Mask customer PII (partial email, last 4 of phone)
