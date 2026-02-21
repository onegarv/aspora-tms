# My Tickets

Shows your assigned ECM tickets with actionable instructions.

## Trigger
- `/my-tickets`
- "my queue"
- "what should I work on"

---

## Flow

### 1. Get Your Identity
Ask once: "What's your email?" → cache for session

### 2. Fetch Your Assignments
Read Google Sheet → Assignments tab:
```
Spreadsheet: 1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks
Filter: Assigned Agent = {your_email} AND Status IN ('OPEN', 'IN_PROGRESS')
```

### 3. Refresh Live Data
For each order, get current status from Redshift:
```sql
SELECT order_id, status, sub_state,
       meta_postscript_pricing_info_send_amount AS amount,
       ROUND(EXTRACT(EPOCH FROM (GETDATE() - created_at)) / 3600, 1) AS hours_stuck
FROM orders_goms
WHERE order_id IN ({your_order_ids})
```

### 4. Check for Auto-Completions
If any order shows `status = 'COMPLETED'` in Redshift:
- Flag it: "Order {id} completed — run `/resolve {id} 'Auto-completed'`"

### 5. Calculate SLA Status
| Diagnosis | SLA |
|-----------|-----|
| PAYMENT_FAILED | 2h |
| STATUS_SYNC_ISSUE | 1h |
| BRN_PENDING | 4h |
| RFI_PENDING | 24h |
| DEFAULT | 8h |

SLA Status:
- 🔴 BREACHED — past deadline
- ⚠️ CRITICAL — < 25% remaining
- 🟡 WARNING — 25-50% remaining
- 🟢 OK — > 50% remaining

### 6. Sort and Display
Sort by: Priority (P1 first) → SLA breached → SLA remaining

---

## Output Format

```
🎫 Your Queue — {count} tickets

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 🔴 P1 | AE12Y0K4BU00 | 60,100 AED | 297h | SLA: BREACHED
   Diagnosis: status_sync_issue
   ┌─────────────────────────────────────────────┐
   │ WHAT TO DO:                                 │
   │ 1. Open AlphaDesk → Search order            │
   │ 2. Verify Lulu shows CREDITED               │
   │ 3. Trigger webhook replay / force sync      │
   │ 4. Verify GOMS updates to COMPLETED         │
   │                                             │
   │ DONE? → /resolve AE12Y0K4BU00 "synced"      │
   │ STUCK? → /escalate AE12Y0K4BU00 "reason"    │
   └─────────────────────────────────────────────┘

2. 🟠 P2 | AE13IZSV2O00 | 35,000 AED | 48h | SLA: ⚠️ 2h
   Diagnosis: brn_issue
   → Push BRN to Lulu via AlphaDesk
   → Run `/order AE13IZSV2O00` for full steps

3. 🟡 P3 | AE14ABC1234 | 2,500 AED | 24h | SLA: 🟢 4h
   Diagnosis: rfi_order_grtr_than_24_hr
   → Send reminder to customer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ Start with #1 — longest overdue, highest amount

📊 Today: 0 resolved | Avg: — | SLA met: —%
```

### Empty Queue

```
🎫 Your Queue — 0 tickets

✨ All clear! No tickets assigned to you.

Next triage runs at 7 AM / 2 PM / 8 PM UAE.
Check #wg-asap-agent-pilot for updates.
```

---

## Guardrails
- Only show YOUR tickets (by email)
- Order data from Redshift via `ecm-gateway` MCP only
- Do NOT auto-resolve — you must confirm with `/resolve`
- If Diagnosis is empty, say "Run `/order {id}` for diagnosis"
