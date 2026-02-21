# Escalate Ticket

Escalate an order that needs help from another team.

## Trigger
- `/escalate {order_id} "{reason}"`
- "escalate {order_id}"
- "need help with {order_id}"

---

## Flow

### 1. Validate Order
Check that order exists in your assignments:
```
Spreadsheet: 1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks
Sheet: Assignments
Filter: Order ID = {order_id} AND Assigned Agent = {your_email}
```

### 2. Get Order Details
Query Redshift for current status and stuck_reason.

### 3. Determine Escalation Path
From `stuck-reasons.yaml`, get the escalation team:

| stuck_reason | Escalate To |
|--------------|-------------|
| `status_sync_issue` | TechOps (Ahsan) |
| `stuck_at_lulu` | Lulu Team (Binoy) |
| `stuck_at_lean_recon` | TechOps (Ahsan) |
| `refund_pending` | FinOps |
| `stuck_due_trm` | Compliance |
| `uncategorized` | Senior Ops |

### 4. Update Google Sheet
Update the row in Assignments tab:
```
Column D (Status) = "ESCALATED"
Column J (Notes) = "ESCALATED: {reason} | To: {team} | By: {agent} at {timestamp}"
```

### 5. Confirm with Contact Info

```
🚨 Escalated: {order_id}

Reason: {reason}
Escalated to: {team}
Contact: {contact_info}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next steps:
1. Reach out to {contact} via Slack/email
2. Share order ID and your notes
3. Follow up if no response in 4h
```

---

## Escalation Contacts

| Team | Contact | Slack | When |
|------|---------|-------|------|
| TechOps | Ahsan | @ahsan | Sync issues, webhook failures |
| Lulu | Binoy | @binoy | Lulu stuck >48h |
| FinOps | Finance Team | #finance-ops | Refunds, payment issues |
| Compliance | Compliance | #compliance | TRM holds, KYC blocks |
| Senior Ops | Dinesh | @dinesh | Uncategorized, complex cases |

---

## Output Format

### Success
```
🚨 Escalated: AE12Y0K4BU00

Reason: Lulu stuck for 72h, no response to previous requests
Escalated to: Lulu Team
Contact: Binoy (@binoy on Slack)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next steps:
1. Message @binoy on Slack with order ID
2. Share this context: "AED order stuck at Lulu for 72h,
   status shows TXN_TRANSMITTED but no payout"
3. Follow up if no response in 4h

📊 Queue: 11 tickets remaining
```

### Not Your Ticket
```
❌ Order AE12Y0K4BU00 not in your queue.

This order is assigned to: vishnu.r@aspora.com
Contact them or your manager to escalate.
```

---

## Guardrails
- Only escalate tickets assigned to YOU
- Reason is REQUIRED — explain why you're stuck
- Include what you already tried
- Update Google Sheet via `ecm-gateway` MCP only
