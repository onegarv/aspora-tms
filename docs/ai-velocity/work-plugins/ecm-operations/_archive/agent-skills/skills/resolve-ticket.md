# Resolve Ticket

Mark an order as resolved with notes.

## Trigger
- `/resolve {order_id} "{notes}"`
- "resolve {order_id}"
- "mark {order_id} done"

---

## Flow

### 1. Validate Order
Check that order exists in your assignments:
```
Spreadsheet: 1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks
Sheet: Assignments
Filter: Order ID = {order_id} AND Assigned Agent = {your_email}
```

If not found:
```
❌ Order {order_id} not in your queue.
Run `/my-tickets` to see your assignments.
```

### 2. Check Current Status
If Status is already RESOLVED or CLOSED:
```
⚠️ Order {order_id} already resolved.
```

### 3. High-Value Check
If Amount > 50,000 AED:
```
⚠️ HIGH-VALUE ORDER — Requires supervisor approval

Amount: {amount} AED
Please confirm with your supervisor before resolving.

Continue? (yes/no)
```

### 4. Update Google Sheet
Update the row in Assignments tab:
```
Column D (Status) = "RESOLVED"
Column J (Notes) = {notes} + " | Resolved by {agent} at {timestamp}"
```

### 5. Confirm

```
✅ Resolved: {order_id}

Notes: {notes}
Resolved at: {timestamp}

📊 Today: {resolved_count} resolved | Queue: {remaining} left
```

---

## Output Format

### Success
```
✅ Resolved: AE12Y0K4BU00

Notes: Status synced via AlphaDesk webhook replay
Resolved at: 2026-02-13 14:30 UAE

📊 Today: 5 resolved | Queue: 12 left
```

### Not Your Ticket
```
❌ Order AE12Y0K4BU00 not in your queue.

This order is assigned to: akshay@aspora.com
Run `/my-tickets` to see your assignments.
```

### High-Value Warning
```
⚠️ HIGH-VALUE ORDER — Requires supervisor approval

Order: AE12Y0K4BU00
Amount: 75,000 AED

Please confirm with your supervisor before resolving.
Type "yes" to continue or "no" to cancel.
```

---

## Guardrails
- Only resolve tickets assigned to YOU
- High-value (>50K AED) requires confirmation
- Notes are REQUIRED — explain what you did
- Update Google Sheet via `ecm-gateway` MCP only
