# Assign Tickets Skill

## Trigger
- "assign tickets"
- "assign tickets to {agent}"
- "assign {order_id} to {agent}"
- "distribute tickets"

## Google Sheet
**Spreadsheet ID:** `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`

### Assignments Tab Schema (10 columns)
```
A: Order ID | B: Assigned Agent | C: Assigned At | D: Status | E: Priority
F: Currency | G: Amount         | H: Hours Stuck | I: Diagnosis | J: Notes
```

## Description
Assigns stuck orders to agents by writing to **Google Sheets** (Assignments tab). Pulls unassigned orders from Redshift.

> For full priority-scored triage + assignment, use `triage and assign` (manager persona).

## Modes

### Mode 1: Auto-assign to self
```
assign tickets
```
Gets unassigned stuck orders and assigns them to the current agent.

### Mode 2: Assign to specific agent
```
assign tickets to ravi@aspora.com
```
Assigns unassigned orders to the specified agent.

### Mode 3: Assign specific order
```
assign AE136JM2JF00 to ravi@aspora.com
```
Assigns a single order to the specified agent.

## Data Flow

### Step 1: Get already-assigned orders
Read **Assignments** tab where `Status IN ('OPEN', 'IN_PROGRESS', 'ESCALATED')`:
- Get list of already-assigned order_ids

### Step 2: Get pending orders from Redshift
Run `../shared/queries/ecm-pending-list.sql` or:
```sql
SELECT order_id, order_status, currency_from, send_amount, created_at,
       EXTRACT(EPOCH FROM (GETDATE() - created_at)) / 3600 AS hours_diff
FROM orders_goms
WHERE status IN ('PROCESSING_DEAL_IN', 'PENDING', 'FAILED')
  AND order_type = 'REMITTANCE'
  AND currency_from IN ('AED', 'GBP', 'EUR')
  AND created_at::date >= CURRENT_DATE - 7
ORDER BY created_at ASC
LIMIT 50
```

### Step 3: Filter out already-assigned
Remove orders that are already in Assignments tab.

### Step 4: Check agent capacity
Read **Agents** tab:
- Get agent's `Max Tickets` limit
- Count current OPEN/IN_PROGRESS tickets
- Calculate available slots

### Step 5: Assign orders
For each order (up to available slots):
- Append row to **Assignments** tab with ALL 10 columns:
```
Order ID | Assigned Agent | Assigned At | Status | Priority | Currency | Amount | Hours Stuck | Diagnosis | Notes
{order_id} | {agent_email} | {now} | OPEN | P{1-4} | {currency} | {amount} | {hours} | {stuck_reason} | {one_liner}
```

Use `sheets_update_cells`:
- spreadsheet_id: `1r50OEZlFVSUmU1tkLBqx2_BzlilZ3s0pArNHV83tRks`
- sheet: `Assignments`
- range: next available row

### Step 6: Determine priority (from knowledge-graph scoring model)

Use the priority scoring formula from `../shared/config/knowledge-graph.yaml` and `skills/triage-and-assign.md`:

For each order, enrich with `../shared/queries/ecm-triage-fast.sql` to get `stuck_reason`, then compute:
```
PRIORITY_SCORE = 0.25*age + 0.20*amount + 0.25*severity + 0.15*rfi + 0.10*payment + 0.05*attempts
```

Priority mapping:
- **P1** (Score > 0.7): Critical — immediate action required
- **P2** (Score 0.5-0.7): High — resolve within SLA
- **P3** (Score 0.3-0.5): Medium — standard handling
- **P4** (Score < 0.3): Low — monitor

Write priority AND a one-line actionable in the Notes column:
```
{order_id} | {agent} | {now} | OPEN | P{1-4} | [Auto] {stuck_reason}: {one_line_action}
```

See `skills/triage-and-assign.md` → "Action Generation Rules" for the exact actionable text per stuck_reason.

## Output Format

### Auto-assign
```
✅ Assigned 5 tickets to @{agent}

# | Order ID      | Amount    | Stuck | Priority
--|---------------|-----------|-------|----------
1 | AE136JLXUG00  | 721 AED   | 4h    | P2
2 | AE136JM2JF00  | 260 AED   | 6h    | P4
3 | AE136JM6JF00  | 1,250 AED | 2h    | P2
4 | AE136JM6UG00  | 8,010 AED | 1h    | P1
5 | AE136JM7AB00  | 500 AED   | 3h    | P4

📊 Queue Status:
│ Your tickets: 5 / 10 (max)
│ Unassigned remaining: 12

Type `my tickets` to see your queue with SLA status.
```

### Assign to specific agent
```
✅ Assigned 3 tickets to @priya

# | Order ID      | Amount    | Stuck
--|---------------|-----------|-------
1 | AE136KL1XX00  | 450 AED   | 2h
2 | AE136KL2XX00  | 890 AED   | 4h
3 | AE136KL3XX00  | 1,200 AED | 1h

@priya now has 6 / 8 tickets.
```

### Assign single order
```
✅ Assigned AE136JM2JF00 to @ravi

Order: 260 AED | Status: PENDING | Stuck: 6h

@ravi now has 4 / 10 tickets.
```

## Error Cases

### Agent at capacity
```
⚠️ @{agent} is at capacity ({current} / {max} tickets)

Options:
1. Wait for them to resolve some tickets
2. Assign to another agent
3. Override: `assign {order_id} to {agent} --force`
```

### No unassigned orders
```
✨ No unassigned stuck orders!

All {total} stuck orders are already assigned:
- @ravi: 5 tickets
- @priya: 3 tickets
- @dinesh: 2 tickets
```

### Order already assigned
```
⚠️ Order {order_id} is already assigned to @{current_agent}

To reassign: `reassign {order_id} to {new_agent}`
```

## Bulk Distribution
```
distribute tickets
```
Evenly distributes all unassigned orders across available agents based on capacity.

## Guardrails
- Only assign orders that exist in Redshift
- Respect agent capacity limits (from Agents tab)
- Never write to Redshift
- All writes go to Google Sheets (Assignments tab)
