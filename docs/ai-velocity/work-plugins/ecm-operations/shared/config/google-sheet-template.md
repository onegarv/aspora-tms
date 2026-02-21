# ECM Google Sheet — Operations Database

**Google Sheets is the source of truth for agent operations.** Redshift is read-only for order data.

## Sheet Name
`ECM Operations` (or your preferred name)

---

## Tab 1: Assignments

**Who is working on what.** This is the "database" for `my tickets`.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| A: Order ID | Text | ✅ | Order ID from Redshift |
| B: Assigned Agent | Text | ✅ | Agent email/username |
| C: Assigned At | DateTime | ✅ | When assigned |
| D: Status | Text | ✅ | OPEN / IN_PROGRESS / RESOLVED / ESCALATED |
| E: Priority | Text | | P1 / P2 / P3 / P4 (optional override) |
| F: Notes | Text | | Any notes from assignment |

### Sample Data
```
Order ID      | Assigned Agent | Assigned At         | Status      | Priority | Notes
AE136JLXUG00  | ravi@aspora.com| 2026-02-04 08:00:00 | OPEN        | P2       | 
AE136JM2JF00  | ravi@aspora.com| 2026-02-04 08:15:00 | IN_PROGRESS | P1       | VIP customer
AE136JM6JF00  | priya@aspora.com| 2026-02-04 08:30:00| OPEN        |          |
```

### How "my tickets" works
1. Agent says "my tickets"
2. Claude reads **Assignments** tab filtered by `Assigned Agent = {agent}` and `Status IN (OPEN, IN_PROGRESS)`
3. Claude joins with Redshift order data to get amounts, stuck time, etc.
4. Shows the queue

---

## Tab 2: Resolutions

**Completed tickets.** Written when agent says `resolve {order_id} "{notes}"`.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| A: Timestamp | DateTime | ✅ | When resolved |
| B: Order ID | Text | ✅ | Order ID |
| C: Agent | Text | ✅ | Who resolved it |
| D: Resolution Notes | Text | ✅ | What was done |
| E: Assigned At | DateTime | | When originally assigned |
| F: Resolution Time (min) | Number | | Calculated: Timestamp - Assigned At |
| G: SLA Target (min) | Number | | From diagnosis mapping |
| H: SLA Status | Text | | MET or MISSED |
| I: Stuck Reason | Text | | From Redshift |
| J: Amount | Number | | Transaction amount |
| K: Currency | Text | | AED / GBP / EUR |
| L: User Segment | Text | | VIP / active / etc |

### Sample Data
```
Timestamp           | Order ID      | Agent          | Resolution Notes               | Assigned At         | Time | SLA  | Status | Reason            | Amount | Curr
2026-02-04 10:30:00 | AE136JM2JF00  | ravi@aspora.com| Replayed webhook, LULU confirmed| 2026-02-04 08:15:00 | 135  | 240  | MET    | CNR_RESERVED_WAIT | 260    | AED
```

---

## Tab 3: Escalations

**Tickets that needed senior help.** Written when agent says `stuck {order_id} "{reason}"`.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| A: Timestamp | DateTime | ✅ | When escalated |
| B: Order ID | Text | ✅ | Order ID |
| C: Escalated By | Text | ✅ | Agent who escalated |
| D: Reason | Text | ✅ | Why escalation needed |
| E: Order Status | Text | | From Redshift |
| F: Stuck Hours | Number | | How long stuck |
| G: Amount | Number | | Transaction amount |
| H: Currency | Text | | AED / GBP / EUR |
| I: Priority | Text | | P1/P2/P3/P4 |
| J: Assigned To | Text | | Who picked it up (filled later) |
| K: Resolved At | DateTime | | When resolved (filled later) |
| L: Resolution Notes | Text | | How resolved (filled later) |

---

## Tab 4: Agents

**Agent roster.** For assignment and identification.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| A: Email | Text | ✅ | Agent email (primary key) |
| B: Name | Text | ✅ | Display name |
| C: Team | Text | | Ops / KYC_ops / VDA_ops |
| D: Slack Handle | Text | | For notifications |
| E: Active | Boolean | | TRUE if currently working |
| F: Max Tickets | Number | | Max concurrent tickets (default: 10) |

### Sample Data
```
Email              | Name  | Team | Slack   | Active | Max
ravi@aspora.com    | Ravi  | Ops  | @ravi   | TRUE   | 10
priya@aspora.com   | Priya | Ops  | @priya  | TRUE   | 8
dinesh@aspora.com  | Dinesh| Ops  | @dinesh | TRUE   | 15
```

---

## Tab 5: Daily Stats (Auto-calculated)

**Aggregated stats.** Can be formulas or written by Claude.

| Column | Type | Description |
|--------|------|-------------|
| A: Date | Date | Date |
| B: Agent | Text | Agent email |
| C: Resolved | Number | Count from Resolutions tab |
| D: Avg Time (min) | Number | Average resolution time |
| E: SLA Met % | Number | % of resolutions meeting SLA |
| F: Escalated | Number | Count from Escalations tab |

### Formulas (in row 2, copy down)
```
C2: =COUNTIFS(Resolutions!C:C, B2, Resolutions!A:A, ">="&A2, Resolutions!A:A, "<"&A2+1)
D2: =AVERAGEIFS(Resolutions!F:F, Resolutions!C:C, B2, Resolutions!A:A, ">="&A2, Resolutions!A:A, "<"&A2+1)
E2: =COUNTIFS(Resolutions!C:C, B2, Resolutions!H:H, "MET", Resolutions!A:A, ">="&A2)/C2*100
```

---

## Tab 6: Pattern Intelligence

**Systemic failure pattern tracking.** Written by the Pattern Intelligence skill after each daily run.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| A: Date (IST) | DateTime | Yes | Run timestamp in IST (UTC+5:30) |
| B: Pattern_ID | Text | Yes | P1, P2, P3... (rank by impact score) |
| C: Stuck_Reason | Text | Yes | Mapped stuck_reason, "novel", or "uncategorized" |
| D: Currency | Text | Yes | AED / GBP / EUR |
| E: Signature | Text | Yes | State combo: sub_state\|lulu\|falcon\|payout\|rfi |
| F: Order_Count | Number | Yes | Orders in this cluster |
| G: Total_Amount | Number | Yes | Sum of send_amount |
| H: Avg_Hours | Number | Yes | Average hours stuck |
| I: High_Value_Count | Number | Yes | Orders >= 5,000 in cluster |
| J: Critical_Count | Number | Yes | Orders > 36h stuck in cluster |
| K: Delta_vs_Previous | Text | Yes | UP +N / DOWN -N / NEW / GONE / baseline |
| L: Recommendation | Text | Yes | One-line action from stuck-reasons.yaml |

### How It Works
1. Pattern Intelligence skill runs clustering query against Redshift
2. Groups orders by failure signature, scores by impact
3. Compares to previous day's rows (same tab) for trends
4. Appends new rows with current date — historical data preserved for trend analysis

---

## How It All Works

### "my tickets" Flow
```
1. Read Assignments tab → filter by agent, status = OPEN/IN_PROGRESS
2. Get order_ids from that list
3. Query Redshift for order details (amounts, stuck time, etc.)
4. Join and display
```

### "resolve {id} {notes}" Flow
```
1. Find order in Assignments tab
2. Get assignment timestamp → calculate resolution time
3. Append row to Resolutions tab
4. Update Assignments tab → Status = RESOLVED
5. Show confirmation
```

### "stuck {id} {reason}" Flow
```
1. Append row to Escalations tab
2. Update Assignments tab → Status = ESCALATED
3. (Optional) Notify Slack
4. Show confirmation
```

### Assignment Flow (assign tickets)
```
1. Run ECM pending list from Redshift
2. Filter out already-assigned orders (check Assignments tab)
3. Add rows to Assignments tab for new assignments
4. Agent can now see them in "my tickets"
```

---

## Setup

Google Sheets access is handled through `ecm-gateway` — no local credentials needed.

1. **Create Google Sheet** named "ECM Operations"
2. **Create 5 tabs**: Assignments, Resolutions, Escalations, Agents, Daily Stats
3. **Add headers** as shown above
4. **Add agents** to Agents tab
5. **Start assigning!**
