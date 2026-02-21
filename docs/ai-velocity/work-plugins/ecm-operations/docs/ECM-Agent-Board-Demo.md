# ECM Agent — Board Demo

> **Date:** February 11, 2026
> **Presenter:** ECM Operations Team
> **Duration:** 15 minutes

---

## Executive Summary

**The Problem:** Remittance orders get stuck at various points — payments, compliance, partners. Ops teams manually check 8+ dashboards to diagnose and fix each one.

**The Solution:** An AI agent that instantly diagnoses issues, prioritizes by impact, and guides resolution with step-by-step runbooks.

| Metric | Before | After |
|--------|--------|-------|
| Time to triage 10 orders | 45 min | **30 seconds** |
| Knowledge required | Senior Ops | Any L1 agent |
| Steps per order | 8 dashboard checks | **1 command** |
| Diagnosis consistency | Variable | **22 mapped patterns** |

---

## Live Demo Scenarios

### Scenario 1: Dashboard View

**Command:** `run ECM` or `stuck orders`

**What it shows:**
- All orders stuck > 12 hours
- Prioritized by business impact (Sentinel algorithm)
- Grouped by team (Ops, KYC_ops, VDA_ops)

**Sample Output:**
```
🚨 ECM Dashboard — 10 stuck orders

| # | Order ID     | Amount        | Age    | Issue           | Priority |
|---|--------------|---------------|--------|-----------------|----------|
| 1 | AE126XD9MS00 | 8,554 AED     | 30 days| Status Sync     | P1 🔴    |
| 2 | AE126XEWKY00 | 12,000 AED    | 30 days| Investigate     | P1 🔴    |
| 3 | AE126X9LMS00 | 4,700 AED     | 30 days| Investigate     | P2 🟠    |

💰 Total at risk: ₹8,31,285
```

---

### Scenario 2: Order Deep Dive

**Command:** `order AE126XD9MS00`

**What it shows:**
- Instant diagnosis with plain English explanation
- Step-by-step resolution instructions
- Customer context (masked PII)
- Priority score breakdown

**Sample Output:**
```
## 🔶 AE126XD9MS00 | P1 | Ops

### 😰 What's Wrong

Hey team! **Sidhique's** order has been stuck for **30 DAYS**!

- 💳 Customer paid **8,554 AED** via Checkout ✅
- 🏦 Lulu processed it and shows **CREDITED** ✅
- 🚫 But GOMS still shows **PENDING** — webhook missed!

### 🛠️ What To Do

1. 🔍 Open **AlphaDesk** → Search `AE126XD9MS00`
2. ✅ Verify Lulu shows **CREDITED**
3. 🔄 Click "Replay Webhook" or "Force Status Update"
4. ✅ Verify GOMS now shows **COMPLETED**
5. 📝 Run: `resolve AE126XD9MS00 "Force-synced"`

### 📊 Order Facts

| Field    | Value                          |
|----------|--------------------------------|
| Status   | PENDING / CNR_RESERVED_WAIT    |
| Amount   | 8,554 AED → ₹2,10,000 INR      |
| Age      | 720h (30 days) 🔴              |
| Payment  | ✅ COMPLETED via Checkout       |
| Lulu     | ✅ CREDITED                     |

### 👤 Customer

| Field | Value            |
|-------|------------------|
| Name  | Sidhique K.      |
| Email | sid***@gmail.com |
| Phone | ***9691          |

**SLA:** 1h | **Escalation:** TechOps
```

---

### Scenario 3: Agent Queue

**Command:** `my tickets`

**What it shows:**
- Agent's assigned tickets from Google Sheets
- Live data from Redshift (hours stuck, current status)
- Actionable instructions per ticket
- SLA status (breached/warning/ok)

**Sample Output:**
```
🎫 @snita's Queue — 2 tickets

1. 🔴 P1 | AE120PTGL200 | 10,000 AED | 729h | SLA: BREACHED

   Diagnosis: stuck_at_lulu

   ┌─────────────────────────────────────────────┐
   │ 🛠️ WHAT TO DO:                              │
   │ 1. Open LEANTECH Dashboard                  │
   │ 2. Search for AE120PTGL200                  │
   │ 3. Initiate refund for 10,000 AED           │
   │                                             │
   │ ✅ DONE? → resolve AE120PTGL200 "Refund done"|
   └─────────────────────────────────────────────┘

2. 🔴 P1 | AE134VEAJF00 | 25,000 AED | 209h | SLA: BREACHED
   → REFUND 25,000 AED via LEANTECH
   → Type `order AE134VEAJF00` for details

📊 Your Stats: 2/10 tickets | 67% SLA met
```

---

### Scenario 4: Resolution Flow

**Command:** `resolve AE120PTGL200 "Refund initiated, ref #LT20260211"`

**What happens:**
1. ✏️ Appends row to **Resolutions** tab (timestamp, notes, SLA status)
2. ✏️ Updates **Assignments** tab (Status → RESOLVED)
3. ✅ Shows confirmation to agent

**Sample Output:**
```
✅ TICKET RESOLVED

📋 Order: AE120PTGL200
👤 Agent: snita@aspora.com
📝 Notes: Refund initiated, ref #LT20260211
⏱️ Resolution: 4,290 min | SLA: MISSED
✅ Diagnosis: CORRECT | Action: FOLLOWED

🎫 Remaining queue: 1 ticket
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   📊 REDSHIFT                 📋 GOOGLE SHEETS              │
│   (Read-only)                 (Operations DB)               │
│                                                             │
│   • orders_goms               • Assignments                 │
│   • payments_goms             • Resolutions                 │
│   • lulu_data                 • Escalations                 │
│   • falcon_transactions       • Agents                      │
│   • transfer_rfi              • Daily Stats                 │
│                                                             │
│          └──────────────┬──────────────┘                    │
│                         ▼                                   │
│               ┌─────────────────┐                           │
│               │   🤖 ECM AGENT  │                           │
│               │                 │                           │
│               │ Diagnoses       │◄── stuck-reasons.yaml     │
│               │ Prioritizes     │◄── Sentinel scoring       │
│               │ Guides          │◄── runbooks/*.md          │
│               │ Tracks          │                           │
│               └────────┬────────┘                           │
│                        ▼                                    │
│               ┌─────────────────┐                           │
│               │   👤 OPS AGENT  │                           │
│               │                 │                           │
│               │ "my tickets"    │                           │
│               │ "order X"       │                           │
│               │ "resolve X"     │                           │
│               └─────────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Instant Diagnosis
- Queries 8 systems in parallel
- Computes `stuck_reason` using Metabase SOP logic
- 22 distinct stuck patterns mapped

### 2. Priority Scoring (Sentinel Algorithm)
```
score = 0.25 × age + 0.20 × amount + 0.25 × severity + 0.15 × rfi + 0.10 × payment

Priority: P1 (≥0.7) | P2 (≥0.5) | P3 (≥0.3) | P4 (<0.3)
```

### 3. Runbook-Guided Resolution
- Every `stuck_reason` maps to a runbook
- Step-by-step instructions
- Escalation contacts included

### 4. Full Tracking
- Google Sheets as operations database
- Resolution time, SLA status, diagnosis accuracy
- Agent performance metrics

---

## Team Performance (Sample Week)

| Agent   | Resolved | Avg Time | SLA Met | Accuracy |
|---------|----------|----------|---------|----------|
| Dinesh  | 3        | 1,560m   | 67%     | 100% ✅  |
| Snita   | 4        | 1,163m   | 50%     | 75%      |
| Aakash  | 2        | 182m     | 100%    | 100% ✅  |
| Akshay  | 4        | 17m      | 100%    | 25% ⚠️   |

**Key Insight:** 31% false positive rate on `status_sync_issue` — tuning needed.

---

## Production Readiness

| Component | Status | Notes |
|-----------|--------|-------|
| ✅ Redshift Queries | Ready | Optimized for <10s |
| ✅ Google Sheets | Ready | Read + Write working |
| ✅ Stuck Reason Mapping | Ready | 22 patterns |
| ✅ Priority Scoring | Ready | Sentinel integrated |
| ✅ Runbooks | Ready | 25 playbooks |
| ⚠️ Output Templates | 90% | Need emoji updates |
| ⚠️ False Positive Rate | 31% | Tune detection |

---

## Pilot Proposal

**Ask:** Approve 2-week pilot with Ops team (4 agents)

**Success Metrics:**
- 📉 Reduce avg resolution time by 50%
- 📈 Increase diagnosis accuracy to 90%+
- 💰 Process ₹10L+ in stuck orders
- 😊 Agent satisfaction > 4/5

---

## Appendix: Available Commands

| Command | Description |
|---------|-------------|
| `run ECM` | Dashboard of all stuck orders |
| `order {id}` | Deep dive on specific order |
| `my tickets` | Agent's assigned queue |
| `resolve {id} "{notes}"` | Close a ticket |
| `escalate {id} "{reason}"` | Escalate to senior |
| `assign tickets` | Manager: distribute work |
| `triage` | Manager: daily briefing |

---

*Generated by ECM Agent | February 2026*
