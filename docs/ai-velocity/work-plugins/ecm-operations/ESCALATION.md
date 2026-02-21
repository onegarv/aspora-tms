# ECM Escalation Matrix

Reference: Ops SOPs (Lulu flows, Partner-dependent, CNR, Edge Case Management). Use these contacts only as defined in runbooks and SOPs — do not invent escalation paths.

---

## TechOps Team (Vance / Aspora)

| Level | Contact | Use When |
|-------|---------|----------|
| **Level 1** | albert.roshan@pearldatadirect.com, sreejith.parameswaran@pearldatadirect.com | First escalation for tech/sync issues |
| **Level 2** | ajaimoan.antony@pearldatadirect.com, vinu.xavier@pearldatadirect.com | Breach of SLA, unresolved tech issues |
| **Level 3** | sreekanth.ramakrishnan@pearldatadirect.com, ankur.sharma@pearldatadirect.com | Critical / repeated failures |

**Exception:** Status sync error (e.g. transaction incorrectly marked Processing Deal In) → raise internally to TechOps.

---

## FinOps / Lulu Accounts (UAE)

| Purpose | Contact |
|---------|---------|
| **Lulu Accounts (UAE)** | accounts.uae@ae.luluexchange.com |
| **Lulu FinOps** | shehin.abdul@ae.luluexchange.com, gokul.gopikumar@ae.luluexchange.com |

**Use for:** Excess credit sheet, reconciliation, refund coordination, BRN/refund queries.

---

## Lulu Support (Agent / Operations)

| Purpose | Contact |
|---------|---------|
| **Pending non-terminal transactions (batch)** | agent.support@pearldatadirect.com |
| **Peak (>50 txns)** | Share SQL output with Lulu via email to above |
| **Txn Released / RFI peak (>10 txns)** | agent.support@pearldatadirect.com |

---

## Partner-Dependent Cases (HDFC / Yes Bank)

| Role | Owner |
|------|--------|
| **Day 1–3** | Normal follow-ups per SLA — @Rakesh M |
| **Day 4** | Level-2 escalation to partner SPOC + internal DL / WhatsApp — @Rakesh M |
| **Day 5–6** | Loop manager: collective cases, ageing, group email/WhatsApp — @Rakesh M |

**Tag:** `>3D_PENDING_PARTNER` in Partner Pending Tracker.

---

## CNR (Completed Not Received)

| Role | Owner |
|------|--------|
| **Operations Owner** | @Vishnu R |
| **CX Owner** | @Asaf Ali |
| **Tech Owner** | @Raj Vishwakarma |

Manual payout / SLA breach: log in Emergency Payout Sheet – CNR SLA Breach Tab; CX owns recovery if double credit.

---

## Edge Case Management (ECM)

| Threshold | Action | Owner |
|-----------|--------|-------|
| **16h – Warning** | Tag to owner, add to Incident Log (if >10 orders), capture TTD, check SOP link, start RCA stub | Ops |
| **24h – Action Required** | Escalate to @dinesh + Product/Backend, confirm customer comms, add TTM, update ECM dashboard | Ops / Dinesh |
| **48h – Critical** | Daily updates, roll-back/guard-rail or hotfix, add TTR | Ops + Backend |

**Daily ritual:** Reconcile ECM Report + Incident Log counts; 16h/24h/48h breach queues; post 24h dump on ops-alert (except VDA Bulk).

---

## SLA Summary (from SOPs)

| Scenario | SLA | Action After Breach |
|----------|-----|---------------------|
| Processing deal in – Payment pending / Txn transmitted | 3 hours | Escalate |
| Processing deal in – Payment awaiting clearance | T+1 | Escalate |
| Processing deal in – AML check / Txn released | 4 hours | Escalate |
| Excess credit – Refund | T+3 from user details | UAE FTS within 2 working days |
| Partner-dependent (HDFC/Yes Bank) | >3 working days | Level-2 escalation, WhatsApp alert |
| CNR – Transferred to correct account | 72 hours | Manual payout process per CNR SOP |
| Edge case (new / unmonitored) | 72 hours | Identify → assign → fix → improve |

Do not invent escalation recipients or SLAs; use only the above or the specific runbook/SOP referenced for the case type.
