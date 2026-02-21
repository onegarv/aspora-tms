# Aspora Treasury Management System (TMS)
## Complete Technical Package

**Version:** 1.0 | **Date:** 2026-02-20 | **Status:** Draft

---

# PART 1 — PRODUCT REQUIREMENTS DOCUMENT

## 1. Executive Summary

Aspera operates a prefunded remittance model (Rupee Drawing Arrangement) for NRI corridors (USD→INR, GBP→INR, EUR→INR, AED→INR). The TMS is a multi-agent system comprising three autonomous but coordinated agents — **Liquidity**, **FX Analyst**, and **Operations** — that collectively forecast daily INR requirements, optimize FX deal timing, and execute fund movements under strict banking-hour and compliance constraints.

---

## 2. Agent 1 — Liquidity Agent

### 2.1 Purpose
Predict the total INR payout requirement for each business day and determine how much capital (in each source currency) must be prefunded into bank nostro accounts before the market opens.

### 2.2 Inputs
| Source | Data |
|---|---|
| Internal DB | Historical daily payment volumes by corridor (USD, GBP, EUR, AED) — 12+ months |
| Internal DB | Current RDA balances per currency per bank |
| FX Agent | Current spot rates for USD/INR, GBP/INR, EUR/INR |
| Calendar service | India/US/UK/UAE holiday calendars, salary-cycle calendar |
| Config | Multiplier table (payday boost, holiday-adjacent boost, favorable-rate elasticity curve) |

### 2.3 Decision Logic
1. **Base forecast:** Weighted moving average of payment volume for same weekday, same week-of-month, trailing 8 weeks — with exponential decay weighting.
2. **Multiplier stack (multiplicative):**
   - `payday_multiplier`: 1.3–1.8× on 25th–1st of month (configurable).
   - `holiday_multiplier`: 1.2× day before Indian public holiday; 0.6× on day after (pent-up demand shifts).
   - `fx_elasticity_multiplier`: When current rate is >1σ above 30-day mean, apply `1 + 0.15 × z_score` (users remit more when INR is weak). Capped at 1.5×.
   - `day_of_week_multiplier`: Learned per-weekday factor from historical data.
3. **Currency split:** Allocate total INR requirement to source currencies proportional to historical corridor mix, adjusted for current rate attractiveness.
4. **RDA sufficiency check:** Compare required prefunding per currency to current nostro balance. Flag shortfall if `required > balance × 0.9` (10% safety buffer).

### 2.4 Outputs
| Output | Consumer |
|---|---|
| `daily_inr_forecast` (in crores, with confidence interval) | FX Agent, Ops Agent, Dashboard |
| `currency_split` { USD: X, GBP: Y, EUR: Z } | FX Agent |
| `rda_shortfall_alert` per currency | Ops Agent (triggers fund movement) |
| `forecast_confidence` (low / medium / high) | FX Agent (affects hedging aggressiveness) |

### 2.5 Edge Cases
- **Double holiday:** India holiday + US holiday on consecutive days — forecast must look ahead 2 days and prefund accordingly on the last open day.
- **Month-end + holiday overlap:** Payday and holiday multipliers stack; cap total multiplier at 2.5× to prevent over-prefunding.
- **Sudden rate spike:** If intra-day rate moves >1% from morning open, re-run forecast mid-day and alert FX Agent.
- **Data gap:** If volume data is missing for a day (system outage), fall back to corridor-level monthly average ÷ business days.

---

## 3. Agent 2 — FX Analyst / Trading Agent

### 3.1 Purpose
Determine optimal timing, sizing, and instrument selection for FX deal booking to cover the day's INR exposure at the best achievable rate, while managing risk within defined limits.

### 3.2 Inputs
| Source | Data |
|---|---|
| Liquidity Agent | `daily_inr_forecast`, `currency_split`, `forecast_confidence` |
| Market feeds | Real-time spot, TOM, cash rates from bank deal desk (9 AM–3:30 PM IST) |
| Composite Rate Engine | Continuous implied-spot from multiple sources (see §3.3.1) |
| News/sentiment | RSS/API feeds: RBI announcements, FOMC, UK BoE, US CPI/NFP, India trade deficit |
| Internal DB | Historical deal logs (rate achieved vs. RBI reference rate, slippage analysis) |
| Config | Risk limits: max single-deal size, max open exposure %, stop-loss thresholds |

### 3.3 Decision Logic

#### 3.3.1 Composite Rate Engine (24/7 Price Discovery)

USD/INR onshore spot trades only **9 AM – 3:30 PM IST**. To maintain continuous price discovery, the FX Agent consumes a priority-ranked feed:

| IST Window | Primary Source | Backup | Confidence |
|---|---|---|---|
| 09:00–15:30 | Onshore spot (bank quotes) | NSE currency futures | 1.0 |
| 15:30–17:00 | NSE/BSE currency futures | CME INR futures | 0.9 |
| 17:00–20:30 | DGCX INR futures (Dubai) | Early London NDF | 0.75 |
| 20:30–02:30 | London NDF + CME futures | CME futures | 0.7 |
| 02:30–06:30 | CME INR futures (NY session) | Thin NDF | 0.6 |
| 06:30–09:00 | Early Asia NDF + CME | CME pre-open | 0.5 |

**Key design decisions:**
- All non-spot sources are **basis-adjusted** to implied spot (removing cost-of-carry for futures, NDF premium/discount). Basis is calibrated daily at 15:30 IST against onshore close.
- GBP/INR and EUR/INR can be **synthesized** from USD/INR × GBP/USD or EUR/USD (majors trade 24h and are always liquid).
- Every composite rate carries a **confidence score** (source reliability × liquidity/spread width). The FX Agent won't auto-book tranches based on low-confidence rates.
- **Basis anomaly detection:** If NDF-spot basis blows out beyond 50 paise, it signals market dislocation → pause automated trading.

#### 3.3.2 Pre-Market Phase (6 AM–9 AM IST)
- Ingest overnight composite rate (CME/NDF) and compute overnight move vs. yesterday's onshore close.
- Generate **directional signal**: bullish-INR / bearish-INR / neutral, with magnitude score (−5 to +5).
- Monitor NDF-spot basis: if wider than 50 paise, flag potential market dislocation.
- If strong directional signal + high liquidity confidence → pre-commit to aggressive early booking.
- Produce `market_brief` for human review.

#### 3.3.3 Active Trading Phase (9 AM–3:30 PM IST)
- **Tranche strategy:** Split total exposure into N tranches (default 3–5). Book tranches based on:
  - **Time-weighted baseline:** Equal tranches at 9:30, 11:00, 13:00, 14:30.
  - **Signal-adjusted:** If bearish-INR signal, front-load (book 50%+ in first tranche). If bullish-INR, back-load.
  - **VWAP-chase mode:** If intra-day volatility < threshold, use market microstructure to chase VWAP.
- **Instrument selection:**
  - Default: **Spot** (best rate, T+2 settlement) — use when INR liquidity can wait 2 days.
  - **TOM**: When tomorrow's prefunding is critical and spot won't settle in time.
  - **Cash**: Emergency same-day settlement; accept the cash premium only when same-day INR shortfall is confirmed.
- **Stop-loss:** If cumulative booked rate is worse than RBI reference rate by > X paise (configurable), pause and escalate to human.
- **Exposure cap:** Never leave more than 30% of daily exposure unhedged past 2 PM IST.

#### 3.3.4 Post-Market Phase (3:30 PM–6 AM IST)
- Calibrate basis: record onshore close vs. each source's concurrent price. This calibration feeds the next day's implied-spot calculation.
- Monitor NDF/CME for next-day positioning.
- Generate end-of-day P&L: compare achieved blended rate vs. RBI reference rate.
- Feed performance data back for model retraining.

### 3.4 FX Forecast Model
- **1-hour:** LSTM/transformer on 5-min NDF tick data + order flow features. Target: direction + magnitude.
- **1-day:** Ensemble of (a) technical features from NDF/onshore rates, (b) macro surprise index, (c) sentiment score from news NLP. Output: point estimate + 80% confidence band.
- **1-week:** Macro-driven regression (US-India rate differential, oil prices, FII flows, trade balance). Lower weight in real-time decisions; used for strategic hedging.

### 3.5 Outputs
| Output | Consumer |
|---|---|
| `deal_instructions` { currency, amount, instrument, target_rate, time_window } | Ops Agent (for execution) |
| `market_brief` (pre-market and intra-day) | Dashboard, human traders |
| `exposure_status` { total, covered, open, blended_rate } | Liquidity Agent, Dashboard |
| `eod_pnl_report` | Analytics, model retraining pipeline |

### 3.6 Edge Cases
- **Flash crash / spike:** If rate moves >0.5% in 5 min, freeze auto-booking and alert human. Resume after 15-min cooldown or manual override.
- **Bank deal desk unreachable:** Retry 3× with 30s backoff, then switch to backup bank. Alert Ops Agent.
- **Settlement date collision:** If T+2 lands on Indian holiday, auto-select TOM or cash to avoid settlement failure.
- **Partial fill:** If bank can only fill partial amount, log remainder as open exposure and re-queue.

---

## 4. Agent 3 — Operations Agent

### 4.1 Purpose
Execute and monitor all fund movements (operating account → nostro), enforce compliance workflows (maker-checker), and manage banking-hour and holiday constraints across four jurisdictions.

### 4.2 Inputs
| Source | Data |
|---|---|
| Liquidity Agent | `rda_shortfall_alert`, `currency_split` |
| FX Agent | `deal_instructions` (triggers post-deal nostro top-up if needed) |
| Bank APIs / SWIFT | Account balance confirmations, transfer status |
| Calendar service | US Fed, BoE, RBI, UAE CB holiday calendars |
| Config | Account mappings, cut-off times, approval thresholds |

### 4.3 Fund Movement Windows
| Currency | Rail | Hours (local) | IST equivalent (winter) | IST equivalent (summer) | Holiday Calendar |
|---|---|---|---|---|---|
| USD | Fedwire | 9:00 AM – 6:00 PM ET | 7:30 PM – 4:30 AM (+1d) | 6:30 PM – 3:30 AM (+1d) | US Federal Reserve |
| GBP | CHAPS | 8:00 AM – 4:00 PM UK | 1:30 PM – 9:30 PM | 12:30 PM – 8:30 PM | Bank of England |
| EUR | SEPA/TARGET2 | 7:00 AM – 6:00 PM CET | 11:30 AM – 10:30 PM | 10:30 AM – 9:30 PM | TARGET2 calendar |
| AED | → USD conversion | Follows USD rail | Follows USD | Follows USD | UAE CB + US Fed |
| INR (deal booking) | Bank treasury desk | 9:00 AM – 3:00 PM IST | — | — | RBI / NSE holidays |

> ⚠️ IST equivalents shift with US/UK/EU DST transitions (US: second Sunday of March & first Sunday of November; UK/EU: last Sunday of March & October). The values above are approximate references only. Code uses IANA `ZoneInfo` identifiers and is always correct. For exact IST times on any date call `WindowManager.get_window(currency).window_ist_summary(date)`. DST transition warnings are surfaced proactively in the `ops.holiday.lookahead` event payload under the `dst_transitions` key.

### 4.4 Decision Logic

#### 4.4.1 Daily Pre-Open Routine (6 AM IST)
1. Receive `daily_inr_forecast` and `currency_split` from Liquidity Agent.
2. Check nostro balances via bank API.
3. If shortfall in any currency: calculate transfer amount = shortfall + 10% buffer. Check if the relevant fund movement window is open or will open today. If window is closed and won't open before INR market open (9 AM IST), **escalate immediately** — this is a critical prefunding failure.
4. Generate `fund_movement_proposal` → enter maker-checker queue.

#### 4.4.2 Maker-Checker Workflow
- **Maker** (system or junior ops): Proposes action with amount, source account, destination nostro, currency, rail, expected arrival time, purpose code.
- **Validation checks (automated):** Amount within daily/per-transaction limits. Destination account matches approved nostro registry. Transfer window is currently open (or will be before deadline). No duplicate transfer in last 2 hours for same amount + destination.
- **Checker** (senior ops / treasury head): Reviews proposal + validation results. Approves or rejects with comment. Approval required within 30 min of proposal; auto-escalate if not actioned. Transfers above threshold X require dual-checker (two senior approvals).
- **Execution:** On approval, submit transfer via bank API/portal. Log transaction ID.
- **Confirmation tracking:** Poll for settlement confirmation. Alert if not confirmed within expected SLA (Fedwire: 30 min, CHAPS: 2 hr, SEPA: 4 hr).

#### 4.4.3 Intra-Day Monitoring
- After each FX deal booked by FX Agent, verify that nostro balance can support the deal settlement.
- If cumulative deals exceed nostro balance, trigger top-up proposal.
- Track all pending transfers and their expected arrival times.

### 4.5 Outputs
| Output | Consumer |
|---|---|
| `fund_movement_status` { pending, approved, executed, confirmed } | Dashboard, Liquidity Agent |
| `window_closing_alert` (30 min before cut-off) | FX Agent (to rush any pending deals), Dashboard |
| `holiday_lookahead` (next 3 business days) | All agents (planning) |
| `transfer_confirmation` with settlement reference | Audit log, reconciliation |

### 4.6 Edge Cases
- **US holiday on Monday + India open:** GBP/EUR nostro can still be funded, but USD cannot. Liquidity Agent must shift corridor mix or pre-fund USD on Friday.
- **CHAPS outage:** Fall back to SWIFT gpi for GBP; accept longer settlement time and alert FX Agent to delay GBP deals.
- **Maker proposes transfer but window closes before checker approves:** Auto-reject with reason "window closed", reschedule for next available window, alert all agents.
- **Double-spend prevention:** Idempotency key on every transfer proposal; reject duplicates.
- **DST transitions:** US/UK clock changes shift IST equivalents; calendar service must account for this dynamically.

---

## 5. Inter-Agent Communication

### 5.1 Message Bus
All agents communicate via an async message bus (Redis Streams or Kafka). Messages are typed and versioned.

### 5.2 Key Flows

**Flow 1 — Morning Prefunding**
```
Liquidity Agent → [daily_inr_forecast, currency_split, rda_shortfall_alert]
    → FX Agent: receives exposure to plan tranches
    → Ops Agent: receives shortfall, initiates fund movement
```

**Flow 2 — Deal Booking Cycle**
```
FX Agent → [deal_instruction] → Ops Agent
Ops Agent → validates timing/balance → [deal_execution_status] → FX Agent
FX Agent → updates exposure_status → Liquidity Agent
```

**Flow 3 — Alert Cascade**
```
Ops Agent detects window closing in 30 min
    → FX Agent: "book remaining exposure NOW or wait until tomorrow"
    → Liquidity Agent: "if unhedged, tomorrow's forecast needs adjustment"
```

**Flow 4 — Emergency Re-forecast**
```
FX Agent detects >1% rate move
    → Liquidity Agent: "re-run forecast with updated rate"
    → Liquidity Agent re-forecasts → updated exposure → FX Agent adjusts tranches
```

---

## 6. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Forecast latency | < 5s for daily forecast generation |
| Deal instruction latency | < 2s from signal to instruction emission |
| Maker-checker SLA | Proposal surfaced to checker within 5s; auto-escalation at 30 min |
| Uptime | 99.9% during IST business hours (9 AM – 6 PM) |
| Audit trail | Every state change, decision, and approval immutably logged |
| Data retention | Tick data: 2 years. Deal logs: 7 years. Audit logs: 10 years |
| Security | All inter-service communication over mTLS. Secrets in Vault. SOC2 compliant |

## 7. Success Metrics

1. **FX efficiency:** Blended deal rate vs. RBI reference rate — target: within 2 paise on average.
2. **Prefunding accuracy:** Forecast vs. actual daily volume — target: <10% MAPE.
3. **Zero settlement failures** due to timing/holiday misses.
4. **Ops throughput:** Maker-checker cycle time < 15 min for routine transfers.
5. **Alert lead time:** Window-closing alerts issued ≥30 min before cut-off 100% of the time.

---
---

# PART 2 — SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ASPERA TMS                                  │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  LIQUIDITY   │◄──►│   MESSAGE    │◄──►│  FX ANALYST  │          │
│  │    AGENT     │    │     BUS      │    │    AGENT     │          │
│  └──────┬───────┘    │  (Redis /    │    └──────┬───────┘          │
│         │            │   Kafka)     │           │                   │
│         │            └──────┬───────┘           │                   │
│         │                   │                   │                   │
│         │            ┌──────┴───────┐           │                   │
│         └───────────►│  OPERATIONS  │◄──────────┘                   │
│                      │    AGENT     │                               │
│                      └──────┬───────┘                               │
│                             │                                       │
│  ┌──────────────────────────┼──────────────────────────────────┐   │
│  │              SHARED SERVICES LAYER                          │   │
│  │  ┌────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐│   │
│  │  │Calendar│ │ Account │ │  Audit  │ │ Alert  │ │  Auth  ││   │
│  │  │Service │ │ Ledger  │ │  Log    │ │ Router │ │ & RBAC ││   │
│  │  └────────┘ └─────────┘ └─────────┘ └────────┘ └────────┘│   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              COMPOSITE RATE ENGINE                          │   │
│  │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────┐ ┌────────┐│   │
│  │  │Onshore  │ │NSE/BSE   │ │ DGCX   │ │ CME  │ │  NDF   ││   │
│  │  │Spot Feed│ │Futures   │ │Futures │ │Futures│ │Offshore││   │
│  │  └────┬────┘ └────┬─────┘ └───┬────┘ └──┬───┘ └───┬────┘│   │
│  │       └───────────┴───────────┴──────────┴─────────┘      │   │
│  │                    ▼                                       │   │
│  │  ┌─────────────────────────────────────────────────────┐  │   │
│  │  │  Basis Adjuster → Priority Selector → Cross-Rate    │  │   │
│  │  │  Calibration      (by time of day)    Synthesizer   │  │   │
│  │  │                                       (GBP/EUR×USD) │  │   │
│  │  └──────────────────────┬──────────────────────────────┘  │   │
│  │                         ▼                                  │   │
│  │              Implied Spot (24/7)                           │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              OTHER EXTERNAL INTEGRATIONS                    │   │
│  │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐          │   │
│  │  │Bank APIs│ │News / NLP│ │  RBI   │ │Bloomberg│          │   │
│  │  │(Nostro) │ │ Pipeline │ │ Rates  │ │/Reuters │          │   │
│  │  └─────────┘ └──────────┘ └────────┘ └────────┘          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              DATA STORES                                    │   │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐│   │
│  │  │ Postgres │  │TimescaleDB│  │  Redis   │  │ S3/Blob  ││   │
│  │  │ (config, │  │(tick data,│  │ (cache,  │  │ (model   ││   │
│  │  │  deals)  │  │ volumes)  │  │ pub/sub) │  │artifacts)││   │
│  │  └──────────┘  └───────────┘  └──────────┘  └──────────┘│   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              DASHBOARD & CONTROLS                           │   │
│  │  ┌────────────┐  ┌───────────────┐  ┌──────────────┐     │   │
│  │  │  Treasury  │  │ Maker-Checker │  │    Admin     │     │   │
│  │  │  Dashboard │  │      UI       │  │   Console    │     │   │
│  │  └────────────┘  └───────────────┘  └──────────────┘     │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---
---

# PART 3 — CODEBASE SCAFFOLD

## 3.1 Directory Structure

```
aspera-tms/
├── pyproject.toml
├── docker-compose.yml
├── config/
│   ├── settings.py              # Global config, env vars
│   ├── risk_limits.yaml         # FX risk parameters
│   ├── multipliers.yaml         # Liquidity forecast multipliers
│   └── accounts.yaml            # Nostro account registry
├── agents/
│   ├── __init__.py
│   ├── base.py                  # Abstract agent base class
│   ├── liquidity/
│   │   ├── __init__.py
│   │   ├── agent.py             # Liquidity agent orchestrator
│   │   ├── forecaster.py        # Volume forecasting models
│   │   ├── multipliers.py       # Payday, holiday, FX elasticity
│   │   └── rda_checker.py       # RDA balance sufficiency
│   ├── fx_analyst/
│   │   ├── __init__.py
│   │   ├── agent.py             # FX agent orchestrator
│   │   ├── composite_rate.py    # 24/7 rate engine (see §3.4)
│   │   ├── market_intel.py      # News/sentiment ingestion
│   │   ├── forecaster.py        # Rate forecasting (1h/1d/1w)
│   │   ├── tranche_engine.py    # Tranche sizing & timing
│   │   └── risk_manager.py      # Stop-loss, exposure caps
│   └── operations/
│       ├── __init__.py
│       ├── agent.py             # Ops agent orchestrator
│       ├── fund_mover.py        # Transfer execution logic
│       ├── maker_checker.py     # Approval workflow
│       └── window_manager.py    # Banking hour & holiday logic
├── bus/
│   ├── __init__.py
│   ├── events.py                # Event type definitions
│   ├── publisher.py             # Publish to message bus
│   └── consumer.py              # Subscribe & dispatch
├── services/
│   ├── __init__.py
│   ├── calendar_service.py      # Multi-jurisdiction holiday calendar
│   ├── account_ledger.py        # Nostro balance tracking
│   ├── audit_log.py             # Immutable audit trail
│   └── alert_router.py          # Multi-channel alerting
├── integrations/
│   ├── __init__.py
│   ├── bank_api.py              # Bank deal desk & transfer API
│   ├── feeds/
│   │   ├── onshore_spot.py      # Bank dealer quote feed
│   │   ├── nse_futures.py       # NSE currency futures
│   │   ├── cme_futures.py       # CME INR futures (6I)
│   │   ├── dgcx_futures.py      # DGCX INR futures
│   │   ├── ndf_feed.py          # Offshore NDF
│   │   └── fx_majors.py         # GBP/USD, EUR/USD for cross-rates
│   ├── news_feed.py             # News/macro data ingestion
│   ├── rbi_rates.py             # RBI reference rate fetcher
│   └── bloomberg.py             # Bloomberg/Reuters adapter
├── models/
│   ├── __init__.py
│   ├── domain.py                # Core domain models
│   └── db.py                    # SQLAlchemy / ORM models
├── api/
│   ├── __init__.py
│   ├── dashboard.py             # Dashboard REST endpoints
│   ├── maker_checker_ui.py      # Approval UI endpoints
│   └── admin.py                 # Config management endpoints
├── tests/
│   ├── test_liquidity.py
│   ├── test_fx_analyst.py
│   ├── test_operations.py
│   ├── test_composite_rate.py
│   └── test_integration.py
└── scripts/
    ├── seed_holidays.py
    ├── backfill_volumes.py
    └── run_agents.py            # Entry point
```

## 3.2 Config & Domain Models

```python
# config/settings.py

from pydantic_settings import BaseSettings
from enum import Enum

class Currency(str, Enum):
    USD = "USD"
    GBP = "GBP"
    EUR = "EUR"
    AED = "AED"
    INR = "INR"

class DealType(str, Enum):
    SPOT = "spot"       # T+2
    TOM = "tom"         # T+1
    CASH = "cash"       # T+0

class Settings(BaseSettings):
    postgres_url: str = "postgresql://localhost:5432/aspera_tms"
    timescale_url: str = "postgresql://localhost:5433/aspera_ts"
    redis_url: str = "redis://localhost:6379"
    bus_type: str = "redis"
    kafka_brokers: str = "localhost:9092"

    # Risk limits
    max_single_deal_usd: float = 5_000_000
    max_open_exposure_pct: float = 0.30
    stop_loss_paise: int = 5
    flash_crash_threshold_pct: float = 0.5

    # Ops
    maker_checker_timeout_min: int = 30
    dual_checker_threshold_usd: float = 10_000_000
    prefunding_buffer_pct: float = 0.10

    # Forecast
    forecast_lookback_weeks: int = 8
    max_total_multiplier: float = 2.5

    class Config:
        env_prefix = "ASPERA_"

settings = Settings()
```

```python
# models/domain.py

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

@dataclass
class DailyForecast:
    forecast_date: date
    total_inr_crores: float
    confidence: str                          # "low" | "medium" | "high"
    currency_split: dict[str, float]         # {"USD": 45.2, "GBP": 12.1, ...}
    multipliers_applied: dict[str, float]
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RDAShortfall:
    currency: str
    required_amount: float
    available_balance: float
    shortfall: float
    severity: str  # "warning" | "critical"

@dataclass
class DealInstruction:
    id: str
    currency_pair: str
    amount_foreign: float
    amount_inr: float
    deal_type: str
    target_rate: float
    time_window_start: datetime
    time_window_end: datetime
    tranche_number: int
    total_tranches: int

@dataclass
class FundMovementProposal:
    id: str
    currency: str
    amount: float
    source_account: str
    destination_nostro: str
    rail: str
    proposed_by: str
    purpose: str
    idempotency_key: str
    status: str = "pending"
    approved_by: Optional[str] = None
    executed_at: Optional[datetime] = None
    settlement_ref: Optional[str] = None

@dataclass
class ExposureStatus:
    as_of: datetime
    total_inr_required: float
    covered_inr: float
    open_inr: float
    blended_rate: float
    deal_count: int
```

## 3.3 Event Bus

```python
# bus/events.py

from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass
class Event:
    event_type: str
    source_agent: str
    timestamp: datetime
    payload: dict[str, Any]
    correlation_id: str

# Event type constants
FORECAST_READY       = "forecast.daily.ready"
SHORTFALL_ALERT      = "forecast.rda.shortfall"
DEAL_INSTRUCTION     = "fx.deal.instruction"
EXPOSURE_UPDATE      = "fx.exposure.update"
MARKET_BRIEF         = "fx.market.brief"
REFORECAST_TRIGGER   = "fx.reforecast.trigger"
FUND_MOVEMENT_REQ    = "ops.fund.movement.request"
FUND_MOVEMENT_STATUS = "ops.fund.movement.status"
WINDOW_CLOSING       = "ops.window.closing"
HOLIDAY_LOOKAHEAD    = "ops.holiday.lookahead"
```

```python
# bus/publisher.py

import json
import redis.asyncio as aioredis

class EventPublisher:
    def __init__(self, redis_url: str):
        self._redis = aioredis.from_url(redis_url)

    async def publish(self, event):
        stream_key = f"tms:{event.event_type}"
        await self._redis.xadd(
            stream_key, {"data": json.dumps(event.__dict__, default=str)}
        )

    async def close(self):
        await self._redis.close()
```

```python
# bus/consumer.py

import json, asyncio
import redis.asyncio as aioredis
from typing import Callable, Awaitable

class EventConsumer:
    def __init__(self, redis_url: str, group: str, consumer_name: str):
        self._redis = aioredis.from_url(redis_url)
        self._group = group
        self._consumer = consumer_name
        self._handlers: dict[str, Callable] = {}

    def on(self, event_type: str, handler: Callable[[dict], Awaitable[None]]):
        self._handlers[event_type] = handler

    async def start(self):
        streams = {f"tms:{et}": ">" for et in self._handlers}
        for stream in streams:
            try:
                await self._redis.xgroup_create(stream, self._group, mkstream=True)
            except Exception:
                pass
        while True:
            results = await self._redis.xreadgroup(
                self._group, self._consumer, streams, count=10, block=1000
            )
            for stream_name, messages in results:
                et = stream_name.decode().replace("tms:", "")
                handler = self._handlers.get(et)
                if handler:
                    for msg_id, data in messages:
                        payload = json.loads(data[b"data"])
                        await handler(payload)
                        await self._redis.xack(stream_name, self._group, msg_id)
```

## 3.4 Agent Base Class

```python
# agents/base.py

from abc import ABC, abstractmethod
import logging

class BaseAgent(ABC):
    def __init__(self, name, publisher, consumer):
        self.name = name
        self.publisher = publisher
        self.consumer = consumer
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    async def setup(self):
        """Register event handlers and initialize state."""

    @abstractmethod
    async def run_daily(self):
        """Execute daily routine."""

    async def emit(self, event):
        self.logger.info(f"Emitting {event.event_type}")
        await self.publisher.publish(event)

    async def start(self):
        await self.setup()
        await self.consumer.start()
```

## 3.5 Liquidity Agent

```python
# agents/liquidity/agent.py

from datetime import date, datetime
import uuid

class LiquidityAgent(BaseAgent):
    def __init__(self, publisher, consumer, forecaster, multiplier_engine, rda_checker):
        super().__init__("liquidity", publisher, consumer)
        self.forecaster = forecaster
        self.multipliers = multiplier_engine
        self.rda_checker = rda_checker

    async def setup(self):
        self.consumer.on("fx.reforecast.trigger", self.handle_reforecast)
        self.consumer.on("fx.exposure.update", self.handle_exposure_update)

    async def run_daily(self):
        today = date.today()
        base = await self.forecaster.predict(today)
        mults = await self.multipliers.compute(today, current_rates=await self._get_rates())
        adjusted_total = min(base.total * mults.combined(), base.total * 2.5)
        split = await self.forecaster.currency_split(today, adjusted_total)
        shortfalls = await self.rda_checker.check(split)

        forecast = DailyForecast(
            forecast_date=today,
            total_inr_crores=adjusted_total,
            confidence=self._confidence_level(mults),
            currency_split=split,
            multipliers_applied=mults.as_dict(),
        )
        await self.emit(Event(
            event_type="forecast.daily.ready",
            source_agent=self.name,
            timestamp=datetime.utcnow(),
            payload=forecast.__dict__,
            correlation_id=str(uuid.uuid4()),
        ))
        for s in shortfalls:
            await self.emit(Event(
                event_type="forecast.rda.shortfall",
                source_agent=self.name,
                timestamp=datetime.utcnow(),
                payload=s.__dict__,
                correlation_id=str(uuid.uuid4()),
            ))

    async def handle_reforecast(self, payload):
        self.logger.warning(f"Re-forecast triggered: {payload.get('reason')}")
        await self.run_daily()

    async def handle_exposure_update(self, payload):
        pass  # Store for end-of-day MAPE calculation

    def _confidence_level(self, mults):
        if mults.combined() > 2.0: return "low"
        if mults.combined() > 1.5: return "medium"
        return "high"

    async def _get_rates(self):
        return {}  # TODO: integrate with composite rate engine cache
```

```python
# agents/liquidity/forecaster.py

import numpy as np
from datetime import date
from dataclasses import dataclass

@dataclass
class BaseForecast:
    total: float
    weekday: int

class VolumeForecaster:
    def __init__(self, db_session):
        self.db = db_session

    async def predict(self, target_date: date) -> BaseForecast:
        weekday = target_date.weekday()
        volumes = await self._fetch_historical(weekday, weeks=8)
        if not volumes:
            return await self._fallback_forecast(target_date)
        weights = np.exp(-0.3 * np.arange(len(volumes)))
        weights /= weights.sum()
        return BaseForecast(total=float(np.dot(volumes, weights)), weekday=weekday)

    async def currency_split(self, target_date: date, total_inr: float) -> dict:
        mix = await self._fetch_corridor_mix(weeks=4)
        return {ccy: round(total_inr * pct, 2) for ccy, pct in mix.items()}

    async def _fetch_historical(self, weekday, weeks):
        return []  # TODO: query TimescaleDB

    async def _fetch_corridor_mix(self, weeks):
        return {"USD": 0.55, "GBP": 0.20, "EUR": 0.15, "AED": 0.10}

    async def _fallback_forecast(self, target_date):
        return BaseForecast(total=50.0, weekday=target_date.weekday())
```

```python
# agents/liquidity/multipliers.py

from datetime import date
from dataclasses import dataclass

@dataclass
class MultiplierResult:
    payday: float
    holiday: float
    fx_elasticity: float
    day_of_week: float

    def combined(self):
        return self.payday * self.holiday * self.fx_elasticity * self.day_of_week

    def as_dict(self):
        return self.__dict__

class MultiplierEngine:
    def __init__(self, calendar_service, config):
        self.calendar = calendar_service
        self.config = config

    async def compute(self, target_date, current_rates) -> MultiplierResult:
        return MultiplierResult(
            payday=self._payday_factor(target_date),
            holiday=await self._holiday_factor(target_date),
            fx_elasticity=self._fx_elasticity(current_rates),
            day_of_week=self._dow_factor(target_date),
        )

    def _payday_factor(self, d):
        return self.config.get("payday_boost", 1.4) if (25 <= d.day or d.day <= 1) else 1.0

    async def _holiday_factor(self, d):
        if await self.calendar.is_day_before_holiday(d, "IN"): return 1.2
        if await self.calendar.is_day_after_holiday(d, "IN"): return 0.6
        return 1.0

    def _fx_elasticity(self, rates):
        return 1.0  # TODO: z-score based on 30-day mean

    def _dow_factor(self, d):
        factors = self.config.get("dow_factors", {0:1.0, 1:0.95, 2:1.0, 3:1.05, 4:1.1})
        return factors.get(d.weekday(), 1.0)
```

## 3.6 FX Analyst Agent

```python
# agents/fx_analyst/agent.py

import asyncio, uuid
from datetime import datetime

class FXAnalystAgent(BaseAgent):
    def __init__(self, publisher, consumer, rate_engine,
                 forecaster, tranche_engine, risk_manager, market_intel):
        super().__init__("fx_analyst", publisher, consumer)
        self.rates = rate_engine       # CompositeRateEngine
        self.forecaster = forecaster
        self.tranches = tranche_engine
        self.risk = risk_manager
        self.intel = market_intel
        self.daily_exposure = None

    async def setup(self):
        self.consumer.on("forecast.daily.ready", self.handle_forecast)
        self.consumer.on("ops.fund.movement.status", self.handle_fund_status)
        self.consumer.on("ops.window.closing", self.handle_window_closing)

    async def run_daily(self):
        usdinr = self.rates.get_rate(CurrencyPair.USDINR)
        sentiment = await self.intel.get_sentiment_score()
        signal = self.forecaster.generate_signal(usdinr, sentiment)

        brief = {
            "composite_rate": usdinr.mid if usdinr else None,
            "primary_source": usdinr.primary_source.value if usdinr else None,
            "confidence": usdinr.confidence if usdinr else 0,
            "signal": signal.direction,
            "magnitude": signal.magnitude,
            "key_events": await self.intel.get_upcoming_events(),
            "basis_alert": self.rates.check_basis_anomaly(CurrencyPair.USDINR),
            "recommendation": self._pre_market_rec(signal),
        }
        await self.emit(Event(
            event_type="fx.market.brief",
            source_agent=self.name,
            timestamp=datetime.utcnow(),
            payload=brief,
            correlation_id=str(uuid.uuid4()),
        ))

    async def handle_forecast(self, payload):
        self.daily_exposure = ExposureStatus(
            as_of=datetime.utcnow(),
            total_inr_required=payload["total_inr_crores"],
            covered_inr=0,
            open_inr=payload["total_inr_crores"],
            blended_rate=0, deal_count=0,
        )
        signal = await self._current_signal()
        plan = self.tranches.plan(
            exposure=self.daily_exposure,
            split=payload["currency_split"],
            signal=signal,
            confidence=payload["confidence"],
        )
        for t in plan.tranches:
            instr = DealInstruction(
                id=str(uuid.uuid4()),
                currency_pair=f"{t.currency}/INR",
                amount_foreign=t.amount_foreign,
                amount_inr=t.amount_inr,
                deal_type=t.instrument,
                target_rate=t.target_rate,
                time_window_start=t.window_start,
                time_window_end=t.window_end,
                tranche_number=t.number,
                total_tranches=plan.total_tranches,
            )
            await self.emit(Event(
                event_type="fx.deal.instruction",
                source_agent=self.name,
                timestamp=datetime.utcnow(),
                payload=instr.__dict__,
                correlation_id=str(uuid.uuid4()),
            ))

    async def handle_window_closing(self, payload):
        if self.daily_exposure and self.daily_exposure.open_inr > 0:
            self.logger.warning(f"Window closing! Open: {self.daily_exposure.open_inr} Cr")
            if self.risk.should_force_book(self.daily_exposure):
                await self._book_remaining()

    async def monitor_intraday(self):
        while self._is_trading_hours():
            rate = self.rates.get_rate(CurrencyPair.USDINR)
            if rate and self.risk.is_flash_event(rate.mid):
                self.logger.critical(f"Flash event: {rate.mid}")
                await asyncio.sleep(900)
            # Check for rate move > 1% → trigger reforecast
            if rate and self._is_large_move(rate.mid):
                await self.emit(Event(
                    event_type="fx.reforecast.trigger",
                    source_agent=self.name,
                    timestamp=datetime.utcnow(),
                    payload={"reason": "Rate move > 1%", "current_rate": rate.mid},
                    correlation_id=str(uuid.uuid4()),
                ))
            await asyncio.sleep(30)

    def _pre_market_rec(self, signal):
        if signal.magnitude > 3: return "Front-load — strong bearish INR signal"
        if signal.magnitude < -3: return "Back-load — INR likely to strengthen"
        return "Standard time-weighted tranches"

    async def _current_signal(self):
        rate = self.rates.get_rate(CurrencyPair.USDINR)
        sentiment = await self.intel.get_sentiment_score()
        return self.forecaster.generate_signal(rate, sentiment)

    def _is_trading_hours(self): return True  # TODO
    def _is_large_move(self, mid): return False  # TODO
    async def _book_remaining(self): pass  # TODO
    async def handle_fund_status(self, payload): pass
```

```python
# agents/fx_analyst/tranche_engine.py

from dataclasses import dataclass
from datetime import datetime, time

@dataclass
class Tranche:
    number: int
    currency: str
    amount_foreign: float
    amount_inr: float
    instrument: str
    target_rate: float
    window_start: datetime
    window_end: datetime

@dataclass
class TranchePlan:
    tranches: list
    total_tranches: int
    strategy: str

class TrancheEngine:
    SLOTS = [
        (time(9, 30), time(10, 30)),
        (time(11, 0), time(12, 0)),
        (time(13, 0), time(14, 0)),
        (time(14, 30), time(15, 0)),
    ]

    def plan(self, exposure, split, signal, confidence) -> TranchePlan:
        n = len(self.SLOTS)
        weights = self._signal_to_weights(signal, n)
        tranches = []
        for i, (ccy, amount_inr) in enumerate(self._expand(split, weights)):
            tranches.append(Tranche(
                number=i+1, currency=ccy,
                amount_foreign=0, amount_inr=amount_inr,
                instrument=self._pick_instrument(i, n, confidence),
                target_rate=0,
                window_start=datetime.combine(datetime.today(), self.SLOTS[i%n][0]),
                window_end=datetime.combine(datetime.today(), self.SLOTS[i%n][1]),
            ))
        strategy = ("front_loaded" if signal.magnitude > 2
                     else "back_loaded" if signal.magnitude < -2
                     else "time_weighted")
        return TranchePlan(tranches=tranches, total_tranches=len(tranches), strategy=strategy)

    def _signal_to_weights(self, signal, n):
        if signal.magnitude > 2: return [0.4, 0.3, 0.2, 0.1][:n]
        if signal.magnitude < -2: return [0.1, 0.2, 0.3, 0.4][:n]
        return [1/n] * n

    def _pick_instrument(self, idx, total, confidence):
        return "spot"  # Default; TOM/Cash selected by settlement logic

    def _expand(self, split, weights):
        for i, w in enumerate(weights):
            for ccy, total in split.items():
                yield ccy, total * w
```

## 3.7 Composite Rate Engine

```python
# agents/fx_analyst/composite_rate.py

from dataclasses import dataclass
from datetime import datetime, time, date, timedelta
from enum import Enum
from zoneinfo import ZoneInfo
from typing import Optional
import logging

logger = logging.getLogger("rate_engine")
IST = ZoneInfo("Asia/Kolkata")
CT  = ZoneInfo("US/Central")
GST = ZoneInfo("Asia/Dubai")
LDN = ZoneInfo("Europe/London")

class RateSource(str, Enum):
    ONSHORE_SPOT = "onshore_spot"
    NSE_FUTURES  = "nse_futures"
    DGCX_FUTURES = "dgcx_futures"
    CME_FUTURES  = "cme_futures"
    NDF_OFFSHORE = "ndf_offshore"
    SYNTHETIC    = "synthetic"

class CurrencyPair(str, Enum):
    USDINR = "USD/INR"
    GBPINR = "GBP/INR"
    EURINR = "EUR/INR"
    GBPUSD = "GBP/USD"
    EURUSD = "EUR/USD"

@dataclass
class RawTick:
    source: RateSource
    pair: CurrencyPair
    bid: float
    ask: float
    timestamp: datetime
    tenor: str = "spot"
    volume: float = 0
    exchange_code: str = ""

@dataclass
class ImpliedSpot:
    pair: CurrencyPair
    mid: float
    bid: float
    ask: float
    source: RateSource
    raw_price: float
    basis_adjustment: float
    confidence: float
    timestamp: datetime
    stale: bool = False

@dataclass
class CompositeRate:
    pair: CurrencyPair
    mid: float
    bid: float
    ask: float
    primary_source: RateSource
    sources_used: list
    confidence: float
    timestamp: datetime
    is_market_hours: bool

@dataclass
class SourceWindow:
    source: RateSource
    open_time: time
    close_time: time
    tz: ZoneInfo
    priority: int
    confidence_base: float
    staleness_sec: int

SOURCE_SCHEDULE = [
    SourceWindow(RateSource.ONSHORE_SPOT, time(9,0),  time(15,30), IST, 1, 1.0,  30),
    SourceWindow(RateSource.NSE_FUTURES,  time(9,0),  time(17,0),  IST, 2, 0.9,  15),
    SourceWindow(RateSource.DGCX_FUTURES, time(7,0),  time(23,59), GST, 3, 0.75, 60),
    SourceWindow(RateSource.CME_FUTURES,  time(17,0), time(16,0),  CT,  4, 0.7,  30),
    SourceWindow(RateSource.NDF_OFFSHORE, time(0,0),  time(23,59), LDN, 5, 0.6, 120),
]


class BasisCalculator:
    def __init__(self):
        self._basis_cache: dict[tuple, float] = {}

    async def calibrate(self, onshore_close, pair, source_closes):
        for src, price in source_closes.items():
            self._basis_cache[(src, pair)] = price - onshore_close

    def adjust(self, tick: RawTick) -> float:
        basis = self._basis_cache.get((tick.source, tick.pair), 0.0)
        if tick.source in (RateSource.NSE_FUTURES, RateSource.CME_FUTURES, RateSource.DGCX_FUTURES):
            dte = self._days_to_expiry(tick.tenor)
            basis *= (dte / 30.0) if dte > 0 else 0.0
        return (tick.bid + tick.ask) / 2 - basis

    def get_basis(self, source, pair):
        return self._basis_cache.get((source, pair), 0.0)

    def _days_to_expiry(self, tenor):
        if tenor == "near_month":
            import calendar
            today = date.today()
            last_day = calendar.monthrange(today.year, today.month)[1]
            last_date = date(today.year, today.month, last_day)
            while last_date.weekday() != 2: last_date -= timedelta(days=1)
            return max((last_date - today).days, 0)
        return 30


class CrossRateSynth:
    def __init__(self):
        self._latest: dict[CurrencyPair, RawTick] = {}

    def update(self, tick: RawTick):
        self._latest[tick.pair] = tick

    def synthesize_gbpinr(self, usdinr_mid):
        t = self._latest.get(CurrencyPair.GBPUSD)
        if t and self._fresh(t, 60):
            return ((t.bid + t.ask) / 2) * usdinr_mid
        return None

    def synthesize_eurinr(self, usdinr_mid):
        t = self._latest.get(CurrencyPair.EURUSD)
        if t and self._fresh(t, 60):
            return ((t.bid + t.ask) / 2) * usdinr_mid
        return None

    def _fresh(self, tick, max_sec):
        return (datetime.now(tz=IST) - tick.timestamp.astimezone(IST)).total_seconds() < max_sec


class CompositeRateEngine:
    def __init__(self):
        self.basis = BasisCalculator()
        self.cross = CrossRateSynth()
        self._ticks: dict[tuple, ImpliedSpot] = {}
        self._composite: dict[CurrencyPair, CompositeRate] = {}
        self._subscribers = []

    async def on_tick(self, tick: RawTick):
        if tick.pair in (CurrencyPair.GBPUSD, CurrencyPair.EURUSD):
            self.cross.update(tick)
            await self._update_cross_pairs()
            return

        implied_mid = self.basis.adjust(tick)
        half_spread = (tick.ask - tick.bid) / 2
        self._ticks[(tick.source, tick.pair)] = ImpliedSpot(
            pair=tick.pair, mid=implied_mid,
            bid=implied_mid - half_spread, ask=implied_mid + half_spread,
            source=tick.source, raw_price=(tick.bid+tick.ask)/2,
            basis_adjustment=self.basis.get_basis(tick.source, tick.pair),
            confidence=self._base_confidence(tick.source),
            timestamp=tick.timestamp,
        )
        await self._recompute_composite(tick.pair)

    def get_rate(self, pair) -> Optional[CompositeRate]:
        return self._composite.get(pair)

    def subscribe(self, callback):
        self._subscribers.append(callback)

    def check_basis_anomaly(self, pair, threshold=0.50):
        onshore = self._ticks.get((RateSource.ONSHORE_SPOT, pair))
        ndf = self._ticks.get((RateSource.NDF_OFFSHORE, pair))
        if onshore and ndf and not onshore.stale and not ndf.stale:
            if abs(ndf.mid - onshore.mid) > threshold:
                return f"ANOMALY: {pair.value} basis at {abs(ndf.mid-onshore.mid):.2f}"
        return None

    async def _recompute_composite(self, pair):
        candidates = []
        for sw in SOURCE_SCHEDULE:
            imp = self._ticks.get((sw.source, pair))
            if not imp: continue
            age = (datetime.now(IST) - imp.timestamp.astimezone(IST)).total_seconds()
            if age > sw.staleness_sec:
                imp.stale = True
                continue
            if self._is_active(sw):
                imp.confidence = sw.confidence_base * self._liq_factor(imp)
                candidates.append(imp)
        if not candidates: return
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        best = candidates[0]
        self._composite[pair] = CompositeRate(
            pair=pair, mid=best.mid, bid=best.bid, ask=best.ask,
            primary_source=best.source,
            sources_used=[c.source for c in candidates[:3]],
            confidence=best.confidence, timestamp=best.timestamp,
            is_market_hours=self._is_onshore(),
        )
        for cb in self._subscribers:
            await cb(self._composite[pair])

    async def _update_cross_pairs(self):
        usdinr = self._composite.get(CurrencyPair.USDINR)
        if not usdinr: return
        for pair, fn in [(CurrencyPair.GBPINR, self.cross.synthesize_gbpinr),
                         (CurrencyPair.EURINR, self.cross.synthesize_eurinr)]:
            mid = fn(usdinr.mid)
            if mid:
                self._ticks[(RateSource.SYNTHETIC, pair)] = ImpliedSpot(
                    pair=pair, mid=mid, bid=mid*0.9999, ask=mid*1.0001,
                    source=RateSource.SYNTHETIC, raw_price=mid,
                    basis_adjustment=0, confidence=0.5,
                    timestamp=datetime.now(IST),
                )
                await self._recompute_composite(pair)

    def _is_active(self, sw):
        now = datetime.now(sw.tz).time()
        if sw.open_time < sw.close_time: return sw.open_time <= now <= sw.close_time
        return now >= sw.open_time or now <= sw.close_time

    def _is_onshore(self):
        return time(9,0) <= datetime.now(IST).time() <= time(15,30)

    def _liq_factor(self, imp):
        spread = (imp.ask - imp.bid) * 100
        if spread < 1: return 1.0
        if spread < 3: return 0.9
        if spread < 10: return 0.7
        return 0.4

    def _base_confidence(self, source):
        for sw in SOURCE_SCHEDULE:
            if sw.source == source: return sw.confidence_base
        return 0.5
```

## 3.8 Operations Agent

```python
# agents/operations/agent.py

import asyncio, uuid
from datetime import datetime, date, time

class OperationsAgent(BaseAgent):
    def __init__(self, publisher, consumer, fund_mover, maker_checker, window_manager):
        super().__init__("operations", publisher, consumer)
        self.fund_mover = fund_mover
        self.mc = maker_checker
        self.windows = window_manager

    async def setup(self):
        self.consumer.on("forecast.rda.shortfall", self.handle_shortfall)
        self.consumer.on("fx.deal.instruction", self.handle_deal_instruction)

    async def run_daily(self):
        lookahead = await self.windows.holiday_lookahead(days=3)
        await self.emit(Event(
            event_type="ops.holiday.lookahead",
            source_agent=self.name,
            timestamp=datetime.utcnow(),
            payload={"holidays": lookahead},
            correlation_id=str(uuid.uuid4()),
        ))
        asyncio.create_task(self._monitor_windows())

    async def handle_shortfall(self, payload):
        ccy = payload["currency"]
        amount = payload["shortfall"] * 1.10  # 10% buffer
        window = self.windows.get_window(ccy)
        if not window.is_open_now() and not window.opens_before(time(9, 0)):
            self.logger.critical(f"CRITICAL: {ccy} window won't open before INR market!")
            return

        proposal = FundMovementProposal(
            id=str(uuid.uuid4()), currency=ccy, amount=amount,
            source_account=self.fund_mover.get_operating_account(ccy),
            destination_nostro=self.fund_mover.get_nostro_account(ccy),
            rail=self.windows.get_rail(ccy),
            proposed_by="system:liquidity_agent",
            purpose=f"RDA shortfall cover {date.today()}",
            idempotency_key=f"{ccy}-{date.today()}-shortfall",
        )
        await self.mc.submit_proposal(proposal)

    async def handle_deal_instruction(self, payload):
        pass  # TODO: verify nostro can cover settlement

    async def _monitor_windows(self):
        while True:
            for ccy in ["USD", "GBP", "EUR"]:
                w = self.windows.get_window(ccy)
                mins = w.minutes_until_close()
                if mins is not None and mins <= 30:
                    await self.emit(Event(
                        event_type="ops.window.closing",
                        source_agent=self.name,
                        timestamp=datetime.utcnow(),
                        payload={"currency": ccy, "minutes_remaining": mins},
                        correlation_id=str(uuid.uuid4()),
                    ))
            await asyncio.sleep(60)
```

```python
# agents/operations/maker_checker.py

import asyncio
from datetime import datetime

class MakerCheckerWorkflow:
    def __init__(self, db, auth_service, alert_router):
        self.db = db
        self.auth = auth_service
        self.alerts = alert_router

    async def submit_proposal(self, proposal):
        errors = await self._validate(proposal)
        if errors:
            proposal.status = "rejected"
            await self.db.save(proposal)
            return {"status": "rejected", "errors": errors}

        proposal.status = "pending_approval"
        await self.db.save(proposal)
        required = 2 if proposal.amount > 10_000_000 else 1
        await self.alerts.notify_checkers(proposal, required)
        asyncio.create_task(self._auto_escalate(proposal.id, timeout=30))
        return {"status": "pending_approval", "proposal_id": proposal.id}

    async def approve(self, proposal_id, checker_id):
        proposal = await self.db.get(proposal_id)
        if not await self.auth.can_approve(checker_id, proposal):
            raise PermissionError("Insufficient authority")
        proposal.status = "approved"
        proposal.approved_by = checker_id
        await self.db.save(proposal)
        return await self._execute(proposal)

    async def _validate(self, p):
        errors = []
        if not await self.db.is_approved_nostro(p.destination_nostro):
            errors.append(f"Unknown nostro: {p.destination_nostro}")
        if await self.db.has_recent_duplicate(p.idempotency_key):
            errors.append("Duplicate transfer (idempotency check)")
        return errors

    async def _execute(self, proposal):
        proposal.status = "executed"
        proposal.executed_at = datetime.utcnow()
        await self.db.save(proposal)
        return {"status": "executed"}

    async def _auto_escalate(self, proposal_id, timeout):
        await asyncio.sleep(timeout * 60)
        p = await self.db.get(proposal_id)
        if p.status == "pending_approval":
            await self.alerts.escalate(p, reason="Approval timeout")
```

```python
# agents/operations/window_manager.py

from datetime import time, datetime, timedelta
from zoneinfo import ZoneInfo

class TransferWindow:
    def __init__(self, currency, open_time, close_time, tz, rail, holiday_cal):
        self.currency = currency
        self.open_time = open_time
        self.close_time = close_time
        self.tz = ZoneInfo(tz)
        self.rail = rail

    def is_open_now(self):
        now = datetime.now(self.tz)
        return self.open_time <= now.time() <= self.close_time and now.weekday() < 5

    def minutes_until_close(self):
        if not self.is_open_now(): return None
        now = datetime.now(self.tz)
        close_dt = datetime.combine(now.date(), self.close_time, tzinfo=self.tz)
        return int((close_dt - now).total_seconds() / 60)

    def opens_before(self, deadline_ist):
        next_op = self._next_open()
        return next_op.astimezone(ZoneInfo("Asia/Kolkata")).time() < deadline_ist

    def _next_open(self):
        now = datetime.now(self.tz)
        candidate = datetime.combine(now.date(), self.open_time, tzinfo=self.tz)
        if candidate <= now: candidate += timedelta(days=1)
        while candidate.weekday() >= 5: candidate += timedelta(days=1)
        return candidate

class WindowManager:
    WINDOWS = {
        "USD": TransferWindow("USD", time(9,0),  time(18,0), "US/Eastern",     "fedwire", "us_fed"),
        "GBP": TransferWindow("GBP", time(8,0),  time(16,0), "Europe/London",  "chaps",   "uk_boe"),
        "EUR": TransferWindow("EUR", time(7,0),  time(18,0), "Europe/Berlin",  "target2", "eu_target2"),
        "INR": TransferWindow("INR", time(9,0),  time(15,0), "Asia/Kolkata",   "bank_desk","in_rbi"),
    }

    def get_window(self, currency):
        return self.WINDOWS.get("USD" if currency == "AED" else currency)

    def get_rail(self, currency):
        return self.get_window(currency).rail

    async def holiday_lookahead(self, days=3):
        return {}  # TODO: query calendar service
```

## 3.9 Shared Services

```python
# services/calendar_service.py

from datetime import date, timedelta

class CalendarService:
    def __init__(self, db):
        self.db = db

    async def is_holiday(self, d, jurisdiction):
        return await self.db.check_holiday(d, jurisdiction)

    async def is_day_before_holiday(self, d, jurisdiction):
        nxt = await self._next_biz_day(d, jurisdiction)
        return (nxt - d).days > 1

    async def is_day_after_holiday(self, d, jurisdiction):
        prev = await self._prev_biz_day(d, jurisdiction)
        return (d - prev).days > 1

    async def _next_biz_day(self, d, jur):
        c = d + timedelta(days=1)
        while await self.is_holiday(c, jur) or c.weekday() >= 5:
            c += timedelta(days=1)
        return c

    async def _prev_biz_day(self, d, jur):
        c = d - timedelta(days=1)
        while await self.is_holiday(c, jur) or c.weekday() >= 5:
            c -= timedelta(days=1)
        return c
```

```python
# services/audit_log.py

import hashlib, json
from datetime import datetime

class AuditLog:
    def __init__(self, db):
        self.db = db

    async def log(self, event_type, agent, action, details, user="system"):
        await self.db.insert("audit_log", {
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "agent": agent,
            "action": action,
            "details": details,
            "user": user,
            "checksum": hashlib.sha256(
                json.dumps(details, sort_keys=True).encode()
            ).hexdigest(),
        })
```

## 3.10 Entry Point

```python
# scripts/run_agents.py

import asyncio

async def main():
    publisher = EventPublisher(settings.redis_url)
    liq_consumer = EventConsumer(settings.redis_url, "liquidity_group", "liq_1")
    fx_consumer  = EventConsumer(settings.redis_url, "fx_group", "fx_1")
    ops_consumer = EventConsumer(settings.redis_url, "ops_group", "ops_1")

    # Initialize composite rate engine
    rate_engine = CompositeRateEngine()

    # Wire up agents with dependencies (TODO: full DI)
    # liquidity_agent = LiquidityAgent(publisher, liq_consumer, ...)
    # fx_agent = FXAnalystAgent(publisher, fx_consumer, rate_engine, ...)
    # ops_agent = OperationsAgent(publisher, ops_consumer, ...)

    # Run daily routines via scheduler (APScheduler @ 6 AM IST)
    # Then start event consumers
    await asyncio.gather(
        # liquidity_agent.start(),
        # fx_agent.start(),
        # ops_agent.start(),
    )

if __name__ == "__main__":
    asyncio.run(main())
```

---

# PART 4 — NEXT STEPS

| Priority | Task | Owner |
|---|---|---|
| P0 | Wire up database layer (SQLAlchemy + TimescaleDB migrations) | Backend |
| P0 | Implement NSE/CME/DGCX feed connectors (start with one, add others) | Data Eng |
| P0 | Set up Redis Streams / Kafka for message bus | Infra |
| P1 | Build maker-checker UI (FastAPI + React) | Fullstack |
| P1 | Implement FX forecast model (start with simple ensemble → LSTM later) | ML |
| P1 | Seed holiday calendars (RBI, Fed, BoE, UAE CB) | Ops |
| P2 | Treasury dashboard with real-time exposure via WebSocket | Frontend |
| P2 | Add APScheduler to trigger `run_daily()` at 6 AM IST | Backend |
| P2 | Integration tests: holiday conflicts, flash crash, window-closing races | QA |
| P3 | Basis calibration pipeline (automated daily at 15:30 IST) | Data Eng |
| P3 | News/sentiment NLP pipeline for macro signal | ML |