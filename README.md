# Aspora TMS — Treasury Management System

An event-driven, multi-agent treasury management platform that automates FX deal execution, nostro prefunding, fund movement approvals, and banking window monitoring across USD, GBP, EUR, AED, and INR.

---

## Architecture Overview

```
┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Liquidity Agent │  │  FX Analyst Agent │  │  Operations Agent   │  │  FX Band Predictor       │
│  (forecast/RDA) │  │  (deals/exposure) │  │  (fund movements)   │  │  (FX rate prediction)│
└────────┬────────┘  └────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
         │                    │                         │                        │
         └────────────────────┴─────────────────────────┘                        │
                                 │                                               │
                    ┌────────────▼────────────┐                     ┌────────────▼────────────┐
                    │   Redis Streams Bus      │                     │   FX Intelligence API    │
                    │   (tms:<event_type>)     │                     │   :8001 /predict         │
                    └────────────┬────────────┘                     └─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FastAPI Dashboard API  │
                    │   :8000 /api/v1/...      │
                    └─────────────────────────┘
```

### Agents & Responsibilities

| Agent | Branch | Key responsibilities |
|---|---|---|
| **Operations Agent** | `feature/operations/...` | Fund movement proposals, maker-checker, window alerts, holiday lookahead |
| **Liquidity Agent** | `feature/liquidity-agent/...` | Daily cash forecasting, RDA shortfall detection |
| **FX Analyst Agent** | `feature/fx-analyst/...` | FX deal instructions, exposure management, reforecast triggers |
| **FX Band Predictor** | `integration/fx` | 48h USD/INR rate band prediction, 5-model ensemble, 7-day forecast |

### Event Bus (Redis Streams)

All inter-agent communication uses typed events defined in `bus/events.py`:

| Event type | Publisher | Consumers |
|---|---|---|
| `forecast.daily.ready` | Liquidity Agent | Operations Agent |
| `forecast.rda.shortfall` | Liquidity Agent | Operations Agent |
| `fx.deal.instruction` | FX Analyst Agent | Operations Agent |
| `fx.reforecast.trigger` | FX Analyst Agent | Operations Agent |
| `maker_checker.proposal.approved` | MakerChecker | Operations Agent |
| `ops.fund.movement.status` | Operations Agent | All |
| `ops.nostro.balance.update` | Operations Agent | All |
| `ops.window.closing` | Operations Agent | All |
| `ops.holiday.lookahead` | Operations Agent | All |

---

## Key Features

- **Automated nostro prefunding** — detects RDA shortfalls and submits fund movement proposals with a configurable safety buffer (default 10%)
- **Maker-checker approvals** — dual-approval required for transfers above $10M; auto-escalates stale proposals after 30 minutes
- **Banking window monitoring** — fires `ops.window.closing` alerts 30 minutes before Fedwire / CHAPS / TARGET2 cut-offs
- **Idempotent event handling** — deduplication per currency per calendar day prevents duplicate proposals
- **Holiday-aware scheduling** — 3-day lookahead across RBI, Fed, BoE, ECB calendars
- **Multi-rail support** — Fedwire, CHAPS, TARGET2, SWIFT gpi, INR treasury desk

---

## FX Band Predictor — FX Rate Prediction Engine

> *Hackathon-winning integration by the FX Intelligence team.*

**FX Band Predictor** ("merchant's intelligence") is a 5-model ensemble that predicts the 48-hour USD/INR rate band so treasury knows exactly how much INR to keep ready. Built for $9.5M daily volume (₹86.3 crore at stake), trained on 22 years of market data.

### How it works

Five models vote on direction and range, then a safety gate decides whether to act:

| Model | Role | What it does |
|---|---|---|
| **XGBoost Regime** | Context | Classifies market as trending_up / trending_down / high_vol / range_bound (99.3% accuracy) |
| **LSTM Range** | Range provider | Predicts 48h high/low/close band using 30-day sequences (76% containment) |
| **Sentiment** | Primary vote | News-driven direction signal via Alpha Vantage / NewsAPI |
| **Macro Signal** | Secondary vote | Rule-based FRED fundamentals (Fed funds, yield curve, CPI) |
| **Intraday LSTM** | Momentum check | 4h bar momentum confirmation (asymmetric UP-only, 74% UP accuracy) |

Direction is decided by sentiment + macro voting. If both agree, the system acts. A 5-rule safety gate blocks ~65% of signals to keep the system conservative. When confidence exceeds 50%, Claude Opus 4.6 generates a CFO-ready reasoning brief.

### Running it

Everything is handled by a single command:

```bash
cd fx_band_predictor/fx-agent
pip install -r requirements.txt && pip install fredapi torch
python run.py
```

`run.py` fetches market data, trains any missing models, and starts the prediction API on **port 8001**. See `fx_band_predictor/README.md` for the full reference.

### API endpoints (port 8001)

| Endpoint | Description |
|---|---|
| `GET /predict` | 48h prediction with full model breakdown and AI reasoning |
| `GET /predict/weekly` | 7-day calendar-aware forecast with best conversion day |
| `GET /intraday` | 4h momentum signal with session breakdown |
| `GET /health` | System status and uptime |
| `GET /history?days=30` | Historical prediction accuracy |
| `GET /backtest?days=90` | Walk-forward backtest results |

---

## Project Structure

```
aspora-tms/
├── agents/
│   ├── base.py                    # BaseAgent (subscribe/emit helpers)
│   └── operations/
│       ├── agent.py               # OperationsAgent v2.0
│       ├── fund_mover.py          # FundMover — executes proposals via bank APIs
│       ├── maker_checker.py       # MakerCheckerWorkflow — approval state machine
│       └── window_manager.py      # WindowManager — cut-off times per rail
├── api/
│   ├── app.py                     # FastAPI app factory (:8000)
│   ├── auth.py                    # JWT / RBAC middleware
│   ├── schemas.py                 # Pydantic request/response models
│   └── routers/
│       ├── balances.py            # GET /api/v1/balances
│       ├── proposals.py           # GET/POST /api/v1/proposals
│       ├── transfers.py           # GET /api/v1/transfers
│       ├── exposure.py            # GET /api/v1/exposure
│       ├── windows.py             # GET /api/v1/windows
│       ├── holidays.py            # GET /api/v1/holidays
│       └── events.py              # GET /api/v1/events (SSE stream)
├── fx_band_predictor/                  # FX Rate Prediction Engine (hackathon winner)
│   └── fx-agent/
│       ├── run.py                 # Single entry point — train + serve (:8001)
│       ├── agents/                # 5-model ensemble + sentiment + macro
│       ├── api/main.py            # FastAPI prediction API (:8001)
│       ├── data/                  # Market data fetchers + feature engineering
│       ├── models/                # Training scripts + saved artifacts
│       └── requirements.txt
├── bus/
│   ├── events.py                  # Typed event constants + Event dataclass
│   ├── base.py                    # EventBus ABC
│   ├── redis_bus.py               # Redis Streams implementation
│   ├── memory_bus.py              # In-memory bus (dev / tests)
│   ├── publisher.py               # Publisher helper
│   └── consumer.py                # Consumer helper
├── config/
│   └── settings.py                # Pydantic Settings (ASPORA_ env prefix)
├── models/
│   └── domain.py                  # Core dataclasses (FundMovementProposal, etc.)
├── services/
│   ├── calendar_service.py        # Holiday calendar queries
│   ├── audit_log.py               # Structured audit trail
│   └── integrations/              # Bank API adapters
├── tests/                         # pytest test suite
└── pyproject.toml
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Redis 7+ (for production message bus)
- PostgreSQL 15+ with TimescaleDB extension (for production persistence)

### Local Setup

```bash
# Clone the repo
git clone git@github.com:onegarv/aspora-tms.git
cd aspora-tms

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Copy environment template and configure
cp .env.example .env              # edit values as needed

# Run the full test suite
pytest tests/ -v
```

### Running the API Locally

```bash
uvicorn api.app:app --reload --port 8000
```

The API starts with stub services by default — no Redis or Postgres required for local development.

Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Configuration

All settings are read from environment variables prefixed `ASPORA_` (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `ASPORA_POSTGRES_URL` | `postgresql+asyncpg://localhost:5432/aspora_tms` | Primary database |
| `ASPORA_REDIS_URL` | `redis://localhost:6379/0` | Message bus |
| `ASPORA_BUS_TYPE` | `redis` | `redis` or `kafka` |
| `ASPORA_MAX_SINGLE_DEAL_USD` | `5000000` | Max per FX deal (USD equiv) |
| `ASPORA_MAX_OPEN_EXPOSURE_PCT` | `0.30` | Max unhedged exposure after 2PM |
| `ASPORA_PREFUNDING_BUFFER_PCT` | `0.10` | Safety buffer on shortfall transfers |
| `ASPORA_MAKER_CHECKER_TIMEOUT_MIN` | `30` | Auto-escalate stale proposals after N minutes |
| `ASPORA_DUAL_CHECKER_THRESHOLD_USD` | `10000000` | Transfers above this require 2 approvers |
| `ASPORA_JWT_SECRET` | *(must override)* | JWT signing secret — **change in production** |

See `config/settings.py` for the full list.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_operations_agent.py -v

# Type-check
mypy .

# Lint
ruff check .
```

Tests use `fakeredis` for the message bus and `freezegun` for time-sensitive scenarios — no external services required.

---

## Contributing

### Branch Naming

```
feature/<agent>/<short-description>
  e.g.  feature/liquidity-agent/forecast-model
        feature/fx-analyst/exposure-calculator
        fix/fund-mover/sla-breach-handling
        feature/infra/db-layer
```

### Workflow

1. Branch off `master` using the convention above
2. Open a PR — at least 1 review required before merge
3. CI must pass (lint + type-check + tests)
4. Do **not** push directly to `master`

### Agent Ownership

| Agent / Area | Owner | Do not modify without coordinating |
|---|---|---|
| `agents/operations/`, `api/` | D1 (core team) | Yes — coordinate first |
| `agents/liquidity/` | Liquidity Agent team | No |
| `agents/fx_analyst/` | FX Analyst team | No |
| `bus/events.py` | All teams | Coordinate — shared contract |
| `config/settings.py`, `db/` | Infra team | No |

---

## Payment Rails & SLAs

| Rail | Currencies | Confirmation SLA |
|---|---|---|
| Fedwire | USD | 30 min |
| CHAPS | GBP | 120 min |
| TARGET2 / SEPA | EUR | 240 min |
| SWIFT gpi | USD, GBP, AED | 480 min |
| Bank Desk | INR | Intraday |

Window-closing alerts fire 30 minutes before each rail's operational cut-off.

---

## License

Proprietary — Aspora Financial Technologies. All rights reserved.
