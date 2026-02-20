# Aspora TMS — Build Progress

Last updated: 2026-02-21
Kanban board: `docs/KANBAN.md`

## Overall: 6/25 complete

## Operations Agent (You) — 6/6 ✅
- [x] R1: Window Manager — Banking hours, timezone-aware transfer windows, DST
- [x] R2: Calendar Service — Multi-jurisdiction holiday calendar (US/UK/IN/UAE)
- [x] R3: Event Bus — Redis Streams typed events, consumer groups
- [x] R4: Maker-Checker — Approval workflow, dual-checker, auto-escalation
- [x] R5: Fund Mover — Transfer execution, bank API, retry logic
- [x] R6: Ops Agent Wiring — Orchestrator, decision tree, edge cases

## Liquidity Agent (Colleague A) — 0/3
- [~] R7: Volume Forecaster — Weighted moving avg, same-weekday prediction
- [ ] R8: Multiplier Engine — Payday, holiday, FX elasticity, day-of-week
- [ ] R9: RDA Checker & Agent Wiring — Balance sufficiency, currency split

## Dashboard & UI (You) — 0/5
- [ ] D1: Dashboard API (FastAPI) — REST endpoints for balances, deals, exposure
- [ ] D2: WebSocket Live Feed — Real-time event stream, live updates
- [ ] D3: Maker-Checker UI — Approval interface, pending proposals, audit trail
- [ ] D4: Treasury Dashboard — Exposure view, deal blotter, P&L tracker
- [ ] D5: Admin Console — Multiplier config, risk limits, holiday overrides

## FX Analyst Agent (Colleague B) — 0/5
- [~] R10: Composite Rate Engine — 24/7 implied spot, basis adjustment
- [ ] R11: Market Intel & NDF Monitor — News/sentiment, macro data, RBI
- [ ] R12: FX Forecaster — 1h/1d/1w rate forecasts (LSTM, ensemble, macro)
- [ ] R13: Tranche Engine — Sizing, signal-weighted timing, instrument selection
- [ ] R14: FX Agent Wiring — Pre-market, active trading, post-market phases

## Infrastructure (Colleague C) — 0/5
- [ ] I1: Bank API Integration — Real bank adapters for deals + transfers
- [ ] I2: Postgres + Migrations — Deals, accounts, approvals, config
- [ ] I3: TimescaleDB Setup — Tick data, volume time series, NDF history
- [ ] I4: Redis Deployment — Production event bus, rate cache
- [ ] I5: Monitoring & Alerting — Prometheus, Grafana, PagerDuty

---

## Status Key
- [x] Done
- [~] In Progress
- [ ] Not Started
- BLOCKED — noted inline

## Changelog
- 2026-02-20: Ops Agent (R1-R6) complete
- {date}: {what changed}

---

Rules for maintaining this file:
1. Anyone who finishes a round updates their checkbox and the count
2. Add a changelog entry every time something moves
3. If something is blocked, write BLOCKED and the reason next to it
4. Keep the "Last updated" date current
