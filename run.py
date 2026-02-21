"""
Aspora TMS — Combined Runner (local dev / hackathon mode).

Starts FastAPI + all agents in ONE process sharing a single InMemoryBus.
This eliminates the bus-isolation problem that exists when running
`python main.py` (agents) and `uvicorn api.app:app` (API) as separate
processes — they each create their own InMemoryBus and never share events.

Usage:
    python run.py
    PORT=3001 python run.py      # override port (default 3001)

What runs:
  - FastAPI dashboard API        → http://localhost:3001
  - FXAnalystAgent               → polls FX Band Predictor (port 8001)
  - LiquidityAgent               → forecasts + RDA checks
  - OperationsAgent              → fund movement + window monitoring
  - APScheduler                  → daily cron at 00:30 / 03:30 UTC

Bus: InMemoryBus (in-process; all agents and the API share the same instance)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from decimal import Decimal

import structlog
import uvicorn

# ── Logging setup ──────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)

log = structlog.get_logger("tms.run")


async def main() -> None:
    log.info("aspora-tms combined runner starting")

    # ── Shared bus (single instance for agents + API) ──────────────────────────
    from bus.memory_bus import InMemoryBus
    bus = InMemoryBus()

    # ── Services ───────────────────────────────────────────────────────────────
    from agents.operations.fund_mover import (
        BalanceTracker,
        FundMover,
        InMemoryExecutionStore,
        MockBankAPI,
    )
    from agents.operations.maker_checker import MakerCheckerWorkflow
    from agents.operations.window_manager import WindowManager
    from services.calendar_service import CalendarService
    from services.sheets_balance_sync import fetch_nostro_balances
    from config.settings import settings

    calendar = CalendarService()

    # ── Seed nostro balances from Google Sheets (falls back to hardcoded) ──────
    _row_numbers = [int(r) for r in settings.sheets_balance_rows.split(",")]
    initial_balances = fetch_nostro_balances(
        sheet_id=settings.sheets_balance_id,
        gid=settings.sheets_balance_gid,
        row_numbers=_row_numbers,
    )
    log.info("nostro balances loaded from sheets", currencies=list(initial_balances.keys()))

    tracker = BalanceTracker(initial_balances)
    fund_mover = FundMover(MockBankAPI(), InMemoryExecutionStore(), tracker)

    class _StubDB:
        async def get(self, _id): return None
        async def save(self, _p): pass
        async def list_all(self): return []
        async def is_approved_nostro(self, _n): return True
        async def has_recent_duplicate(self, _k): return False

    class _StubAuth:
        async def can_approve(self, _c, _p): return True

    class _StubAlerts:
        async def notify_checkers(self, _p, _n): pass
        async def escalate(self, _p, **_kw): pass
        async def notify_executed(self, _p): pass

    class _StubAudit:
        async def log(self, **_kw): pass

    maker_checker = MakerCheckerWorkflow(
        _StubDB(), _StubAuth(), _StubAlerts(), _StubAudit()
    )

    no_holiday = lambda _d: False  # noqa: E731
    window_manager = WindowManager(no_holiday, no_holiday, no_holiday, no_holiday)

    # ── Build FastAPI with shared bus ──────────────────────────────────────────
    from api.app import create_app
    app = create_app(fund_mover, maker_checker, window_manager, calendar, bus)

    # ── Build agents (same shared bus) ─────────────────────────────────────────
    from agents.fx_analyst.agent import FXAnalystAgent
    from agents.liquidity.agent import LiquidityAgent
    from agents.operations.agent import OperationsAgent

    fx_analyst     = FXAnalystAgent(bus=bus)
    liquidity_agent = LiquidityAgent(bus=bus, calendar=calendar)
    ops_agent      = OperationsAgent(
        bus=bus,
        calendar=calendar,
        window_manager=window_manager,
        maker_checker=maker_checker,
        fund_mover=fund_mover,
    )

    await fx_analyst.start()
    await liquidity_agent.start()
    await ops_agent.start()
    log.info("agents started", agents=["fx_analyst", "liquidity", "operations"])

    # Attach agent references to app.state so /dev/trigger can reach them
    app.state.fx_agent  = fx_analyst
    app.state.liq_agent = liquidity_agent
    app.state.ops_agent = ops_agent

    # ── Scheduler ─────────────────────────────────────────────────────────────
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        liquidity_agent.run_daily, trigger="cron", hour=0, minute=30,
        id="liquidity_daily", name="LiquidityAgent daily run",
    )
    scheduler.add_job(
        ops_agent.run_daily, trigger="cron", hour=0, minute=30,
        id="ops_daily", name="OperationsAgent daily run",
    )
    scheduler.add_job(
        fx_analyst.run_daily, trigger="cron", hour=3, minute=30,
        id="fx_daily", name="FXAnalystAgent daily market brief",
    )

    async def _refresh_balances() -> None:
        balances = fetch_nostro_balances(
            sheet_id=settings.sheets_balance_id,
            gid=settings.sheets_balance_gid,
            row_numbers=[int(r) for r in settings.sheets_balance_rows.split(",")],
        )
        await fund_mover._tracker.refresh(balances)
        log.info("nostro balances refreshed from sheets", currencies=list(balances.keys()))

    scheduler.add_job(
        _refresh_balances, trigger="cron", hour=0, minute=30,
        id="balance_refresh", name="Nostro balance sync from Google Sheets",
    )
    scheduler.start()
    log.info("scheduler started", cron="00:30 UTC (06:00 IST)")

    # ── Start uvicorn in the same event loop ───────────────────────────────────
    port = int(os.environ.get("PORT", 3001))
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        # Don't create a new event loop — reuse the running one
        loop="none",
    )
    server = uvicorn.Server(config)
    log.info("combined server starting", port=port)

    try:
        await server.serve()
    finally:
        scheduler.shutdown(wait=False)
        log.info("aspora-tms combined runner stopped")


if __name__ == "__main__":
    asyncio.run(main())
