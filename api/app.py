"""
FastAPI application factory for the Aspora TMS Dashboard API.

Usage:
    uvicorn api.app:app --reload --port 3001   # agents start automatically
    python run.py                               # combined runner (also works)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    balances, deals, dev, events, exposure, forecast,
    holidays, pnl, proposals, risk, transfers, windows,
)


def create_app(
    fund_mover,
    maker_checker,
    window_manager,
    calendar,
    bus,
    cb_client=None,
    clearbank_nostro_map: dict | None = None,
    lifespan=None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    All service instances are stored on app.state so routers can retrieve
    them via request.app.state.<name>.
    """
    app = FastAPI(
        title="Aspora TMS Dashboard API",
        version="1.0",
        lifespan=lifespan,
    )

    # Inject service instances
    app.state.fm  = fund_mover
    app.state.mc  = maker_checker
    app.state.wm  = window_manager
    app.state.cal = calendar
    app.state.bus = bus

    app.state.cb_client            = cb_client
    app.state.clearbank_nostro_map = clearbank_nostro_map or {}

    # CORS — lock down in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register all routers under /api/v1
    PREFIX = "/api/v1"
    app.include_router(balances.router,  prefix=PREFIX)
    app.include_router(proposals.router, prefix=PREFIX)
    app.include_router(transfers.router, prefix=PREFIX)
    app.include_router(exposure.router,  prefix=PREFIX)
    app.include_router(windows.router,   prefix=PREFIX)
    app.include_router(holidays.router,  prefix=PREFIX)
    app.include_router(events.router,    prefix=PREFIX)
    app.include_router(deals.router,     prefix=PREFIX)
    app.include_router(pnl.router,       prefix=PREFIX)
    app.include_router(forecast.router,  prefix=PREFIX)
    app.include_router(risk.router,      prefix=PREFIX)
    app.include_router(dev.router,       prefix=PREFIX)

    return app


# ── Module-level app for `uvicorn api.app:app` ────────────────────────────────

def _make_default_app() -> FastAPI:
    """
    Build the app with all agents started via lifespan.
    Works with plain `uvicorn api.app:app --port 3001`.
    """
    import asyncio
    import logging

    from agents.operations.fund_mover import (
        BalanceTracker, FundMover, InMemoryExecutionStore, MockBankAPI,
    )
    from agents.operations.maker_checker import MakerCheckerWorkflow
    from agents.operations.window_manager import WindowManager
    from bus.memory_bus import InMemoryBus
    from config.settings import settings
    from services.calendar_service import CalendarService
    from services.sheets_balance_sync import fetch_nostro_balances

    log = logging.getLogger("tms.app")

    # ── Build services synchronously (safe at import time) ────────────────────
    _row_numbers = [int(r) for r in settings.sheets_balance_rows.split(",")]
    initial_balances = fetch_nostro_balances(
        sheet_id=settings.sheets_balance_id,
        gid=settings.sheets_balance_gid,
        row_numbers=_row_numbers,
    )
    tracker = BalanceTracker(initial_balances)
    fm = FundMover(MockBankAPI(), InMemoryExecutionStore(), tracker)

    class _StubDB:
        _proposals: dict = {}
        async def get(self, _id): return self._proposals.get(_id)
        async def save(self, p): self._proposals[p.id] = p
        async def list_all(self): return list(self._proposals.values())
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

    mc  = MakerCheckerWorkflow(_StubDB(), _StubAuth(), _StubAlerts(), _StubAudit())
    wm  = WindowManager(lambda d: False, lambda d: False, lambda d: False, lambda d: False)
    cal = CalendarService()
    bus = InMemoryBus()

    # ── Lifespan: start agents when uvicorn starts ─────────────────────────────
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        from agents.fx_analyst.agent import FXAnalystAgent
        from agents.liquidity.agent import LiquidityAgent
        from agents.operations.agent import OperationsAgent

        fx_agent  = FXAnalystAgent(bus=bus)
        liq_agent = LiquidityAgent(bus=bus, calendar=cal)
        ops_agent = OperationsAgent(
            bus=bus, calendar=cal, window_manager=wm,
            maker_checker=mc, fund_mover=fm,
        )

        await fx_agent.start()
        await liq_agent.start()
        await ops_agent.start()

        # Attach so /dev/trigger can reach them
        app.state.fx_agent  = fx_agent
        app.state.liq_agent = liq_agent
        app.state.ops_agent = ops_agent

        log.info("all agents started via lifespan")
        yield

        # Graceful shutdown
        await ops_agent.shutdown()
        log.info("agents shut down")

    return create_app(fm, mc, wm, cal, bus, lifespan=lifespan)


app = _make_default_app()
