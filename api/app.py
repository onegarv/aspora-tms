"""
FastAPI application factory for the Aspora TMS Dashboard API.

Usage:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

from decimal import Decimal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import balances, deals, events, exposure, forecast, holidays, pnl, proposals, risk, transfers, windows


def create_app(fund_mover, maker_checker, window_manager, calendar, bus) -> FastAPI:
    """
    Create and configure the FastAPI application.

    All service instances are stored on app.state so routers can retrieve
    them via request.app.state.<name>.
    """
    app = FastAPI(title="Aspora TMS Dashboard API", version="1.0")

    # Inject service instances
    app.state.fm  = fund_mover
    app.state.mc  = maker_checker
    app.state.wm  = window_manager
    app.state.cal = calendar
    app.state.bus = bus

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

    return app


# ── Module-level app for `uvicorn api.app:app` ────────────────────────────────

def _make_default_app() -> FastAPI:
    """
    Build a lightweight default app using stub/empty service instances.
    Suitable for `uvicorn api.app:app` during local development.
    """
    from agents.operations.fund_mover import (
        BalanceTracker,
        FundMover,
        InMemoryExecutionStore,
        MockBankAPI,
    )
    from agents.operations.maker_checker import MakerCheckerWorkflow
    from agents.operations.window_manager import WindowManager
    from bus.memory_bus import InMemoryBus
    from services.calendar_service import CalendarService

    # Stub services
    tracker = BalanceTracker({"USD": Decimal("0"), "GBP": Decimal("0")})
    fm = FundMover(MockBankAPI(), InMemoryExecutionStore(), tracker)

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

    mc = MakerCheckerWorkflow(_StubDB(), _StubAuth(), _StubAlerts(), _StubAudit())

    no_holiday = lambda d: False
    wm = WindowManager(no_holiday, no_holiday, no_holiday, no_holiday)
    cal = CalendarService()
    bus = InMemoryBus()

    return create_app(fm, mc, wm, cal, bus)


app = _make_default_app()
