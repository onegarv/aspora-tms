"""
FastAPI application factory for the Aspora TMS Dashboard API.

Usage:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

from decimal import Decimal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import balances, clearbank, events, exposure, holidays, proposals, transfers, windows
from services.guardrail_service import GuardrailService


def create_app(
    fund_mover,
    maker_checker,
    window_manager,
    calendar,
    bus,
    cb_client=None,
    clearbank_nostro_map: dict | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    All service instances are stored on app.state so routers can retrieve
    them via request.app.state.<name>.

    cb_client: optional ClearBankClient — injected when ASPORA_CLEARBANK_ENABLED=true.
    clearbank_nostro_map: maps TMS nostro IDs to ClearBank IBANs.
    """
    app = FastAPI(title="Aspora TMS Dashboard API", version="1.0")

    # Inject service instances
    app.state.fm  = fund_mover
    app.state.mc  = maker_checker
    app.state.wm  = window_manager
    app.state.cal = calendar
    app.state.bus = bus

    # ClearBank
    app.state.cb_client            = cb_client
    app.state.clearbank_nostro_map = clearbank_nostro_map or {}
    app.state.guardrail            = GuardrailService()

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
    app.include_router(clearbank.router, prefix=PREFIX)

    return app


# ── Module-level app for `uvicorn api.app:app` ────────────────────────────────

def _make_default_app() -> FastAPI:
    """
    Build a lightweight default app using stub/empty service instances.
    Suitable for `uvicorn api.app:app` during local development.

    When ASPORA_CLEARBANK_ENABLED=true and credentials are set, injects
    ClearBankClient so the dispatch endpoint works in live mode.
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
    from config.settings import settings
    from services.calendar_service import CalendarService

    # Stub services
    tracker = BalanceTracker({"USD": Decimal("0"), "GBP": Decimal("0")})
    fm = FundMover(MockBankAPI(), InMemoryExecutionStore(), tracker)

    class _StubDB:
        _proposals: dict = {}

        async def get(self, _id):
            return self._proposals.get(_id)

        async def save(self, p):
            self._proposals[p.id] = p

        async def list_all(self):
            return list(self._proposals.values())

        async def is_approved_nostro(self, _n):
            return True

        async def has_recent_duplicate(self, _k):
            return False

    class _StubAuth:
        async def can_approve(self, _c, _p): return True

    class _StubAlerts:
        async def notify_checkers(self, _p, _n): pass
        async def escalate(self, _p, **_kw): pass
        async def notify_executed(self, _p): pass

    class _StubAudit:
        async def log(self, **_kw): pass

    stub_db = _StubDB()
    mc = MakerCheckerWorkflow(stub_db, _StubAuth(), _StubAlerts(), _StubAudit())

    no_holiday = lambda d: False
    wm  = WindowManager(no_holiday, no_holiday, no_holiday, no_holiday)
    cal = CalendarService()
    bus = InMemoryBus()

    # ── ClearBank client (optional — only when enabled + credentials present) ──
    cb_client        = None
    clearbank_nostro = {}
    if settings.clearbank_enabled and settings.clearbank_token:
        try:
            from agents.operations.clearbank_client import ClearBankClient, ClearBankConfig
            cb_config = ClearBankConfig(
                token               = settings.clearbank_token,
                base_url            = settings.clearbank_base_url,
                private_key         = settings.clearbank_private_key,
                clearbank_account_id = settings.clearbank_account_id,
                source_account_id   = settings.clearbank_source_account_id,
                legal_name          = settings.clearbank_legal_name,
                legal_address       = settings.clearbank_legal_address,
            )
            cb_client = ClearBankClient(cb_config)
            # Default nostro map: TMS internal ID → ClearBank IBAN
            clearbank_nostro = {
                "NOSTRO-GBP-001": {
                    "iban": settings.clearbank_source_account_id or "GB00CLRB04010000000001",
                    "name": "Aspora GBP Nostro",
                }
            }
            import logging
            logging.getLogger("tms.app").info(
                "ClearBank client initialised (base_url=%s)", settings.clearbank_base_url
            )
        except Exception as exc:
            import logging
            logging.getLogger("tms.app").warning(
                "ClearBank client init failed (continuing without it): %s", exc
            )

    return create_app(fm, mc, wm, cal, bus, cb_client=cb_client, clearbank_nostro_map=clearbank_nostro)


app = _make_default_app()
