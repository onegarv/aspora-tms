"""
Aspora TMS — Agent entrypoint.

Starts OperationsAgent + LiquidityAgent with APScheduler (daily at 06:00 IST / 00:30 UTC).

Bus selection:
  ASPORA_BUS_TYPE=memory (default) — InMemoryBus, no external deps required
  ASPORA_BUS_TYPE=redis            — RedisBus; requires Redis at ASPORA_REDIS_URL

Usage:
    python main.py
    ASPORA_BUS_TYPE=redis python main.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from decimal import Decimal

import structlog

# ── Logging setup ─────────────────────────────────────────────────────────────

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

log = structlog.get_logger("tms.main")


# ── Bus factory ───────────────────────────────────────────────────────────────

async def _build_bus():
    bus_type = os.environ.get("ASPORA_BUS_TYPE", "memory").lower()

    if bus_type == "redis":
        redis_url = os.environ.get("ASPORA_REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(redis_url, socket_connect_timeout=2)
            await client.ping()
            await client.aclose()

            from bus.redis_bus import RedisBus
            log.info("bus.type=redis", url=redis_url)
            return RedisBus(redis_url=redis_url)
        except Exception as exc:
            log.warning(
                "Redis unreachable — falling back to InMemoryBus",
                url=redis_url,
                error=str(exc),
            )

    from bus.memory_bus import InMemoryBus
    log.info("bus.type=memory")
    return InMemoryBus()


# ── Service factory ───────────────────────────────────────────────────────────

def _build_services():
    from agents.operations.fund_mover import (
        BalanceTracker,
        FundMover,
        InMemoryExecutionStore,
        MockBankAPI,
    )
    from agents.operations.maker_checker import MakerCheckerWorkflow
    from agents.operations.window_manager import WindowManager
    from services.calendar_service import CalendarService

    calendar = CalendarService()

    # Balance tracker — seeded with dev-safe starting balances
    tracker = BalanceTracker({
        "USD": Decimal("10_000_000"),
        "GBP": Decimal("5_000_000"),
        "AED": Decimal("2_000_000"),
    })
    fund_mover = FundMover(MockBankAPI(), InMemoryExecutionStore(), tracker)

    class _StubDB:
        async def get(self, _id):
            return None
        async def save(self, _p):
            pass
        async def list_all(self):
            return []

    class _StubAudit:
        async def record(self, *_a, **_kw):
            pass

    maker_checker = MakerCheckerWorkflow(db=_StubDB(), audit_log=_StubAudit())
    window_manager = WindowManager(calendar)

    return calendar, fund_mover, maker_checker, window_manager


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("aspora-tms starting")

    bus = await _build_bus()
    calendar, fund_mover, maker_checker, window_manager = _build_services()

    # ── Build agents ──────────────────────────────────────────────────────────
    from agents.liquidity.agent import LiquidityAgent
    from agents.operations.agent import OperationsAgent

    liquidity_agent = LiquidityAgent(bus=bus, calendar=calendar)
    ops_agent = OperationsAgent(
        bus=bus,
        calendar=calendar,
        window_manager=window_manager,
        maker_checker=maker_checker,
        fund_mover=fund_mover,
    )

    # ── Start agents (registers event handlers) ───────────────────────────────
    await liquidity_agent.start()
    await ops_agent.start()
    log.info("agents started", agents=["liquidity", "operations"])

    # ── Scheduler: 06:00 IST = 00:30 UTC ─────────────────────────────────────
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        liquidity_agent.run_daily,
        trigger="cron",
        hour=0,
        minute=30,
        id="liquidity_daily",
        name="LiquidityAgent daily run",
    )
    scheduler.add_job(
        ops_agent.run_daily,
        trigger="cron",
        hour=0,
        minute=30,
        id="ops_daily",
        name="OperationsAgent daily run",
    )
    scheduler.start()
    log.info("scheduler started", cron="00:30 UTC (06:00 IST)")

    # ── Run until interrupted ─────────────────────────────────────────────────
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal():
        log.info("shutdown signal received")
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        import signal as _signal
        sig = getattr(_signal, sig_name, None)
        if sig is not None:
            loop.add_signal_handler(sig, _handle_signal)

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        log.info("shutting down")
        scheduler.shutdown(wait=False)
        await ops_agent.shutdown()
        await bus.stop()
        log.info("aspora-tms stopped")


if __name__ == "__main__":
    asyncio.run(main())
