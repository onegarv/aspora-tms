"""
Dev trigger router — POST /dev/trigger

Seeds the InMemoryBus by immediately running all agent daily routines and
forcing the FXAnalystAgent to poll (bypassing the trading-hours guard).

Agents run as background tasks so this endpoint returns instantly.
ONLY registered in development / local mode (run.py).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/dev", tags=["dev"])
logger = logging.getLogger("tms.dev")


async def _run_agents(app) -> None:
    """Background coroutine: fire all agents and log results."""
    # ── FXAnalystAgent ────────────────────────────────────────────────────────
    fx_agent = getattr(app.state, "fx_agent", None)
    if fx_agent is not None:
        try:
            await fx_agent._poll_and_emit(force=True)
            logger.info("dev trigger: fx_analyst ok")
        except Exception as exc:
            logger.warning("dev trigger: fx_analyst error: %s", exc)

    # ── LiquidityAgent ────────────────────────────────────────────────────────
    liq_agent = getattr(app.state, "liq_agent", None)
    if liq_agent is not None:
        try:
            await liq_agent.run_daily()
            logger.info("dev trigger: liquidity ok")
        except Exception as exc:
            logger.warning("dev trigger: liquidity error: %s", exc)

    # ── OperationsAgent ───────────────────────────────────────────────────────
    ops_agent = getattr(app.state, "ops_agent", None)
    if ops_agent is not None:
        try:
            await ops_agent.run_daily()
            logger.info("dev trigger: operations ok")
        except Exception as exc:
            logger.warning("dev trigger: operations error: %s", exc)

    # Log bus state after all agents complete
    bus = getattr(app.state, "bus", None)
    if bus is not None:
        all_events = bus.get_events()
        summary: dict[str, int] = {}
        for e in all_events:
            summary[e.event_type] = summary.get(e.event_type, 0) + 1
        logger.info("dev trigger complete — bus events: %s", summary)


@router.post("/trigger")
async def trigger_all(request: Request) -> JSONResponse:
    """
    Fire all agents as a background task and return immediately.
    Check /api/v1/events to watch events populate in real time.
    """
    attached = {
        "fx_analyst":  getattr(request.app.state, "fx_agent",  None) is not None,
        "liquidity":   getattr(request.app.state, "liq_agent", None) is not None,
        "operations":  getattr(request.app.state, "ops_agent", None) is not None,
    }

    # Fire all agents in the background — don't block the HTTP response
    asyncio.create_task(_run_agents(request.app))

    return JSONResponse({
        "status": "triggered",
        "agents": attached,
        "note": "Agents running in background. Watch /api/v1/events for results.",
    })


@router.get("/status")
async def bus_status(request: Request) -> JSONResponse:
    """Return a count of events per type currently in the bus."""
    bus = getattr(request.app.state, "bus", None)
    if bus is None:
        return JSONResponse({"error": "bus not available"})

    all_events = bus.get_events()
    summary: dict[str, int] = {}
    for e in all_events:
        summary[e.event_type] = summary.get(e.event_type, 0) + 1

    return JSONResponse({"total": len(all_events), "by_type": summary})
