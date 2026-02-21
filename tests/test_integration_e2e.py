"""
End-to-end integration test: Liquidity Agent ↔ Operations Agent on a holiday-eve.

Scenario
--------
Today is Thursday 2026-02-26.
The next day (2026-02-27) is injected as a confirmed IN_RBI_FX bank holiday.

Architecture note
-----------------
BalanceTracker   = operating/source account  (rich — this is what the OpsAgent
                   checks when it wants to SEND money to fund a nostro)
Nostro balances  = destination account state  (thin — published to the bus via
                   NOSTRO_BALANCE_UPDATE so LiquidityAgent can detect a shortfall)

In production these come from different sources: the BalanceTracker is updated
from bank API responses for the operating account; nostro balances are fetched
separately.  In this test we inject a synthetic NOSTRO_BALANCE_UPDATE directly
onto the bus so we can keep the operating account healthy.

Expected event chain
--------------------
  1. Synthetic NOSTRO_BALANCE_UPDATE published (USD nostro = 1 M — thin)
       → LiquidityAgent._handle_nostro_update stores the balance
         BUG UNDER TEST: OpsAgent emits balances as *strings*; the original
         `isinstance(v, (int, float))` filter silently discarded them.
         The fixed code uses float(v) with try/except so strings are accepted.

  2. liq.run_daily()
       → MultiplierEngine detects day-before-holiday → "holiday" 1.2× applied
       → base 5 M/corridor × 1.2 = 6 M adjusted
       → FORECAST_READY  (elevated total, multipliers_applied["holiday"] == 1.2)
       → ops.handle_forecast stores it, re-emits NOSTRO_BALANCE_UPDATE
       → RDAChecker: 6 M × 1.1 buffer = 6.6 M required vs 1 M available → CRITICAL
       → SHORTFALL_ALERT (currency=USD, severity=critical)
         carries the same correlation_id as FORECAST_READY

  3. ops.handle_shortfall()
       → BalanceTracker (operating account) has 20 M USD → balance check passes
       → WindowManager: Fedwire opens today at 14:00 UTC, before tomorrow's
         INR deadline (2026-02-27 03:30 UTC) → window feasible
       → FundMovementProposal submitted to MakerChecker
       → FUND_MOVEMENT_STATUS (status=pending_approval, currency=USD)
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from freezegun import freeze_time

from agents.liquidity.agent import LiquidityAgent
from agents.operations.agent import OperationsAgent
from agents.operations.fund_mover import (
    BalanceTracker,
    FundMover,
    FundMoverConfig,
    InMemoryExecutionStore,
    MockBankAPI,
)
from agents.operations.window_manager import WindowManager
from bus.events import (
    FORECAST_READY,
    FUND_MOVEMENT_STATUS,
    NOSTRO_BALANCE_UPDATE,
    SHORTFALL_ALERT,
    create_event,
)
from bus.memory_bus import InMemoryBus
from models.domain import FundMovementProposal
from services.calendar_service import CalendarService, IN_RBI_FX


# ── Helpers ───────────────────────────────────────────────────────────────────


def _no_holidays(_d):
    return False


def _make_window_manager() -> WindowManager:
    return WindowManager(
        usd_holidays=_no_holidays,
        gbp_holidays=_no_holidays,
        inr_holidays=_no_holidays,
        aed_holidays=_no_holidays,
        operational_buffer_min=0,
    )


def _eight_thursdays(base_vol_usd: float = 5_000_000.0) -> dict[str, list[dict]]:
    """8 weeks of Thursday (dow=3) history for every corridor."""
    from datetime import timedelta
    anchor = date(2025, 12, 25)          # a known Thursday
    rows = [
        {"date": (anchor - timedelta(weeks=w)).isoformat(),
         "volume_usd": base_vol_usd,
         "dow": 3}
        for w in range(8)
    ]
    return {corridor: rows for corridor in ["USD_INR", "GBP_INR", "AED_INR"]}


class _MockMC:
    """Minimal MakerChecker stub — records proposals, always returns pending_approval."""

    def __init__(self) -> None:
        self.submitted: list[FundMovementProposal] = []

    async def submit_proposal(self, proposal: FundMovementProposal) -> dict:
        self.submitted.append(proposal)
        return {"status": "pending_approval", "proposal_id": proposal.id}


# ── Scenario ──────────────────────────────────────────────────────────────────

# Thursday 2026-02-26 09:30 UTC = 15:00 IST (trading hours).
# Fedwire opens today at 14:00 UTC, which is before tomorrow's INR deadline
# (2026-02-27 03:30 UTC), so the window feasibility check returns True.
@freeze_time("2026-02-26 09:30:00+00:00")
async def test_holiday_eve_full_chain():
    """Full event chain on a holiday-eve: elevated forecast → shortfall → fund movement."""

    # ── 1. Shared infrastructure ──────────────────────────────────────────────
    bus = InMemoryBus()
    cal = CalendarService()

    # Inject 2026-02-27 as a confirmed RBI bank holiday (Layer 3 runtime override).
    cal.add_holiday(
        date(2026, 2, 27), IN_RBI_FX,
        name="Test Bank Holiday",
        user="integration_test",
    )

    # ── 2. OperationsAgent — RICH operating/source account (20 M USD) ─────────
    #
    # BalanceTracker = source operating account, NOT the nostro.
    # OpsAgent checks this when deciding whether it can fund a transfer.
    # The nostro balance (thin: 1 M USD) is injected separately via an event.
    bank    = MockBankAPI()
    store   = InMemoryExecutionStore()
    tracker = BalanceTracker({
        "USD": Decimal("20_000_000"),   # operating account — plenty to fund
        "GBP": Decimal("10_000_000"),
        "AED": Decimal("10_000_000"),
    })
    fm = FundMover(
        bank_api=bank, store=store, balance_tracker=tracker,
        calendar_svc=cal, config=FundMoverConfig(poll_interval_sec=0.0),
    )
    mc  = _MockMC()
    wm  = _make_window_manager()
    ops = OperationsAgent(
        bus=bus, calendar=cal, window_manager=wm,
        maker_checker=mc, fund_mover=fm,
        config={
            "prefunding_buffer_pct":    0.10,
            "window_closing_alert_min": 30,
            "nostro_topup_trigger_pct": 0.90,
            "topup_target_pct":         1.20,
            "monitor_interval_sec":     0.0,
            "stale_proposal_age_min":   90,
            "lookahead_days":           3,
            "monitored_currencies":     ["USD", "GBP", "AED"],
        },
    )

    # ── 3. LiquidityAgent — pre-loaded with 8 weeks of Thursday history ───────
    liq = LiquidityAgent(bus=bus, calendar=cal)
    liq._forecaster.load(_eight_thursdays(base_vol_usd=5_000_000.0))
    liq._spot_rates = {
        "USD_INR": 86.0,
        "GBP_INR": 108.0,
        "EUR_INR":  92.0,
        "AED_INR":  23.4,
    }

    # ── 4. Wire agents, start bus ──────────────────────────────────────────────
    await ops.setup()
    await liq.setup()
    await bus.start()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Inject thin nostro balances as string values (as OpsAgent sends them)
    #
    # This directly exercises the type-parse fix in _handle_nostro_update:
    # original code silently dropped str values (isinstance int/float filter).
    # ═══════════════════════════════════════════════════════════════════════════
    nostro_update = create_event(
        event_type="ops.nostro.balance.update",
        source_agent="operations",
        payload={
            "balances": {
                "USD": "1000000",    # string — as emitted by OpsAgent (Decimal → str)
                "GBP": "8000000",
                "AED": "8000000",
            }
        },
    )
    await bus.publish(nostro_update)

    # Verify the parse fix: string values must be accepted
    assert "USD" in liq._nostro_balances, (
        "BUG: LiquidityAgent._handle_nostro_update dropped string balance values. "
        "OpsAgent emits Decimal-as-str; isinstance(v, (int,float)) silently discards them."
    )
    assert liq._nostro_balances["USD"] == pytest.approx(1_000_000.0), (
        "USD nostro balance should be parsed from string '1000000' → float 1_000_000.0"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Liquidity runs daily routine on a holiday-eve
    #
    # Key sequencing: pause the bus BEFORE liq.run_daily() so the FORECAST_READY
    # event it emits is queued but NOT delivered to OpsAgent yet.  This prevents
    # OpsAgent.handle_forecast() from overwriting our thin nostro balances (with
    # its own 20 M operating-account values) before the RDA check runs.
    # We flush the queue afterwards so OpsAgent processes SHORTFALL_ALERT.
    # ═══════════════════════════════════════════════════════════════════════════
    await bus.stop()          # pause — published events will queue, not deliver
    await liq.run_daily()     # RDA check sees 1 M USD → CRITICAL shortfall queued
    await bus.start()         # flush: FORECAST_READY → ops, SHORTFALL_ALERT → ops

    # ── 2a. FORECAST_READY must carry the pre-holiday 1.2× multiplier ─────────
    forecast_events = bus.get_events(FORECAST_READY)
    assert len(forecast_events) == 1, "Exactly one FORECAST_READY should be on the bus"

    forecast_payload = forecast_events[0].payload
    multipliers      = forecast_payload["multipliers_applied"]

    assert "holiday" in multipliers, (
        "MultiplierEngine should detect day-before-IN_RBI_FX-holiday and add "
        "'holiday' key — 2026-02-26 is the eve of the injected 2026-02-27 holiday"
    )
    assert multipliers["holiday"] == pytest.approx(1.2), (
        "Pre-holiday multiplier should be 1.2× (SPEC §3.2.2 — day before IN_RBI_FX)"
    )
    assert forecast_payload["total_inr_crores"] > 0

    # ── 2b. SHORTFALL_ALERT fires for the thin USD nostro ─────────────────────
    shortfall_events = bus.get_events(SHORTFALL_ALERT)
    assert len(shortfall_events) >= 1, (
        "RDAChecker must fire SHORTFALL_ALERT: adjusted USD demand (~6.6 M) "
        "exceeds the thin nostro balance (1 M)"
    )

    usd_sf = next(
        (e for e in shortfall_events if e.payload["currency"] == "USD"), None
    )
    assert usd_sf is not None, "USD corridor must be flagged — it is the most underfunded"
    assert usd_sf.payload["severity"] == "critical", (
        "1 M balance vs ~6 M raw requirement → below raw threshold → CRITICAL"
    )

    # ── 2c. Correlation ID threads FORECAST_READY → SHORTFALL_ALERT ───────────
    assert usd_sf.correlation_id == forecast_events[0].correlation_id, (
        "SHORTFALL_ALERT must carry the same correlation_id as the FORECAST_READY "
        "that triggered the RDA check — required for end-to-end event chain tracing"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Ops handles shortfall: window check → MakerChecker → status event
    # ═══════════════════════════════════════════════════════════════════════════
    fund_status_events = bus.get_events(FUND_MOVEMENT_STATUS)
    assert len(fund_status_events) >= 1, (
        "OpsAgent must emit FUND_MOVEMENT_STATUS after handling SHORTFALL_ALERT"
    )

    # Find the USD proposal status (skip GBP/AED which may also be underfunded)
    usd_status = next(
        (e for e in fund_status_events if e.payload.get("currency") == "USD"), None
    )
    assert usd_status is not None, "USD FUND_MOVEMENT_STATUS should be present"
    assert usd_status.payload["status"] == "pending_approval", (
        f"Expected pending_approval but got {usd_status.payload['status']!r}. "
        "Check window feasibility and operating account balance."
    )

    # MakerChecker received the proposal
    usd_proposals = [p for p in mc.submitted if p.currency == "USD"]
    assert len(usd_proposals) == 1, "Exactly one USD fund movement proposal should be submitted"

    proposal = usd_proposals[0]
    # shortfall ≈ 6.6 M − 1 M = 5.6 M;  transfer = 5.6 M × 1.1 buffer ≈ 6.16 M
    assert proposal.amount > 5_000_000, (
        f"Transfer {proposal.amount:,.0f} too low — expected ~6.16 M for holiday-eve shortfall"
    )
    assert proposal.amount < 10_000_000, "Transfer should not breach single-deal risk limit (10 M)"

    # ── Summary (visible with pytest -s) ──────────────────────────────────────
    print("\n── Holiday-Eve Integration Chain ──────────────────────────────────")
    print(f"  Injected holiday:       2026-02-27 (IN_RBI_FX)")
    print(f"  Multipliers applied:    {multipliers}")
    print(f"  Total INR crores:       {forecast_payload['total_inr_crores']:.2f}")
    print(f"  USD nostro (thin):      1,000,000")
    print(f"  USD shortfall:          {usd_sf.payload['shortfall']:,.0f}  severity={usd_sf.payload['severity']}")
    print(f"  Transfer proposed:      {float(proposal.amount):,.0f} USD")
    print(f"  Proposal status:        {usd_status.payload['status']}")
    print(f"  Correlation chain:      FORECAST_READY ──▶ SHORTFALL_ALERT ✓ (id={forecast_events[0].correlation_id[:8]}…)")
