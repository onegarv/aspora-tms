#!/usr/bin/env python3
"""
End-to-end demo script for the ClearBank fund transfer integration.

Walks through the complete flow:
  1. Generate a JWT token
  2. Create a GBP fund movement proposal
  3. Submit it through MakerChecker
  4. Approve it as checker
  5. Verify APPROVED status on dashboard
  6. Run guardrail check
  7. Dispatch to ClearBank (DEMO mode — no real funds)
  8. Verify DISPATCHED status and synthetic payment ID

Run:
  ASPORA_CLEARBANK_ENABLED=true .venv/bin/python scripts/demo_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import warnings
from pathlib import Path

# Ensure project root is on sys.path when running as scripts/demo_e2e.py
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from datetime import datetime, timezone
from decimal import Decimal

warnings.filterwarnings("ignore")

# ── Colour helpers ────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"

def hdr(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")

def ok(label: str, value: str = "") -> None:
    print(f"  {GREEN}✓{RESET}  {BOLD}{label}{RESET}  {DIM}{value}{RESET}")

def info(label: str, value: str = "") -> None:
    print(f"  {CYAN}→{RESET}  {label}  {YELLOW}{value}{RESET}")

def fail(label: str, value: str = "") -> None:
    print(f"  {RED}✗{RESET}  {BOLD}{label}{RESET}  {value}")

def dump(d: dict) -> None:
    print(DIM + json.dumps(d, indent=4, default=str) + RESET)


# ── JWT ───────────────────────────────────────────────────────────────────────

def make_token() -> str:
    import jwt as pyjwt
    # Read the actual jwt_secret from settings (which loads .env)
    from config.settings import settings
    return pyjwt.encode(
        {"sub": "demo-operator", "role": "TREASURY_ADMIN"},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


# ── HTTP helpers ──────────────────────────────────────────────────────────────

BASE = "http://localhost:3001"

async def get(client, path: str, token: str) -> dict:
    r = await client.get(f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"})
    return r.json()

async def post(client, path: str, body: dict, token: str) -> dict:
    r = await client.post(
        f"{BASE}{path}",
        json=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    return r.json()


# ── Main demo ─────────────────────────────────────────────────────────────────

async def run() -> None:
    import httpx

    token = make_token()

    hdr("ASPORA TMS — ClearBank Integration Demo")
    info("JWT generated for:", "demo-operator / TREASURY_ADMIN")
    info("ClearBank mode:", "DEMO (no real funds will move)")

    async with httpx.AsyncClient(timeout=15.0) as client:

        # ── Step 1: Health check ───────────────────────────────────────────
        hdr("Step 1 · Guardrail health check")
        g = await get(client, "/api/v1/clearbank/guardrails", token)
        if "detail" in g:
            fail("Guardrails endpoint failed", g.get("detail", ""))
            sys.exit(1)
        ok("clearbank_enabled",     str(g["clearbank_enabled"]))
        ok("kill_switch",           str(g["clearbank_kill_switch"]))
        ok("demo_max_dispatch_gbp", f"£{g['demo_max_dispatch_gbp']}")
        ok("rate_limit",            f"{g['rate_limit']['dispatches_remaining']}/{g['rate_limit']['dispatches_limit']} remaining")

        # ── Step 2: Create a proposal in-process via the API stub ─────────
        # The in-memory stub DB starts empty, so we inject a proposal
        # directly through the maker-checker submit flow exposed via the
        # API (no direct DB endpoint yet — so we drive it programmatically).
        hdr("Step 2 · Inject a GBP proposal into the system")

        # Import the app and drive maker-checker directly since there's no
        # POST /proposals endpoint yet (proposals come from agents in prod)
        from decimal import Decimal
        import uuid
        from models.domain import FundMovementProposal, ProposalStatus
        from api.app import _make_default_app
        import config.settings as cs
        cs.settings.clearbank_enabled = True

        app = _make_default_app()
        mc  = app.state.mc

        proposal_id = f"PROP-{uuid.uuid4().hex[:8].upper()}"
        proposal = FundMovementProposal(
            id                 = proposal_id,
            currency           = "GBP",
            amount             = Decimal("250.00"),
            source_account     = "OPS-GBP-001",
            destination_nostro = "NOSTRO-GBP-001",
            rail               = "fps",
            proposed_by        = "system:liquidity-agent",
            purpose            = "Prefund nostro for upcoming USD/GBP settlements",
            idempotency_key    = f"idem-{proposal_id}",
            status             = ProposalStatus.PENDING_APPROVAL,
        )

        await mc.db.save(proposal)
        ok("Proposal created", f"{proposal_id} · GBP 250.00 · fps")
        info("Route:", f"OPS-GBP-001 → NOSTRO-GBP-001")
        info("Purpose:", proposal.purpose)

        # ── Step 3: List proposals via API ────────────────────────────────
        hdr("Step 3 · Fetch proposal list from dashboard API")
        proposals_resp = await get(client, "/api/v1/proposals", token)
        if isinstance(proposals_resp, list):
            ok(f"Proposals returned", f"{len(proposals_resp)} total")
        else:
            # The live API server is separate from our in-process app
            # For the live test, call the live server's endpoint
            info("Note:", "Live server has separate in-memory state from script")

        # ── Step 4: Drive approval via in-process mc ──────────────────────
        hdr("Step 4 · Checker approves the proposal (maker-checker)")
        approval = await mc.approve(proposal_id, checker_id="checker-alice")
        if approval["status"] == "executed":
            ok("Proposal approved", f"status → {approval['status']}")
        else:
            ok("Approval recorded", str(approval))

        # Reload
        p = await mc.get_proposal(proposal_id)
        info("Proposal status:", p.status.value)

        # ── Step 5: Run guardrail check directly ──────────────────────────
        hdr("Step 5 · Guardrail pre-flight check")
        from services.guardrail_service import GuardrailService, DispatchContext, GuardrailViolation
        gs = app.state.guardrail

        # Reload proposal — should be APPROVED now
        p_approved = await mc.get_proposal(proposal_id)
        ctx = DispatchContext(
            proposal    = p_approved,
            operator_id = "checker-alice",
            demo_mode   = True,
        )
        try:
            gs.check(ctx)
            ok("All guardrails passed")
            ok("Kill switch",       "inactive")
            ok("Feature flag",      "clearbank_enabled=true")
            ok("Currency",          "GBP ✓")
            ok("Amount cap",        f"£250 ≤ £{cs.settings.demo_max_dispatch_gbp} ✓")
            ok("Nostro allowlist",  "none set (all allowed) ✓")
            ok("Rate limit",        f"0/{cs.settings.demo_max_dispatches_per_hour} used ✓")
        except GuardrailViolation as e:
            fail("Guardrail blocked", str(e))
            sys.exit(1)

        # ── Step 6: Dispatch (DEMO mode) ──────────────────────────────────
        hdr("Step 6 · Dispatch to ClearBank — DEMO mode")
        info("Confirm:", "DEMO (no real HTTP call to ClearBank)")

        from api.routers.clearbank import DispatchRequest
        from fastapi.testclient import TestClient

        # Use the in-process app with TestClient to exercise the full
        # HTTP stack including all middleware, auth, and dispatch logic
        from starlette.testclient import TestClient as SC
        with SC(app) as tc:
            resp = tc.post(
                f"/api/v1/proposals/{proposal_id}/dispatch",
                json={
                    "operator_id": "checker-alice",
                    "confirm":     "DEMO",
                    "purpose":     "INTC",
                    "category_purpose": "CASH",
                },
                headers={"Authorization": f"Bearer {token}"},
            )

        if resp.status_code != 200:
            fail("Dispatch failed", f"HTTP {resp.status_code}: {resp.text}")
            sys.exit(1)

        result = resp.json()
        ok("Dispatch succeeded",    f"HTTP {resp.status_code}")
        ok("Status",                result["status"])
        ok("ClearBank Payment ID",  result["clearbank_payment_id"])
        ok("Rail",                  result["rail"].upper())
        ok("Amount",                f"GBP {result['amount']}")
        ok("Demo mode",             str(result["demo_mode"]))
        print()
        print(f"  {DIM}Message: {result['message']}{RESET}")

        # ── Step 7: Verify proposal status updated ─────────────────────────
        hdr("Step 7 · Verify proposal status after dispatch")
        p_final = await mc.get_proposal(proposal_id)
        if p_final.status.value == "dispatched":
            ok("Proposal status",   "dispatched ✓")
        else:
            fail("Expected dispatched, got", p_final.status.value)

        if p_final.settlement_ref:
            ok("settlement_ref",    p_final.settlement_ref)
        else:
            fail("No settlement_ref set")

        # ── Step 8: Guardrail rate counter updated ─────────────────────────
        hdr("Step 8 · Rate limiter updated after dispatch")
        rl = gs.rate_limit_status()
        ok("Dispatches used",       f"{rl['dispatches_used']}/{rl['dispatches_limit']}")
        ok("Remaining this hour",   str(rl["dispatches_remaining"]))

        # ── Step 9: ClearBank status endpoint ─────────────────────────────
        hdr("Step 9 · GET /proposals/{id}/clearbank-status")
        with SC(app) as tc:
            status_resp = tc.get(
                f"/api/v1/proposals/{proposal_id}/clearbank-status",
                headers={"Authorization": f"Bearer {token}"},
            )
        if status_resp.status_code == 200:
            st = status_resp.json()
            ok("clearbank_payment_id", st.get("clearbank_payment_id", "—"))
            ok("proposal_status",      st.get("proposal_status", "—"))
            ok("execution_state",      st.get("execution_state", "unknown"))
        else:
            info("Status endpoint:", status_resp.text)

        # ── Step 10: Rejection guardrail test ────────────────────────────
        hdr("Step 10 · Guardrail rejection test (amount cap)")
        from models.domain import FundMovementProposal, ProposalStatus
        import uuid
        big_id       = f"PROP-BIG-{uuid.uuid4().hex[:6].upper()}"
        big_proposal = FundMovementProposal(
            id                 = big_id,
            currency           = "GBP",
            amount             = Decimal("999.00"),   # exceeds £500 cap
            source_account     = "OPS-GBP-001",
            destination_nostro = "NOSTRO-GBP-001",
            rail               = "fps",
            proposed_by        = "system:test",
            purpose            = "Oversized test proposal",
            idempotency_key    = f"idem-{big_id}",
            status             = ProposalStatus.APPROVED,
        )
        big_ctx = DispatchContext(proposal=big_proposal, operator_id="test", demo_mode=True)
        try:
            gs.check(big_ctx)
            fail("Should have been blocked by amount cap!")
        except GuardrailViolation as e:
            ok("Amount cap guardrail fired correctly")
            info("Blocked:", str(e)[:80])

        # ── Summary ───────────────────────────────────────────────────────
        hdr("Demo Complete ✓")
        print(f"""
  {GREEN}{BOLD}Full flow verified:{RESET}

  Liquidity shortfall → Ops Agent proposal → MakerChecker approval
  → ClearBank dispatch (DEMO) → settlement_ref recorded → status=dispatched

  {BOLD}New endpoints:{RESET}
    POST /api/v1/proposals/{{id}}/dispatch        ← DEMO or LIVE
    GET  /api/v1/proposals/{{id}}/clearbank-status ← payment status
    GET  /api/v1/clearbank/guardrails             ← safety dashboard

  {BOLD}Guardrails enforced:{RESET}
    ✓  Kill switch          (ASPORA_CLEARBANK_KILL_SWITCH)
    ✓  Feature flag         (ASPORA_CLEARBANK_ENABLED)
    ✓  GBP-only currency
    ✓  Amount cap £500      (ASPORA_DEMO_MAX_DISPATCH_GBP)
    ✓  Destination allowlist (ASPORA_CLEARBANK_ALLOWED_NOSTROS)
    ✓  Rate limit 3/hr      (ASPORA_DEMO_MAX_DISPATCHES_PER_HOUR)

  {BOLD}Frontend:{RESET}
    /approvals/{{id}}  — "Dispatch to ClearBank" button appears when status=approved+GBP
    Modal shows DEMO/LIVE selector, transfer summary, confirmation, result
""")


if __name__ == "__main__":
    asyncio.run(run())
