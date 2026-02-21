#!/usr/bin/env python3
"""
TMS Orchestrator — Fetch FX rates → Compute daily TPV requirement.

Flow:
  1. Call FX Band Predictor API (GET /predict) to get USD/INR predicted rate.
     On weekends, the predictor returns Friday's last known rate.
  2. Feed rates into the Liquidity Agent's forecaster + multiplier engine.
  3. Compute total TPV required per corridor (in USD and INR crores).
  4. Print a clean summary.

Usage:
    python orchestrator.py              # normal run (uses live FX Band Predictor)
    python orchestrator.py --mock       # use mock data (no API needed)

Requires:
    FX Band Predictor running on localhost:8001  (unless --mock is used)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from data.fx_predictor_client import fetch_prediction, extract_spot_rate, extract_band
from agents.liquidity.forecaster import VolumeForecaster
from agents.liquidity.multipliers import MultiplierEngine
from services.calendar_service import CalendarService


# ── Fallback rates if FX Band Predictor is down ────────────────────────────
_FALLBACK_RATES = {
    "USD_INR": 90.73,
    "GBP_INR": 122.31,
    "EUR_INR": 107.22,
    "AED_INR": 24.71,
}

# ── Representative corridor volumes (USD) for when Metabase is unavailable ─
# These are illustrative defaults; in production, Metabase supplies real data.
_SAMPLE_VOLUMES = {
    "AED_INR": [],
    "GBP_INR": [],
    "USD_INR": [],
    "EUR_INR": [],
}


def _generate_sample_volumes() -> dict[str, list[dict]]:
    """Generate 10 weeks of synthetic same-weekday volume data per corridor."""
    base_volumes = {
        "AED_INR": 4_500_000,   # $4.5M avg daily
        "GBP_INR": 1_800_000,   # $1.8M
        "USD_INR": 2_500_000,   # $2.5M
        "EUR_INR":   900_000,   # $0.9M
    }
    import random
    random.seed(42)
    volumes: dict[str, list[dict]] = {c: [] for c in base_volumes}
    today = date.today()

    for weeks_back in range(10):
        for dow in range(5):  # Mon-Fri
            d = today - timedelta(days=today.weekday()) - timedelta(weeks=weeks_back) + timedelta(days=dow)
            for corridor, base in base_volumes.items():
                jitter = random.uniform(0.85, 1.15)
                volumes[corridor].append({
                    "date": d.isoformat(),
                    "volume_usd": round(base * jitter, 2),
                    "dow": dow,
                })
    return volumes


def _resolve_forecast_date() -> date:
    """
    Return the next trading day.
    If today is Saturday or Sunday, return next Monday.
    Otherwise return tomorrow (or today if before market open).
    """
    today = date.today()
    dow = today.weekday()

    if dow == 5:        # Saturday → Monday
        return today + timedelta(days=2)
    elif dow == 6:      # Sunday → Monday
        return today + timedelta(days=1)
    else:
        # Weekday: forecast for next trading day
        nxt = today + timedelta(days=1)
        if nxt.weekday() == 5:  # Friday → forecast for Monday
            return nxt + timedelta(days=2)
        return nxt


def run(use_mock: bool = False) -> None:
    forecast_date = _resolve_forecast_date()

    print()
    print("=" * 70)
    print("  TMS ORCHESTRATOR — Daily TPV Forecast")
    print(f"  Run date : {date.today().isoformat()} ({date.today().strftime('%A')})")
    print(f"  Forecast : {forecast_date.isoformat()} ({forecast_date.strftime('%A')})")
    print("=" * 70)

    # ── Step 1: Fetch FX prediction ────────────────────────────────────
    print("\n  STEP 1: Fetch USD/INR rate from FX Band Predictor")

    usd_inr = _FALLBACK_RATES["USD_INR"]
    band = None
    prediction = None

    if use_mock:
        usd_inr = 85.50
        band = {
            "direction": "UP",
            "range_low": 85.20,
            "range_high": 85.80,
            "most_likely": 85.50,
            "confidence": 0.72,
        }
        print(f"    [mock] USD/INR spot  : {usd_inr:.4f}")
    else:
        prediction = fetch_prediction()
        if prediction:
            rate = extract_spot_rate(prediction)
            if rate:
                usd_inr = rate
            band = extract_band(prediction)
            print(f"    USD/INR current rate : {usd_inr:.4f}")
        else:
            print(f"    FX Band Predictor unreachable — using fallback: {usd_inr:.4f}")

    if band:
        direction = band.get("direction", "N/A")
        conf = band.get("confidence", 0)
        low = band.get("range_low", 0)
        high = band.get("range_high", 0)
        likely = band.get("most_likely", 0)
        action = band.get("action", "N/A")
        print(f"    48h direction       : {direction} (confidence: {conf:.0%})")
        print(f"    48h range           : {low:.2f} — {high:.2f}")
        print(f"    Most likely close   : {likely:.2f}")
        print(f"    Prefunding action   : {action}")

    # ── Use the HIGHER band rate for conservative TPV calculation ──────
    # Prefunding should use worst-case rate to avoid under-funding.
    # Higher USD/INR = more INR needed per dollar = conservative estimate.
    spot_rate = usd_inr
    if band and band.get("range_high"):
        usd_inr = band["range_high"]
        print(f"\n    >>> Using HIGHER BAND rate for TPV: {usd_inr:.4f} (spot was {spot_rate:.4f})")
    else:
        print(f"\n    >>> No band available — using spot rate: {usd_inr:.4f}")

    rates = dict(_FALLBACK_RATES)
    rates["USD_INR"] = usd_inr

    # ── Step 2: Load volume history ────────────────────────────────────
    print("\n  STEP 2: Load corridor volume history")

    forecaster = VolumeForecaster(lookback_weeks=8, decay=0.85)

    # Try Metabase first, fall back to sample data
    volumes = None
    try:
        from data.metabase import fetch_corridor_volumes
        volumes = fetch_corridor_volumes(lookback_days=70)
    except Exception:
        pass

    if volumes:
        forecaster.load(volumes)
        print("    Loaded real volume data from Metabase")
    else:
        sample = _generate_sample_volumes()
        forecaster.load(sample)
        print("    Metabase unavailable — using sample volume data")

    # ── Step 3: Forecast TPV per corridor (with detailed breakdown) ────
    print(f"\n  STEP 3: Forecast TPV for {forecast_date.isoformat()} ({forecast_date.strftime('%A')})")

    detailed = forecaster.forecast_detailed(forecast_date)
    forecast_usd = {c: d["forecast_usd"] for c, d in detailed.items()}
    confidence = forecaster.confidence(forecast_date)

    # ── Step 3b: Per-currency breakdown (EMA vs prev week vs forecast) ─
    print(f"\n  Per-Currency Breakdown (floor = never below previous week)")
    print(f"  {'—' * 86}")
    print(f"  {'Corridor':<12} {'Prev Week Date':>16} {'Prev Week USD':>15} {'EMA USD':>14} {'Forecast USD':>14} {'Floor?':>8}")
    print(f"  {'—' * 86}")

    for corridor in ["AED_INR", "GBP_INR", "USD_INR", "EUR_INR"]:
        d = detailed.get(corridor, {})
        ccy = corridor.split("_")[0]
        prev_date = d.get("prev_week_date", "N/A") or "N/A"
        prev_usd = d.get("prev_week_actual_usd", 0)
        ema_usd = d.get("ema_usd", 0)
        fc_usd = d.get("forecast_usd", 0)
        floored = "YES" if d.get("floor_applied") else ""
        print(f"  {ccy + ' → INR':<12} {prev_date:>16} ${prev_usd:>13,.0f} ${ema_usd:>13,.0f} ${fc_usd:>13,.0f} {floored:>8}")

    print(f"  {'—' * 86}")

    # ── Step 4: Apply multipliers ──────────────────────────────────────
    print(f"\n  STEP 4: Apply payday / holiday multipliers")

    calendar = CalendarService()
    multiplier_engine = MultiplierEngine()
    multipliers = multiplier_engine.compute(forecast_date, calendar)

    if multipliers:
        for name, val in multipliers.items():
            print(f"    {name:>12s} : {val:.1f}x")
    else:
        print("    No multipliers active (1.0x)")

    adjusted_usd = {
        corridor: multiplier_engine.apply(vol, multipliers, cap=2.5)
        for corridor, vol in forecast_usd.items()
    }

    # ── Step 5: Final TPV in USD and INR per corridor ─────────────────
    print(f"\n  STEP 5: Total TPV Required")
    print(f"  {'—' * 78}")
    print(f"  {'Corridor':<12} {'Forecast USD':>14} {'Adjusted USD':>14} {'INR Crores':>12} {'INR Lakhs':>12} {'INR':>14}")
    print(f"  {'—' * 78}")

    total_usd = 0.0
    total_inr_crores = 0.0

    for corridor in ["AED_INR", "GBP_INR", "USD_INR", "EUR_INR"]:
        raw = forecast_usd.get(corridor, 0.0)
        adj = adjusted_usd.get(corridor, 0.0)
        inr_crores = round(adj * usd_inr / 1e7, 4)
        inr_lakhs = round(adj * usd_inr / 1e5, 2)
        inr_abs = round(adj * usd_inr, 2)

        total_usd += adj
        total_inr_crores += inr_crores

        ccy = corridor.split("_")[0]
        print(f"  {ccy + ' → INR':<12} ${raw:>13,.0f} ${adj:>13,.0f} {inr_crores:>11.2f} {inr_lakhs:>11,.0f} ₹{inr_abs:>13,.0f}")

    total_inr_lakhs = round(total_usd * usd_inr / 1e5, 2)
    total_inr_abs = round(total_usd * usd_inr, 2)

    print(f"  {'—' * 78}")
    print(f"  {'TOTAL':<12} {'':>14} ${total_usd:>13,.0f} {total_inr_crores:>11.2f} {total_inr_lakhs:>11,.0f} ₹{total_inr_abs:>13,.0f}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  DAILY TPV FORECAST SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Forecast date      : {forecast_date.isoformat()} ({forecast_date.strftime('%A')})")
    print(f"  USD/INR spot rate  : {spot_rate:.4f}")
    print(f"  USD/INR band high  : {usd_inr:.4f}  ← used for TPV (conservative)")
    print(f"  Forecast confidence: {confidence.value.upper()}")
    if multipliers:
        total_mult = multiplier_engine.total(multipliers, cap=2.5)
        print(f"  Multiplier applied : {total_mult:.2f}x ({', '.join(f'{k}={v}' for k,v in multipliers.items())})")

    floors_applied = sum(1 for d in detailed.values() if d.get("floor_applied"))
    if floors_applied:
        print(f"  Floor applied on   : {floors_applied} corridor(s) (forecast raised to >= prev week)")

    print()
    print(f"  Total TPV (USD)    : ${total_usd:>15,.2f}")
    print(f"  Total TPV (INR)    : ₹{total_inr_abs:>15,.2f}")
    print(f"  Total TPV (Crores) : ₹{total_inr_crores:>12.2f} Cr")
    print(f"  Total TPV (Lakhs)  : ₹{total_inr_lakhs:>12,.0f} L")
    if band:
        print()
        print(f"  FX Band Prediction : {band.get('direction', 'N/A')} ({band.get('confidence', 0):.0%} confidence)")
        print(f"  Expected range     : {band.get('range_low', 0):.2f} — {band.get('range_high', 0):.2f}")
    print(f"{'=' * 70}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TMS Orchestrator — Daily TPV Forecast")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of live APIs")
    args = parser.parse_args()
    run(use_mock=args.mock)
