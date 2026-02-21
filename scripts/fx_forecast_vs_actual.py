#!/usr/bin/env python
"""
Predicted FX Liquidity vs Actual — Last 7 Days
===============================================

Walk-forward evaluation: for each of the last 7 days, the forecaster is
trained on all data *strictly before* that day (simulating what the model
would have predicted at 06:00 IST that morning), then compared against
the actual completed-order volume recorded in Redshift that day.

Usage:
    python scripts/fx_forecast_vs_actual.py
    python scripts/fx_forecast_vs_actual.py --days 14
    python scripts/fx_forecast_vs_actual.py --csv   (emit raw CSV to stdout)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.metabase import fetch_corridor_volumes
from agents.liquidity.forecaster import VolumeForecaster

CORRIDORS = ["USD_INR", "GBP_INR", "EUR_INR", "AED_INR"]
CORRIDOR_CCY = {"USD_INR": "USD", "GBP_INR": "GBP", "EUR_INR": "EUR", "AED_INR": "AED"}
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_table(days: int = 7) -> list[dict]:
    """
    Fetch history from Metabase and run a walk-forward comparison.
    Returns a list of row-dicts, one per (date × corridor).
    """
    # Need enough history to train the EMA (8 weeks same-DOW lookback)
    lookback_fetch = days + 8 * 7 + 7   # fetch window: evaluation + training buffer
    raw = fetch_corridor_volumes(lookback_days=lookback_fetch)
    if raw is None:
        raise RuntimeError(
            "Cannot reach Metabase — is Cloudflare WARP active?"
        )

    today = date.today()
    eval_dates = [today - timedelta(days=i) for i in range(days, 0, -1)]

    rows = []
    for eval_date in eval_dates:
        # Training set: every record strictly before eval_date
        history_before: dict[str, list[dict]] = {}
        for corridor, records in raw.items():
            history_before[corridor] = [
                r for r in records if r["date"] < eval_date.isoformat()
            ]

        forecaster = VolumeForecaster()
        forecaster.load(history_before)
        forecast = forecaster.forecast(eval_date)
        confidence = forecaster.confidence(eval_date)

        for corridor in CORRIDORS:
            # Actual: find the record for eval_date in the raw data
            actual_records = [
                r for r in raw[corridor] if r["date"] == eval_date.isoformat()
            ]
            actual_usd = actual_records[0]["volume_usd"] if actual_records else None
            pred_usd   = forecast.get(corridor, 0.0)

            if actual_usd is not None:
                delta    = actual_usd - pred_usd
                error_pct = (delta / pred_usd * 100) if pred_usd else None
            else:
                delta     = None
                error_pct = None

            rows.append({
                "date":        eval_date.isoformat(),
                "dow":         DOW_NAMES[eval_date.weekday()],
                "corridor":    corridor,
                "currency":    CORRIDOR_CCY[corridor],
                "forecast_usd":    pred_usd,
                "actual_usd":      actual_usd,
                "delta_usd":       delta,
                "error_pct":       error_pct,
                "confidence":  confidence.value,
            })

    return rows


def fmt_usd(v) -> str:
    if v is None:
        return "  —    "
    m = v / 1_000_000
    return f"{m:>7.2f}M"


def fmt_pct(v) -> str:
    if v is None:
        return "    —  "
    sign = "+" if v > 0 else ""
    return f"{sign}{v:>+6.1f}%"


def print_table(rows: list[dict]) -> None:
    # Group by corridor for presentation
    print()
    print(f"  {'─'*78}")
    print(f"  {'FX LIQUIDITY FORECAST vs ACTUAL — LAST 7 DAYS':^78}")
    print(f"  {'(USD volumes from analytics_orders_master_data)':^78}")
    print(f"  {'─'*78}")
    print()

    # Print per corridor
    for corridor in CORRIDORS:
        ccy = CORRIDOR_CCY[corridor]
        corridor_rows = [r for r in rows if r["corridor"] == corridor]

        print(f"  ┌─ {corridor} ({ccy}) {'─'*50}")
        print(f"  │  {'Date':>10}  {'Day':>3}  {'Forecast':>9}  {'Actual':>9}  "
              f"{'Delta':>9}  {'Error %':>8}  {'Conf':>6}")
        print(f"  │  {'─'*10}  {'─'*3}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*6}")

        for r in corridor_rows:
            marker = ""
            if r["error_pct"] is not None and abs(r["error_pct"]) > 20:
                marker = " ⚠"
            elif r["actual_usd"] is None:
                marker = " ?"
            print(
                f"  │  {r['date']:>10}  {r['dow']:>3}  "
                f"{fmt_usd(r['forecast_usd']):>9}  "
                f"{fmt_usd(r['actual_usd']):>9}  "
                f"{fmt_usd(r['delta_usd']):>9}  "
                f"{fmt_pct(r['error_pct']):>8}  "
                f"{r['confidence']:>6}"
                f"{marker}"
            )

        # Summary line: MAE for this corridor
        valid = [r for r in corridor_rows if r["error_pct"] is not None]
        if valid:
            mae = sum(abs(r["delta_usd"]) for r in valid) / len(valid)
            mape = sum(abs(r["error_pct"]) for r in valid) / len(valid)
            print(f"  │  {'':>10}  {'':>3}  {'':>9}  {'':>9}  "
                  f"MAE: {fmt_usd(mae):>9}  MAPE: {mape:>5.1f}%  {'':>6}")
        print()

    # Overall error summary
    all_valid = [r for r in rows if r["error_pct"] is not None]
    if all_valid:
        overall_mape = sum(abs(r["error_pct"]) for r in all_valid) / len(all_valid)
        overall_mae  = sum(abs(r["delta_usd"])  for r in all_valid) / len(all_valid)
        over = sum(1 for r in all_valid if r["delta_usd"] > 0)
        under = sum(1 for r in all_valid if r["delta_usd"] < 0)
        print(f"  {'─'*78}")
        print(f"  Overall MAPE: {overall_mape:.1f}%   Overall MAE: {fmt_usd(overall_mae).strip()}")
        print(f"  Actual > Forecast (over-delivered): {over}/{len(all_valid)} days  |  "
              f"Under: {under}/{len(all_valid)} days")
        print(f"  ⚠  = error > 20%     ?  = no data for that day")
    print(f"  {'─'*78}")
    print()


def print_csv(rows: list[dict]) -> None:
    import csv, io
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    print(out.getvalue(), end="")


def build_demo_table(days: int = 7) -> list[dict]:
    """
    Generate realistic mock data when Metabase is unreachable.
    Volumes are representative of a mid-size remittance corridor (USD millions/day).
    """
    import random, math
    rng = random.Random(42)  # deterministic

    # Typical mean volumes and noise levels per corridor
    MEANS = {"USD_INR": 9_400_000, "GBP_INR": 4_800_000,
              "EUR_INR": 2_900_000, "AED_INR": 6_200_000}
    NOISE = 0.18   # ±18% actual variance around forecast

    today = date.today()
    eval_dates = [today - timedelta(days=i) for i in range(days, 0, -1)]

    rows = []
    for eval_date in eval_dates:
        dow = eval_date.weekday()
        # Weekend naturally lower
        dow_mult = 0.35 if dow >= 5 else 1.0
        # Simulate a payday bump on 28th
        payday = 1.35 if eval_date.day >= 27 else 1.0

        for corridor in CORRIDORS:
            base        = MEANS[corridor] * dow_mult * payday
            pred_usd    = round(base * rng.uniform(0.97, 1.03))
            actual_usd  = round(pred_usd * (1 + rng.uniform(-NOISE, NOISE)))
            delta       = actual_usd - pred_usd
            error_pct   = delta / pred_usd * 100 if pred_usd else None
            samples     = 8 if dow < 5 else 4
            confidence  = "high" if samples >= 6 else "medium"
            rows.append({
                "date":         eval_date.isoformat(),
                "dow":          DOW_NAMES[dow],
                "corridor":     corridor,
                "currency":     CORRIDOR_CCY[corridor],
                "forecast_usd": float(pred_usd),
                "actual_usd":   float(actual_usd),
                "delta_usd":    float(delta),
                "error_pct":    error_pct,
                "confidence":   confidence,
            })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7,
                        help="Number of recent days to evaluate (default: 7)")
    parser.add_argument("--csv", action="store_true",
                        help="Emit raw CSV instead of formatted table")
    parser.add_argument("--demo", action="store_true",
                        help="Use mock data (no Metabase/WARP required)")
    args = parser.parse_args()

    try:
        rows = build_demo_table(days=args.days) if args.demo else build_table(days=args.days)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}", file=sys.stderr)
        print(f"  Tip: activate Cloudflare WARP, then retry.", file=sys.stderr)
        print(f"  For a preview with mock data run:  python scripts/fx_forecast_vs_actual.py --demo\n",
              file=sys.stderr)
        sys.exit(1)

    if args.csv:
        print_csv(rows)
    else:
        print_table(rows)
