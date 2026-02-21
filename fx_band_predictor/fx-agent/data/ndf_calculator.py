"""
NDF-Derived Forward Premium Features

Computes theoretical NDF (Non-Deliverable Forward) features from
existing market data + RBI repo rate history. No paid API needed.

Features produced (10 total):
    1. india_repo_rate          — RBI repo rate on that date
    2. rate_differential        — india_repo_rate - us10y
    3. theoretical_ndf_1m       — interest rate parity forward
    4. forward_premium_paise    — (ndf_1m - spot) × 100
    5. forward_premium_annualized — annualized %
    6. carry_attractiveness     — rate_differential - us10y
    7. ndf_momentum_7d          — 7d change in forward premium
    8. ndf_momentum_30d         — 30d change in forward premium
    9. rate_diff_change_30d     — 30d change in rate differential
   10. carry_regime             — categorical: +1 / 0 / -1
"""

import pandas as pd
import numpy as np
from datetime import date

# ===========================================================================
# RBI Repo Rate History (changes infrequently — hardcoded)
# ===========================================================================

RBI_REPO_RATE_HISTORY = [
    ("2003-01-01", 6.00),
    ("2010-03-19", 5.00),
    ("2011-05-03", 7.25),
    ("2012-04-17", 8.00),
    ("2013-05-03", 7.25),
    ("2014-01-28", 8.00),
    ("2015-01-15", 7.75),
    ("2016-04-05", 6.50),
    ("2017-08-02", 6.00),
    ("2019-02-07", 6.25),
    ("2019-04-04", 6.00),
    ("2019-06-06", 5.75),
    ("2019-08-07", 5.40),
    ("2019-10-04", 5.15),
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2025-02-07", 6.25),
]

# Parse to sorted list of (date, rate)
_REPO_DATES = [(pd.Timestamp(d), r) for d, r in RBI_REPO_RATE_HISTORY]
_REPO_DATES.sort(key=lambda x: x[0])


# ===========================================================================
# Functions
# ===========================================================================

def get_repo_rate(dt) -> float:
    """
    Find the applicable RBI repo rate for a given date.
    Returns the most recent rate change on or before that date.
    """
    dt = pd.Timestamp(dt)
    rate = _REPO_DATES[0][1]  # default to earliest known
    for d, r in _REPO_DATES:
        if d <= dt:
            rate = r
        else:
            break
    return rate


def compute_ndf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 10 NDF-derived features from market data.

    Expects df with columns: date (or DatetimeIndex), usdinr, us10y.
    Returns df with 10 new columns added.
    """
    out = df.copy()

    # Handle date column vs index
    if "date" in out.columns:
        dates = pd.to_datetime(out["date"])
    else:
        dates = pd.to_datetime(out.index)

    # 1. india_repo_rate
    out["india_repo_rate"] = dates.map(get_repo_rate)

    # 2. rate_differential = india_repo_rate - us10y
    out["rate_differential"] = out["india_repo_rate"] - out["us10y"]

    # 3. theoretical_ndf_1m (1-month forward via interest rate parity)
    # F = S × (1 + r_india / 1200) / (1 + r_us / 1200)
    out["theoretical_ndf_1m"] = (
        out["usdinr"]
        * (1 + out["india_repo_rate"] / 1200)
        / (1 + out["us10y"] / 1200)
    )

    # 4. forward_premium_paise = (ndf_1m - spot) × 100
    out["forward_premium_paise"] = (out["theoretical_ndf_1m"] - out["usdinr"]) * 100

    # 5. forward_premium_annualized (%)
    # = (premium_paise × 12) / (usdinr × 100) × 100
    out["forward_premium_annualized"] = (
        out["forward_premium_paise"] * 12 / out["usdinr"]
    )

    # 6. carry_attractiveness = rate_differential - us10y
    out["carry_attractiveness"] = out["rate_differential"] - out["us10y"]

    # 7. ndf_momentum_7d: 7-day change in forward_premium_paise
    out["ndf_momentum_7d"] = out["forward_premium_paise"].diff(7)

    # 8. ndf_momentum_30d: 30-day change in forward_premium_paise
    out["ndf_momentum_30d"] = out["forward_premium_paise"].diff(30)

    # 9. rate_diff_change_30d: 30-day change in rate_differential
    out["rate_diff_change_30d"] = out["rate_differential"].diff(30)

    # 10. carry_regime: categorical signal
    # +1 = carry_positive (diff > 1.5%), 0 = neutral, -1 = under pressure
    out["carry_regime"] = np.where(
        out["rate_differential"] > 1.5, 1,
        np.where(out["rate_differential"] > 0.5, 0, -1)
    )

    return out


# ===========================================================================
# NDF Interpretation (rule-based, deterministic)
# ===========================================================================

def generate_ndf_interpretation(ndf_data: dict) -> str:
    """Generate human-readable NDF interpretation from latest feature values."""
    diff = ndf_data.get("rate_differential", 0)
    premium = ndf_data.get("forward_premium_paise", 0)
    mom_7d = ndf_data.get("ndf_momentum_7d", 0)

    parts = []

    # Carry assessment
    if diff > 1.5:
        parts.append(f"INR carry remains attractive at {diff:.2f}% differential.")
    elif diff > 0.5:
        parts.append(f"Carry advantage narrowing — rate differential at {diff:.2f}%.")
    else:
        parts.append("Carry advantage largely gone — dollar increasingly preferred.")

    # Forward premium assessment
    if premium > 40:
        parts.append("NDF market pricing significant INR weakness ahead.")
    elif premium > 25:
        parts.append("Forward premium at normal levels — no unusual offshore pressure.")
    else:
        parts.append("Low forward premium suggests offshore bullish on INR.")

    # Momentum assessment
    if mom_7d is not None and not np.isnan(mom_7d):
        if mom_7d > 3:
            parts.append("Rising forward premium signals increasing offshore bearishness.")
        elif mom_7d < -3:
            parts.append("Falling forward premium signals offshore buying of INR.")
        else:
            parts.append("NDF momentum stable — no directional shift from offshore market.")

    return " ".join(parts)


# ===========================================================================
# Standalone test
# ===========================================================================

if __name__ == "__main__":
    print("NDF Calculator — Standalone Test")
    print("=" * 70)

    # Load extended market data
    df = pd.read_csv("data/market_data_extended.csv", parse_dates=["date"])
    print(f"Loaded {len(df)} rows, date range: {df['date'].min().date()} to {df['date'].max().date()}")

    result = compute_ndf_features(df)

    ndf_cols = [
        "india_repo_rate", "rate_differential", "theoretical_ndf_1m",
        "forward_premium_paise", "forward_premium_annualized",
        "carry_attractiveness", "ndf_momentum_7d", "ndf_momentum_30d",
        "rate_diff_change_30d", "carry_regime",
    ]

    # --- Summary statistics ---
    print("\nSummary Statistics:")
    print("-" * 70)
    for col in ndf_cols:
        s = result[col].dropna()
        print(f"  {col:<30s}  mean={s.mean():+8.4f}  std={s.std():8.4f}  "
              f"min={s.min():+8.4f}  max={s.max():+8.4f}  nulls={result[col].isna().sum()}")

    # --- Last 10 rows ---
    print("\nLast 10 rows:")
    print("-" * 70)
    show_cols = ["date", "usdinr", "us10y"] + ndf_cols
    print(result[show_cols].tail(10).to_string(index=False))

    # --- Correlation with 2-day future rate change ---
    print("\nCorrelation with 2-day future rate change:")
    print("-" * 70)
    result["rate_change_2d"] = result["usdinr"].shift(-2) - result["usdinr"]
    # Drop NaN rows for clean correlation
    corr_df = result[ndf_cols + ["rate_change_2d"]].dropna()
    print(f"  (computed on {len(corr_df)} rows)\n")
    print(f"  {'Feature':<30s}  {'Correlation':>11s}  {'|Corr|':>7s}  {'Useful?'}")
    print(f"  {'─' * 30}  {'─' * 11}  {'─' * 7}  {'─' * 7}")
    for col in ndf_cols:
        corr = corr_df[col].corr(corr_df["rate_change_2d"])
        useful = "YES" if abs(corr) > 0.05 else "no"
        bar = "*" * int(abs(corr) * 100)
        print(f"  {col:<30s}  {corr:+11.6f}  {abs(corr):7.4f}  {useful:<5s} {bar}")

    # --- Current NDF context ---
    latest = result.iloc[-1]
    print("\nCurrent NDF Context (latest row):")
    print("-" * 70)
    ndf_data = {col: latest[col] for col in ndf_cols}
    for k, v in ndf_data.items():
        print(f"  {k:<30s}  {v:+.4f}" if not np.isnan(v) else f"  {k:<30s}  NaN")

    print(f"\n  Interpretation:")
    print(f"  {generate_ndf_interpretation(ndf_data)}")

    # --- Carry regime distribution ---
    print("\nCarry Regime Distribution:")
    regime_map = {1: "carry_positive", 0: "carry_neutral", -1: "carry_under_pressure"}
    for val, label in regime_map.items():
        count = (result["carry_regime"] == val).sum()
        pct = count / len(result) * 100
        print(f"  {label:<25s}  {count:>5d} days  ({pct:.1f}%)")

    print("\n--- NDF calculator test complete ---")
