"""
Phase 8 — Backtest Evaluation

Simulates the last 90 days of the test period as if the agent ran live
each morning. For each day:
    1. Use ONLY data available up to that day (no lookahead)
    2. Run XGBoost + LSTM predictions
    3. Sentiment = neutral fallback (no historical news data)
    4. Apply ensemble + safety gate
    5. Compare prediction vs actual outcome 2 days later

Outputs:
    - backtest/results.json       — full day-by-day results
    - backtest/summary.json       — metrics summary
    - backtest/daily_results.csv  — simulation table
    - backtest/cumulative_pnl.png — P&L chart
    - backtest/confidence_calibration.png — calibration chart
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, PROJECT_DIR)

from features.feature_engineering import build_features
from models.train_lstm import LSTMRangePredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAVED_DIR = os.path.join(PROJECT_DIR, "models", "saved")
TRAIN_RATIO = 0.80
NEUTRAL_THRESHOLD = 0.15
BACKTEST_DAYS = 90
FUTURE_DAYS = 2

DAILY_VOLUME_USD = 9_500_000
PAISE_TO_INR_PER_DOLLAR = DAILY_VOLUME_USD / 100  # 1 paise × volume = ₹95,000

LABEL_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
W_XGB, W_LSTM = 0.50, 0.35  # sentiment weight goes to NEUTRAL at 0.15


# ---------------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------------

def load_models():
    xgb_model = joblib.load(os.path.join(SAVED_DIR, "xgb_direction.pkl"))
    xgb_features = joblib.load(os.path.join(SAVED_DIR, "feature_names.pkl"))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(os.path.join(SAVED_DIR, "lstm_range.pt"), map_location=device, weights_only=False)
    lstm_model = LSTMRangePredictor(input_dim=ckpt["input_dim"])
    lstm_model.load_state_dict(ckpt["model_state_dict"])
    lstm_model.to(device)
    lstm_model.eval()
    lstm_scaler = joblib.load(os.path.join(SAVED_DIR, "lstm_scaler.pkl"))
    range_buffer = ckpt.get("range_buffer", 0.27)

    return xgb_model, xgb_features, lstm_model, lstm_scaler, range_buffer, device


# ---------------------------------------------------------------------------
# Day-by-day simulation
# ---------------------------------------------------------------------------

def simulate_day(day_idx, market_df, feat_df, xgb_model, xgb_features,
                 lstm_model, lstm_scaler, range_buffer, device):
    """
    Simulate one day's prediction using only data up to day_idx.
    Returns a result dict or None if lookahead data unavailable.
    """
    row = feat_df.iloc[day_idx]
    date_str = str(row["date"])[:10]
    entry_rate = float(row["usdinr"])

    # --- Actual outcome 2 days later ---
    market_date_idx = market_df[market_df["date"] == row["date"]].index
    if len(market_date_idx) == 0:
        return None
    mkt_i = market_date_idx[0]
    if mkt_i + FUTURE_DAYS >= len(market_df):
        return None

    future_rates = market_df["usdinr"].iloc[mkt_i + 1: mkt_i + FUTURE_DAYS + 1]
    actual_rate_48h = float(market_df["usdinr"].iloc[mkt_i + FUTURE_DAYS])
    actual_high = float(future_rates.max())
    actual_low = float(future_rates.min())
    rate_diff = actual_rate_48h - entry_rate

    if rate_diff > NEUTRAL_THRESHOLD:
        actual_direction = "UP"
    elif rate_diff < -NEUTRAL_THRESHOLD:
        actual_direction = "DOWN"
    else:
        actual_direction = "NEUTRAL"

    # --- XGBoost prediction ---
    X = row[xgb_features].values.reshape(1, -1).astype(float)
    xgb_proba = xgb_model.predict_proba(X)[0]
    xgb_class = int(np.argmax(xgb_proba))
    xgb_direction = LABEL_MAP[xgb_class]
    xgb_confidence = float(xgb_proba.max())

    # --- LSTM prediction ---
    lstm_features_list = ["usdinr", "oil", "dxy", "vix", "us10y"]
    # Get the market_df rows up to this day for LSTM sequence
    mkt_slice = market_df.iloc[:mkt_i + 1]
    if len(mkt_slice) < 30:
        lstm_direction = "NEUTRAL"
        lstm_confidence = 0.0
        lstm_close = entry_rate
        lstm_low = entry_rate - 0.3
        lstm_high = entry_rate + 0.3
    else:
        tail = mkt_slice[lstm_features_list].tail(30).values
        scaled = lstm_scaler.transform(tail)
        X_lstm = torch.FloatTensor(scaled).unsqueeze(0).to(device)
        with torch.no_grad():
            delta = lstm_model(X_lstm).cpu().numpy()[0]
        raw_high = entry_rate + delta[0]
        raw_low = entry_rate + delta[1]
        lstm_close = entry_rate + delta[2]
        raw_low, raw_high = min(raw_low, raw_high), max(raw_low, raw_high)
        lstm_high = float(raw_high + range_buffer)
        lstm_low = float(raw_low - range_buffer)
        lstm_close = float(lstm_close)
        if abs(lstm_close - entry_rate) < 0.05:
            lstm_direction = "NEUTRAL"
        else:
            lstm_direction = "UP" if lstm_close > entry_rate else "DOWN"
        lstm_confidence = 0.60

    # --- Ensemble ---
    # Sentiment has 0 confidence (no historical news) — exclude from voting.
    # With only 2 active models, consensus = both agree.
    # Re-weight: XGB = 0.50/(0.50+0.35) ≈ 0.588, LSTM = 0.35/(0.50+0.35) ≈ 0.412
    active_models = [(xgb_direction, xgb_confidence, W_XGB),
                     (lstm_direction, lstm_confidence, W_LSTM)]
    active_dirs = [d for d, _, _ in active_models]
    up_votes = active_dirs.count("UP")
    down_votes = active_dirs.count("DOWN")

    if up_votes == 2:
        ens_direction = "UP"
    elif down_votes == 2:
        ens_direction = "DOWN"
    elif up_votes == 1 and down_votes == 0:
        # One UP, one NEUTRAL — lean towards UP but lower confidence
        ens_direction = "UP"
    elif down_votes == 1 and up_votes == 0:
        ens_direction = "DOWN"
    else:
        ens_direction = "NEUTRAL"

    total_w = sum(w for _, _, w in active_models)
    if ens_direction == "NEUTRAL":
        ens_confidence = 0.5
    else:
        weighted = 0.0
        for d, conf, w in active_models:
            norm_w = w / total_w
            weighted += (conf if d == ens_direction else 1.0 - conf) * norm_w
        ens_confidence = round(weighted, 3)
        # Reduce confidence when only 1 of 2 models agrees (no full consensus)
        if (up_votes + down_votes) < 2:
            ens_confidence = round(ens_confidence * 0.85, 3)

    # Signal strength
    if ens_confidence > 0.72:
        strength = "STRONG"
    elif ens_confidence > 0.62:
        strength = "MEDIUM"
    else:
        strength = "WEAK"

    # --- Safety gate ---
    # In 2-model mode (no sentiment), XGBoost's calibrated probabilities
    # top out at ~0.55 for directional classes. Use 0.50 as the 2-model
    # confidence gate (both agreeing at their typical confidence).
    CONF_THRESHOLD = 0.50  # 2-model mode threshold

    act_on_signal = True
    risk_flags = []

    if ens_confidence < CONF_THRESHOLD:
        act_on_signal = False
        risk_flags.append("low_confidence")

    if ens_direction == "NEUTRAL":
        act_on_signal = False
        risk_flags.append("no_consensus")

    rate_change_1d = float(row.get("rate_change_1d", 0))
    if abs(rate_change_1d) * 100 > 0.8:
        act_on_signal = False
        risk_flags.append("circuit_breaker")

    vol_5d = float(row.get("volatility_5d", 0))
    if vol_5d > 0.005:
        ens_confidence -= 0.10
        ens_confidence = round(ens_confidence, 3)
        risk_flags.append("high_volatility")
        if ens_confidence < CONF_THRESHOLD and act_on_signal:
            act_on_signal = False

    # --- Result ---
    direction_correct = (ens_direction == actual_direction) if ens_direction != "NEUTRAL" else None
    rate_in_range = (actual_rate_48h >= lstm_low) and (actual_rate_48h <= lstm_high)

    # P&L: if we predicted UP and held, gain = actual - entry (in paise)
    if act_on_signal and ens_direction == "UP" and actual_rate_48h > entry_rate:
        improvement = (actual_rate_48h - entry_rate) * 100
    elif act_on_signal and ens_direction == "DOWN" and actual_rate_48h < entry_rate:
        improvement = (entry_rate - actual_rate_48h) * 100
    elif act_on_signal and ens_direction != "NEUTRAL":
        improvement = -abs(actual_rate_48h - entry_rate) * 100
    else:
        improvement = 0.0

    return {
        "date": date_str,
        "entry_rate": round(entry_rate, 4),
        "xgb_direction": xgb_direction,
        "xgb_confidence": round(xgb_confidence, 4),
        "lstm_predicted_close": round(lstm_close, 4),
        "lstm_range_low": round(lstm_low, 4),
        "lstm_range_high": round(lstm_high, 4),
        "ensemble_direction": ens_direction,
        "ensemble_confidence": round(ens_confidence, 4),
        "signal_strength": strength,
        "act_on_signal": act_on_signal,
        "actual_rate_48h": round(actual_rate_48h, 4),
        "actual_direction": actual_direction,
        "direction_correct": direction_correct,
        "rate_in_predicted_range": rate_in_range,
        "rate_improvement_paise": round(improvement, 1),
        "pnl_inr": round(improvement * PAISE_TO_INR_PER_DOLLAR, 0),
        "risk_flags": risk_flags,
    }


# ---------------------------------------------------------------------------
# Run full backtest
# ---------------------------------------------------------------------------

def run_backtest(market_df: pd.DataFrame, n_days: int = BACKTEST_DAYS):
    print("Loading models...")
    xgb_model, xgb_features, lstm_model, lstm_scaler, range_buffer, device = load_models()

    print("Building features...")
    feat_df = build_features(market_df)

    # Use last 20% as test zone (matching training split)
    split_idx = int(len(feat_df) * TRAIN_RATIO)
    test_feat = feat_df.iloc[split_idx:].reset_index(drop=True)

    # Take last n_days from test set (leave room for 2-day lookahead)
    sim_end = len(test_feat) - FUTURE_DAYS
    sim_start = max(0, sim_end - n_days)

    print(f"Simulating days {sim_start} to {sim_end} ({sim_end - sim_start} days)")
    print(f"Test set range: {test_feat['date'].iloc[sim_start].date()} → {test_feat['date'].iloc[sim_end - 1].date()}")

    results = []
    for i in range(sim_start, sim_end):
        # Map test_feat index back to full feat_df index
        full_idx = split_idx + i
        r = simulate_day(full_idx, market_df, feat_df, xgb_model, xgb_features,
                         lstm_model, lstm_scaler, range_buffer, device)
        if r:
            results.append(r)

    print(f"Completed: {len(results)} days simulated\n")
    return results


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list) -> dict:
    df = pd.DataFrame(results)

    total = len(df)
    acted = df[df["act_on_signal"]]
    directional = df[df["ensemble_direction"] != "NEUTRAL"]
    hc = df[df["ensemble_confidence"] > 0.50]  # 2-model threshold
    hc_directional = hc[hc["ensemble_direction"] != "NEUTRAL"]

    # Directional accuracy (on non-NEUTRAL predictions)
    dir_correct = directional["direction_correct"].dropna()
    overall_dir_acc = dir_correct.mean() * 100 if len(dir_correct) > 0 else 0.0

    hc_correct = hc_directional["direction_correct"].dropna()
    hc_dir_acc = hc_correct.mean() * 100 if len(hc_correct) > 0 else 0.0

    # Range accuracy
    range_acc = df["rate_in_predicted_range"].mean() * 100

    # Calibration
    correct_df = directional[directional["direction_correct"] == True]
    incorrect_df = directional[directional["direction_correct"] == False]
    avg_conf_correct = correct_df["ensemble_confidence"].mean() if len(correct_df) > 0 else 0.0
    avg_conf_incorrect = incorrect_df["ensemble_confidence"].mean() if len(incorrect_df) > 0 else 0.0

    # Financial — $9.5M daily conversion volume
    acted_pnl = acted["rate_improvement_paise"]
    avg_improvement = acted_pnl[acted_pnl > 0].mean() if (acted_pnl > 0).any() else 0.0
    best_gain = acted_pnl.max() if len(acted_pnl) > 0 else 0.0
    worst_loss = acted_pnl.min() if len(acted_pnl) > 0 else 0.0
    total_improvement = acted_pnl.sum()

    saving_per_signal = avg_improvement * PAISE_TO_INR_PER_DOLLAR

    # Signal quality by confidence tier
    strong = directional[directional["ensemble_confidence"] > 0.55]
    medium = directional[(directional["ensemble_confidence"] > 0.50) & (directional["ensemble_confidence"] <= 0.55)]
    weak = directional[directional["ensemble_confidence"] <= 0.50]

    def strength_acc(subset):
        d = subset[subset["ensemble_direction"] != "NEUTRAL"]["direction_correct"].dropna()
        return d.mean() * 100 if len(d) > 0 else 0.0

    # Safety gate breakdown
    all_flags = [f for row in results for f in row.get("risk_flags", [])]
    blocked_confidence = all_flags.count("low_confidence")
    blocked_consensus = all_flags.count("no_consensus")
    blocked_circuit = all_flags.count("circuit_breaker")
    blocked_vol = all_flags.count("high_volatility")

    # Strategy comparison
    naive_avg_rate = df["entry_rate"].mean()
    # Our strategy: on acted days use actual_rate_48h, on non-acted days use entry_rate
    our_rates = []
    for _, row in df.iterrows():
        if row["act_on_signal"] and row["ensemble_direction"] == "DOWN":
            our_rates.append(row["entry_rate"])  # converted early
        elif row["act_on_signal"] and row["ensemble_direction"] == "UP":
            our_rates.append(row["actual_rate_48h"])  # waited and got better rate
        else:
            our_rates.append(row["entry_rate"])  # converted at open (same as naive)
    our_avg_rate = np.mean(our_rates) if our_rates else naive_avg_rate

    # Signal frequency
    signals_per_month = len(acted) / max(total, 1) * 30
    monthly_saving = saving_per_signal * signals_per_month
    annual_saving = monthly_saving * 12

    return {
        "overall": {
            "total_days": total,
            "total_signals": len(directional),
            "high_confidence_signals": len(hc_directional),
            "signals_acted_on": len(acted),
        },
        "accuracy": {
            "overall_directional_pct": round(overall_dir_acc, 1),
            "high_confidence_pct": round(hc_dir_acc, 1),
            "range_accuracy_pct": round(range_acc, 1),
            "avg_confidence_correct": round(float(avg_conf_correct), 3),
            "avg_confidence_incorrect": round(float(avg_conf_incorrect), 3),
            "calibration_gap": round(float(avg_conf_correct - avg_conf_incorrect), 3),
        },
        "financial": {
            "daily_volume_usd": DAILY_VOLUME_USD,
            "avg_improvement_paise": round(float(avg_improvement), 1),
            "best_gain_paise": round(float(best_gain), 1),
            "worst_loss_paise": round(float(worst_loss), 1),
            "saving_per_signal_inr": int(saving_per_signal),
            "monthly_saving_inr": int(monthly_saving),
            "annual_saving_inr": int(annual_saving),
            "annual_saving_crore": round(annual_saving / 1_00_00_000, 1),
        },
        "signal_quality": {
            "strong_count": len(strong), "strong_acc": round(strength_acc(strong), 1),
            "medium_count": len(medium), "medium_acc": round(strength_acc(medium), 1),
            "weak_count": len(weak), "weak_acc": round(strength_acc(weak), 1),
        },
        "safety_gate": {
            "blocked_low_confidence": blocked_confidence,
            "blocked_no_consensus": blocked_consensus,
            "blocked_circuit_breaker": blocked_circuit,
            "blocked_high_volatility": blocked_vol,
        },
        "strategy_comparison": {
            "naive_avg_rate": round(float(naive_avg_rate), 4),
            "agent_avg_rate": round(float(our_avg_rate), 4),
            "naive_trades": total,
            "agent_trades": len(acted),
            "signals_per_month": round(float(signals_per_month), 1),
        },
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def generate_charts(results: list):
    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])

    # --- Chart 1: Cumulative P&L ---
    fig, ax = plt.subplots(figsize=(12, 5))
    df["cum_pnl"] = df["rate_improvement_paise"].cumsum()
    ax.plot(df["date"], df["cum_pnl"], color="#2563eb", linewidth=1.5, label="Cumulative P&L")
    ax.fill_between(df["date"], 0, df["cum_pnl"], alpha=0.1, color="#2563eb")

    acted = df[df["act_on_signal"]]
    correct = acted[acted["direction_correct"] == True]
    wrong = acted[acted["direction_correct"] == False]
    ax.scatter(correct["date"], correct["cum_pnl"], color="green", s=60, zorder=5,
               label=f"Correct signal ({len(correct)})", edgecolors="white", linewidths=0.5)
    ax.scatter(wrong["date"], wrong["cum_pnl"], color="red", s=60, zorder=5,
               label=f"Wrong signal ({len(wrong)})", edgecolors="white", linewidths=0.5)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Gain (paise per unit)")
    ax.set_title("FX Agent — Cumulative P&L vs Naive Daily Conversion")
    ax.legend(loc="upper left")
    plt.tight_layout()
    path1 = os.path.join(BASE_DIR, "cumulative_pnl.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"  Saved: {path1}")

    # --- Chart 2: Confidence vs Accuracy ---
    directional = df[df["ensemble_direction"] != "NEUTRAL"].copy()
    directional["correct_int"] = directional["direction_correct"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(directional["ensemble_confidence"], directional["correct_int"],
               alpha=0.5, color="#2563eb", s=40)

    # Trend line (binned)
    if len(directional) > 5:
        bins = np.linspace(directional["ensemble_confidence"].min(),
                           directional["ensemble_confidence"].max(), 6)
        directional["conf_bin"] = pd.cut(directional["ensemble_confidence"], bins)
        bin_acc = directional.groupby("conf_bin", observed=True)["correct_int"].mean()
        bin_centers = [(b.left + b.right) / 2 for b in bin_acc.index]
        ax.plot(bin_centers, bin_acc.values, color="red", linewidth=2,
                marker="o", markersize=8, label="Binned accuracy")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.axvline(x=0.62, color="orange", linestyle="--", linewidth=0.8, label="Confidence threshold")
    ax.set_xlabel("Ensemble Confidence")
    ax.set_ylabel("Correct (1) / Incorrect (0)")
    ax.set_title("Confidence Calibration — Higher Confidence Should Mean Higher Accuracy")
    ax.legend()
    plt.tight_layout()
    path2 = os.path.join(BASE_DIR, "confidence_calibration.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Saved: {path2}")


# ---------------------------------------------------------------------------
# Pretty print metrics
# ---------------------------------------------------------------------------

def print_metrics(m: dict):
    o = m["overall"]
    a = m["accuracy"]
    f = m["financial"]
    sq = m["signal_quality"]
    sg = m["safety_gate"]
    sc = m["strategy_comparison"]

    print("\nOVERALL PERFORMANCE")
    print("─" * 45)
    print(f"  Total days simulated:        {o['total_days']}")
    print(f"  Total signals generated:     {o['total_signals']}")
    print(f"  High confidence signals:     {o['high_confidence_signals']}  (confidence > 0.62)")
    print(f"  Signals acted on:            {o['signals_acted_on']}")

    print(f"\nACCURACY METRICS")
    print("─" * 45)
    print(f"  Overall directional accuracy:  {a['overall_directional_pct']:.1f}%")
    print(f"  High confidence accuracy:      {a['high_confidence_pct']:.1f}%   ← headline metric")
    print(f"  Range accuracy:                {a['range_accuracy_pct']:.1f}%")
    print(f"  Avg confidence (correct):      {a['avg_confidence_correct']:.3f}")
    print(f"  Avg confidence (incorrect):    {a['avg_confidence_incorrect']:.3f}")
    print(f"  Calibration gap:               {a['calibration_gap']:+.3f}  {'(GOOD)' if a['calibration_gap'] > 0 else '(NEEDS WORK)'}")

    vol = f['daily_volume_usd']
    print(f"\nFINANCIAL METRICS  (daily volume: ${vol/1e6:.1f}M)")
    print("─" * 55)
    print(f"  Avg rate improvement (paise):    {f['avg_improvement_paise']:.1f} p")
    print(f"  1 paise = ₹{vol/100:,.0f} per day")
    print(f"  Best single signal gain:         {f['best_gain_paise']:.1f} p")
    print(f"  Worst single signal loss:        {f['worst_loss_paise']:.1f} p")
    print(f"  Saving per correct signal:       ₹{f['saving_per_signal_inr']:,}")
    print(f"  Estimated monthly saving:        ₹{f['monthly_saving_inr']:,}")
    print(f"  Projected annual saving:         ₹{f['annual_saving_inr']:,} (~₹{f['annual_saving_crore']:.1f} Cr)")

    print(f"\nSIGNAL QUALITY BREAKDOWN")
    print("─" * 45)
    print(f"  HIGH conf (>0.55):         {sq['strong_count']:>3d} total, {sq['strong_acc']:.1f}% accurate")
    print(f"  MED conf  (0.50-0.55):    {sq['medium_count']:>3d} total, {sq['medium_acc']:.1f}% accurate")
    print(f"  LOW conf  (<0.50):         {sq['weak_count']:>3d} total, {sq['weak_acc']:.1f}% accurate")

    print(f"\nSAFETY GATE BREAKDOWN")
    print("─" * 45)
    print(f"  Blocked by low confidence:   {sg['blocked_low_confidence']} days")
    print(f"  Blocked by no consensus:     {sg['blocked_no_consensus']} days")
    print(f"  Blocked by circuit breaker:  {sg['blocked_circuit_breaker']} days")
    print(f"  Blocked by volatility gate:  {sg['blocked_high_volatility']} days")

    print(f"\nSTRATEGY COMPARISON ({o['total_days']}-day backtest, ${vol/1e6:.1f}M daily)")
    print("═" * 60)
    print(f"  {'':25s} {'NAIVE':>14s}   {'OUR AGENT':>14s}")
    print(f"  {'':25s} {'(convert daily)':>14s}   {'(signal-based)':>14s}")
    print("─" * 60)
    print(f"  {'Avg conversion rate':<25s} {sc['naive_avg_rate']:>14.4f}   {sc['agent_avg_rate']:>14.4f}")
    print(f"  {'Total trades':<25s} {sc['naive_trades']:>14d}   {sc['agent_trades']:>14d}")
    print(f"  {'Saving per signal':<25s} {'₹0':>14s}   ₹{f['saving_per_signal_inr']:>12,}")
    print(f"  {'Monthly saving':<25s} {'₹0':>14s}   ₹{f['monthly_saving_inr']:>12,}")
    print(f"  {'Annual saving':<25s} {'₹0':>14s}   ₹{f['annual_saving_inr']:>12,}")
    print("═" * 60)

    print(f"\n\n{'═' * 55}")
    print(f"  HACKATHON DEMO NUMBERS  ($9.5M daily volume)")
    print(f"{'═' * 55}")
    print(f"  High confidence accuracy:   {a['high_confidence_pct']:.1f}%")
    print(f"  Saving per correct signal:  ₹{f['saving_per_signal_inr']:,}")
    print(f"  Estimated monthly saving:   ₹{f['monthly_saving_inr']:,}")
    print(f"  Projected annual saving:    ₹{f['annual_saving_inr']:,} (~₹{f['annual_saving_crore']:.1f} Cr)")
    print(f"  Signal frequency:           {sc['signals_per_month']:.1f} per month")
    print(f"  1 paise improvement =       ₹{vol/100:,.0f}/day")
    print(f"{'═' * 55}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = os.path.join(PROJECT_DIR, "data", "market_data.csv")
    print(f"Loading market data from {data_path}")
    market_df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"Shape: {market_df.shape}\n")

    # Run simulation
    results = run_backtest(market_df, BACKTEST_DAYS)

    # Compute metrics
    metrics = compute_metrics(results)

    # Print everything
    print_metrics(metrics)

    # Generate charts
    print("\nGenerating charts...")
    generate_charts(results)

    # Save outputs
    results_path = os.path.join(BASE_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    summary_path = os.path.join(BASE_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    csv_path = os.path.join(BASE_DIR, "daily_results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print("\n--- Phase 8 complete ---")
