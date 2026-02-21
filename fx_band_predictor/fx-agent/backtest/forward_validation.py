"""
Forward Validation — 1-Month Prediction vs Actuals

Validates the prediction system by:
    1. Taking market data up to each day in the validation window
    2. Running the full FXPredictionAgent (XGBoost + LSTM + neutral sentiment)
    3. Comparing each prediction against the actual rate 2 days later
    4. Producing a scorecard showing accuracy

Validation window: last ~30 calendar days of available data.
Default: sentiment uses neutral fallback (no LLM calls) for speed.

Reports two views:
    - Gated: After the safety gate (what the agent would actually recommend)
    - Pre-gate (Ensemble): Raw XGBoost + LSTM ensemble direction before safety gate
      This is more informative in 2-model mode where the 0.62 threshold gates most signals.

Usage:
    source .venv/bin/activate && python backtest/forward_validation.py
"""

import io
import json
import os
import sys
import contextlib
from collections import Counter

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, PROJECT_DIR)

from agents.fx_prediction_agent import FXPredictionAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VALIDATION_DAYS = 30        # calendar days to validate
FUTURE_DAYS = 2             # compare against rate N days later
NEUTRAL_THRESHOLD = 0.15    # INR movement to classify as UP/DOWN

DATA_PATH = os.path.join(PROJECT_DIR, "data", "market_data.csv")


# ---------------------------------------------------------------------------
# Neutral sentiment override
# ---------------------------------------------------------------------------

def _neutral_sentiment_override():
    """
    Return a neutral sentiment result matching the format of
    FXPredictionAgent._run_sentiment() (not raw NEUTRAL_FALLBACK).
    """
    return {
        "direction": "NEUTRAL",
        "confidence": 0.0,
        "score": 0.0,
        "explanation": "Neutral fallback — no live sentiment for validation",
        "data_quality": "none",
        "high_impact_event_detected": False,
        "event_type": None,
        "event_description": None,
        "bullish_inr_signals": [],
        "bearish_inr_signals": [],
    }


# ---------------------------------------------------------------------------
# Derive pre-gate ensemble direction from model breakdown
# ---------------------------------------------------------------------------

def _derive_ensemble_direction(prediction: dict) -> str:
    """
    Extract the pre-safety-gate ensemble direction from the prediction.
    Direction comes from sentiment + macro votes (XGBoost is regime only,
    LSTM is range only). The vote_outcome field captures this.
    """
    mb = prediction.get("model_breakdown", {})
    # Direction signals come from sentiment and macro
    sent_dir = mb.get("sentiment", {}).get("direction", "NEUTRAL")
    macro_dir = mb.get("macro", {}).get("direction", "NEUTRAL")

    # If both have direction signals, check agreement
    if sent_dir != "NEUTRAL" and macro_dir != "NEUTRAL":
        if sent_dir == macro_dir:
            return sent_dir
        return "NEUTRAL"  # disagree → neutral
    elif sent_dir != "NEUTRAL":
        return sent_dir
    elif macro_dir != "NEUTRAL":
        return macro_dir
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Forward validation loop
# ---------------------------------------------------------------------------

def run_forward_validation(market_df: pd.DataFrame, n_days: int = VALIDATION_DAYS,
                           use_live_sentiment: bool = False):
    """
    Run day-by-day forward validation over the last n_days calendar days.

    For each day in the window:
        1. Slice data up to that day
        2. Run FXPredictionAgent.predict() on the slice
        3. Compare prediction against actual rate 2 days later

    Returns:
        List of result dicts, one per validated day
    """
    print("Initializing FX Prediction Agent...")
    agent = FXPredictionAgent()

    # Override sentiment if not using live
    if not use_live_sentiment:
        agent._run_sentiment = _neutral_sentiment_override
        print("  [config] Sentiment: neutral fallback (no LLM calls)")
    else:
        print("  [config] Sentiment: LIVE (OpenRouter calls enabled)")

    # Determine validation window
    # Need FUTURE_DAYS of lookahead, so last prediction day = len - FUTURE_DAYS - 1
    max_pred_idx = len(market_df) - FUTURE_DAYS - 1
    start_idx = max(0, max_pred_idx - n_days + 1)

    # Need enough history for features (at least 35 rows for rolling windows)
    MIN_HISTORY = 35
    if start_idx < MIN_HISTORY:
        start_idx = MIN_HISTORY
        print(f"  [config] Adjusted start to index {start_idx} (need {MIN_HISTORY} rows for features)")

    start_date = market_df["date"].iloc[start_idx]
    end_date = market_df["date"].iloc[max_pred_idx]
    total_days = max_pred_idx - start_idx + 1

    print(f"\n  Validation window: {start_date.date()} → {end_date.date()} ({total_days} days)")
    print(f"  Comparing against actual rate {FUTURE_DAYS} days later")
    print()

    results = []
    for day_idx in range(start_idx, max_pred_idx + 1):
        date_val = market_df["date"].iloc[day_idx]
        entry_rate = float(market_df["usdinr"].iloc[day_idx])

        # Actual rate FUTURE_DAYS later
        actual_idx = day_idx + FUTURE_DAYS
        actual_rate = float(market_df["usdinr"].iloc[actual_idx])
        rate_diff = actual_rate - entry_rate

        if rate_diff > NEUTRAL_THRESHOLD:
            actual_direction = "UP"
        elif rate_diff < -NEUTRAL_THRESHOLD:
            actual_direction = "DOWN"
        else:
            actual_direction = "NEUTRAL"

        # Slice data up to this day (inclusive)
        df_slice = market_df.iloc[:day_idx + 1].copy()

        # Run agent prediction (suppress verbose output)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                prediction = agent.predict(df_slice)
            except Exception as e:
                print(f"  ERROR on {date_val.date()}: {e}", file=sys.stderr)
                continue

        # Extract prediction details (post safety gate)
        pred_48h = prediction["prediction_48h"]
        pred_direction = pred_48h["direction"]
        pred_confidence = pred_48h["confidence"]
        pred_range_low = pred_48h["range_low"]
        pred_range_high = pred_48h["range_high"]
        pred_most_likely = pred_48h["most_likely"]

        # Extract pre-gate ensemble direction from model breakdown
        model_breakdown = prediction["model_breakdown"]
        ensemble_dir = _derive_ensemble_direction(prediction)
        xgb_regime = model_breakdown["xgboost"]["regime"]
        xgb_conf = model_breakdown["xgboost"]["regime_confidence"]
        macro_dir = model_breakdown.get("macro", {}).get("direction", "NEUTRAL")
        lstm_close = model_breakdown["lstm"].get("most_likely")

        # Evaluate gated prediction
        if pred_direction == "NEUTRAL":
            gated_correct = None
        else:
            gated_correct = (pred_direction == actual_direction)

        # Evaluate pre-gate ensemble prediction
        if ensemble_dir == "NEUTRAL":
            ensemble_correct = None
        else:
            ensemble_correct = (ensemble_dir == actual_direction)

        rate_in_range = (pred_range_low <= actual_rate <= pred_range_high)

        result = {
            "date": date_val.strftime("%Y-%m-%d"),
            "day_of_week": date_val.strftime("%a"),
            "entry_rate": round(entry_rate, 4),
            # Pre-gate model details
            "xgb_regime": xgb_regime,
            "xgb_confidence": round(xgb_conf, 4),
            "macro_direction": macro_dir,
            "lstm_close": round(lstm_close, 4) if lstm_close else None,
            "ensemble_direction": ensemble_dir,
            "ensemble_correct": ensemble_correct,
            # Post-gate (final agent output)
            "gated_direction": pred_direction,
            "gated_confidence": round(pred_confidence, 4),
            "signal_strength": prediction["signal_strength"],
            "act_on_signal": prediction["act_on_signal"],
            "gated_correct": gated_correct,
            # Range
            "range_low": round(pred_range_low, 4),
            "range_high": round(pred_range_high, 4),
            "most_likely": round(pred_most_likely, 4),
            "rate_in_range": rate_in_range,
            # Actuals
            "actual_rate_2d": round(actual_rate, 4),
            "actual_direction": actual_direction,
            "rate_diff": round(rate_diff, 4),
            "risk_flags": prediction.get("risk_flags", []),
        }
        results.append(result)

        # Progress indicator
        n = len(results)
        symbol = "." if n % 5 != 0 else str(n)
        print(symbol, end="", flush=True)

    print(f"\n\nCompleted: {len(results)} days validated\n")
    return results


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def print_scorecard(results: list) -> dict:
    """Print the detailed scorecard table and summary metrics."""
    if not results:
        print("No results to display.")
        return {}

    df = pd.DataFrame(results)

    # === TABLE ===
    print("=" * 120)
    print(f"  FORWARD VALIDATION SCORECARD — {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print("=" * 120)
    print(f"{'Date':<12s}{'Day':>3s} {'Rate':>8s} "
          f"{'Regime':<13s} {'Macro':>5s} {'Ens':>5s} "
          f"{'Actual+2d':>9s} {'ActDir':>7s} {'EnsOK':>6s} {'InRange':>8s}")
    print("-" * 120)

    for _, r in df.iterrows():
        ens_ok = "---" if r["ensemble_correct"] is None else ("YES" if r["ensemble_correct"] else " NO")
        range_str = "YES" if r["rate_in_range"] else " NO"
        regime_short = r["xgb_regime"][:12] if isinstance(r["xgb_regime"], str) else str(r["xgb_regime"])[:12]

        print(f"{r['date']:<12s}{r['day_of_week']:>3s} {r['entry_rate']:>8.4f} "
              f"{regime_short:<13s} {r['macro_direction']:>5s} {r['ensemble_direction']:>5s} "
              f"{r['actual_rate_2d']:>9.4f} {r['actual_direction']:>7s} {ens_ok:>6s} {range_str:>8s}")

    print("-" * 120)

    # === SUMMARY METRICS ===
    total = len(df)

    # --- Pre-gate ensemble metrics ---
    ens_directional = df[df["ensemble_direction"] != "NEUTRAL"]
    ens_neutral = df[df["ensemble_direction"] == "NEUTRAL"]
    ens_correct = ens_directional["ensemble_correct"].dropna()
    ens_acc = ens_correct.mean() * 100 if len(ens_correct) > 0 else 0.0

    # --- Gated metrics ---
    gated_directional = df[df["gated_direction"] != "NEUTRAL"]
    gated_neutral = df[df["gated_direction"] == "NEUTRAL"]
    gated_correct = gated_directional["gated_correct"].dropna()
    gated_acc = gated_correct.mean() * 100 if len(gated_correct) > 0 else 0.0
    acted = df[df["act_on_signal"]]

    # --- Range ---
    range_acc = df["rate_in_range"].mean() * 100

    # --- Macro standalone ---
    macro_directional = df[df["macro_direction"] != "NEUTRAL"]
    macro_correct_list = []
    for _, r in macro_directional.iterrows():
        macro_correct_list.append(r["macro_direction"] == r["actual_direction"])
    macro_acc = (sum(macro_correct_list) / len(macro_correct_list) * 100) if macro_correct_list else 0.0

    # --- Regime breakdown ---
    regime_counts = df["xgb_regime"].value_counts()

    # --- Confidence calibration (on ensemble) ---
    ens_correct_df = ens_directional[ens_directional["ensemble_correct"] == True]
    ens_incorrect_df = ens_directional[ens_directional["ensemble_correct"] == False]
    avg_conf_correct = ens_correct_df["gated_confidence"].mean() if len(ens_correct_df) > 0 else 0.0
    avg_conf_incorrect = ens_incorrect_df["gated_confidence"].mean() if len(ens_incorrect_df) > 0 else 0.0

    print(f"\n{'SUMMARY METRICS':^120s}")
    print("=" * 120)

    print(f"\n  OVERVIEW")
    print(f"  ─────────────────────────────────────")
    print(f"  Total days validated:           {total}")
    print(f"  Ensemble directional signals:   {len(ens_directional)} ({len(ens_neutral)} neutral)")
    print(f"  Safety-gated directional:       {len(gated_directional)} ({len(gated_neutral)} gated to neutral)")
    print(f"  Act-on-signal days:             {len(acted)}")

    print(f"\n  DIRECTIONAL ACCURACY (pre-gate ensemble)")
    print(f"  ─────────────────────────────────────")
    print(f"  Ensemble (Sent+Macro):          {ens_acc:.1f}%  ({int(ens_correct.sum())}/{len(ens_correct)})")
    print(f"  Macro signal alone:             {macro_acc:.1f}%  ({sum(macro_correct_list)}/{len(macro_correct_list)})")

    print(f"\n  REGIME DISTRIBUTION (XGBoost)")
    print(f"  ─────────────────────────────────────")
    for regime, count in regime_counts.items():
        print(f"  {regime:<20s}  {count:>3d} days ({count/total*100:.0f}%)")

    if len(gated_correct) > 0:
        print(f"\n  DIRECTIONAL ACCURACY (post safety gate)")
        print(f"  ─────────────────────────────────────")
        print(f"  Gated predictions:              {gated_acc:.1f}%  ({int(gated_correct.sum())}/{len(gated_correct)})")

    print(f"\n  RANGE ACCURACY (LSTM predicted range)")
    print(f"  ─────────────────────────────────────")
    print(f"  Actual rate within range:       {range_acc:.1f}%  ({int(df['rate_in_range'].sum())}/{total})")

    print(f"\n  CONFIDENCE CALIBRATION")
    print(f"  ─────────────────────────────────────")
    print(f"  Avg confidence (correct):       {avg_conf_correct:.3f}")
    print(f"  Avg confidence (incorrect):     {avg_conf_incorrect:.3f}")
    gap = avg_conf_correct - avg_conf_incorrect
    cal_label = "(GOOD)" if gap > 0 else "(NEEDS WORK)"
    print(f"  Calibration gap:                {gap:+.3f}  {cal_label}")

    # Range error analysis
    if "lstm_close" in df.columns:
        valid = df[df["lstm_close"].notna()]
        if len(valid) > 0:
            position_errors = (valid["lstm_close"] - valid["actual_rate_2d"]).abs() * 100
            print(f"\n  POSITION ERROR (LSTM most_likely vs actual)")
            print(f"  ─────────────────────────────────────")
            print(f"  Mean error:                     {position_errors.mean():.1f}p")
            print(f"  Median error:                   {position_errors.median():.1f}p")
            print(f"  Max error:                      {position_errors.max():.1f}p")

    # Safety gate triggers
    all_flags = [f for row in results for f in row.get("risk_flags", [])]
    if all_flags:
        # Group similar confidence flags
        conf_flags = [f for f in all_flags if f.startswith("Confidence")]
        other_flags = [f for f in all_flags if not f.startswith("Confidence")]

        print(f"\n  SAFETY GATE TRIGGERS")
        print(f"  ─────────────────────────────────────")
        if conf_flags:
            print(f"  Confidence below threshold:     {len(conf_flags)} days")
        for flag, count in Counter(other_flags).most_common():
            print(f"  {flag}: {count} days")

    print("\n" + "=" * 120)

    return {
        "total_days": total,
        "ensemble_directional": len(ens_directional),
        "ensemble_neutral": len(ens_neutral),
        "ensemble_accuracy_pct": round(ens_acc, 1),
        "macro_accuracy_pct": round(macro_acc, 1),
        "gated_directional": len(gated_directional),
        "gated_accuracy_pct": round(gated_acc, 1),
        "range_accuracy_pct": round(range_acc, 1),
        "avg_conf_correct": round(float(avg_conf_correct), 3),
        "avg_conf_incorrect": round(float(avg_conf_incorrect), 3),
        "calibration_gap": round(float(gap), 3),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forward Validation — 1-Month Prediction vs Actuals")
    parser.add_argument("--days", type=int, default=VALIDATION_DAYS,
                        help=f"Number of calendar days to validate (default: {VALIDATION_DAYS})")
    parser.add_argument("--live-sentiment", action="store_true",
                        help="Use live sentiment (Bedrock calls) instead of neutral fallback")
    args = parser.parse_args()

    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run data/fetch_market_data.py first.")
        sys.exit(1)

    print(f"Loading market data from {DATA_PATH}")
    market_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Shape: {market_df.shape}")
    print(f"Date range: {market_df['date'].iloc[0].date()} → {market_df['date'].iloc[-1].date()}\n")

    # Run validation
    results = run_forward_validation(market_df, n_days=args.days,
                                     use_live_sentiment=args.live_sentiment)

    if not results:
        print("ERROR: No results produced. Check data availability.")
        sys.exit(1)

    # Print scorecard
    summary = print_scorecard(results)

    # Save outputs
    results_path = os.path.join(BASE_DIR, "forward_validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    summary_path = os.path.join(BASE_DIR, "forward_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    csv_path = os.path.join(BASE_DIR, "forward_validation_daily.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print("\n--- Forward validation complete ---")
