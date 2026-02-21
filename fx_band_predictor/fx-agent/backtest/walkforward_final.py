"""
Walk-Forward Final Validation — Jan 1 to Feb 20, 2026

Runs the FULL 5-model FXPredictionAgent day-by-day:
  1. XGBoost regime classifier
  2. LSTM range predictor
  3. Sentiment (live Alpha Vantage w/ cache, neutral fallback for old dates)
  4. Macro signal (FRED)
  5. Intraday LSTM momentum

Generates:
  - backtest/walkforward_final.png (demo chart)
  - Summary metrics table (printed)

Usage:
    source .venv/bin/activate && python backtest/walkforward_final.py
"""

import contextlib
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, ".env"), override=True)

from agents.fx_prediction_agent import FXPredictionAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

START_DATE = "2026-01-01"
END_DATE = "2026-02-20"
FUTURE_DAYS = 2              # compare against rate N days later
NEUTRAL_THRESHOLD = 0.15     # INR movement to classify as UP/DOWN

DATA_PATH = os.path.join(PROJECT_DIR, "data", "market_data.csv")
INTRADAY_PATH = os.path.join(PROJECT_DIR, "data", "intraday_4h.csv")
OUTPUT_PNG = os.path.join(BASE_DIR, "walkforward_final.png")


# ---------------------------------------------------------------------------
# Run walk-forward
# ---------------------------------------------------------------------------

def run_walkforward():
    # Load daily data
    print(f"Loading daily data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"  Shape: {df.shape}, range: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    # Load intraday data
    print(f"Loading intraday data from {INTRADAY_PATH}")
    df_4h = pd.read_csv(INTRADAY_PATH)
    df_4h["Datetime"] = pd.to_datetime(df_4h["Datetime"], utc=True)
    print(f"  Shape: {df_4h.shape}, range: {df_4h['Datetime'].iloc[0].date()} → {df_4h['Datetime'].iloc[-1].date()}")

    # Filter to validation window
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    # Find index range in daily data
    mask = (df["date"] >= start) & (df["date"] <= end)
    val_indices = df.index[mask].tolist()

    # We need FUTURE_DAYS of lookahead after each prediction day
    max_available = df.index[-1]
    val_indices = [i for i in val_indices if i + FUTURE_DAYS <= max_available]

    print(f"\n  Validation window: {start.date()} → {end.date()}")
    print(f"  Trading days to validate: {len(val_indices)}")
    print(f"  Comparing against actual rate {FUTURE_DAYS} days later")

    # Initialize agent
    print("\nInitializing FX Prediction Agent (all 5 models)...")
    agent = FXPredictionAgent()
    print()

    results = []
    for count, day_idx in enumerate(val_indices, 1):
        date_val = df["date"].iloc[day_idx]
        entry_rate = float(df["usdinr"].iloc[day_idx])

        # Actual rate FUTURE_DAYS later
        actual_idx = day_idx + FUTURE_DAYS
        actual_rate = float(df["usdinr"].iloc[actual_idx])
        rate_diff = actual_rate - entry_rate
        paise_diff = round(rate_diff * 100)  # in paise

        if rate_diff > NEUTRAL_THRESHOLD:
            actual_direction = "UP"
        elif rate_diff < -NEUTRAL_THRESHOLD:
            actual_direction = "DOWN"
        else:
            actual_direction = "NEUTRAL"

        # Slice daily data up to this day
        df_slice = df.iloc[:day_idx + 1].copy()

        # Slice intraday data up to this day's end
        day_end_utc = pd.Timestamp(date_val).tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59)
        df_4h_slice = df_4h[df_4h["Datetime"] <= day_end_utc].copy()

        # Run full agent prediction (suppress verbose output)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                prediction = agent.predict(df_slice, df_4h=df_4h_slice)
            except Exception as e:
                print(f"  ERROR on {date_val.date()}: {e}", file=sys.stderr)
                continue

        # Extract results
        pred_48h = prediction["prediction_48h"]
        pred_direction = pred_48h["direction"]
        pred_confidence = pred_48h["confidence"]
        pred_range_low = pred_48h["range_low"]
        pred_range_high = pred_48h["range_high"]
        act_on_signal = prediction["act_on_signal"]

        # Model breakdown
        mb = prediction.get("model_breakdown", {})
        sentiment_dir = mb.get("sentiment", {}).get("direction", "N/A")
        sentiment_score = mb.get("sentiment", {}).get("score", 0.0)
        macro_dir = mb.get("macro", {}).get("direction", "N/A")
        regime = mb.get("xgboost", {}).get("regime", "N/A")
        intraday_signal = mb.get("intraday_lstm", {}).get("signal", "N/A")
        intraday_raw = mb.get("intraday_lstm", {}).get("raw_direction", "N/A")

        # Confidence breakdown
        cb = prediction.get("confidence_breakdown", {})
        agreement_bonus = cb.get("agreement_bonus_applied", 0.0)
        sentiment_contrib = cb.get("sentiment_contribution", 0.0)
        macro_contrib = cb.get("macro_contribution", 0.0)

        # Reasoning
        reasoning = prediction.get("reasoning", {})
        reasoning_generated = reasoning.get("generated", False)

        # Evaluate
        rate_in_range = (pred_range_low <= actual_rate <= pred_range_high)

        if act_on_signal and pred_direction != "NEUTRAL":
            correct = (pred_direction == actual_direction)
        else:
            correct = None  # not acted on

        # --- Pre-gate "model lean" ---
        # Derive directional lean from regime + macro + intraday
        # (independent of sentiment, which is unavailable in backtest)
        lean_votes = []
        if regime == "trending_up":
            lean_votes.append("UP")
        elif regime == "trending_down":
            lean_votes.append("DOWN")
        if macro_dir in ("UP", "DOWN"):
            lean_votes.append(macro_dir)
        if intraday_raw in ("UP", "DOWN"):
            lean_votes.append(intraday_raw)

        up_votes = lean_votes.count("UP")
        down_votes = lean_votes.count("DOWN")
        if up_votes > down_votes:
            model_lean = "UP"
        elif down_votes > up_votes:
            model_lean = "DOWN"
        else:
            model_lean = "NEUTRAL"

        if model_lean != "NEUTRAL":
            lean_correct = (model_lean == actual_direction)
        else:
            lean_correct = None

        regime_conf = mb.get("xgboost", {}).get("regime_confidence", 0.0)

        result = {
            "date": date_val.strftime("%Y-%m-%d"),
            "day_of_week": date_val.strftime("%a"),
            "entry_rate": round(entry_rate, 4),
            "pred_direction": pred_direction,
            "pred_confidence": round(pred_confidence, 4),
            "act_on_signal": act_on_signal,
            "correct": correct,
            "range_low": round(pred_range_low, 4),
            "range_high": round(pred_range_high, 4),
            "rate_in_range": rate_in_range,
            "actual_rate_2d": round(actual_rate, 4),
            "actual_direction": actual_direction,
            "rate_diff": round(rate_diff, 4),
            "paise_diff": paise_diff,
            # Model details
            "regime": regime,
            "regime_conf": round(regime_conf, 4),
            "sentiment_dir": sentiment_dir,
            "sentiment_score": round(sentiment_score, 3),
            "macro_dir": macro_dir,
            "intraday_signal": intraday_signal,
            "intraday_raw": intraday_raw,
            "agreement_bonus": round(agreement_bonus, 3),
            "sentiment_contrib": round(sentiment_contrib, 3),
            "macro_contrib": round(macro_contrib, 3),
            "reasoning_generated": reasoning_generated,
            "risk_flags": prediction.get("risk_flags", []),
            # Pre-gate model lean
            "model_lean": model_lean,
            "lean_correct": lean_correct,
        }
        results.append(result)

        # Progress
        lean_tag = ""
        if model_lean != "NEUTRAL":
            lean_ok = "✓" if lean_correct else "✗"
            lean_tag = f" | lean={model_lean} {lean_ok}"
        status = f"conf={pred_confidence:.2f} regime={regime}{lean_tag}"
        if act_on_signal:
            ok = "CORRECT" if correct else "WRONG"
            status = f"ACTED {pred_direction} conf={pred_confidence:.2f} → {ok} ({paise_diff:+d}p)"
        print(f"  [{count:2d}/{len(val_indices)}] {date_val.date()} rate={entry_rate:.4f} → {status}")

    print(f"\nCompleted: {len(results)} days validated\n")
    return results, df


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(results: list, full_df: pd.DataFrame):
    """Generate walkforward_final.png — the demo centerpiece chart."""
    rdf = pd.DataFrame(results)
    rdf["date"] = pd.to_datetime(rdf["date"])

    # Get full rate series for the chart background
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    chart_df = full_df[(full_df["date"] >= start) & (full_df["date"] <= end)].copy()

    fig, ax = plt.subplots(figsize=(18, 9))

    # LSTM predicted range as shaded band (for trading days only)
    trading = rdf[rdf["day_of_week"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])]
    ax.fill_between(trading["date"], trading["range_low"], trading["range_high"],
                     alpha=0.12, color="#3498DB", label="LSTM predicted range", zorder=1)

    # Plot rate line
    ax.plot(chart_df["date"], chart_df["usdinr"], color="#2C3E50", linewidth=1.8,
            label="USD/INR rate", zorder=2, alpha=0.9)

    # Categorize model lean signals
    lean_up = rdf[rdf["model_lean"] == "UP"]
    lean_down = rdf[rdf["model_lean"] == "DOWN"]
    lean_neutral = rdf[rdf["model_lean"] == "NEUTRAL"]

    lean_up_correct = lean_up[lean_up["lean_correct"] == True]
    lean_up_wrong = lean_up[lean_up["lean_correct"] == False]
    lean_down_correct = lean_down[lean_down["lean_correct"] == True]
    lean_down_wrong = lean_down[lean_down["lean_correct"] == False]

    # Acted signals (if any)
    acted_correct = rdf[(rdf["act_on_signal"]) & (rdf["correct"] == True)]
    acted_wrong = rdf[(rdf["act_on_signal"]) & (rdf["correct"] == False)]

    # Plot neutral/no-lean as small grey dots
    ax.scatter(lean_neutral["date"], lean_neutral["entry_rate"],
               color="#BDC3C7", s=25, zorder=3, alpha=0.5,
               label=f"No directional lean ({len(lean_neutral)})")

    # Plot model lean signals — triangles
    ax.scatter(lean_up_correct["date"], lean_up_correct["entry_rate"],
               color="#27AE60", s=80, zorder=4, marker="^", edgecolors="#1E8449", linewidth=0.8,
               label=f"Lean UP, correct ({len(lean_up_correct)})")
    ax.scatter(lean_up_wrong["date"], lean_up_wrong["entry_rate"],
               color="#F5B7B1", s=80, zorder=4, marker="^", edgecolors="#E74C3C", linewidth=0.8,
               label=f"Lean UP, wrong ({len(lean_up_wrong)})")
    ax.scatter(lean_down_correct["date"], lean_down_correct["entry_rate"],
               color="#27AE60", s=80, zorder=4, marker="v", edgecolors="#1E8449", linewidth=0.8,
               label=f"Lean DOWN, correct ({len(lean_down_correct)})")
    ax.scatter(lean_down_wrong["date"], lean_down_wrong["entry_rate"],
               color="#F5B7B1", s=80, zorder=4, marker="v", edgecolors="#E74C3C", linewidth=0.8,
               label=f"Lean DOWN, wrong ({len(lean_down_wrong)})")

    # Large circles for acted signals (overlay on top)
    if len(acted_correct) > 0:
        ax.scatter(acted_correct["date"], acted_correct["entry_rate"],
                   color="#27AE60", s=200, zorder=5, edgecolors="white", linewidth=2,
                   label=f"ACTED & correct ({len(acted_correct)})", marker="o")
    if len(acted_wrong) > 0:
        ax.scatter(acted_wrong["date"], acted_wrong["entry_rate"],
                   color="#E74C3C", s=200, zorder=5, edgecolors="white", linewidth=2,
                   label=f"ACTED & wrong ({len(acted_wrong)})", marker="o")

    # Annotate directional lean signals with text
    lean_directional = rdf[rdf["model_lean"] != "NEUTRAL"].copy()
    annotations_placed = []
    for idx, row in lean_directional.iterrows():
        paise = row["paise_diff"]
        sign = "+" if paise >= 0 else ""
        ok = "✓" if row["lean_correct"] else "✗"
        label = (f"{row['date'].strftime('%b %d')}: {row['model_lean']} "
                 f"({row['regime']}) → {ok} {sign}{paise}p")

        # Stagger annotations to reduce overlap
        pos_idx = len(annotations_placed)
        y_dir = 1 if pos_idx % 2 == 0 else -1
        y_offset = y_dir * (30 + (pos_idx % 3) * 12)

        color = "#1E8449" if row["lean_correct"] else "#C0392B"
        ax.annotate(label,
                     xy=(row["date"], row["entry_rate"]),
                     xytext=(0, y_offset),
                     textcoords="offset points",
                     fontsize=6.5, fontweight="bold", color=color,
                     ha="center",
                     va="bottom" if y_dir > 0 else "top",
                     arrowprops=dict(arrowstyle="-", color="#BDC3C7", lw=0.6),
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                               edgecolor=color, alpha=0.85))
        annotations_placed.append(row["date"])

    # Annotate acted signals with larger boxes
    for _, row in pd.concat([acted_correct, acted_wrong]).iterrows():
        paise = row["paise_diff"]
        sign = "+" if paise >= 0 else ""
        label = (f"ACTED: {row['pred_direction']} "
                 f"conf {row['pred_confidence']:.0%} → "
                 f"{'✓' if row['correct'] else '✗'} {sign}{paise}p")
        color = "#27AE60" if row["correct"] else "#E74C3C"
        ax.annotate(label,
                     xy=(row["date"], row["entry_rate"]),
                     xytext=(0, 45),
                     textcoords="offset points",
                     fontsize=8, fontweight="bold", color=color,
                     ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor=color, alpha=0.95, linewidth=2))

    # Formatting
    ax.set_title("FX Prediction Agent — Walk-Forward Validation (Jan 1 – Feb 20, 2026)\n"
                 "5-Model Ensemble: XGBoost Regime + LSTM Range + Sentiment + Macro + Intraday LSTM",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("USD/INR Rate", fontsize=11)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    plt.xticks(rotation=45, ha="right")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    # Summary stats box
    total = len(rdf)
    n_lean_dir = len(lean_directional)
    lean_correct_n = lean_directional["lean_correct"].sum()
    lean_acc = (lean_correct_n / n_lean_dir * 100) if n_lean_dir > 0 else 0
    range_acc = rdf["rate_in_range"].mean() * 100
    n_acted = len(rdf[rdf["act_on_signal"]])
    block_rate = (total - n_acted) / total * 100 if total > 0 else 0

    stats_text = (
        f"Total days: {total}\n"
        f"Safety gate blocked: {total - n_acted}/{total} ({block_rate:.0f}%)\n"
        f"─────────────────────────\n"
        f"Model lean signals: {n_lean_dir}\n"
        f"  Correct: {int(lean_correct_n)}  Wrong: {n_lean_dir - int(lean_correct_n)}\n"
        f"  Lean accuracy: {lean_acc:.0f}%\n"
        f"─────────────────────────\n"
        f"LSTM range accuracy: {range_acc:.0f}%"
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="#ECF0F1", edgecolor="#2C3E50", alpha=0.92)
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=props, family="monospace")

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def print_summary(results: list):
    """Print the final summary metrics table."""
    if not results:
        print("No results to display.")
        return

    rdf = pd.DataFrame(results)
    total = len(rdf)

    # Acted signals
    acted = rdf[rdf["act_on_signal"]]
    blocked = rdf[~rdf["act_on_signal"]]
    n_acted = len(acted)

    acted_correct = acted[acted["correct"] == True]
    acted_wrong = acted[acted["correct"] == False]
    act_acc = (len(acted_correct) / n_acted * 100) if n_acted > 0 else 0

    # All directional signals (including blocked)
    directional = rdf[rdf["pred_direction"] != "NEUTRAL"]
    dir_eval = []
    for _, r in directional.iterrows():
        dir_eval.append(r["pred_direction"] == r["actual_direction"])
    all_dir_acc = (sum(dir_eval) / len(dir_eval) * 100) if dir_eval else 0

    # Range accuracy
    range_acc = rdf["rate_in_range"].mean() * 100

    # Confidence stats
    avg_conf_correct = acted_correct["pred_confidence"].mean() if len(acted_correct) > 0 else 0
    avg_conf_wrong = acted_wrong["pred_confidence"].mean() if len(acted_wrong) > 0 else 0

    # P&L calculation (paise gained/lost on acted signals)
    if n_acted > 0:
        # For UP signals: profit if rate went up (positive paise)
        # For DOWN signals: profit if rate went down (negative paise = we saved by converting)
        total_paise = 0
        for _, r in acted.iterrows():
            if r["pred_direction"] == "UP" and r["actual_direction"] == "UP":
                total_paise += abs(r["paise_diff"])  # waited, rate went up = good
            elif r["pred_direction"] == "UP" and r["actual_direction"] != "UP":
                total_paise -= abs(r["paise_diff"])  # waited, rate didn't go up = bad
            elif r["pred_direction"] == "DOWN":
                total_paise += r["paise_diff"]  # converted early: benefit if rate dropped
    else:
        total_paise = 0

    # Regime distribution
    regime_counts = Counter(rdf["regime"])

    # Reasoning
    reasoning_count = rdf["reasoning_generated"].sum()

    # Risk flags
    all_flags = [f for row in results for f in row.get("risk_flags", [])]
    conf_flags = len([f for f in all_flags if f.startswith("Confidence")])

    print("=" * 90)
    print(f"  WALK-FORWARD FINAL RESULTS — {rdf['date'].iloc[0]} to {rdf['date'].iloc[-1]}")
    print("=" * 90)

    # Detail table
    print(f"\n  {'Date':<12s}{'Day':>3s} {'Rate':>8s} "
          f"{'Dir':>6s} {'Conf':>5s} {'Act':>4s} "
          f"{'Actual+2d':>9s} {'ActDir':>7s} {'OK':>4s} {'Paise':>6s} {'Range':>6s} "
          f"{'Lean':>6s} {'L-OK':>5s}")
    print("  " + "-" * 100)

    for _, r in rdf.iterrows():
        act_str = "YES" if r["act_on_signal"] else " no"
        if r["correct"] is True:
            ok_str = " ✓"
        elif r["correct"] is False:
            ok_str = " ✗"
        else:
            ok_str = "  —"
        paise_str = f"{r['paise_diff']:+d}p" if r["act_on_signal"] else ""
        range_str = "✓" if r["rate_in_range"] else "✗"
        lean_str = r["model_lean"][:4] if r["model_lean"] != "NEUTRAL" else "  —"
        if r["lean_correct"] is True:
            lean_ok = "  ✓"
        elif r["lean_correct"] is False:
            lean_ok = "  ✗"
        else:
            lean_ok = "  —"

        print(f"  {r['date']:<12s}{r['day_of_week']:>3s} {r['entry_rate']:>8.4f} "
              f"{r['pred_direction']:>6s} {r['pred_confidence']:>5.2f} {act_str:>4s} "
              f"{r['actual_rate_2d']:>9.4f} {r['actual_direction']:>7s} {ok_str:>4s} {paise_str:>6s} {range_str:>6s} "
              f"{lean_str:>6s} {lean_ok:>5s}")

    print("  " + "-" * 86)

    # Summary
    print(f"\n  ACTED SIGNAL ACCURACY")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total trading days:             {total}")
    print(f"  Safety gate blocked:            {len(blocked)} ({len(blocked)/total*100:.0f}%)")
    print(f"  Acted signals:                  {n_acted}")
    print(f"    Correct:                      {len(acted_correct)}")
    print(f"    Wrong:                        {len(acted_wrong)}")
    print(f"    Accuracy:                     {act_acc:.1f}%")

    print(f"\n  ALL DIRECTIONAL SIGNALS (including blocked)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Directional predictions:        {len(directional)}")
    print(f"  Directional accuracy:           {all_dir_acc:.1f}%")

    # Pre-gate model lean analysis
    lean_dir = rdf[rdf["model_lean"] != "NEUTRAL"]
    lean_correct_cnt = lean_dir["lean_correct"].sum()
    lean_total = len(lean_dir)
    lean_acc = (lean_correct_cnt / lean_total * 100) if lean_total > 0 else 0
    lean_neutral_cnt = total - lean_total

    print(f"\n  PRE-GATE MODEL LEAN (regime + macro + intraday)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Lean signals (UP/DOWN):         {lean_total}")
    print(f"  Neutral (no lean):              {lean_neutral_cnt}")
    print(f"    Correct:                      {int(lean_correct_cnt)}")
    print(f"    Wrong:                        {lean_total - int(lean_correct_cnt)}")
    print(f"    Lean accuracy:                {lean_acc:.1f}%")

    # Lean breakdown by direction
    lean_up = lean_dir[lean_dir["model_lean"] == "UP"]
    lean_down = lean_dir[lean_dir["model_lean"] == "DOWN"]
    up_acc = (lean_up["lean_correct"].sum() / len(lean_up) * 100) if len(lean_up) > 0 else 0
    dn_acc = (lean_down["lean_correct"].sum() / len(lean_down) * 100) if len(lean_down) > 0 else 0
    print(f"    UP leans:  {len(lean_up):>3d}  accuracy: {up_acc:.0f}%")
    print(f"    DOWN leans:{len(lean_down):>3d}  accuracy: {dn_acc:.0f}%")

    print(f"\n  RANGE ACCURACY (LSTM)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Actual rate within range:       {range_acc:.1f}% ({int(rdf['rate_in_range'].sum())}/{total})")

    print(f"\n  CONFIDENCE CALIBRATION")
    print(f"  ─────────────────────────────────────────")
    print(f"  Avg confidence (correct):       {avg_conf_correct:.3f}")
    print(f"  Avg confidence (wrong):         {avg_conf_wrong:.3f}")
    gap = avg_conf_correct - avg_conf_wrong
    cal = "(GOOD — higher conf = more accurate)" if gap > 0 else "(NEEDS WORK)"
    print(f"  Calibration gap:                {gap:+.3f}  {cal}")

    print(f"\n  P&L ESTIMATE (acted signals)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Net paise:                      {total_paise:+d}p")
    print(f"  Per $100K USD equivalent:       ₹{total_paise * 10:+,d}")

    print(f"\n  REGIME DISTRIBUTION")
    print(f"  ─────────────────────────────────────────")
    for regime, count in regime_counts.most_common():
        print(f"  {regime:<25s} {count:>3d} days ({count/total*100:.0f}%)")

    print(f"\n  OPUS REASONING")
    print(f"  ─────────────────────────────────────────")
    print(f"  Reasoning generated:            {int(reasoning_count)} of {total} days")

    if conf_flags:
        print(f"\n  SAFETY GATE TRIGGERS")
        print(f"  ─────────────────────────────────────────")
        print(f"  Confidence below threshold:     {conf_flags} days")
        other = [f for f in all_flags if not f.startswith("Confidence")]
        for flag, cnt in Counter(other).most_common():
            print(f"  {flag}: {cnt} days")

    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  WALK-FORWARD FINAL VALIDATION")
    print(f"  {START_DATE} → {END_DATE} | All 5 models active")
    print("=" * 70)
    print()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run data/fetch_market_data.py first.")
        sys.exit(1)
    if not os.path.exists(INTRADAY_PATH):
        print(f"ERROR: {INTRADAY_PATH} not found. Run data/fetch_intraday.py first.")
        sys.exit(1)

    results, full_df = run_walkforward()

    if not results:
        print("ERROR: No results produced.")
        sys.exit(1)

    # Print summary
    print_summary(results)

    # Generate chart
    print("\nGenerating chart...")
    generate_chart(results, full_df)

    # Save results
    results_path = os.path.join(BASE_DIR, "walkforward_final_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")

    csv_path = os.path.join(BASE_DIR, "walkforward_final_daily.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    print("\n--- Walk-forward final validation complete ---")
