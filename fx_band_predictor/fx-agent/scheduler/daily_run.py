"""
Phase 9 — Daily Scheduler / Entry Point

Runs the full FX prediction pipeline end-to-end:
    1. Fetch fresh market data (last 60 days)
    2. Run FXPredictionAgent.predict()
    3. Save prediction JSON to S3
    4. Log key metrics to CloudWatch
    5. Print formatted summary to stdout
    6. Exit cleanly with code 0 (success) or 1 (failure)

AWS EventBridge cron to run at 8:00 AM IST (2:30 AM UTC) Monday-Friday:
cron(30 2 ? * MON-FRI *)
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

import boto3
import pandas as pd

from data.fetch_market_data import fetch_all
from data.fetch_intraday import fetch_4h_data
from agents.fx_prediction_agent import FXPredictionAgent

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "")
STALE_DATA_HOURS = 48  # refuse to predict on data older than this
RUNTIME_WARNING_SECS = 120

# ---------------------------------------------------------------------------
# Step 1 — Fetch fresh market data
# ---------------------------------------------------------------------------

def fetch_fresh_data() -> pd.DataFrame:
    """
    Fetch ~1.5 years of market data from Yahoo Finance.
    Needs 252+ days for rolling features (rate_percentile_1y, long_term_trend_1y).
    Does NOT overwrite data/market_data.csv — models need full history.
    Returns a fresh dataframe for today's prediction.
    """
    print("  [data] Fetching fresh market data (~1.5 years for feature warm-up)...")
    df = fetch_all(years=1, buffer_days=180)

    if df.empty or "usdinr" not in df.columns:
        raise RuntimeError("Data fetch returned empty or malformed dataframe")

    # Check staleness — latest date should be within STALE_DATA_HOURS
    latest_date = pd.to_datetime(df["date"].iloc[-1])
    now = datetime.now()
    hours_old = (now - latest_date).total_seconds() / 3600

    if hours_old > STALE_DATA_HOURS:
        raise RuntimeError(
            f"Data is stale: latest date is {latest_date.date()} "
            f"({hours_old:.0f}h old, limit is {STALE_DATA_HOURS}h). "
            f"Will not run prediction on stale data."
        )

    print(f"  [data] Got {len(df)} rows, latest: {latest_date.date()} ({hours_old:.0f}h old)")
    return df

# ---------------------------------------------------------------------------
# Step 2 — Run prediction (handled in main pipeline)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 3 — Save to S3
# ---------------------------------------------------------------------------

def save_to_s3(prediction: dict, run_date: datetime) -> str:
    """
    Save prediction JSON to S3.
    - Dated key: predictions/YYYY/MM/DD/prediction.json
    - Latest key: predictions/latest.json (always overwritten)

    Returns the S3 key on success, or empty string if skipped/failed.
    """
    if not S3_BUCKET:
        print("  [s3] WARNING: S3_BUCKET_NAME not set — skipping S3 save")
        return ""

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        body = json.dumps(prediction, indent=2, default=str)

        # Dated key
        date_key = run_date.strftime("predictions/%Y/%m/%d/prediction.json")
        s3.put_object(Bucket=S3_BUCKET, Key=date_key, Body=body,
                      ContentType="application/json",
                      Metadata={"source": "fx-band-predictor-v1"})

        # Latest key
        s3.put_object(Bucket=S3_BUCKET, Key="predictions/latest.json", Body=body,
                      ContentType="application/json",
                      Metadata={"source": "fx-band-predictor-v1"})

        print(f"  [s3] Saved to {date_key}")
        print(f"  [s3] Updated predictions/latest.json")
        return date_key

    except Exception as e:
        print(f"  [s3] WARNING: S3 save failed: {e}")
        return ""

# ---------------------------------------------------------------------------
# Step 4 — CloudWatch metrics
# ---------------------------------------------------------------------------

def log_to_cloudwatch(prediction: dict) -> int:
    """
    Log 4 custom metrics to CloudWatch namespace 'FXAgent'.
    Returns number of metrics logged (0 if failed).
    """
    try:
        cw = boto3.client("cloudwatch", region_name=AWS_REGION)

        pred_48h = prediction.get("prediction_48h", {})
        direction = pred_48h.get("direction", "NEUTRAL")
        confidence = pred_48h.get("confidence", 0.0)
        act = 1.0 if prediction.get("act_on_signal", False) else 0.0

        metrics = [
            {"MetricName": "PredictionConfidence", "Value": float(confidence),
             "Unit": "None"},
            {"MetricName": "ActOnSignal", "Value": act,
             "Unit": "None"},
            {"MetricName": "DirectionUP", "Value": 1.0 if direction == "UP" else 0.0,
             "Unit": "None"},
            {"MetricName": "DirectionDOWN", "Value": 1.0 if direction == "DOWN" else 0.0,
             "Unit": "None"},
        ]

        timestamp = datetime.now(timezone.utc)
        metric_data = []
        for m in metrics:
            metric_data.append({
                "MetricName": m["MetricName"],
                "Timestamp": timestamp,
                "Value": m["Value"],
                "Unit": m["Unit"],
            })

        cw.put_metric_data(Namespace="FXAgent", MetricData=metric_data)
        print(f"  [cloudwatch] {len(metrics)} metrics logged to FXAgent namespace")
        return len(metrics)

    except Exception as e:
        print(f"  [cloudwatch] WARNING: CloudWatch logging failed: {e}")
        return 0

# ---------------------------------------------------------------------------
# Step 5 — Save error to S3
# ---------------------------------------------------------------------------

def save_error_to_s3(error_msg: str, tb: str, run_date: datetime) -> None:
    """Save error details to S3 at predictions/errors/YYYY-MM-DD.json."""
    if not S3_BUCKET:
        return

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        error_key = run_date.strftime("predictions/errors/%Y-%m-%d.json")
        body = json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_msg,
            "traceback": tb,
        }, indent=2)

        s3.put_object(Bucket=S3_BUCKET, Key=error_key, Body=body,
                      ContentType="application/json")
        print(f"  [s3] Error saved to {error_key}")

    except Exception as e2:
        print(f"  [s3] WARNING: Failed to save error to S3: {e2}")

# ---------------------------------------------------------------------------
# Step 6 — Formatted summary
# ---------------------------------------------------------------------------

def print_formatted_summary(prediction: dict, run_date: datetime,
                            s3_key: str, cw_count: int,
                            runtime_secs: float) -> None:
    """Print the production-grade formatted summary to stdout."""
    pred_48h = prediction.get("prediction_48h", {})
    breakdown = prediction.get("model_breakdown", {})
    direction = pred_48h.get("direction", "NEUTRAL")
    confidence = pred_48h.get("confidence", 0.0)
    current_rate = prediction.get("current_rate", 0.0)
    act = prediction.get("act_on_signal", False)

    # Determine action
    if not act:
        action = "HOLD"
    elif direction == "DOWN":
        action = "CONVERT_NOW"
    elif direction == "UP":
        action = "CONVERT_PARTIAL"
    else:
        action = "HOLD"

    xgb = breakdown.get("xgboost", {})
    lstm = breakdown.get("lstm", {})
    sent = breakdown.get("sentiment", {})
    macro = breakdown.get("macro", {})
    vote = prediction.get("vote_outcome", "?")

    date_str = run_date.strftime("%Y-%m-%d %H:%M")

    bar = "\u2550" * 40  # ═

    print()
    print(bar)
    print(f"FX BAND PREDICTOR \u2014 {date_str}")
    print(bar)
    print(f"Status:          SUCCESS")
    print(f"Current Rate:    {current_rate:.2f} USD/INR")
    print(f"Volume context:  $9.5M daily | \u20b9{current_rate * 9_500_000 / 1_00_00_000:.1f} crore")
    print(f"1 paise =        \u20b995,000 saved today")
    print(f"Direction:       {direction}")
    print(f"Confidence:      {confidence * 100:.1f}%")
    print(f"Act on Signal:   {'YES' if act else 'NO'}")
    print(f"Action:          {action}")
    print(f"Vote:            {vote}")
    print()
    print("Model Outputs:")
    regime = xgb.get("regime", "N/A")
    regime_adj = xgb.get("confidence_adjustment", 0.0)
    print(f"  XGBoost:   REGIME={regime:<13s} (adj: {regime_adj:+.2f})")
    range_str = f"{lstm.get('range_low', 'N/A')} — {lstm.get('range_high', 'N/A')}"
    print(f"  LSTM:      RANGE  ({range_str}) [RANGE PROVIDER]")
    score = sent.get("score", 0.0)
    sent_conf = sent.get("confidence", 0.0)
    print(f"  Sentiment: {sent.get('direction', 'N/A'):<8s} (score: {score:+.2f}, conf: {sent_conf:.2f}) [VOTE 1]")
    if macro.get("available"):
        m_score = macro.get("score", 0.0)
        m_conf = macro.get("confidence", 0.0)
        print(f"  Macro:     {macro.get('direction', 'N/A'):<8s} (score: {m_score:+.3f}, conf: {m_conf:.2f}) [VOTE 2]")
        print(f"             {macro.get('interpretation', '')}")
    else:
        print(f"  Macro:     unavailable [VOTE 2]")
    intraday = breakdown.get("intraday_lstm", {})
    i_signal = intraday.get("signal", "N/A")
    i_raw = intraday.get("raw_direction", "N/A")
    i_conf = intraday.get("raw_confidence", 0.0)
    i_adj = intraday.get("confidence_adjustment", 0.0)
    i_bars = intraday.get("bars_analyzed", 0)
    print(f"  Intraday:  {i_signal:<15s} (raw: {i_raw}, conf: {i_conf:.2f}, adj: {i_adj:+.2f}) [MOMENTUM]")
    print(f"             Bars analyzed: {i_bars}")
    print()

    risk_flags = prediction.get("risk_flags", [])
    if risk_flags:
        print("Risk Flags:")
        for f in risk_flags:
            print(f"  \u2192 {f}")
    else:
        print("Risk Flags:      None")

    if act and direction != "NEUTRAL":
        range_low = pred_48h.get("range_low", current_rate)
        range_high = pred_48h.get("range_high", current_rate)
        if direction == "UP":
            paise_gain = (range_high - current_rate) * 100
        else:
            paise_gain = (current_rate - range_low) * 100
        expected_gain_inr = paise_gain * 95_000
        print(f"\nExpected gain:   ~{paise_gain:.1f} paise \u2192 \u20b9{expected_gain_inr:,.0f} if correct")

    print()

    if s3_key:
        print(f"S3:              Saved to {s3_key}")
    else:
        print(f"S3:              Skipped (no bucket configured)")

    print(f"CloudWatch:      {cw_count} metrics logged" if cw_count > 0
          else "CloudWatch:      Skipped")
    print(f"Runtime:         {runtime_secs:.1f} seconds")

    if runtime_secs > RUNTIME_WARNING_SECS:
        print(f"WARNING:         Runtime exceeded {RUNTIME_WARNING_SECS}s — investigate")

    print(bar)

    # --- Opus reasoning ---
    reasoning = prediction.get("reasoning", {})
    if reasoning.get("generated"):
        print()
        print("FX BAND PREDICTOR — OPUS ANALYSIS:")
        print("\u2500" * 60)
        print(reasoning["analysis"])
        print("\u2500" * 60)
        tokens_used = reasoning.get("input_tokens", 0) + reasoning.get("output_tokens", 0)
        print(f"Tokens used: {tokens_used} | Est. cost: ${tokens_used * 0.00005:.4f}")
    elif reasoning.get("skip_reason"):
        print(f"\nOpus analysis: {reasoning['skip_reason']}")

    # --- Confidence breakdown ---
    conf_breakdown = prediction.get("confidence_breakdown", {})
    if conf_breakdown:
        agr = conf_breakdown.get("agreement_bonus_applied", 0)
        n_agr = conf_breakdown.get("signals_in_agreement", 0)
        sent_c = conf_breakdown.get("sentiment_contribution", 0)
        macro_c = conf_breakdown.get("macro_contribution", 0)
        intra_a = conf_breakdown.get("intraday_adjustment", 0)
        regime_a = conf_breakdown.get("regime_adjustment", 0)
        print(f"\nConfidence breakdown:")
        print(f"  Sentiment contribution:  {sent_c:+.3f}")
        print(f"  Macro contribution:      {macro_c:+.3f}")
        print(f"  Regime adjustment:       {regime_a:+.3f}")
        print(f"  Agreement bonus:         {agr:+.3f} ({n_agr} signals agree)")
        print(f"  Intraday adjustment:     {intra_a:+.3f}")
        print(f"  Hard limits:             {conf_breakdown.get('hard_limits', 'N/A')}")

    print()

# ---------------------------------------------------------------------------
# Step 7 — Weekly forecast to stdout
# ---------------------------------------------------------------------------

def print_weekly_forecast(weekly: dict) -> None:
    """Print the weekly forecast in a formatted table."""
    forecasts = weekly.get("daily_forecasts", [])
    summary = weekly.get("week_summary", {})

    bar = "\u2500" * 52  # ─

    print()
    print("FX BAND PREDICTOR — WEEKLY FORECAST")
    print(bar)

    for day in forecasts:
        d = day.get("date", "")
        name = day.get("day_name", "")[:3]
        # Format: Mon Feb 23
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            label = dt.strftime("%a %b %d")
        except Exception:
            label = f"{name} {d}"

        if not day.get("is_trading_day", False):
            print(f"{label}: CLOSED")
            continue

        pred = day.get("prediction", {})
        if pred is None:
            print(f"{label}: CLOSED")
            continue

        direction = pred.get("direction", "NEUTRAL")
        conf = pred.get("confidence", 0.0)
        r_low = pred.get("range_low", 0.0)
        r_high = pred.get("range_high", 0.0)
        print(f"{label}: {direction:<8s} (conf: {conf * 100:>2.0f}%) | {r_low:.2f} — {r_high:.2f}")

    print()
    best = summary.get("best_conversion_day")
    if best:
        print(f"Best conversion day: {best.get('day_name', '')} {best.get('date', '')}")
    else:
        print(f"Best conversion day: None this week")
    print(f"Week bias: {summary.get('overall_bias', 'NEUTRAL')}")
    rec = summary.get("recommendation", "No recommendation available")
    print(f"Recommendation: {rec}")
    print(bar)
    print()


def save_weekly_forecast(weekly: dict, run_date: datetime) -> None:
    """Save weekly forecast to S3 (or locally if no bucket configured)."""
    body = json.dumps(weekly, indent=2, default=str)

    if S3_BUCKET:
        try:
            s3 = boto3.client("s3", region_name=AWS_REGION)
            key = run_date.strftime("predictions/%Y/%m/%d/weekly_forecast.json")
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body,
                          ContentType="application/json")
            print(f"  [s3] Weekly forecast saved to {key}")
        except Exception as e:
            print(f"  [s3] WARNING: Failed to save weekly forecast to S3: {e}")
            # Fall through to local save
    else:
        local_path = os.path.join(PROJECT_DIR, "data", "latest_weekly_forecast.json")
        try:
            with open(local_path, "w") as f:
                f.write(body)
            print(f"  [local] Weekly forecast saved to {local_path}")
        except Exception as e:
            print(f"  [local] WARNING: Failed to save weekly forecast locally: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> int:
    """
    Execute the full daily pipeline. Returns exit code (0=success, 1=failure).
    """
    start_time = time.time()
    run_date = datetime.now()
    prediction = None
    s3_key = ""
    cw_count = 0

    print()
    print("=" * 60)
    print("  FX BAND PREDICTOR — FX Intelligence Engine")
    print("  $9.5M daily | ₹86.3 crore at stake")
    print("  Powered by 5 AI models across 2 time scales")
    print("=" * 60)
    print()

    # --- Step 1: Fetch fresh data ---
    try:
        df = fetch_fresh_data()
    except Exception as e:
        print(f"\n  FATAL: Data fetch failed: {e}")
        tb = traceback.format_exc()
        print(tb)
        save_error_to_s3(str(e), tb, run_date)
        return 1

    # --- Step 1b: Fetch intraday data ---
    df_4h = None
    try:
        print("\n  [data] Fetching intraday 4-hour data...")
        df_4h = fetch_4h_data(years_back=2)
        print(f"  [data] Got {len(df_4h)} intraday bars")
    except Exception as e:
        print(f"  [data] WARNING: Intraday fetch failed: {e} (continuing without)")

    # --- Step 2: Run prediction ---
    try:
        print("\n  [agent] Initializing FX Prediction Agent...")
        agent = FXPredictionAgent()
        prediction = agent.predict(df, df_4h=df_4h)
        print(f"\n  [agent] Prediction complete")
    except Exception as e:
        print(f"\n  FATAL: Prediction failed: {e}")
        tb = traceback.format_exc()
        print(tb)
        save_error_to_s3(str(e), tb, run_date)
        return 1

    # --- Step 3: Run weekly forecast ---
    weekly = None
    try:
        print("\n  [agent] Running weekly forecast...")
        weekly = agent.predict_weekly(df)
        print(f"  [agent] Weekly forecast complete")
    except Exception as e:
        print(f"  [weekly] WARNING: Weekly forecast failed: {e}")
        # Non-fatal — continue

    # --- Step 4: Save to S3 ---
    try:
        s3_key = save_to_s3(prediction, run_date)
    except Exception as e:
        print(f"  [s3] WARNING: Unexpected error: {e}")
        # Non-fatal — continue

    # Save weekly forecast
    if weekly:
        try:
            save_weekly_forecast(weekly, run_date)
        except Exception as e:
            print(f"  [weekly] WARNING: Failed to save weekly forecast: {e}")

    # --- Step 5: CloudWatch metrics ---
    try:
        cw_count = log_to_cloudwatch(prediction)
    except Exception as e:
        print(f"  [cloudwatch] WARNING: Unexpected error: {e}")
        # Non-fatal — continue

    # --- Step 6: Print formatted summary ---
    runtime = time.time() - start_time
    print_formatted_summary(prediction, run_date, s3_key, cw_count, runtime)

    # --- Step 7: Print weekly forecast ---
    if weekly:
        print_weekly_forecast(weekly)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        exit_code = run_pipeline()
    except Exception as e:
        print(f"\n  UNHANDLED EXCEPTION: {e}")
        tb = traceback.format_exc()
        print(tb)
        save_error_to_s3(str(e), tb, datetime.now())
        exit_code = 1

    sys.exit(exit_code)
