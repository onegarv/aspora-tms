#!/usr/bin/env python3
"""
Baniya Buddhi — Unified Orchestrator

Single entry point that handles the full lifecycle:
  1. Fetch market data (daily + full history + intraday)
  2. Build feature matrices (daily 51 features + intraday 20 features)
  3. Train models if artifacts are missing (XGBoost regime, LSTM range, Intraday LSTM)
  4. Run a one-shot prediction (optional)
  5. Start the FastAPI server on port 8001

Usage:
    python run.py                  # train (if needed) + start API server
    python run.py --train-only     # train all models, skip API server
    python run.py --retrain        # force retrain all models + start API
    python run.py --api-only       # skip training, just start API server
    python run.py --predict        # one-shot prediction, print result, exit
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — all imports resolve relative to fx-agent/
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

SAVED_DIR = os.path.join(PROJECT_DIR, "models", "saved")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# ---------------------------------------------------------------------------
# Required model artifacts — if any are missing, training is triggered
# ---------------------------------------------------------------------------

ARTIFACTS = {
    "xgb_regime": {
        "files": [
            os.path.join(SAVED_DIR, "xgb_regime.pkl"),
            os.path.join(SAVED_DIR, "feature_names_regime.pkl"),
        ],
        "trainer": "train_xgb_regime",
    },
    "lstm_range": {
        "files": [
            os.path.join(SAVED_DIR, "lstm_range.pt"),
            os.path.join(SAVED_DIR, "lstm_scaler.pkl"),
        ],
        "trainer": "train_lstm",
    },
    "intraday_lstm": {
        "files": [
            os.path.join(SAVED_DIR, "intraday_lstm.pt"),
            os.path.join(SAVED_DIR, "intraday_scaler.pkl"),
            os.path.join(SAVED_DIR, "intraday_lstm_meta.pkl"),
        ],
        "trainer": "train_intraday_lstm",
    },
}


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _banner(title: str):
    width = 64
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Step 1: Data fetching
# ---------------------------------------------------------------------------

def fetch_data(force: bool = False):
    """Fetch all required market data CSVs."""
    from data.fetch_market_data import (
        fetch_all, fetch_full_history, save,
        OUTPUT_FILE, FULL_HISTORY_OUTPUT_FILE,
    )
    from data.fetch_intraday import fetch_4h_data

    _banner("STEP 1: Fetching Market Data")

    # 1a. Daily data (3 years) — used by API /predict
    if force or not os.path.exists(OUTPUT_FILE):
        _log("Fetching 3-year daily data (Yahoo Finance)...")
        df_daily = fetch_all(years=3, buffer_days=180)
        save(df_daily, OUTPUT_FILE)
        _log(f"  Saved {len(df_daily)} rows → {OUTPUT_FILE}")
    else:
        _log(f"Daily data exists → {OUTPUT_FILE} (skip, use --retrain to force)")

    # 1b. Full history (2003-present) — used by XGBoost regime training
    if force or not os.path.exists(FULL_HISTORY_OUTPUT_FILE):
        _log("Fetching full history 2003-present (Yahoo Finance + FRED)...")
        df_full = fetch_full_history()
        save(df_full, FULL_HISTORY_OUTPUT_FILE)
        _log(f"  Saved {len(df_full)} rows → {FULL_HISTORY_OUTPUT_FILE}")
    else:
        _log(f"Full history exists → {FULL_HISTORY_OUTPUT_FILE} (skip)")

    # 1c. Intraday 4h bars — used by intraday LSTM training + API /intraday
    intraday_csv = os.path.join(DATA_DIR, "intraday_4h.csv")
    if force or not os.path.exists(intraday_csv):
        _log("Fetching 2-year intraday 1h data, resampling to 4h bars...")
        try:
            df_4h = fetch_4h_data(years_back=2)
            _log(f"  Saved {len(df_4h)} bars → {intraday_csv}")
        except Exception as e:
            _log(f"  WARNING: Intraday fetch failed: {e}")
            _log("  Intraday LSTM training will be skipped if features CSV also missing.")
    else:
        _log(f"Intraday data exists → {intraday_csv} (skip)")


# ---------------------------------------------------------------------------
# Step 2: Feature engineering
# ---------------------------------------------------------------------------

def build_features_step(force: bool = False):
    """Build daily + intraday feature matrices."""
    import pandas as pd
    from features.feature_engineering import build_features
    from data.intraday_features import build_intraday_features

    _banner("STEP 2: Feature Engineering")

    # 2a. Daily features (51 columns)
    feature_csv = os.path.join(DATA_DIR, "feature_matrix.csv")
    daily_csv = os.path.join(DATA_DIR, "market_data.csv")

    if force or not os.path.exists(feature_csv):
        if not os.path.exists(daily_csv):
            _log("ERROR: market_data.csv missing — run data fetch first")
            return
        _log("Building 51 daily features...")
        df = pd.read_csv(daily_csv, parse_dates=["date"])
        features_df = build_features(df)
        features_df.to_csv(feature_csv, index=False)
        _log(f"  Saved {len(features_df)} rows × {len(features_df.columns)} cols → {feature_csv}")
    else:
        _log(f"Feature matrix exists → {feature_csv} (skip)")

    # 2b. Intraday features (20 columns)
    intraday_feat_csv = os.path.join(DATA_DIR, "intraday_features.csv")
    intraday_csv = os.path.join(DATA_DIR, "intraday_4h.csv")

    if force or not os.path.exists(intraday_feat_csv):
        if not os.path.exists(intraday_csv):
            _log("WARNING: intraday_4h.csv missing — skipping intraday features")
            return
        _log("Building 20 intraday features from 4h bars...")
        df_4h = pd.read_csv(intraday_csv, index_col=0, parse_dates=True)
        feat_df, feat_cols = build_intraday_features(df_4h)
        feat_df.to_csv(intraday_feat_csv)
        _log(f"  Saved {len(feat_df)} rows × {len(feat_cols)} features → {intraday_feat_csv}")
    else:
        _log(f"Intraday features exist → {intraday_feat_csv} (skip)")


# ---------------------------------------------------------------------------
# Step 3: Model training
# ---------------------------------------------------------------------------

def check_artifacts() -> dict:
    """Check which model artifacts are present. Returns {name: bool}."""
    status = {}
    for name, info in ARTIFACTS.items():
        status[name] = all(os.path.exists(f) for f in info["files"])
    return status


def train_xgb_regime():
    """Train XGBoost 4-class regime classifier."""
    _banner("TRAINING: XGBoost Regime Classifier")
    from models.train_xgb_regime import main as train_main
    train_main()


def train_lstm():
    """Train LSTM range predictor (v1, 5 features)."""
    _banner("TRAINING: LSTM Range Predictor (v1)")
    import pandas as pd
    from models.train_lstm import train_pipeline

    data_path = os.path.join(DATA_DIR, "market_data.csv")
    if not os.path.exists(data_path):
        _log(f"ERROR: {data_path} not found — cannot train LSTM")
        return

    raw = pd.read_csv(data_path, parse_dates=["date"])
    _log(f"Loaded {len(raw)} rows from market_data.csv")
    train_pipeline(raw, version="v1")


def train_intraday_lstm():
    """Train intraday LSTM momentum classifier."""
    _banner("TRAINING: Intraday LSTM Momentum Classifier")

    intraday_feat_csv = os.path.join(DATA_DIR, "intraday_features.csv")
    if not os.path.exists(intraday_feat_csv):
        _log(f"WARNING: {intraday_feat_csv} not found — skipping intraday LSTM training")
        _log("  This model is optional; predictions will work without it.")
        return

    from models.train_intraday_lstm import train_pipeline
    train_pipeline()


def train_models(force: bool = False):
    """Train any models whose artifacts are missing."""
    _banner("STEP 3: Model Training")

    os.makedirs(SAVED_DIR, exist_ok=True)
    status = check_artifacts()

    _log("Artifact check:")
    for name, present in status.items():
        mark = "OK" if present else "MISSING"
        _log(f"  {name:<20s}: {mark}")
        if not present:
            for f in ARTIFACTS[name]["files"]:
                exists = "exists" if os.path.exists(f) else "MISSING"
                _log(f"    {os.path.basename(f):<40s} {exists}")

    trainers = {
        "xgb_regime": train_xgb_regime,
        "lstm_range": train_lstm,
        "intraday_lstm": train_intraday_lstm,
    }

    any_trained = False
    for name, present in status.items():
        if force or not present:
            _log(f"\nTraining {name}...")
            try:
                trainers[name]()
                any_trained = True
                _log(f"  {name} training complete.")
            except Exception as e:
                _log(f"  ERROR training {name}: {e}")
                traceback.print_exc()
                if name == "intraday_lstm":
                    _log("  (intraday_lstm is optional — continuing)")
                else:
                    raise

    if not any_trained and not force:
        _log("All model artifacts present — no training needed.")

    # Post-training verification
    final_status = check_artifacts()
    _log("\nPost-training artifact check:")
    all_ok = True
    for name, present in final_status.items():
        mark = "OK" if present else "MISSING"
        _log(f"  {name:<20s}: {mark}")
        if not present and name != "intraday_lstm":
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Step 4: One-shot prediction
# ---------------------------------------------------------------------------

def run_prediction():
    """Run a single prediction and print the result."""
    import pandas as pd
    from agents.fx_prediction_agent import FXPredictionAgent
    from data.fetch_market_data import fetch_all
    from data.fetch_intraday import fetch_4h_data

    _banner("ONE-SHOT PREDICTION")

    _log("Fetching fresh market data...")
    df = fetch_all(years=1, buffer_days=180)
    _log(f"  {len(df)} rows, latest: {df['date'].max()}")

    df_4h = None
    try:
        _log("Fetching intraday data...")
        df_4h = fetch_4h_data(years_back=2)
        _log(f"  {len(df_4h)} bars")
    except Exception as e:
        _log(f"  Intraday fetch failed: {e} (continuing without)")

    _log("Loading prediction agent...")
    agent = FXPredictionAgent()

    _log("Running prediction...")
    prediction = agent.predict(df, df_4h=df_4h)

    # Print formatted summary
    pred = prediction["prediction_48h"]
    act = prediction["act_on_signal"]
    rate = prediction["current_rate"]
    breakdown = prediction["model_breakdown"]

    if act and pred["direction"] == "UP":
        action = "HOLD — Rate likely rising, hold USD"
    elif act and pred["direction"] == "DOWN":
        action = "CONVERT_NOW — Rate likely falling, convert USD to INR"
    else:
        action = "CONVERT_PARTIAL — Uncertain, convert operational minimum only"

    print()
    print("=" * 64)
    print(f"  BANIYA BUDDHI — {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    print("=" * 64)
    print(f"  Current Rate:    {rate:.4f} USD/INR")
    print(f"  Direction:       {pred['direction']}")
    print(f"  Confidence:      {pred['confidence']*100:.1f}%")
    print(f"  Range:           {pred['range_low']:.2f} — {pred['range_high']:.2f}")
    print(f"  Act on Signal:   {'YES' if act else 'NO'}")
    print(f"  Action:          {action}")
    print()
    print("  Model Breakdown:")
    xgb = breakdown.get("xgboost", {})
    lstm = breakdown.get("lstm", {})
    sent = breakdown.get("sentiment", {})
    macro = breakdown.get("macro", {})
    intra = breakdown.get("intraday_lstm", {})
    print(f"    XGBoost:   regime={xgb.get('regime', 'N/A'):<16s} (adj: {xgb.get('confidence_adjustment', 0):+.2f})")
    print(f"    LSTM:      range {lstm.get('range_low', 0):.2f} — {lstm.get('range_high', 0):.2f}")
    print(f"    Sentiment: {sent.get('direction', 'N/A'):<8s} (score: {sent.get('score', 0):+.2f}, conf: {sent.get('confidence', 0):.2f})")
    print(f"    Macro:     {macro.get('direction', 'N/A'):<8s} (score: {macro.get('score', 0):+.2f}, conf: {macro.get('confidence', 0):.2f})")
    if intra:
        print(f"    Intraday:  {intra.get('signal', 'N/A'):<16s} (adj: {intra.get('confidence_adjustment', 0):+.2f})")
    print()
    print(f"  Vote Outcome:    {prediction.get('vote_outcome', 'N/A')}")

    risk = prediction.get("risk_flags", [])
    if risk:
        print()
        print("  Risk Flags:")
        for flag in risk:
            print(f"    - {flag}")

    print("=" * 64)

    return prediction


# ---------------------------------------------------------------------------
# Step 5: Start API server
# ---------------------------------------------------------------------------

def start_api():
    """Start the FastAPI server on port 8001."""
    import uvicorn

    _banner("STEP 5: Starting API Server")
    _log("Starting Baniya Buddhi API on http://0.0.0.0:8001")
    _log("Endpoints:")
    _log("  GET /health              — System status")
    _log("  GET /predict             — 48h prediction (cached 15 min)")
    _log("  GET /predict/weekly      — 7-day forecast (cached 60 min)")
    _log("  GET /intraday            — 4h momentum signal")
    _log("  GET /history?days=30     — Historical predictions")
    _log("  GET /backtest?days=90    — Walk-forward backtest results")
    print()

    # Import the FastAPI app from api/main.py
    from api.main import app
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baniya Buddhi — FX Intelligence Engine (Unified Orchestrator)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                  Train if needed, then start API on port 8001
  python run.py --train-only     Train all models, then exit
  python run.py --retrain        Force retrain all models, then start API
  python run.py --api-only       Skip training, just start API
  python run.py --predict        Run one-shot prediction and exit
        """,
    )
    parser.add_argument("--train-only", action="store_true",
                        help="Train models and exit (no API server)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain all models even if artifacts exist")
    parser.add_argument("--api-only", action="store_true",
                        help="Skip all training, just start the API server")
    parser.add_argument("--predict", action="store_true",
                        help="Run a single prediction, print result, and exit")
    args = parser.parse_args()

    start_time = time.time()

    print()
    print("=" * 64)
    print("  BANIYA BUDDHI — Sharp money. Sharper timing.")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 64)

    try:
        if args.api_only:
            start_api()
            return

        if args.predict:
            # Quick prediction — fetch fresh data, predict, exit
            status = check_artifacts()
            missing = [k for k, v in status.items() if not v and k != "intraday_lstm"]
            if missing:
                _log(f"Missing required models: {missing}")
                _log("Run `python run.py --train-only` first, or `python run.py` to auto-train.")
                sys.exit(1)
            run_prediction()
            return

        force = args.retrain

        # Step 1: Fetch data
        fetch_data(force=force)

        # Step 2: Build features
        build_features_step(force=force)

        # Step 3: Train models
        all_ok = train_models(force=force)
        if not all_ok:
            _log("WARNING: Some required models failed to train.")
            _log("The API will start but predictions may be degraded.")

        elapsed = time.time() - start_time
        _log(f"\nSetup complete in {elapsed:.1f}s")

        if args.train_only:
            _banner("TRAINING COMPLETE")
            _log("Models are ready. Start the API with: python run.py --api-only")
            return

        # Step 5: Start API
        start_api()

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        _log(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
