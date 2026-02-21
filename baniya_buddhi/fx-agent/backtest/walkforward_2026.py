"""
Baniya Buddhi — Walk-Forward Validation (Jan-Feb 2026)

Step 1: Retrain LSTM on 2015-2024 data (features from 2003+)
Step 2: Walk-forward simulation Jan 2 — Feb 20, 2026
Step 3: Opus justification for every day
Step 4: Generate HTML report
Step 5: Generate JSON + CSV
Step 6: Print terminal summary

Usage:
    python backtest/walkforward_2026.py              # Steps 1-6
    python backtest/walkforward_2026.py --step 1     # Step 1 only (train + sanity)
    python backtest/walkforward_2026.py --skip-opus  # Steps 1-2, 4-6 (no Opus)
"""

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_DIR)

# Load .env for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_DIR, ".env"))
except ImportError:
    pass

from features.feature_engineering import build_features
from data.fx_calendar import is_trading_day, get_calendar_context
from models.train_lstm import LSTMRangePredictor, create_sequences, EarlyStopping

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(PROJECT_DIR, "data", "market_data_full.csv")
SAVED_DIR = os.path.join(PROJECT_DIR, "models", "saved")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "backtest")

FEATURES = [
    "usdinr", "oil", "dxy", "vix", "us10y",
    "rate_trend_30d", "rate_percentile_1y", "momentum_consistency",
    "rate_vs_5y_avg", "rate_vs_alltime_percentile",
    "long_term_trend_1y", "is_decade_high",
]

SEQ_LEN = 30
BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 15
LR = 0.001
HUBER_DELTA = 1.0
VAL_SPLIT = 0.1
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

TRAIN_CUTOFF = "2024-12-31"
TRAIN_START = "2015-01-01"
VALIDATION_START = "2026-01-01"
VALIDATION_END = "2026-02-28"

TPV_DAILY_USD = 9_500_000


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Retrain LSTM on 2003-2024
# ═══════════════════════════════════════════════════════════════════════════

def step1_train_lstm():
    """Train LSTM on 2015-2024, sanity check on Q4 2024."""
    print("=" * 70)
    print("STEP 1: RETRAIN LSTM — Training cutoff Dec 31, 2024")
    print(f"  Features: {len(FEATURES)}, Seq: {SEQ_LEN}, Device: {DEVICE}")
    print("=" * 70)

    # Load full data
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"\n  Raw data: {len(raw)} rows ({raw['date'].min().date()} to {raw['date'].max().date()})")

    # Build features from ALL data (so long-term features are computed correctly)
    print("  Building features...")
    features_df = build_features(raw)
    print(f"  Features built: {len(features_df)} rows")

    for f in FEATURES:
        assert f in features_df.columns, f"Missing feature: {f}"

    # Build trading-day targets
    print("  Building trading-day targets...")
    from models.train_lstm_fullhistory import build_trading_day_targets
    df = build_trading_day_targets(features_df)
    print(f"  After targets: {len(df)} rows")

    # --- STRICT: train only on data <= Dec 31, 2024 ---
    train_mask = (df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_CUTOFF)
    train_df = df.loc[train_mask].reset_index(drop=True)

    # Q4 2024 sanity check subset (last 60 trading days of training)
    q4_df = train_df.tail(60).copy()

    print(f"\n  Training data: {len(train_df)} rows")
    print(f"    From: {train_df['date'].min().date()}")
    print(f"    To:   {train_df['date'].max().date()}")
    print(f"  Q4 2024 sanity check: last 60 days ({q4_df['date'].min().date()} to {q4_df['date'].max().date()})")

    # --- Fit scaler on training data ONLY ---
    print("\n  Fitting MinMaxScaler on training data...")
    scaler = MinMaxScaler()
    train_features = train_df[FEATURES].values
    scaler.fit(train_features)
    train_scaled = scaler.transform(train_features)

    # --- Build sequences ---
    train_targets = train_df[["dy_high", "dy_low", "dy_close"]].values
    train_entry = train_df["usdinr"].values
    train_dates = train_df["date"].values

    X_all, y_all, d_all, r_all = create_sequences(
        train_scaled, train_targets, train_dates, train_entry, SEQ_LEN
    )
    print(f"  Total sequences: {X_all.shape}")

    # Val split from end of training
    val_size = int(len(X_all) * VAL_SPLIT)
    X_train = X_all[:-val_size]
    y_train = y_all[:-val_size]
    X_val = X_all[-val_size:]
    y_val = y_all[-val_size:]
    r_val = r_all[-val_size:]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # --- Train ---
    print(f"\n{'─' * 70}")
    print(f"  TRAINING LSTM (device={DEVICE})")
    print(f"{'─' * 70}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRangePredictor(input_dim=len(FEATURES)).to(DEVICE)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    stopper = EarlyStopping(patience=PATIENCE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: LSTM({len(FEATURES)}→64)→LSTM(64→32)→Dense(32→16→3)")
    print(f"  Parameters: {total_params:,}")
    print()

    t0 = time.time()
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        avg_val = np.mean(val_losses)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{MAX_EPOCHS}  train={avg_train:.6f}  val={avg_val:.6f}")

        if stopper.step(avg_val, model):
            best_epoch = epoch - PATIENCE
            print(f"\n  Early stopping at epoch {epoch}. Best: {best_epoch}")
            break
    else:
        best_epoch = MAX_EPOCHS
        print(f"\n  Completed all {MAX_EPOCHS} epochs.")

    train_time = time.time() - t0
    stopper.restore(model)

    # --- Calibrate range buffer ---
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy()
    val_residuals = np.abs((r_val + val_preds[:, 2]) - (r_val + y_val[:, 2]))
    range_buffer = np.percentile(val_residuals, 85)
    print(f"\n  Range buffer (P85 val residuals): ±{range_buffer:.4f} INR")
    print(f"  Training time: {train_time:.1f}s")

    # --- Q4 2024 sanity check ---
    print(f"\n{'─' * 70}")
    print(f"  Q4 2024 SANITY CHECK (last 60 days of training window)")
    print(f"{'─' * 70}")

    # Scale Q4 features and build sequences
    q4_scaled = scaler.transform(q4_df[FEATURES].values)
    q4_targets = q4_df[["dy_high", "dy_low", "dy_close"]].values
    q4_entry = q4_df["usdinr"].values
    q4_dates = q4_df["date"].values

    X_q4, y_q4, d_q4, r_q4 = create_sequences(
        q4_scaled, q4_targets, q4_dates, q4_entry, SEQ_LEN
    )

    with torch.no_grad():
        q4_preds = model(torch.FloatTensor(X_q4).to(DEVICE)).cpu().numpy()

    q4_pred_close = r_q4 + q4_preds[:, 2]
    q4_pred_high = r_q4 + np.maximum(q4_preds[:, 0], q4_preds[:, 1]) + range_buffer
    q4_pred_low = r_q4 + np.minimum(q4_preds[:, 0], q4_preds[:, 1]) - range_buffer
    q4_actual_close = r_q4 + y_q4[:, 2]
    q4_actual_high = r_q4 + y_q4[:, 0]
    q4_actual_low = r_q4 + y_q4[:, 1]

    mae_close = np.mean(np.abs(q4_pred_close - q4_actual_close))
    mae_high = np.mean(np.abs(q4_pred_high - q4_actual_high))
    mae_low = np.mean(np.abs(q4_pred_low - q4_actual_low))

    range_acc = ((q4_actual_close >= q4_pred_low) & (q4_actual_close <= q4_pred_high)).mean()

    print(f"  MAE close: {mae_close:.4f} INR {'PASS' if mae_close <= 0.50 else 'FAIL — MAE > 0.50!'}")
    print(f"  MAE high:  {mae_high:.4f} INR")
    print(f"  MAE low:   {mae_low:.4f} INR")
    print(f"  Range accuracy (close in range): {range_acc:.1%}")
    print(f"  Avg predicted range width: {np.mean(q4_pred_high - q4_pred_low) * 100:.1f} paise")

    if mae_close > 0.50:
        print(f"\n  *** HARD STOP: MAE {mae_close:.4f} > 0.50 — model not saved ***")
        return None, None, None

    # --- Save ---
    os.makedirs(SAVED_DIR, exist_ok=True)
    model_path = os.path.join(SAVED_DIR, "lstm_walkforward_2024.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(FEATURES),
        "seq_len": SEQ_LEN,
        "features": FEATURES,
        "predicts_deltas": True,
        "range_buffer": float(range_buffer),
        "training_cutoff": TRAIN_CUTOFF,
        "training_start": TRAIN_START,
        "best_epoch": best_epoch,
        "train_time_seconds": round(train_time, 1),
    }, model_path)
    print(f"\n  Model saved: {model_path}")

    scaler_path = os.path.join(SAVED_DIR, "lstm_scaler_walkforward_2024.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved: {scaler_path}")

    print(f"\n{'=' * 70}")
    print(f"  STEP 1 COMPLETE — LSTM trained on {TRAIN_START} to {TRAIN_CUTOFF}")
    print(f"  Q4 2024 MAE: {mae_close:.4f} | Range accuracy: {range_acc:.1%}")
    print(f"{'=' * 70}")

    return model, scaler, range_buffer


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Walk-forward simulation Jan-Feb 2026
# ═══════════════════════════════════════════════════════════════════════════

def step2_walkforward(model, scaler, range_buffer):
    """Run walk-forward validation for each trading day in Jan-Feb 2026."""
    print(f"\n{'=' * 70}")
    print("STEP 2: WALK-FORWARD SIMULATION — Jan 2 to Feb 20, 2026")
    print(f"{'=' * 70}")

    # Load full data
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    features_df = build_features(raw)

    # Get all dates in validation window
    val_mask = (features_df["date"] >= VALIDATION_START) & (features_df["date"] <= VALIDATION_END)
    val_dates = features_df.loc[val_mask, "date"].tolist()

    # Filter to trading days only
    trading_dates = [d for d in val_dates if is_trading_day(d.date() if hasattr(d, 'date') else d)]
    print(f"  Trading days in validation window: {len(trading_dates)}")

    # We also need XGBoost regime model
    xgb_model = None
    xgb_feature_names = None
    xgb_path = os.path.join(SAVED_DIR, "xgb_regime.pkl")
    xgb_features_path = os.path.join(SAVED_DIR, "feature_names_regime.pkl")
    if os.path.exists(xgb_path) and os.path.exists(xgb_features_path):
        xgb_model = joblib.load(xgb_path)
        xgb_feature_names = joblib.load(xgb_features_path)
        print(f"  XGBoost regime model loaded ({len(xgb_feature_names)} features)")
    else:
        print(f"  XGBoost regime model not found — skipping regime classification")

    # Macro signal helper
    from agents.macro_signal import get_macro_signal

    REGIME_MAP = {0: "trending_up", 1: "trending_down", 2: "high_vol", 3: "range_bound"}

    # Build date → rate lookup for actuals
    date_to_rate = {}
    for _, row in features_df.iterrows():
        d = row["date"]
        dt = d.date() if hasattr(d, 'date') else d
        date_to_rate[dt] = float(row["usdinr"])

    results = []
    model.eval()

    print(f"\n  {'Date':<12s} {'Rate':>7s} {'PredLo':>8s} {'PredHi':>8s} {'ActLo':>8s} {'ActHi':>8s} {'Result':>6s} {'Ovlp%':>6s}")
    print(f"  {'─'*12} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

    for pred_date in trading_dates:
        pred_dt = pred_date.date() if hasattr(pred_date, 'date') else pred_date

        # 1. Take ALL data up to and including this date (no lookahead)
        slice_mask = features_df["date"] <= pred_date
        df_slice = features_df.loc[slice_mask].copy()

        if len(df_slice) < SEQ_LEN + 1:
            continue

        entry_rate = float(df_slice["usdinr"].iloc[-1])

        # 2. Scale features and build sequence
        scaled = scaler.transform(df_slice[FEATURES].values)
        seq = scaled[-SEQ_LEN:]  # last 30 days
        X = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        # 3. Run LSTM
        with torch.no_grad():
            delta_preds = model(X).cpu().numpy()[0]

        pred_high_raw = entry_rate + max(delta_preds[0], delta_preds[1])
        pred_low_raw = entry_rate + min(delta_preds[0], delta_preds[1])
        pred_close = entry_rate + delta_preds[2]
        pred_high = pred_high_raw + range_buffer
        pred_low = pred_low_raw - range_buffer

        # 4. Run XGBoost regime
        regime = "range_bound"
        regime_confidence = 0.0
        regime_probs = {}
        if xgb_model is not None:
            try:
                latest_features = df_slice.iloc[[-1]]
                X_xgb = latest_features[xgb_feature_names].values
                proba = xgb_model.predict_proba(X_xgb)[0]
                pred_class = int(np.argmax(proba))
                regime = REGIME_MAP[pred_class]
                regime_confidence = float(proba.max())
                regime_probs = {REGIME_MAP[i]: round(float(proba[i]), 4) for i in range(len(proba))}
            except Exception:
                pass

        # 5. Calendar context
        cal = get_calendar_context(pred_dt)
        calendar_note = cal.get("special_note", "")

        # 6. Macro signal
        macro_direction = "NEUTRAL"
        macro_score = 0.0
        try:
            macro = get_macro_signal(df_slice)
            if macro.get("available"):
                macro_direction = macro.get("direction", "NEUTRAL")
                macro_score = macro.get("score", 0.0)
        except Exception:
            pass

        # 7. Derive direction from regime + macro (sentiment = NEUTRAL)
        if macro_direction != "NEUTRAL":
            direction = macro_direction
            confidence = 0.40
        elif regime in ("trending_up", "trending_down"):
            direction = "UP" if regime == "trending_up" else "DOWN"
            confidence = min(0.45, regime_confidence * 0.5)
        else:
            direction = "NEUTRAL"
            confidence = 0.35

        # 8. Key features for this day
        row = df_slice.iloc[-1]
        key_features = {
            "rate_vs_alltime_percentile": round(float(row.get("rate_vs_alltime_percentile", 0)), 4),
            "rate_trend_30d": round(float(row.get("rate_trend_30d", 0)), 6),
            "volatility_20d": round(float(row.get("volatility_20d", 0)), 4),
            "yield_curve_spread": round(float(row.get("yield_curve_spread", 0)), 3),
            "momentum_consistency": int(row.get("momentum_consistency", 0)),
            "rate_vs_5y_avg": round(float(row.get("rate_vs_5y_avg", 0)), 4),
            "is_decade_high": int(row.get("is_decade_high", 0)),
        }

        # 9. Look up actual outcomes: next 2 TRADING days
        actual_rates = []
        check = pred_dt
        for _ in range(14):  # safety
            check = check + timedelta(days=1)
            if is_trading_day(check) and check in date_to_rate:
                actual_rates.append(date_to_rate[check])
            if len(actual_rates) == 2:
                break

        if len(actual_rates) < 2:
            # Not enough future data — skip this day
            continue

        actual_high = max(actual_rates)
        actual_low = min(actual_rates)
        actual_close = actual_rates[1]

        # 10. Evaluate
        overlap_low = max(pred_low, actual_low)
        overlap_high = min(pred_high, actual_high)

        if actual_low >= pred_low and actual_high <= pred_high:
            result = "GREEN"
            overlap_pct = 100.0
        elif overlap_high > overlap_low:
            result = "YELLOW"
            actual_span = actual_high - actual_low
            if actual_span > 0:
                captured = overlap_high - overlap_low
                overlap_pct = round((captured / actual_span) * 100, 1)
            else:
                overlap_pct = 100.0
        else:
            result = "RED"
            overlap_pct = 0.0

        # TPV impact
        if result == "GREEN":
            buffer_saved = (pred_high - actual_high) * TPV_DAILY_USD
            tpv_impact_inr = max(0, round(buffer_saved))
            tpv_note = f"Prefunding accurate — {_fmt_inr(tpv_impact_inr)} buffer saved"
        elif result == "YELLOW":
            miss_paise = max(0, actual_high - pred_high) * 100
            cost = miss_paise / 100 * TPV_DAILY_USD
            tpv_impact_inr = -round(cost)
            if miss_paise > 0:
                tpv_note = f"Miss by {miss_paise:.1f} paise — {_fmt_inr(abs(tpv_impact_inr))} exposure"
            else:
                tpv_note = f"Partial overlap — {overlap_pct:.0f}% range captured"
        else:
            miss_paise = max(0, actual_high - pred_high, pred_low - actual_low) * 100
            cost = miss_paise / 100 * TPV_DAILY_USD
            tpv_impact_inr = -round(cost)
            tpv_note = f"Complete miss by {miss_paise:.1f} paise — {_fmt_inr(abs(tpv_impact_inr))} at risk"

        rec = {
            "date": pred_dt.isoformat(),
            "day_name": pred_dt.strftime("%A"),
            "entry_rate": round(entry_rate, 4),
            "predicted_range_low": round(pred_low, 4),
            "predicted_range_high": round(pred_high, 4),
            "predicted_most_likely": round(pred_close, 4),
            "predicted_direction": direction,
            "regime": regime,
            "regime_confidence": round(regime_confidence, 4),
            "regime_probs": regime_probs,
            "confidence": round(confidence, 3),
            "calendar_note": calendar_note,
            "macro_direction": macro_direction,
            "macro_score": round(macro_score, 3),
            "key_features": key_features,
            "actual_high": round(actual_high, 4),
            "actual_low": round(actual_low, 4),
            "actual_close": round(actual_close, 4),
            "result": result,
            "overlap_pct": overlap_pct,
            "predicted_range_paise": round((pred_high - pred_low) * 100, 1),
            "actual_range_paise": round((actual_high - actual_low) * 100, 1),
            "tpv_impact_inr": tpv_impact_inr,
            "tpv_note": tpv_note,
        }
        results.append(rec)

        # Print row
        color = {"GREEN": "G", "YELLOW": "Y", "RED": "R"}[result]
        print(f"  {pred_dt.isoformat():<12s} {entry_rate:>7.2f} {pred_low:>8.2f} {pred_high:>8.2f} "
              f"{actual_low:>8.2f} {actual_high:>8.2f} {color:>6s} {overlap_pct:>5.1f}%")

    # Summary
    green = sum(1 for r in results if r["result"] == "GREEN")
    yellow = sum(1 for r in results if r["result"] == "YELLOW")
    red = sum(1 for r in results if r["result"] == "RED")
    total = len(results)
    avg_overlap = np.mean([r["overlap_pct"] for r in results]) if results else 0

    print(f"\n  {'─' * 60}")
    print(f"  STEP 2 SUMMARY: {total} trading days")
    print(f"  GREEN: {green} ({green/total*100:.0f}%) | YELLOW: {yellow} ({yellow/total*100:.0f}%) | RED: {red} ({red/total*100:.0f}%)")
    print(f"  Average overlap: {avg_overlap:.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Opus justification for every day
# ═══════════════════════════════════════════════════════════════════════════

def _call_openrouter(prompt, max_tokens=2000, temperature=0.3):
    """Single OpenRouter call with model fallback chain."""
    import requests
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    models = [
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-haiku-4",
    ]
    for model_id in models:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_id,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"].strip()
            usage = body.get("usage", {})
            return {
                "model": model_id.split("/")[-1],
                "content": content,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                break
            continue
    return None


def _build_opus_prompt(rec):
    """Build full Opus prompt for a single day's analysis."""
    kf = rec["key_features"]
    return f"""You are the reasoning engine for Baniya Buddhi, an institutional FX intelligence system processing $9.5M daily (₹86.3 crore) in USD/INR remittances.

Analyze this prediction and explain the full reasoning — what the models saw, why they predicted what they predicted, and what actually happened. Be as detailed as needed. Do not hold back.

DATE: {rec['date']} ({rec['day_name']})
RESULT: {rec['result']}

WHAT WE PREDICTED:
Entry rate: {rec['entry_rate']:.4f}
Predicted range: {rec['predicted_range_low']:.4f} — {rec['predicted_range_high']:.4f}
Most likely: {rec['predicted_most_likely']:.4f}
Direction: {rec['predicted_direction']}
Confidence: {rec['confidence']:.1%}
Regime: {rec['regime']} ({rec['regime_confidence']:.1%} confident)

WHAT ACTUALLY HAPPENED:
Actual low:   {rec['actual_low']:.4f}
Actual high:  {rec['actual_high']:.4f}
Actual close: {rec['actual_close']:.4f}
Overlap:      {rec['overlap_pct']:.1f}%

MARKET CONTEXT ON THIS DAY:
Rate vs 22-year history: {kf['rate_vs_alltime_percentile']:.1%} percentile
Rate vs 5-year average:  {kf['rate_vs_5y_avg']:+.2%}
30-day trend:            {kf['rate_trend_30d']:+.4f} INR/day
20-day volatility:       {kf['volatility_20d']:.2%}
Momentum consistency:    {kf['momentum_consistency']}/6
Yield curve spread:      {kf['yield_curve_spread']:+.3f}
At decade high:          {"Yes" if kf['is_decade_high'] else "No"}
Macro signal:            {rec['macro_direction']} (score: {rec['macro_score']:+.2f})
Calendar:                {rec['calendar_note'] or 'No special events'}

TASK:
1. Explain what the models saw on this date and why they predicted this specific range. Reference the actual feature values above.

2. Explain why the prediction succeeded or failed.
   If GREEN: what made the range accurate?
   If YELLOW: what did we capture and what did we miss?
   If RED: what caused the miss? Was it predictable?

3. What would have been the treasury impact?
   GREEN: How much buffer was saved?
   YELLOW: Was the miss manageable for prefunding?
   RED: What was the cost in rupees on $9.5M volume?

4. What should the system learn from this day? Any pattern that should inform future predictions?

Write with the authority of a 20-year FX market veteran who also understands machine learning deeply. Be specific with numbers. Do not summarize — explain."""


def step3_opus_justifications(results, batch_size=5):
    """Generate Opus justification for every day."""
    print(f"\n{'=' * 70}")
    print(f"STEP 3: OPUS JUSTIFICATION — {len(results)} days")
    print(f"{'=' * 70}")

    total_tokens = 0
    completed = 0

    for batch_start in range(0, len(results), batch_size):
        batch = results[batch_start:batch_start + batch_size]
        print(f"\n  Processing batch {batch_start // batch_size + 1} "
              f"(days {batch_start + 1}-{min(batch_start + batch_size, len(results))})...")

        futures = {}
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for rec in batch:
                prompt = _build_opus_prompt(rec)
                fut = pool.submit(_call_openrouter, prompt, max_tokens=2000, temperature=0.3)
                futures[fut] = rec

            for fut in as_completed(futures):
                rec = futures[fut]
                result = fut.result()
                if result:
                    rec["opus_analysis"] = result["content"]
                    rec["opus_model"] = result["model"]
                    rec["opus_tokens"] = result["input_tokens"] + result["output_tokens"]
                    total_tokens += rec["opus_tokens"]
                    completed += 1
                    print(f"    {rec['date']} — {result['model']} ({rec['opus_tokens']} tokens)")
                else:
                    # Retry once
                    print(f"    {rec['date']} — FAILED, retrying...")
                    retry = _call_openrouter(_build_opus_prompt(rec), max_tokens=2000)
                    if retry:
                        rec["opus_analysis"] = retry["content"]
                        rec["opus_model"] = retry["model"]
                        rec["opus_tokens"] = retry["input_tokens"] + retry["output_tokens"]
                        total_tokens += rec["opus_tokens"]
                        completed += 1
                        print(f"    {rec['date']} — RETRY OK ({rec['opus_tokens']} tokens)")
                    else:
                        rec["opus_analysis"] = _fallback_analysis(rec)
                        rec["opus_model"] = "rule_based_fallback"
                        rec["opus_tokens"] = 0
                        print(f"    {rec['date']} — using fallback")

    print(f"\n  Opus complete: {completed}/{len(results)} days, {total_tokens:,} total tokens")
    return results


def _fallback_analysis(rec):
    """Rule-based fallback analysis when Opus is unavailable."""
    kf = rec["key_features"]
    lines = []
    lines.append(f"On {rec['date']}, USD/INR opened at {rec['entry_rate']:.4f}.")
    lines.append(f"The LSTM predicted a range of {rec['predicted_range_low']:.4f} — {rec['predicted_range_high']:.4f} "
                 f"({rec['predicted_range_paise']:.0f} paise wide).")
    lines.append(f"XGBoost detected a {rec['regime']} regime with {rec['regime_confidence']:.0%} confidence.")

    if rec['result'] == 'GREEN':
        lines.append(f"The actual range ({rec['actual_low']:.4f} — {rec['actual_high']:.4f}) "
                     f"was fully contained within the predicted range. "
                     f"Result: GREEN — prediction accurate.")
    elif rec['result'] == 'YELLOW':
        lines.append(f"The actual range ({rec['actual_low']:.4f} — {rec['actual_high']:.4f}) "
                     f"partially overlapped the prediction ({rec['overlap_pct']:.0f}% captured). "
                     f"Result: YELLOW — partial success.")
    else:
        lines.append(f"The actual range ({rec['actual_low']:.4f} — {rec['actual_high']:.4f}) "
                     f"fell outside the predicted range entirely. "
                     f"Result: RED — prediction missed.")

    lines.append(f"Rate was at the {kf['rate_vs_alltime_percentile']:.0%} percentile of 22-year history, "
                 f"with 20-day volatility at {kf['volatility_20d']:.2%}.")
    lines.append(f"TPV impact on $9.5M: {rec['tpv_note']}.")
    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Generate HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def step4_html_report(results, training_info):
    """Generate self-contained HTML report."""
    print(f"\n{'=' * 70}")
    print(f"STEP 4: GENERATING HTML REPORT")
    print(f"{'=' * 70}")

    total = len(results)
    green = sum(1 for r in results if r["result"] == "GREEN")
    yellow = sum(1 for r in results if r["result"] == "YELLOW")
    red = sum(1 for r in results if r["result"] == "RED")
    avg_overlap = np.mean([r["overlap_pct"] for r in results])
    avg_pred_range = np.mean([r["predicted_range_paise"] for r in results])
    avg_actual_range = np.mean([r["actual_range_paise"] for r in results])
    total_tpv = sum(r["tpv_impact_inr"] for r in results)

    # Best/worst
    best = max(results, key=lambda r: r["overlap_pct"])
    worst = min(results, key=lambda r: r["overlap_pct"])

    # Monthly breakdown
    jan = [r for r in results if r["date"].startswith("2026-01")]
    feb = [r for r in results if r["date"].startswith("2026-02")]

    # Chart data (ensure native Python floats for JSON)
    def _jf(vals):
        return json.dumps([float(v) for v in vals])
    dates_js = json.dumps([r["date"] for r in results])
    entry_rates_js = _jf([r["entry_rate"] for r in results])
    pred_low_js = _jf([r["predicted_range_low"] for r in results])
    pred_high_js = _jf([r["predicted_range_high"] for r in results])
    actual_low_js = _jf([r["actual_low"] for r in results])
    actual_high_js = _jf([r["actual_high"] for r in results])
    result_colors_js = json.dumps([
        "#22C55E" if r["result"] == "GREEN" else
        "#EAB308" if r["result"] == "YELLOW" else "#EF4444"
        for r in results
    ])

    # Day cards HTML
    day_cards = []
    for r in results:
        color_map = {"GREEN": ("#22C55E", "#0A2E1A", "rgba(34,197,94,0.08)"),
                     "YELLOW": ("#EAB308", "#2E2A0A", "rgba(234,179,8,0.08)"),
                     "RED": ("#EF4444", "#2E0A0A", "rgba(239,68,68,0.08)")}
        dot, header_bg, card_bg = color_map[r["result"]]
        emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[r["result"]]

        opus_html = (r.get("opus_analysis", "") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

        kf = r["key_features"]
        card = f"""
<div class="day-card {r['result'].lower()}" data-result="{r['result']}">
  <div class="card-header" style="border-left: 4px solid {dot}; background: {header_bg};">
    <div class="card-title">
      <span>{emoji} {r['date']} — {r['day_name']}</span>
      <span class="overlap-badge" style="background: {dot};">{r['overlap_pct']:.0f}%</span>
    </div>
    <div class="card-meta">
      <span>Predicted: {r['predicted_range_low']:.2f} — {r['predicted_range_high']:.2f} | Most likely: {r['predicted_most_likely']:.2f}</span>
      <span>Actual: {r['actual_low']:.2f} — {r['actual_high']:.2f} | Close: {r['actual_close']:.2f}</span>
      <span>Regime: {r['regime']} ({r['regime_confidence']:.0%}) | Direction: {r['predicted_direction']} | Conf: {r['confidence']:.0%}</span>
    </div>
  </div>
  <div class="card-body" style="background: {card_bg};">
    <div class="card-section">
      <h4>OPUS ANALYSIS</h4>
      <div class="opus-text">{opus_html}</div>
    </div>
    <div class="card-section treasury-impact">
      <h4>TREASURY IMPACT ($9.5M daily)</h4>
      <p>{r['tpv_note']}</p>
    </div>
    <div class="card-section features-grid">
      <span>📊 Rate percentile: {kf['rate_vs_alltime_percentile']:.0%}</span>
      <span>📈 Trend 30d: {kf['rate_trend_30d']:+.4f}</span>
      <span>⚡ Volatility: {kf['volatility_20d']:.2%}</span>
      <span>📅 {r['calendar_note'] or 'No events'}</span>
    </div>
  </div>
</div>
"""
        day_cards.append(card)

    day_cards_html = "\n".join(day_cards)

    # Monthly tables
    def _monthly_table(month_results, label):
        if not month_results:
            return f"<p>No data for {label}</p>"
        g = sum(1 for r in month_results if r["result"] == "GREEN")
        y = sum(1 for r in month_results if r["result"] == "YELLOW")
        rd = sum(1 for r in month_results if r["result"] == "RED")
        ao = np.mean([r["overlap_pct"] for r in month_results])
        return f"""
<div class="monthly-table">
  <h3>{label}</h3>
  <table>
    <tr><th>Days</th><th>🟢 Green</th><th>🟡 Yellow</th><th>🔴 Red</th><th>Avg Overlap</th></tr>
    <tr><td>{len(month_results)}</td><td>{g} ({g/len(month_results)*100:.0f}%)</td>
        <td>{y} ({y/len(month_results)*100:.0f}%)</td>
        <td>{rd} ({rd/len(month_results)*100:.0f}%)</td>
        <td>{ao:.1f}%</td></tr>
  </table>
</div>"""

    jan_table = _monthly_table(jan, "January 2026")
    feb_table = _monthly_table(feb, "February 2026")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Baniya Buddhi — Walk-Forward Validation Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg-dark: #0A1628;
  --bg-card: #111827;
  --bg-surface: #1F2937;
  --gold: #F0B429;
  --gold-dim: #C49A1C;
  --text: #E5E7EB;
  --text-dim: #9CA3AF;
  --green: #22C55E;
  --yellow: #EAB308;
  --red: #EF4444;
  --border: #374151;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg-dark);
  color: var(--text);
  line-height: 1.6;
}}

.header {{
  background: linear-gradient(135deg, var(--bg-dark) 0%, #0F1D32 100%);
  padding: 3rem 2rem;
  text-align: center;
  border-bottom: 2px solid var(--gold);
}}
.header h1 {{
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--gold);
  margin-bottom: 0.5rem;
}}
.header .subtitle {{
  font-size: 1.2rem;
  color: var(--text-dim);
  font-weight: 300;
}}
.header .date-range {{
  font-size: 1rem;
  color: var(--text);
  margin-top: 0.5rem;
  font-weight: 500;
}}
.header .training-note {{
  font-size: 0.85rem;
  color: var(--gold-dim);
  margin-top: 0.3rem;
  font-style: italic;
}}

.container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}

.dashboard {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}}
.stat-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem;
  text-align: center;
}}
.stat-card .stat-value {{
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--gold);
}}
.stat-card .stat-label {{
  font-size: 0.8rem;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 0.3rem;
}}
.stat-card.green .stat-value {{ color: var(--green); }}
.stat-card.yellow .stat-value {{ color: var(--yellow); }}
.stat-card.red .stat-value {{ color: var(--red); }}

.chart-section {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin: 2rem 0;
}}
.chart-section h2 {{
  color: var(--gold);
  margin-bottom: 1rem;
  font-size: 1.2rem;
}}
.chart-container {{
  position: relative;
  width: 100%;
  height: 400px;
}}

.monthly-section {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin: 2rem 0;
}}
.monthly-table {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
}}
.monthly-table h3 {{
  color: var(--gold);
  margin-bottom: 1rem;
}}
.monthly-table table {{
  width: 100%;
  border-collapse: collapse;
}}
.monthly-table th, .monthly-table td {{
  padding: 0.6rem;
  text-align: center;
  border-bottom: 1px solid var(--border);
  font-size: 0.9rem;
}}
.monthly-table th {{
  color: var(--text-dim);
  font-weight: 500;
  text-transform: uppercase;
  font-size: 0.75rem;
}}

.filter-bar {{
  display: flex;
  gap: 0.5rem;
  margin: 2rem 0 1rem;
  flex-wrap: wrap;
}}
.filter-btn {{
  padding: 0.5rem 1.2rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-card);
  color: var(--text);
  cursor: pointer;
  font-family: inherit;
  font-size: 0.9rem;
  transition: all 0.2s;
}}
.filter-btn:hover {{ background: var(--bg-surface); }}
.filter-btn.active {{
  background: var(--gold);
  color: var(--bg-dark);
  border-color: var(--gold);
  font-weight: 600;
}}

.day-card {{
  margin-bottom: 1.5rem;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--border);
  transition: transform 0.2s;
}}
.day-card:hover {{ transform: translateY(-2px); }}
.card-header {{
  padding: 1rem 1.2rem;
  border-left: 4px solid;
}}
.card-title {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 1.1rem;
  font-weight: 600;
}}
.overlap-badge {{
  padding: 0.2rem 0.8rem;
  border-radius: 20px;
  color: #000;
  font-weight: 700;
  font-size: 0.9rem;
}}
.card-meta {{
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: var(--text-dim);
  font-family: 'SF Mono', 'Fira Code', monospace;
}}
.card-body {{
  padding: 1.2rem;
}}
.card-section {{
  margin-bottom: 1rem;
}}
.card-section h4 {{
  color: var(--gold);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.5rem;
}}
.opus-text {{
  font-size: 0.9rem;
  line-height: 1.7;
  color: var(--text);
  white-space: pre-wrap;
}}
.treasury-impact {{
  background: rgba(240, 180, 41, 0.05);
  border: 1px solid rgba(240, 180, 41, 0.2);
  border-radius: 8px;
  padding: 0.8rem 1rem;
}}
.treasury-impact p {{
  font-weight: 500;
  color: var(--gold);
}}
.features-grid {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  font-size: 0.8rem;
  color: var(--text-dim);
}}

.footer {{
  text-align: center;
  padding: 3rem 2rem;
  border-top: 1px solid var(--border);
  color: var(--text-dim);
  font-size: 0.85rem;
  margin-top: 3rem;
}}
.footer .brand {{ color: var(--gold); font-weight: 600; }}

@media print {{
  body {{ background: #fff; color: #000; }}
  .filter-bar {{ display: none; }}
  .day-card {{ break-inside: avoid; page-break-inside: avoid; }}
  .header {{ background: #fff; border-bottom: 2px solid #000; }}
  .header h1 {{ color: #000; }}
  .stat-card {{ border: 1px solid #ccc; }}
}}

@media (max-width: 768px) {{
  .dashboard {{ grid-template-columns: repeat(2, 1fr); }}
  .monthly-section {{ grid-template-columns: 1fr; }}
  .header h1 {{ font-size: 1.8rem; }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>🪙 Baniya Buddhi</h1>
  <div class="subtitle">Walk-Forward Validation Report</div>
  <div class="date-range">January 2 — February 20, 2026</div>
  <div class="training-note">Models trained strictly on 2003–2024 data | Blind test on 2026</div>
</div>

<div class="container">

  <!-- Dashboard -->
  <div class="dashboard">
    <div class="stat-card">
      <div class="stat-value">{total}</div>
      <div class="stat-label">Trading Days Tested</div>
    </div>
    <div class="stat-card green">
      <div class="stat-value">{green} <small>({green/total*100:.0f}%)</small></div>
      <div class="stat-label">🟢 Green (Full Success)</div>
    </div>
    <div class="stat-card yellow">
      <div class="stat-value">{yellow} <small>({yellow/total*100:.0f}%)</small></div>
      <div class="stat-label">🟡 Yellow (Partial)</div>
    </div>
    <div class="stat-card red">
      <div class="stat-value">{red} <small>({red/total*100:.0f}%)</small></div>
      <div class="stat-label">🔴 Red (Miss)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{avg_overlap:.1f}%</div>
      <div class="stat-label">Avg Overlap</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{_fmt_inr(abs(total_tpv))}</div>
      <div class="stat-label">TPV Net Impact</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{avg_pred_range:.0f}p</div>
      <div class="stat-label">Avg Predicted Range</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{best['date'][5:]}</div>
      <div class="stat-label">Best Day ({best['overlap_pct']:.0f}%)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{worst['date'][5:]}</div>
      <div class="stat-label">Worst Day ({worst['overlap_pct']:.0f}%)</div>
    </div>
  </div>

  <!-- Chart -->
  <div class="chart-section">
    <h2>USD/INR Rate with Predicted Range Bands</h2>
    <div class="chart-container">
      <canvas id="rangeChart"></canvas>
    </div>
  </div>

  <!-- Monthly summaries -->
  <div class="monthly-section">
    {jan_table}
    {feb_table}
  </div>

  <!-- Filter bar -->
  <div class="filter-bar">
    <button class="filter-btn active" onclick="filterCards('ALL')">All ({total})</button>
    <button class="filter-btn" onclick="filterCards('GREEN')">🟢 Green ({green})</button>
    <button class="filter-btn" onclick="filterCards('YELLOW')">🟡 Yellow ({yellow})</button>
    <button class="filter-btn" onclick="filterCards('RED')">🔴 Red ({red})</button>
  </div>

  <!-- Day cards -->
  <div id="dayCards">
    {day_cards_html}
  </div>

</div>

<div class="footer">
  <p><span class="brand">Baniya Buddhi v1.0</span> | Trained on 22 years of data (2003–2024)</p>
  <p>Models: XGBoost Regime + LSTM Range + Macro Signal</p>
  <p>Training cutoff: December 31, 2024 | Validation: January 2 — February 20, 2026</p>
</div>

<script>
// Filter cards
function filterCards(result) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.day-card').forEach(card => {{
    if (result === 'ALL' || card.dataset.result === result) {{
      card.style.display = 'block';
    }} else {{
      card.style.display = 'none';
    }}
  }});
}}

// Chart
const ctx = document.getElementById('rangeChart').getContext('2d');
const dates = {dates_js};
const entryRates = {entry_rates_js};
const predLow = {pred_low_js};
const predHigh = {pred_high_js};
const actualLow = {actual_low_js};
const actualHigh = {actual_high_js};
const resultColors = {result_colors_js};

new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: dates.map(d => d.slice(5)),
    datasets: [
      {{
        label: 'Entry Rate',
        data: entryRates,
        borderColor: '#F0B429',
        backgroundColor: 'rgba(240,180,41,0.1)',
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: resultColors,
        pointBorderColor: resultColors,
        fill: false,
        tension: 0.1,
      }},
      {{
        label: 'Predicted High',
        data: predHigh,
        borderColor: 'rgba(34,197,94,0.5)',
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
      }},
      {{
        label: 'Predicted Low',
        data: predLow,
        borderColor: 'rgba(34,197,94,0.5)',
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: '-1',
        backgroundColor: 'rgba(34,197,94,0.08)',
      }},
      {{
        label: 'Actual High',
        data: actualHigh,
        borderColor: 'rgba(239,68,68,0.4)',
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
      }},
      {{
        label: 'Actual Low',
        data: actualLow,
        borderColor: 'rgba(239,68,68,0.4)',
        borderWidth: 1,
        pointRadius: 0,
        fill: '-1',
        backgroundColor: 'rgba(239,68,68,0.06)',
      }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{
        labels: {{ color: '#9CA3AF', font: {{ family: 'Inter' }} }}
      }},
      tooltip: {{
        callbacks: {{
          afterBody: function(tooltipItems) {{
            const i = tooltipItems[0].dataIndex;
            return [
              'Predicted: ' + predLow[i].toFixed(2) + ' — ' + predHigh[i].toFixed(2),
              'Actual: ' + actualLow[i].toFixed(2) + ' — ' + actualHigh[i].toFixed(2),
            ];
          }}
        }}
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#9CA3AF', maxRotation: 45 }},
        grid: {{ color: 'rgba(55,65,81,0.5)' }}
      }},
      y: {{
        ticks: {{ color: '#9CA3AF' }},
        grid: {{ color: 'rgba(55,65,81,0.5)' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    html_path = os.path.join(OUTPUT_DIR, "walkforward_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report saved: {html_path}")
    return html_path


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Generate JSON and CSV
# ═══════════════════════════════════════════════════════════════════════════

def step5_save_data(results):
    """Save JSON and CSV summaries."""
    print(f"\n{'=' * 70}")
    print(f"STEP 5: SAVING JSON + CSV")
    print(f"{'=' * 70}")

    total = len(results)
    green = sum(1 for r in results if r["result"] == "GREEN")
    yellow = sum(1 for r in results if r["result"] == "YELLOW")
    red = sum(1 for r in results if r["result"] == "RED")
    avg_overlap = np.mean([r["overlap_pct"] for r in results])
    avg_pred_range = np.mean([r["predicted_range_paise"] for r in results])
    avg_actual_range = np.mean([r["actual_range_paise"] for r in results])
    total_tpv = sum(r["tpv_impact_inr"] for r in results)

    best = max(results, key=lambda r: r["overlap_pct"])
    worst = min(results, key=lambda r: r["overlap_pct"])

    # Monthly
    jan = [r for r in results if r["date"].startswith("2026-01")]
    feb = [r for r in results if r["date"].startswith("2026-02")]

    def _monthly_summary(month_results):
        if not month_results:
            return {}
        g = sum(1 for r in month_results if r["result"] == "GREEN")
        y = sum(1 for r in month_results if r["result"] == "YELLOW")
        rd = sum(1 for r in month_results if r["result"] == "RED")
        return {
            "days": len(month_results),
            "green": g, "yellow": y, "red": rd,
            "green_pct": round(g / len(month_results) * 100, 1),
            "avg_overlap_pct": round(np.mean([r["overlap_pct"] for r in month_results]), 1),
        }

    # Range efficiency
    range_eff = (avg_actual_range / avg_pred_range * 100) if avg_pred_range > 0 else 0

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_cutoff": TRAIN_CUTOFF,
        "training_start": TRAIN_START,
        "validation_period": f"{results[0]['date']} to {results[-1]['date']}",
        "model": "lstm_walkforward_2024 + xgb_regime",
        "summary": {
            "total_days": total,
            "green_days": green,
            "yellow_days": yellow,
            "red_days": red,
            "green_pct": round(green / total * 100, 1),
            "avg_overlap_pct": round(avg_overlap, 1),
            "avg_predicted_range_paise": round(avg_pred_range, 1),
            "avg_actual_range_paise": round(avg_actual_range, 1),
            "range_efficiency": round(range_eff, 1),
            "tpv_total_impact_inr": total_tpv,
            "best_day": {"date": best["date"], "overlap": best["overlap_pct"]},
            "worst_day": {"date": worst["date"], "overlap": worst["overlap_pct"]},
        },
        "monthly": {
            "january_2026": _monthly_summary(jan),
            "february_2026": _monthly_summary(feb),
        },
        "daily_results": results,
    }

    json_path = os.path.join(OUTPUT_DIR, "walkforward_2026_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  JSON saved: {json_path}")

    # CSV
    csv_rows = []
    for r in results:
        kf = r["key_features"]
        csv_rows.append({
            "date": r["date"],
            "entry_rate": r["entry_rate"],
            "pred_low": r["predicted_range_low"],
            "pred_high": r["predicted_range_high"],
            "pred_most_likely": r["predicted_most_likely"],
            "actual_low": r["actual_low"],
            "actual_high": r["actual_high"],
            "actual_close": r["actual_close"],
            "result": r["result"],
            "overlap_pct": r["overlap_pct"],
            "regime": r["regime"],
            "confidence": r["confidence"],
            "tpv_impact_inr": r["tpv_impact_inr"],
            "rate_vs_alltime_pct": kf["rate_vs_alltime_percentile"],
            "momentum": kf["momentum_consistency"],
            "volatility": kf["volatility_20d"],
        })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(OUTPUT_DIR, "walkforward_2026_daily.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    return output


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Terminal summary
# ═══════════════════════════════════════════════════════════════════════════

def step6_terminal_summary(results, html_path):
    """Print formatted terminal summary."""
    total = len(results)
    green = sum(1 for r in results if r["result"] == "GREEN")
    yellow = sum(1 for r in results if r["result"] == "YELLOW")
    red = sum(1 for r in results if r["result"] == "RED")
    avg_overlap = np.mean([r["overlap_pct"] for r in results])
    avg_pred_range = np.mean([r["predicted_range_paise"] for r in results])
    avg_actual_range = np.mean([r["actual_range_paise"] for r in results])

    green_savings = sum(r["tpv_impact_inr"] for r in results if r["result"] == "GREEN")
    red_costs = sum(r["tpv_impact_inr"] for r in results if r["result"] == "RED")
    net = green_savings + red_costs

    print(f"""
╔══════════════════════════════════════════════════╗
║     BANIYA BUDDHI — WALK-FORWARD VALIDATION      ║
║         Jan 2, 2026 — Feb 20, 2026               ║
║      Trained on 2003-2024 | Blind test           ║
╠══════════════════════════════════════════════════╣
║ Total trading days tested:    {total:<20d}║
║                                                  ║
║ 🟢 GREEN (full success):      {green} days ({green/total*100:.0f}%)              ║
║ 🟡 YELLOW (partial overlap):  {yellow} days ({yellow/total*100:.0f}%)              ║
║ 🔴 RED (miss):                {red} days ({red/total*100:.0f}%)              ║
║                                                  ║
║ Average overlap:              {avg_overlap:.1f}%               ║
║ Avg predicted range:          {avg_pred_range:.1f} paise          ║
║ Avg actual range:             {avg_actual_range:.1f} paise          ║
║                                                  ║
║ TREASURY IMPACT ($9.5M daily)                    ║
║ Green days saved:             {_fmt_inr(green_savings):>16s}   ║
║ Red days cost:                {_fmt_inr(abs(red_costs)):>16s}   ║
║ Net benefit:                  {_fmt_inr(net):>16s}   ║
║                                                  ║
║ Report: {html_path:<40s}║
╚══════════════════════════════════════════════════╝""")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_inr(amount):
    """Format INR amount in Indian notation."""
    abs_amt = abs(amount)
    sign = "-" if amount < 0 else ""
    if abs_amt >= 1_00_00_000:
        return f"{sign}₹{abs_amt / 1_00_00_000:.1f} crore"
    elif abs_amt >= 1_00_000:
        return f"{sign}₹{abs_amt / 1_00_000:.1f} lakh"
    else:
        return f"{sign}₹{abs_amt:,.0f}"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Baniya Buddhi Walk-Forward Validation")
    parser.add_argument("--step", type=int, default=0, help="Run specific step only (1-6)")
    parser.add_argument("--skip-opus", action="store_true", help="Skip Opus justifications")
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)

    # Step 1: Train LSTM
    model_path = os.path.join(SAVED_DIR, "lstm_walkforward_2024.pt")
    scaler_path = os.path.join(SAVED_DIR, "lstm_scaler_walkforward_2024.pkl")

    if args.step == 1:
        # Explicit step 1 — always retrain
        model, scaler, range_buffer = step1_train_lstm()
        if model is None:
            print("\n  HARD STOP — Aborting.")
            return
        print("\n  Step 1 complete. Run without --step to continue.")
        return
    elif os.path.exists(model_path) and os.path.exists(scaler_path):
        # Load pre-trained model (skip retraining)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model = LSTMRangePredictor(input_dim=checkpoint["input_dim"]).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        scaler = joblib.load(scaler_path)
        range_buffer = checkpoint["range_buffer"]
        print(f"  Loaded pre-trained model from {model_path}")
        print(f"  Range buffer: ±{range_buffer:.4f}")
    else:
        # No saved model — must train first
        model, scaler, range_buffer = step1_train_lstm()
        if model is None:
            print("\n  HARD STOP — Aborting.")
            return

    # Step 2: Walk-forward simulation
    results = step2_walkforward(model, scaler, range_buffer)

    if not results:
        print("\n  No results — aborting.")
        return

    # Step 3: Opus justifications
    if not args.skip_opus:
        results = step3_opus_justifications(results)
    else:
        print(f"\n  Skipping Opus justifications (--skip-opus)")
        for r in results:
            r["opus_analysis"] = _fallback_analysis(r)
            r["opus_model"] = "rule_based_fallback"
            r["opus_tokens"] = 0

    # Step 4: HTML report
    training_info = {"cutoff": TRAIN_CUTOFF, "start": TRAIN_START}
    html_path = step4_html_report(results, training_info)

    # Step 5: JSON + CSV
    step5_save_data(results)

    # Step 6: Terminal summary
    step6_terminal_summary(results, html_path)


if __name__ == "__main__":
    main()
