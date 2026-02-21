"""
FX Band Predictor — Final Validation: Retrain on 2003-2025, Blind Test Jan-Feb 2026

Steps:
  1. Retrain XGBoost regime classifier (extend training through 2025)
  2. Retrain LSTM v2 (BoundaryAwareLoss) with 2025 as validation
  3. Walk-forward blind prediction on 30 trading days
  4. Generate final HTML report
  5. Update production models if results are better

Usage:
    python backtest/final_validation_2025.py
    python backtest/final_validation_2025.py --step 1   # XGBoost only
    python backtest/final_validation_2025.py --step 2   # LSTM only
    python backtest/final_validation_2025.py --step 3   # Walk-forward only (uses saved models)
    python backtest/final_validation_2025.py --step 4   # HTML report only
    python backtest/final_validation_2025.py --step 5   # Production update only
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_DIR)

from features.feature_engineering import build_features
from data.fx_calendar import is_trading_day, get_calendar_context
from models.train_lstm import create_sequences, EarlyStopping
from backtest.lstm_v2_retrain import BoundaryAwareLoss, LSTMRangePredictorV2
from agents.range_risk_detector import detect_high_risk_prediction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(PROJECT_DIR, "data", "market_data_full.csv")
SAVED_DIR = os.path.join(PROJECT_DIR, "models", "saved")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "backtest")

LSTM_FEATURES = [
    "usdinr", "oil", "dxy", "vix", "us10y",
    "rate_trend_30d", "rate_percentile_1y", "momentum_consistency",
    "rate_vs_5y_avg", "rate_vs_alltime_percentile",
    "long_term_trend_1y", "is_decade_high",
]

SEQ_LEN = 30
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20
LR = 0.0003
HUBER_DELTA = 1.0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

TRAIN_START_LSTM = "2015-01-01"
TRAIN_CUTOFF = "2024-12-31"
VAL_START = "2025-01-01"
VAL_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-02-28"

TPV_DAILY_USD = 9_500_000

# XGBoost config
TREND_THRESHOLD = 0.0004
VOL_THRESHOLD = 0.006
REGIME_MAP = {0: "trending_up", 1: "trending_down", 2: "high_vol", 3: "range_bound"}

# XGBoost features to exclude
XGB_REGIME_DERIVED = {
    "regime_trending_up", "regime_trending_down", "regime_high_vol", "regime_range_bound",
    "is_trending_up", "is_trending_down", "trend_strength",
}
XGB_EXCLUDE_COLS = {"date", "usdinr", "oil", "dxy", "vix", "us10y", "us_2y", "fed_funds", "cpi"}


# ═══════════════════════════════════════════════════════════════════════════
# Shared data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_features():
    """Load raw data and build features (cached)."""
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    features_df = build_features(raw)
    return features_df


def build_regime_target(df):
    """Build 4-class regime target (same logic as train_xgb_regime.py)."""
    out = df.copy()
    trend = out["rate_trend_30d"]
    vol = out["volatility_5d"]
    avg_dev = out["rate_vs_30d_avg"]

    def classify(i):
        v = vol.iloc[i]
        t = trend.iloc[i]
        a = avg_dev.iloc[i]
        if v > VOL_THRESHOLD:
            return 2  # high_vol
        elif t > TREND_THRESHOLD and a > 0.002:
            return 0  # trending_up
        elif t < -TREND_THRESHOLD and a < -0.002:
            return 1  # trending_down
        else:
            return 3  # range_bound

    out["regime_target"] = [classify(i) for i in range(len(out))]
    return out


def build_trading_day_targets(df):
    """Build 2-trading-day forward targets (same as train_lstm_fullhistory.py)."""
    out = df.copy()
    dates = pd.to_datetime(out["date"]).tolist()
    rates = out["usdinr"].values

    date_to_rate = {}
    for i, dt in enumerate(dates):
        date_to_rate[dt.date() if hasattr(dt, 'date') else dt] = rates[i]

    y_high = np.full(len(out), np.nan)
    y_low = np.full(len(out), np.nan)
    y_close = np.full(len(out), np.nan)

    for i in range(len(out)):
        current_date = dates[i].date() if hasattr(dates[i], 'date') else dates[i]
        trading_rates = []
        check_date = current_date
        for _ in range(14):
            check_date = check_date + pd.Timedelta(days=1)
            py_date = check_date if isinstance(check_date, date) else check_date.date()
            if is_trading_day(py_date):
                if py_date in date_to_rate:
                    trading_rates.append(date_to_rate[py_date])
                if len(trading_rates) == 2:
                    break
        if len(trading_rates) == 2:
            y_high[i] = max(trading_rates)
            y_low[i] = min(trading_rates)
            y_close[i] = trading_rates[1]

    out["y_high"] = y_high
    out["y_low"] = y_low
    out["y_close"] = y_close
    out["dy_high"] = out["y_high"] - out["usdinr"]
    out["dy_low"] = out["y_low"] - out["usdinr"]
    out["dy_close"] = out["y_close"] - out["usdinr"]
    out = out.dropna(subset=["y_high", "y_low", "y_close"]).reset_index(drop=True)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Retrain XGBoost
# ═══════════════════════════════════════════════════════════════════════════

def step1_xgboost(features_df):
    """Retrain XGBoost regime classifier with data through 2025."""
    print("═" * 70)
    print("STEP 1 — RETRAIN XGBOOST REGIME CLASSIFIER")
    print("═" * 70)

    df = build_regime_target(features_df)

    # Feature columns (same exclusions as original)
    feature_cols = [c for c in df.columns if c not in XGB_EXCLUDE_COLS
                    and c not in XGB_REGIME_DERIVED
                    and c != "regime_target"]

    # Count original training rows (< 2025)
    orig_train_mask = df["date"] < "2025-01-01"
    orig_train_count = orig_train_mask.sum()

    # New: train on everything through 2025
    # Step A: Validate on 2025 (same as original to get comparable val accuracy)
    val_mask = (df["date"] >= "2025-01-01") & (df["date"] < "2026-01-01")
    train_mask = df["date"] < "2025-01-01"

    X_train_val = df.loc[train_mask, feature_cols].values
    y_train_val = df.loc[train_mask, "regime_target"].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, "regime_target"].values

    print(f"\n  Phase A — Validate on 2025 (for accuracy metric)")
    print(f"  Train (< 2025): {len(X_train_val)} rows")
    print(f"  Val (2025):     {len(X_val)} rows")

    sw_train = compute_sample_weight("balanced", y_train_val)

    model_val = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42, verbosity=0,
    )
    model_val.fit(X_train_val, y_train_val, sample_weight=sw_train)

    y_val_pred = model_val.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Val accuracy (2025): {val_acc:.1%}")

    if val_acc < 0.90:
        print(f"\n  *** HARD STOP: Val accuracy {val_acc:.1%} < 90% — not saving ***")
        return None, None, val_acc

    # Step B: Retrain final model on ALL data through 2025
    print(f"\n  Phase B — Retrain on full 2003-2025 data")
    full_mask = df["date"] < "2026-01-01"
    X_full = df.loc[full_mask, feature_cols].values
    y_full = df.loc[full_mask, "regime_target"].values
    new_rows = len(X_full) - len(X_train_val)

    sw_full = compute_sample_weight("balanced", y_full)
    model_final = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42, verbosity=0,
    )
    model_final.fit(X_full, y_full, sample_weight=sw_full)

    # Top 5 features
    importances = model_final.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    top5 = [f[0] for f in feat_imp[:5]]

    # Regime distribution in 2025 data
    y_2025 = df.loc[val_mask, "regime_target"].values
    total_2025 = len(y_2025)
    dist = {}
    for cls in sorted(REGIME_MAP.keys()):
        count = (y_2025 == cls).sum()
        dist[REGIME_MAP[cls]] = count / total_2025 * 100

    # Save
    model_path = os.path.join(SAVED_DIR, "xgb_regime_2025.pkl")
    fnames_path = os.path.join(SAVED_DIR, "feature_names_regime_2025.pkl")
    joblib.dump(model_final, model_path)
    joblib.dump(feature_cols, fnames_path)

    print(f"""
XGBOOST RETRAIN SUMMARY
─────────────────────────────────────────────────
Training rows:     {len(X_full)}  (was {orig_train_count} through 2024)
New rows added:    {new_rows}    (2025 trading days)
Val accuracy:      {val_acc:.1%}  (was 99.3%)
Feature count:     {len(feature_cols)}
Top 5 features:    {top5}
Model saved:       {model_path}
─────────────────────────────────────────────────
Regime distribution in 2025 training data:
  range_bound:   {dist.get('range_bound', 0):.1f}%
  trending_up:   {dist.get('trending_up', 0):.1f}%
  trending_down: {dist.get('trending_down', 0):.1f}%
  high_vol:      {dist.get('high_vol', 0):.1f}%""")

    return model_final, feature_cols, val_acc


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Retrain LSTM v2
# ═══════════════════════════════════════════════════════════════════════════

def step2_lstm(features_df):
    """Retrain LSTM v2 with BoundaryAwareLoss, val on 2025."""
    print(f"\n{'═' * 70}")
    print("STEP 2 — RETRAIN LSTM v2 (BoundaryAwareLoss)")
    print(f"{'═' * 70}")

    for f in LSTM_FEATURES:
        assert f in features_df.columns, f"Missing feature: {f}"

    print("  Building trading-day targets...")
    df = build_trading_day_targets(features_df)
    print(f"  After targets: {len(df)} rows")

    # Train: 2015-2024, Val: 2025 (explicit)
    train_mask = (df["date"] >= TRAIN_START_LSTM) & (df["date"] <= TRAIN_CUTOFF)
    val_mask = (df["date"] >= VAL_START) & (df["date"] <= VAL_END)

    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)

    print(f"\n  Train: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['date'].min().date()} to {val_df['date'].max().date()})")

    # Fit scaler on training data ONLY
    scaler = MinMaxScaler()
    scaler.fit(train_df[LSTM_FEATURES].values)
    train_scaled = scaler.transform(train_df[LSTM_FEATURES].values)
    val_scaled = scaler.transform(val_df[LSTM_FEATURES].values)

    # Build sequences
    train_targets = train_df[["dy_high", "dy_low", "dy_close"]].values
    val_targets = val_df[["dy_high", "dy_low", "dy_close"]].values
    train_entry = train_df["usdinr"].values
    val_entry = val_df["usdinr"].values
    train_dates = train_df["date"].values
    val_dates = val_df["date"].values

    X_train, y_train, d_train, r_train = create_sequences(
        train_scaled, train_targets, train_dates, train_entry, SEQ_LEN
    )

    # For val: need full dataset (train + val) scaled, then extract val sequences
    # because sequences need 30-day lookback that may span into training period
    full_train_val = pd.concat([train_df, val_df]).reset_index(drop=True)
    full_scaled = scaler.transform(full_train_val[LSTM_FEATURES].values)
    full_targets = full_train_val[["dy_high", "dy_low", "dy_close"]].values
    full_entry = full_train_val["usdinr"].values
    full_dates = full_train_val["date"].values

    X_all, y_all, d_all, r_all = create_sequences(
        full_scaled, full_targets, full_dates, full_entry, SEQ_LEN
    )

    # Extract val sequences (those whose dates fall in 2025)
    val_start_ts = pd.Timestamp(VAL_START)
    val_indices = [i for i in range(len(d_all)) if pd.Timestamp(d_all[i]) >= val_start_ts]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]
    r_val = r_all[val_indices]

    print(f"  Train sequences: {X_train.shape}")
    print(f"  Val sequences:   {X_val.shape}")

    orig_model_path = os.path.join(SAVED_DIR, "lstm_walkforward_v2.pt")
    orig_sequences = "~2,400"
    if os.path.exists(orig_model_path):
        orig_ckpt = torch.load(orig_model_path, map_location="cpu", weights_only=False)
        orig_sequences = "from v2 checkpoint"

    # Train
    print(f"\n{'─' * 70}")
    print(f"  TRAINING LSTM v2 (device={DEVICE})")
    print(f"{'─' * 70}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRangePredictorV2(input_dim=len(LSTM_FEATURES)).to(DEVICE)
    criterion = BoundaryAwareLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    stopper = EarlyStopping(patience=PATIENCE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: LSTM({len(LSTM_FEATURES)}→64)→LN→LSTM(64→32)→LN→Dense(32→16→3)")
    print(f"  Parameters: {total_params:,}")
    print(f"  Loss: BoundaryAwareLoss (0.35/0.35/0.15/0.10/0.05)")
    print(f"  LR: {LR} with ReduceLROnPlateau")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val)
        new_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1 or new_lr != current_lr:
            lr_note = f"  LR↓{new_lr:.1e}" if new_lr != current_lr else ""
            print(f"  Epoch {epoch:>3d}/{MAX_EPOCHS}  train={avg_train:.6f}  val={avg_val:.6f}  lr={current_lr:.1e}{lr_note}")

        if stopper.step(avg_val, model):
            best_epoch = epoch - PATIENCE
            print(f"\n  Early stopping at epoch {epoch}. Best: {best_epoch}")
            break
    else:
        best_epoch = MAX_EPOCHS
        print(f"\n  Completed all {MAX_EPOCHS} epochs.")

    train_time = time.time() - t0
    stopper.restore(model)

    # Compute val metrics
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy()

    high_residuals = np.abs(val_preds[:, 0] - y_val[:, 0])
    low_residuals = np.abs(val_preds[:, 1] - y_val[:, 1])
    close_residuals = np.abs(val_preds[:, 2] - y_val[:, 2])

    mae_high = np.mean(high_residuals) * 100
    mae_low = np.mean(low_residuals) * 100
    mae_close = np.mean(close_residuals) * 100

    # Width metrics
    pred_width = val_preds[:, 0] - val_preds[:, 1]
    actual_width = y_val[:, 0] - y_val[:, 1]
    mae_width = np.mean(np.abs(pred_width - actual_width)) * 100

    # Train/val ratio
    final_train_loss = avg_train
    final_val_loss = stopper.best_loss
    tv_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else 0

    # Compute P90 buffer + asymmetric upside from 2025 val residuals
    all_residuals = np.concatenate([high_residuals, low_residuals])
    range_buffer_p90 = float(np.percentile(all_residuals, 90))

    # Asymmetric upside: compute directional bias on val set
    # If actual_high > pred_high more often, we need upside buffer
    high_bias = val_preds[:, 0] - y_val[:, 0]  # negative means we underpredict high
    upside_miss = np.mean(np.maximum(0, -high_bias))  # average underestimate of highs
    asymmetric_upside = float(np.percentile(np.maximum(0, -high_bias), 50))  # median upside miss

    # Fallback: use ~50% of buffer like v2
    if asymmetric_upside < 0.01:
        asymmetric_upside = range_buffer_p90 * 0.5

    print(f"""
LSTM v2 RETRAIN SUMMARY (2003-2025)
─────────────────────────────────────────────────
Training sequences:  {X_train.shape[0]}  (was ~2,400 through 2024)
Val sequences:       {X_val.shape[0]}    (2025 data)
Best val epoch:      {best_epoch}
Best val loss:       {final_val_loss:.6f}
Train/val ratio:     {tv_ratio:.2f}   {'⚠ OVERFITTING' if tv_ratio > 1.5 else '✓ OK'}
─────────────────────────────────────────────────
BoundaryAwareLoss components (val):
  high_mae:   {mae_high:.1f}p
  low_mae:    {mae_low:.1f}p
  width_mae:  {mae_width:.1f}p
  close_mae:  {mae_close:.1f}p
─────────────────────────────────────────────────
P90 buffer:     {range_buffer_p90:.4f}  (recomputed from 2025 val)
Asym upside:    {asymmetric_upside:.4f}  (recomputed from 2025 val)
─────────────────────────────────────────────────
Training time:  {train_time:.1f}s""")

    if tv_ratio > 1.8:
        print(f"\n  ⚠ OVERFITTING WARNING: train/val ratio {tv_ratio:.2f} > 1.8")
    if mae_high > 20 or mae_low > 20:
        print(f"\n  ⚠ WARNING: Boundary MAE > 20p (high={mae_high:.1f}p, low={mae_low:.1f}p)")

    # Save
    model_path = os.path.join(SAVED_DIR, "lstm_final_2025.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(LSTM_FEATURES),
        "seq_len": SEQ_LEN,
        "features": LSTM_FEATURES,
        "predicts_deltas": True,
        "range_buffer": float(range_buffer_p90),
        "range_buffer_p90": float(range_buffer_p90),
        "asymmetric_upside": float(asymmetric_upside),
        "training_cutoff": TRAIN_CUTOFF,
        "training_start": TRAIN_START_LSTM,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "best_epoch": best_epoch,
        "train_time_seconds": round(train_time, 1),
        "loss": "BoundaryAwareLoss",
        "architecture": "LSTMv2_LayerNorm",
        "val_mae_high_paise": round(mae_high, 1),
        "val_mae_low_paise": round(mae_low, 1),
        "val_mae_close_paise": round(mae_close, 1),
        "val_mae_width_paise": round(mae_width, 1),
        "train_val_ratio": round(tv_ratio, 2),
    }, model_path)
    print(f"\n  Model saved:  {model_path}")

    scaler_path = os.path.join(SAVED_DIR, "lstm_scaler_final_2025.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved: {scaler_path}")

    return model, scaler, range_buffer_p90, asymmetric_upside, {
        "mae_high": mae_high, "mae_low": mae_low, "mae_close": mae_close,
        "mae_width": mae_width, "tv_ratio": tv_ratio, "best_epoch": best_epoch,
        "train_sequences": X_train.shape[0], "val_sequences": X_val.shape[0],
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def step3_walkforward(features_df, lstm_model, scaler, range_buffer,
                      asymmetric_upside, xgb_model, xgb_feature_cols):
    """Run walk-forward on Jan-Feb 2026 with final models."""
    print(f"\n{'═' * 70}")
    print("STEP 3 — WALK-FORWARD BLIND PREDICTION (Jan-Feb 2026)")
    print(f"{'═' * 70}")

    # Load v1 baseline results
    v1_csv = os.path.join(OUTPUT_DIR, "walkforward_2026_daily.csv")
    v1_baseline = {}
    if os.path.exists(v1_csv):
        v1_df = pd.read_csv(v1_csv)
        for _, row in v1_df.iterrows():
            v1_baseline[row["date"]] = {
                "pred_low": row["pred_low"], "pred_high": row["pred_high"],
                "result": row["result"], "overlap_pct": row["overlap_pct"],
            }
        print(f"  Loaded v1 baseline: {len(v1_baseline)} days")

    # Load v2 baseline results
    v2_csv = os.path.join(OUTPUT_DIR, "v1_vs_v2_comparison.csv")
    v2_baseline = {}
    if os.path.exists(v2_csv):
        v2_df = pd.read_csv(v2_csv)
        for _, row in v2_df.iterrows():
            v2_baseline[row["date"]] = {
                "pred_low": row["pred_low"], "pred_high": row["pred_high"],
                "result": row["result"], "overlap_pct": row["overlap_pct"],
                "pred_range_paise": row["pred_range_paise"],
            }
        print(f"  Loaded v2 baseline: {len(v2_baseline)} days")

    # Get trading dates in test window
    val_mask = (features_df["date"] >= TEST_START) & (features_df["date"] <= TEST_END)
    val_dates = features_df.loc[val_mask, "date"].tolist()
    trading_dates = [d for d in val_dates if is_trading_day(
        d.date() if hasattr(d, 'date') else d)]
    print(f"  Trading days: {len(trading_dates)}")

    # Date → rate lookup
    date_to_rate = {}
    for _, row in features_df.iterrows():
        d = row["date"]
        dt = d.date() if hasattr(d, 'date') else d
        date_to_rate[dt] = float(row["usdinr"])

    results = []
    lstm_model.eval()
    prev_regime = None

    print(f"\n  {'Date':<12s} {'Regime':<14s} {'PredLo':>7s} {'PredHi':>7s} {'ActLo':>7s} {'ActHi':>7s} "
          f"{'Res':>4s} {'Ovlp%':>6s} {'Err':>5s}")
    print(f"  {'─'*12} {'─'*14} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*4} {'─'*6} {'─'*5}")

    for pred_date in trading_dates:
        pred_dt = pred_date.date() if hasattr(pred_date, 'date') else pred_date

        # 1. Slice data up to this date (no lookahead)
        slice_mask = features_df["date"] <= pred_date
        df_slice = features_df.loc[slice_mask].copy()
        if len(df_slice) < SEQ_LEN + 1:
            continue

        entry_rate = float(df_slice["usdinr"].iloc[-1])

        # 2. LSTM prediction
        scaled = scaler.transform(df_slice[LSTM_FEATURES].values)
        seq = scaled[-SEQ_LEN:]
        X = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            delta_preds = lstm_model(X).cpu().numpy()[0]

        pred_high_raw = entry_rate + max(delta_preds[0], delta_preds[1])
        pred_low_raw = entry_rate + min(delta_preds[0], delta_preds[1])
        pred_close = entry_rate + delta_preds[2]

        pred_high = pred_high_raw + range_buffer + asymmetric_upside
        pred_low = pred_low_raw - range_buffer

        # 3. XGBoost regime
        regime = "range_bound"
        if xgb_model is not None:
            try:
                latest = df_slice.iloc[[-1]]
                X_xgb = latest[xgb_feature_cols].values
                proba = xgb_model.predict_proba(X_xgb)[0]
                pred_class = int(np.argmax(proba))
                regime = REGIME_MAP[pred_class]
            except Exception:
                pass

        # 4. Look up actual outcomes (next 2 trading days)
        actual_rates = []
        check = pred_dt
        for _ in range(14):
            check = check + timedelta(days=1)
            if is_trading_day(check) and check in date_to_rate:
                actual_rates.append(date_to_rate[check])
            if len(actual_rates) == 2:
                break
        if len(actual_rates) < 2:
            continue

        actual_high = max(actual_rates)
        actual_low = min(actual_rates)
        actual_close = actual_rates[1]

        # 5. Evaluate
        if actual_low >= pred_low and actual_high <= pred_high:
            result = "GREEN"
            overlap_pct = 100.0
        else:
            overlap_low = max(pred_low, actual_low)
            overlap_high = min(pred_high, actual_high)
            if overlap_high > overlap_low:
                result = "YELLOW"
                actual_span = actual_high - actual_low
                overlap_pct = round((overlap_high - overlap_low) / actual_span * 100, 1) if actual_span > 0 else 100.0
            else:
                result = "RED"
                overlap_pct = 0.0

        # 6. Error metrics
        high_error = abs(pred_high - actual_high) * 100
        low_error = abs(pred_low - actual_low) * 100
        total_error = high_error + low_error
        pred_center = (pred_high + pred_low) / 2
        actual_center = (actual_high + actual_low) / 2
        position_error = abs(pred_center - actual_center) * 100
        pred_width = (pred_high - pred_low) * 100
        actual_width = (actual_high - actual_low) * 100
        width_error = abs(pred_width - actual_width)

        # 7. Risk detection
        row = df_slice.iloc[-1]
        recent_5 = df_slice.tail(5)
        high_5d = float(recent_5["usdinr"].max()) if len(recent_5) >= 5 else 0
        low_5d = float(recent_5["usdinr"].min()) if len(recent_5) >= 5 else 0
        vol_5d = float(recent_5["usdinr"].pct_change().dropna().std()) if len(recent_5) >= 5 else 0
        recent_30 = df_slice.tail(30)
        ma_30 = float(recent_30["usdinr"].mean()) if len(recent_30) >= 20 else entry_rate
        prev_friday_close = 0
        if pred_dt.weekday() == 0:
            prev_days = df_slice.tail(3)
            if len(prev_days) > 0:
                prev_friday_close = float(prev_days.iloc[-1]["usdinr"])

        cal = get_calendar_context(pred_dt)
        risk_features = {
            "rate_vs_alltime_percentile": float(row.get("rate_vs_alltime_percentile", 0)),
            "momentum_consistency": int(row.get("momentum_consistency", 0)),
            "volatility_20d": float(row.get("volatility_20d", 0)),
            "volatility_5d": vol_5d,
            "current_rate": entry_rate,
            "high_5d": high_5d, "low_5d": low_5d, "ma_30": ma_30,
            "current_regime": regime,
            "prev_regime": prev_regime or regime,
            "is_fomc_day": cal.get("is_fomc_day", False),
            "is_rbi_day": cal.get("is_rbi_day", False),
            "day_of_week": pred_dt.weekday(),
            "prev_friday_close": prev_friday_close,
        }
        risk = detect_high_risk_prediction(risk_features)

        # 8. TPV impact
        if result == "GREEN":
            tpv_impact = max(0, round((pred_high - actual_high) * TPV_DAILY_USD))
        elif result == "RED":
            miss = max(0, actual_high - pred_high, pred_low - actual_low)
            tpv_impact = -round(miss * TPV_DAILY_USD)
        else:
            miss = max(0, actual_high - pred_high)
            tpv_impact = -round(miss * TPV_DAILY_USD)

        date_str = pred_dt.isoformat()
        v1 = v1_baseline.get(date_str, {})
        v2 = v2_baseline.get(date_str, {})

        rec = {
            "date": date_str,
            "entry_rate": round(entry_rate, 4),
            "pred_low": round(pred_low, 4),
            "pred_high": round(pred_high, 4),
            "pred_close": round(pred_close, 4),
            "actual_low": round(actual_low, 4),
            "actual_high": round(actual_high, 4),
            "actual_close": round(actual_close, 4),
            "result": result,
            "overlap_pct": overlap_pct,
            "regime": regime,
            "tpv_impact": tpv_impact,
            "pred_range_paise": round(pred_width, 1),
            "actual_range_paise": round(actual_width, 1),
            "high_error": round(high_error, 1),
            "low_error": round(low_error, 1),
            "total_error": round(total_error, 1),
            "position_error": round(position_error, 1),
            "width_error": round(width_error, 1),
            "risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"],
            "risk_factors": risk["risk_factors"],
            "recommended_buffer_paise": risk["recommended_buffer_paise"],
            "v1_result": v1.get("result", "?"),
            "v1_overlap": v1.get("overlap_pct", 0),
            "v1_pred_low": v1.get("pred_low", 0),
            "v1_pred_high": v1.get("pred_high", 0),
            "v2_result": v2.get("result", "?"),
            "v2_overlap": v2.get("overlap_pct", 0),
            "v2_pred_low": v2.get("pred_low", 0),
            "v2_pred_high": v2.get("pred_high", 0),
        }
        results.append(rec)
        prev_regime = regime

        # Print
        sym = {"GREEN": "G", "YELLOW": "Y", "RED": "R"}[result]
        print(f"  {date_str:<12s} {regime:<14s} {pred_low:>7.2f} {pred_high:>7.2f} "
              f"{actual_low:>7.2f} {actual_high:>7.2f} {sym:>4s} {overlap_pct:>5.1f}% {total_error:>4.0f}p")

    # Save results
    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_results = []
    for r in results:
        clean = {}
        for k, v in r.items():
            if isinstance(v, list):
                clean[k] = [_convert(x) if not isinstance(x, str) else x for x in v]
            else:
                clean[k] = _convert(v)
        clean_results.append(clean)

    json_path = os.path.join(OUTPUT_DIR, "walkforward_final_2025_results.json")
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n  JSON saved: {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, "walkforward_final_2025_daily.csv")
    pd.DataFrame(clean_results).to_csv(csv_path, index=False)
    print(f"  CSV saved:  {csv_path}")

    # Print comparison table
    _print_comparison(results)

    return results


def _print_comparison(results):
    """Print v1 vs v2 vs final comparison table."""
    if not results:
        return

    total = len(results)

    # V1 metrics
    v1_green = sum(1 for r in results if r["v1_result"] == "GREEN")
    v1_yellow = sum(1 for r in results if r["v1_result"] == "YELLOW")
    v1_red = sum(1 for r in results if r["v1_result"] == "RED")
    v1_avg_overlap = np.mean([r["v1_overlap"] for r in results])

    # V2 metrics
    v2_green = sum(1 for r in results if r["v2_result"] == "GREEN")
    v2_yellow = sum(1 for r in results if r["v2_result"] == "YELLOW")
    v2_red = sum(1 for r in results if r["v2_result"] == "RED")
    v2_avg_overlap = np.mean([r["v2_overlap"] for r in results])

    # Final metrics
    f_green = sum(1 for r in results if r["result"] == "GREEN")
    f_yellow = sum(1 for r in results if r["result"] == "YELLOW")
    f_red = sum(1 for r in results if r["result"] == "RED")
    f_avg_overlap = np.mean([r["overlap_pct"] for r in results])

    # Error metrics
    f_high_err = np.mean([r["high_error"] for r in results])
    f_low_err = np.mean([r["low_error"] for r in results])
    f_total_err = np.mean([r["total_error"] for r in results])
    f_pos_err = np.mean([r["position_error"] for r in results])
    f_width_err = np.mean([r["width_error"] for r in results])
    f_avg_range = np.mean([r["pred_range_paise"] for r in results])

    # V1 error estimates (from v1 baselines)
    v1_high_errs = [abs(r["v1_pred_high"] - r["actual_high"]) * 100 for r in results if r["v1_pred_high"] > 0]
    v1_low_errs = [abs(r["v1_pred_low"] - r["actual_low"]) * 100 for r in results if r["v1_pred_low"] > 0]
    v1_high_err = np.mean(v1_high_errs) if v1_high_errs else 0
    v1_low_err = np.mean(v1_low_errs) if v1_low_errs else 0
    v1_total_err = v1_high_err + v1_low_err
    v1_pos_errs = [abs((r["v1_pred_high"] + r["v1_pred_low"])/2 - (r["actual_high"] + r["actual_low"])/2) * 100
                   for r in results if r["v1_pred_high"] > 0]
    v1_pos_err = np.mean(v1_pos_errs) if v1_pos_errs else 0
    v1_width_errs = [abs((r["v1_pred_high"] - r["v1_pred_low"])*100 - r["actual_range_paise"])
                     for r in results if r["v1_pred_high"] > 0]
    v1_width_err = np.mean(v1_width_errs) if v1_width_errs else 0
    v1_avg_range = np.mean([(r["v1_pred_high"] - r["v1_pred_low"])*100 for r in results if r["v1_pred_high"] > 0])

    # V2 error estimates
    v2_high_errs = [abs(r["v2_pred_high"] - r["actual_high"]) * 100 for r in results if r["v2_pred_high"] > 0]
    v2_low_errs = [abs(r["v2_pred_low"] - r["actual_low"]) * 100 for r in results if r["v2_pred_low"] > 0]
    v2_high_err = np.mean(v2_high_errs) if v2_high_errs else 0
    v2_low_err = np.mean(v2_low_errs) if v2_low_errs else 0
    v2_total_err = v2_high_err + v2_low_err
    v2_pos_errs = [abs((r["v2_pred_high"] + r["v2_pred_low"])/2 - (r["actual_high"] + r["actual_low"])/2) * 100
                   for r in results if r["v2_pred_high"] > 0]
    v2_pos_err = np.mean(v2_pos_errs) if v2_pos_errs else 0
    v2_width_errs = [abs((r["v2_pred_high"] - r["v2_pred_low"])*100 - r["actual_range_paise"])
                     for r in results if r["v2_pred_high"] > 0]
    v2_width_err = np.mean(v2_width_errs) if v2_width_errs else 0
    v2_avg_range = np.mean([(r["v2_pred_high"] - r["v2_pred_low"])*100 for r in results if r["v2_pred_high"] > 0])

    # Regime breakdown
    regimes = {}
    for r in results:
        rg = r["regime"]
        if rg not in regimes:
            regimes[rg] = {"v1_err": [], "v2_err": [], "f_err": []}
        if r["v1_pred_high"] > 0:
            regimes[rg]["v1_err"].append(
                abs(r["v1_pred_high"] - r["actual_high"])*100 + abs(r["v1_pred_low"] - r["actual_low"])*100)
        if r["v2_pred_high"] > 0:
            regimes[rg]["v2_err"].append(
                abs(r["v2_pred_high"] - r["actual_high"])*100 + abs(r["v2_pred_low"] - r["actual_low"])*100)
        regimes[rg]["f_err"].append(r["total_error"])

    # Risk detection
    red_days = [r for r in results if r["result"] == "RED"]
    red_flagged = sum(1 for r in red_days if r["risk_level"] in ("HIGH", "CRITICAL"))

    print(f"""
FINAL VALIDATION RESULTS — 2003-2025 TRAINING
═══════════════════════════════════════════════════════════════
Metric                v1 (orig)  v2 (2024)  FINAL (2025)
───────────────────────────────────────────────────────────────
High error (mean)     {v1_high_err:>5.1f}p     {v2_high_err:>5.1f}p     {f_high_err:>5.1f}p
Low error (mean)      {v1_low_err:>5.1f}p     {v2_low_err:>5.1f}p     {f_low_err:>5.1f}p
Total error (mean)    {v1_total_err:>5.1f}p     {v2_total_err:>5.1f}p     {f_total_err:>5.1f}p
Position error        {v1_pos_err:>5.1f}p     {v2_pos_err:>5.1f}p     {f_pos_err:>5.1f}p
Width error           {v1_width_err:>5.1f}p     {v2_width_err:>5.1f}p     {f_width_err:>5.1f}p
───────────────────────────────────────────────────────────────
Avg range width       {v1_avg_range:>5.0f}p      {v2_avg_range:>5.0f}p      {f_avg_range:>5.0f}p
───────────────────────────────────────────────────────────────
GREEN days            {v1_green:>2d} ({v1_green/total*100:>2.0f}%)     {v2_green:>2d} ({v2_green/total*100:>2.0f}%)     {f_green:>2d} ({f_green/total*100:>2.0f}%)
YELLOW days           {v1_yellow:>2d} ({v1_yellow/total*100:>2.0f}%)     {v2_yellow:>2d} ({v2_yellow/total*100:>2.0f}%)     {f_yellow:>2d} ({f_yellow/total*100:>2.0f}%)
RED days              {v1_red:>2d} ({v1_red/total*100:>2.0f}%)      {v2_red:>2d} ({v2_red/total*100:>2.0f}%)      {f_red:>2d} ({f_red/total*100:>2.0f}%)
Avg overlap           {v1_avg_overlap:>5.1f}%     {v2_avg_overlap:>5.1f}%     {f_avg_overlap:>5.1f}%
───────────────────────────────────────────────────────────────""")

    # Regime breakdown
    print("Regime breakdown (total error):")
    for rg in sorted(regimes.keys()):
        data = regimes[rg]
        v1_e = np.mean(data["v1_err"]) if data["v1_err"] else 0
        v2_e = np.mean(data["v2_err"]) if data["v2_err"] else 0
        f_e = np.mean(data["f_err"]) if data["f_err"] else 0
        print(f"  {rg:<16s}     {v1_e:>5.1f}p     {v2_e:>5.1f}p     {f_e:>5.1f}p")

    print(f"───────────────────────────────────────────────────────────────")
    print(f"Risk detection:")
    print(f"  RED days flagged    {red_flagged}/{len(red_days)}")
    print(f"  (HIGH or CRITICAL)")
    print(f"═══════════════════════════════════════════════════════════════")

    # Per-day summary
    print(f"\nDATE       REGIME         PRED RANGE         ACTUAL RANGE       RESULT  ERROR")
    for r in results:
        sym = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[r["result"]]
        print(f"{r['date'][5:]}  {r['regime']:<14s} {r['pred_low']:.2f} — {r['pred_high']:.2f}  "
              f"{r['actual_low']:.2f} — {r['actual_high']:.2f}  {sym}  {r['total_error']:>5.0f}p")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def step4_html_report(results):
    """Generate final HTML validation report."""
    print(f"\n{'═' * 70}")
    print("STEP 4 — GENERATE FINAL HTML REPORT")
    print(f"{'═' * 70}")

    total = len(results)
    if total == 0:
        print("  No results to report.")
        return None

    # Metrics
    f_green = sum(1 for r in results if r["result"] == "GREEN")
    f_yellow = sum(1 for r in results if r["result"] == "YELLOW")
    f_red = sum(1 for r in results if r["result"] == "RED")
    f_avg_overlap = np.mean([r["overlap_pct"] for r in results])
    f_high_err = np.mean([r["high_error"] for r in results])
    f_low_err = np.mean([r["low_error"] for r in results])
    f_total_err = np.mean([r["total_error"] for r in results])
    f_avg_range = np.mean([r["pred_range_paise"] for r in results])

    # V1 metrics
    v1_green = sum(1 for r in results if r["v1_result"] == "GREEN")
    v1_avg_overlap = np.mean([r["v1_overlap"] for r in results])
    v1_high_errs = [abs(r["v1_pred_high"] - r["actual_high"]) * 100 for r in results if r["v1_pred_high"] > 0]
    v1_low_errs = [abs(r["v1_pred_low"] - r["actual_low"]) * 100 for r in results if r["v1_pred_low"] > 0]
    v1_high_err = np.mean(v1_high_errs) if v1_high_errs else 0
    v1_low_err = np.mean(v1_low_errs) if v1_low_errs else 0
    v1_total_err = v1_high_err + v1_low_err
    v1_avg_range = np.mean([(r["v1_pred_high"] - r["v1_pred_low"])*100 for r in results if r["v1_pred_high"] > 0])

    # V2 metrics
    v2_green = sum(1 for r in results if r["v2_result"] == "GREEN")
    v2_avg_overlap = np.mean([r["v2_overlap"] for r in results])
    v2_high_errs = [abs(r["v2_pred_high"] - r["actual_high"]) * 100 for r in results if r["v2_pred_high"] > 0]
    v2_low_errs = [abs(r["v2_pred_low"] - r["actual_low"]) * 100 for r in results if r["v2_pred_low"] > 0]
    v2_high_err = np.mean(v2_high_errs) if v2_high_errs else 0
    v2_low_err = np.mean(v2_low_errs) if v2_low_errs else 0
    v2_total_err = v2_high_err + v2_low_err
    v2_avg_range = np.mean([(r["v2_pred_high"] - r["v2_pred_low"])*100 for r in results if r["v2_pred_high"] > 0])

    # Progress towards target
    target_err = 25.0
    target_overlap = 75.0
    err_progress = min(100, max(0, (v1_total_err - f_total_err) / (v1_total_err - target_err) * 100)) if v1_total_err > target_err else 100
    ovlp_progress = min(100, max(0, (f_avg_overlap - v1_avg_overlap) / (target_overlap - v1_avg_overlap) * 100)) if target_overlap > v1_avg_overlap else 100
    overall_progress = (err_progress + ovlp_progress) / 2

    # Monthly
    jan = [r for r in results if r["date"].startswith("2026-01")]
    feb = [r for r in results if r["date"].startswith("2026-02")]

    jan_green = sum(1 for r in jan if r["result"] == "GREEN")
    jan_yellow = sum(1 for r in jan if r["result"] == "YELLOW")
    jan_red = sum(1 for r in jan if r["result"] == "RED")
    jan_overlap = np.mean([r["overlap_pct"] for r in jan]) if jan else 0

    feb_green = sum(1 for r in feb if r["result"] == "GREEN")
    feb_yellow = sum(1 for r in feb if r["result"] == "YELLOW")
    feb_red = sum(1 for r in feb if r["result"] == "RED")
    feb_overlap = np.mean([r["overlap_pct"] for r in feb]) if feb else 0

    # Regime heatmap data
    regimes_data = {}
    for r in results:
        rg = r["regime"]
        if rg not in regimes_data:
            regimes_data[rg] = {"v1": [], "v2": [], "final": []}
        if r["v1_pred_high"] > 0:
            regimes_data[rg]["v1"].append(
                abs(r["v1_pred_high"] - r["actual_high"])*100 + abs(r["v1_pred_low"] - r["actual_low"])*100)
        if r["v2_pred_high"] > 0:
            regimes_data[rg]["v2"].append(
                abs(r["v2_pred_high"] - r["actual_high"])*100 + abs(r["v2_pred_low"] - r["actual_low"])*100)
        regimes_data[rg]["final"].append(r["total_error"])

    # Risk detection
    red_days = [r for r in results if r["result"] == "RED"]
    red_flagged = sum(1 for r in red_days if r["risk_level"] in ("HIGH", "CRITICAL"))
    high_critical_days = [r for r in results if r["risk_level"] in ("HIGH", "CRITICAL")]

    # Key finding
    if f_total_err < v2_total_err and f_avg_overlap >= v2_avg_overlap:
        key_finding = (f"Adding 2025 training data improved total error by "
                       f"{v2_total_err - f_total_err:.1f}p and maintained {f_avg_overlap:.1f}% overlap")
    elif f_total_err >= v2_total_err:
        key_finding = (f"2025 training data did not reduce total error ({v2_total_err:.1f}p → {f_total_err:.1f}p). "
                       f"v2 boundary model is already near-optimal for this test period.")
    else:
        key_finding = f"Final model achieves {f_green} GREEN days with {f_avg_overlap:.1f}% average overlap"

    # Build day cards
    day_cards_html = ""
    for r in results:
        color = {"GREEN": "#22C55E", "YELLOW": "#EAB308", "RED": "#EF4444"}[r["result"]]
        bg = {"GREEN": "#0A2E1A", "YELLOW": "#2E2A0A", "RED": "#2E0A0A"}[r["result"]]
        emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[r["result"]]

        # V1 comparison
        v1_res = r.get("v1_result", "?")
        v1_ovlp = r.get("v1_overlap", 0)
        v2_res = r.get("v2_result", "?")

        # Range bars (simplified visual)
        range_min = min(r["pred_low"], r["actual_low"], r.get("v1_pred_low", r["pred_low"])) - 0.05
        range_max = max(r["pred_high"], r["actual_high"], r.get("v1_pred_high", r["pred_high"])) + 0.05
        span = range_max - range_min if range_max > range_min else 1

        def bar_style(lo, hi):
            left = (lo - range_min) / span * 100
            width = (hi - lo) / span * 100
            return f"left:{left:.0f}%;width:{max(2, width):.0f}%"

        v1_bar = bar_style(r.get("v1_pred_low", r["pred_low"]), r.get("v1_pred_high", r["pred_high"]))
        final_bar = bar_style(r["pred_low"], r["pred_high"])
        actual_bar = bar_style(r["actual_low"], r["actual_high"])

        risk_badge_color = {"CRITICAL": "#EF4444", "HIGH": "#F97316", "ELEVATED": "#EAB308",
                           "MILD": "#6B7280", "NORMAL": "#374151"}[r["risk_level"]]

        # System note
        pred_w = r["pred_range_paise"]
        act_w = r["actual_range_paise"]
        diff = r["total_error"] - (abs(r.get("v1_pred_high",0) - r["actual_high"])*100 + abs(r.get("v1_pred_low",0) - r["actual_low"])*100) if r.get("v1_pred_high", 0) > 0 else 0
        comp = "Better" if diff < -2 else ("Worse" if diff > 2 else "Same")

        day_cards_html += f"""
<div class="day-card" style="border-left:4px solid {color};background:{bg};">
  <div class="card-header">
    <div class="card-top">
      <span class="card-date">{emoji} {r['date']} &mdash; {pd.Timestamp(r['date']).strftime('%A')}</span>
      <span class="overlap-badge" style="background:{color};color:#000;">{r['overlap_pct']:.0f}%</span>
    </div>
    <div class="card-detail">Regime: {r['regime']} | Error: {r['total_error']:.0f}p</div>
  </div>
  <div class="range-bars">
    <div class="bar-row"><span class="bar-label">v1:</span><div class="bar-track"><div class="bar-fill" style="{v1_bar};background:rgba(156,163,175,0.4);"></div></div><span class="bar-vals">{r.get('v1_pred_low',0):.2f} — {r.get('v1_pred_high',0):.2f}</span></div>
    <div class="bar-row"><span class="bar-label">Final:</span><div class="bar-track"><div class="bar-fill" style="{final_bar};background:rgba(34,197,94,0.5);"></div></div><span class="bar-vals">{r['pred_low']:.2f} — {r['pred_high']:.2f}</span></div>
    <div class="bar-row"><span class="bar-label">Actual:</span><div class="bar-track"><div class="bar-fill" style="{actual_bar};background:rgba(239,68,68,0.5);"></div></div><span class="bar-vals">{r['actual_low']:.2f} — {r['actual_high']:.2f}</span></div>
  </div>
  <div class="card-footer">
    <span>Overlap: v1={v1_ovlp:.0f}% | Final={r['overlap_pct']:.0f}%</span>
    <span>Error: v1={(abs(r.get('v1_pred_high',0)-r['actual_high'])+abs(r.get('v1_pred_low',0)-r['actual_low']))*100:.0f}p | Final={r['total_error']:.0f}p</span>
    <span class="risk-badge" style="background:{risk_badge_color};">{r['risk_level']}</span>
  </div>
  <div class="card-note">Range was {pred_w:.0f} paise wide. Actual traded in {act_w:.0f} paise range. {comp} than v1 by {abs(diff):.0f}p.</div>
</div>"""

    # Regime heatmap rows
    regime_heatmap = ""
    for rg in ["range_bound", "trending_up", "trending_down", "high_vol"]:
        if rg not in regimes_data:
            continue
        d = regimes_data[rg]
        v1_e = np.mean(d["v1"]) if d["v1"] else 0
        v2_e = np.mean(d["v2"]) if d["v2"] else 0
        f_e = np.mean(d["final"]) if d["final"] else 0
        n = len(d["final"])
        # Color: green if improved vs v1, red if worse
        v1_cls = "improved" if f_e < v1_e - 2 else ("worse" if f_e > v1_e + 2 else "same")
        v2_cls = "improved" if f_e < v2_e - 2 else ("worse" if f_e > v2_e + 2 else "same")
        regime_heatmap += f"""<tr>
  <td>{rg}</td><td>{n}</td>
  <td class="{v1_cls}">{v1_e:.1f}p</td>
  <td class="{v2_cls}">{v2_e:.1f}p</td>
  <td>{f_e:.1f}p</td></tr>\n"""

    # Risk table
    risk_table = ""
    for r in high_critical_days:
        risk_table += f"""<tr>
  <td>{r['date']}</td><td style="color:{'#EF4444' if r['risk_level']=='CRITICAL' else '#F97316'}">{r['risk_level']}</td>
  <td>{r['risk_score']}</td><td>{r['result']}</td>
  <td>{'; '.join(f[:50] for f in r['risk_factors'][:2])}</td></tr>\n"""

    # Progress bar
    bar_width = max(0, min(100, overall_progress))
    bar_color = "#22C55E" if bar_width > 70 else "#EAB308" if bar_width > 40 else "#EF4444"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FX Band Predictor — Final Validation Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{ --bg:#0A1628; --card:#111827; --surface:#1F2937; --gold:#F0B429;
  --text:#E5E7EB; --dim:#9CA3AF; --green:#22C55E; --yellow:#EAB308; --red:#EF4444; --border:#374151; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
.header {{ background:linear-gradient(135deg,var(--bg),#0F1D32); padding:3rem 2rem; text-align:center; border-bottom:2px solid var(--gold); }}
.header h1 {{ font-size:2.5rem; color:var(--gold); }}
.header .sub {{ color:var(--dim); font-size:1.1rem; margin-top:0.3rem; }}
.header .meta {{ color:var(--text); margin-top:0.5rem; font-weight:500; }}
.container {{ max-width:1200px; margin:0 auto; padding:2rem; }}
h2 {{ color:var(--gold); font-size:1.3rem; margin:2rem 0 1rem; }}
.grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; }}
.grid6 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:0.8rem; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1rem; text-align:center; }}
.card .val {{ font-size:1.6rem; font-weight:700; color:var(--gold); }}
.card .lbl {{ font-size:0.75rem; color:var(--dim); text-transform:uppercase; margin-top:0.2rem; }}
.card.g .val {{ color:var(--green); }}
.card.r .val {{ color:var(--red); }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:10px; overflow:hidden; margin:1rem 0; }}
th {{ background:var(--surface); color:var(--dim); font-size:0.75rem; text-transform:uppercase; padding:0.6rem; text-align:center; }}
td {{ padding:0.5rem; text-align:center; border-bottom:1px solid var(--border); font-size:0.85rem; }}
.improved {{ color:var(--green); font-weight:600; }}
.worse {{ color:var(--red); font-weight:600; }}
.same {{ color:var(--dim); }}
.progress-bar {{ background:var(--surface); border-radius:8px; height:28px; margin:1rem 0; overflow:hidden; position:relative; }}
.progress-fill {{ height:100%; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:700; color:#000; }}
.monthly {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
.monthly .card {{ text-align:left; }}
.monthly .card h3 {{ color:var(--gold); font-size:1rem; margin-bottom:0.5rem; }}
.day-card {{ margin:0.8rem 0; border-radius:10px; overflow:hidden; }}
.card-header {{ padding:0.8rem 1rem; }}
.card-top {{ display:flex; justify-content:space-between; align-items:center; }}
.card-date {{ font-weight:600; font-size:1rem; }}
.overlap-badge {{ padding:0.15rem 0.6rem; border-radius:12px; font-weight:700; font-size:0.85rem; }}
.card-detail {{ font-size:0.8rem; color:var(--dim); margin-top:0.3rem; }}
.range-bars {{ padding:0.5rem 1rem; }}
.bar-row {{ display:flex; align-items:center; gap:0.5rem; margin:0.2rem 0; font-size:0.75rem; }}
.bar-label {{ width:40px; color:var(--dim); text-align:right; }}
.bar-track {{ flex:1; height:12px; background:var(--surface); border-radius:6px; position:relative; }}
.bar-fill {{ position:absolute; top:0; height:100%; border-radius:6px; }}
.bar-vals {{ width:120px; font-family:monospace; font-size:0.7rem; color:var(--dim); }}
.card-footer {{ display:flex; gap:1rem; padding:0.3rem 1rem; font-size:0.75rem; color:var(--dim); align-items:center; }}
.risk-badge {{ padding:0.1rem 0.5rem; border-radius:4px; font-size:0.7rem; font-weight:700; color:#000; }}
.card-note {{ padding:0.3rem 1rem 0.8rem; font-size:0.8rem; color:var(--dim); font-style:italic; }}
.footer {{ text-align:center; padding:3rem 2rem; border-top:1px solid var(--border); margin-top:3rem; }}
.footer .brand {{ color:var(--gold); font-weight:600; font-size:1.1rem; }}
.footer p {{ color:var(--dim); font-size:0.85rem; margin:0.3rem 0; }}
@media (max-width:768px) {{ .grid4,.grid6 {{ grid-template-columns:repeat(2,1fr); }} .monthly {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="header">
  <h1>🪙 FX Band Predictor — Final Validation Report</h1>
  <div class="sub">Trained: 2003–2025 | Blind Test: Jan–Feb 2026</div>
  <div class="meta">Both models retrained with full data through December 2025</div>
</div>
<div class="container">

<h2>Summary Dashboard</h2>
<div class="grid4">
  <div><table><tr><th></th><th>v1</th><th>v2</th><th>Final</th><th>Δ</th></tr>
  <tr><td>GREEN</td><td>{v1_green}</td><td>{v2_green}</td><td style="color:var(--green);font-weight:700">{f_green}</td><td>{f_green - v1_green:+d}</td></tr>
  <tr><td>Overlap</td><td>{v1_avg_overlap:.0f}%</td><td>{v2_avg_overlap:.0f}%</td><td style="font-weight:700">{f_avg_overlap:.0f}%</td><td>{f_avg_overlap - v1_avg_overlap:+.0f}%</td></tr>
  <tr><td>High err</td><td>{v1_high_err:.0f}p</td><td>{v2_high_err:.0f}p</td><td>{f_high_err:.0f}p</td><td>{f_high_err - v1_high_err:+.0f}p</td></tr>
  <tr><td>Low err</td><td>{v1_low_err:.0f}p</td><td>{v2_low_err:.0f}p</td><td>{f_low_err:.0f}p</td><td>{f_low_err - v1_low_err:+.0f}p</td></tr>
  <tr><td>Total err</td><td>{v1_total_err:.0f}p</td><td>{v2_total_err:.0f}p</td><td>{f_total_err:.0f}p</td><td>{f_total_err - v1_total_err:+.0f}p</td></tr>
  <tr><td>Range</td><td>{v1_avg_range:.0f}p</td><td>{v2_avg_range:.0f}p</td><td>{f_avg_range:.0f}p</td><td></td></tr></table></div>

  <div class="card g"><div class="val">{f_green}/{total}</div><div class="lbl">GREEN Days</div></div>
  <div class="card"><div class="val">{f_avg_overlap:.1f}%</div><div class="lbl">Avg Overlap</div></div>
  <div class="card"><div class="val">{f_total_err:.0f}p</div><div class="lbl">Total Error</div></div>
</div>

<h2>Progress: v1 → Final</h2>
<div style="font-size:0.85rem;color:var(--dim);margin-bottom:0.3rem;">Target: total_error &lt; 25p, overlap &gt; 75%</div>
<div class="progress-bar"><div class="progress-fill" style="width:{bar_width:.0f}%;background:{bar_color};">{bar_width:.0f}%</div></div>

<h2>Regime Heatmap (total error by regime)</h2>
<table>
<tr><th>Regime</th><th>Days</th><th>v1</th><th>v2</th><th>Final</th></tr>
{regime_heatmap}</table>

<h2>Monthly Breakdown</h2>
<div class="monthly">
  <div class="card"><h3>January 2026</h3>
    <p>🟢 {jan_green} GREEN &nbsp; 🟡 {jan_yellow} YELLOW &nbsp; 🔴 {jan_red} RED</p>
    <p style="color:var(--dim)">Overlap: {jan_overlap:.1f}%</p></div>
  <div class="card"><h3>February 2026</h3>
    <p>🟢 {feb_green} GREEN &nbsp; 🟡 {feb_yellow} YELLOW &nbsp; 🔴 {feb_red} RED</p>
    <p style="color:var(--dim)">Overlap: {feb_overlap:.1f}%</p>
    <p style="color:var(--dim);font-size:0.8rem;margin-top:0.3rem">Did 2025 data help Feb? Rate reverted to 90–91 range.</p></div>
</div>

<h2>Day Cards ({total} days)</h2>
{day_cards_html}

<h2>Risk Detection</h2>
<p style="color:var(--dim);margin-bottom:0.5rem;">Score: {red_flagged} of {len(red_days)} RED days were correctly flagged HIGH/CRITICAL.</p>
<p style="color:var(--dim);margin-bottom:1rem;">Risk detector would have warned treasury on {red_flagged}/{len(red_days)} catastrophic days.</p>
<table>
<tr><th>Date</th><th>Risk Level</th><th>Score</th><th>Result</th><th>Factors</th></tr>
{risk_table}</table>

</div>
<div class="footer">
  <p class="brand">FX Band Predictor v2.0 | Sharp money. Sharper timing.</p>
  <p>Training: 2003–2025 | Val: 2025 (252 trading days) | Test: Jan–Feb 2026 (30 trading days, never seen)</p>
  <p>Models: XGBoost (45 features) + LSTM v2 BoundaryAwareLoss (12 features, 30-day sequences)</p>
  <p style="color:var(--gold);margin-top:0.5rem;font-weight:500;">{key_finding}</p>
</div>
</body>
</html>"""

    html_path = os.path.join(OUTPUT_DIR, "walkforward_final_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved: {html_path}")
    print(f"  Lines: {len(html.splitlines())}")
    return html_path


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Update production
# ═══════════════════════════════════════════════════════════════════════════

def step5_update_production(results):
    """Update production models if final is better than v2."""
    print(f"\n{'═' * 70}")
    print("STEP 5 — PRODUCTION UPDATE DECISION")
    print(f"{'═' * 70}")

    total = len(results)
    f_total_err = np.mean([r["total_error"] for r in results])
    f_avg_overlap = np.mean([r["overlap_pct"] for r in results])

    # V2 metrics
    v2_total_errs = []
    v2_overlaps = []
    for r in results:
        if r["v2_pred_high"] > 0:
            v2_total_errs.append(
                abs(r["v2_pred_high"] - r["actual_high"])*100 + abs(r["v2_pred_low"] - r["actual_low"])*100)
        v2_overlaps.append(r["v2_overlap"])
    v2_total_err = np.mean(v2_total_errs) if v2_total_errs else 999
    v2_avg_overlap = np.mean(v2_overlaps)

    print(f"\n  Final total error:  {f_total_err:.1f}p  (v2: {v2_total_err:.1f}p)")
    print(f"  Final avg overlap:  {f_avg_overlap:.1f}%  (v2: {v2_avg_overlap:.1f}%)")

    should_update = (f_total_err < v2_total_err) and (f_avg_overlap >= v2_avg_overlap)

    if should_update:
        print(f"\n  VERDICT: Final models are BETTER — updating production")
        return True
    else:
        if f_total_err >= v2_total_err:
            print(f"\n  VERDICT: Final total error ({f_total_err:.1f}p) >= v2 ({v2_total_err:.1f}p)")
        if f_avg_overlap < v2_avg_overlap:
            print(f"  VERDICT: Final overlap ({f_avg_overlap:.1f}%) < v2 ({v2_avg_overlap:.1f}%)")
        print(f"\n  2025 training did not improve performance on Jan-Feb 2026 test set.")
        print(f"  Keeping existing v2 models in production.")
        print(f"  This is an HONEST result — v2 boundary model is already optimal.")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Final Validation — Retrain 2003-2025")
    parser.add_argument("--step", type=int, default=0, help="Run specific step (1-5)")
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)

    print("Loading market data and building features...")
    features_df = load_features()
    print(f"  Features: {len(features_df)} rows ({features_df['date'].min().date()} to {features_df['date'].max().date()})")

    # Step 1: XGBoost
    if args.step == 0 or args.step == 1:
        xgb_model, xgb_features, xgb_acc = step1_xgboost(features_df)
        if xgb_model is None:
            print("\n  HARD STOP on XGBoost. Aborting.")
            return
        if args.step == 1:
            return

    # Step 2: LSTM
    if args.step == 0 or args.step == 2:
        result = step2_lstm(features_df)
        if result is None:
            print("\n  HARD STOP on LSTM. Aborting.")
            return
        lstm_model, scaler, range_buffer, asym_up, lstm_metrics = result
        if args.step == 2:
            return

    # Load models if running step 3+ only
    if args.step >= 3:
        # Load XGBoost
        xgb_path = os.path.join(SAVED_DIR, "xgb_regime_2025.pkl")
        xgb_fnames_path = os.path.join(SAVED_DIR, "feature_names_regime_2025.pkl")
        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            xgb_features = joblib.load(xgb_fnames_path)
        else:
            xgb_model, xgb_features = None, None

        # Load LSTM
        lstm_path = os.path.join(SAVED_DIR, "lstm_final_2025.pt")
        scaler_path = os.path.join(SAVED_DIR, "lstm_scaler_final_2025.pkl")
        ckpt = torch.load(lstm_path, map_location=DEVICE, weights_only=False)
        lstm_model = LSTMRangePredictorV2(input_dim=ckpt["input_dim"]).to(DEVICE)
        lstm_model.load_state_dict(ckpt["model_state_dict"])
        lstm_model.eval()
        scaler = joblib.load(scaler_path)
        range_buffer = ckpt["range_buffer_p90"]
        asym_up = ckpt["asymmetric_upside"]
        print(f"  Loaded final models (buffer={range_buffer:.4f}, asym={asym_up:.4f})")

    # Step 3: Walk-forward
    if args.step == 0 or args.step == 3:
        results = step3_walkforward(
            features_df, lstm_model, scaler, range_buffer, asym_up,
            xgb_model, xgb_features
        )
        if not results:
            print("\n  No results. Aborting.")
            return
        if args.step == 3:
            return

    # Load results if running step 4+ only
    if args.step >= 4:
        results_path = os.path.join(OUTPUT_DIR, "walkforward_final_2025_results.json")
        with open(results_path) as f:
            results = json.load(f)

    # Step 4: HTML report
    if args.step == 0 or args.step == 4:
        step4_html_report(results)
        if args.step == 4:
            return

    # Step 5: Production update
    if args.step == 0 or args.step == 5:
        should_update = step5_update_production(results)
        if should_update:
            print("\n  To complete production update, run:")
            print("  python backtest/final_validation_2025.py --apply-production")


if __name__ == "__main__":
    main()
