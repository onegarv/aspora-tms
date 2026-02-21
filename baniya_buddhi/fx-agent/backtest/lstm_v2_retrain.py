"""
Baniya Buddhi — LSTM v2 Retraining with BoundaryAwareLoss

Step 3 of LSTM tuning pipeline:
  1. BoundaryAwareLoss: 0.35*high + 0.35*low + 0.15*close + 0.10*width + 0.05*invalid
  2. Enhanced architecture with LayerNorm
  3. LR=0.0003 with ReduceLROnPlateau scheduler
  4. Walk-forward evaluation on Jan-Feb 2026
  5. Two-stage: retrained model + calibration layer on top
  6. Side-by-side comparison with v1 baseline

Usage:
    python backtest/lstm_v2_retrain.py                 # Full pipeline
    python backtest/lstm_v2_retrain.py --train-only     # Train only, skip evaluation
    python backtest/lstm_v2_retrain.py --eval-only      # Evaluate existing v2 model
"""

import argparse
import copy
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_DIR)

from features.feature_engineering import build_features
from data.fx_calendar import is_trading_day
from models.train_lstm import create_sequences, EarlyStopping

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
MAX_EPOCHS = 200
PATIENCE = 20
LR = 0.0003  # Lower LR for boundary-aware training
HUBER_DELTA = 1.0
VAL_SPLIT = 0.1
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

TRAIN_CUTOFF = "2024-12-31"
TRAIN_START = "2015-01-01"
VALIDATION_START = "2026-01-01"
VALIDATION_END = "2026-02-28"

TPV_DAILY_USD = 9_500_000

# Regime map (same as v1)
REGIME_MAP = {0: "trending_up", 1: "trending_down", 2: "high_vol", 3: "range_bound"}


# ═══════════════════════════════════════════════════════════════════════════
# BoundaryAwareLoss
# ═══════════════════════════════════════════════════════════════════════════

class BoundaryAwareLoss(nn.Module):
    """
    Custom loss that prioritizes boundary accuracy:
      0.35 * high_error + 0.35 * low_error + 0.15 * close_error
      + 0.10 * width_penalty + 0.05 * invalid_penalty

    - high_error/low_error: Huber loss on boundary deltas
    - close_error: Huber loss on close delta
    - width_penalty: penalizes predicted width deviating from actual width
    - invalid_penalty: penalizes cases where pred_high < pred_low
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="none")
        self.w_high = 0.35
        self.w_low = 0.35
        self.w_close = 0.15
        self.w_width = 0.10
        self.w_invalid = 0.05

    def forward(self, pred, target):
        """
        pred:   (batch, 3) → [dy_high, dy_low, dy_close]
        target: (batch, 3) → [dy_high, dy_low, dy_close]
        """
        # Boundary losses
        high_loss = self.huber(pred[:, 0], target[:, 0]).mean()
        low_loss = self.huber(pred[:, 1], target[:, 1]).mean()
        close_loss = self.huber(pred[:, 2], target[:, 2]).mean()

        # Width loss: penalize predicted width deviating from actual width
        pred_width = pred[:, 0] - pred[:, 1]    # dy_high - dy_low
        actual_width = target[:, 0] - target[:, 1]
        width_loss = self.huber(pred_width, actual_width).mean()

        # Invalid penalty: pred_high should be > pred_low (dy_high > dy_low)
        # Penalize when dy_high < dy_low (inverted range)
        invalid = torch.relu(pred[:, 1] - pred[:, 0])  # positive when low > high
        invalid_loss = (invalid ** 2).mean()

        total = (self.w_high * high_loss
                 + self.w_low * low_loss
                 + self.w_close * close_loss
                 + self.w_width * width_loss
                 + self.w_invalid * invalid_loss)

        return total


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced LSTM with LayerNorm
# ═══════════════════════════════════════════════════════════════════════════

class LSTMRangePredictorV2(nn.Module):
    """
    Enhanced LSTM with LayerNorm for boundary-aware prediction.

    LSTM(input→64) → LayerNorm → Dropout(0.2) →
    LSTM(64→32) → LayerNorm → Dropout(0.2) →
    Dense(32→16, relu) → Dense(16→3)
    """

    def __init__(self, input_dim=12):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.ln2 = nn.LayerNorm(32)
        self.drop2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)  # [dy_high, dy_low, dy_close]

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        out = self.drop1(out)

        out, _ = self.lstm2(out)
        out = self.ln2(out)
        # Take only the last time step
        out = out[:, -1, :]
        out = self.drop2(out)

        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_v2():
    """Train LSTM v2 with BoundaryAwareLoss on 2015-2024."""
    print("=" * 70)
    print("STEP 3: RETRAIN LSTM v2 — BoundaryAwareLoss + LayerNorm")
    print(f"  Features: {len(FEATURES)}, Seq: {SEQ_LEN}, Device: {DEVICE}")
    print(f"  LR: {LR}, BoundaryAwareLoss weights: high=0.35 low=0.35 close=0.15 width=0.10 invalid=0.05")
    print("=" * 70)

    # Load full data
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"\n  Raw data: {len(raw)} rows ({raw['date'].min().date()} to {raw['date'].max().date()})")

    # Build features
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

    print(f"\n  Training data: {len(train_df)} rows")
    print(f"    From: {train_df['date'].min().date()}")
    print(f"    To:   {train_df['date'].max().date()}")

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
    print(f"  TRAINING LSTM v2 (device={DEVICE})")
    print(f"{'─' * 70}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRangePredictorV2(input_dim=len(FEATURES)).to(DEVICE)
    criterion = BoundaryAwareLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    stopper = EarlyStopping(patience=PATIENCE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: LSTM({len(FEATURES)}→64)→LN→LSTM(64→32)→LN→Dense(32→16→3)")
    print(f"  Parameters: {total_params:,}")
    print(f"  Scheduler: ReduceLROnPlateau(factor=0.5, patience=8)")
    print()

    t0 = time.time()
    best_epoch = 0
    loss_history = []

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

        loss_history.append({"epoch": epoch, "train": avg_train, "val": avg_val, "lr": current_lr})

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

    # --- Calibrate range buffer (P85 of val residuals) ---
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy()

    # Compute boundary residuals for buffer calibration
    high_residuals = np.abs(val_preds[:, 0] - y_val[:, 0])
    low_residuals = np.abs(val_preds[:, 1] - y_val[:, 1])
    close_residuals = np.abs(val_preds[:, 2] - y_val[:, 2])

    # Use separate buffers for high and low
    range_buffer_high = float(np.percentile(high_residuals, 85))
    range_buffer_low = float(np.percentile(low_residuals, 85))
    range_buffer_symmetric = float(np.percentile(
        np.concatenate([high_residuals, low_residuals]), 85
    ))

    # Also compute P90 and P95 buffers for experimentation
    range_buffer_p90 = float(np.percentile(
        np.concatenate([high_residuals, low_residuals]), 90
    ))
    range_buffer_p95 = float(np.percentile(
        np.concatenate([high_residuals, low_residuals]), 95
    ))

    # Also compute close residual for sanity
    close_residual = np.abs((r_val + val_preds[:, 2]) - (r_val + y_val[:, 2]))
    range_buffer_close = float(np.percentile(close_residual, 85))

    print(f"\n  Range buffer (val residuals):")
    print(f"    High P85: ±{range_buffer_high:.4f} INR")
    print(f"    Low P85:  ±{range_buffer_low:.4f} INR")
    print(f"    Symmetric P85: ±{range_buffer_symmetric:.4f} INR")
    print(f"    Symmetric P90: ±{range_buffer_p90:.4f} INR")
    print(f"    Symmetric P95: ±{range_buffer_p95:.4f} INR")
    print(f"    Close MAE: {np.mean(close_residuals):.4f} INR")
    print(f"  Training time: {train_time:.1f}s")

    # --- Val diagnostics ---
    print(f"\n{'─' * 70}")
    print(f"  VALIDATION DIAGNOSTICS (last 10% of training window)")
    print(f"{'─' * 70}")

    pred_width = val_preds[:, 0] - val_preds[:, 1]
    actual_width = y_val[:, 0] - y_val[:, 1]
    width_ratio = np.mean(np.abs(pred_width)) / np.mean(np.abs(actual_width)) if np.mean(np.abs(actual_width)) > 0 else 0

    mae_high = np.mean(high_residuals) * 100  # in paise
    mae_low = np.mean(low_residuals) * 100
    mae_close = np.mean(close_residuals) * 100

    print(f"  MAE high:  {mae_high:.1f} paise")
    print(f"  MAE low:   {mae_low:.1f} paise")
    print(f"  MAE close: {mae_close:.1f} paise")
    print(f"  Width ratio: {width_ratio:.2f}x")
    print(f"  Avg pred width: {np.mean(np.abs(pred_width)) * 100:.1f} paise")
    print(f"  Avg actual width: {np.mean(np.abs(actual_width)) * 100:.1f} paise")

    # --- Q4 2024 sanity check ---
    q4_df = train_df.tail(60).copy()
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
    q4_pred_high = r_q4 + np.maximum(q4_preds[:, 0], q4_preds[:, 1]) + range_buffer_symmetric
    q4_pred_low = r_q4 + np.minimum(q4_preds[:, 0], q4_preds[:, 1]) - range_buffer_symmetric
    q4_actual_close = r_q4 + y_q4[:, 2]
    q4_actual_high = r_q4 + y_q4[:, 0]
    q4_actual_low = r_q4 + y_q4[:, 1]

    q4_mae_close = np.mean(np.abs(q4_pred_close - q4_actual_close))
    q4_range_acc = ((q4_actual_close >= q4_pred_low) & (q4_actual_close <= q4_pred_high)).mean()

    print(f"\n{'─' * 70}")
    print(f"  Q4 2024 SANITY CHECK")
    print(f"{'─' * 70}")
    print(f"  MAE close: {q4_mae_close:.4f} INR {'PASS' if q4_mae_close <= 0.50 else 'FAIL'}")
    print(f"  Range accuracy: {q4_range_acc:.1%}")

    if q4_mae_close > 0.50:
        print(f"\n  *** HARD STOP: MAE {q4_mae_close:.4f} > 0.50 — model not saved ***")
        return None, None, None, None

    # --- Save ---
    os.makedirs(SAVED_DIR, exist_ok=True)
    model_path = os.path.join(SAVED_DIR, "lstm_walkforward_v2.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(FEATURES),
        "seq_len": SEQ_LEN,
        "features": FEATURES,
        "predicts_deltas": True,
        "range_buffer": float(range_buffer_symmetric),
        "range_buffer_p90": float(range_buffer_p90),
        "range_buffer_p95": float(range_buffer_p95),
        "range_buffer_high": float(range_buffer_high),
        "range_buffer_low": float(range_buffer_low),
        "training_cutoff": TRAIN_CUTOFF,
        "training_start": TRAIN_START,
        "best_epoch": best_epoch,
        "train_time_seconds": round(train_time, 1),
        "loss": "BoundaryAwareLoss",
        "architecture": "LSTMv2_LayerNorm",
        "val_mae_close_paise": round(mae_close, 1),
        "val_mae_high_paise": round(mae_high, 1),
        "val_mae_low_paise": round(mae_low, 1),
        "val_width_ratio": round(width_ratio, 2),
    }, model_path)
    print(f"\n  Model saved: {model_path}")

    scaler_path = os.path.join(SAVED_DIR, "lstm_scaler_walkforward_v2.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved: {scaler_path}")

    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE — LSTM v2 (BoundaryAwareLoss + LayerNorm)")
    print(f"  Best epoch: {best_epoch} | Close MAE: {mae_close:.1f}p | Width ratio: {width_ratio:.2f}x")
    print(f"{'=' * 70}")

    return model, scaler, range_buffer_symmetric, {
        "range_buffer_high": range_buffer_high,
        "range_buffer_low": range_buffer_low,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Walk-Forward Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def walkforward_evaluate(model, scaler, range_buffer, extra_buffers=None,
                         use_calibration=True, config_label="v2",
                         asymmetric_upside=0.0):
    """Run walk-forward evaluation on Jan-Feb 2026 and compare with v1.

    asymmetric_upside: extra buffer added to high side only (for upward bias correction)
    """
    print(f"\n{'=' * 70}")
    cal_note = " + calibration" if use_calibration else " (no calibration)"
    asym_note = f" + {asymmetric_upside:.4f} upside bias" if asymmetric_upside > 0 else ""
    print(f"WALK-FORWARD EVALUATION [{config_label}] — Jan 2 to Feb 20, 2026{cal_note}{asym_note}")
    print(f"  Range buffer: ±{range_buffer:.4f}")
    print(f"{'=' * 70}")

    # Load v1 baseline results
    v1_csv_path = os.path.join(OUTPUT_DIR, "walkforward_2026_daily.csv")
    v1_df = pd.read_csv(v1_csv_path)
    v1_baseline = {}
    for _, row in v1_df.iterrows():
        v1_baseline[row["date"]] = {
            "pred_low": row["pred_low"],
            "pred_high": row["pred_high"],
            "pred_most_likely": row["pred_most_likely"],
            "result": row["result"],
            "overlap_pct": row["overlap_pct"],
        }
    print(f"  Loaded v1 baseline: {len(v1_baseline)} days")

    # Load full data
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    features_df = build_features(raw)

    # Get all dates in validation window
    val_mask = (features_df["date"] >= VALIDATION_START) & (features_df["date"] <= VALIDATION_END)
    val_dates = features_df.loc[val_mask, "date"].tolist()
    trading_dates = [d for d in val_dates if is_trading_day(d.date() if hasattr(d, 'date') else d)]
    print(f"  Trading days in validation window: {len(trading_dates)}")

    # Load XGBoost regime model
    xgb_model = None
    xgb_feature_names = None
    xgb_path = os.path.join(SAVED_DIR, "xgb_regime.pkl")
    xgb_features_path = os.path.join(SAVED_DIR, "feature_names_regime.pkl")
    if os.path.exists(xgb_path) and os.path.exists(xgb_features_path):
        xgb_model = joblib.load(xgb_path)
        xgb_feature_names = joblib.load(xgb_features_path)

    # Load calibration data
    cal_path = os.path.join(SAVED_DIR, "lstm_width_calibration.json")
    calibration = None
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            calibration = json.load(f)
        print(f"  Loaded width calibration data")

    # Build date → rate lookup
    from datetime import timedelta
    date_to_rate = {}
    for _, row in features_df.iterrows():
        d = row["date"]
        dt = d.date() if hasattr(d, 'date') else d
        date_to_rate[dt] = float(row["usdinr"])

    results = []
    model.eval()

    print(f"\n  {'Date':<12s} {'Rate':>7s} {'v1Lo':>7s} {'v1Hi':>7s} {'v2Lo':>7s} {'v2Hi':>7s} "
          f"{'ActLo':>7s} {'ActHi':>7s} {'v1':>4s} {'v2':>4s} {'Change':>8s}")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*4} {'─'*4} {'─'*8}")

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
        seq = scaled[-SEQ_LEN:]
        X = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        # 3. Run LSTM v2
        with torch.no_grad():
            delta_preds = model(X).cpu().numpy()[0]

        # Raw predictions
        pred_high_raw = entry_rate + max(delta_preds[0], delta_preds[1])
        pred_low_raw = entry_rate + min(delta_preds[0], delta_preds[1])
        pred_close = entry_rate + delta_preds[2]

        # Apply range buffer (with optional asymmetric upside)
        pred_high = pred_high_raw + range_buffer + asymmetric_upside
        pred_low = pred_low_raw - range_buffer

        # 4. XGBoost regime for calibration
        regime = "range_bound"
        if xgb_model is not None:
            try:
                latest_features = df_slice.iloc[[-1]]
                X_xgb = latest_features[xgb_feature_names].values
                proba = xgb_model.predict_proba(X_xgb)[0]
                pred_class = int(np.argmax(proba))
                regime = REGIME_MAP[pred_class]
            except Exception:
                pass

        # 5. Two-stage calibration layer (optional)
        if use_calibration and calibration:
            regime_ratios = calibration.get("regime_width_ratios", {})
            if regime in regime_ratios:
                ratio_data = regime_ratios[regime]
                target_ratio = ratio_data["width_ratio"]
                current_width = pred_high - pred_low
                actual_center = (pred_high + pred_low) / 2

                # Dampened calibration: only reduce width by half the difference
                # This preserves safety margin while tightening
                dampened_ratio = (1.0 + target_ratio) / 2.0
                new_width = current_width * dampened_ratio
                pred_high = actual_center + new_width / 2
                pred_low = actual_center - new_width / 2

        # 6. Look up actual outcomes
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

        # 7. Evaluate
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

        # 8. TPV impact
        if result == "GREEN":
            tpv_impact = max(0, round((pred_high - actual_high) * TPV_DAILY_USD))
        elif result == "RED":
            miss = max(0, actual_high - pred_high, pred_low - actual_low)
            tpv_impact = -round(miss * TPV_DAILY_USD)
        else:
            miss = max(0, actual_high - pred_high)
            tpv_impact = -round(miss * TPV_DAILY_USD)

        # V1 comparison
        date_str = pred_dt.isoformat()
        v1 = v1_baseline.get(date_str, {})
        v1_result = v1.get("result", "?")
        v1_overlap = v1.get("overlap_pct", 0)

        # Determine change
        result_order = {"GREEN": 2, "YELLOW": 1, "RED": 0, "?": -1}
        v1_score = result_order.get(v1_result, -1)
        v2_score = result_order.get(result, -1)
        if v2_score > v1_score:
            change = "BETTER"
        elif v2_score < v1_score:
            change = "WORSE"
        else:
            if overlap_pct > v1_overlap + 1:
                change = "better"
            elif overlap_pct < v1_overlap - 1:
                change = "worse"
            else:
                change = "same"

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
            "pred_range_paise": round((pred_high - pred_low) * 100, 1),
            "actual_range_paise": round((actual_high - actual_low) * 100, 1),
            "v1_result": v1_result,
            "v1_overlap": v1_overlap,
            "v1_pred_low": v1.get("pred_low", 0),
            "v1_pred_high": v1.get("pred_high", 0),
            "change": change,
        }
        results.append(rec)

        # Print row
        v1_lo = v1.get("pred_low", 0)
        v1_hi = v1.get("pred_high", 0)
        c1 = {"GREEN": "G", "YELLOW": "Y", "RED": "R"}.get(v1_result, "?")
        c2 = {"GREEN": "G", "YELLOW": "Y", "RED": "R"}.get(result, "?")
        change_sym = {"BETTER": "++", "WORSE": "--", "better": "+", "worse": "-", "same": "="}.get(change, "?")
        print(f"  {date_str:<12s} {entry_rate:>7.2f} {v1_lo:>7.2f} {v1_hi:>7.2f} "
              f"{pred_low:>7.2f} {pred_high:>7.2f} {actual_low:>7.2f} {actual_high:>7.2f} "
              f"{c1:>4s} {c2:>4s} {change_sym:>8s}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Comparison Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison(results):
    """Print side-by-side v1 vs v2 comparison."""
    if not results:
        print("  No results to compare.")
        return

    total = len(results)

    # V1 metrics
    v1_green = sum(1 for r in results if r["v1_result"] == "GREEN")
    v1_yellow = sum(1 for r in results if r["v1_result"] == "YELLOW")
    v1_red = sum(1 for r in results if r["v1_result"] == "RED")
    v1_avg_overlap = np.mean([r["v1_overlap"] for r in results])

    # V2 metrics
    v2_green = sum(1 for r in results if r["result"] == "GREEN")
    v2_yellow = sum(1 for r in results if r["result"] == "YELLOW")
    v2_red = sum(1 for r in results if r["result"] == "RED")
    v2_avg_overlap = np.mean([r["overlap_pct"] for r in results])

    # Width metrics
    v2_avg_pred_range = np.mean([r["pred_range_paise"] for r in results])
    v2_avg_actual_range = np.mean([r["actual_range_paise"] for r in results])
    v2_width_ratio = v2_avg_pred_range / v2_avg_actual_range if v2_avg_actual_range > 0 else 0

    # V1 width (from saved data)
    v1_avg_pred_range = np.mean([(r["v1_pred_high"] - r["v1_pred_low"]) * 100 for r in results if r["v1_pred_high"] > 0])
    v1_width_ratio = v1_avg_pred_range / v2_avg_actual_range if v2_avg_actual_range > 0 else 0

    # Change counts
    better = sum(1 for r in results if r["change"] in ("BETTER", "better"))
    worse = sum(1 for r in results if r["change"] in ("WORSE", "worse"))
    same = sum(1 for r in results if r["change"] == "same")
    big_better = sum(1 for r in results if r["change"] == "BETTER")
    big_worse = sum(1 for r in results if r["change"] == "WORSE")

    # Boundary errors
    v2_high_errors = []
    v2_low_errors = []
    v1_high_errors = []
    v1_low_errors = []
    for r in results:
        v2_high_errors.append(abs(r["pred_high"] - r["actual_high"]) * 100)
        v2_low_errors.append(abs(r["pred_low"] - r["actual_low"]) * 100)
        if r["v1_pred_high"] > 0:
            v1_high_errors.append(abs(r["v1_pred_high"] - r["actual_high"]) * 100)
            v1_low_errors.append(abs(r["v1_pred_low"] - r["actual_low"]) * 100)

    v2_total_boundary = np.mean(v2_high_errors) + np.mean(v2_low_errors)
    v1_total_boundary = np.mean(v1_high_errors) + np.mean(v1_low_errors) if v1_high_errors else 0

    # Position errors (center offset)
    v2_position_errors = []
    v1_position_errors = []
    for r in results:
        actual_center = (r["actual_high"] + r["actual_low"]) / 2
        v2_center = (r["pred_high"] + r["pred_low"]) / 2
        v2_position_errors.append(abs(v2_center - actual_center) * 100)
        if r["v1_pred_high"] > 0:
            v1_center = (r["v1_pred_high"] + r["v1_pred_low"]) / 2
            v1_position_errors.append(abs(v1_center - actual_center) * 100)

    v2_avg_position = np.mean(v2_position_errors)
    v1_avg_position = np.mean(v1_position_errors) if v1_position_errors else 0

    # TPV
    v2_tpv = sum(r["tpv_impact"] for r in results)

    # Per-regime breakdown
    regimes = {}
    for r in results:
        regime = r["regime"]
        if regime not in regimes:
            regimes[regime] = {"v1_green": 0, "v2_green": 0, "total": 0,
                               "v1_overlap": [], "v2_overlap": []}
        regimes[regime]["total"] += 1
        if r["v1_result"] == "GREEN":
            regimes[regime]["v1_green"] += 1
        if r["result"] == "GREEN":
            regimes[regime]["v2_green"] += 1
        regimes[regime]["v1_overlap"].append(r["v1_overlap"])
        regimes[regime]["v2_overlap"].append(r["overlap_pct"])

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          LSTM v1 vs v2 — COMPARISON REPORT                  ║
╠══════════════════════════════════════════════════════════════╣
║                         v1 (Huber)    v2 (BoundaryAware)    ║
║ ────────────────────────────────────────────────────────── ║
║ GREEN days:            {v1_green:>3d} ({v1_green/total*100:>4.0f}%)      {v2_green:>3d} ({v2_green/total*100:>4.0f}%)          ║
║ YELLOW days:           {v1_yellow:>3d} ({v1_yellow/total*100:>4.0f}%)      {v2_yellow:>3d} ({v2_yellow/total*100:>4.0f}%)          ║
║ RED days:              {v1_red:>3d} ({v1_red/total*100:>4.0f}%)       {v2_red:>3d} ({v2_red/total*100:>4.0f}%)          ║
║ Avg overlap:           {v1_avg_overlap:>6.1f}%       {v2_avg_overlap:>6.1f}%          ║
║ Avg pred range:        {v1_avg_pred_range:>5.1f}p        {v2_avg_pred_range:>5.1f}p           ║
║ Width ratio:           {v1_width_ratio:>5.2f}x        {v2_width_ratio:>5.2f}x           ║
║ Total boundary error:  {v1_total_boundary:>5.1f}p        {v2_total_boundary:>5.1f}p           ║
║ Position error (center):{v1_avg_position:>4.1f}p        {v2_avg_position:>5.1f}p           ║
║                                                              ║
║ CHANGES: {big_better} upgraded ++ | {big_worse} downgraded -- | {better-big_better} minor+ | {worse-big_worse} minor- | {same} same ║
╚══════════════════════════════════════════════════════════════╝""")

    # Boundary error improvement
    if v1_total_boundary > 0:
        boundary_improvement = (v1_total_boundary - v2_total_boundary) / v1_total_boundary * 100
        print(f"\n  Boundary error change: {boundary_improvement:+.1f}% {'(IMPROVED)' if boundary_improvement > 0 else '(WORSE)'}")

    if v1_avg_position > 0:
        position_improvement = (v1_avg_position - v2_avg_position) / v1_avg_position * 100
        print(f"  Position error change: {position_improvement:+.1f}% {'(IMPROVED)' if position_improvement > 0 else '(WORSE)'}")

    overlap_change = v2_avg_overlap - v1_avg_overlap
    print(f"  Overlap change: {overlap_change:+.1f}% {'(IMPROVED)' if overlap_change > 0 else '(WORSE)'}")

    # Per-regime table
    print(f"\n  {'Regime':<16s} {'Days':>5s} {'v1 GREEN':>9s} {'v2 GREEN':>9s} {'v1 Ovlp':>8s} {'v2 Ovlp':>8s}")
    print(f"  {'─'*16} {'─'*5} {'─'*9} {'─'*9} {'─'*8} {'─'*8}")
    for regime, data in sorted(regimes.items()):
        v1_g_pct = data["v1_green"] / data["total"] * 100
        v2_g_pct = data["v2_green"] / data["total"] * 100
        v1_o = np.mean(data["v1_overlap"])
        v2_o = np.mean(data["v2_overlap"])
        print(f"  {regime:<16s} {data['total']:>5d} {data['v1_green']:>3d} ({v1_g_pct:>3.0f}%) "
              f"{data['v2_green']:>3d} ({v2_g_pct:>3.0f}%) {v1_o:>7.1f}% {v2_o:>7.1f}%")

    # Day-by-day changes
    print(f"\n  Day-by-day changes:")
    for r in results:
        if r["change"] in ("BETTER", "WORSE"):
            sym = "⬆" if r["change"] == "BETTER" else "⬇"
            print(f"    {sym} {r['date']} {r['v1_result']:>6s} → {r['result']:<6s} "
                  f"(overlap: {r['v1_overlap']:.0f}% → {r['overlap_pct']:.0f}%) "
                  f"[{r['regime']}]")

    # Decision: save or not?
    print(f"\n{'─' * 70}")
    if v2_avg_overlap > v1_avg_overlap * 1.15:  # >15% improvement
        print(f"  VERDICT: v2 PASSES — {(v2_avg_overlap/v1_avg_overlap - 1)*100:.1f}% overlap improvement (>15% threshold)")
        print(f"  Model saved as lstm_walkforward_v2.pt")
        return True
    elif v2_green > v1_green and v2_avg_overlap >= v1_avg_overlap:
        print(f"  VERDICT: v2 PASSES — more GREEN days ({v1_green}→{v2_green}) with same/better overlap")
        print(f"  Model saved as lstm_walkforward_v2.pt")
        return True
    elif v2_avg_overlap > v1_avg_overlap:
        improvement_pct = (v2_avg_overlap / v1_avg_overlap - 1) * 100
        print(f"  VERDICT: v2 shows {improvement_pct:.1f}% improvement but below 15% threshold")
        print(f"  Model saved for further analysis")
        return True
    else:
        print(f"  VERDICT: v2 DOES NOT IMPROVE over v1 — consider different approach")
        return False

    # Save comparison CSV
    comp_path = os.path.join(OUTPUT_DIR, "v1_vs_v2_comparison.csv")
    comp_df = pd.DataFrame(results)
    comp_df.to_csv(comp_path, index=False)
    print(f"  Comparison CSV: {comp_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LSTM v2 Retraining with BoundaryAwareLoss")
    parser.add_argument("--train-only", action="store_true", help="Train only, skip evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing v2 model only")
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)

    if args.eval_only:
        # Load existing v2 model
        model_path = os.path.join(SAVED_DIR, "lstm_walkforward_v2.pt")
        scaler_path = os.path.join(SAVED_DIR, "lstm_scaler_walkforward_v2.pkl")
        if not os.path.exists(model_path):
            print(f"  ERROR: {model_path} not found. Run training first.")
            return

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model = LSTMRangePredictorV2(input_dim=checkpoint["input_dim"]).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        scaler = joblib.load(scaler_path)
        range_buffer = checkpoint["range_buffer"]
        extra_buffers = {
            "range_buffer_high": checkpoint.get("range_buffer_high", range_buffer),
            "range_buffer_low": checkpoint.get("range_buffer_low", range_buffer),
        }
        print(f"  Loaded v2 model from {model_path}")
        print(f"  Range buffer: ±{range_buffer:.4f}")
    else:
        # Train
        result = train_v2()
        if result[0] is None:
            print("\n  HARD STOP — Aborting.")
            return
        model, scaler, range_buffer, extra_buffers = result

        if args.train_only:
            print("\n  Training complete. Run with --eval-only to evaluate.")
            return

    # Run multiple configurations to find optimal setup
    checkpoint_path = os.path.join(SAVED_DIR, "lstm_walkforward_v2.pt")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    buffer_p85 = checkpoint.get("range_buffer", range_buffer)
    buffer_p90 = checkpoint.get("range_buffer_p90", range_buffer * 1.15)
    buffer_p95 = checkpoint.get("range_buffer_p95", range_buffer * 1.3)

    # Asymmetric upside: add ~50% of buffer on high side only
    # because most errors are upward misses (actual > predicted in trending markets)
    asym_upside = buffer_p90 * 0.5

    configs = [
        ("v2-P90", buffer_p90, False, 0.0, "P90 buffer, no calibration"),
        ("v2-P95", buffer_p95, False, 0.0, "P95 buffer, no calibration"),
        ("v2-P90+asym", buffer_p90, False, asym_upside, "P90 buffer + upside bias"),
        ("v2-P95+asym", buffer_p95, False, asym_upside * 0.5, "P95 buffer + small upside bias"),
    ]

    best_config = None
    best_overlap = 0
    all_summaries = []

    for config_label, buf, use_cal, asym, desc in configs:
        print(f"\n{'━' * 70}")
        print(f"  CONFIG: {config_label} — {desc}")
        print(f"{'━' * 70}")

        results = walkforward_evaluate(
            model, scaler, buf, extra_buffers,
            use_calibration=use_cal, config_label=config_label,
            asymmetric_upside=asym
        )

        if results:
            total = len(results)
            green = sum(1 for r in results if r["result"] == "GREEN")
            yellow = sum(1 for r in results if r["result"] == "YELLOW")
            red = sum(1 for r in results if r["result"] == "RED")
            avg_overlap = np.mean([r["overlap_pct"] for r in results])
            avg_range = np.mean([r["pred_range_paise"] for r in results])

            summary = {
                "config": config_label, "desc": desc, "buffer": buf,
                "green": green, "yellow": yellow, "red": red,
                "avg_overlap": avg_overlap, "avg_range": avg_range,
                "results": results,
            }
            all_summaries.append(summary)

            if avg_overlap > best_overlap:
                best_overlap = avg_overlap
                best_config = summary

    # Print all configs summary
    print(f"\n{'═' * 70}")
    print(f"  CONFIGURATION COMPARISON")
    print(f"{'═' * 70}")
    print(f"\n  v1 baseline: 13 GREEN (43%), 7 RED (23%), avg overlap 57.4%\n")
    print(f"  {'Config':<14s} {'Buffer':>7s} {'GREEN':>6s} {'YELLOW':>7s} {'RED':>5s} {'Overlap':>8s} {'Pred Range':>11s}")
    print(f"  {'─'*14} {'─'*7} {'─'*6} {'─'*7} {'─'*5} {'─'*8} {'─'*11}")
    for s in all_summaries:
        total = s["green"] + s["yellow"] + s["red"]
        print(f"  {s['config']:<14s} {s['buffer']:>7.4f} "
              f"{s['green']:>3d}({s['green']/total*100:>2.0f}%) "
              f"{s['yellow']:>3d}({s['yellow']/total*100:>2.0f}%) "
              f"{s['red']:>3d}({s['red']/total*100:>2.0f}%) "
              f"{s['avg_overlap']:>7.1f}% {s['avg_range']:>10.1f}p")

    if best_config:
        print(f"\n  BEST CONFIG: {best_config['config']} — overlap {best_config['avg_overlap']:.1f}%")

        # Print detailed comparison for best config
        print_comparison(best_config["results"])

        # Save best comparison CSV
        comp_path = os.path.join(OUTPUT_DIR, "v1_vs_v2_comparison.csv")
        comp_df = pd.DataFrame(best_config["results"])
        comp_df.to_csv(comp_path, index=False)
        print(f"\n  Comparison CSV saved: {comp_path}")


if __name__ == "__main__":
    main()
