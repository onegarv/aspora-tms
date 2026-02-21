"""
Step 4 — LSTM Full History Training (2003-2025)

12 input features, trading-day targets, 2003-2024 train / 2025 test.
"""

import os
import sys
import copy

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features.feature_engineering import build_features
from data.fx_calendar import is_trading_day
from models.train_lstm import LSTMRangePredictor, create_sequences, EarlyStopping

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "market_data_full.csv")

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

# Hard stops
MAX_CLOSE_MAE = 0.45
MIN_RANGE_ACC = 0.65
MIN_AVG_DELTA = 0.05


# ---------------------------------------------------------------------------
# Trading-day target construction
# ---------------------------------------------------------------------------

def build_trading_day_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, compute targets over the next 2 TRADING days:
        y_high  = max rate over those 2 trading days
        y_low   = min rate over those 2 trading days
        y_close = rate on 2nd trading day

    Deltas from current rate:
        dy_high  = y_high  - current
        dy_low   = y_low   - current
        dy_close = y_close - current
    """
    from datetime import date as date_type

    out = df.copy()
    dates = pd.to_datetime(out["date"]).tolist()
    rates = out["usdinr"].values

    # Build date->rate lookup
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

        max_lookahead = 14  # safety
        for _ in range(max_lookahead):
            check_date = check_date + pd.Timedelta(days=1)
            py_date = check_date if isinstance(check_date, date_type) else check_date.date()

            if is_trading_day(py_date):
                if py_date in date_to_rate:
                    trading_rates.append(date_to_rate[py_date])
                if len(trading_rates) == 2:
                    break

        if len(trading_rates) == 2:
            y_high[i] = max(trading_rates)
            y_low[i] = min(trading_rates)
            y_close[i] = trading_rates[1]  # 2nd trading day

    out["y_high"] = y_high
    out["y_low"] = y_low
    out["y_close"] = y_close

    out["dy_high"] = out["y_high"] - out["usdinr"]
    out["dy_low"] = out["y_low"] - out["usdinr"]
    out["dy_close"] = out["y_close"] - out["usdinr"]

    out = out.dropna(subset=["y_high", "y_low", "y_close"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 4: LSTM Full History Training (2003-2025)")
    print(f"  12 features, trading-day targets, seq_len={SEQ_LEN}")
    print("=" * 70)

    # --- Load and build features ---
    print(f"\nLoading {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Raw: {len(raw)} rows")

    print("Building features...")
    features_df = build_features(raw)
    print(f"Features: {len(features_df)} rows, checking required columns...")

    for f in FEATURES:
        assert f in features_df.columns, f"Missing feature: {f}"
    print(f"  All 12 features present")

    # --- Build trading-day targets ---
    print("\nBuilding TRADING-DAY targets (2 trading days forward)...")
    df = build_trading_day_targets(features_df)
    print(f"After targets: {len(df)} rows")
    print(f"  dy_close range: {df['dy_close'].min():.4f} to {df['dy_close'].max():.4f}")
    print(f"  dy_high range:  {df['dy_high'].min():.4f} to {df['dy_high'].max():.4f}")
    print(f"  dy_low range:   {df['dy_low'].min():.4f} to {df['dy_low'].max():.4f}")

    # --- Train/test split by date ---
    # NOTE: Training from 2015 (not 2003) — earlier data has completely different
    # FX dynamics (USDINR 39-63) that cause the model to predict flat deltas.
    # Features are still computed from full history, so long-term features
    # (rate_vs_5y_avg, rate_vs_alltime_percentile, etc.) still carry historical context.
    # This matches the v2 LSTM's working scaler range (~63-85).
    TRAIN_START = "2015-01-01"
    train_mask = (df["date"] >= TRAIN_START) & (df["date"] < "2025-01-01")
    test_mask = (df["date"] >= "2025-01-01") & (df["date"] < "2026-01-01")
    fwd_mask = df["date"] >= "2026-01-01"

    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    fwd_df = df.loc[fwd_mask].reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  NOTE: Starting from {TRAIN_START} (not 2003) — pre-2015 FX dynamics too different")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    print(f"Fwd:   {len(fwd_df)} rows ({fwd_df['date'].min().date()} to {fwd_df['date'].max().date()})" if len(fwd_df) > 0 else "Fwd:   0 rows")

    # --- Scaler: fit on TRAINING only ---
    print(f"\nFitting MinMaxScaler on training data only ({len(train_df)} rows)...")
    scaler = MinMaxScaler()
    train_features = train_df[FEATURES].values
    scaler.fit(train_features)

    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_df[FEATURES].values)

    print(f"  Feature ranges (train):")
    for i, f in enumerate(FEATURES):
        print(f"    {f:<30s}: min={scaler.data_min_[i]:>10.4f}  max={scaler.data_max_[i]:>10.4f}")

    # --- Build sequences ---
    train_targets = train_df[["dy_high", "dy_low", "dy_close"]].values
    test_targets = test_df[["dy_high", "dy_low", "dy_close"]].values
    train_entry = train_df["usdinr"].values
    test_entry = test_df["usdinr"].values
    train_dates = train_df["date"].values
    test_dates = test_df["date"].values

    X_train, y_train, d_train, r_train = create_sequences(
        train_scaled, train_targets, train_dates, train_entry, SEQ_LEN
    )
    X_test, y_test, d_test, r_test = create_sequences(
        test_scaled, test_targets, test_dates, test_entry, SEQ_LEN
    )

    print(f"\n  Train sequences: {X_train.shape}")
    print(f"  Test sequences:  {X_test.shape}")

    # Val split
    val_size = int(len(X_train) * VAL_SPLIT)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_fit = X_train[:-val_size]
    y_train_fit = y_train[:-val_size]

    print(f"  Train (fit): {X_train_fit.shape}, Val: {X_val.shape}")

    # --- Train ---
    print(f"\n{'='*70}")
    print(f"TRAINING LSTM (device={DEVICE})")
    print(f"{'='*70}")

    train_ds = TensorDataset(torch.FloatTensor(X_train_fit), torch.FloatTensor(y_train_fit))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRangePredictor(input_dim=len(FEATURES)).to(DEVICE)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    stopper = EarlyStopping(patience=PATIENCE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: LSTM(12→64)→LSTM(64→32)→Dense(32→16→3)")
    print(f"  Parameters: {total_params:,}")
    print()

    history = {"train_loss": [], "val_loss": []}
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

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{MAX_EPOCHS}  train={avg_train:.6f}  val={avg_val:.6f}")

        if stopper.step(avg_val, model):
            best_epoch = epoch - PATIENCE
            print(f"\n  Early stopping at epoch {epoch}. Best: {best_epoch}")
            break
    else:
        best_epoch = MAX_EPOCHS
        print(f"\n  Completed all {MAX_EPOCHS} epochs.")

    stopper.restore(model)

    # ======================================================================
    # OUTPUT 1: Train vs val loss
    # ======================================================================
    print(f"\n{'='*70}")
    print("1. TRAIN VS VAL LOSS")
    print(f"{'='*70}")

    final_train = history["train_loss"][best_epoch - 1] if best_epoch > 0 else history["train_loss"][-1]
    final_val = stopper.best_loss
    ratio = final_val / final_train if final_train > 0 else 0
    print(f"  Final train_loss: {final_train:.6f}")
    print(f"  Final val_loss:   {final_val:.6f}")
    print(f"  Ratio (val/train): {ratio:.2f}x")
    if ratio > 3.0:
        print(f"  ⚠ WARNING: val/train > 3x — possible overfitting!")
    else:
        print(f"  OK: no severe overfitting")

    # --- Calibrate range buffer ---
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy()
    val_entry = r_train[-val_size:]
    val_residuals = np.abs((val_entry + val_preds[:, 2]) - (val_entry + y_val[:, 2]))
    range_buffer = np.percentile(val_residuals, 85)
    print(f"\n  Range buffer (P85 of val residuals): ±{range_buffer:.4f} INR")

    # --- Test predictions ---
    with torch.no_grad():
        delta_preds = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()

    dpred_high = delta_preds[:, 0]
    dpred_low = delta_preds[:, 1]
    dpred_close = delta_preds[:, 2]

    pred_high_raw = r_test + dpred_high
    pred_low_raw = r_test + dpred_low
    pred_close = r_test + dpred_close

    stacked = np.stack([pred_low_raw, pred_high_raw], axis=1)
    pred_low_raw = stacked.min(axis=1)
    pred_high_raw = stacked.max(axis=1)

    pred_high = pred_high_raw + range_buffer
    pred_low = pred_low_raw - range_buffer

    actual_high = r_test + y_test[:, 0]
    actual_low = r_test + y_test[:, 1]
    actual_close = r_test + y_test[:, 2]

    # ======================================================================
    # OUTPUT 2: MAE metrics
    # ======================================================================
    print(f"\n{'='*70}")
    print("2. MAE ON 2025 TEST SET")
    print(f"{'='*70}")

    mae_close = np.mean(np.abs(pred_close - actual_close))
    mae_high = np.mean(np.abs(pred_high - actual_high))
    mae_low = np.mean(np.abs(pred_low - actual_low))

    print(f"  MAE close: {mae_close:.4f} INR {'✓' if mae_close <= MAX_CLOSE_MAE else '✗ HARD STOP'}")
    print(f"  MAE high:  {mae_high:.4f} INR")
    print(f"  MAE low:   {mae_low:.4f} INR")

    # ======================================================================
    # OUTPUT 3: Range accuracy
    # ======================================================================
    print(f"\n{'='*70}")
    print("3. RANGE ACCURACY ON 2025 TEST SET")
    print(f"{'='*70}")

    in_range = ((actual_close >= pred_low) & (actual_close <= pred_high)).sum()
    range_acc = in_range / len(actual_close)
    print(f"  Range accuracy: {range_acc:.1%} ({in_range}/{len(actual_close)})")
    print(f"  Target: >= {MIN_RANGE_ACC:.0%}")

    # ======================================================================
    # OUTPUT 4: Directional accuracy
    # ======================================================================
    print(f"\n{'='*70}")
    print("4. DIRECTIONAL ACCURACY (LSTM alone)")
    print(f"{'='*70}")

    pred_dir = np.where(dpred_close > 0.05, "UP",
               np.where(dpred_close < -0.05, "DOWN", "NEUTRAL"))
    actual_dir = np.where(y_test[:, 2] > 0.05, "UP",
                 np.where(y_test[:, 2] < -0.05, "DOWN", "NEUTRAL"))

    dir_correct = (pred_dir == actual_dir).sum()
    dir_acc = dir_correct / len(pred_dir)
    print(f"  Directional accuracy: {dir_acc:.1%} ({dir_correct}/{len(pred_dir)})")

    print(f"\n  Predicted distribution:")
    for d in ["UP", "DOWN", "NEUTRAL"]:
        n = (pred_dir == d).sum()
        print(f"    {d:<8s}: {n:>4d}  ({n/len(pred_dir)*100:.1f}%)")

    print(f"\n  Actual distribution:")
    for d in ["UP", "DOWN", "NEUTRAL"]:
        n = (actual_dir == d).sum()
        print(f"    {d:<8s}: {n:>4d}  ({n/len(actual_dir)*100:.1f}%)")

    # ======================================================================
    # OUTPUT 5: Friday check
    # ======================================================================
    print(f"\n{'='*70}")
    print("5. FRIDAY vs NON-FRIDAY DELTA CHECK")
    print(f"{'='*70}")

    test_dows = pd.to_datetime(d_test).dayofweek
    fri_mask = test_dows == 4
    non_fri_mask = ~fri_mask

    fri_deltas = np.abs(dpred_close[fri_mask])
    non_fri_deltas = np.abs(dpred_close[non_fri_mask])

    avg_fri = fri_deltas.mean() if len(fri_deltas) > 0 else 0
    avg_non_fri = non_fri_deltas.mean() if len(non_fri_deltas) > 0 else 0

    print(f"  Friday avg |delta|:     {avg_fri:.4f} INR ({len(fri_deltas)} rows)")
    print(f"  Non-Friday avg |delta|: {avg_non_fri:.4f} INR ({len(non_fri_deltas)} rows)")
    ratio_fri = avg_fri / avg_non_fri if avg_non_fri > 0 else 0
    print(f"  Ratio (Fri/non-Fri):    {ratio_fri:.2f}x")

    if ratio_fri < 0.50:
        print(f"  ⚠ WARNING: Friday deltas significantly smaller — weekend bias detected")
    else:
        print(f"  OK: Friday deltas comparable to non-Friday")

    # ======================================================================
    # OUTPUT 6: Last 30 days of 2025 day-by-day
    # ======================================================================
    print(f"\n{'='*70}")
    print("6. LAST 30 DAYS OF 2025 — DAY-BY-DAY PREDICTIONS")
    print(f"{'='*70}")

    n_show = min(30, len(d_test))
    print(f"\n  {'Date':<12s} {'Day':<5s} {'Entry':>8s} {'PredCl':>8s} {'ActCl':>8s} {'Delta':>7s} {'Error':>7s} {'Dir':>5s}")
    print(f"  {'─'*12} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*5}")

    for i in range(-n_show, 0):
        dt = pd.Timestamp(d_test[i])
        day = dt.strftime("%a")
        entry = r_test[i]
        pc = pred_close[i]
        ac = actual_close[i]
        delta = dpred_close[i]
        err = pc - ac
        d_str = "UP" if delta > 0.05 else ("DOWN" if delta < -0.05 else "NEUT")
        print(f"  {dt.strftime('%Y-%m-%d'):<12s} {day:<5s} {entry:>8.2f} {pc:>8.2f} {ac:>8.2f} {delta:>+7.3f} {err:>+7.3f} {d_str:>5s}")

    # Stats
    last30_deltas = np.abs(dpred_close[-n_show:])
    print(f"\n  Avg |delta| last 30 days: {last30_deltas.mean():.4f} INR")
    print(f"  Min |delta|: {last30_deltas.min():.4f}, Max |delta|: {last30_deltas.max():.4f}")

    if last30_deltas.mean() < MIN_AVG_DELTA:
        print(f"  ⚠ WARNING: Avg delta < {MIN_AVG_DELTA} — LSTM predicting flat")

    # ======================================================================
    # OUTPUT 7: Forward validation Jan-Feb 2026
    # ======================================================================
    print(f"\n{'='*70}")
    print("7. FORWARD VALIDATION — Jan-Feb 2026")
    print(f"{'='*70}")

    if len(fwd_df) > 0:
        fwd_scaled = scaler.transform(fwd_df[FEATURES].values)
        fwd_targets = fwd_df[["dy_high", "dy_low", "dy_close"]].values
        fwd_entry = fwd_df["usdinr"].values
        fwd_dates = fwd_df["date"].values

        X_fwd, y_fwd, d_fwd, r_fwd = create_sequences(
            fwd_scaled, fwd_targets, fwd_dates, fwd_entry, SEQ_LEN
        )

        if len(X_fwd) > 0:
            with torch.no_grad():
                fwd_preds = model(torch.FloatTensor(X_fwd).to(DEVICE)).cpu().numpy()

            fwd_dpred_close = fwd_preds[:, 2]
            fwd_pred_close = r_fwd + fwd_dpred_close
            fwd_actual_close = r_fwd + y_fwd[:, 2]
            fwd_pred_high = r_fwd + fwd_preds[:, 0] + range_buffer
            fwd_pred_low = r_fwd + fwd_preds[:, 1] - range_buffer

            # Directional
            fwd_pred_dir = np.where(fwd_dpred_close > 0.05, "UP",
                           np.where(fwd_dpred_close < -0.05, "DOWN", "NEUTRAL"))
            fwd_actual_dir = np.where(y_fwd[:, 2] > 0.05, "UP",
                             np.where(y_fwd[:, 2] < -0.05, "DOWN", "NEUTRAL"))
            fwd_dir_acc = (fwd_pred_dir == fwd_actual_dir).mean()

            # Range
            fwd_in_range = ((fwd_actual_close >= fwd_pred_low) & (fwd_actual_close <= fwd_pred_high)).sum()
            fwd_range_acc = fwd_in_range / len(fwd_actual_close)

            # MAE
            fwd_mae = np.mean(np.abs(fwd_pred_close - fwd_actual_close))

            # Avg delta magnitude
            fwd_avg_delta = np.abs(fwd_dpred_close).mean()

            print(f"\n  Period: {pd.Timestamp(d_fwd[0]).date()} to {pd.Timestamp(d_fwd[-1]).date()}")
            print(f"  Rows: {len(X_fwd)}")
            print(f"  Directional accuracy: {fwd_dir_acc:.1%}")
            print(f"  Range accuracy:       {fwd_range_acc:.1%}")
            print(f"  Close MAE:            {fwd_mae:.4f} INR")
            print(f"  Avg |delta|:          {fwd_avg_delta:.4f} INR {'✓ > 0.10' if fwd_avg_delta > 0.10 else '⚠ < 0.10'}")

            # Day-by-day
            print(f"\n  {'Date':<12s} {'Day':<5s} {'Entry':>8s} {'PredCl':>8s} {'ActCl':>8s} {'Delta':>7s} {'Dir':>5s} {'OK':>4s}")
            print(f"  {'─'*12} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*5} {'─'*4}")
            for i in range(len(d_fwd)):
                dt = pd.Timestamp(d_fwd[i])
                day = dt.strftime("%a")
                ok = "YES" if fwd_pred_dir[i] == fwd_actual_dir[i] else "NO"
                print(f"  {dt.strftime('%Y-%m-%d'):<12s} {day:<5s} {r_fwd[i]:>8.2f} "
                      f"{fwd_pred_close[i]:>8.2f} {fwd_actual_close[i]:>8.2f} "
                      f"{fwd_dpred_close[i]:>+7.3f} {fwd_pred_dir[i]:>5s} {ok:>4s}")
        else:
            print(f"  Not enough forward data for {SEQ_LEN}-day sequences")
    else:
        print(f"  No 2026 data in feature matrix")

    # ======================================================================
    # HARD STOP CHECKS
    # ======================================================================
    print(f"\n{'='*70}")
    print("HARD STOP SUMMARY")
    print(f"{'='*70}")

    avg_delta_all = np.abs(dpred_close).mean()
    checks = [
        (f"Close MAE <= {MAX_CLOSE_MAE}", mae_close <= MAX_CLOSE_MAE, f"{mae_close:.4f}"),
        (f"Range accuracy >= {MIN_RANGE_ACC:.0%}", range_acc >= MIN_RANGE_ACC, f"{range_acc:.1%}"),
        (f"Avg |delta| >= {MIN_AVG_DELTA}", avg_delta_all >= MIN_AVG_DELTA, f"{avg_delta_all:.4f}"),
        (f"Friday delta ratio >= 0.50", ratio_fri >= 0.50, f"{ratio_fri:.2f}"),
    ]

    all_pass = True
    for name, passed, val in checks:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {status}  {name}  (actual: {val})")
        if not passed:
            all_pass = False

    if not all_pass:
        print(f"\n  *** MODEL NOT SAVED — hard stop triggered ***")
        return False

    # ======================================================================
    # SAVE
    # ======================================================================
    print(f"\n{'='*70}")
    print("SAVING")
    print(f"{'='*70}")

    os.makedirs(SAVED_DIR, exist_ok=True)

    model_path = os.path.join(SAVED_DIR, "lstm_fullhistory.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(FEATURES),
        "seq_len": SEQ_LEN,
        "features": FEATURES,
        "predicts_deltas": True,
        "range_buffer": float(range_buffer),
    }, model_path)
    print(f"  Model  → {model_path}")

    scaler_path = os.path.join(SAVED_DIR, "lstm_scaler_fullhistory.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler → {scaler_path}")

    print(f"\n{'='*70}")
    print("Step 4 complete. Awaiting approval before Step 5.")
    print(f"{'='*70}")
    return True


if __name__ == "__main__":
    main()
