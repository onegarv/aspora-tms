"""
Step 3 — Intraday LSTM Momentum Classifier (PyTorch)

Predicts 24-hour (6 × 4h bars) directional movement:
    0 = DOWN  (rate falls > 0.10 INR)
    1 = NEUTRAL (movement within ±0.10 INR)
    2 = UP    (rate rises > 0.10 INR)

Input:  48-bar sequences (8 trading days) of 20 intraday features
Output: 3-class softmax

Architecture:
    LSTM(128, return_sequences=True) → Dropout(0.3) →
    LSTM(64) → Dropout(0.3) →
    Dense(32, relu) → Dense(3, softmax)

Train/test split: chronological at 2025-11-01
Class balancing: sklearn compute_class_weight → CrossEntropyLoss weights
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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
SAVED_DIR = os.path.join(BASE_DIR, "saved")
os.makedirs(SAVED_DIR, exist_ok=True)

SEQ_LEN = 24              # 24 bars = 4 trading days (more appropriate for 24h prediction)
FUTURE_BARS = 6            # predict next 6 bars = 24 hours
THRESHOLD = 0.10           # ±0.10 INR for UP/DOWN classification
SPLIT_DATE = "2025-11-01"  # chronological split

# Training hyperparameters
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LR = 0.001
GRAD_CLIP = 1.0
MAX_CLASS_WEIGHT = 1.5  # cap to prevent mode collapse
VAL_SPLIT = 0.1  # 10% of training data for validation

# Feature columns — 19 stationary features (Close dropped: non-stationary,
# causes distribution shift between train period ~82-88 and test period ~88-91)
FEATURE_COLS = [
    "ret_4h", "ret_8h", "ret_24h", "ret_48h",
    "bar_range", "range_ma_5", "is_high_vol_bar",
    "last_4h_direction", "momentum_consistency_24h",
    "rate_vs_24h_high", "rate_vs_24h_low", "rate_vs_48h_avg",
    "rsi_4h",
    "is_asian", "is_london", "is_ny",
    "hour", "day_of_week", "session_num",
]

# Class labels
CLASS_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class IntradayLSTM(nn.Module):
    """
    LSTM(128) → LayerNorm → Dropout(0.3) → LSTM(64) → LayerNorm →
    Dropout(0.3) → Dense(32, relu) → Dense(3)

    Xavier init on linear layers, orthogonal init on LSTM weights.
    """
    def __init__(self, input_dim=20):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.ln1 = nn.LayerNorm(128)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.ln2 = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for LSTM (helps gradient flow)
        for name, param in self.lstm1.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for name, param in self.lstm2.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Xavier for linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        out, _ = self.lstm1(x)          # (batch, seq, 128)
        out = self.ln1(out)
        out = self.drop1(out)
        out, _ = self.lstm2(out)        # (batch, seq, 64)
        out = out[:, -1, :]             # last time step only
        out = self.ln2(out)
        out = self.drop2(out)
        out = self.relu(self.fc1(out))  # (batch, 32)
        out = self.fc2(out)             # (batch, 3) — raw logits
        return out


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame) -> np.ndarray:
    """
    For each bar i, compute target class from Close[i+6] - Close[i]:
        0 = DOWN:    delta < -THRESHOLD
        1 = NEUTRAL: |delta| <= THRESHOLD
        2 = UP:      delta > +THRESHOLD
    Returns array of class labels (last FUTURE_BARS rows will be NaN/dropped).
    """
    close = df["Close"].values
    n = len(close)
    targets = np.full(n, np.nan)

    for i in range(n - FUTURE_BARS):
        delta = close[i + FUTURE_BARS] - close[i]
        if delta > THRESHOLD:
            targets[i] = 2   # UP
        elif delta < -THRESHOLD:
            targets[i] = 0   # DOWN
        else:
            targets[i] = 1   # NEUTRAL

    return targets


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def create_sequences(features_scaled: np.ndarray, targets: np.ndarray,
                     timestamps: np.ndarray):
    """
    Create overlapping sequences of length SEQ_LEN.
    Only includes samples where target is not NaN.
    """
    X, y, ts = [], [], []
    for i in range(SEQ_LEN, len(features_scaled)):
        if np.isnan(targets[i]):
            continue
        X.append(features_scaled[i - SEQ_LEN:i])
        y.append(int(targets[i]))
        ts.append(timestamps[i])
    return np.array(X), np.array(y), np.array(ts)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = PATIENCE):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_pipeline():
    """Full intraday LSTM training pipeline."""

    print("=" * 60)
    print("  INTRADAY LSTM — 3-Class Momentum Classifier")
    print("=" * 60)
    print()

    # --- Load features ---
    data_path = os.path.join(PROJECT_DIR, "data", "intraday_features.csv")
    df = pd.read_csv(data_path, index_col="Datetime", parse_dates=True)
    print(f"Loaded {len(df)} rows from intraday_features.csv")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # --- Build targets ---
    targets = build_targets(df)
    valid_mask = ~np.isnan(targets)
    print(f"Valid target rows: {valid_mask.sum()} (dropped {(~valid_mask).sum()} future-lookahead rows)")

    # --- Overall class distribution ---
    valid_targets = targets[valid_mask].astype(int)
    print(f"\nOverall class distribution:")
    for cls in [0, 1, 2]:
        count = (valid_targets == cls).sum()
        pct = count / len(valid_targets) * 100
        flag = " ⚠ BELOW 15%" if pct < 15 else ""
        print(f"  {CLASS_NAMES[cls]:<8s}: {count:>5d} ({pct:>5.1f}%){flag}")

    # --- Chronological split ---
    split_ts = pd.Timestamp(SPLIT_DATE, tz=df.index.tz)
    train_mask = df.index < split_ts
    test_mask = df.index >= split_ts

    print(f"\nChronological split at {SPLIT_DATE}:")
    print(f"  Train: {train_mask.sum()} bars  ({df.index[train_mask][0].date()} to {df.index[train_mask][-1].date()})")
    print(f"  Test:  {test_mask.sum()} bars  ({df.index[test_mask][0].date()} to {df.index[test_mask][-1].date()})")

    # --- Scale features ---
    # StandardScaler (not MinMax) — more robust to distribution shift
    # between train period (Close ~82-88) and test period (Close ~88-91)
    features = df[FEATURE_COLS].values
    timestamps = df.index.values

    scaler = StandardScaler()
    # Fit on training data only
    scaler.fit(features[train_mask])
    features_scaled = scaler.transform(features)

    # --- Create sequences ---
    X, y, ts = create_sequences(features_scaled, targets, timestamps)
    print(f"\nTotal sequences: {len(X)} (seq_len={SEQ_LEN})")

    # Split sequences by timestamp
    split_ts_np = np.datetime64(SPLIT_DATE)
    train_idx = ts < split_ts_np
    test_idx = ts >= split_ts_np

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    ts_test = ts[test_idx]

    print(f"  Train sequences: {len(X_train)}")
    print(f"  Test sequences:  {len(X_test)}")

    # --- 1. Class distribution in train and test ---
    print(f"\n{'─' * 50}")
    print("1. CLASS DISTRIBUTION")
    print(f"{'─' * 50}")
    for label, name in CLASS_NAMES.items():
        tr_count = (y_train == label).sum()
        tr_pct = tr_count / len(y_train) * 100
        te_count = (y_test == label).sum()
        te_pct = te_count / len(y_test) * 100
        tr_flag = " ⚠" if tr_pct < 15 else ""
        te_flag = " ⚠" if te_pct < 15 else ""
        print(f"  {name:<8s}:  Train {tr_count:>5d} ({tr_pct:>5.1f}%){tr_flag}  |  Test {te_count:>5d} ({te_pct:>5.1f}%){te_flag}")

    # --- Compute class weights ---
    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2]), y=y_train
    )
    print(f"\n  Class weights (raw):      DOWN={class_weights[0]:.3f}, NEUTRAL={class_weights[1]:.3f}, UP={class_weights[2]:.3f}")

    # Cap weights to prevent mode collapse (raw DOWN=1.9 caused all-DOWN predictions)
    class_weights = np.clip(class_weights, 0.5, MAX_CLASS_WEIGHT)
    print(f"  Class weights (capped):   DOWN={class_weights[0]:.3f}, NEUTRAL={class_weights[1]:.3f}, UP={class_weights[2]:.3f}")

    # --- Validation split from training data ---
    val_size = int(len(X_train) * VAL_SPLIT)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    print(f"\n  Training: {len(X_train_final)} sequences")
    print(f"  Validation: {len(X_val)} sequences")
    print(f"  Test: {len(X_test)} sequences")

    # --- Build DataLoaders ---
    train_ds = TensorDataset(
        torch.tensor(X_train_final, dtype=torch.float32),
        torch.tensor(y_train_final, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- Build model ---
    model = IntradayLSTM(input_dim=len(FEATURE_COLS)).to(DEVICE)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    print(f"\n  Device: {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Gradient clip: {GRAD_CLIP}")
    print(f"  Label smoothing: 0.1")
    print(f"  LR scheduler: ReduceLROnPlateau(factor=0.5, patience=5)")
    print(f"\nTraining...")

    # --- Training loop ---
    early_stop = EarlyStopping(patience=PATIENCE)
    train_losses = []
    val_losses = []

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                n_val += 1
        val_loss = val_loss / n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Step LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:>3d}/{MAX_EPOCHS}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.6f}")

        if early_stop.step(val_loss, model):
            print(f"  Early stopping at epoch {epoch + 1} (patience={PATIENCE})")
            break

    # Restore best weights
    early_stop.restore(model)
    best_epoch = len(train_losses) - early_stop.counter
    print(f"  Best model from epoch {best_epoch}")

    # --- 2. Training curve ---
    print(f"\n{'─' * 50}")
    print("2. TRAINING CURVE")
    print(f"{'─' * 50}")
    final_train = train_losses[best_epoch - 1]
    final_val = val_losses[best_epoch - 1]
    ratio = final_val / final_train if final_train > 0 else 0
    print(f"  Best epoch train loss: {final_train:.4f}")
    print(f"  Best epoch val loss:   {final_val:.4f}")
    print(f"  Val/Train ratio:       {ratio:.2f}x")
    if ratio > 2.0:
        print(f"  ⚠ OVERFITTING WARNING: val loss > 2x train loss")
    else:
        print(f"  ✓ No severe overfitting detected")

    # --- Evaluate on test set ---
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # --- 3. Overall accuracy ---
    overall_acc = (all_preds == all_labels).mean()
    print(f"\n{'─' * 50}")
    print("3. OVERALL TEST ACCURACY")
    print(f"{'─' * 50}")
    print(f"  {overall_acc * 100:.1f}%  ({(all_preds == all_labels).sum()}/{len(all_labels)})")

    # --- 4. Per-class accuracy ---
    print(f"\n{'─' * 50}")
    print("4. PER-CLASS ACCURACY")
    print(f"{'─' * 50}")
    for cls in [0, 1, 2]:
        mask = all_labels == cls
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == cls).sum() / mask.sum()
            print(f"  {CLASS_NAMES[cls]:<8s} accuracy: {cls_acc * 100:>5.1f}%  ({(all_preds[mask] == cls).sum()}/{mask.sum()})")
        else:
            print(f"  {CLASS_NAMES[cls]:<8s} accuracy: N/A (no samples)")

    # --- 5. Confusion matrix ---
    print(f"\n{'─' * 50}")
    print("5. CONFUSION MATRIX")
    print(f"{'─' * 50}")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print(f"{'':>14s} Pred DOWN  Pred NEUT  Pred UP")
    for i, name in CLASS_NAMES.items():
        print(f"  True {name:<8s}  {cm[i, 0]:>6d}     {cm[i, 1]:>6d}     {cm[i, 2]:>5d}")

    # --- 6. High-confidence accuracy ---
    print(f"\n{'─' * 50}")
    print("6. HIGH-CONFIDENCE ACCURACY (confidence > 0.6)")
    print(f"{'─' * 50}")
    max_probs = all_probs.max(axis=1)
    high_conf_mask = max_probs > 0.6
    if high_conf_mask.sum() > 0:
        hc_acc = (all_preds[high_conf_mask] == all_labels[high_conf_mask]).mean()
        print(f"  High-conf samples: {high_conf_mask.sum()} / {len(all_labels)} ({high_conf_mask.mean() * 100:.1f}%)")
        print(f"  High-conf accuracy: {hc_acc * 100:.1f}%")
        if hc_acc < 0.50:
            print(f"  ⚠ HIGH-CONF ACCURACY BELOW 50% — do NOT integrate into ensemble")
    else:
        print(f"  No samples with confidence > 0.6")

    # --- 7. Classification report ---
    print(f"\n{'─' * 50}")
    print("7. FULL CLASSIFICATION REPORT")
    print(f"{'─' * 50}")
    print(classification_report(all_labels, all_preds,
                                target_names=["DOWN", "NEUTRAL", "UP"],
                                digits=3))

    # --- Save model + scaler ---
    model_path = os.path.join(SAVED_DIR, "intraday_lstm.pt")
    scaler_path = os.path.join(SAVED_DIR, "intraday_scaler.pkl")
    meta_path = os.path.join(SAVED_DIR, "intraday_lstm_meta.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "future_bars": FUTURE_BARS,
        "threshold": THRESHOLD,
        "class_names": CLASS_NAMES,
        "split_date": SPLIT_DATE,
        "overall_accuracy": float(overall_acc),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_epoch": best_epoch,
        "device": str(DEVICE),
    }, meta_path)

    print(f"\nSaved:")
    print(f"  Model:   {model_path}")
    print(f"  Scaler:  {scaler_path}")
    print(f"  Meta:    {meta_path}")

    # --- Save test results ---
    results_path = os.path.join(SAVED_DIR, "intraday_lstm_test_results.csv")
    results_df = pd.DataFrame({
        "timestamp": ts_test,
        "actual": all_labels,
        "predicted": all_preds,
        "prob_down": all_probs[:, 0],
        "prob_neutral": all_probs[:, 1],
        "prob_up": all_probs[:, 2],
        "correct": all_preds == all_labels,
    })
    results_df.to_csv(results_path, index=False)
    print(f"  Results: {results_path}")

    print(f"\n{'=' * 60}")
    print(f"  DONE — Overall accuracy: {overall_acc * 100:.1f}%")
    if overall_acc >= 0.50:
        print(f"  ✓ Passes 50% threshold — eligible for ensemble integration")
    else:
        print(f"  ⚠ Below 50% — use for /intraday endpoint only, do NOT integrate")
    print(f"{'=' * 60}")

    return model, scaler, overall_acc


if __name__ == "__main__":
    train_pipeline()
