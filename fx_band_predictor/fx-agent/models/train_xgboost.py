"""
Phase 3 — XGBoost Direction Classifier

Predicts 48-hour USD/INR direction as 3 classes:
    0 = DOWN  (rate drops > 0.15 INR)
    1 = NEUTRAL (rate moves < 0.15 INR in either direction)
    2 = UP    (rate rises > 0.15 INR)

Pipeline:
    1. Construct target variable from 48h future rate
    2. Chronological 80/20 train/test split
    3. Train XGBClassifier
    4. Wrap with CalibratedClassifierCV (isotonic) for trustworthy probabilities
    5. Evaluate: classification report, directional accuracy, confusion matrix,
       high-confidence accuracy, feature importance, calibration quality
    6. Save model + artifacts
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
SAVED_DIR = os.path.join(BASE_DIR, "saved")
os.makedirs(SAVED_DIR, exist_ok=True)

NEUTRAL_THRESHOLD = 0.15          # INR dead-zone band
FUTURE_SHIFT = 2                  # 48 hours ≈ 2 calendar days
TRAIN_RATIO = 0.80
HIGH_CONF_THRESHOLD = 0.65        # for high-confidence evaluation

LABEL_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

# Features used by the model — everything from feature_engineering except
# raw market data columns and date
EXCLUDE_COLS = {"date", "usdinr", "oil", "dxy", "vix", "us10y", "us_2y", "fed_funds", "cpi"}


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target column based on 48h future rate movement.
    Drops rows where future rate is unavailable (last FUTURE_SHIFT rows).
    """
    out = df.copy()
    out["future_rate_48h"] = out["usdinr"].shift(-FUTURE_SHIFT)
    out["rate_diff"] = out["future_rate_48h"] - out["usdinr"]

    def classify(diff):
        if pd.isna(diff):
            return np.nan
        if diff < -NEUTRAL_THRESHOLD:
            return 0  # DOWN
        elif diff > NEUTRAL_THRESHOLD:
            return 2  # UP
        else:
            return 1  # NEUTRAL

    out["target"] = out["rate_diff"].apply(classify)
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(int)
    return out


# ---------------------------------------------------------------------------
# Train/test split (chronological)
# ---------------------------------------------------------------------------

def time_split(df: pd.DataFrame, feature_cols: list):
    """
    Chronological 80/20 split. Returns X_train, X_test, y_train, y_test,
    plus date arrays for both splits.
    """
    split_idx = int(len(df) * TRAIN_RATIO)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols].values
    X_test = test[feature_cols].values
    y_train = train["target"].values
    y_test = test["target"].values

    dates_train = train["date"].values
    dates_test = test["date"].values

    return X_train, X_test, y_train, y_test, dates_train, dates_test


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def directional_accuracy(y_true, y_pred):
    """
    Among predictions where the model says UP or DOWN (not NEUTRAL),
    what percentage are correct?
    """
    mask = (y_pred != 1)  # non-NEUTRAL predictions
    if mask.sum() == 0:
        return 0.0, 0
    correct = (y_true[mask] == y_pred[mask]).sum()
    total = mask.sum()
    return correct / total, total


def high_confidence_accuracy(y_true, y_pred, proba, threshold=HIGH_CONF_THRESHOLD):
    """
    Among predictions where max probability exceeds threshold,
    what percentage are correct?
    """
    max_conf = proba.max(axis=1)
    mask = max_conf > threshold
    if mask.sum() == 0:
        return 0.0, 0
    correct = (y_true[mask] == y_pred[mask]).sum()
    total = mask.sum()
    return correct / total, total


def calibration_check(y_true, y_pred, proba):
    """
    Compare average confidence for correct vs incorrect predictions.
    """
    max_conf = proba.max(axis=1)
    correct_mask = y_true == y_pred
    avg_correct = max_conf[correct_mask].mean() if correct_mask.sum() > 0 else 0.0
    avg_incorrect = max_conf[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0.0
    return avg_correct, avg_incorrect


def print_confusion(y_true, y_pred):
    """Print a labeled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    labels = ["DOWN", "NEUTRAL", "UP"]
    print(f"\n{'':>12s}  {'Pred DOWN':>10s} {'Pred NEUT':>10s} {'Pred UP':>10s}")
    print("  " + "-" * 45)
    for i, row_label in enumerate(labels):
        print(f"  True {row_label:<7s}  {cm[i, 0]:>10d} {cm[i, 1]:>10d} {cm[i, 2]:>10d}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_pipeline(feature_df: pd.DataFrame, version: str = "v1",
                    extra_data: dict = None):
    """
    Full training pipeline:
        1. Build target
        2. Split
        3. Train XGBoost
        4. Calibrate
        5. Evaluate
        6. Save
    Returns the calibrated model and feature column names.

    Args:
        feature_df: Feature matrix DataFrame
        version: "v1" or "v2" — controls output filenames
        extra_data: Optional dict with "test_df" for regime analysis
    """
    # Output filenames based on version
    if version == "v2":
        model_filename = "xgb_direction_v2.pkl"
        fnames_filename = "feature_names_v2.pkl"
        imp_filename = "feature_importance_v2.csv"
        results_filename = "xgb_test_results_v2.csv"
    else:
        model_filename = "xgb_direction.pkl"
        fnames_filename = "feature_names.pkl"
        imp_filename = "feature_importance.csv"
        results_filename = "xgb_test_results.csv"
    # ------------------------------------------------------------------
    # 1. Target construction
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Building target variable")
    print("=" * 70)

    df = build_target(feature_df)

    class_counts = df["target"].value_counts().sort_index()
    print(f"\nClass distribution:")
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"  {LABEL_MAP[cls]:>7s} ({cls}): {count:>5d}  ({pct:.1f}%)")
    print(f"  {'TOTAL':>7s}:     {len(df):>5d}")

    # ------------------------------------------------------------------
    # 2. Chronological split
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 2: Chronological train/test split (80/20)")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                    and c not in ("target", "future_rate_48h", "rate_diff")]

    X_train, X_test, y_train, y_test, dates_train, dates_test = time_split(df, feature_cols)

    print(f"\n  Train: {len(X_train)} rows  |  {str(dates_train[0])[:10]} → {str(dates_train[-1])[:10]}")
    print(f"  Test:  {len(X_test)} rows  |  {str(dates_test[0])[:10]} → {str(dates_test[-1])[:10]}")

    train_dist = pd.Series(y_train).value_counts().sort_index()
    test_dist = pd.Series(y_test).value_counts().sort_index()
    print(f"\n  Train class distribution: {dict(train_dist)}")
    print(f"  Test  class distribution: {dict(test_dist)}")

    # ------------------------------------------------------------------
    # 3. Train base XGBoost
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 3: Training XGBClassifier")
    print("=" * 70)

    # Compute sample weights to handle class imbalance
    # DOWN and UP are minority classes (~15-19%) vs NEUTRAL (~67%)
    # 'balanced' makes each class equally important
    sw_train = compute_sample_weight("balanced", y_train)
    print(f"\n  Sample weight range: {sw_train.min():.3f} – {sw_train.max():.3f}")
    for cls in sorted(set(y_train)):
        w = sw_train[y_train == cls][0]
        print(f"    {LABEL_MAP[cls]:<8s} weight: {w:.3f}")

    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )

    base_model.fit(X_train, y_train, sample_weight=sw_train)
    print("  Base XGBoost training complete (with balanced class weights).")

    # ------------------------------------------------------------------
    # 4. Calibrate with CalibratedClassifierCV
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 4: Calibrating probabilities (isotonic, 5-fold)")
    print("=" * 70)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        cv=5,
        method="isotonic",
    )
    # Pass sample weights to calibration as well so folds are balanced
    calibrated_model.fit(X_train, y_train, sample_weight=sw_train)
    print("  Calibration complete.")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 5: Evaluation on TEST SET")
    print("=" * 70)

    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)

    # 5a. Classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "NEUTRAL", "UP"]))

    # 5b. Overall accuracy
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {overall_acc:.4f} ({overall_acc * 100:.1f}%)")

    # 5c. Directional accuracy
    dir_acc, dir_count = directional_accuracy(y_test, y_pred)
    print(f"\nDirectional accuracy (UP/DOWN only, excluding NEUTRAL):")
    print(f"  {dir_acc:.4f} ({dir_acc * 100:.1f}%) on {dir_count} non-NEUTRAL predictions")

    # 5d. Confusion matrix
    print("\n--- Confusion Matrix ---")
    print_confusion(y_test, y_pred)

    # 5e. High-confidence accuracy
    hc_acc, hc_count = high_confidence_accuracy(y_test, y_pred, y_proba, HIGH_CONF_THRESHOLD)
    print(f"\n--- High Confidence Signals (max prob > {HIGH_CONF_THRESHOLD}) ---")
    print(f"  Accuracy: {hc_acc:.4f} ({hc_acc * 100:.1f}%)")
    print(f"  Count:    {hc_count} out of {len(y_test)} test samples")
    print(f"  Frequency: ~{hc_count / max(len(y_test), 1) * 30:.1f} per month (if test ≈ 1 month)")

    # 5f. Calibration check
    avg_correct, avg_incorrect = calibration_check(y_test, y_pred, y_proba)
    print(f"\n--- Calibration Check ---")
    print(f"  Avg confidence on CORRECT   predictions: {avg_correct:.4f}")
    print(f"  Avg confidence on INCORRECT predictions: {avg_incorrect:.4f}")
    gap = avg_correct - avg_incorrect
    print(f"  Gap: {gap:+.4f} {'(GOOD — correct preds have higher confidence)' if gap > 0 else '(WARNING — calibration may be poor)'}")

    # 5g. Per-regime accuracy (if regime features exist)
    regime_cols = [c for c in feature_cols if c.startswith("regime_")]
    if regime_cols and "test_df" in (extra_data or {}):
        print(f"\n--- Accuracy by Regime ---")
        test_df_regime = extra_data["test_df"]
        for regime_col in regime_cols:
            if regime_col not in test_df_regime.columns:
                continue
            mask = test_df_regime[regime_col].values == 1
            if mask.sum() == 0:
                print(f"  {regime_col}: 0 days (no samples)")
                continue
            regime_correct = (y_test[mask] == y_pred[mask]).sum()
            regime_total = mask.sum()
            regime_acc = regime_correct / regime_total * 100

            # Directional accuracy in this regime
            dir_mask = mask & (y_pred != 1)
            if dir_mask.sum() > 0:
                dir_corr = (y_test[dir_mask] == y_pred[dir_mask]).sum()
                dir_regime_acc = dir_corr / dir_mask.sum() * 100
                print(f"  {regime_col:<25s}  {regime_total:>4d} days | overall {regime_acc:.1f}% | directional {dir_regime_acc:.1f}% ({dir_corr}/{dir_mask.sum()})")
            else:
                print(f"  {regime_col:<25s}  {regime_total:>4d} days | overall {regime_acc:.1f}% | directional N/A (no directional preds)")

    # ------------------------------------------------------------------
    # 6. Feature importance
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 6: Feature Importance (top 15 by gain)")
    print("=" * 70)

    importance = base_model.get_booster().get_score(importance_type="gain")
    # Map feature names (xgboost uses f0, f1, ... internally)
    fname_map = {f"f{i}": name for i, name in enumerate(feature_cols)}
    importance_named = {fname_map.get(k, k): v for k, v in importance.items()}
    imp_sorted = sorted(importance_named.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  {'Rank':<5s} {'Feature':<25s} {'Gain':>12s}")
    print("  " + "-" * 45)
    for rank, (feat, gain) in enumerate(imp_sorted[:15], 1):
        marker = " ★" if feat.startswith(("rate_trend", "regime_", "rate_vs_52w", "rate_percentile", "momentum_", "is_trending", "trend_")) else ""
        print(f"  {rank:<5d} {feat:<25s} {gain:>12.4f}{marker}")

    # Check how many regime features in top 10
    regime_feature_names = {f for f in feature_cols if f.startswith(("rate_trend", "regime_", "rate_vs_52w", "rate_percentile", "momentum_7d", "momentum_consistency", "is_trending", "trend_"))}
    top10_names = {f for f, _ in imp_sorted[:10]}
    regime_in_top10 = top10_names & regime_feature_names
    print(f"\n  Regime features in top 10: {len(regime_in_top10)} — {sorted(regime_in_top10) if regime_in_top10 else 'NONE'}")

    # Save feature importance
    imp_df = pd.DataFrame(imp_sorted, columns=["feature", "gain"])
    imp_path = os.path.join(SAVED_DIR, imp_filename)
    imp_df.to_csv(imp_path, index=False)
    print(f"\n  Saved full feature importance to {imp_path}")

    # ------------------------------------------------------------------
    # 7. Save artifacts
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 7: Saving artifacts")
    print("=" * 70)

    model_path = os.path.join(SAVED_DIR, model_filename)
    joblib.dump(calibrated_model, model_path)
    print(f"  Saved calibrated model → {model_path}")

    fnames_path = os.path.join(SAVED_DIR, fnames_filename)
    joblib.dump(feature_cols, fnames_path)
    print(f"  Saved feature names    → {fnames_path}")

    # Save test results with dates
    results_df = pd.DataFrame({
        "date": dates_test,
        "actual": y_test,
        "predicted": y_pred,
        "conf_down": y_proba[:, 0],
        "conf_neutral": y_proba[:, 1],
        "conf_up": y_proba[:, 2],
        "max_confidence": y_proba.max(axis=1),
    })
    results_path = os.path.join(SAVED_DIR, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"  Saved test results     → {results_path}")

    # ------------------------------------------------------------------
    # 8. Sanity check — last 5 predictions
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SANITY CHECK: Last 5 test predictions (raw predict_proba)")
    print("=" * 70)

    last5_X = X_test[-5:]
    last5_dates = dates_test[-5:]
    last5_proba = calibrated_model.predict_proba(last5_X)
    last5_pred = calibrated_model.predict(last5_X)
    last5_actual = y_test[-5:]

    print(f"\n  {'Date':<12s} {'Actual':<9s} {'Predicted':<10s} {'P(DOWN)':>8s} {'P(NEUT)':>8s} {'P(UP)':>8s} {'MaxConf':>8s}")
    print("  " + "-" * 70)
    for i in range(5):
        dt = str(last5_dates[i])[:10]
        act = LABEL_MAP[last5_actual[i]]
        pred = LABEL_MAP[last5_pred[i]]
        p_down = last5_proba[i, 0]
        p_neut = last5_proba[i, 1]
        p_up = last5_proba[i, 2]
        mx = max(p_down, p_neut, p_up)
        print(f"  {dt:<12s} {act:<9s} {pred:<10s} {p_down:>8.4f} {p_neut:>8.4f} {p_up:>8.4f} {mx:>8.4f}")

    return calibrated_model, feature_cols


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    sys.path.insert(0, PROJECT_DIR)

    parser = argparse.ArgumentParser(description="Train XGBoost direction classifier")
    parser.add_argument("--v2", action="store_true",
                        help="Train v2 model on extended data with regime features")
    args = parser.parse_args()

    if args.v2:
        # V2: Load extended market data and build features from scratch
        from features.feature_engineering import build_features

        data_path = os.path.join(PROJECT_DIR, "data", "market_data_extended.csv")
        if not os.path.exists(data_path):
            print(f"ERROR: {data_path} not found. Run: python data/fetch_market_data.py --extended")
            sys.exit(1)

        print(f"Loading EXTENDED market data from {data_path}")
        raw = pd.read_csv(data_path, parse_dates=["date"])
        print(f"Raw shape: {raw.shape}")

        print("Building features (with regime features)...")
        df = build_features(raw)
        print(f"Feature matrix shape: {df.shape}\n")

        # We need the test portion of the feature df for regime analysis
        split_idx = int(len(df) * TRAIN_RATIO)
        # build_target will drop last FUTURE_SHIFT rows, but regime cols stay
        # We pass the full feature df and let train_pipeline handle the split
        # But we need the test_df after target construction for regime analysis
        # So we build target first, then extract test_df
        from models.train_xgboost import build_target
        df_with_target = build_target(df)
        test_split = int(len(df_with_target) * TRAIN_RATIO)
        test_df = df_with_target.iloc[test_split:]

        model, features = train_pipeline(df, version="v2",
                                          extra_data={"test_df": test_df})

        print("\n--- V2 XGBoost training complete ---")
    else:
        feature_path = os.path.join(PROJECT_DIR, "data", "feature_matrix.csv")
        if not os.path.exists(feature_path):
            print(f"ERROR: {feature_path} not found. Run features/feature_engineering.py first.")
            sys.exit(1)

        print(f"Loading feature matrix from {feature_path}")
        df = pd.read_csv(feature_path, parse_dates=["date"])
        print(f"Shape: {df.shape}\n")

        model, features = train_pipeline(df)

        print("\n--- Phase 3 complete ---")
