"""
XGBoost Regime Classifier

Replaces the direction classifier. XGBoost now classifies the
current market regime (4-class):
    0: trending_up
    1: trending_down
    2: high_vol
    3: range_bound

Regime is derived from rate_trend_30d and volatility_5d thresholds
(same as feature_engineering.py regime features).

Train: 2003-2024, Test: 2025
No CalibratedClassifierCV needed — raw probabilities are fine for regime.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features.feature_engineering import build_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TREND_THRESHOLD = 0.0004  # matches feature_engineering.py
VOL_THRESHOLD = 0.006     # matches feature_engineering.py

REGIME_MAP = {0: "trending_up", 1: "trending_down", 2: "high_vol", 3: "range_bound"}

# Features to EXCLUDE from model input (they are the target or derived from it)
REGIME_DERIVED_FEATURES = {
    "regime_trending_up", "regime_trending_down", "regime_high_vol", "regime_range_bound",
    "is_trending_up", "is_trending_down", "trend_strength",
}

EXCLUDE_COLS = {"date", "usdinr", "oil", "dxy", "vix", "us10y", "us_2y", "fed_funds", "cpi"}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "market_data_full.csv")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")


def build_regime_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 4-class regime target from rate_trend_30d and volatility_5d.

    Priority: high_vol > trending_up/down > range_bound
    """
    out = df.copy()

    trend = out["rate_trend_30d"]
    vol = out["volatility_5d"]
    avg_dev = out["rate_vs_30d_avg"]

    def classify(row_idx):
        t = trend.iloc[row_idx]
        v = vol.iloc[row_idx]
        a = avg_dev.iloc[row_idx]

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


def main():
    print("=" * 70)
    print("XGBoost REGIME CLASSIFIER Training")
    print("  4-class: trending_up / trending_down / high_vol / range_bound")
    print("=" * 70)

    # --- Load and build features ---
    print(f"\nLoading {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Raw: {len(raw)} rows")

    print("Building features...")
    features = build_features(raw)
    print(f"Features: {len(features)} rows")

    # --- Build regime target ---
    df = build_regime_target(features)

    # --- Feature columns (exclude regime-derived) ---
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                    and c not in REGIME_DERIVED_FEATURES
                    and c != "regime_target"]
    print(f"Model features: {len(feature_cols)} (excluded {len(REGIME_DERIVED_FEATURES)} regime-derived)")

    # --- Train/test split ---
    train_mask = df["date"] < "2025-01-01"
    test_mask = df["date"] >= "2025-01-01"

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "regime_target"].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "regime_target"].values

    print(f"\nTrain: {len(X_train)} rows")
    print(f"Test:  {len(X_test)} rows")

    # --- Class distribution ---
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*70}")

    for label, y in [("Train", y_train), ("Test", y_test)]:
        print(f"\n  {label}:")
        for cls in sorted(set(y)):
            count = (y == cls).sum()
            pct = count / len(y) * 100
            print(f"    {REGIME_MAP[cls]:<15s}: {count:>5d}  ({pct:.1f}%)")

    # --- Sample weights ---
    sw_train = compute_sample_weight("balanced", y_train)

    # --- Train ---
    print(f"\nTraining XGBClassifier (regime, 300 trees)...")
    model = XGBClassifier(
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
    model.fit(X_train, y_train, sample_weight=sw_train)
    print("  Trained.")

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"\n{'='*70}")
    print("CLASSIFICATION REPORT (Test 2025)")
    print(f"{'='*70}")
    target_names = [REGIME_MAP[i] for i in sorted(REGIME_MAP.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))

    overall_acc = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {overall_acc:.1%}")

    # --- Confusion matrix ---
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'Predicted →':<15s}", end="")
    for name in target_names:
        print(f" {name[:8]:>8s}", end="")
    print()
    for i, name in enumerate(target_names):
        print(f"  {name:<15s}", end="")
        for j in range(len(target_names)):
            print(f" {cm[i][j]:>8d}", end="")
        print()

    # --- Top 10 features ---
    print(f"\n{'='*70}")
    print("TOP 10 FEATURES")
    print(f"{'='*70}")

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    for i, row in feat_imp.head(10).iterrows():
        print(f"  {i+1:<3d} {row['feature']:<32s} {row['importance']:.4f}")

    # --- Per-regime confidence distribution ---
    print(f"\n{'='*70}")
    print("REGIME CONFIDENCE DISTRIBUTION")
    print(f"{'='*70}")

    max_conf = y_proba.max(axis=1)
    for cls in sorted(REGIME_MAP.keys()):
        mask = y_pred == cls
        if mask.sum() > 0:
            confs = max_conf[mask]
            print(f"\n  {REGIME_MAP[cls]:<15s}: {mask.sum():>4d} signals")
            print(f"    Confidence: min={confs.min():.3f}  mean={confs.mean():.3f}  max={confs.max():.3f}")

    # --- Save ---
    print(f"\n{'='*70}")
    print("SAVING")
    print(f"{'='*70}")

    os.makedirs(SAVED_DIR, exist_ok=True)

    model_path = os.path.join(SAVED_DIR, "xgb_regime.pkl")
    fnames_path = os.path.join(SAVED_DIR, "feature_names_regime.pkl")
    fimp_path = os.path.join(SAVED_DIR, "feature_importance_regime.csv")

    joblib.dump(model, model_path)
    print(f"  Model   → {model_path}")

    joblib.dump(feature_cols, fnames_path)
    print(f"  Features → {fnames_path}")

    feat_imp.to_csv(fimp_path, index=False)
    print(f"  Importance → {fimp_path}")

    print(f"\n{'='*70}")
    print("Regime classifier training complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
