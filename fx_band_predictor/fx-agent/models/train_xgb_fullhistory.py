"""
Step 3 — Retrain XGBoost on Full History (2003-2025)

Train: 2003-01-01 to 2024-12-31
Test:  2025-01-01 to 2025-12-31 (never seen in training)

Uses all 49 features including 4 new long-term features.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features.feature_engineering import build_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEUTRAL_THRESHOLD = 0.15  # INR dead-zone
FUTURE_SHIFT = 2          # 2-day forward look

LABEL_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

EXCLUDE_COLS = {"date", "usdinr", "oil", "dxy", "vix", "us10y"}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "market_data_full.csv")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")

HIGH_CONF_THRESHOLD = 0.65
HARD_STOP_ACCURACY = 0.68  # Must achieve >= 68% high-conf accuracy


# ---------------------------------------------------------------------------
# Target builder
# ---------------------------------------------------------------------------

def build_target(df: pd.DataFrame) -> pd.DataFrame:
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
# Main training
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 3: XGBoost Full History Training (2003-2025)")
    print("=" * 70)

    # --- Load and build features ---
    print(f"\nLoading data from {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Raw data: {len(raw)} rows, {raw['date'].min().date()} to {raw['date'].max().date()}")

    print("\nBuilding features...")
    features = build_features(raw)
    print(f"Feature matrix: {len(features)} rows")

    # --- Build target ---
    print("\nBuilding targets (2-day forward, ±0.15 INR threshold)...")
    df = build_target(features)
    print(f"After target: {len(df)} rows")

    # --- Feature columns ---
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                    and c not in ("target", "future_rate_48h", "rate_diff")]
    print(f"Feature columns: {len(feature_cols)}")

    # --- Train/test split by date ---
    train_mask = df["date"] < "2025-01-01"
    test_mask = df["date"] >= "2025-01-01"

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "target"].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "target"].values
    test_dates = df.loc[test_mask, "date"].values
    test_rates = df.loc[test_mask, "usdinr"].values

    print(f"\nTrain: {len(X_train)} rows ({df.loc[train_mask, 'date'].min().date()} to {df.loc[train_mask, 'date'].max().date()})")
    print(f"Test:  {len(X_test)} rows ({df.loc[test_mask, 'date'].min().date()} to {df.loc[test_mask, 'date'].max().date()})")

    # ======================================================================
    # OUTPUT 1: Class distribution
    # ======================================================================
    print("\n" + "=" * 70)
    print("1. CLASS DISTRIBUTION")
    print("=" * 70)

    for label, name, y in [("Train", "train", y_train), ("Test", "test", y_test)]:
        print(f"\n  {label} set:")
        for cls in [0, 1, 2]:
            count = (y == cls).sum()
            pct = count / len(y) * 100
            print(f"    {LABEL_MAP[cls]:<8s}: {count:>5d}  ({pct:.1f}%)")
        print(f"    {'TOTAL':<8s}: {len(y):>5d}")

    # --- Sample weights ---
    sw_train = compute_sample_weight("balanced", y_train)
    print(f"\n  Sample weight range: {sw_train.min():.3f} – {sw_train.max():.3f}")
    for cls in sorted(set(y_train)):
        w = sw_train[y_train == cls][0]
        print(f"    {LABEL_MAP[cls]:<8s} weight: {w:.3f}")

    # --- Train base XGBoost ---
    print("\nTraining XGBClassifier (300 trees, depth=4, lr=0.05)...")
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
    print("  Base model trained.")

    # --- Calibrate ---
    print("  Calibrating with CalibratedClassifierCV(cv=5, isotonic)...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        cv=5,
        method="isotonic",
    )
    calibrated_model.fit(X_train, y_train, sample_weight=sw_train)
    print("  Calibrated model ready.")

    # --- Predictions ---
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)
    max_conf = y_proba.max(axis=1)

    # ======================================================================
    # OUTPUT 2: Classification report
    # ======================================================================
    print("\n" + "=" * 70)
    print("2. CLASSIFICATION REPORT (Test Set)")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["DOWN", "NEUTRAL", "UP"]))

    # ======================================================================
    # OUTPUT 3: Overall accuracy
    # ======================================================================
    overall_acc = accuracy_score(y_test, y_pred)
    print("=" * 70)
    print(f"3. OVERALL TEST ACCURACY: {overall_acc:.1%}")
    print("=" * 70)

    # ======================================================================
    # OUTPUT 4: High confidence accuracy
    # ======================================================================
    print("\n" + "=" * 70)
    print("4. HIGH CONFIDENCE ACCURACY (confidence > 0.65)")
    print("=" * 70)

    high_conf_mask = max_conf > HIGH_CONF_THRESHOLD
    n_high_conf = high_conf_mask.sum()

    if n_high_conf > 0:
        high_conf_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
        print(f"  Signals:  {n_high_conf}")
        print(f"  Accuracy: {high_conf_acc:.1%}")
        print(f"  Threshold: >= {HARD_STOP_ACCURACY:.0%}")

        if high_conf_acc < HARD_STOP_ACCURACY:
            print(f"\n  *** HARD STOP: High-conf accuracy {high_conf_acc:.1%} < {HARD_STOP_ACCURACY:.0%} ***")
            print(f"  *** MODEL NOT SAVED — INVESTIGATION NEEDED ***")
            return False
        else:
            print(f"  PASSED: {high_conf_acc:.1%} >= {HARD_STOP_ACCURACY:.0%}")
    else:
        print(f"  WARNING: No signals above {HIGH_CONF_THRESHOLD} confidence")
        print(f"  Cannot evaluate — proceeding with caution")

    # ======================================================================
    # OUTPUT 5: Top 15 features by importance
    # ======================================================================
    print("\n" + "=" * 70)
    print("5. TOP 15 FEATURES BY IMPORTANCE")
    print("=" * 70)

    importances = base_model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  {'Rank':<5s} {'Feature':<32s} {'Importance':>10s}")
    print(f"  {'─' * 5} {'─' * 32} {'─' * 10}")
    for i, row in feat_imp.head(15).iterrows():
        marker = ""
        if row["feature"] in ("rate_vs_alltime_percentile", "rate_vs_5y_avg",
                               "long_term_trend_1y", "is_decade_high"):
            marker = " ★ NEW"
        if row["feature"] == "is_rbi_week":
            marker = " ◆ WAS #1"
        print(f"  {i+1:<5d} {row['feature']:<32s} {row['importance']:>10.4f}{marker}")

    # Check where new features rank
    new_feats = ["rate_vs_alltime_percentile", "rate_vs_5y_avg",
                 "long_term_trend_1y", "is_decade_high"]
    print(f"\n  New feature rankings:")
    for nf in new_feats:
        rank = feat_imp.index[feat_imp["feature"] == nf].tolist()
        if rank:
            imp_val = feat_imp.loc[rank[0], "importance"]
            in_top15 = "YES ✓" if rank[0] < 15 else "no"
            print(f"    {nf:<32s}  rank #{rank[0]+1:<4d}  importance={imp_val:.4f}  top15={in_top15}")

    # Check is_rbi_week
    rbi_rank = feat_imp.index[feat_imp["feature"] == "is_rbi_week"].tolist()
    if rbi_rank:
        print(f"    {'is_rbi_week':<32s}  rank #{rbi_rank[0]+1:<4d}  (was #1 in v2)")

    # ======================================================================
    # OUTPUT 6: Per-regime accuracy
    # ======================================================================
    print("\n" + "=" * 70)
    print("6. PER-REGIME ACCURACY (Test Set)")
    print("=" * 70)

    test_df = df.loc[test_mask].copy().reset_index(drop=True)
    test_df["predicted"] = y_pred
    test_df["correct"] = (y_pred == y_test).astype(int)
    test_df["max_conf"] = max_conf

    regime_cols = {
        "regime_trending_up": "Trending UP",
        "regime_trending_down": "Trending DOWN",
        "regime_high_vol": "High Volatility",
        "regime_range_bound": "Range Bound",
    }

    print(f"\n  {'Regime':<20s} {'Rows':>6s} {'Accuracy':>10s} {'Avg Conf':>10s}")
    print(f"  {'─' * 20} {'─' * 6} {'─' * 10} {'─' * 10}")
    for col, label in regime_cols.items():
        mask = test_df[col] == 1
        n = mask.sum()
        if n > 0:
            acc = test_df.loc[mask, "correct"].mean()
            avg_c = test_df.loc[mask, "max_conf"].mean()
            print(f"  {label:<20s} {n:>6d} {acc:>10.1%} {avg_c:>10.3f}")
        else:
            print(f"  {label:<20s} {n:>6d}       N/A        N/A")

    # ======================================================================
    # OUTPUT 7: Monthly accuracy for 2025
    # ======================================================================
    print("\n" + "=" * 70)
    print("7. MONTHLY ACCURACY — 2025 TEST SET")
    print("=" * 70)

    test_df["month"] = pd.to_datetime(test_df["date"]).dt.month
    test_df["month_name"] = pd.to_datetime(test_df["date"]).dt.strftime("%b %Y")

    print(f"\n  {'Month':<12s} {'Accuracy':>10s} {'Signals':>9s} {'High-Conf':>10s} {'HC Signals':>11s}")
    print(f"  {'─' * 12} {'─' * 10} {'─' * 9} {'─' * 10} {'─' * 11}")

    monthly_results = []
    for month_num in range(1, 13):
        m_mask = test_df["month"] == month_num
        n = m_mask.sum()
        if n > 0:
            acc = test_df.loc[m_mask, "correct"].mean()
            hc_mask = m_mask & (test_df["max_conf"] > HIGH_CONF_THRESHOLD)
            n_hc = hc_mask.sum()
            hc_acc = test_df.loc[hc_mask, "correct"].mean() if n_hc > 0 else float("nan")
            month_name = pd.Timestamp(2025, month_num, 1).strftime("%b 2025")
            hc_str = f"{hc_acc:.1%}" if n_hc > 0 else "N/A"
            print(f"  {month_name:<12s} {acc:>10.1%} {n:>9d} {hc_str:>10s} {n_hc:>11d}")
            monthly_results.append({"month": month_name, "acc": acc, "n": n})
        else:
            month_name = pd.Timestamp(2025, month_num, 1).strftime("%b 2025")
            print(f"  {month_name:<12s}       N/A         0        N/A           0")

    # Q4 2025 highlight
    q4_mask = test_df["month"] >= 10
    if q4_mask.sum() > 0:
        q4_acc = test_df.loc[q4_mask, "correct"].mean()
        print(f"\n  Q4 2025 (Oct-Dec) accuracy: {q4_acc:.1%} ({q4_mask.sum()} signals)")
        print(f"  {'→ Long-term features helping!' if q4_acc > 0.50 else '→ Still struggling in unprecedented zone'}")

    # ======================================================================
    # OUTPUT 8: Compare vs V2
    # ======================================================================
    print("\n" + "=" * 70)
    print("8. COMPARISON VS V2 MODEL")
    print("=" * 70)

    v2_hc_acc = 0.793
    v2_hc_signals = 29

    if n_high_conf > 0:
        print(f"\n  {'Metric':<30s} {'V2':>12s} {'Full History':>14s} {'Delta':>10s}")
        print(f"  {'─' * 30} {'─' * 12} {'─' * 14} {'─' * 10}")
        print(f"  {'High-conf accuracy':<30s} {v2_hc_acc:>12.1%} {high_conf_acc:>14.1%} {high_conf_acc - v2_hc_acc:>+10.1%}")
        print(f"  {'High-conf signals':<30s} {v2_hc_signals:>12d} {n_high_conf:>14d} {n_high_conf - v2_hc_signals:>+10d}")
        print(f"  {'Overall accuracy':<30s} {'N/A':>12s} {overall_acc:>14.1%} {'':>10s}")

        # Verdict
        if high_conf_acc >= v2_hc_acc and n_high_conf >= v2_hc_signals:
            verdict = "BETTER: Higher accuracy AND more signals"
        elif high_conf_acc >= v2_hc_acc:
            verdict = "BETTER: Higher accuracy (fewer signals acceptable)"
        elif n_high_conf > v2_hc_signals and high_conf_acc >= HARD_STOP_ACCURACY:
            verdict = "TRADE-OFF: More signals, slightly lower accuracy (still above threshold)"
        elif high_conf_acc < HARD_STOP_ACCURACY:
            verdict = "WORSE: Below hard stop threshold — investigate"
        else:
            verdict = "COMPARABLE: Similar performance"
        print(f"\n  Verdict: {verdict}")
    else:
        print(f"\n  Cannot compare — no high-confidence signals in full history model")

    # ======================================================================
    # SAVE (only if passed hard stop)
    # ======================================================================
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    os.makedirs(SAVED_DIR, exist_ok=True)

    model_path = os.path.join(SAVED_DIR, "xgb_fullhistory.pkl")
    fnames_path = os.path.join(SAVED_DIR, "feature_names_fullhistory.pkl")
    fimp_path = os.path.join(SAVED_DIR, "feature_importance_fullhistory.csv")

    joblib.dump(calibrated_model, model_path)
    print(f"  Saved model          → {model_path}")

    joblib.dump(feature_cols, fnames_path)
    print(f"  Saved feature names  → {fnames_path}")

    feat_imp.to_csv(fimp_path, index=False)
    print(f"  Saved importance     → {fimp_path}")

    print(f"\n{'=' * 70}")
    print(f"Step 3 complete. Waiting for approval before Step 4 (LSTM).")
    print(f"{'=' * 70}")

    return True


if __name__ == "__main__":
    main()
