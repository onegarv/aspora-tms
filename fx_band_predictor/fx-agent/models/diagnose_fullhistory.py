"""
Diagnostic checks for XGBoost fullhistory model.
Investigating 100% high-confidence accuracy.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features.feature_engineering import build_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEUTRAL_THRESHOLD = 0.15
FUTURE_SHIFT = 2
LABEL_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
EXCLUDE_COLS = {"date", "usdinr", "oil", "dxy", "vix", "us10y"}

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "market_data_full.csv")


def build_target(df):
    out = df.copy()
    out["future_rate_48h"] = out["usdinr"].shift(-FUTURE_SHIFT)
    out["rate_diff"] = out["future_rate_48h"] - out["usdinr"]

    def classify(diff):
        if pd.isna(diff):
            return np.nan
        if diff < -NEUTRAL_THRESHOLD:
            return 0
        elif diff > NEUTRAL_THRESHOLD:
            return 2
        else:
            return 1

    out["target"] = out["rate_diff"].apply(classify)
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(int)
    return out


def main():
    # --- Load ---
    print("Loading data and model...")
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    features = build_features(raw)
    df = build_target(features)

    feature_cols = joblib.load(os.path.join(SAVED_DIR, "feature_names_fullhistory.pkl"))
    calibrated_model = joblib.load(os.path.join(SAVED_DIR, "xgb_fullhistory.pkl"))

    # --- Split ---
    test_2025_mask = (df["date"] >= "2025-01-01") & (df["date"] < "2026-01-01")
    test_2025 = df.loc[test_2025_mask].copy().reset_index(drop=True)

    X_test_2025 = test_2025[feature_cols].values
    y_test_2025 = test_2025["target"].values

    y_pred_2025 = calibrated_model.predict(X_test_2025)
    y_proba_2025 = calibrated_model.predict_proba(X_test_2025)
    max_conf_2025 = y_proba_2025.max(axis=1)

    high_conf_mask = max_conf_2025 > 0.65
    n_hc = high_conf_mask.sum()

    # ==================================================================
    # CHECK 1: What is the model actually predicting?
    # ==================================================================
    print("\n" + "=" * 70)
    print("CHECK 1: HIGH-CONFIDENCE SIGNAL BREAKDOWN")
    print("=" * 70)

    hc_pred = y_pred_2025[high_conf_mask]
    hc_actual = y_test_2025[high_conf_mask]
    hc_conf = max_conf_2025[high_conf_mask]
    hc_dates = test_2025.loc[high_conf_mask, "date"].values
    hc_rates = test_2025.loc[high_conf_mask, "usdinr"].values

    # Confusion matrix for high-conf signals
    print(f"\nTotal high-confidence signals: {n_hc}")
    print(f"\n  {'Predicted':<12s} {'Actual':<12s} {'Count':>7s}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 7}")
    for pred_cls in [0, 1, 2]:
        for act_cls in [0, 1, 2]:
            count = ((hc_pred == pred_cls) & (hc_actual == act_cls)).sum()
            if count > 0:
                print(f"  {LABEL_MAP[pred_cls]:<12s} {LABEL_MAP[act_cls]:<12s} {count:>7d}")

    # Distribution of predicted classes
    print(f"\n  Predicted class distribution (high-conf only):")
    for cls in [0, 1, 2]:
        count = (hc_pred == cls).sum()
        pct = count / n_hc * 100 if n_hc > 0 else 0
        print(f"    {LABEL_MAP[cls]:<8s}: {count:>4d}  ({pct:.1f}%)")

    print(f"\n  Actual class distribution (high-conf only):")
    for cls in [0, 1, 2]:
        count = (hc_actual == cls).sum()
        pct = count / n_hc * 100 if n_hc > 0 else 0
        print(f"    {LABEL_MAP[cls]:<8s}: {count:>4d}  ({pct:.1f}%)")

    # Print all 59 signals with dates
    print(f"\n  All {n_hc} high-confidence signals:")
    print(f"  {'Date':<12s} {'Rate':>8s} {'Pred':<8s} {'Actual':<8s} {'Conf':>6s} {'Correct':>8s}")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 8}")
    for i in range(n_hc):
        dt = pd.Timestamp(hc_dates[i]).strftime("%Y-%m-%d")
        rate = hc_rates[i]
        pred = LABEL_MAP[hc_pred[i]]
        actual = LABEL_MAP[hc_actual[i]]
        conf = hc_conf[i]
        correct = "YES" if hc_pred[i] == hc_actual[i] else "NO"
        print(f"  {dt:<12s} {rate:>8.2f} {pred:<8s} {actual:<8s} {conf:>6.3f} {correct:>8s}")

    # ==================================================================
    # CHECK 2: Accuracy at different confidence thresholds
    # ==================================================================
    print("\n" + "=" * 70)
    print("CHECK 2: ACCURACY AT DIFFERENT CONFIDENCE THRESHOLDS")
    print("=" * 70)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    print(f"\n  {'Threshold':>10s} {'Signals':>9s} {'Accuracy':>10s} {'Pred UP':>9s} {'Pred NEU':>10s} {'Pred DOWN':>11s}")
    print(f"  {'─' * 10} {'─' * 9} {'─' * 10} {'─' * 9} {'─' * 10} {'─' * 11}")
    for t in thresholds:
        mask = max_conf_2025 > t
        n = mask.sum()
        if n > 0:
            acc = (y_pred_2025[mask] == y_test_2025[mask]).mean()
            n_up = (y_pred_2025[mask] == 2).sum()
            n_neu = (y_pred_2025[mask] == 1).sum()
            n_down = (y_pred_2025[mask] == 0).sum()
            print(f"  {t:>10.2f} {n:>9d} {acc:>10.1%} {n_up:>9d} {n_neu:>10d} {n_down:>11d}")
        else:
            print(f"  {t:>10.2f} {0:>9d}        N/A")

    # ==================================================================
    # CHECK 3: SHAP analysis for high-confidence signals
    # ==================================================================
    print("\n" + "=" * 70)
    print("CHECK 3: SHAP ANALYSIS — WHAT DRIVES HIGH-CONFIDENCE SIGNALS?")
    print("=" * 70)

    # Get the base (uncalibrated) XGBoost model from the calibrated wrapper
    # CalibratedClassifierCV stores calibrated_classifiers_ list, each has a base_estimator
    base_models = []
    for cc in calibrated_model.calibrated_classifiers_:
        base_models.append(cc.estimator)

    # Use the first fold's base model for SHAP (they're all trained on similar data)
    base_xgb = base_models[0]

    print("\n  Computing SHAP values for high-confidence signals...")
    X_hc = test_2025.loc[high_conf_mask, feature_cols].values
    explainer = shap.TreeExplainer(base_xgb)
    shap_values = explainer.shap_values(X_hc)

    # shap_values is (n_samples, n_features, n_classes) or list of 3 arrays
    # For multiclass XGBoost, shap_values is a list of 3 arrays (one per class)
    if isinstance(shap_values, list):
        # Stack: shape becomes (n_classes, n_samples, n_features)
        shap_array = np.array(shap_values)
    else:
        shap_array = shap_values

    # For each high-conf signal, find which class was predicted and use that class's SHAP
    print(f"\n  SHAP values shape: {shap_array.shape}")

    # Aggregate: mean |SHAP| across all high-conf signals, for the PREDICTED class
    mean_abs_shap = np.zeros(len(feature_cols))
    for i in range(n_hc):
        pred_class = hc_pred[i]
        if len(shap_array.shape) == 3 and shap_array.shape[0] == 3:
            # shape: (n_classes, n_samples, n_features)
            mean_abs_shap += np.abs(shap_array[pred_class, i, :])
        else:
            # shape: (n_samples, n_features, n_classes) or (n_samples, n_features)
            mean_abs_shap += np.abs(shap_array[i, :, pred_class] if len(shap_array.shape) == 3 else shap_array[i, :])
    mean_abs_shap /= n_hc

    shap_ranking = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n  Top 15 features by mean |SHAP| (high-conf signals only):")
    print(f"  {'Rank':<5s} {'Feature':<32s} {'Mean |SHAP|':>12s}")
    print(f"  {'─' * 5} {'─' * 32} {'─' * 12}")
    for i, row in shap_ranking.head(15).iterrows():
        marker = ""
        if row["feature"] in ("rate_vs_alltime_percentile", "rate_vs_5y_avg",
                               "long_term_trend_1y", "is_decade_high"):
            marker = " ★ NEW"
        print(f"  {i+1:<5d} {row['feature']:<32s} {row['mean_abs_shap']:>12.4f}{marker}")

    # Check if one feature dominates
    top_shap = shap_ranking.iloc[0]["mean_abs_shap"]
    second_shap = shap_ranking.iloc[1]["mean_abs_shap"]
    dominance_ratio = top_shap / second_shap if second_shap > 0 else float("inf")
    print(f"\n  Dominance ratio (top / 2nd): {dominance_ratio:.2f}x")
    if dominance_ratio > 3.0:
        print(f"  ⚠ WARNING: {shap_ranking.iloc[0]['feature']} dominates — possible shortcut")
    else:
        print(f"  OK: No single feature dominates SHAP values")

    # Check new features specifically
    print(f"\n  New feature SHAP rankings:")
    new_feats = ["rate_vs_alltime_percentile", "rate_vs_5y_avg",
                 "long_term_trend_1y", "is_decade_high"]
    for nf in new_feats:
        rank_idx = shap_ranking.index[shap_ranking["feature"] == nf].tolist()
        if rank_idx:
            sv = shap_ranking.loc[rank_idx[0], "mean_abs_shap"]
            print(f"    {nf:<32s}  rank #{rank_idx[0]+1:<4d}  SHAP={sv:.4f}")

    # ==================================================================
    # CHECK 4: Forward validation on Jan-Feb 2026
    # ==================================================================
    print("\n" + "=" * 70)
    print("CHECK 4: FORWARD VALIDATION — Jan 1 to Feb 20, 2026")
    print("=" * 70)

    fwd_mask = df["date"] >= "2026-01-01"
    fwd_df = df.loc[fwd_mask].copy().reset_index(drop=True)

    if len(fwd_df) == 0:
        print("  No data for 2026 — skipping")
    else:
        X_fwd = fwd_df[feature_cols].values
        y_fwd = fwd_df["target"].values

        y_pred_fwd = calibrated_model.predict(X_fwd)
        y_proba_fwd = calibrated_model.predict_proba(X_fwd)
        max_conf_fwd = y_proba_fwd.max(axis=1)

        fwd_acc = (y_pred_fwd == y_fwd).mean()
        print(f"\n  Period: {fwd_df['date'].min().date()} to {fwd_df['date'].max().date()}")
        print(f"  Rows:   {len(fwd_df)}")
        print(f"\n  Overall accuracy: {fwd_acc:.1%}")

        # Class breakdown
        print(f"\n  Predicted class distribution:")
        for cls in [0, 1, 2]:
            count = (y_pred_fwd == cls).sum()
            pct = count / len(y_pred_fwd) * 100
            print(f"    {LABEL_MAP[cls]:<8s}: {count:>4d}  ({pct:.1f}%)")

        print(f"\n  Actual class distribution:")
        for cls in [0, 1, 2]:
            count = (y_fwd == cls).sum()
            pct = count / len(y_fwd) * 100
            print(f"    {LABEL_MAP[cls]:<8s}: {count:>4d}  ({pct:.1f}%)")

        # High-conf on forward
        hc_fwd_mask = max_conf_fwd > 0.65
        n_hc_fwd = hc_fwd_mask.sum()
        if n_hc_fwd > 0:
            hc_fwd_acc = (y_pred_fwd[hc_fwd_mask] == y_fwd[hc_fwd_mask]).mean()
            print(f"\n  High-confidence (>0.65): {n_hc_fwd} signals, {hc_fwd_acc:.1%} accuracy")
        else:
            print(f"\n  High-confidence (>0.65): 0 signals")

        # Comparison
        v2_fwd_acc = 0.24
        print(f"\n  {'Metric':<30s} {'V2':>10s} {'FullHist':>10s}")
        print(f"  {'─' * 30} {'─' * 10} {'─' * 10}")
        print(f"  {'Jan-Feb 2026 overall acc':<30s} {v2_fwd_acc:>10.1%} {fwd_acc:>10.1%}")
        if n_hc_fwd > 0:
            print(f"  {'Jan-Feb 2026 high-conf acc':<30s} {'N/A':>10s} {hc_fwd_acc:>10.1%}")
            print(f"  {'Jan-Feb 2026 high-conf #':<30s} {'N/A':>10s} {n_hc_fwd:>10d}")

        if fwd_acc < v2_fwd_acc:
            print(f"\n  ⚠ WARNING: Fullhistory model WORSE than V2 on forward validation!")
            print(f"  ⚠ Possible overfit to 2025 trend — investigate before production use")
        elif fwd_acc > v2_fwd_acc:
            print(f"\n  ✓ Fullhistory model BETTER than V2 on forward validation")
        else:
            print(f"\n  Fullhistory model EQUAL to V2 on forward validation")

        # Print individual predictions for forward period
        print(f"\n  All Jan-Feb 2026 predictions:")
        print(f"  {'Date':<12s} {'Rate':>8s} {'Pred':<8s} {'Actual':<8s} {'Conf':>6s} {'Correct':>8s}")
        print(f"  {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 8}")
        for i in range(len(fwd_df)):
            dt = fwd_df.iloc[i]["date"].strftime("%Y-%m-%d")
            rate = fwd_df.iloc[i]["usdinr"]
            pred = LABEL_MAP[y_pred_fwd[i]]
            actual = LABEL_MAP[y_fwd[i]]
            conf = max_conf_fwd[i]
            correct = "YES" if y_pred_fwd[i] == y_fwd[i] else "NO"
            hc_marker = " ◆" if conf > 0.65 else ""
            print(f"  {dt:<12s} {rate:>8.2f} {pred:<8s} {actual:<8s} {conf:>6.3f} {correct:>8s}{hc_marker}")

    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTICS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
