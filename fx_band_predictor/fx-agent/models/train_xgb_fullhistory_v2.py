"""
Step 3 (v2) — XGBoost Full History with CORRECTED Targets

Fixes:
  1. Target uses 2 TRADING days (not calendar days) — Fridays now measure
     to Tuesday, not Sunday. No more free NEUTRAL on weekends.
  2. Removed is_friday and is_monday features (calendar shortcuts).
  3. Same hyperparameters, same train/test split.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import date as date_type

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features.feature_engineering import build_features
from data.fx_calendar import is_trading_day

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEUTRAL_THRESHOLD = 0.15
LABEL_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
EXCLUDE_COLS = {"date", "usdinr", "oil", "dxy", "vix", "us10y"}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "market_data_full.csv")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")

HIGH_CONF_THRESHOLD = 0.65
HARD_STOP_ACCURACY = 0.68


# ---------------------------------------------------------------------------
# FIX 1: Trading-day target construction
# ---------------------------------------------------------------------------

def get_nth_trading_day_rate(df: pd.DataFrame, n: int = 2) -> pd.Series:
    """
    For each row, find the rate on the nth TRADING day after that date.
    Trading days = weekdays that are not Indian/US holidays.

    This means:
      - Monday prediction → targets Wednesday (2 trading days)
      - Friday prediction → targets Tuesday (skips Sat/Sun)
      - Pre-holiday prediction → skips the holiday
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    dates = pd.to_datetime(df["date"]).tolist()
    rates = df["usdinr"].values

    # Build a lookup: date -> usdinr rate for O(1) access
    date_to_rate = {}
    for i, dt in enumerate(dates):
        date_to_rate[dt.date()] = rates[i]

    for i in range(len(dates)):
        current_date = dates[i].date()
        trading_count = 0
        check_date = current_date

        # Walk forward day by day until we find n trading days
        max_lookahead = n * 7  # safety: at most 7x calendar days per trading day
        for _ in range(max_lookahead):
            check_date = check_date + pd.Timedelta(days=1)
            py_date = check_date if isinstance(check_date, date_type) else check_date.date()

            if is_trading_day(py_date):
                trading_count += 1

            if trading_count == n:
                # Found the nth trading day — look up its rate
                if py_date in date_to_rate:
                    result.iloc[i] = date_to_rate[py_date]
                break

    return result


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Build target using 2 TRADING days forward (not calendar days)."""
    out = df.copy()

    print("  Computing 2-trading-day forward rates...")
    out["future_rate_2td"] = get_nth_trading_day_rate(out, n=2)
    out["rate_diff"] = out["future_rate_2td"] - out["usdinr"]

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
    print("STEP 3 (v2): XGBoost Full History — CORRECTED TARGETS")
    print("  Target: 2 TRADING days (not calendar days)")
    print("  Removed: is_friday, is_monday features")
    print("=" * 70)

    # --- Load and build features ---
    print(f"\nLoading data from {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Raw data: {len(raw)} rows, {raw['date'].min().date()} to {raw['date'].max().date()}")

    print("\nBuilding features (without is_friday, is_monday)...")
    features = build_features(raw)
    print(f"Feature matrix: {len(features)} rows, {len([c for c in features.columns if c not in EXCLUDE_COLS])} feature cols")

    # Verify is_friday is gone
    assert "is_friday" not in features.columns, "is_friday should be removed!"
    assert "is_monday" not in features.columns, "is_monday should be removed!"
    print("  Confirmed: is_friday and is_monday removed")

    # --- Build corrected target ---
    print("\nBuilding CORRECTED targets (2 trading days forward, ±0.15 INR)...")
    df = build_target(features)
    print(f"After target: {len(df)} rows")

    # --- Show Friday target distribution to prove fix works ---
    df["_dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    friday_rows = df[df["_dow"] == 4]
    print(f"\n  Friday target distribution (should NOT be mostly NEUTRAL now):")
    for cls in [0, 1, 2]:
        count = (friday_rows["target"] == cls).sum()
        pct = count / len(friday_rows) * 100 if len(friday_rows) > 0 else 0
        print(f"    {LABEL_MAP[cls]:<8s}: {count:>5d}  ({pct:.1f}%)")
    print(f"    Total Fridays: {len(friday_rows)}")

    non_friday_rows = df[df["_dow"] != 4]
    print(f"\n  Non-Friday target distribution (for comparison):")
    for cls in [0, 1, 2]:
        count = (non_friday_rows["target"] == cls).sum()
        pct = count / len(non_friday_rows) * 100 if len(non_friday_rows) > 0 else 0
        print(f"    {LABEL_MAP[cls]:<8s}: {count:>5d}  ({pct:.1f}%)")
    print(f"    Total non-Friday: {len(non_friday_rows)}")

    df = df.drop(columns=["_dow"])

    # --- Feature columns ---
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                    and c not in ("target", "future_rate_2td", "rate_diff")]
    print(f"\nFeature columns: {len(feature_cols)}")

    # --- Train/test split ---
    train_mask = df["date"] < "2025-01-01"
    test_mask = df["date"] >= "2025-01-01"

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "target"].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "target"].values
    test_dates = pd.to_datetime(df.loc[test_mask, "date"]).values
    test_rates = df.loc[test_mask, "usdinr"].values

    print(f"\nTrain: {len(X_train)} rows ({df.loc[train_mask, 'date'].min().date()} to {df.loc[train_mask, 'date'].max().date()})")
    print(f"Test:  {len(X_test)} rows ({df.loc[test_mask, 'date'].min().date()} to {df.loc[test_mask, 'date'].max().date()})")

    # ======================================================================
    # OUTPUT 1: Class distribution
    # ======================================================================
    print("\n" + "=" * 70)
    print("1. CLASS DISTRIBUTION")
    print("=" * 70)

    for label, y in [("Train", y_train), ("Test", y_test)]:
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

    # --- Train ---
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
    # OUTPUT 2: High confidence signal breakdown (CRITICAL CHECK)
    # ======================================================================
    print("\n" + "=" * 70)
    print("2. HIGH CONFIDENCE SIGNAL BREAKDOWN (confidence > 0.65)")
    print("=" * 70)

    high_conf_mask = max_conf > HIGH_CONF_THRESHOLD
    n_hc = high_conf_mask.sum()

    hc_pred = y_pred[high_conf_mask]
    hc_actual = y_test[high_conf_mask]
    hc_conf = max_conf[high_conf_mask]
    hc_dates = test_dates[high_conf_mask]
    hc_rates = test_rates[high_conf_mask]

    print(f"\n  Total high-confidence signals: {n_hc}")

    if n_hc > 0:
        # How many are Friday?
        hc_dows = pd.to_datetime(hc_dates).dayofweek
        n_friday_hc = (hc_dows == 4).sum()
        print(f"  Of which Friday calls: {n_friday_hc}  (was 52/52 before fix)")

        # Predicted class distribution
        print(f"\n  Predicted class distribution (high-conf):")
        for cls in [0, 1, 2]:
            count = (hc_pred == cls).sum()
            pct = count / n_hc * 100 if n_hc > 0 else 0
            print(f"    {LABEL_MAP[cls]:<8s}: {count:>4d}  ({pct:.1f}%)")

        # Confusion
        print(f"\n  {'Predicted':<12s} {'Actual':<12s} {'Count':>7s}")
        print(f"  {'─' * 12} {'─' * 12} {'─' * 7}")
        for pred_cls in [0, 1, 2]:
            for act_cls in [0, 1, 2]:
                count = ((hc_pred == pred_cls) & (hc_actual == act_cls)).sum()
                if count > 0:
                    correct = "✓" if pred_cls == act_cls else "✗"
                    print(f"  {LABEL_MAP[pred_cls]:<12s} {LABEL_MAP[act_cls]:<12s} {count:>7d}  {correct}")

        # Are there directional (UP/DOWN) high-confidence signals?
        n_directional_hc = ((hc_pred == 0) | (hc_pred == 2)).sum()
        print(f"\n  Directional (UP or DOWN) high-conf signals: {n_directional_hc}")
        if n_directional_hc == 0:
            print(f"  ⚠ CRITICAL: Still no directional high-confidence signals!")
            print(f"  ⚠ Model is still only confident about NEUTRAL")
        else:
            n_dir_correct = (
                ((hc_pred == 0) & (hc_actual == 0)) |
                ((hc_pred == 2) & (hc_actual == 2))
            ).sum()
            dir_acc = n_dir_correct / n_directional_hc * 100 if n_directional_hc > 0 else 0
            print(f"  Directional high-conf accuracy: {n_dir_correct}/{n_directional_hc} = {dir_acc:.1f}%")

        # Full accuracy
        hc_acc = accuracy_score(hc_actual, hc_pred)
        print(f"\n  Overall high-conf accuracy: {hc_acc:.1%} ({n_hc} signals)")

        if hc_acc < HARD_STOP_ACCURACY:
            print(f"\n  *** HARD STOP: {hc_acc:.1%} < {HARD_STOP_ACCURACY:.0%} ***")
            print(f"  *** MODEL NOT SAVED ***")
            return False

        # Print all signals
        print(f"\n  All {n_hc} high-confidence signals:")
        print(f"  {'Date':<12s} {'Day':<5s} {'Rate':>8s} {'Pred':<8s} {'Actual':<8s} {'Conf':>6s} {'OK':>4s}")
        print(f"  {'─' * 12} {'─' * 5} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 4}")
        for i in range(n_hc):
            dt = pd.Timestamp(hc_dates[i])
            day_name = dt.strftime("%a")
            print(f"  {dt.strftime('%Y-%m-%d'):<12s} {day_name:<5s} {hc_rates[i]:>8.2f} "
                  f"{LABEL_MAP[hc_pred[i]]:<8s} {LABEL_MAP[hc_actual[i]]:<8s} "
                  f"{hc_conf[i]:>6.3f} {'YES' if hc_pred[i] == hc_actual[i] else 'NO':>4s}")
    else:
        print("  No high-confidence signals at all")
        hc_acc = 0.0

    # ======================================================================
    # OUTPUT 3: Classification report
    # ======================================================================
    print("\n" + "=" * 70)
    print("3. CLASSIFICATION REPORT (Test Set)")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["DOWN", "NEUTRAL", "UP"]))

    # ======================================================================
    # OUTPUT 4: Overall accuracy
    # ======================================================================
    overall_acc = accuracy_score(y_test, y_pred)
    print("=" * 70)
    print(f"4. OVERALL TEST ACCURACY: {overall_acc:.1%}")
    print("=" * 70)

    # ======================================================================
    # OUTPUT 5: Top 10 feature importance
    # ======================================================================
    print("\n" + "=" * 70)
    print("5. TOP 10 FEATURE IMPORTANCE")
    print("=" * 70)

    importances = base_model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  {'Rank':<5s} {'Feature':<32s} {'Importance':>10s}")
    print(f"  {'─' * 5} {'─' * 32} {'─' * 10}")
    for i, row in feat_imp.head(10).iterrows():
        marker = ""
        if row["feature"] in ("rate_vs_alltime_percentile", "rate_vs_5y_avg",
                               "long_term_trend_1y", "is_decade_high"):
            marker = " ★ NEW"
        print(f"  {i+1:<5d} {row['feature']:<32s} {row['importance']:>10.4f}{marker}")

    # Verify is_friday gone
    print(f"\n  is_friday in features? {'YES ⚠' if 'is_friday' in feature_cols else 'NO ✓ (removed)'}")
    print(f"  is_monday in features? {'YES ⚠' if 'is_monday' in feature_cols else 'NO ✓ (removed)'}")

    # Where are new features?
    print(f"\n  New feature rankings:")
    for nf in ["rate_vs_alltime_percentile", "rate_vs_5y_avg", "long_term_trend_1y", "is_decade_high"]:
        rank_idx = feat_imp.index[feat_imp["feature"] == nf].tolist()
        if rank_idx:
            print(f"    {nf:<32s}  rank #{rank_idx[0]+1}")

    # ======================================================================
    # OUTPUT 6: Monthly accuracy 2025
    # ======================================================================
    print("\n" + "=" * 70)
    print("6. MONTHLY ACCURACY — 2025 TEST SET")
    print("=" * 70)

    test_df = df.loc[test_mask].copy().reset_index(drop=True)
    test_df["predicted"] = y_pred
    test_df["correct"] = (y_pred == y_test).astype(int)
    test_df["max_conf"] = max_conf
    test_df["month"] = pd.to_datetime(test_df["date"]).dt.month
    test_df["year"] = pd.to_datetime(test_df["date"]).dt.year

    # Only 2025 rows for monthly breakdown
    test_2025 = test_df[test_df["year"] == 2025]

    print(f"\n  {'Month':<12s} {'Accuracy':>10s} {'Signals':>9s} {'HC Acc':>10s} {'HC #':>6s}")
    print(f"  {'─' * 12} {'─' * 10} {'─' * 9} {'─' * 10} {'─' * 6}")
    for month_num in range(1, 13):
        m_mask = test_2025["month"] == month_num
        n = m_mask.sum()
        if n > 0:
            acc = test_2025.loc[m_mask, "correct"].mean()
            hc_m = m_mask & (test_2025["max_conf"] > HIGH_CONF_THRESHOLD)
            n_hc_m = hc_m.sum()
            hc_acc_m = test_2025.loc[hc_m, "correct"].mean() if n_hc_m > 0 else float("nan")
            month_name = pd.Timestamp(2025, month_num, 1).strftime("%b 2025")
            hc_str = f"{hc_acc_m:.1%}" if n_hc_m > 0 else "N/A"
            print(f"  {month_name:<12s} {acc:>10.1%} {n:>9d} {hc_str:>10s} {n_hc_m:>6d}")
        else:
            month_name = pd.Timestamp(2025, month_num, 1).strftime("%b 2025")
            print(f"  {month_name:<12s}       N/A         0        N/A      0")

    # Q4 highlight
    q4_mask = (test_2025["month"] >= 10)
    if q4_mask.sum() > 0:
        q4_acc = test_2025.loc[q4_mask, "correct"].mean()
        print(f"\n  Q4 2025 overall: {q4_acc:.1%} ({q4_mask.sum()} signals)")

    # ======================================================================
    # OUTPUT 7: Forward validation Jan-Feb 2026
    # ======================================================================
    print("\n" + "=" * 70)
    print("7. FORWARD VALIDATION — Jan-Feb 2026")
    print("=" * 70)

    fwd_mask = df["date"] >= "2026-01-01"
    fwd_df = df.loc[fwd_mask].copy().reset_index(drop=True)

    if len(fwd_df) == 0:
        print("  No 2026 data available")
    else:
        X_fwd = fwd_df[feature_cols].values
        y_fwd = fwd_df["target"].values

        y_pred_fwd = calibrated_model.predict(X_fwd)
        y_proba_fwd = calibrated_model.predict_proba(X_fwd)
        max_conf_fwd = y_proba_fwd.max(axis=1)

        fwd_acc = (y_pred_fwd == y_fwd).mean()
        print(f"\n  Period: {fwd_df['date'].min().date()} to {fwd_df['date'].max().date()}")
        print(f"  Rows: {len(fwd_df)}")
        print(f"  Overall accuracy: {fwd_acc:.1%}")

        # Class breakdown
        print(f"\n  Predicted → Actual:")
        for pred_cls in [0, 1, 2]:
            for act_cls in [0, 1, 2]:
                count = ((y_pred_fwd == pred_cls) & (y_fwd == act_cls)).sum()
                if count > 0:
                    print(f"    {LABEL_MAP[pred_cls]:<8s} → {LABEL_MAP[act_cls]:<8s}: {count}")

        # High conf
        hc_fwd_mask = max_conf_fwd > HIGH_CONF_THRESHOLD
        n_hc_fwd = hc_fwd_mask.sum()
        if n_hc_fwd > 0:
            hc_fwd_acc = (y_pred_fwd[hc_fwd_mask] == y_fwd[hc_fwd_mask]).mean()
            hc_fwd_pred = y_pred_fwd[hc_fwd_mask]
            n_dir_fwd = ((hc_fwd_pred == 0) | (hc_fwd_pred == 2)).sum()
            print(f"\n  High-conf (>0.65): {n_hc_fwd} signals, {hc_fwd_acc:.1%} accuracy")
            print(f"  Of which directional: {n_dir_fwd}")
        else:
            print(f"\n  High-conf (>0.65): 0 signals")

        # Comparison table
        print(f"\n  {'Model':<25s} {'Jan-Feb 2026':>14s}")
        print(f"  {'─' * 25} {'─' * 14}")
        print(f"  {'V2 (3yr, old target)':<25s} {'24.0%':>14s}")
        print(f"  {'FH v1 (broken target)':<25s} {'50.0%':>14s}")
        print(f"  {'FH v2 (trading-day)':<25s} {fwd_acc:>14.1%}")

    # ======================================================================
    # SAVE DECISION
    # ======================================================================
    print("\n" + "=" * 70)
    print("SAVE DECISION")
    print("=" * 70)

    if n_hc > 0 and hc_acc >= HARD_STOP_ACCURACY:
        # Check for directional signals
        n_directional = ((hc_pred == 0) | (hc_pred == 2)).sum()
        if n_directional == 0:
            print(f"\n  ⚠ CRITICAL: No directional high-confidence signals")
            print(f"  ⚠ All {n_hc} high-conf signals are NEUTRAL")
            print(f"  ⚠ NOT SAVING — user review needed")
            return False

        os.makedirs(SAVED_DIR, exist_ok=True)

        model_path = os.path.join(SAVED_DIR, "xgb_fullhistory_v2.pkl")
        fnames_path = os.path.join(SAVED_DIR, "feature_names_fullhistory_v2.pkl")
        fimp_path = os.path.join(SAVED_DIR, "feature_importance_fullhistory_v2.csv")

        joblib.dump(calibrated_model, model_path)
        print(f"\n  Saved model          → {model_path}")
        joblib.dump(feature_cols, fnames_path)
        print(f"  Saved feature names  → {fnames_path}")
        feat_imp.to_csv(fimp_path, index=False)
        print(f"  Saved importance     → {fimp_path}")
    elif n_hc == 0:
        print(f"\n  No high-confidence signals — NOT SAVING")
        print(f"  Review needed before proceeding")
        return False
    else:
        print(f"\n  High-conf accuracy {hc_acc:.1%} below {HARD_STOP_ACCURACY:.0%} — NOT SAVING")
        return False

    print(f"\n{'=' * 70}")
    print(f"Step 3 (v2) complete. Awaiting approval before Step 4.")
    print(f"{'=' * 70}")
    return True


if __name__ == "__main__":
    main()
