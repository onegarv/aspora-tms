"""Test range risk detector on all 30 walk-forward days."""
import os, sys
import pandas as pd
import numpy as np
import joblib

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_DIR)

from features.feature_engineering import build_features
from agents.range_risk_detector import detect_high_risk_prediction
from data.fx_calendar import is_trading_day, get_calendar_context

DATA_PATH = os.path.join(PROJECT_DIR, "data", "market_data_full.csv")
SAVED_DIR = os.path.join(PROJECT_DIR, "models", "saved")
CSV_PATH = os.path.join(PROJECT_DIR, "backtest", "walkforward_2026_daily.csv")

REGIME_MAP = {0: "trending_up", 1: "trending_down", 2: "high_vol", 3: "range_bound"}


def main():
    # Load walk-forward results
    wf = pd.read_csv(CSV_PATH)

    # Load full features
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    features_df = build_features(raw)

    # Load XGBoost for regime detection
    xgb_model = None
    xgb_feature_names = None
    xgb_path = os.path.join(SAVED_DIR, "xgb_regime.pkl")
    xgb_features_path = os.path.join(SAVED_DIR, "feature_names_regime.pkl")
    if os.path.exists(xgb_path) and os.path.exists(xgb_features_path):
        xgb_model = joblib.load(xgb_path)
        xgb_feature_names = joblib.load(xgb_features_path)

    # Build date → features lookup
    date_to_features = {}
    for _, row in features_df.iterrows():
        d = row["date"]
        dt = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
        date_to_features[dt] = row

    print("=" * 90)
    print("RISK DETECTION ON JAN-FEB 2026 WALK-FORWARD")
    print("=" * 90)
    print(f"\n  {'Date':<12s} {'Result':>6s} {'Risk':>9s} {'Score':>5s} {'Key Factors'}")
    print(f"  {'─'*12} {'─'*6} {'─'*9} {'─'*5} {'─'*50}")

    risk_results = []
    prev_regime = None

    for _, row in wf.iterrows():
        date_str = row["date"]

        # Get full feature row
        feat_row = date_to_features.get(date_str)
        if feat_row is None:
            continue

        current_rate = float(feat_row["usdinr"])

        # Compute 5-day high/low
        mask = features_df["date"] <= pd.Timestamp(date_str)
        recent = features_df.loc[mask].tail(5)
        high_5d = float(recent["usdinr"].max()) if len(recent) >= 5 else 0
        low_5d = float(recent["usdinr"].min()) if len(recent) >= 5 else 0

        # Compute 5-day volatility
        if len(recent) >= 5:
            returns = recent["usdinr"].pct_change().dropna()
            vol_5d = float(returns.std()) if len(returns) > 1 else 0
        else:
            vol_5d = 0

        # 30d MA
        recent_30 = features_df.loc[mask].tail(30)
        ma_30 = float(recent_30["usdinr"].mean()) if len(recent_30) >= 20 else current_rate

        # XGBoost regime
        current_regime = row.get("regime", "range_bound")

        # Calendar context
        from datetime import date as dt_date
        pred_dt = dt_date.fromisoformat(date_str)
        cal = get_calendar_context(pred_dt)

        # Previous Friday close for Monday gap
        prev_friday_close = 0
        if pred_dt.weekday() == 0:  # Monday
            prev_days = features_df.loc[features_df["date"] < pd.Timestamp(date_str)].tail(3)
            if len(prev_days) > 0:
                prev_friday_close = float(prev_days.iloc[-1]["usdinr"])

        features_dict = {
            "rate_vs_alltime_percentile": float(row.get("rate_vs_alltime_pct", 0)),
            "momentum_consistency": int(row.get("momentum", 0)),
            "volatility_20d": float(row.get("volatility", 0)),
            "volatility_5d": vol_5d,
            "current_rate": current_rate,
            "high_5d": high_5d,
            "low_5d": low_5d,
            "ma_30": ma_30,
            "current_regime": current_regime,
            "prev_regime": prev_regime or current_regime,
            "is_fomc_day": cal.get("is_fomc_day", False),
            "is_rbi_day": cal.get("is_rbi_day", False),
            "day_of_week": pred_dt.weekday(),
            "prev_friday_close": prev_friday_close,
        }

        risk = detect_high_risk_prediction(features_dict)

        factors_short = "; ".join(f[:50] for f in risk["risk_factors"][:2]) if risk["risk_factors"] else "—"

        result_sym = {"GREEN": "G", "YELLOW": "Y", "RED": "R"}.get(row["result"], "?")
        print(f"  {date_str:<12s} {result_sym:>6s} {risk['risk_level']:>9s} {risk['risk_score']:>5d} {factors_short}")

        risk_results.append({
            "date": date_str,
            "result": row["result"],
            "risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"],
            "risk_factors": risk["risk_factors"],
            "recommended_buffer_paise": risk["recommended_buffer_paise"],
        })

        prev_regime = current_regime

    # Correlation analysis
    print(f"\n{'─' * 90}")
    print("  RISK DETECTION CORRELATION ANALYSIS")
    print(f"{'─' * 90}")

    red_days = [r for r in risk_results if r["result"] == "RED"]
    green_days = [r for r in risk_results if r["result"] == "GREEN"]
    yellow_days = [r for r in risk_results if r["result"] == "YELLOW"]

    red_flagged = sum(1 for r in red_days if r["risk_level"] in ("HIGH", "CRITICAL"))
    red_elevated = sum(1 for r in red_days if r["risk_level"] in ("HIGH", "CRITICAL", "ELEVATED"))
    green_normal = sum(1 for r in green_days if r["risk_level"] in ("NORMAL", "MILD"))
    green_elevated = sum(1 for r in green_days if r["risk_level"] in ("ELEVATED", "HIGH", "CRITICAL"))

    print(f"\n  RED days ({len(red_days)} total):")
    print(f"    Flagged HIGH/CRITICAL: {red_flagged}/{len(red_days)} ({red_flagged/len(red_days)*100:.0f}%)")
    print(f"    Flagged ELEVATED+:     {red_elevated}/{len(red_days)} ({red_elevated/len(red_days)*100:.0f}%)")
    for r in red_days:
        print(f"      {r['date']} — {r['risk_level']} (score {r['risk_score']})")

    print(f"\n  GREEN days ({len(green_days)} total):")
    print(f"    NORMAL/MILD:     {green_normal}/{len(green_days)} ({green_normal/len(green_days)*100:.0f}%)")
    print(f"    FALSE ALARMS:    {green_elevated}/{len(green_days)} ({green_elevated/len(green_days)*100:.0f}%)")

    print(f"\n  YELLOW days ({len(yellow_days)} total):")
    for r in yellow_days:
        print(f"      {r['date']} — {r['risk_level']} (score {r['risk_score']})")

    # Summary
    avg_red_score = np.mean([r["risk_score"] for r in red_days]) if red_days else 0
    avg_green_score = np.mean([r["risk_score"] for r in green_days]) if green_days else 0
    avg_yellow_score = np.mean([r["risk_score"] for r in yellow_days]) if yellow_days else 0

    print(f"\n  Average risk score by outcome:")
    print(f"    RED:    {avg_red_score:.1f}")
    print(f"    YELLOW: {avg_yellow_score:.1f}")
    print(f"    GREEN:  {avg_green_score:.1f}")

    # Save for Step 5
    import json
    risk_path = os.path.join(PROJECT_DIR, "backtest", "risk_detection_results.json")
    with open(risk_path, "w") as f:
        json.dump(risk_results, f, indent=2)
    print(f"\n  Risk results saved: {risk_path}")

    return risk_results


if __name__ == "__main__":
    main()
