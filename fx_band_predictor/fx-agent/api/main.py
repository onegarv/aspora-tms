"""
Phase 7 — FastAPI API
Endpoints: GET /health, GET /predict (+ weekly preview, cached 15 min),
           GET /predict/weekly (cached 60 min), GET /backtest?days=90,
           GET /history?days=30
"""

import json
import os, sys, time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_DIR)
from agents.fx_prediction_agent import FXPredictionAgent
from data.fx_calendar import get_next_7_calendar_days, get_calendar_context, get_week_risk_events
from data.ndf_calculator import compute_ndf_features, generate_ndf_interpretation
from data.intraday_features import build_intraday_features

app = FastAPI(title="FX Band Predictor — FX Intelligence Engine", version="1.0.0")

# Branding metadata injected into every response
SYSTEM_META = {
    "system": "FX Band Predictor",
    "tagline": "Sharp money. Sharper timing.",
    "version": "1.0.0",
}
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()
CACHE = {"prediction": None, "timestamp": 0.0}
WEEKLY_CACHE = {"prediction": None, "timestamp": 0.0}
INTRADAY_CACHE = {"prediction": None, "timestamp": 0.0}
CACHE_TTL = 900  # 15 minutes
WEEKLY_CACHE_TTL = 3600  # 60 minutes
INTRADAY_CACHE_TTL = 14400  # 4 hours — refresh at each new 4h bar
DATA_PATH = os.path.join(PROJECT_DIR, "data", "market_data.csv")
SAVED_DIR = os.path.join(PROJECT_DIR, "models", "saved")
BACKTEST_CSV = os.path.join(PROJECT_DIR, "backtest", "forward_validation_daily.csv")

# S3 config (from env)
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "")

agent: FXPredictionAgent | None = None
last_prediction_ts: str = "never"


@app.on_event("startup")
def startup():
    global agent
    print("Loading FX Band Predictor — FX Intelligence Engine...")
    agent = FXPredictionAgent()
    print("FX Band Predictor ready. Sharp money. Sharper timing.\n")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    ms = (time.time() - t0) * 1000
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {request.method} {request.url.path} → {response.status_code} ({ms:.0f}ms)")
    return response


def _error(msg: str, status: int = 500) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "error": msg, "timestamp": datetime.now(timezone.utc).isoformat(),
        "fallback": "Convert operational minimum only"})


def _build_summary(output: dict) -> dict:
    pred = output["prediction_48h"]
    act = output["act_on_signal"]
    if act and pred["direction"] == "UP":
        action, msg = "HOLD", "Rate likely rising — hold USD, wait to convert"
    elif act and pred["direction"] == "DOWN":
        action, msg = "CONVERT_NOW", "Rate likely falling — convert USD to INR now"
    else:
        action, msg = "CONVERT_PARTIAL", "Uncertain — convert operational minimum only"
    return {"action": action, "message": msg,
            "confidence_pct": round(pred["confidence"] * 100, 1),
            "rate": output["current_rate"],
            "range": f"{pred['range_low']:.2f} — {pred['range_high']:.2f}"}


def _build_weekly_preview(output: dict, df: pd.DataFrame = None) -> dict:
    """
    Build a slim weekly_preview from the base prediction + calendar.
    Avoids calling predict_weekly() (which re-runs the full agent).
    Uses the same three-force projection model as predict_weekly().
    """
    try:
        pred = output["prediction_48h"]
        current_rate = output["current_rate"]
        base_dir = pred["direction"]
        base_conf = pred["confidence"]

        xgb = output["model_breakdown"]["xgboost"]
        lstm = output["model_breakdown"]["lstm"]
        sent = output["model_breakdown"]["sentiment"]

        # --- Three-force setup (sentiment-driven drift) ---
        sentiment_score = sent.get("score", 0.0)

        # Sentiment direction drives drift:
        #   score < -0.20 → UP direction → positive drift
        #   score > +0.20 → DOWN direction → negative drift
        if sentiment_score < -0.20:
            momentum_per_day = abs(sentiment_score) * 0.05
        elif sentiment_score > 0.20:
            momentum_per_day = -abs(sentiment_score) * 0.05
        else:
            momentum_per_day = 0.0

        # Mean reversion and trend — need df
        if df is not None and "usdinr" in df.columns and len(df) >= 30:
            ma_30 = float(df["usdinr"].rolling(30).mean().iloc[-1])
            reversion_gap = ma_30 - current_rate
            # Compute rate_trend_30d inline (avoid importing build_features)
            rates = df["usdinr"].tail(30).values
            if len(rates) >= 30:
                x = np.arange(len(rates))
                slope = np.polyfit(x, rates, 1)[0]
                rate_trend_30d = slope / rates.mean() if rates.mean() != 0 else 0.0
            else:
                rate_trend_30d = 0.0
        else:
            reversion_gap = 0.0
            rate_trend_30d = 0.0

        today = date.today()
        days = get_next_7_calendar_days(today)
        DECAY_RATE = 0.90   # multiplicative: each day retains 90%
        CONF_FLOOR = 0.20   # never below 20%

        preview_days = []
        best_day = None
        best_conf = 0.0
        hc_count = 0
        trading_day_n = 1

        for i, d in enumerate(days):
            offset = i + 1
            ctx = get_calendar_context(d)

            if not ctx["is_trading_day"]:
                preview_days.append({
                    "date": d.isoformat(),
                    "day_name": ctx["day_name"],
                    "trading_day": False,
                    "direction": None,
                    "confidence": None,
                    "range_low": None,
                    "range_high": None,
                    "act_on_signal": False,
                    "is_best_day": False,
                    "calendar_note": ctx["skip_reason"],
                })
                continue

            # Multiplicative decay with floor
            decay_base = max(0.35, base_conf)
            conf = max(CONF_FLOOR, decay_base * (DECAY_RATE ** (trading_day_n - 1)))
            trading_day_n += 1
            direction = base_dir

            # Calendar overrides
            if ctx["is_rbi_day"] or ctx["is_fed_day"]:
                direction = "NEUTRAL"
                conf = 0.15
            elif ctx["is_rbi_week"]:
                conf = max(CONF_FLOOR, conf * 0.90)
            elif ctx["is_fed_week"]:
                conf = max(CONF_FLOOR, conf * 0.90)

            if ctx["is_month_end"] and direction == "DOWN":
                conf *= 0.80

            conf = round(conf, 3)

            # Three-force projection (same as predict_weekly)
            momentum_weight = max(0.20, 1.0 - (offset - 1) * 0.13)
            momentum_c = momentum_per_day * offset * momentum_weight
            reversion_weight = min(0.40, offset * 0.06)
            reversion_c = reversion_gap * reversion_weight
            capped_trend = min(0.10, max(-0.10, rate_trend_30d))
            trend_c = capped_trend * offset
            total = max(-2.0, min(2.0, momentum_c + reversion_c + trend_c))
            most_likely = round(current_rate + total, 4)

            range_buffer = 0.20 + (offset * 0.07)
            if momentum_per_day > 0.05:
                day_low = round(most_likely - range_buffer * 0.7, 2)
                day_high = round(most_likely + range_buffer * 1.3, 2)
            elif momentum_per_day < -0.05:
                day_low = round(most_likely - range_buffer * 1.3, 2)
                day_high = round(most_likely + range_buffer * 0.7, 2)
            else:
                day_low = round(most_likely - range_buffer, 2)
                day_high = round(most_likely + range_buffer, 2)

            day_act = conf >= 0.55 and direction != "NEUTRAL"

            if day_act:
                hc_count += 1

            # Treasury enrichment per preview day
            DAILY_VOL = 9_500_000
            tpv_prefund = round(most_likely * DAILY_VOL)
            rate_risk = round(day_high * DAILY_VOL) - round(day_low * DAILY_VOL)

            if direction == "DOWN" and day_act:
                t_note = f"Rate falling — prefund at {most_likely:.2f}"
            elif direction == "UP" and day_act:
                t_note = f"Rate rising — hold dollars, convert later"
            elif ctx.get("is_rbi_day") or ctx.get("is_fed_day"):
                t_note = "Central bank day — caution"
            else:
                t_note = f"No strong signal — prefund at {day_high:.2f}"

            preview_days.append({
                "date": d.isoformat(),
                "day_name": ctx["day_name"],
                "trading_day": True,
                "direction": direction,
                "confidence": conf,
                "range_low": day_low,
                "range_high": day_high,
                "most_likely": round(most_likely, 2),
                "act_on_signal": day_act,
                "is_best_day": False,  # set below
                "treasury_note": t_note,
                "prefunding_rate": round(most_likely, 2),
                "tpv_at_prefunding_rate": tpv_prefund,
                "rate_risk_inr": rate_risk,
                "calendar_note": ctx["special_note"],
            })

            # Track best UP day
            if direction == "UP" and conf > best_conf:
                best_conf = conf
                best_day = d.isoformat()

        # Mark best day
        if best_day:
            for day in preview_days:
                if day["date"] == best_day:
                    day["is_best_day"] = True

        # Overall bias
        scores = []
        for day in preview_days:
            if not day["trading_day"]:
                continue
            d_dir = day["direction"]
            d_conf = day["confidence"]
            if d_dir == "UP":
                scores.append(d_conf)
            elif d_dir == "DOWN":
                scores.append(-d_conf)
            else:
                scores.append(0.0)

        if scores:
            avg = sum(scores) / len(scores)
            if avg > 0.05:
                bias = "UP"
            elif avg < -0.05:
                bias = "DOWN"
            else:
                bias = "NEUTRAL"
        else:
            bias = "NEUTRAL"

        return {
            "days": preview_days,
            "best_conversion_day": best_day,
            "week_bias": bias,
            "high_confidence_days_count": hc_count,
        }
    except Exception as e:
        return {"days": [], "best_conversion_day": None,
                "week_bias": "NEUTRAL", "high_confidence_days_count": 0,
                "error": str(e)}


def _build_ndf_context(df: pd.DataFrame) -> dict | None:
    """Compute NDF context from market data — informational only, not a model input."""
    try:
        ndf_df = compute_ndf_features(df)
        latest = ndf_df.iloc[-1]
        ndf_data = {
            "india_repo_rate": round(float(latest["india_repo_rate"]), 2),
            "us_10y_yield": round(float(latest["us10y"]), 3),
            "rate_differential": round(float(latest["rate_differential"]), 3),
            "forward_premium_paise": round(float(latest["forward_premium_paise"]), 1),
            "forward_premium_annualized_pct": round(float(latest["forward_premium_annualized"]), 2),
            "carry_regime": (
                "carry_positive" if latest["carry_regime"] == 1
                else "carry_neutral" if latest["carry_regime"] == 0
                else "carry_under_pressure"
            ),
            "ndf_momentum_7d": round(float(latest["ndf_momentum_7d"]), 1) if pd.notna(latest["ndf_momentum_7d"]) else None,
        }
        ndf_data["interpretation"] = generate_ndf_interpretation(ndf_data)
        return ndf_data
    except Exception:
        return None


# --- GET /health ---

@app.get("/health")
def health():
    return {
        **SYSTEM_META,
        "status": "operational",
        "models_loaded": 5 if agent else 0,
        "model_list": agent.models_loaded if agent else [],
        "last_prediction": last_prediction_ts,
        "uptime_seconds": int(time.time() - START_TIME),
    }


# --- GET /predict ---

@app.get("/predict")
def predict():
    global last_prediction_ts
    try:
        now = time.time()
        if CACHE["prediction"] and (now - CACHE["timestamp"]) < CACHE_TTL:
            result = CACHE["prediction"].copy()
            result["cache_hit"] = True
            return result
        if agent is None:
            return _error("Agent not initialized", 503)
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        output = agent.predict(df)
        output = {**SYSTEM_META, **output}
        output["summary"] = _build_summary(output)
        output["weekly_preview"] = _build_weekly_preview(output, df)

        # NDF context with Opus interpretation merge
        ndf = _build_ndf_context(df)
        ctx_interp = output.pop("_context_interpretations", {})
        if ndf and ctx_interp.get("ndf_interpretation"):
            ndf["opus_interpretation"] = ctx_interp["ndf_interpretation"]
        output["ndf_context"] = ndf

        # Macro context enrichment (from existing model_breakdown + Opus)
        macro_bd = output.get("model_breakdown", {}).get("macro", {})
        if macro_bd.get("available"):
            output["macro_context"] = {
                "direction": macro_bd.get("direction"),
                "score": macro_bd.get("score"),
                "fed_funds_change_3m": macro_bd.get("fed_funds_change_3m"),
                "yield_curve_spread": macro_bd.get("yield_curve_spread"),
                "cpi_yoy": macro_bd.get("cpi_yoy"),
                "interpretation": macro_bd.get("interpretation"),
                "opus_interpretation": ctx_interp.get("macro_interpretation"),
            }

        output["cache_hit"] = False
        CACHE["prediction"] = output
        CACHE["timestamp"] = now
        last_prediction_ts = datetime.now(timezone.utc).isoformat()
        return output
    except Exception as e:
        return _error(str(e))


# --- GET /predict/weekly ---

@app.get("/predict/weekly")
def predict_weekly():
    global last_prediction_ts
    try:
        now = time.time()
        if WEEKLY_CACHE["prediction"] and (now - WEEKLY_CACHE["timestamp"]) < WEEKLY_CACHE_TTL:
            result = WEEKLY_CACHE["prediction"].copy()
            result["cache_hit"] = True
            return result
        if agent is None:
            return _error("Agent not initialized", 503)
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        output = agent.predict_weekly(df)
        output = {**SYSTEM_META, **output}
        output["cache_hit"] = False
        WEEKLY_CACHE["prediction"] = output
        WEEKLY_CACHE["timestamp"] = now
        last_prediction_ts = datetime.now(timezone.utc).isoformat()
        return output
    except Exception as e:
        return _error(str(e))


# --- GET /history ---

def _fetch_s3_history(days: int) -> list[dict] | None:
    """Try to read prediction history from S3. Returns None if unavailable."""
    if not S3_BUCKET:
        return None
    try:
        import boto3
        s3 = boto3.client("s3", region_name=AWS_REGION)
        today = date.today()
        predictions = []

        for i in range(days):
            d = today - timedelta(days=i)
            key = d.strftime("predictions/%Y/%m/%d/prediction.json")
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                data = json.loads(obj["Body"].read().decode("utf-8"))
                pred_48h = data.get("prediction_48h", {})
                predictions.append({
                    "date": d.isoformat(),
                    "rate": data.get("current_rate"),
                    "direction": pred_48h.get("direction", "NEUTRAL"),
                    "confidence": pred_48h.get("confidence", 0.0),
                    "acted": data.get("act_on_signal", False),
                    "actual_rate_next_day": None,  # filled in post-processing
                    "was_correct": None,
                })
            except s3.exceptions.NoSuchKey:
                continue
            except Exception:
                continue

        if not predictions:
            return None

        # Post-process: fill in actual_rate_next_day from the next day's prediction
        by_date = {p["date"]: p for p in predictions}
        for p in predictions:
            d = date.fromisoformat(p["date"])
            next_d = (d + timedelta(days=1)).isoformat()
            if next_d in by_date and by_date[next_d]["rate"] is not None:
                p["actual_rate_next_day"] = by_date[next_d]["rate"]
                if p["direction"] == "UP":
                    p["was_correct"] = by_date[next_d]["rate"] > p["rate"]
                elif p["direction"] == "DOWN":
                    p["was_correct"] = by_date[next_d]["rate"] < p["rate"]
                else:
                    p["was_correct"] = None

        predictions.sort(key=lambda x: x["date"])
        return predictions
    except Exception:
        return None


def _mock_history_from_backtest(days: int) -> list[dict]:
    """Fall back to forward_validation_daily.csv for mock history data."""
    try:
        df = pd.read_csv(BACKTEST_CSV)
        df = df.tail(days).reset_index(drop=True)

        predictions = []
        for _, row in df.iterrows():
            # Use ensemble direction (model's actual call) — gated is always NEUTRAL
            # when safety gate blocks, which isn't useful for history display
            direction = row.get("ensemble_direction", "NEUTRAL")
            acted = str(row.get("act_on_signal", "False")).lower() == "true"
            conf = row.get("xgb_confidence", 0.0)

            # Determine correctness using ensemble direction
            was_correct = None
            if direction in ("UP", "DOWN"):
                actual_dir = row.get("actual_direction", "")
                if actual_dir in ("UP", "DOWN", "NEUTRAL"):
                    was_correct = (direction == actual_dir)

            predictions.append({
                "date": row["date"],
                "rate": round(float(row["entry_rate"]), 4),
                "direction": direction,
                "confidence": round(float(conf), 3),
                "acted": acted,
                "actual_rate_next_day": round(float(row["actual_rate_2d"]), 4) if pd.notna(row.get("actual_rate_2d")) else None,
                "was_correct": was_correct,
            })

        return predictions
    except Exception:
        return []


def _compute_accuracy(predictions: list[dict]) -> dict:
    """Compute accuracy summary from a list of prediction records."""
    directional = [p for p in predictions if p["direction"] in ("UP", "DOWN")]
    acted = [p for p in directional if p["acted"]]
    evaluated = [p for p in directional if p["was_correct"] is not None]
    correct = [p for p in evaluated if p["was_correct"]]

    return {
        "signals_generated": len(directional),
        "signals_acted": len(acted),
        "correct": len(correct),
        "total_evaluated": len(evaluated),
        "accuracy_pct": round(len(correct) / len(evaluated) * 100, 1) if evaluated else 0.0,
    }


@app.get("/history")
def history(days: int = Query(default=30, ge=1, le=90)):
    try:
        # Try S3 first — need at least 3 days to be useful
        predictions = _fetch_s3_history(days)
        source = "s3"

        # Fall back to backtest CSV if S3 unavailable or too sparse
        if predictions is None or len(predictions) < 3:
            predictions = _mock_history_from_backtest(days)
            source = "backtest_csv" if predictions else "none"

        accuracy = _compute_accuracy(predictions)

        return {
            **SYSTEM_META,
            "days": days,
            "source": source,
            "predictions": predictions,
            "accuracy": accuracy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return _error(str(e))


# --- GET /backtest ---

@app.get("/backtest")
def backtest(days: int = Query(default=90, ge=7, le=365)):
    try:
        xgb_df = pd.read_csv(os.path.join(SAVED_DIR, "xgb_test_results.csv"))
        lstm_df = pd.read_csv(os.path.join(SAVED_DIR, "lstm_test_results.csv"))
        xgb_df = xgb_df.tail(days).reset_index(drop=True)
        lstm_df = lstm_df.tail(days).reset_index(drop=True)

        label_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
        xgb_df["actual_dir"] = xgb_df["actual"].map(label_map)
        xgb_df["pred_dir"] = xgb_df["predicted"].map(label_map)

        # All high-confidence signals (including NEUTRAL — headline metric)
        hc_all = xgb_df[xgb_df["max_confidence"] > 0.65]
        hc_all_correct = int((hc_all["actual"] == hc_all["predicted"]).sum()) if len(hc_all) else 0
        hc_all_acc = (hc_all_correct / len(hc_all) * 100) if len(hc_all) else 0.0

        # High-confidence directional signals (UP/DOWN only — actionable)
        non_neutral_hc = hc_all[hc_all["pred_dir"] != "NEUTRAL"]
        total_signals = len(xgb_df[xgb_df["pred_dir"] != "NEUTRAL"])
        hc_signals = len(non_neutral_hc)
        correct_hc = int((non_neutral_hc["actual"] == non_neutral_hc["predicted"]).sum()) if hc_signals else 0
        dir_acc = (correct_hc / hc_signals * 100) if hc_signals else 0.0

        # Confidence split: correct vs wrong
        if hc_signals:
            correct_mask = non_neutral_hc["actual"] == non_neutral_hc["predicted"]
            avg_conf_correct = float(non_neutral_hc.loc[correct_mask, "max_confidence"].mean()) if correct_mask.any() else 0.0
            avg_conf_wrong = float(non_neutral_hc.loc[~correct_mask, "max_confidence"].mean()) if (~correct_mask).any() else 0.0
        else:
            avg_conf_correct = avg_conf_wrong = 0.0

        # Range accuracy from LSTM
        range_acc = 0.0
        if {"actual_close", "pred_low", "pred_high"}.issubset(lstm_df.columns):
            in_range = (lstm_df["actual_close"] >= lstm_df["pred_low"]) & (lstm_df["actual_close"] <= lstm_df["pred_high"])
            range_acc = in_range.mean() * 100

        # P&L: average paise gained on correct HOLD decisions
        DAILY_VOLUME_USD = 9_500_000
        avg_improvement = 0.0
        if {"actual_close", "entry_rate"}.issubset(lstm_df.columns) and len(lstm_df) > 0:
            deltas = lstm_df["actual_close"] - lstm_df["entry_rate"]
            positive = deltas[deltas > 0]
            avg_improvement = float(positive.mean() * 100) if len(positive) > 0 else 0.0

        signals_per_month = round(hc_signals / max(days, 1) * 30, 1)
        saving_per_signal = avg_improvement * DAILY_VOLUME_USD / 100
        monthly_saving = saving_per_signal * signals_per_month
        annual_saving = monthly_saving * 12
        current_rate_approx = 85.0
        if {"entry_rate"}.issubset(lstm_df.columns) and len(lstm_df) > 0:
            current_rate_approx = float(lstm_df["entry_rate"].iloc[-1])
        daily_volume_inr = int(DAILY_VOLUME_USD * current_rate_approx)
        cost_per_call = 0.0005
        monthly_cost_inr = int(cost_per_call * 22 * current_rate_approx)
        roi = f"{int(monthly_saving / max(monthly_cost_inr, 1)):,}x" if monthly_cost_inr > 0 else "N/A"

        return {**SYSTEM_META, "backtest": {
            "period_days": days,
            "high_confidence_all_accuracy_pct": round(hc_all_acc, 1),
            "high_confidence_all_count": len(hc_all),
            "total_directional_signals": int(total_signals),
            "high_confidence_directional_signals": int(hc_signals),
            "signals_acted_on": int(hc_signals),
            "correct_direction": correct_hc,
            "directional_accuracy_pct": round(dir_acc, 1),
            "range_accuracy_pct": round(range_acc, 1),
            "avg_confidence_when_correct": round(avg_conf_correct, 3),
            "avg_confidence_when_wrong": round(avg_conf_wrong, 3),
            "avg_rate_improvement_paise": round(avg_improvement, 1),
            "naive_strategy": "Convert everything daily at open",
            "our_strategy": "Convert on high-confidence signals only"},
            "financial_impact": {
                "daily_volume_usd": DAILY_VOLUME_USD,
                "daily_volume_inr_approx": daily_volume_inr,
                "avg_gain_per_correct_signal_paise": round(avg_improvement, 1),
                "avg_gain_per_correct_signal_inr": int(saving_per_signal),
                "signals_per_month": signals_per_month,
                "estimated_monthly_saving_inr": int(monthly_saving),
                "estimated_annual_saving_inr": int(annual_saving),
                "estimated_annual_saving_crore": round(annual_saving / 1_00_00_000, 1),
                "cost_to_run_monthly_inr": monthly_cost_inr,
                "roi": roi,
            },
            "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return _error(str(e))


# --- GET /intraday ---

INTRADAY_CSV = os.path.join(PROJECT_DIR, "data", "intraday_4h.csv")


@app.get("/intraday")
def intraday():
    try:
        now = time.time()
        if INTRADAY_CACHE["prediction"] and (now - INTRADAY_CACHE["timestamp"]) < INTRADAY_CACHE_TTL:
            result = INTRADAY_CACHE["prediction"].copy()
            result["cache_hit"] = True
            return result
        if agent is None:
            return _error("Agent not initialized", 503)

        # Load 4h data
        if not os.path.exists(INTRADAY_CSV):
            return _error("Intraday data not available. Run data/fetch_intraday.py first.", 404)

        df_4h = pd.read_csv(INTRADAY_CSV, index_col="Datetime", parse_dates=True)

        # Run intraday model
        intraday_result = agent._run_intraday_lstm(df_4h)

        # Build features for last_24h_bars display
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            df_features, _ = build_intraday_features(df_4h)

        current_rate = float(df_4h["Close"].iloc[-1])
        last6 = df_features.tail(6)

        # Last 24h bars detail
        last_24h_bars = []
        for ts, row in last6.iterrows():
            last_24h_bars.append({
                "timestamp": ts.isoformat(),
                "open": round(float(df_4h.loc[ts, "Open"]) if ts in df_4h.index else 0, 4),
                "high": round(float(df_4h.loc[ts, "High"]) if ts in df_4h.index else 0, 4),
                "low": round(float(df_4h.loc[ts, "Low"]) if ts in df_4h.index else 0, 4),
                "close": round(float(row["Close"]), 4),
                "session": row.get("session", "off"),
                "ret_4h": round(float(row["ret_4h"]), 6),
                "rsi_4h": round(float(row["rsi_4h"]), 1),
                "momentum_consistency_24h": int(row["momentum_consistency_24h"]),
            })

        # Session summary — last bar from each session
        session_summary = {}
        for sess_name in ["asian", "london", "newyork"]:
            sess_bars = last6[last6["session"] == sess_name]
            if len(sess_bars) > 0:
                last_bar = sess_bars.iloc[-1]
                move = float(last_bar["ret_4h"])
                session_summary[sess_name] = {
                    "last_move": f"{move:+.4f}",
                    "direction": "UP" if move > 0 else "DOWN" if move < 0 else "FLAT",
                }
            else:
                session_summary[sess_name] = {"last_move": "N/A", "direction": "N/A"}

        # Momentum and interpretation
        mom_24h = int(df_features["momentum_consistency_24h"].iloc[-1])
        sessions_up = sum(1 for s in session_summary.values() if s["direction"] == "UP")
        sessions_down = sum(1 for s in session_summary.values() if s["direction"] == "DOWN")

        if sessions_up > sessions_down:
            interp = "Majority sessions positive — upward intraday pressure"
        elif sessions_down > sessions_up:
            interp = "Majority sessions negative — downward intraday pressure"
        else:
            interp = "Mixed sessions — no clear intraday direction"

        # Identify strongest session move
        strongest = max(session_summary.items(),
                       key=lambda x: abs(float(x[1]["last_move"])) if x[1]["last_move"] != "N/A" else 0)
        if strongest[1]["last_move"] != "N/A":
            move_val = float(strongest[1]["last_move"])
            direction_word = "upward" if move_val > 0 else "downward"
            interp += f" | {strongest[0].title()} session drove {direction_word} move"

        output = {
            **SYSTEM_META,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "current_rate": round(current_rate, 4),
            "intraday_signal": {
                "direction": intraday_result["signal"],
                "raw_prediction": intraday_result["raw_direction"],
                "confidence": intraday_result["raw_confidence"],
                "integrated_into_ensemble": intraday_result.get("integrated_into_ensemble", False),
                "confidence_adjustment_applied": intraday_result["confidence_adjustment"],
            },
            "last_24h_bars": last_24h_bars,
            "session_summary": session_summary,
            "momentum_consistency_24h": mom_24h,
            "interpretation": interp,
            "cache_hit": False,
        }

        INTRADAY_CACHE["prediction"] = output
        INTRADAY_CACHE["timestamp"] = now
        return output
    except Exception as e:
        return _error(str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
