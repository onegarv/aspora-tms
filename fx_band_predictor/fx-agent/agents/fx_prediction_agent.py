"""
Phase 6 — FX Prediction Agent (Ensemble + Safety Gate)

The brain of the system. Loads all models, runs them on live data,
combines outputs, applies the safety gate, and returns a structured prediction JSON.

Architecture:
    1. XGBoost  — REGIME CLASSIFIER (detects market regime, adjusts confidence)
    2. LSTM     — RANGE PROVIDER (predicted high/low/close; direction always NEUTRAL)
    3. Sentiment — PRIMARY DIRECTION SIGNAL (news-driven)
    4. Macro    — SECONDARY DIRECTION SIGNAL (FRED rule-based fundamentals)

Direction voting:
    Vote 1: Sentiment (primary — news driven)
    Vote 2: Macro signal (secondary — fundamentals from FRED)
    Context: XGBoost regime (confidence modifier)
    Range: LSTM (range provider)

    If sentiment and macro agree → use that direction
    If sentiment and macro disagree → NEUTRAL
    If one is NEUTRAL → use the other with reduced confidence

Confidence:
    Both agree:     sentiment_conf × 0.55 + macro_conf × 0.35 + regime_adj
    Only sentiment: sentiment_conf × 0.70 + regime_adj
    Only macro:     macro_conf × 0.50 + regime_adj
    Neither:        0.35 (baseline)

act_on_signal = confidence > 0.55 AND direction != NEUTRAL
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import joblib
import requests as _requests
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, PROJECT_DIR)

from features.feature_engineering import build_features, get_latest_features
from agents.sentiment_agent import get_sentiment, sentiment_to_direction
from agents.macro_signal import get_macro_signal
from models.train_lstm import LSTMRangePredictor
from models.train_intraday_lstm import IntradayLSTM, FEATURE_COLS as INTRADAY_FEATURE_COLS
from data.intraday_features import build_intraday_features
from agents.range_risk_detector import detect_high_risk_prediction
from data.fx_calendar import (
    get_calendar_context, get_next_7_calendar_days,
    get_week_risk_events,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAVED_DIR = os.path.join(PROJECT_DIR, "models", "saved")

# Regime classifier paths (XGBoost classifies market regime, not direction)
XGB_REGIME_MODEL_PATH = os.path.join(SAVED_DIR, "xgb_regime.pkl")
XGB_REGIME_FEATURES_PATH = os.path.join(SAVED_DIR, "feature_names_regime.pkl")
XGB_REGIME_IMPORTANCE_PATH = os.path.join(SAVED_DIR, "feature_importance_regime.csv")

# LSTM paths (range provider — direction always NEUTRAL)
# v2-boundary: BoundaryAwareLoss + LayerNorm + asymmetric upside (walkforward winner)
LSTM_MODEL_PATH_V2_BOUNDARY = os.path.join(SAVED_DIR, "lstm_walkforward_v2.pt")
LSTM_SCALER_PATH_V2_BOUNDARY = os.path.join(SAVED_DIR, "lstm_scaler_walkforward_v2.pkl")
# v2: 12-feature model trained on full history
LSTM_MODEL_PATH_V2 = os.path.join(SAVED_DIR, "lstm_range_v2.pt")
LSTM_SCALER_PATH_V2 = os.path.join(SAVED_DIR, "lstm_scaler_v2.pkl")
# v1: 5-feature baseline
LSTM_MODEL_PATH = os.path.join(SAVED_DIR, "lstm_range.pt")
LSTM_SCALER_PATH = os.path.join(SAVED_DIR, "lstm_scaler.pkl")

# Intraday LSTM paths (momentum classifier — asymmetric UP-only signal)
INTRADAY_MODEL_PATH = os.path.join(SAVED_DIR, "intraday_lstm.pt")
INTRADAY_SCALER_PATH = os.path.join(SAVED_DIR, "intraday_scaler.pkl")
INTRADAY_META_PATH = os.path.join(SAVED_DIR, "intraday_lstm_meta.pkl")
INTRADAY_SEQ_LEN = 24

# Raw market data columns (v1 LSTM uses these directly)
RAW_MARKET_COLS = {"usdinr", "oil", "dxy", "vix", "us10y"}

# Regime map
REGIME_MAP = {0: "trending_up", 1: "trending_down", 2: "high_vol", 3: "range_bound"}

LABEL_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

# Human-readable feature name mapping for key_drivers
FEATURE_DESCRIPTIONS = {
    "rate_change_1d": ("Rate moved {val:+.2%} today", None),
    "rate_change_3d": ("Rate moved {val:+.2%} over 3 days", None),
    "rate_change_5d": ("Rate moved {val:+.2%} over 5 days", None),
    "rate_vs_7d_avg": ("Rate {dir} vs 7-day average by {abs_val:.2%}", "strengthening|weakening"),
    "rate_vs_30d_avg": ("Rate {dir} vs 30-day average by {abs_val:.2%}", "strengthening|weakening"),
    "rsi_14": ("RSI at {val:.0f} ({zone})", "overbought|oversold|neutral"),
    "macd_histogram": ("MACD histogram {dir} ({val:+.4f})", "bullish|bearish"),
    "bb_position": ("Bollinger position at {val:.2f} ({zone})", None),
    "volatility_5d": ("5-day volatility at {val:.2%}", None),
    "volatility_20d": ("20-day volatility at {val:.2%}", None),
    "oil_change_1d": ("Oil {dir} {abs_val:.1%} today", "up|down"),
    "oil_vs_30d_avg": ("Oil {dir} vs 30-day average by {abs_val:.1%}", "above|below"),
    "dxy_change_1d": ("DXY {dir} {abs_val:.1%} today", "up|down"),
    "dxy_vs_7d_avg": ("USD {dir} vs 7-day average", "strengthening|weakening"),
    "vix_level": ("VIX fear index at {val:.1f}", None),
    "vix_change_1d": ("VIX {dir} {abs_val:.1%} today", "up|down"),
    "us10y_change_1d": ("US 10Y yield {dir} {abs_val:.3f}", "up|down"),
    "us10y_level": ("US 10Y yield at {val:.3f}%", None),
    "oil_dxy_divergence": ("Oil-DXY divergence: {val:+.3f}", None),
    "yield_spread_proxy": ("Yield spread proxy at {val:.3f}", None),
    "is_rbi_week": ("RBI meeting approaching", None),
    "is_fed_week": ("Fed meeting approaching", None),
    "is_month_end": ("Month-end flows expected", None),
    "days_to_next_rbi": ("RBI meeting in {val:.0f} days", None),
    "days_to_next_fed": ("Fed meeting in {val:.0f} days", None),
    "high_vol_regime": ("High volatility regime active", None),
}


def _format_driver(feature_name: str, value: float) -> str:
    """Convert a feature name + value into a human-readable string."""
    if feature_name not in FEATURE_DESCRIPTIONS:
        return f"{feature_name}: {value:.4f}"

    template, directions = FEATURE_DESCRIPTIONS[feature_name]

    abs_val = abs(value)

    # Direction words
    if directions:
        parts = directions.split("|")
        if len(parts) == 2:
            direction = parts[0] if value > 0 else parts[1]
        else:
            direction = parts[0]
    else:
        direction = ""

    # Special zones for RSI
    if feature_name == "rsi_14":
        if value > 70:
            zone = "overbought"
        elif value < 30:
            zone = "oversold"
        elif value > 55:
            zone = "mild bullish momentum"
        elif value < 45:
            zone = "mild bearish momentum"
        else:
            zone = "neutral"
        return template.format(val=value, zone=zone)

    # Special zones for Bollinger
    if feature_name == "bb_position":
        if value > 0.85:
            zone = "near upper band"
        elif value < 0.15:
            zone = "near lower band"
        else:
            zone = "mid-band"
        return template.format(val=value, zone=zone)

    # Binary features
    if feature_name in ("is_rbi_week", "is_fed_week",
                         "is_month_end", "high_vol_regime"):
        return template if value > 0 else ""

    try:
        return template.format(val=value, abs_val=abs_val, dir=direction)
    except (KeyError, ValueError):
        return f"{feature_name}: {value:.4f}"


# ===========================================================================
# FXPredictionAgent
# ===========================================================================

class FXPredictionAgent:
    """
    Ensemble FX prediction agent. Loads trained models on init,
    produces structured 48h USD/INR predictions.
    """

    def __init__(self):
        """Load all models, scalers, and metadata."""
        self.models_loaded = []
        self.model_versions = {}

        # --- XGBoost Regime Classifier ---
        try:
            self.xgb_model = joblib.load(XGB_REGIME_MODEL_PATH)
            self.xgb_feature_names = joblib.load(XGB_REGIME_FEATURES_PATH)
            self.xgb_importance = pd.read_csv(XGB_REGIME_IMPORTANCE_PATH)
            self.model_versions["xgboost"] = "regime"
            self.models_loaded.append("xgboost")
            print(f"  [init] XGBoost regime classifier loaded ({len(self.xgb_feature_names)} features)")
        except Exception as e:
            print(f"  [init] WARNING: Failed to load XGBoost regime classifier: {e}")
            self.xgb_model = None
            self.xgb_feature_names = []
            self.xgb_importance = pd.DataFrame()
            self.model_versions["xgboost"] = "none"

        # --- LSTM (try v2-boundary first, then v2, then v1) ---
        try:
            self.lstm_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            if os.path.exists(LSTM_MODEL_PATH_V2_BOUNDARY):
                # v2-boundary: BoundaryAwareLoss + LayerNorm (walkforward winner)
                from backtest.lstm_v2_retrain import LSTMRangePredictorV2
                checkpoint = torch.load(LSTM_MODEL_PATH_V2_BOUNDARY, map_location=self.lstm_device, weights_only=False)
                self.lstm_scaler = joblib.load(LSTM_SCALER_PATH_V2_BOUNDARY)
                self.lstm_model = LSTMRangePredictorV2(input_dim=checkpoint["input_dim"])
                self.model_versions["lstm"] = "v2_boundary_optimized"
                version_label = "v2-boundary"
                # Optimal config: P90 buffer + asymmetric upside
                self.lstm_range_buffer = checkpoint.get("optimal_range_buffer",
                                                        checkpoint.get("range_buffer", 0.1775))
                self.lstm_asymmetric_upside = checkpoint.get("optimal_asymmetric_upside", 0.0888)
            elif os.path.exists(LSTM_MODEL_PATH_V2):
                checkpoint = torch.load(LSTM_MODEL_PATH_V2, map_location=self.lstm_device, weights_only=False)
                self.lstm_scaler = joblib.load(LSTM_SCALER_PATH_V2)
                self.lstm_model = LSTMRangePredictor(input_dim=checkpoint["input_dim"])
                self.model_versions["lstm"] = "v2"
                version_label = "v2"
                self.lstm_range_buffer = checkpoint.get("range_buffer", 0.27)
                self.lstm_asymmetric_upside = 0.0
            else:
                checkpoint = torch.load(LSTM_MODEL_PATH, map_location=self.lstm_device, weights_only=False)
                self.lstm_scaler = joblib.load(LSTM_SCALER_PATH)
                self.lstm_model = LSTMRangePredictor(input_dim=checkpoint["input_dim"])
                self.model_versions["lstm"] = "v1"
                version_label = "v1"
                self.lstm_range_buffer = checkpoint.get("range_buffer", 0.27)
                self.lstm_asymmetric_upside = 0.0

            self.lstm_model.load_state_dict(checkpoint["model_state_dict"])
            self.lstm_model.to(self.lstm_device)
            self.lstm_model.eval()
            self.lstm_seq_len = checkpoint["seq_len"]
            self.lstm_features = checkpoint["features"]
            # Check if LSTM needs engineered features (v2) vs raw market data (v1)
            self.lstm_needs_features = not set(self.lstm_features).issubset(RAW_MARKET_COLS)
            self.models_loaded.append("lstm")
            asym_note = f", asym_up={self.lstm_asymmetric_upside:.4f}" if self.lstm_asymmetric_upside > 0 else ""
            print(f"  [init] LSTM {version_label} loaded (seq={self.lstm_seq_len}, "
                  f"input_dim={checkpoint['input_dim']}, buffer={self.lstm_range_buffer:.4f}{asym_note})")
        except Exception as e:
            print(f"  [init] WARNING: Failed to load LSTM: {e}")
            import traceback; traceback.print_exc()
            self.lstm_model = None
            self.lstm_needs_features = False
            self.lstm_asymmetric_upside = 0.0
            self.model_versions["lstm"] = "none"

        # --- Sentiment (no model to load — runs live) ---
        self.models_loaded.append("sentiment")
        self.model_versions["sentiment"] = "live"
        print(f"  [init] Sentiment agent ready")

        # --- Intraday LSTM (momentum classifier — asymmetric UP-only) ---
        try:
            if os.path.exists(INTRADAY_MODEL_PATH):
                self.intraday_model = IntradayLSTM(input_dim=len(INTRADAY_FEATURE_COLS))
                state_dict = torch.load(INTRADAY_MODEL_PATH, map_location=self.lstm_device, weights_only=True)
                self.intraday_model.load_state_dict(state_dict)
                self.intraday_model.to(self.lstm_device)
                self.intraday_model.eval()
                self.intraday_scaler = joblib.load(INTRADAY_SCALER_PATH)
                self.intraday_meta = joblib.load(INTRADAY_META_PATH)
                self.models_loaded.append("intraday_lstm")
                self.model_versions["intraday_lstm"] = "v1"
                print(f"  [init] Intraday LSTM loaded (seq={INTRADAY_SEQ_LEN}, "
                      f"features={len(INTRADAY_FEATURE_COLS)}, "
                      f"accuracy={self.intraday_meta.get('overall_accuracy', 0)*100:.1f}%)")
            else:
                self.intraday_model = None
                self.model_versions["intraday_lstm"] = "none"
                print(f"  [init] Intraday LSTM not found — skipping")
        except Exception as e:
            print(f"  [init] WARNING: Failed to load intraday LSTM: {e}")
            self.intraday_model = None
            self.model_versions["intraday_lstm"] = "none"

        # --- OpenRouter client (for Opus reasoning layer) ---
        # Fallback chain: Opus 4.6 → Sonnet 4.6 → Sonnet 3.5 → Haiku
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        if self.openrouter_api_key:
            self.reasoning_model_chain = [
                "anthropic/claude-opus-4.6",
                "anthropic/claude-sonnet-4.6",
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-3-haiku",
            ]
            print(f"  [init] OpenRouter reasoning client ready")
        else:
            print(f"  [init] WARNING: OPENROUTER_API_KEY not set — reasoning disabled")
            self.reasoning_model_chain = []

        print(f"  [init] Models loaded: {self.models_loaded}")
        print(f"  [init] Versions: {self.model_versions}")

    # -----------------------------------------------------------------------
    # Model 1: XGBoost Regime Classifier
    # -----------------------------------------------------------------------

    def _run_xgboost(self, df: pd.DataFrame, features_df: pd.DataFrame = None) -> dict:
        """
        Run XGBoost regime classifier. Returns regime context, NOT direction.

        Regime adjustment modifies LSTM confidence:
          trending_up + LSTM UP:   +0.08
          trending_up + LSTM DOWN: -0.08
          trending_down + LSTM DOWN: +0.08
          trending_down + LSTM UP: -0.08
          range_bound: 0.0 (uncertain)
          high_vol: -0.10 (reduce all confidence)
        """
        fallback = {
            "regime": "range_bound",
            "regime_confidence": 0.0,
            "regime_probs": {},
            "contributes_to_vote": False,
            "error": None,
        }

        if self.xgb_model is None:
            fallback["error"] = "model_not_loaded"
            return fallback

        try:
            if features_df is None:
                features_df = build_features(df)
            latest = features_df.iloc[[-1]]
            X = latest[self.xgb_feature_names].values

            proba = self.xgb_model.predict_proba(X)[0]
            pred_class = int(np.argmax(proba))
            regime = REGIME_MAP[pred_class]
            confidence = float(proba.max())

            # Track regime for risk detector
            self._prev_xgb_regime = getattr(self, "_last_xgb_regime", regime)
            self._last_xgb_regime = regime

            return {
                "regime": regime,
                "regime_confidence": round(confidence, 4),
                "regime_probs": {REGIME_MAP[i]: round(float(proba[i]), 4)
                                 for i in range(len(proba))},
                "contributes_to_vote": False,
                "error": None,
            }
        except Exception as e:
            print(f"  [xgboost-regime] ERROR: {e}")
            fallback["error"] = str(e)
            return fallback

    # -----------------------------------------------------------------------
    # Model 2: LSTM Range
    # -----------------------------------------------------------------------

    def _run_lstm(self, df: pd.DataFrame, features_df: pd.DataFrame = None) -> dict:
        """
        Run LSTM range predictor on last 30 days of data.

        LSTM provides price RANGE only — direction always NEUTRAL.
        Close-delta converges to near-zero for any properly trained model
        on 2-day FX returns. The value is in high/low range bounds.

        v2-boundary model uses BoundaryAwareLoss + asymmetric upside buffer.
        """
        fallback = {
            "direction": "NEUTRAL",
            "role": "range_provider",
            "model_version": self.model_versions.get("lstm", "none"),
            "predicted_high": None, "predicted_low": None,
            "predicted_close": None, "range_width": 0.0,
            "range_width_label": "N/A",
            "range_risk": detect_high_risk_prediction({}),
            "note": "LSTM provides price range. Direction from sentiment.",
        }

        if self.lstm_model is None:
            fallback["error"] = "model_not_loaded"
            return fallback

        try:
            current_rate = float(df["usdinr"].iloc[-1])

            # Get the right data source for LSTM features
            if self.lstm_needs_features:
                if features_df is None:
                    features_df = build_features(df)
                source_df = features_df
            else:
                source_df = df

            # Get last seq_len rows
            tail = source_df[self.lstm_features].tail(self.lstm_seq_len).values
            if len(tail) < self.lstm_seq_len:
                fallback["error"] = f"need {self.lstm_seq_len} rows, got {len(tail)}"
                return fallback

            # Scale using training scaler
            scaled = self.lstm_scaler.transform(tail)
            X = torch.FloatTensor(scaled).unsqueeze(0).to(self.lstm_device)

            with torch.no_grad():
                delta_preds = self.lstm_model(X).cpu().numpy()[0]  # [dy_high, dy_low, dy_close]

            # Convert deltas to absolute values
            pred_high_raw = current_rate + delta_preds[0]
            pred_low_raw = current_rate + delta_preds[1]
            pred_close = current_rate + delta_preds[2]

            # Enforce low <= high
            pred_low_raw, pred_high_raw = min(pred_low_raw, pred_high_raw), max(pred_low_raw, pred_high_raw)

            # Apply calibrated range buffer (with asymmetric upside for v2-boundary)
            asym = getattr(self, "lstm_asymmetric_upside", 0.0)
            pred_high = float(pred_high_raw + self.lstm_range_buffer + asym)
            pred_low = float(pred_low_raw - self.lstm_range_buffer)
            pred_close = float(pred_close)
            range_width = pred_high - pred_low

            # Run risk detector
            risk_features = {}
            if features_df is not None and len(features_df) > 0:
                row = features_df.iloc[-1]
                recent_5 = source_df.tail(5)
                risk_features = {
                    "rate_vs_alltime_percentile": float(row.get("rate_vs_alltime_percentile", 0)),
                    "momentum_consistency": int(row.get("momentum_consistency", 0)),
                    "volatility_20d": float(row.get("volatility_20d", 0)),
                    "volatility_5d": float(row.get("volatility_5d", 0)) if "volatility_5d" in row.index else 0,
                    "current_rate": current_rate,
                    "high_5d": float(recent_5["usdinr"].max()) if len(recent_5) >= 5 else 0,
                    "low_5d": float(recent_5["usdinr"].min()) if len(recent_5) >= 5 else 0,
                    "ma_30": float(source_df["usdinr"].tail(30).mean()),
                    "current_regime": getattr(self, "_last_xgb_regime", "range_bound"),
                    "prev_regime": getattr(self, "_prev_xgb_regime", "range_bound"),
                    "day_of_week": pd.Timestamp.now().weekday(),
                }
            range_risk = detect_high_risk_prediction(risk_features)

            model_version = self.model_versions.get("lstm", "v1")

            return {
                "direction": "NEUTRAL",  # Always — LSTM cannot predict direction
                "role": "range_provider",
                "model_version": model_version,
                "predicted_high": round(pred_high, 4),
                "predicted_low": round(pred_low, 4),
                "predicted_close": round(pred_close, 4),
                "range_width": round(range_width, 4),
                "range_width_label": f"{range_width * 100:.0f} paise",
                "range_risk": range_risk,
                "note": ("BoundaryAwareLoss v2 — optimized for high/low boundary accuracy"
                         if "boundary" in model_version else
                         "LSTM provides price range. Direction from sentiment."),
            }
        except Exception as e:
            print(f"  [lstm] ERROR: {e}")
            fallback["error"] = str(e)
            return fallback

    # -----------------------------------------------------------------------
    # Model 3: Sentiment
    # -----------------------------------------------------------------------

    def _run_sentiment(self) -> dict:
        """Run sentiment agent (NewsAPI + Bedrock)."""
        try:
            raw = get_sentiment()
            direction, confidence = sentiment_to_direction(raw)

            # If no data, zero out confidence so it has no weight
            if raw.get("data_quality") == "none":
                confidence = 0.0

            return {
                "direction": direction,
                "confidence": confidence,
                "score": raw.get("sentiment_score", 0.0),
                "explanation": raw.get("explanation", ""),
                "data_quality": raw.get("data_quality", "none"),
                "high_impact_event_detected": raw.get("high_impact_event_detected", False),
                "event_type": raw.get("event_type"),
                "event_description": raw.get("event_description"),
                "bullish_inr_signals": raw.get("bullish_inr_signals", []),
                "bearish_inr_signals": raw.get("bearish_inr_signals", []),
            }
        except Exception as e:
            print(f"  [sentiment] ERROR: {e}")
            return {
                "direction": "NEUTRAL", "confidence": 0.0,
                "score": 0.0, "explanation": f"Sentiment failed: {e}",
                "data_quality": "none",
                "high_impact_event_detected": False,
                "event_type": None, "event_description": None,
                "bullish_inr_signals": [], "bearish_inr_signals": [],
            }

    # -----------------------------------------------------------------------
    # Model 4: Macro Signal (rule-based FRED)
    # -----------------------------------------------------------------------

    def _run_macro(self, features_df: pd.DataFrame) -> dict:
        """Run rule-based macro signal from FRED features."""
        try:
            return get_macro_signal(features_df)
        except Exception as e:
            print(f"  [macro] ERROR: {e}")
            return {
                "role": "macro_direction",
                "score": 0.0, "direction": "NEUTRAL", "confidence": 0.0,
                "fed_funds_change_3m": None, "yield_curve_spread": None,
                "cpi_yoy": None, "interpretation": f"Macro signal error: {e}",
                "available": False,
            }

    # -----------------------------------------------------------------------
    # Model 5: Intraday LSTM (asymmetric — UP-only momentum signal)
    # -----------------------------------------------------------------------

    def _run_intraday_lstm(self, df_4h: pd.DataFrame = None) -> dict:
        """
        Run intraday LSTM momentum classifier on 4-hour bars.

        Asymmetric integration:
          UP with confidence > 0.60:   signal = "UP_CONFIRMED", adjustment = +0.06
          DOWN or NEUTRAL:             signal = "UNCONFIRMED",  adjustment = 0.00

        The model has 74% accuracy on UP predictions but only 26% on DOWN,
        so we only use the signal where it's reliable.

        Args:
            df_4h: Raw 4-hour OHLCV DataFrame (from fetch_4h_data).
                   If None, reads from data/intraday_4h.csv.
        """
        fallback = {
            "role": "intraday_momentum",
            "signal": "UNCONFIRMED",
            "raw_direction": "NEUTRAL",
            "raw_confidence": 0.0,
            "confidence_adjustment": 0.0,
            "accuracy_note": "Intraday model unavailable",
            "bars_analyzed": 0,
            "model_limitation": "Asymmetric — only UP signals integrated",
            "probabilities": {"DOWN": 0.0, "NEUTRAL": 0.0, "UP": 0.0},
        }

        if self.intraday_model is None:
            fallback["accuracy_note"] = "Intraday LSTM not loaded"
            return fallback

        try:
            # Build features from raw 4h data
            import io
            from contextlib import redirect_stdout

            # Suppress feature-building print output
            f = io.StringIO()
            with redirect_stdout(f):
                df_features, feature_cols = build_intraday_features(df_4h)

            if len(df_features) < INTRADAY_SEQ_LEN:
                fallback["accuracy_note"] = f"Need {INTRADAY_SEQ_LEN} bars, got {len(df_features)}"
                return fallback

            # Scale using training scaler
            features = df_features[INTRADAY_FEATURE_COLS].values
            features_scaled = self.intraday_scaler.transform(features)

            # Take last SEQ_LEN bars as input sequence
            seq = features_scaled[-INTRADAY_SEQ_LEN:]
            X = torch.FloatTensor(seq).unsqueeze(0).to(self.lstm_device)

            # Run inference
            with torch.no_grad():
                logits = self.intraday_model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_class = int(probs.argmax())
            class_names = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
            raw_direction = class_names[pred_class]
            raw_confidence = float(probs.max())

            # Asymmetric signal logic:
            # Only boost confidence when model says UP with high confidence
            if raw_direction == "UP" and raw_confidence > 0.60:
                signal = "UP_CONFIRMED"
                confidence_adjustment = +0.06
                note = "Intraday momentum confirms upward pressure (74% historical accuracy)"
            elif raw_direction == "DOWN":
                signal = "UNCONFIRMED"
                confidence_adjustment = 0.0
                note = "Intraday DOWN signal unreliable — no adjustment applied"
            else:
                signal = "UNCONFIRMED"
                confidence_adjustment = 0.0
                note = "Intraday neutral — no adjustment"

            return {
                "role": "intraday_momentum",
                "signal": signal,
                "raw_direction": raw_direction,
                "raw_confidence": round(raw_confidence, 4),
                "confidence_adjustment": confidence_adjustment,
                "accuracy_note": note,
                "bars_analyzed": INTRADAY_SEQ_LEN,
                "model_limitation": "Asymmetric — only UP signals integrated",
                "probabilities": {
                    "DOWN": round(float(probs[0]), 4),
                    "NEUTRAL": round(float(probs[1]), 4),
                    "UP": round(float(probs[2]), 4),
                },
            }

        except Exception as e:
            print(f"  [intraday-lstm] ERROR: {e}")
            fallback["accuracy_note"] = f"Intraday model error: {e}"
            return fallback

    # -----------------------------------------------------------------------
    # Ensemble
    # -----------------------------------------------------------------------

    def _ensemble(self, xgb_result: dict, lstm_result: dict,
                  sentiment_result: dict, macro_result: dict,
                  current_rate: float, intraday_result: dict = None) -> dict:
        """
        Multi-signal direction model with proportional confidence scaling.

        Signals:
            Vote 1: Sentiment (primary — news-driven, proportional scaling)
            Vote 2: Macro signal (secondary — FRED fundamentals, proportional scaling)
            Context: XGBoost regime (confidence modifier + directional signal)
            Momentum: Intraday LSTM (asymmetric — UP confirmed or partial DOWN)
            Range: LSTM (range provider)

        Agreement bonus: when 2+ non-NEUTRAL signals agree, +0.06 per agreeing signal
        Confidence hard limits: [0.30, 0.82]
        """
        regime = xgb_result.get("regime", "range_bound")
        macro_available = macro_result.get("available", False)
        macro_score = macro_result.get("score", 0.0) if macro_available else 0.0

        # --- Sentiment direction + proportional contribution ---
        sent_score = sentiment_result.get("score", 0.0)
        sent_conf = sentiment_result.get("confidence", 0.0)
        sent_quality = sentiment_result.get("data_quality", "none")

        if sent_quality == "none":
            sent_dir = "NEUTRAL"
            sent_conf = 0.0
            sentiment_contribution = 0.0
        elif sent_score < -0.20:
            sent_dir = "UP"      # bearish INR news → rate goes up
            sentiment_contribution = min(0.18, abs(sent_score) * 0.25)
        elif sent_score > 0.20:
            sent_dir = "DOWN"    # bullish INR news → rate goes down
            sentiment_contribution = min(0.18, abs(sent_score) * 0.25)
        else:
            sent_dir = "NEUTRAL"
            sentiment_contribution = abs(sent_score) * 0.10  # weak signal still contributes

        # --- Macro direction + proportional contribution ---
        macro_dir = macro_result.get("direction", "NEUTRAL") if macro_available else "NEUTRAL"
        macro_conf = macro_result.get("confidence", 0.0) if macro_available else 0.0

        if macro_available and abs(macro_score) > 0.25:
            macro_contribution = min(0.12, abs(macro_score) * 0.20)
        elif macro_available:
            macro_contribution = abs(macro_score) * 0.08
        else:
            macro_contribution = 0.0

        # --- Regime as directional signal ---
        if regime == "trending_up":
            regime_dir = "UP"
        elif regime == "trending_down":
            regime_dir = "DOWN"
        else:
            regime_dir = "NEUTRAL"

        # --- Direction voting ---
        sent_has_dir = sent_dir != "NEUTRAL"
        macro_has_dir = macro_dir != "NEUTRAL"

        if sent_has_dir and macro_has_dir:
            if sent_dir == macro_dir:
                # Both agree
                final_direction = sent_dir
                ensemble_confidence = sent_conf * 0.55 + macro_conf * 0.35
                ensemble_confidence += sentiment_contribution + macro_contribution
                vote_outcome = "agree"
            else:
                # Disagree → NEUTRAL
                final_direction = "NEUTRAL"
                ensemble_confidence = 0.35
                vote_outcome = "disagree"
        elif sent_has_dir:
            # Only sentiment has direction
            final_direction = sent_dir
            ensemble_confidence = sent_conf * 0.70 + sentiment_contribution
            vote_outcome = "sentiment_only"
        elif macro_has_dir:
            # Only macro has direction
            final_direction = macro_dir
            ensemble_confidence = macro_conf * 0.50 + macro_contribution
            vote_outcome = "macro_only"
        else:
            # Neither has direction
            final_direction = "NEUTRAL"
            ensemble_confidence = 0.35
            vote_outcome = "neither"

        # --- Regime adjustment ---
        regime_adj = self._compute_regime_adjustment(regime, final_direction)
        if final_direction != "NEUTRAL":
            ensemble_confidence += regime_adj

        # --- Multi-signal agreement bonus ---
        signals = []
        if sent_dir != "NEUTRAL":
            signals.append(sent_dir)
        if macro_dir != "NEUTRAL":
            signals.append(macro_dir)
        if regime_dir != "NEUTRAL":
            signals.append(regime_dir)

        up_count = signals.count("UP")
        down_count = signals.count("DOWN")
        agreement_bonus = 0.0

        if up_count >= 2:
            agreement_bonus = 0.06 * up_count
            if final_direction == "NEUTRAL" and up_count > down_count:
                final_direction = "UP"
        elif down_count >= 2:
            agreement_bonus = 0.06 * down_count
            if final_direction == "NEUTRAL" and down_count > up_count:
                final_direction = "DOWN"

        if final_direction != "NEUTRAL":
            ensemble_confidence += agreement_bonus

        signals_in_agreement = max(up_count, down_count)

        # --- Intraday confirmation ---
        intraday_adj = 0.0
        if intraday_result:
            intra_signal = intraday_result.get("signal", "UNCONFIRMED")
            intra_raw_dir = intraday_result.get("raw_direction", "NEUTRAL")

            if intra_signal == "UP_CONFIRMED" and final_direction == "UP":
                intraday_adj = +0.06
            elif intra_raw_dir == "DOWN" and final_direction == "DOWN":
                # Partial credit — DOWN accuracy is only 26%
                intraday_adj = +0.04

            ensemble_confidence += intraday_adj

        # --- Regime fallback (last resort when no signals at all) ---
        if sent_quality == "none" and not macro_available and agreement_bonus == 0:
            if regime == "trending_up":
                final_direction = "UP"
                ensemble_confidence = 0.45 + regime_adj
            elif regime == "trending_down":
                final_direction = "DOWN"
                ensemble_confidence = 0.45 + regime_adj
            else:
                ensemble_confidence = 0.35
            ensemble_confidence = min(0.58, ensemble_confidence)
            sentiment_result["data_quality"] = "regime_fallback"
            vote_outcome = "regime_fallback"

        # --- Hard limits ---
        ensemble_confidence = round(max(0.30, min(0.82, ensemble_confidence)), 3)

        # Signal strength
        if ensemble_confidence > 0.72:
            signal_strength = "STRONG"
        elif ensemble_confidence > 0.55:
            signal_strength = "MEDIUM"
        else:
            signal_strength = "WEAK"

        # Range from LSTM (or fallback)
        pred_high = lstm_result.get("predicted_high")
        pred_low = lstm_result.get("predicted_low")
        pred_close = lstm_result.get("predicted_close")

        if pred_high is None or pred_low is None:
            pred_high = round(current_rate + 0.40, 4)
            pred_low = round(current_rate - 0.40, 4)
            pred_close = round(current_rate, 4)

        return {
            "direction": final_direction,
            "confidence": ensemble_confidence,
            "signal_strength": signal_strength,
            "range_high": pred_high,
            "range_low": pred_low,
            "most_likely": pred_close,
            "regime": regime,
            "regime_adjustment": regime_adj,
            "vote_outcome": vote_outcome,
            "sent_direction": sent_dir,
            "macro_direction": macro_dir,
            "agreement_bonus_applied": round(agreement_bonus, 3),
            "signals_in_agreement": signals_in_agreement,
            "sentiment_contribution": round(sentiment_contribution, 4),
            "macro_contribution": round(macro_contribution, 4),
            "intraday_adjustment": round(intraday_adj, 3),
        }

    @staticmethod
    def _regime_note(regime: str) -> str:
        """Human-readable note for the detected regime."""
        notes = {
            "trending_up": "Uptrend detected — directional confidence boosted for UP signals",
            "trending_down": "Downtrend detected — directional confidence boosted for DOWN signals",
            "high_vol": "High volatility regime — reducing all directional confidence",
            "range_bound": "Range-bound market — no regime-based confidence adjustment",
        }
        return notes.get(regime, "Unknown regime")

    def _compute_regime_adjustment(self, regime: str, direction: str) -> float:
        """
        Compute confidence adjustment based on regime + direction alignment.

        trending_up + UP:    +0.08  (regime supports direction)
        trending_up + DOWN:  -0.08  (regime opposes direction)
        trending_down + DOWN: +0.08
        trending_down + UP:  -0.08
        range_bound:          0.00  (no adjustment)
        high_vol:            -0.10  (reduce all confidence)
        """
        if regime == "high_vol":
            return -0.10
        elif regime == "range_bound":
            return 0.00
        elif regime == "trending_up":
            return +0.08 if direction == "UP" else -0.08
        elif regime == "trending_down":
            return +0.08 if direction == "DOWN" else -0.08
        return 0.00

    # -----------------------------------------------------------------------
    # Safety Gate
    # -----------------------------------------------------------------------

    def _safety_gate(self, ensemble_result: dict, sentiment_result: dict,
                     df: pd.DataFrame, features_df: pd.DataFrame = None) -> dict:
        """
        Apply safety rules in order. Modifies ensemble_result in-place.
        Returns dict with act_on_signal and risk_flags.
        """
        act_on_signal = True
        risk_flags = []
        confidence = ensemble_result["confidence"]
        direction = ensemble_result["direction"]

        # Get latest features for volatility/calendar checks
        try:
            if features_df is None:
                features_df = build_features(df)
            latest = features_df.iloc[-1]
        except Exception:
            latest = pd.Series(dtype=float)

        # --- Rule 0: Regime fallback warning ---
        if sentiment_result.get("data_quality") == "regime_fallback":
            risk_flags.append("Sentiment unavailable — regime-only signal (lower reliability)")
            risk_flags.append("Verify manually before acting on this signal")

        # --- Rule 1: Confidence gate ---
        if confidence < 0.55:
            act_on_signal = False
            direction = "NEUTRAL"
            risk_flags.append(f"Confidence below threshold ({confidence:.2f})")

        # --- Rule 2: Direction gate ---
        if direction == "NEUTRAL":
            act_on_signal = False
            if not any("Confidence below" in f for f in risk_flags):
                risk_flags.append("No directional signal from sentiment")

        # --- Rule 3: High impact event blackout ---
        if sentiment_result.get("high_impact_event_detected", False):
            act_on_signal = False
            event_type = sentiment_result.get("event_type", "UNKNOWN")
            risk_flags.append(f"High impact event: {event_type}")

        # --- Rule 4: Circuit breaker (>0.8% daily move) ---
        rate_change_1d = latest.get("rate_change_1d", 0.0) if len(latest) > 0 else 0.0
        rate_change_pct = abs(float(rate_change_1d)) * 100
        if rate_change_pct > 0.8:
            act_on_signal = False
            risk_flags.append(f"Circuit breaker: rate moved {rate_change_pct:.2f}% in 24h")

        # --- Rule 5: Volatility gate (5d vol > 0.5%) ---
        vol_5d = float(latest.get("volatility_5d", 0.0)) if len(latest) > 0 else 0.0
        if vol_5d > 0.005:
            confidence -= 0.10
            confidence = round(confidence, 3)
            risk_flags.append(f"Elevated volatility: {vol_5d * 100:.2f}%")
            # Check if confidence reduction triggers Rule 1
            if confidence < 0.55 and act_on_signal:
                act_on_signal = False
                direction = "NEUTRAL"
                risk_flags.append(f"Confidence dropped below threshold after volatility adjustment ({confidence:.2f})")

        # --- Rule 6: RBI/Fed week caution ---
        is_rbi_week = int(latest.get("is_rbi_week", 0)) if len(latest) > 0 else 0
        is_fed_week = int(latest.get("is_fed_week", 0)) if len(latest) > 0 else 0

        if is_rbi_week:
            confidence -= 0.05
            confidence = round(confidence, 3)
            risk_flags.append("RBI week — increased uncertainty")
        if is_fed_week:
            confidence -= 0.05
            confidence = round(confidence, 3)
            risk_flags.append("Fed week — increased uncertainty")

        # Check if reductions from 4/5 now trigger Rule 1
        if confidence < 0.55 and act_on_signal:
            act_on_signal = False
            direction = "NEUTRAL"
            risk_flags.append(f"Confidence dropped below threshold after adjustments ({confidence:.2f})")

        # Update signal strength after adjustments
        if confidence > 0.72:
            signal_strength = "STRONG"
        elif confidence > 0.55:
            signal_strength = "MEDIUM"
        else:
            signal_strength = "WEAK"

        return {
            "direction": direction,
            "confidence": confidence,
            "signal_strength": signal_strength,
            "act_on_signal": act_on_signal,
            "risk_flags": risk_flags,
        }

    # -----------------------------------------------------------------------
    # Opus 4.6 Reasoning Layer
    # -----------------------------------------------------------------------

    REASONING_CONFIDENCE_THRESHOLD = 0.50

    def _generate_reasoning(self, prediction_data: dict,
                            features_row: dict) -> dict:
        """
        Call Claude Opus via OpenRouter to generate human-readable
        explanation of the prediction. Only called when
        confidence > threshold to save cost and ensure meaningful context.
        """
        # Extract values safely
        breakdown = prediction_data.get("model_breakdown", {})
        pred_48h = prediction_data.get("prediction_48h", {})
        summary = prediction_data.get("summary", {})

        regime = breakdown.get("xgboost", {}).get("regime", "unknown")
        regime_conf = breakdown.get("xgboost", {}).get("regime_confidence", 0.0)
        sentiment_dir = breakdown.get("sentiment", {}).get("direction", "NEUTRAL")
        sentiment_score = breakdown.get("sentiment", {}).get("score", 0.0)
        macro_dir = breakdown.get("macro", {}).get("direction", "NEUTRAL")
        macro_score = breakdown.get("macro", {}).get("score", 0.0)
        intraday_signal = breakdown.get("intraday_lstm", {}).get("signal", "UNCONFIRMED")
        direction = pred_48h.get("direction", "NEUTRAL")
        confidence = pred_48h.get("confidence", 0.0)
        action = summary.get("action", "HOLD")
        range_low = pred_48h.get("range_low", 0.0)
        range_high = pred_48h.get("range_high", 0.0)
        current_rate = prediction_data.get("current_rate", 0.0)

        # Get feature values safely
        yield_curve = float(features_row.get("yield_curve_spread", 0))
        fed_change = float(features_row.get("fed_funds_change_3m", 0))
        cpi = float(features_row.get("cpi_yoy", 0))
        percentile = float(features_row.get("rate_vs_alltime_percentile", 0))
        trend_30d = float(features_row.get("rate_trend_30d", 0))
        volatility = float(features_row.get("volatility_20d", 0))
        rate_vs_5y = float(features_row.get("rate_vs_5y_avg", 0))
        momentum = float(features_row.get("momentum_consistency", 0))

        prompt = f"""You are the reasoning engine behind FX Band Predictor \
— an AI-powered FX timing system built for institutional remittance \
operations processing $9.5M daily (₹86.3 crore).

You think like a seasoned Indian money market dealer — sharp, direct, \
no jargon. You know that 1 paise on $9.5M is ₹95,000. You respect \
capital preservation above all else.

Five specialized models have analyzed the market. Your job is to \
explain their collective judgment clearly to the CFO in 5 sentences.

═══ MARKET CONTEXT ═══
Current USD/INR rate:        {current_rate:.4f}
Daily conversion volume:     $9.5M USD
INR equivalent per day:      ~₹{current_rate * 9_500_000 / 1_00_00_000:.1f} crore
1 paise improvement =        ₹95,000 saved per day
Rate vs 22-year history:     {percentile:.1%} percentile (0%=all-time low, 100%=all-time high)
Rate vs 5-year average:      {rate_vs_5y:+.2%}
30-day trend:                {trend_30d:+.4f} INR per day
20-day volatility:           {volatility:.2%} daily

═══ MODEL SIGNALS ═══
Market Regime (XGBoost):     {regime} ({regime_conf:.1%} confidence)
Expected rate range (LSTM):  {range_low:.4f} — {range_high:.4f} INR
News Sentiment:              {sentiment_dir} (score: {sentiment_score:+.2f})
Macro Signal:                {macro_dir} (score: {macro_score:+.2f})
  └ Yield curve spread:      {yield_curve:+.3f} (negative = inverted = dollar bullish)
  └ Fed funds 3m change:     {fed_change:+.2f}%
  └ US CPI YoY:              {cpi:.1f}%
Intraday Momentum (4h LSTM): {intraday_signal}
24h momentum consistency:    {momentum:+.0f} (range -6 to +6)

═══ PREDICTION ═══
Direction:    {direction}
Confidence:   {confidence:.1%}
Action:       {action}

Write exactly 5 sentences. No bullet points. No headers.
Be specific with numbers from the data above.
Write for a CFO, not a quant. Plain English only.

Sentence 1: What is the USD/INR rate doing right now \
and what historical context matters most?

Sentence 2: Which 2-3 signals are most important today \
and do they agree or conflict?

Sentence 3: What is the single biggest risk to this \
prediction being wrong?

Sentence 4: What specific action should the treasury \
team take today and why?

Sentence 5: What should they watch for in the next \
24-48 hours that would change this view?

Be direct. If confidence is low, say so plainly.
If signals conflict, acknowledge the uncertainty.
Do not oversell. A wrong confident prediction costs \
real money."""

        if not self.openrouter_api_key:
            return {
                "model": None,
                "analysis": None,
                "generated": False,
                "skip_reason": "OpenRouter API key not configured",
            }

        # Try each model in the fallback chain via OpenRouter
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for model_id in self.reasoning_model_chain:
            try:
                payload = {
                    "model": model_id,
                    "max_tokens": 450,
                    "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}],
                }
                resp = _requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                body = resp.json()

                analysis = body["choices"][0]["message"]["content"].strip()
                usage = body.get("usage", {})

                # Extract model short name
                model_name = model_id.split("/")[-1]

                return {
                    "model": model_name,
                    "analysis": analysis,
                    "generated": True,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }
            except Exception as e:
                last_error = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 404:
                    # Model not available — try next
                    continue
                elif status == 401 or status == 403:
                    # Auth issue — stop trying
                    break
                elif status == 429:
                    # Rate limited — try next model
                    continue
                else:
                    continue

        return {
            "model": None,
            "analysis": None,
            "generated": False,
            "skip_reason": f"All models failed. Last error: {str(last_error)}"
                          if last_error else "No models in chain",
        }

    # -----------------------------------------------------------------------
    # Common OpenRouter helper
    # -----------------------------------------------------------------------

    def _call_openrouter(self, prompt: str, max_tokens: int = 450,
                         temperature: float = 0.3) -> dict | None:
        """Make a single OpenRouter API call with model fallback chain."""
        if not self.openrouter_api_key:
            return None
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        for model_id in self.reasoning_model_chain:
            try:
                resp = _requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model_id,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                body = resp.json()
                content = body["choices"][0]["message"]["content"].strip()
                usage = body.get("usage", {})
                return {
                    "model": model_id.split("/")[-1],
                    "content": content,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 403):
                    break
                continue
        return None

    # -----------------------------------------------------------------------
    # Enrichment helpers (pure computation)
    # -----------------------------------------------------------------------

    DAILY_VOLUME_USD = 9_500_000

    @staticmethod
    def _format_inr(amount: float) -> str:
        """Format INR amount in Indian notation (lakh/crore)."""
        abs_amt = abs(amount)
        if abs_amt >= 1_00_00_000:
            return f"\u20b9{amount / 1_00_00_000:.1f} crore"
        elif abs_amt >= 1_00_000:
            return f"\u20b9{amount / 1_00_000:.1f} lakh"
        else:
            return f"\u20b9{amount:,.0f}"

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 0.72:
            return "HIGH \u2014 strong conviction, act on signal"
        elif confidence >= 0.55:
            return "MEDIUM \u2014 actionable with normal position sizing"
        elif confidence >= 0.40:
            return "LOW-MEDIUM \u2014 monitor closely, reduce position size"
        else:
            return "LOW \u2014 monitor, do not act"

    def _build_prefunding_guidance(self, range_low: float, range_high: float,
                                    most_likely: float, current_rate: float,
                                    direction: str, confidence: float,
                                    act_on_signal: bool) -> dict:
        vol = self.DAILY_VOLUME_USD
        if act_on_signal and direction == "DOWN":
            rec_rate = most_likely
            basis = "most_likely \u2014 strong DOWN signal, rate expected to fall"
            action = f"Prefund at {most_likely:.2f}. Rate likely to fall \u2014 convert early."
        elif act_on_signal and direction == "UP":
            rec_rate = range_high
            basis = "range_high \u2014 rate expected to rise, lock in conservative level"
            action = f"Prefund at {range_high:.2f}. Rate likely to rise \u2014 hold and convert later."
        else:
            rec_rate = range_high
            basis = "range_high \u2014 conservative prefunding on low confidence"
            action = f"Prefund at {rec_rate:.2f}. Convert operational minimum daily."

        tpv_high = range_high * vol
        tpv_low = range_low * vol
        tpv_diff = tpv_high - tpv_low

        return {
            "recommended_rate": round(rec_rate, 4),
            "basis": basis,
            "tpv_at_range_high": round(tpv_high),
            "tpv_at_range_low": round(tpv_low),
            "tpv_difference_inr": round(tpv_diff),
            "tpv_difference_label": f"{self._format_inr(tpv_diff)} rate risk this week",
            "action": action,
        }

    def _build_consensus_view(self, ensemble_result: dict, xgb_result: dict,
                               sentiment_result: dict, macro_result: dict,
                               intraday_result: dict) -> dict:
        signals = {}
        sent_dir = ensemble_result.get("sent_direction", "NEUTRAL")
        signals["Sentiment"] = sent_dir
        macro_dir = ensemble_result.get("macro_direction", "NEUTRAL")
        signals["Macro"] = macro_dir

        regime = xgb_result.get("regime", "range_bound")
        if regime == "trending_up":
            signals["XGBoost regime"] = "UP"
        elif regime == "trending_down":
            signals["XGBoost regime"] = "DOWN"
        else:
            signals["XGBoost regime"] = "NEUTRAL"

        intra_raw = intraday_result.get("raw_direction", "NEUTRAL") if intraday_result else "NEUTRAL"
        signals["Intraday LSTM"] = intra_raw
        signals["LSTM range"] = "NEUTRAL"

        up_count = sum(1 for d in signals.values() if d == "UP")
        down_count = sum(1 for d in signals.values() if d == "DOWN")
        neutral_count = sum(1 for d in signals.values() if d == "NEUTRAL")
        directional_count = up_count + down_count

        if up_count >= 3:
            agreement, label = "strong_bull_consensus", "Strong bull consensus \u2014 multiple models confirm UP"
            verdict, emoji = "CONVERT NOW \u2014 strong multi-model agreement on rising rate", "\U0001f7e2"
        elif down_count >= 3:
            agreement, label = "strong_bear_consensus", "Strong bear consensus \u2014 multiple models confirm DOWN"
            verdict, emoji = "HOLD DOLLARS \u2014 strong multi-model agreement on falling rate", "\U0001f534"
        elif up_count >= 2 or down_count >= 2:
            dominant = "UP" if up_count > down_count else "DOWN"
            agreement = "partial_consensus"
            label = f"Partial consensus \u2014 {max(up_count, down_count)} models lean {dominant}"
            verdict, emoji = "MONITOR CLOSELY \u2014 emerging signal, not yet actionable", "\U0001f7e1"
        else:
            agreement, label = "no_consensus", "Models disagree \u2014 low conviction signal"
            verdict, emoji = "HOLD \u2014 insufficient signal strength to act", "\u23f8\ufe0f"

        # Strongest signal
        regime_conf = xgb_result.get("regime_confidence", 0.0)
        sent_conf = sentiment_result.get("confidence", 0.0)
        macro_conf = macro_result.get("confidence", 0.0) if macro_result.get("available") else 0.0
        intra_conf = intraday_result.get("raw_confidence", 0.0) if intraday_result else 0.0

        strengths = {
            f"XGBoost regime ({regime}, {regime_conf:.1%})": regime_conf,
            f"Sentiment ({sent_dir}, {sent_conf:.1%})": sent_conf if sent_dir != "NEUTRAL" else 0.0,
            f"Macro ({macro_dir}, {macro_conf:.1%})": macro_conf if macro_dir != "NEUTRAL" else 0.0,
            f"Intraday LSTM ({intra_raw}, {intra_conf:.1%})": intra_conf if intra_raw != "NEUTRAL" else 0.0,
        }
        strongest = max(strengths, key=strengths.get)

        if up_count > 0 and down_count > 0:
            up_models = [k for k, v in signals.items() if v == "UP"]
            down_models = [k for k, v in signals.items() if v == "DOWN"]
            conflict = f"{', '.join(up_models)} say UP vs {', '.join(down_models)} say DOWN"
        elif directional_count == 0:
            conflict = "No directional signals \u2014 all models neutral"
        else:
            conflict = "No conflict \u2014 signals align or are neutral"

        return {
            "models_checked": 5,
            "models_with_direction": directional_count,
            "models_neutral": neutral_count,
            "agreement": agreement,
            "agreement_label": label,
            "strongest_signal": strongest,
            "biggest_conflict": conflict,
            "overall_verdict": verdict,
            "verdict_emoji": emoji,
        }

    # -----------------------------------------------------------------------
    # Opus enrichments (parallel calls)
    # -----------------------------------------------------------------------

    def _generate_morning_briefing(self, prediction_data: dict,
                                    features_row: dict) -> dict:
        """Opus call #1: 8-sentence morning briefing."""
        try:
            breakdown = prediction_data.get("model_breakdown", {})
            pred = prediction_data.get("prediction_48h", {})
            current_rate = prediction_data.get("current_rate", 0.0)
            xgb = breakdown.get("xgboost", {})
            lstm = breakdown.get("lstm", {})
            sent = breakdown.get("sentiment", {})
            macro = breakdown.get("macro", {})
            intraday = breakdown.get("intraday_lstm", {})

            percentile = float(features_row.get("rate_vs_alltime_percentile", 0))
            trend_30d = float(features_row.get("rate_trend_30d", 0))
            rate_vs_5y = float(features_row.get("rate_vs_5y_avg", 0))
            volatility = float(features_row.get("volatility_20d", 0))
            yield_curve = float(features_row.get("yield_curve_spread", 0))
            cpi = float(features_row.get("cpi_yoy", 0))

            prompt = (
                "You are FX Band Predictor \u2014 a sharp, experienced FX desk analyst "
                "who thinks like a seasoned Mumbai money market dealer.\n\n"
                "Write a morning briefing for the treasury team. They process $9.5M daily "
                "(\u20b986.3 crore) in USD to INR remittances and need to prefund Indian bank "
                "accounts by 9 AM IST.\n\n"
                f"MARKET DATA:\n"
                f"Current USD/INR rate: {current_rate:.4f}\n"
                f"Rate vs 22-year history: {percentile:.1%} percentile\n"
                f"Rate vs 5-year average: {rate_vs_5y:+.2%}\n"
                f"30-day trend: {trend_30d:+.4f} INR per day\n"
                f"20-day volatility: {volatility:.2%} daily\n"
                f"Market regime: {xgb.get('regime', 'unknown')} ({xgb.get('regime_confidence', 0):.1%} confidence)\n"
                f"LSTM range: {lstm.get('range_low', 'N/A')} \u2014 {lstm.get('range_high', 'N/A')}\n"
                f"Sentiment: {sent.get('direction', 'N/A')} (score: {sent.get('score', 0):.2f})\n"
                f"Macro: {macro.get('direction', 'N/A')} (score: {macro.get('score', 0):.2f}), "
                f"yield curve: {yield_curve:+.3f}, CPI: {cpi:.1f}%\n"
                f"Intraday: {intraday.get('signal', 'N/A')} "
                f"(raw: {intraday.get('raw_direction', 'N/A')}, "
                f"conf: {intraday.get('raw_confidence', 0):.1%})\n"
                f"Prediction: {pred.get('direction', 'NEUTRAL')}, "
                f"confidence: {pred.get('confidence', 0):.1%}, "
                f"act: {prediction_data.get('act_on_signal', False)}\n")

            # Add range risk context if available
            range_risk = prediction_data.get("range_risk", {})
            risk_level = range_risk.get("risk_level", "NORMAL")
            risk_factors = range_risk.get("risk_factors", [])
            if risk_level in ("HIGH", "CRITICAL", "ELEVATED"):
                prompt += (
                    f"Range prediction risk: {risk_level} "
                    f"(score: {range_risk.get('risk_score', 0)})\n"
                    f"Risk factors: {'; '.join(risk_factors)}\n"
                )

            prompt += (
                "\nWrite in this exact structure:\n"
                "SITUATION (2 sentences): What is the rate doing and why. "
                "Reference the 22-year percentile and recent trend.\n"
                "SIGNALS (2 sentences): What the 5 models are saying collectively. "
                "Name agreements and disagreements.\n"
                "RISK (1 sentence): The single biggest risk to being wrong today.\n"
                "TREASURY ACTION (2 sentences): Exactly what to do. Include the rate "
                "to prefund at, whether to convert now or wait, and why. "
                "If range risk is HIGH or CRITICAL, prominently warn and "
                "recommend adding the specific buffer amount.\n"
                "WATCH FOR (1 sentence): One specific thing that would change this view "
                "in the next 24 hours.\n\n"
                "Total: 8 sentences maximum. Be direct. Use numbers. No jargon. "
                "Write like you're talking to a CFO at 8 AM."
            )

            result = self._call_openrouter(prompt, max_tokens=500, temperature=0.3)
            if result:
                conf = pred.get("confidence", 0)
                if conf >= 0.55:
                    ctx = "Actionable signal \u2014 briefing reflects conviction"
                elif conf >= 0.40:
                    ctx = "Moderate conviction \u2014 briefing highlights uncertainty"
                else:
                    ctx = "Low conviction week \u2014 briefing reflects uncertainty"
                return {
                    "generated_by": result["model"],
                    "briefing": result["content"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "confidence_context": ctx,
                    "tokens_used": result["input_tokens"] + result["output_tokens"],
                }
        except Exception as e:
            print(f"  [opus-briefing] ERROR: {e}")

        return self._fallback_morning_briefing(prediction_data)

    def _fallback_morning_briefing(self, prediction_data: dict) -> dict:
        pred = prediction_data.get("prediction_48h", {})
        rate = prediction_data.get("current_rate", 0.0)
        breakdown = prediction_data.get("model_breakdown", {})
        regime = breakdown.get("xgboost", {}).get("regime", "range_bound")
        direction = pred.get("direction", "NEUTRAL")
        confidence = pred.get("confidence", 0.0)
        range_high = pred.get("range_high", rate)

        n_dir = sum(1 for m in ["sentiment", "macro"] if
                    breakdown.get(m, {}).get("direction", "NEUTRAL") != "NEUTRAL")

        if direction != "NEUTRAL" and confidence >= 0.55:
            action = f"Act on {direction} signal at {rate:.2f}."
        else:
            action = f"Prefund at {range_high:.2f} for conservative TPV."

        briefing = (
            f"Rate at {rate:.2f}, regime {regime}. "
            f"5 models checked, {n_dir} with directional signal. "
            f"Confidence {confidence:.0%}. {action}"
        )
        return {
            "generated_by": "rule_based_fallback",
            "briefing": briefing,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "confidence_context": "Opus unavailable \u2014 rule-based summary",
            "tokens_used": 0,
        }

    def _generate_model_whys(self, prediction_data: dict) -> dict:
        """Opus call #2: Why explanations for all 5 models (batched)."""
        try:
            bd = prediction_data.get("model_breakdown", {})
            xgb = bd.get("xgboost", {})
            lstm = bd.get("lstm", {})
            sent = bd.get("sentiment", {})
            macro = bd.get("macro", {})
            intra = bd.get("intraday_lstm", {})

            prompt = (
                "Explain each of these 5 model signals in one plain English "
                "sentence each. Be specific with numbers. Write for a treasury "
                "manager, not a quant.\n\n"
                f"XGBoost: regime={xgb.get('regime', 'N/A')}, "
                f"confidence={xgb.get('regime_confidence', 0):.1%}\n"
                f"LSTM: range {lstm.get('range_low', 'N/A')} \u2014 "
                f"{lstm.get('range_high', 'N/A')}, "
                f"most_likely={lstm.get('most_likely', 'N/A')}, "
                f"range_width={lstm.get('range_width', 0):.2f}\n"
                f"Sentiment: direction={sent.get('direction', 'N/A')}, "
                f"score={sent.get('score', 0):.2f}, "
                f"data_quality={sent.get('data_quality', 'N/A')}\n"
                f"Macro: direction={macro.get('direction', 'N/A')}, "
                f"score={macro.get('score', 0):.2f}, "
                f"available={macro.get('available', False)}, "
                f"yield_curve={macro.get('yield_curve_spread', 'N/A')}, "
                f"CPI={macro.get('cpi_yoy', 'N/A')}\n"
                f"Intraday: signal={intra.get('signal', 'N/A')}, "
                f"raw={intra.get('raw_direction', 'N/A')}, "
                f"confidence={intra.get('raw_confidence', 0):.1%}, "
                f"probs={intra.get('probabilities', {})}\n\n"
                "Return ONLY a JSON object:\n"
                '{"xgboost_why": "one sentence", "lstm_why": "one sentence", '
                '"sentiment_why": "one sentence", "macro_why": "one sentence", '
                '"intraday_why": "one sentence"}'
            )

            result = self._call_openrouter(prompt, max_tokens=400, temperature=0.2)
            if result:
                content = result["content"]
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                whys = json.loads(content)
                whys["_model"] = result["model"]
                whys["_tokens"] = result["input_tokens"] + result["output_tokens"]
                return whys
        except Exception as e:
            print(f"  [opus-whys] ERROR: {e}")

        return self._fallback_model_whys(prediction_data)

    def _fallback_model_whys(self, prediction_data: dict) -> dict:
        bd = prediction_data.get("model_breakdown", {})
        xgb = bd.get("xgboost", {})
        lstm = bd.get("lstm", {})
        sent = bd.get("sentiment", {})
        macro = bd.get("macro", {})
        intra = bd.get("intraday_lstm", {})
        regime = xgb.get("regime", "range_bound")
        range_paise = round(lstm.get("range_width", 0) * 100)
        sent_score = sent.get("score", 0.0)

        return {
            "xgboost_why": f"Market is in {regime} regime with {xgb.get('regime_confidence', 0):.0%} confidence \u2014 shapes how much we trust other signals.",
            "lstm_why": f"LSTM expects a {range_paise}-paise range \u2014 {'tight consolidation' if range_paise < 50 else 'moderate volatility' if range_paise < 80 else 'wide range, uncertainty'}.",
            "sentiment_why": f"News sentiment at {sent_score:+.2f} ({sent.get('data_quality', 'none')}) \u2014 {'not enough to move the needle' if abs(sent_score) < 0.20 else 'directional signal detected'}.",
            "macro_why": f"Macro points {macro.get('direction', 'NEUTRAL')} (score {macro.get('score', 0):+.2f}) \u2014 {'neutral environment' if macro.get('direction') == 'NEUTRAL' else 'macro forces active'}.",
            "intraday_why": f"4h momentum: {intra.get('raw_direction', 'N/A')} at {intra.get('raw_confidence', 0):.0%} \u2014 {'too close to call' if abs(intra.get('probabilities', {}).get('UP', 0) - intra.get('probabilities', {}).get('DOWN', 0)) < 0.10 else 'directional lean'}.",
            "_model": "rule_based_fallback",
            "_tokens": 0,
        }

    def _generate_context_interpretations(self, prediction_data: dict,
                                            features_row: dict) -> dict:
        """Opus call #3: NDF + macro context interpretation."""
        try:
            ndf = prediction_data.get("ndf_context") or {}
            macro = prediction_data.get("model_breakdown", {}).get("macro", {})
            yield_curve = float(features_row.get("yield_curve_spread", 0))
            cpi = float(features_row.get("cpi_yoy", 0))
            fed_change = float(features_row.get("fed_funds_change_3m", 0))

            prompt = (
                "You are FX Band Predictor. Interpret these market context signals "
                "for a treasury team processing $9.5M daily in USD/INR. "
                "Be direct and specific.\n\n"
                f"NDF CONTEXT:\n"
                f"Forward premium: {ndf.get('forward_premium_paise', 'N/A')} paise\n"
                f"Annualized premium: {ndf.get('forward_premium_annualized_pct', 'N/A')}%\n"
                f"Carry regime: {ndf.get('carry_regime', 'N/A')}\n"
                f"India-US rate differential: {ndf.get('rate_differential', 'N/A')}%\n"
                f"India repo rate: {ndf.get('india_repo_rate', 'N/A')}%\n"
                f"US 10Y yield: {ndf.get('us_10y_yield', 'N/A')}%\n\n"
                f"MACRO CONTEXT:\n"
                f"Yield curve spread: {yield_curve:+.3f}\n"
                f"CPI YoY: {cpi:.1f}%\n"
                f"Fed funds 3m change: {fed_change:+.2f}%\n"
                f"Macro direction: {macro.get('direction', 'NEUTRAL')}\n"
                f"Macro score: {macro.get('score', 0):.2f}\n"
                f"Macro interpretation: {macro.get('interpretation', 'N/A')}\n\n"
                "For each context, write 2-3 sentences explaining what it means "
                "for USD/INR and whether it's actionable. Be specific.\n\n"
                "Return ONLY a JSON object:\n"
                '{"ndf_interpretation": "2-3 sentences", '
                '"macro_interpretation": "2-3 sentences"}'
            )

            result = self._call_openrouter(prompt, max_tokens=350, temperature=0.2)
            if result:
                content = result["content"]
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                interps = json.loads(content)
                interps["_model"] = result["model"]
                interps["_tokens"] = result["input_tokens"] + result["output_tokens"]
                return interps
        except Exception as e:
            print(f"  [opus-context] ERROR: {e}")

        return self._fallback_context_interpretations(features_row)

    def _fallback_context_interpretations(self, features_row: dict) -> dict:
        yield_curve = float(features_row.get("yield_curve_spread", 0))
        cpi = float(features_row.get("cpi_yoy", 0))

        if yield_curve < 0:
            yc_text = f"inverted at {yield_curve:+.3f} \u2014 recession signal, dollar-bullish"
        elif yield_curve > 0.5:
            yc_text = f"at {yield_curve:+.3f} \u2014 normalized, no recession signal"
        else:
            yc_text = f"flat at {yield_curve:+.3f} \u2014 watchful"

        if cpi < 2.0:
            cpi_text = "below Fed target, argues for cuts and dollar weakness"
        elif cpi > 3.5:
            cpi_text = "above comfort zone, argues for tighter policy"
        else:
            cpi_text = "near target, neutral for dollar"

        return {
            "ndf_interpretation": "NDF data not available for interpretation.",
            "macro_interpretation": f"Yield curve {yc_text}. CPI at {cpi:.1f}% \u2014 {cpi_text}.",
            "_model": "rule_based_fallback",
            "_tokens": 0,
        }

    # -----------------------------------------------------------------------
    # Key drivers
    # -----------------------------------------------------------------------

    def _extract_key_drivers(self, df: pd.DataFrame, n: int = 3,
                              features_df: pd.DataFrame = None) -> list[str]:
        """
        Extract top N features by XGBoost importance that are currently
        noteworthy (above or below their historical mean).
        """
        if self.xgb_importance.empty:
            return []

        try:
            if features_df is None:
                features_df = build_features(df)
            latest = features_df.iloc[-1]
            means = features_df.mean()

            drivers = []
            top_features = self.xgb_importance.head(15)["feature"].tolist()

            for feat in top_features:
                if feat not in latest.index:
                    continue

                val = float(latest[feat])
                mean_val = float(means.get(feat, 0))

                # Skip calendar features that are 0 (not active)
                if feat in ("is_rbi_week", "is_fed_week",
                            "is_month_end", "high_vol_regime") and val == 0:
                    continue

                # Skip if value is very close to mean (not noteworthy)
                if feat not in ("is_rbi_week", "is_fed_week",
                                "is_month_end", "high_vol_regime"):
                    if mean_val != 0 and abs(val - mean_val) / abs(mean_val) < 0.15:
                        continue

                desc = _format_driver(feat, val)
                if desc:
                    drivers.append(desc)

                if len(drivers) >= n:
                    break

            return drivers if drivers else ["No significant deviations in top features"]

        except Exception as e:
            return [f"Driver extraction failed: {e}"]

    # -----------------------------------------------------------------------
    # Build output
    # -----------------------------------------------------------------------

    def _build_output(self, current_rate: float, gate_result: dict,
                      ensemble_result: dict, xgb_result: dict,
                      lstm_result: dict, sentiment_result: dict,
                      macro_result: dict, key_drivers: list,
                      data_rows: int, feature_count: int,
                      intraday_result: dict = None) -> dict:
        """Construct the final prediction JSON."""
        now = datetime.now(timezone.utc)

        direction = gate_result["direction"]
        confidence = gate_result["confidence"]
        act_on_signal = gate_result["act_on_signal"]
        range_low = ensemble_result["range_low"]
        range_high = ensemble_result["range_high"]
        most_likely = ensemble_result["most_likely"]

        # Enriched prefunding guidance
        prefunding = self._build_prefunding_guidance(
            range_low, range_high, most_likely, current_rate,
            direction, confidence, act_on_signal,
        )

        # Consensus view across all 5 models
        consensus = self._build_consensus_view(
            ensemble_result, xgb_result, sentiment_result,
            macro_result, intraday_result,
        )

        return {
            "timestamp": now.isoformat(),
            "current_rate": round(current_rate, 4),
            "prediction_48h": {
                "direction": direction,
                "range_low": range_low,
                "range_high": range_high,
                "most_likely": most_likely,
                "confidence": confidence,
                "confidence_label": self._confidence_label(confidence),
                "prefunding_guidance": prefunding,
            },
            "consensus_view": consensus,
            "signal_strength": gate_result["signal_strength"],
            "act_on_signal": act_on_signal,
            "model_breakdown": {
                "xgboost": {
                    "role": "regime_classifier",
                    "regime": xgb_result.get("regime", "range_bound"),
                    "regime_confidence": xgb_result.get("regime_confidence", 0.0),
                    "confidence_adjustment": ensemble_result.get("regime_adjustment", 0.0),
                    "regime_probs": xgb_result.get("regime_probs", {}),
                    "note": self._regime_note(xgb_result.get("regime", "range_bound")),
                },
                "lstm": {
                    "role": "range_provider",
                    "model_version": lstm_result.get("model_version", "unknown"),
                    "range_low": lstm_result.get("predicted_low"),
                    "range_high": lstm_result.get("predicted_high"),
                    "most_likely": lstm_result.get("predicted_close"),
                    "range_width": lstm_result.get("range_width", 0.0),
                    "range_width_label": lstm_result.get("range_width_label", "N/A"),
                    "range_risk": lstm_result.get("range_risk", {}),
                    "note": lstm_result.get("note", "Range only — direction from sentiment + macro votes"),
                },
                "sentiment": {
                    "role": "direction_signal",
                    "direction": ensemble_result.get("sent_direction", "NEUTRAL"),
                    "score": round(sentiment_result.get("score", 0.0), 4),
                    "confidence": round(sentiment_result.get("confidence", 0.0), 4),
                    "is_primary_signal": True,
                    "explanation": sentiment_result.get("explanation", ""),
                    "data_quality": sentiment_result.get("data_quality", "none"),
                },
                "macro": {
                    "role": "macro_direction",
                    "direction": macro_result.get("direction", "NEUTRAL"),
                    "score": macro_result.get("score", 0.0),
                    "confidence": macro_result.get("confidence", 0.0),
                    "fed_funds_change_3m": macro_result.get("fed_funds_change_3m"),
                    "yield_curve_spread": macro_result.get("yield_curve_spread"),
                    "cpi_yoy": macro_result.get("cpi_yoy"),
                    "interpretation": macro_result.get("interpretation", ""),
                    "available": macro_result.get("available", False),
                },
                "intraday_lstm": intraday_result if intraday_result else {
                    "role": "intraday_momentum",
                    "signal": "UNCONFIRMED",
                    "raw_direction": "NEUTRAL",
                    "raw_confidence": 0.0,
                    "confidence_adjustment": 0.0,
                    "accuracy_note": "Intraday model not run",
                    "bars_analyzed": 0,
                    "model_limitation": "Asymmetric — only UP signals integrated",
                },
            },
            "vote_outcome": ensemble_result.get("vote_outcome", "unknown"),
            "confidence_breakdown": {
                "base_confidence": ensemble_result.get("confidence", 0.0),
                "agreement_bonus_applied": ensemble_result.get("agreement_bonus_applied", 0.0),
                "signals_in_agreement": ensemble_result.get("signals_in_agreement", 0),
                "sentiment_contribution": ensemble_result.get("sentiment_contribution", 0.0),
                "macro_contribution": ensemble_result.get("macro_contribution", 0.0),
                "regime_adjustment": ensemble_result.get("regime_adjustment", 0.0),
                "intraday_adjustment": ensemble_result.get("intraday_adjustment", 0.0),
                "hard_limits": "[0.30, 0.82]",
            },
            "architecture": {
                "direction_models": "Sentiment (primary) + Macro (secondary)",
                "range_model": f"LSTM-{self.model_versions.get('lstm', 'v2')} (12 features, 30-day sequences, BoundaryAwareLoss)",
                "regime_model": "XGBoost-regime-classifier",
                "intraday_model": "Intraday-LSTM-v1 (asymmetric UP-only)",
                "note": "Direction from sentiment + macro vote. LSTM provides price range. "
                        "XGBoost detects regime and adjusts confidence. "
                        "Intraday LSTM provides momentum confirmation.",
            },
            "key_drivers": key_drivers,
            "range_risk": lstm_result.get("range_risk", {}),
            "risk_flags": gate_result["risk_flags"],
            "metadata": {
                "models_loaded": self.models_loaded,
                "model_versions": self.model_versions,
                "sentiment_source": sentiment_result.get("data_quality", "unknown"),
                "feature_count": feature_count,
                "data_rows_used": data_rows,
                "prediction_generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            },
        }

    # -----------------------------------------------------------------------
    # Main predict method
    # -----------------------------------------------------------------------

    def predict(self, df: pd.DataFrame, df_4h: pd.DataFrame = None) -> dict:
        """
        Main entry point. Takes a market dataframe and returns a full
        structured prediction.

        Args:
            df: DataFrame with columns [date, usdinr, oil, dxy, vix, us10y]
            df_4h: Optional 4-hour OHLCV data for intraday model.
                   If None and intraday model is loaded, reads from CSV.

        Returns:
            Prediction dict (always valid JSON, never raises).
        """
        print("\n  Running FX Prediction Agent...")
        current_rate = float(df["usdinr"].iloc[-1])
        data_rows = len(df)

        # Build features once — used by XGBoost, LSTM (v2), safety gate, key drivers
        features_df = build_features(df)

        # --- Run all models ---
        print("  [1/5] Running XGBoost regime classifier...")
        xgb_result = self._run_xgboost(df, features_df=features_df)
        print(f"        → regime: {xgb_result['regime']} (conf: {xgb_result['regime_confidence']:.3f})")

        print("  [2/5] Running LSTM range predictor...")
        lstm_result = self._run_lstm(df, features_df=features_df)
        print(f"        → range: {lstm_result.get('predicted_low', 'N/A')} — {lstm_result.get('predicted_high', 'N/A')} "
              f"(close: {lstm_result.get('predicted_close', 'N/A')})")

        print("  [3/5] Running sentiment agent (DIRECTION vote 1)...")
        sentiment_result = self._run_sentiment()
        print(f"        → {sentiment_result['direction']} (score: {sentiment_result.get('score', 0):.2f}, "
              f"conf: {sentiment_result.get('confidence', 0):.2f})")

        print("  [4/5] Running macro signal (DIRECTION vote 2)...")
        macro_result = self._run_macro(features_df)
        if macro_result.get("available"):
            print(f"        → {macro_result['direction']} (score: {macro_result['score']:+.3f}, "
                  f"conf: {macro_result['confidence']:.2f})")
            print(f"        → {macro_result['interpretation']}")
        else:
            print("        → unavailable (FRED features missing)")

        print("  [5/5] Running intraday LSTM (momentum)...")
        intraday_result = self._run_intraday_lstm(df_4h)
        print(f"        → signal: {intraday_result['signal']} "
              f"(raw: {intraday_result['raw_direction']}, "
              f"conf: {intraday_result['raw_confidence']:.3f}, "
              f"adj: {intraday_result['confidence_adjustment']:+.2f})")

        # --- Ensemble (with all confidence improvements) ---
        print("  [ensemble] Combining model outputs...")
        ensemble_result = self._ensemble(xgb_result, lstm_result, sentiment_result,
                                          macro_result, current_rate,
                                          intraday_result=intraday_result)
        vote = ensemble_result.get("vote_outcome", "?")
        agr_bonus = ensemble_result.get("agreement_bonus_applied", 0.0)
        n_agree = ensemble_result.get("signals_in_agreement", 0)
        intra_adj = ensemble_result.get("intraday_adjustment", 0.0)
        print(f"        → {ensemble_result['direction']} (conf: {ensemble_result['confidence']:.3f}, "
              f"strength: {ensemble_result['signal_strength']}, vote: {vote})")
        if agr_bonus > 0:
            print(f"        → agreement bonus: +{agr_bonus:.2f} ({n_agree} signals agree)")
        if intra_adj != 0:
            print(f"        → intraday adjustment: {intra_adj:+.2f}")
        sent_c = ensemble_result.get("sentiment_contribution", 0.0)
        macro_c = ensemble_result.get("macro_contribution", 0.0)
        if sent_c > 0 or macro_c > 0:
            print(f"        → proportional: sentiment={sent_c:+.3f}, macro={macro_c:+.3f}")

        # Update intraday integration flag
        intraday_result["integrated_into_ensemble"] = (intra_adj != 0.0)

        # --- Safety gate ---
        print("  [safety] Applying safety gate rules...")
        gate_result = self._safety_gate(ensemble_result, sentiment_result, df,
                                         features_df=features_df)
        print(f"        → act_on_signal: {gate_result['act_on_signal']}")
        if gate_result["risk_flags"]:
            for flag in gate_result["risk_flags"]:
                print(f"        → RISK: {flag}")

        # --- Key drivers ---
        key_drivers = self._extract_key_drivers(df, features_df=features_df)

        # --- Feature count (regime features for XGB, total engineered for display) ---
        try:
            feature_count = len(features_df.columns) - 6  # subtract raw market cols
        except Exception:
            feature_count = len(self.xgb_feature_names) if self.xgb_feature_names else 0

        # --- Build output ---
        output = self._build_output(
            current_rate=current_rate,
            gate_result=gate_result,
            ensemble_result=ensemble_result,
            xgb_result=xgb_result,
            lstm_result=lstm_result,
            sentiment_result=sentiment_result,
            macro_result=macro_result,
            key_drivers=key_drivers,
            data_rows=data_rows,
            feature_count=feature_count,
            intraday_result=intraday_result,
        )

        # --- Surface range risk in risk_flags and adjust prefunding ---
        range_risk = lstm_result.get("range_risk", {})
        risk_level = range_risk.get("risk_level", "NORMAL")
        if risk_level in ("HIGH", "CRITICAL"):
            output["risk_flags"].append(
                f"Range prediction risk: {risk_level} — "
                f"{range_risk.get('treasury_action', '')}"
            )
            # Adjust prefunding guidance upward
            buffer_paise = range_risk.get("recommended_buffer_paise", 0)
            if buffer_paise > 0:
                pf = output.get("prediction_48h", {}).get("prefunding_guidance", {})
                if pf and "recommended_rate" in pf:
                    pf["recommended_rate"] = round(pf["recommended_rate"] + buffer_paise / 100, 4)
                    pf["basis"] = (
                        f"range_high + {buffer_paise}p risk buffer "
                        f"({risk_level} risk day)"
                    )
            print(f"  [risk] {risk_level} — {range_risk.get('risk_factors', ['none'])[0][:60]}")
        elif risk_level == "ELEVATED":
            print(f"  [risk] ELEVATED — {range_risk.get('risk_factors', ['none'])[0][:60]}")

        # --- Opus enrichment: 3 parallel calls ---
        features_row = features_df.iloc[-1].to_dict()
        final_confidence = gate_result["confidence"]

        if final_confidence > self.REASONING_CONFIDENCE_THRESHOLD:
            print("  [opus] Running 3 parallel enrichment calls...")
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_briefing = pool.submit(
                    self._generate_morning_briefing, output, features_row)
                fut_whys = pool.submit(
                    self._generate_model_whys, output)
                fut_context = pool.submit(
                    self._generate_context_interpretations, output, features_row)

            briefing = fut_briefing.result()
            whys = fut_whys.result()
            context = fut_context.result()

            total_tokens = (briefing.get("tokens_used", 0) +
                            whys.get("_tokens", 0) +
                            context.get("_tokens", 0))
            models_used = {briefing.get("generated_by"),
                           whys.get("_model"),
                           context.get("_model")} - {None}
            print(f"  [opus] Done — {total_tokens} tokens across {len(models_used)} model(s)")
        else:
            print(f"  [opus] Skipped (confidence {final_confidence:.2f} < "
                  f"{self.REASONING_CONFIDENCE_THRESHOLD}) — using fallbacks")
            briefing = self._fallback_morning_briefing(output)
            whys = self._fallback_model_whys(output)
            context = self._fallback_context_interpretations(features_row)

        # Inject morning briefing
        output["morning_briefing"] = briefing

        # Inject per-model "why" into model_breakdown
        for model_key in ("xgboost", "lstm", "sentiment", "macro", "intraday_lstm"):
            why_key = model_key.replace("intraday_lstm", "intraday") + "_why"
            if why_key in whys and model_key in output["model_breakdown"]:
                output["model_breakdown"][model_key]["why"] = whys[why_key]

        # Inject context interpretations (stored for API layer to merge with NDF/macro)
        output["_context_interpretations"] = {
            "ndf_interpretation": context.get("ndf_interpretation"),
            "macro_interpretation": context.get("macro_interpretation"),
        }

        # Legacy reasoning field (from existing Opus call — now replaced by briefing)
        output["reasoning"] = {
            "model": briefing.get("generated_by"),
            "analysis": briefing.get("briefing"),
            "generated": briefing.get("generated_by") != "rule_based_fallback",
            "skip_reason": None if briefing.get("generated_by") != "rule_based_fallback"
                          else "Using rule-based fallback",
        }

        return output

    # -----------------------------------------------------------------------
    # Weekly forecast
    # -----------------------------------------------------------------------

    def predict_weekly(self, df: pd.DataFrame) -> dict:
        """
        Generate 7-day USD/INR forecast with full calendar awareness.

        Uses today's ensemble prediction as the base signal, then projects it
        across the next 7 calendar days with:
          - 5% confidence decay per day
          - Calendar event overrides (RBI/Fed blackout, month-end bias)
          - Non-trading day markers (weekends, holidays)

        Returns a structured response with daily_forecasts and week_summary.

        Business logic: company RECEIVES dollars, converts to INR.
        Rate UP = good (more INR per dollar).
        Best conversion day = highest confidence UP signal.
        """
        print("\n  Running Weekly Forecast...")

        # --- Base prediction (today's 48h call) ---
        base = self.predict(df)
        today = date.today()
        current_rate = base["current_rate"]

        base_direction = base["prediction_48h"]["direction"]
        base_confidence = base["prediction_48h"]["confidence"]

        # =================================================================
        # Three-force projection model
        # =================================================================
        xgb = base["model_breakdown"]["xgboost"]
        lstm = base["model_breakdown"]["lstm"]
        sent = base["model_breakdown"]["sentiment"]

        # Force 1: Sentiment-driven drift (sole direction signal)
        sentiment_score = sent.get("score", 0.0)  # -1 to +1
        regime = xgb.get("regime", "range_bound")

        # Sentiment direction drives drift:
        #   score < -0.20 → UP direction → positive drift
        #   score > +0.20 → DOWN direction → negative drift
        #   otherwise     → no drift (mean reversion only)
        if sentiment_score < -0.20:
            momentum_per_day = abs(sentiment_score) * 0.05   # UP drift
        elif sentiment_score > 0.20:
            momentum_per_day = -abs(sentiment_score) * 0.05  # DOWN drift
        else:
            momentum_per_day = 0.0  # no directional drift

        # Force 2: Mean reversion anchor (30-day MA)
        ma_30 = float(df["usdinr"].rolling(30).mean().iloc[-1])
        reversion_gap = ma_30 - current_rate

        # Force 3: Structural trend drift (from features)
        features_df = build_features(df)
        rate_trend_30d = float(features_df["rate_trend_30d"].iloc[-1])

        print(f"  [weekly] current_rate={current_rate:.4f}, ma_30={ma_30:.4f}")
        print(f"  [weekly] sentiment_score={sentiment_score:+.2f}, regime={regime}")
        print(f"  [weekly] drift_per_day={momentum_per_day:+.4f}, "
              f"reversion_gap={reversion_gap:+.4f}, trend_30d={rate_trend_30d:+.6f}")

        # Debug table header
        print(f"\n  {'Day':<4s} | {'Sent.Drift':>10s} | {'Reversion':>9s} | {'Trend':>7s} | "
              f"{'Total':>7s} | {'most_likely':>11s}")
        print(f"  {'─'*4}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*11}")

        # --- Project across 7 calendar days ---
        days = get_next_7_calendar_days(today)
        risk_events = get_week_risk_events(today, 14)

        daily_forecasts = []
        best_up_day = None
        best_up_confidence = 0.0
        days_to_avoid = []
        direction_scores = []  # for overall bias

        CONFIDENCE_DECAY_RATE = 0.90  # each day retains 90% of previous
        CONFIDENCE_FLOOR = 0.20      # never below 20% — we always know something
        trading_day_n = 1            # counts only trading days for decay exponent

        for i, d in enumerate(days):
            day_offset = i + 1
            ctx = get_calendar_context(d)

            if not ctx["is_trading_day"]:
                daily_forecasts.append({
                    "date": d.isoformat(),
                    "day_name": ctx["day_name"],
                    "day_offset": day_offset,
                    "is_trading_day": False,
                    "skip_reason": ctx["skip_reason"],
                    "prediction": None,
                    "calendar_note": ctx["special_note"],
                    "override_reason": None,
                })
                days_to_avoid.append({
                    "date": d.isoformat(),
                    "reason": ctx["skip_reason"],
                })
                continue

            # Apply multiplicative confidence decay with floor
            decay_base = max(0.35, base_confidence)
            decayed_confidence = max(CONFIDENCE_FLOOR,
                                     decay_base * (CONFIDENCE_DECAY_RATE ** (trading_day_n - 1)))
            trading_day_n += 1
            direction = base_direction

            # Calendar overrides
            override_reason = None

            if ctx["is_rbi_day"] or ctx["is_fed_day"]:
                # Central bank decision day — blackout
                direction = "NEUTRAL"
                decayed_confidence = 0.15
                label = "RBI rate decision" if ctx["is_rbi_day"] else "Fed rate decision"
                override_reason = f"{label} day — prediction unreliable"
                days_to_avoid.append({
                    "date": d.isoformat(),
                    "reason": label,
                })
            elif ctx["is_rbi_week"]:
                decayed_confidence = max(CONFIDENCE_FLOOR, decayed_confidence * 0.90)
                override_reason = "RBI meeting week — elevated uncertainty"
            elif ctx["is_fed_week"]:
                decayed_confidence = max(CONFIDENCE_FLOOR, decayed_confidence * 0.90)
                override_reason = "Fed meeting week — dollar volatility likely"

            # Month-end structural pattern: importers buy dollars → rate tends UP
            if ctx["is_month_end"] and direction == "DOWN":
                decayed_confidence *= 0.80
                me_note = "Month-end: structural USD demand may counteract DOWN signal"
                override_reason = f"{override_reason} | {me_note}" if override_reason else me_note

            decayed_confidence = round(decayed_confidence, 3)

            # --- Three-force projection ---
            # Momentum: strong early, fades with time
            momentum_weight = max(0.20, 1.0 - (day_offset - 1) * 0.13)
            momentum_contribution = momentum_per_day * day_offset * momentum_weight

            # Mean reversion: weak early, grows with time
            reversion_weight = min(0.40, day_offset * 0.06)
            reversion_contribution = reversion_gap * reversion_weight

            # Trend drift: constant small push, capped at ±10 paise/day
            capped_trend = min(0.10, max(-0.10, rate_trend_30d))
            trend_contribution = capped_trend * day_offset

            # Combine and cap at ±2.0 INR
            total_change = momentum_contribution + reversion_contribution + trend_contribution
            total_change = max(-2.0, min(2.0, total_change))

            most_likely = round(current_rate + total_change, 4)

            # Range: widens with uncertainty, asymmetric if strong momentum
            base_range = 0.20
            daily_expansion = 0.07
            range_buffer = base_range + (day_offset * daily_expansion)

            if momentum_per_day > 0.05:
                day_range_low = round(most_likely - range_buffer * 0.7, 4)
                day_range_high = round(most_likely + range_buffer * 1.3, 4)
            elif momentum_per_day < -0.05:
                day_range_low = round(most_likely - range_buffer * 1.3, 4)
                day_range_high = round(most_likely + range_buffer * 0.7, 4)
            else:
                day_range_low = round(most_likely - range_buffer, 4)
                day_range_high = round(most_likely + range_buffer, 4)

            # Debug table row
            day_label = d.strftime("%a")
            print(f"  {day_label:<4s} | {momentum_contribution:+10.4f} | {reversion_contribution:+9.4f} | "
                  f"{trend_contribution:+7.4f} | {total_change:+7.4f} | {most_likely:11.4f}")

            # Signal strength
            if decayed_confidence > 0.72:
                signal_strength = "STRONG"
            elif decayed_confidence > 0.58:
                signal_strength = "MEDIUM"
            else:
                signal_strength = "WEAK"

            # Treasury enrichment per day
            vol = self.DAILY_VOLUME_USD
            tpv_prefund = round(most_likely * vol)
            tpv_high = round(day_range_high * vol)
            rate_risk = tpv_high - round(day_range_low * vol)

            if direction == "DOWN" and decayed_confidence >= 0.55:
                treasury_note = f"Rate falling — prefund at {most_likely:.2f}, save {self._format_inr(rate_risk)}"
            elif direction == "UP" and decayed_confidence >= 0.55:
                treasury_note = f"Rate rising — hold dollars, convert later at {day_range_high:.2f}"
            elif override_reason:
                treasury_note = f"Caution: {override_reason}"
            else:
                treasury_note = f"No strong signal — prefund conservatively at {day_range_high:.2f}"

            forecast = {
                "date": d.isoformat(),
                "day_name": ctx["day_name"],
                "day_offset": day_offset,
                "is_trading_day": True,
                "prediction": {
                    "direction": direction,
                    "confidence": decayed_confidence,
                    "signal_strength": signal_strength,
                    "range_low": day_range_low,
                    "range_high": day_range_high,
                    "most_likely": most_likely,
                },
                "treasury_note": treasury_note,
                "prefunding_rate": most_likely,
                "tpv_at_prefunding_rate": tpv_prefund,
                "rate_risk_inr": rate_risk,
                "rate_risk_label": self._format_inr(rate_risk),
                "calendar_note": ctx["special_note"],
                "override_reason": override_reason,
            }
            daily_forecasts.append(forecast)

            # Track direction scores for overall bias
            if direction == "UP":
                dir_score = decayed_confidence
            elif direction == "DOWN":
                dir_score = -decayed_confidence
            else:
                dir_score = 0.0
            direction_scores.append(dir_score)

            # Track best UP day (for best_conversion_day)
            if direction == "UP" and decayed_confidence > best_up_confidence:
                best_up_confidence = decayed_confidence
                best_up_day = {
                    "date": d.isoformat(),
                    "day_name": ctx["day_name"],
                    "confidence": decayed_confidence,
                    "reason": f"Highest confidence UP signal ({decayed_confidence * 100:.0f}%)",
                }

        # --- Overall week bias ---
        if direction_scores:
            avg_score = sum(direction_scores) / len(direction_scores)
            if avg_score > 0.05:
                overall_bias = "UP"
            elif avg_score < -0.05:
                overall_bias = "DOWN"
            else:
                overall_bias = "NEUTRAL"
        else:
            overall_bias = "NEUTRAL"

        trading_days_count = sum(1 for f in daily_forecasts if f["is_trading_day"])

        # --- Enriched week summary ---
        # Weekly rate band: min/max across all trading day forecasts
        trading_lows = [f["prediction"]["range_low"] for f in daily_forecasts
                        if f["is_trading_day"] and f["prediction"]]
        trading_highs = [f["prediction"]["range_high"] for f in daily_forecasts
                         if f["is_trading_day"] and f["prediction"]]
        week_low = min(trading_lows) if trading_lows else current_rate
        week_high = max(trading_highs) if trading_highs else current_rate
        vol = self.DAILY_VOLUME_USD
        tpv_week_diff = round((week_high - week_low) * vol)

        weekly_rate_band = {
            "week_low": round(week_low, 4),
            "week_high": round(week_high, 4),
            "band_width_paise": round((week_high - week_low) * 100, 1),
            "tpv_at_week_high": round(week_high * vol),
            "tpv_at_week_low": round(week_low * vol),
            "weekly_tpv_range_inr": tpv_week_diff,
            "weekly_tpv_range_label": f"{self._format_inr(tpv_week_diff)} at stake this week",
        }

        # Conversion windows
        convert_now_days = [f["date"] for f in daily_forecasts
                            if f["is_trading_day"] and f["prediction"]
                            and f["prediction"]["direction"] == "UP"
                            and f["prediction"]["confidence"] >= 0.55]
        hold_days = [f["date"] for f in daily_forecasts
                     if f["is_trading_day"] and f["prediction"]
                     and f["prediction"]["direction"] == "DOWN"
                     and f["prediction"]["confidence"] >= 0.55]
        conversion_windows = {
            "convert_now_days": convert_now_days,
            "hold_days": hold_days,
            "uncertain_days": trading_days_count - len(convert_now_days) - len(hold_days),
        }

        # Week risk summary
        n_risks = len(risk_events) + len(days_to_avoid)
        if n_risks == 0:
            risk_label = "LOW — clear week, no major events"
        elif n_risks <= 2:
            risk_label = "MODERATE — some events to watch"
        else:
            risk_label = "HIGH — multiple risk events, tread carefully"

        week_summary = {
            "overall_bias": overall_bias,
            "best_conversion_day": best_up_day,
            "days_to_avoid": days_to_avoid,
            "risk_events": risk_events,
            "trading_days_count": trading_days_count,
            "weekly_rate_band": weekly_rate_band,
            "conversion_windows": conversion_windows,
            "week_risk_summary": risk_label,
            "recommendation": self._weekly_recommendation(
                overall_bias, best_up_day, days_to_avoid, trading_days_count,
            ),
        }

        print(f"  [weekly] Bias: {overall_bias}, Trading days: {trading_days_count}")
        if best_up_day:
            print(f"  [weekly] Best conversion day: {best_up_day['day_name']} {best_up_day['date']}")
        print(f"  [weekly] Days to avoid: {len(days_to_avoid)}, Risk events: {len(risk_events)}")

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_rate": current_rate,
            "base_prediction": base["prediction_48h"],
            "daily_forecasts": daily_forecasts,
            "week_summary": week_summary,
            "model_breakdown": base["model_breakdown"],
            "key_drivers": base.get("key_drivers", []),
            "risk_flags": base.get("risk_flags", []),
            "projection_params": {
                "momentum_per_day": round(momentum_per_day, 6),
                "reversion_gap": round(reversion_gap, 4),
                "rate_trend_30d": round(rate_trend_30d, 6),
                "ma_30": round(ma_30, 4),
            },
            "metadata": base["metadata"],
        }

    def _weekly_recommendation(self, overall_bias: str, best_up_day: dict | None,
                                days_to_avoid: list, trading_days_count: int) -> str:
        """Generate human-readable weekly conversion recommendation."""
        if trading_days_count == 0:
            return "No trading days this week. Schedule conversions for next week."

        if overall_bias == "UP" and best_up_day:
            # Rate rising → convert now to lock in current level
            conf_pct = best_up_day.get("confidence", 0) * 100
            # Estimate gain: best_day rate vs current
            gain = 9_500_000 * 0.10  # conservative 10 paise estimate
            return (
                f"FX Band Predictor signals: convert now. "
                f"Rate expected to rise — lock in today's level. "
                f"Estimated gain: ₹{gain:,.0f} on $9.5M."
            )
        elif overall_bias == "DOWN":
            saving = 9_500_000 * 0.10  # conservative 10 paise estimate
            return (
                f"FX Band Predictor signals: hold your dollars. "
                f"Rate expected to fall — wait for better INR rate. "
                f"Potential saving: ₹{saving:,.0f} on $9.5M."
            )
        else:
            return (
                "FX Band Predictor sees no clear edge this week. "
                "Convert operational minimums. Protect ₹86 crore "
                "from unnecessary rate risk."
            )


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def print_summary(output: dict) -> None:
    """Print a formatted human-readable prediction summary."""
    pred = output["prediction_48h"]
    breakdown = output["model_breakdown"]

    date_str = datetime.now().strftime("%b %d, %Y")
    direction = pred["direction"]
    strength = output["signal_strength"]
    act = "YES" if output["act_on_signal"] else "NO"

    print()
    print("=" * 50)
    print(f"  FX BAND PREDICTOR — {date_str}")
    print("=" * 50)
    print(f"  Current Rate:    {output['current_rate']:.4f} USD/INR")
    print(f"  48h Direction:   {direction} ({strength} signal)")
    print(f"  Predicted Range: {pred['range_low']:.2f} — {pred['range_high']:.2f}")
    print(f"  Most Likely:     {pred['most_likely']:.4f}")
    print(f"  Confidence:      {pred['confidence'] * 100:.1f}%")
    print(f"  Act on Signal:   {act}")

    vote = output.get("vote_outcome", "?")

    print()
    print("  Model Outputs:")
    xgb = breakdown["xgboost"]
    lstm = breakdown["lstm"]
    sent = breakdown["sentiment"]
    macro = breakdown.get("macro", {})
    print(f"    XGBoost:   REGIME={xgb['regime']:<13s} (conf: {xgb.get('regime_confidence', 0):.2f}, adj: {xgb['confidence_adjustment']:+.2f})")
    range_str = f"{lstm.get('range_low', 'N/A')} — {lstm.get('range_high', 'N/A')}"
    print(f"    LSTM:      RANGE  ({range_str}) [RANGE PROVIDER]")
    print(f"    Sentiment: {sent['direction']:<8s} (score: {sent['score']:.2f}, conf: {sent.get('confidence', 0):.2f}) [VOTE 1]")
    if macro.get("available"):
        print(f"    Macro:     {macro['direction']:<8s} (score: {macro['score']:+.3f}, conf: {macro.get('confidence', 0):.2f}) [VOTE 2]")
        print(f"               {macro.get('interpretation', '')}")
    else:
        print(f"    Macro:     unavailable [VOTE 2]")
    intraday = breakdown.get("intraday_lstm", {})
    signal = intraday.get("signal", "N/A")
    raw_dir = intraday.get("raw_direction", "N/A")
    intra_conf = intraday.get("raw_confidence", 0.0)
    intra_adj = intraday.get("confidence_adjustment", 0.0)
    print(f"    Intraday:  {signal:<15s} (raw: {raw_dir}, conf: {intra_conf:.2f}, adj: {intra_adj:+.2f}) [MOMENTUM]")
    print(f"    Vote:      {vote}")

    print()
    flags = output.get("risk_flags", [])
    if flags:
        print("  Risk Flags:")
        for f in flags:
            print(f"    ! {f}")
    else:
        print("  Risk Flags: None")

    print()
    drivers = output.get("key_drivers", [])
    if drivers:
        print("  Key Drivers:")
        for d in drivers:
            print(f"    -> {d}")

    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = os.path.join(PROJECT_DIR, "data", "market_data.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run data/fetch_market_data.py first.")
        sys.exit(1)

    print("Loading market data...")
    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"Shape: {df.shape}\n")

    print("Initializing FX Prediction Agent...")
    agent = FXPredictionAgent()

    output = agent.predict(df)

    # Print full JSON
    print("\n\n" + "=" * 50)
    print("  FULL PREDICTION JSON")
    print("=" * 50)
    print(json.dumps(output, indent=2, default=str))

    # Print human-readable summary
    print_summary(output)

    print("\n--- Phase 6 complete ---")
