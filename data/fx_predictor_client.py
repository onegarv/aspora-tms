"""
HTTP client for the FX Band Predictor API.

The FX Band Predictor runs as a separate process (default port 8001).
This module provides a clean interface for agents to fetch predictions.

Analogous to data/metabase.py for Metabase access.
"""
from __future__ import annotations

import http.client
import json
import logging
from typing import Optional

logger = logging.getLogger("tms.data.fx_predictor")

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 8001
_TIMEOUT = 30  # seconds


def fetch_prediction(
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    timeout: int = _TIMEOUT,
) -> Optional[dict]:
    """
    GET /predict from the FX Band Predictor API.

    Returns the full prediction JSON dict, or None on failure.
    Caller should use fallback values when None is returned.
    """
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/predict")
        resp = conn.getresponse()

        if resp.status != 200:
            logger.warning("FX Predictor API returned status %d", resp.status)
            return None

        data = json.loads(resp.read())
        logger.info(
            "FX prediction fetched: rate=%.4f direction=%s confidence=%.2f",
            data.get("current_rate", 0),
            data.get("prediction_48h", {}).get("direction", "N/A"),
            data.get("prediction_48h", {}).get("confidence", 0),
        )
        return data

    except Exception as exc:
        logger.warning(
            "FX Predictor API call failed (%s: %s)",
            type(exc).__name__, exc,
        )
        return None

    finally:
        try:
            conn.close()
        except Exception:
            pass


def extract_spot_rate(prediction: dict) -> Optional[float]:
    """Extract the current USD/INR spot rate from a prediction dict."""
    rate = prediction.get("current_rate")
    return float(rate) if rate is not None else None


def extract_band(prediction: dict) -> Optional[dict]:
    """
    Extract the 48h prediction band.

    Returns dict with direction, range, confidence, prefunding guidance
    or None if prediction is malformed.
    """
    pred = prediction.get("prediction_48h")
    if not pred:
        return None

    guidance = pred.get("prefunding_guidance", {})

    return {
        "direction": pred.get("direction", "NEUTRAL"),
        "range_low": pred.get("range_low"),
        "range_high": pred.get("range_high"),
        "most_likely": pred.get("most_likely"),
        "confidence": pred.get("confidence", 0.0),
        "recommended_rate": guidance.get("recommended_rate"),
        "action": guidance.get("action"),
    }
