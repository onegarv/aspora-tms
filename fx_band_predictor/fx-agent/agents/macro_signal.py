"""
Macro Direction Signal — Rule-based signal from FRED economic data

NOT a trained ML model. Uses simple thresholds on:
  - Fed Funds rate change (3-month) → Fed tightening/easing cycle
  - Yield curve spread (10Y - 2Y) → recession/risk indicator
  - CPI year-over-year → inflation pressure on Fed policy

Score convention (same as sentiment):
    positive score = bearish INR (rate goes UP — USD strengthening)
    negative score = bullish INR (rate goes DOWN — USD weakening)

Gracefully returns NEUTRAL if FRED features are missing.
"""

import pandas as pd


def get_macro_signal(df: pd.DataFrame) -> dict:
    """
    Compute rule-based macro direction signal from FRED features.

    Args:
        df: Feature-engineered DataFrame (output of build_features).
            Needs: fed_funds_change_3m, yield_curve_spread, cpi_yoy

    Returns:
        Dict with role, score, direction, confidence, raw values, interpretation.
        Never raises — returns NEUTRAL on any failure.
    """
    fallback = {
        "role": "macro_direction",
        "score": 0.0,
        "direction": "NEUTRAL",
        "confidence": 0.0,
        "fed_funds_change_3m": None,
        "yield_curve_spread": None,
        "cpi_yoy": None,
        "interpretation": "FRED features unavailable",
        "available": False,
    }

    try:
        latest = df.iloc[-1]

        # Check that FRED features exist
        required = ["fed_funds_change_3m", "yield_curve_spread", "cpi_yoy"]
        if not all(col in df.columns for col in required):
            return fallback

        fed_change = float(latest["fed_funds_change_3m"])
        yc_spread = float(latest["yield_curve_spread"])
        cpi_yoy_val = float(latest["cpi_yoy"])

        # Check for NaN
        if pd.isna(fed_change) or pd.isna(yc_spread) or pd.isna(cpi_yoy_val):
            return fallback

        score = 0.0  # positive = bearish INR (rate goes UP)

        # --- Fed tightening = dollar strengthens = rate UP ---
        if fed_change > 0.25:
            score += 0.3   # Fed hiking
        elif fed_change < -0.25:
            score -= 0.3   # Fed cutting

        # --- Yield curve inversion = dollar strong = rate UP ---
        if yc_spread < 0:
            score += 0.2   # inverted = dollar bullish
        elif yc_spread > 1.0:
            score -= 0.1   # steep curve = risk-on = INR support

        # --- High inflation = Fed stays hawkish = dollar strong ---
        cpi_pct = cpi_yoy_val * 100  # convert from decimal to percentage
        if cpi_pct > 3.5:
            score += 0.15
        elif cpi_pct < 2.5:
            score -= 0.10

        # Cap at -1 to +1
        score = max(-1.0, min(1.0, score))

        # Convert to direction
        if score > 0.25:
            direction = "UP"      # bearish INR
        elif score < -0.25:
            direction = "DOWN"    # bullish INR
        else:
            direction = "NEUTRAL"

        confidence = min(0.70, abs(score) + 0.30)

        return {
            "role": "macro_direction",
            "score": round(score, 3),
            "direction": direction,
            "confidence": round(confidence, 4),
            "fed_funds_change_3m": round(fed_change, 4),
            "yield_curve_spread": round(yc_spread, 4),
            "cpi_yoy": round(cpi_pct, 2),
            "interpretation": _generate_macro_note(fed_change, yc_spread, cpi_pct),
            "available": True,
        }

    except Exception as e:
        fallback["interpretation"] = f"Macro signal error: {e}"
        return fallback


def _generate_macro_note(fed_change: float, yc_spread: float, cpi_pct: float) -> str:
    """Generate human-readable interpretation of macro conditions."""
    notes = []

    if fed_change > 0.25:
        notes.append("Fed tightening cycle — dollar bullish")
    elif fed_change < -0.25:
        notes.append("Fed easing cycle — dollar bearish")

    if yc_spread < 0:
        notes.append("Yield curve inverted — historically dollar positive")
    elif yc_spread > 1.5:
        notes.append("Steep yield curve — risk-on, EM-supportive")

    if cpi_pct > 3.5:
        notes.append(f"US CPI at {cpi_pct:.1f}% — Fed hawkish bias")
    elif cpi_pct < 2.0:
        notes.append(f"US CPI at {cpi_pct:.1f}% — disinflation, easing likely")

    return " | ".join(notes) if notes else "Neutral macro environment"
