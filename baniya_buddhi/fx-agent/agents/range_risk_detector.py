"""
Baniya Buddhi — Range Prediction Risk Detector

Detects when the LSTM range prediction is likely unreliable.
Warns treasury to add manual buffer.

Does NOT widen range automatically — that would defeat the
purpose of tight accurate ranges. Instead: flags clearly
so treasury acts consciously.
"""


def detect_high_risk_prediction(features: dict) -> dict:
    """
    Detect when range prediction is likely unreliable.

    Args:
        features: dict with keys like rate_vs_alltime_percentile,
                  momentum_consistency, volatility_20d, volatility_5d,
                  current_rate, current_regime, prev_regime,
                  is_fomc_day, is_rbi_day, day_of_week, etc.

    Returns:
        dict with risk_level, risk_score, risk_factors,
        prediction_reliability, treasury_action,
        recommended_buffer_paise, tpv_buffer_inr, tpv_buffer_label
    """
    risk_factors = []
    risk_score = 0

    # FACTOR 1: Rate at extreme percentile
    pct = features.get("rate_vs_alltime_percentile", 0)
    if pct >= 0.999:
        risk_factors.append(
            f"Rate at 99.9th percentile of 22yr history "
            f"— mean reversion risk is highest here"
        )
        risk_score += 3
    elif pct >= 0.995:
        risk_factors.append(
            f"Rate at 99.5th+ percentile — "
            f"limited historical precedent for predictions"
        )
        risk_score += 2

    # FACTOR 2: Negative momentum at extreme high
    momentum = features.get("momentum_consistency", 0)
    if pct >= 0.99 and momentum < 0:
        risk_factors.append(
            f"Negative momentum ({momentum:.0f}/6) at "
            f"all-time highs — reversal risk elevated"
        )
        risk_score += 3

    # FACTOR 3: Recent large move (5-day range)
    high_5d = features.get("high_5d", 0)
    low_5d = features.get("low_5d", 0)
    current = features.get("current_rate", 0)
    if high_5d and low_5d and high_5d > low_5d:
        recent_5d_range = high_5d - low_5d
        if recent_5d_range > 1.50:
            risk_factors.append(
                f"5-day rate range is {recent_5d_range:.2f} INR "
                f"— regime shock in progress"
            )
            risk_score += 4
        elif recent_5d_range > 1.00:
            risk_factors.append(
                f"5-day rate range is {recent_5d_range:.2f} INR "
                f"— elevated directional pressure"
            )
            risk_score += 2

    # FACTOR 4: Volatility spike vs recent average
    vol_5d = features.get("volatility_5d", 0)
    vol_20d = features.get("volatility_20d", 0)
    if vol_20d > 0 and vol_5d > 0:
        vol_ratio = vol_5d / vol_20d
        if vol_ratio > 2.5:
            risk_factors.append(
                f"5-day vol is {vol_ratio:.1f}x 20-day vol "
                f"— volatility regime expanding rapidly"
            )
            risk_score += 3
        elif vol_ratio > 1.8:
            risk_factors.append(
                f"5-day vol is {vol_ratio:.1f}x 20-day vol "
                f"— volatility picking up"
            )
            risk_score += 1

    # FACTOR 5: Regime just changed
    current_regime = features.get("current_regime", "")
    prev_regime = features.get("prev_regime", "")
    if current_regime and prev_regime:
        if current_regime != prev_regime:
            risk_factors.append(
                f"Regime changed: {prev_regime} → "
                f"{current_regime} — model recalibrating"
            )
            risk_score += 2

    # FACTOR 6: FOMC or RBI event day
    is_fomc = features.get("is_fomc_day", False)
    is_rbi = features.get("is_rbi_day", False)
    if is_fomc:
        risk_factors.append(
            "FOMC decision day — dollar volatility likely "
            "to spike in either direction"
        )
        risk_score += 2
    if is_rbi:
        risk_factors.append(
            "RBI MPC decision day — INR volatility likely"
        )
        risk_score += 2

    # FACTOR 7: Monday after large Friday move
    day_of_week = features.get("day_of_week", -1)
    if day_of_week == 0:  # Monday
        prev_close = features.get("prev_friday_close", 0)
        if prev_close and current:
            gap_paise = abs(current - prev_close) * 100
            if gap_paise > 30:
                risk_factors.append(
                    f"Monday gap: {gap_paise:.0f} paise "
                    f"from Friday close — gap-fill risk"
                )
                risk_score += 2

    # FACTOR 8: Rate vs 30d MA divergence
    ma_30 = features.get("ma_30", 0)
    if ma_30 and current:
        divergence_pct = abs(current - ma_30) / ma_30 * 100
        if divergence_pct > 1.5:
            risk_factors.append(
                f"Rate is {divergence_pct:.1f}% from 30d MA "
                f"— mean reversion force building"
            )
            risk_score += 1

    # Determine risk level
    if risk_score >= 8:
        risk_level = "CRITICAL"
        treasury_action = (
            "Add 40-50 paise manual buffer above range_high. "
            "Consider splitting conversion across 2 days. "
            "Do not rely solely on model range today."
        )
        reliability = "VERY LOW"
    elif risk_score >= 5:
        risk_level = "HIGH"
        treasury_action = (
            "Add 20-30 paise manual buffer above range_high. "
            "Verify with market desk before large conversions."
        )
        reliability = "LOW"
    elif risk_score >= 3:
        risk_level = "ELEVATED"
        treasury_action = (
            "Add 10-15 paise buffer above range_high. "
            "Monitor intraday — be ready to adjust."
        )
        reliability = "MODERATE"
    elif risk_score >= 1:
        risk_level = "MILD"
        treasury_action = (
            "Standard range_high prefunding is sufficient. "
            "Note the risk factor above."
        )
        reliability = "NORMAL"
    else:
        risk_level = "NORMAL"
        treasury_action = (
            "Prefund at range_high. No elevated risk detected."
        )
        reliability = "NORMAL"

    # TPV impact of the risk buffer
    buffer_paise = {
        "CRITICAL": 45,
        "HIGH": 25,
        "ELEVATED": 12,
        "MILD": 5,
        "NORMAL": 0,
    }[risk_level]

    tpv_daily_usd = 9_500_000
    tpv_buffer_inr = buffer_paise / 100 * tpv_daily_usd

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "prediction_reliability": reliability,
        "treasury_action": treasury_action,
        "recommended_buffer_paise": buffer_paise,
        "tpv_buffer_inr": tpv_buffer_inr,
        "tpv_buffer_label": (
            f"\u20b9{tpv_buffer_inr / 100000:.1f} lakh buffer "
            f"on $9.5M volume"
            if tpv_buffer_inr > 0
            else "No additional buffer needed"
        ),
    }
