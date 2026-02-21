"""
Phase 2 — Feature Engineering
Takes the clean market dataframe from Phase 1 and produces a full feature
matrix for the XGBoost classifier and LSTM range predictor.

Features:
  - Rate momentum  (12 features)
  - Macro          (10 features)
  - FRED macro     (4 features) — yield_curve_spread, fed_funds_level, fed_funds_change_3m, cpi_yoy
  - Calendar       (4 features) — day_of_week, days_to_month_end, is_month_end, is_month_start
  - Event calendar (4 features)
  - Regime         (13 features) — trend, regime classification, historical context, momentum
  - Long-term      (4 features) — multi-year context for full-history training

Total: 51 engineered features + date + usdinr (raw) + target columns.
"""

import os
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hardcoded event calendars (2025-2026)
# ---------------------------------------------------------------------------

# RBI Monetary Policy Committee meeting dates (announcement day)
RBI_MEETING_DATES = [
    date(2025, 2, 7),
    date(2025, 4, 9),
    date(2025, 6, 6),
    date(2025, 8, 8),
    date(2025, 10, 8),
    date(2025, 12, 5),
    date(2026, 2, 6),
    date(2026, 4, 8),
    date(2026, 6, 5),
    date(2026, 8, 7),
    date(2026, 10, 7),
    date(2026, 12, 4),
]

# US Federal Reserve FOMC meeting dates (announcement day)
FED_MEETING_DATES = [
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 17),
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 16),
]


# ---------------------------------------------------------------------------
# RSI — Wilder's Smoothing (from scratch)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's exponential smoothing.

    Wilder's smoothing uses alpha = 1/period (equivalent to EMA with
    span = 2*period - 1). The first average is a simple mean of the
    first `period` values, then each subsequent value is:
        avg = prev_avg * (period - 1)/period + current_value / period
    """
    delta = series.diff()

    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # First average: simple mean over the first `period` changes
    avg_gain = pd.Series(np.nan, index=series.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=series.index, dtype=float)

    # We need at least `period + 1` data points (period changes)
    first_idx = period  # index position (0-based) where first average lands

    avg_gain.iloc[first_idx] = gains.iloc[1 : first_idx + 1].mean()
    avg_loss.iloc[first_idx] = losses.iloc[1 : first_idx + 1].mean()

    # Wilder's smoothing for the rest
    for i in range(first_idx + 1, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gains.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + losses.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def compute_macd_histogram(series: pd.Series,
                           fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram = MACD line − signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


# ---------------------------------------------------------------------------
# Bollinger Band position
# ---------------------------------------------------------------------------

def compute_bb_position(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Position within Bollinger Bands.
    0 = at lower band, 0.5 = at middle (SMA), 1 = at upper band.
    Can exceed [0, 1] if price is outside the bands.
    """
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = upper - lower
    # Avoid division by zero
    position = (series - lower) / band_width.replace(0, np.nan)
    return position


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def _days_to_month_end(dt: datetime) -> int:
    """Days remaining until last day of the month."""
    if dt.month == 12:
        last_day = date(dt.year + 1, 1, 1)
    else:
        last_day = date(dt.year, dt.month + 1, 1)
    return (last_day - dt.date()).days if hasattr(dt, 'date') else (last_day - dt).days


def _days_to_next_event(current_date, event_dates: list) -> int:
    """
    Days until the next event. If current_date is past all known events,
    return 999 (far future — no upcoming event known).
    """
    if hasattr(current_date, 'date'):
        current_date = current_date.date()

    future = [d for d in event_dates if d >= current_date]
    if future:
        return (future[0] - current_date).days
    return 999  # no upcoming event in calendar


# ---------------------------------------------------------------------------
# Regime detection features
# ---------------------------------------------------------------------------

def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the slope of a linear regression over a rolling window.
    Normalized by the mean rate in the window so it's scale-independent.
    Positive = trending up, negative = trending down.
    """
    def _slope(vals):
        if len(vals) < window or np.isnan(vals).any():
            return np.nan
        x = np.arange(len(vals))
        # Normalize: slope per day as fraction of mean
        mean_val = vals.mean()
        if mean_val == 0:
            return 0.0
        slope = np.polyfit(x, vals, 1)[0]
        return slope / mean_val

    return series.rolling(window=window).apply(_slope, raw=True)


def add_regime_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime-aware features to the dataframe.
    Assumes rate momentum features (rate_vs_30d_avg, volatility_5d) already exist.

    Adds 13 features:
      TREND (5): rate_trend_30d, rate_trend_90d, is_trending_up, is_trending_down, trend_strength
      REGIME (4): regime_trending_up, regime_trending_down, regime_high_vol, regime_range_bound
      HISTORICAL CONTEXT (3): rate_vs_52w_high, rate_vs_52w_low, rate_percentile_1y
      MOMENTUM (1): momentum_consistency
    """
    rate = out["usdinr"]

    # ==================================================================
    # TREND FEATURES
    # ==================================================================
    # Slope is normalized (daily change / mean), typical range: -0.002 to +0.002
    # Threshold 0.0004 ≈ 75th percentile of |slope| → captures real trends
    TREND_THRESHOLD = 0.0004
    out["rate_trend_30d"] = _rolling_slope(rate, 30)
    out["rate_trend_90d"] = _rolling_slope(rate, 90)
    out["is_trending_up"] = (out["rate_trend_30d"] > TREND_THRESHOLD).astype(int)
    out["is_trending_down"] = (out["rate_trend_30d"] < -TREND_THRESHOLD).astype(int)
    out["trend_strength"] = out["rate_trend_30d"].abs()

    # ==================================================================
    # REGIME CLASSIFICATION
    # ==================================================================
    out["regime_trending_up"] = (
        (out["rate_trend_30d"] > TREND_THRESHOLD) & (out["rate_vs_30d_avg"] > 0.002)
    ).astype(int)
    out["regime_trending_down"] = (
        (out["rate_trend_30d"] < -TREND_THRESHOLD) & (out["rate_vs_30d_avg"] < -0.002)
    ).astype(int)
    out["regime_high_vol"] = (out["volatility_5d"] > 0.006).astype(int)
    out["regime_range_bound"] = (
        (out["regime_trending_up"] == 0) &
        (out["regime_trending_down"] == 0) &
        (out["regime_high_vol"] == 0)
    ).astype(int)

    # ==================================================================
    # HISTORICAL RANGE CONTEXT
    # ==================================================================
    out["rate_vs_52w_high"] = rate / rate.rolling(252).max() - 1.0
    out["rate_vs_52w_low"] = rate / rate.rolling(252).min() - 1.0

    def _percentile_rank(vals):
        if len(vals) < 252 or np.isnan(vals).any():
            return np.nan
        return (vals[-1] > vals[:-1]).sum() / (len(vals) - 1)

    out["rate_percentile_1y"] = rate.rolling(252).apply(_percentile_rank, raw=True)

    # ==================================================================
    # MOMENTUM REGIME
    # ==================================================================
    # momentum_7d: 7-day rate change
    out["momentum_7d"] = rate.pct_change(7)

    # momentum_consistency: how many of last 5 days moved in same direction
    # +5 = all up, -5 = all down, 0 = mixed
    daily_dir = rate.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    out["momentum_consistency"] = daily_dir.rolling(5).sum()

    return out


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the clean market dataframe (date, usdinr, oil, dxy, vix, us10y)
    and returns a feature matrix with all engineered features.

    The returned dataframe retains the 'date' and 'usdinr' columns for
    reference, plus all feature columns. Rows with incomplete features
    (leading warm-up period) are dropped — output has zero NaNs.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    rate = out["usdinr"]
    oil = out["oil"]
    dxy = out["dxy"]
    vix = out["vix"]
    us10y = out["us10y"]

    # Daily returns (pct change)
    rate_ret = rate.pct_change()

    # ==================================================================
    # RATE MOMENTUM FEATURES
    # ==================================================================
    out["rate_change_1d"] = rate.pct_change(1)
    out["rate_change_3d"] = rate.pct_change(3)
    out["rate_change_5d"] = rate.pct_change(5)

    out["rate_vs_7d_avg"] = (rate / rate.rolling(7).mean()) - 1.0
    out["rate_vs_30d_avg"] = (rate / rate.rolling(30).mean()) - 1.0

    out["rsi_14"] = compute_rsi(rate, period=14)

    out["macd_histogram"] = compute_macd_histogram(rate)

    out["bb_position"] = compute_bb_position(rate)

    out["rate_acceleration"] = out["rate_change_1d"].diff()

    out["volatility_5d"] = rate_ret.rolling(5).std()
    out["volatility_20d"] = rate_ret.rolling(20).std()

    out["high_vol_regime"] = (out["volatility_5d"] > out["volatility_20d"]).astype(int)

    # ==================================================================
    # MACRO FEATURES
    # ==================================================================
    out["oil_change_1d"] = oil.pct_change(1)
    out["oil_vs_30d_avg"] = (oil / oil.rolling(30).mean()) - 1.0

    out["dxy_change_1d"] = dxy.pct_change(1)
    out["dxy_vs_7d_avg"] = (dxy / dxy.rolling(7).mean()) - 1.0

    out["vix_level"] = vix
    out["vix_change_1d"] = vix.pct_change(1)

    out["us10y_change_1d"] = us10y.pct_change(1)
    out["us10y_level"] = us10y

    out["oil_dxy_divergence"] = out["oil_change_1d"] - out["dxy_change_1d"]

    out["yield_spread_proxy"] = us10y - out["oil_vs_30d_avg"]

    # ==================================================================
    # FRED MACRO FEATURES (graceful skip if columns not present)
    # ==================================================================
    if "us_2y" in out.columns:
        # Yield curve spread: 10Y minus 2Y — classic recession/risk indicator
        # Positive = normal curve, negative = inverted (recession signal)
        out["yield_curve_spread"] = us10y - out["us_2y"]
    if "fed_funds" in out.columns:
        out["fed_funds_level"] = out["fed_funds"]
        # 3-month change in fed funds rate (~63 trading days)
        out["fed_funds_change_3m"] = out["fed_funds"].diff(63)
    if "cpi" in out.columns:
        # Year-over-year CPI change (inflation rate) — ~252 trading days
        out["cpi_yoy"] = out["cpi"].pct_change(252)

    # ==================================================================
    # CALENDAR FEATURES
    # ==================================================================
    out["day_of_week"] = out["date"].dt.dayofweek           # 0=Mon … 6=Sun

    out["days_to_month_end"] = out["date"].apply(_days_to_month_end)
    out["is_month_end"] = (out["days_to_month_end"] <= 2).astype(int)
    out["is_month_start"] = (out["date"].dt.day <= 3).astype(int)

    # ==================================================================
    # EVENT CALENDAR FEATURES
    # ==================================================================
    out["days_to_next_rbi"] = out["date"].apply(
        lambda d: _days_to_next_event(d, RBI_MEETING_DATES)
    )
    out["days_to_next_fed"] = out["date"].apply(
        lambda d: _days_to_next_event(d, FED_MEETING_DATES)
    )
    out["is_rbi_week"] = (out["days_to_next_rbi"] <= 3).astype(int)
    out["is_fed_week"] = (out["days_to_next_fed"] <= 3).astype(int)

    # ==================================================================
    # REGIME FEATURES (trend, regime classification, historical context)
    # ==================================================================
    out = add_regime_features(out)

    # ==================================================================
    # LONG-TERM FEATURES (multi-year context for full-history training)
    # ==================================================================

    # 1. rate_vs_5y_avg: how far current rate is from 5-year moving average
    #    Backfill early rows (< 1260 days) with expanding mean
    ma_5y = rate.rolling(1260).mean()
    ma_5y_filled = ma_5y.fillna(rate.expanding().mean())
    out["rate_vs_5y_avg"] = rate / ma_5y_filled - 1.0

    # 2. rate_vs_alltime_percentile: walk-forward percentile rank (no lookahead)
    out["rate_vs_alltime_percentile"] = rate.expanding().rank(pct=True)

    # 3. long_term_trend_1y: raw slope of linear regression over 252 trading days
    #    Units: INR per day (not normalized)
    def _raw_slope_252(vals):
        if len(vals) < 252 or np.isnan(vals).any():
            return np.nan
        x = np.arange(len(vals))
        return np.polyfit(x, vals, 1)[0]

    out["long_term_trend_1y"] = rate.rolling(252).apply(_raw_slope_252, raw=True)

    # 4. is_decade_high: 1 if rate is within 2% of decade (2520-day) max
    #    Use expanding max for first 2520 rows to avoid NaN
    rolling_max_10y = rate.rolling(2520).max()
    expanding_max = rate.expanding().max()
    decade_max = rolling_max_10y.fillna(expanding_max)
    out["is_decade_high"] = (rate >= decade_max * 0.98).astype(int)

    # ==================================================================
    # DROP WARM-UP ROWS (leading NaNs from rolling windows)
    # ==================================================================
    raw_cols = {"date", "usdinr", "oil", "dxy", "vix", "us10y", "us_2y", "fed_funds", "cpi"}
    feature_cols = [c for c in out.columns if c not in raw_cols]
    out = out.dropna(subset=feature_cols).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Helper: latest row as dict (for live prediction agent)
# ---------------------------------------------------------------------------

def get_latest_features(df: pd.DataFrame) -> dict:
    """
    Run build_features() and return the last row as a flat dictionary.
    Used by the live prediction agent to get current feature values.
    """
    features = build_features(df)
    last = features.iloc[-1]
    return last.to_dict()


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_feature_summary(df: pd.DataFrame) -> None:
    """Print a formatted table of feature statistics."""
    raw_cols = {"date", "usdinr", "oil", "dxy", "vix", "us10y", "us_2y", "fed_funds", "cpi"}
    feature_cols = [c for c in df.columns if c not in raw_cols]

    print(f"\n{'Feature':<25s} {'Min':>12s} {'Max':>12s} {'Mean':>12s} {'Nulls':>6s}")
    print("-" * 70)
    for col in feature_cols:
        mn = df[col].min()
        mx = df[col].max()
        avg = df[col].mean()
        nulls = df[col].isnull().sum()
        print(f"{col:<25s} {mn:>12.6f} {mx:>12.6f} {avg:>12.6f} {nulls:>6d}")
    print("-" * 70)
    print(f"Total features: {len(feature_cols)}")
    print(f"Total rows:     {len(df)}")
    total_nulls = df[feature_cols].isnull().sum().sum()
    print(f"Total nulls:    {total_nulls}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "feature_matrix.csv")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "market_data.csv")

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run data/fetch_market_data.py first.")
        sys.exit(1)

    print(f"Loading market data from {data_path}")
    raw = pd.read_csv(data_path, parse_dates=["date"])
    print(f"Raw data shape: {raw.shape}")

    print("\nBuilding features...")
    features = build_features(raw)

    print_feature_summary(features)

    # Assert zero nulls
    feature_cols = [c for c in features.columns if c not in ("date", "usdinr", "oil", "dxy", "vix", "us10y")]
    total_nulls = features[feature_cols].isnull().sum().sum()
    assert total_nulls == 0, f"Expected 0 nulls in features, found {total_nulls}"
    print("\nASSERTION PASSED: zero nulls in feature matrix")

    # Save
    features.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved feature matrix to {OUTPUT_FILE} ({len(features)} rows, {len(features.columns)} cols)")

    # Show latest features (what the live agent would see)
    print("\n--- Latest feature row (for live prediction) ---")
    latest = get_latest_features(raw)
    for k, v in latest.items():
        if k == "date":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n--- Phase 2 complete ---")
