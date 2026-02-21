"""
Intraday 4-Hour Feature Engineering for USD/INR

Builds 20 features from 4-hour OHLCV bars for the intraday LSTM.
Cleans Sunday artifact bars before feature computation.

Features (20):
  ret_4h, ret_8h, ret_24h, ret_48h          — multi-horizon returns
  bar_range, range_ma_5, is_high_vol_bar     — volatility
  last_4h_direction                          — sign of last return
  momentum_consistency_24h                   — sum of sign(ret_4h) over 6 bars
  rate_vs_24h_high, rate_vs_24h_low          — position within 24h range
  rate_vs_48h_avg                            — deviation from 48h mean
  rsi_4h                                     — 14-period RSI on 4h bars
  is_asian, is_london, is_ny                 — session dummies
  hour, day_of_week, session_num             — time features
  Close                                      — current rate
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, "intraday_4h.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "intraday_features.csv")


def build_intraday_features(df_4h: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build 20 intraday features from 4-hour OHLCV bars.

    Args:
        df_4h: DataFrame with OHLCV + session columns.
               If None, reads from intraday_4h.csv.

    Returns:
        DataFrame with 20 features, warmup rows dropped, zero nulls.
    """
    # --- Load data if not provided ---
    if df_4h is None:
        df_4h = pd.read_csv(INPUT_FILE, index_col="Datetime", parse_dates=True)

    # --- Step 0: Remove Sunday bars (market artifacts) ---
    n_before = len(df_4h)
    df_4h = df_4h[df_4h.index.dayofweek != 6].copy()
    n_removed = n_before - len(df_4h)
    print(f"  [cleanup] Removed {n_removed} Sunday bars ({n_before} → {len(df_4h)})")

    # --- Feature engineering ---
    df = df_4h.copy()

    # 1-4. Multi-horizon returns
    df["ret_4h"] = df["Close"].pct_change(1)       # 1 bar  = 4 hours
    df["ret_8h"] = df["Close"].pct_change(2)       # 2 bars = 8 hours
    df["ret_24h"] = df["Close"].pct_change(6)      # 6 bars = 24 hours
    df["ret_48h"] = df["Close"].pct_change(12)     # 12 bars = 48 hours

    # 5-7. Volatility features
    df["bar_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["range_ma_5"] = df["bar_range"].rolling(5).mean()
    df["is_high_vol_bar"] = (df["bar_range"] > df["range_ma_5"] * 1.5).astype(int)

    # 8. Last 4h direction
    df["last_4h_direction"] = np.sign(df["ret_4h"])

    # 9. Momentum consistency over 24h (sum of sign(ret_4h) over last 6 bars)
    #    Range: -6 (all down) to +6 (all up)
    df["momentum_consistency_24h"] = (
        np.sign(df["ret_4h"]).rolling(6).sum()
    )

    # 10-11. Position within 24h range
    rolling_high_24h = df["High"].rolling(6).max()
    rolling_low_24h = df["Low"].rolling(6).min()
    df["rate_vs_24h_high"] = df["Close"] / rolling_high_24h - 1
    df["rate_vs_24h_low"] = df["Close"] / rolling_low_24h - 1

    # 12. Deviation from 48h average
    df["rate_vs_48h_avg"] = df["Close"] / df["Close"].rolling(12).mean() - 1

    # 13. RSI (14-period on 4h bars)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_4h"] = 100 - (100 / (1 + rs))

    # 14-16. Session dummies
    df["is_asian"] = (df["session"] == "asian").astype(int)
    df["is_london"] = (df["session"] == "london").astype(int)
    df["is_ny"] = (df["session"] == "newyork").astype(int)

    # 17-19. Time features (already in df_4h, but ensure they exist)
    if "hour" not in df.columns:
        df["hour"] = df.index.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df.index.dayofweek
    if "session_num" not in df.columns:
        session_map = {"off": 0, "asian": 1, "london": 2, "newyork": 3}
        df["session_num"] = df["session"].map(session_map)

    # 20. Close is already present

    # --- Select final 20 features ---
    feature_cols = [
        "ret_4h", "ret_8h", "ret_24h", "ret_48h",
        "bar_range", "range_ma_5", "is_high_vol_bar",
        "last_4h_direction", "momentum_consistency_24h",
        "rate_vs_24h_high", "rate_vs_24h_low", "rate_vs_48h_avg",
        "rsi_4h",
        "is_asian", "is_london", "is_ny",
        "hour", "day_of_week", "session_num",
        "Close",
    ]

    # Keep session column for diagnostics (will be in output but not counted as feature)
    df_out = df[feature_cols + ["session"]].copy()

    # --- Drop warmup rows (need 14 bars for RSI, 12 for ret_48h) ---
    n_before_warmup = len(df_out)
    df_out = df_out.dropna(subset=feature_cols)
    n_dropped = n_before_warmup - len(df_out)
    print(f"  [warmup] Dropped {n_dropped} warmup rows ({n_before_warmup} → {len(df_out)})")

    return df_out, feature_cols


def print_diagnostics(df: pd.DataFrame, feature_cols: list) -> None:
    """Print the diagnostic table and summary the user requested."""

    bar = "─" * 80

    # --- Diagnostic table: last 6 bars ---
    print()
    print("FEATURE SANITY CHECK (last 6 bars = last 24 hours)")
    print(bar)
    last6 = df.tail(6)
    diag_cols = ["session", "Close", "ret_4h", "rsi_4h", "momentum_consistency_24h"]
    print(f"{'Timestamp':<26s} | {'Session':<9s} | {'Close':>8s} | {'ret_4h':>8s} | {'rsi_4h':>6s} | {'mom_24h':>7s}")
    print(bar)
    for ts, row in last6.iterrows():
        ts_str = ts.strftime("%Y-%m-%d %H:%M")
        print(
            f"{ts_str:<26s} | {row['session']:<9s} | {row['Close']:>8.4f} | "
            f"{row['ret_4h']:>+8.5f} | {row['rsi_4h']:>6.1f} | "
            f"{row['momentum_consistency_24h']:>+7.1f}"
        )
    print(bar)

    # --- Summary stats ---
    print()
    print(f"Total feature count:    {len(feature_cols)}")
    print(f"Rows after warmup:      {len(df)}")
    null_count = df[feature_cols].isnull().sum().sum()
    print(f"Null values:            {null_count}")
    if null_count > 0:
        print("  WARNING: Nulls detected!")
        print(df[feature_cols].isnull().sum()[df[feature_cols].isnull().sum() > 0])
    else:
        print("  ✓ Zero nulls — clean dataset")

    # --- Momentum consistency distribution ---
    print()
    print("momentum_consistency_24h distribution:")
    mom = df["momentum_consistency_24h"]
    counts = mom.value_counts().sort_index()
    for val, cnt in counts.items():
        pct = cnt / len(df) * 100
        bar_chart = "█" * int(pct)
        print(f"  {val:>+4.0f}: {cnt:>5d} ({pct:>5.1f}%) {bar_chart}")

    # --- Additional sanity checks ---
    print()
    print("Feature ranges:")
    for col in ["ret_4h", "ret_24h", "bar_range", "rsi_4h"]:
        print(f"  {col:<25s}: min={df[col].min():>+10.6f}  max={df[col].max():>+10.6f}  mean={df[col].mean():>+10.6f}")
    print()


if __name__ == "__main__":
    df, feature_cols = build_intraday_features()
    print_diagnostics(df, feature_cols)

    # Save
    df.to_csv(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
