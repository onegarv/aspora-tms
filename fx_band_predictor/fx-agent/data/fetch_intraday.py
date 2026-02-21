"""
Intraday 4-Hour Data Fetcher for USD/INR

Fetches 1-hour data from Yahoo Finance (free, max 2 years)
and resamples to 4-hour bars with session labels.

Trading sessions (IST = UTC + 5:30):
  Asian/RBI:  UTC 3:30-10:00  (IST 9:00-15:30)
  London:     UTC 8:00-16:00  (IST 13:30-21:30)
  New York:   UTC 13:30-20:00 (IST 19:00-01:30)
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "intraday_4h.csv")


def fetch_4h_data(years_back=2):
    """
    Fetch 4-hour USD/INR bars.
    yfinance provides 1-hour data for up to ~2 years free.
    We resample to 4-hour bars.
    """
    end = datetime.now()
    start = end - timedelta(days=years_back * 365)

    print(f"Fetching 1-hour USDINR data from {start.date()} to {end.date()}...")

    # yfinance 1-hour interval (max ~730 days)
    df = yf.download(
        "USDINR=X",
        start=start,
        end=end,
        interval="1h",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError("No intraday data returned from yfinance")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"  Raw 1-hour bars: {len(df)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Resample 1h → 4h bars
    df_4h = df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    # Add time-of-day features
    df_4h["hour"] = df_4h.index.hour
    df_4h["day_of_week"] = df_4h.index.dayofweek

    # Tag trading sessions (IST = UTC + 5:30)
    # Asian/RBI session: UTC 3:30-10:00 (IST 9-15:30)
    # London session: UTC 8:00-16:00 (IST 13:30-21:30)
    # NY session: UTC 13:30-20:00 (IST 19:00-01:30)
    df_4h["session"] = "off"
    df_4h.loc[df_4h["hour"].between(4, 8), "session"] = "asian"
    df_4h.loc[df_4h["hour"].between(8, 16), "session"] = "london"
    df_4h.loc[df_4h["hour"].between(12, 20), "session"] = "newyork"

    # Encode session as number
    session_map = {"off": 0, "asian": 1, "london": 2, "newyork": 3}
    df_4h["session_num"] = df_4h["session"].map(session_map)

    # Print summary
    print(f"\n  4-hour bars: {len(df_4h)}")
    print(f"  Date range: {df_4h.index[0]} to {df_4h.index[-1]}")
    print(f"  Rate range: {df_4h['Close'].min():.4f} to {df_4h['Close'].max():.4f}")
    print(f"  Rate mean:  {df_4h['Close'].mean():.4f}")
    print(f"\n  Session distribution:")
    for s, count in df_4h["session"].value_counts().sort_index().items():
        print(f"    {s:<10s}: {count:>5d} bars ({count / len(df_4h) * 100:.1f}%)")
    print(f"\n  Day-of-week distribution:")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for dow, count in df_4h["day_of_week"].value_counts().sort_index().items():
        print(f"    {day_names[dow]:<4s}: {count:>5d} bars")
    print(f"\n  Null counts:\n{df_4h.isnull().sum()}")

    # Save
    df_4h.to_csv(OUTPUT_FILE)
    print(f"\n  Saved to {OUTPUT_FILE}")

    return df_4h


if __name__ == "__main__":
    fetch_4h_data()
