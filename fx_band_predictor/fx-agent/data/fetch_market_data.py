"""
Phase 1 — Market Data Fetcher
Pulls 3 years of historical data from Yahoo Finance:
  - USD/INR exchange rate
  - Brent Crude Oil
  - DXY Dollar Index
  - VIX Fear Index
  - US 10Y Treasury Yield

Plus FRED economic series (when FRED_API_KEY is set):
  - DGS2: US 2-Year Treasury Yield (daily)
  - FEDFUNDS: Fed Funds Effective Rate (monthly → forward-filled to daily)
  - CPIAUCSL: US CPI All Urban (monthly → forward-filled to daily)

Merges into one clean daily dataframe, forward-fills weekends/holidays,
and saves as data/market_data.csv.
"""

import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = {
    "usdinr": "USDINR=X",
    "oil": "BZ=F",        # Brent Crude futures
    "dxy": "DX-Y.NYB",    # US Dollar Index
    "vix": "^VIX",        # CBOE Volatility Index
    "us10y": "^TNX",      # US 10-Year Treasury Yield
}

# 3 years of history for training + buffer for feature warm-up
HISTORY_YEARS = 3
BUFFER_DAYS = 60  # extra days so rolling windows don't create leading NaNs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "market_data.csv")
EXTENDED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "market_data_extended.csv")
FULL_HISTORY_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "market_data_full.csv")

# Extended data: 10+ years covering multiple USD/INR regimes
EXTENDED_START_DATE = "2015-01-01"
FALLBACK_START_DATE = "2018-01-01"  # fallback if 2015 data unavailable

# Full history: max available from 2003
FULL_HISTORY_START = "2003-01-01"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_dates(years: int = HISTORY_YEARS, buffer_days: int = BUFFER_DAYS):
    """Return (start_date, end_date) strings for yfinance download."""
    end = datetime.today()
    start = end - timedelta(days=365 * years + buffer_days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _download_ticker(ticker_symbol: str, start: str, end: str) -> pd.Series:
    """
    Download adjusted close for a single ticker.
    Returns a Series indexed by date.
    """
    print(f"  Downloading {ticker_symbol} ...")
    df = yf.download(ticker_symbol, start=start, end=end, progress=False)

    if df.empty:
        print(f"  WARNING: No data returned for {ticker_symbol}")
        return pd.Series(dtype=float)

    # yfinance may return MultiIndex columns when downloading single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Use 'Close' (adjusted close is default in modern yfinance)
    if "Close" in df.columns:
        series = df["Close"].copy()
    elif "Adj Close" in df.columns:
        series = df["Adj Close"].copy()
    else:
        raise KeyError(f"No 'Close' or 'Adj Close' column found for {ticker_symbol}. "
                       f"Columns: {df.columns.tolist()}")

    series.index = pd.to_datetime(series.index)
    series.index = series.index.tz_localize(None)  # remove timezone if present
    return series


# ---------------------------------------------------------------------------
# FRED economic data
# ---------------------------------------------------------------------------

FRED_SERIES = {
    "us_2y": "DGS2",        # 2-Year Treasury Yield (daily)
    "fed_funds": "FEDFUNDS", # Fed Funds Effective Rate (monthly)
    "cpi": "CPIAUCSL",       # CPI All Urban Consumers (monthly)
}

FRED_CACHE_FILE = os.path.join(OUTPUT_DIR, "fred_cache.json")


def _get_fred_cache(start: str, end: str) -> dict[str, pd.Series] | None:
    """Return cached FRED data if today's cache exists and covers the requested range."""
    if not os.path.exists(FRED_CACHE_FILE):
        return None
    try:
        with open(FRED_CACHE_FILE, "r") as f:
            cache = json.load(f)
        today = datetime.now().strftime("%Y-%m-%d")
        if cache.get("date") != today:
            print(f"  [fred] Cache expired ({cache.get('date')}) — fetching fresh")
            return None
        if cache.get("start") != start or cache.get("end") != end:
            print(f"  [fred] Cache range mismatch — fetching fresh")
            return None
        result = {}
        for col_name, records in cache.get("series", {}).items():
            s = pd.Series(records["values"], index=pd.to_datetime(records["dates"]))
            result[col_name] = s
            print(f"  [fred] Cache hit — {col_name}: {len(s)} obs")
        return result
    except Exception as e:
        print(f"  [fred] Cache read error: {e}")
        return None


def _save_fred_cache(data: dict[str, pd.Series], start: str, end: str) -> None:
    """Save FRED data to daily cache."""
    series_out = {}
    for col_name, s in data.items():
        series_out[col_name] = {
            "dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "values": s.tolist(),
        }
    cache = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "start": start,
        "end": end,
        "cached_at": datetime.now().isoformat(),
        "series": series_out,
    }
    try:
        with open(FRED_CACHE_FILE, "w") as f:
            json.dump(cache, f)
        print(f"  [fred] Cached {len(data)} series for today")
    except Exception as e:
        print(f"  [fred] Cache write error: {e}")


def _fetch_fred_series(start: str, end: str) -> dict[str, pd.Series]:
    """
    Fetch FRED economic series with daily file-based caching.
    Returns dict of {column_name: pd.Series}.
    Returns empty dict if FRED_API_KEY is not set — never raises.
    """
    # Check cache first
    cached = _get_fred_cache(start, end)
    if cached is not None:
        return cached

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        print("  [fred] WARNING: FRED_API_KEY not set — skipping FRED data")
        return {}

    try:
        from fredapi import Fred
    except ImportError:
        print("  [fred] WARNING: fredapi not installed — run: pip install fredapi")
        return {}

    fred = Fred(api_key=api_key)
    result = {}

    for col_name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s = s.dropna()
            if s.empty:
                print(f"  [fred] WARNING: {series_id} returned no data")
                continue
            s.index = pd.to_datetime(s.index)
            s.index = s.index.tz_localize(None) if s.index.tz else s.index
            result[col_name] = s
            print(f"  [fred] {series_id} → {col_name}: {len(s)} obs, "
                  f"{s.index[0].date()} → {s.index[-1].date()}")
        except Exception as e:
            print(f"  [fred] WARNING: Failed to fetch {series_id}: {e}")

    # Cache if we got data
    if result:
        _save_fred_cache(result, start, end)

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fetch_all(years: int = HISTORY_YEARS, buffer_days: int = BUFFER_DAYS) -> pd.DataFrame:
    """
    Download all tickers, merge into a single daily dataframe.

    Returns a DataFrame with columns:
        date, usdinr, oil, dxy, vix, us10y
    Sorted by date ascending, forward-filled for weekends/holidays,
    with no NaN rows in the usdinr column.
    """
    start, end = _compute_dates(years, buffer_days)
    print(f"Fetching market data from {start} to {end}\n")

    series_dict = {}
    for name, symbol in TICKERS.items():
        s = _download_ticker(symbol, start, end)
        if not s.empty:
            series_dict[name] = s

    if "usdinr" not in series_dict or series_dict["usdinr"].empty:
        raise RuntimeError("Failed to download USD/INR data — cannot proceed.")

    # Fetch FRED series and add to series_dict
    fred_data = _fetch_fred_series(start, end)
    series_dict.update(fred_data)

    # Merge all series on date
    merged = pd.DataFrame(series_dict)
    merged.index.name = "date"

    # Create a continuous daily date range (includes weekends)
    full_range = pd.date_range(start=merged.index.min(), end=merged.index.max(), freq="D")
    merged = merged.reindex(full_range)
    merged.index.name = "date"

    # Forward-fill weekends and holidays (markets closed)
    # Also forward-fills monthly FRED series (FEDFUNDS, CPIAUCSL) to daily
    merged = merged.ffill()

    # Drop any leading rows that are still NaN (before first trading day)
    merged = merged.dropna(subset=["usdinr"])

    # Sort chronologically
    merged = merged.sort_index()

    # Reset index so 'date' is a column
    merged = merged.reset_index()

    print(f"\nDataset shape: {merged.shape}")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"Columns: {merged.columns.tolist()}")
    print(f"\nNull counts:\n{merged.isnull().sum()}")
    print(f"\nSample (last 5 rows):\n{merged.tail()}")

    return merged


def fetch_extended(start_date: str = EXTENDED_START_DATE) -> pd.DataFrame:
    """
    Download extended historical data from a fixed start date.
    Falls back to FALLBACK_START_DATE if primary start fails.

    Returns a DataFrame with columns:
        date, usdinr, oil, dxy, vix, us10y
    """
    end = datetime.today().strftime("%Y-%m-%d")
    print(f"Fetching EXTENDED market data from {start_date} to {end}\n")

    series_dict = {}
    for name, symbol in TICKERS.items():
        s = _download_ticker(symbol, start_date, end)
        if not s.empty:
            series_dict[name] = s

    # If USDINR failed, try fallback start date
    if "usdinr" not in series_dict or series_dict["usdinr"].empty:
        if start_date != FALLBACK_START_DATE:
            print(f"\nWARNING: No USDINR data from {start_date}, trying {FALLBACK_START_DATE}...")
            return fetch_extended(start_date=FALLBACK_START_DATE)
        raise RuntimeError("Failed to download USD/INR data — cannot proceed.")

    # Fetch FRED series
    fred_data = _fetch_fred_series(start_date, end)
    series_dict.update(fred_data)

    # Merge all series on date
    merged = pd.DataFrame(series_dict)
    merged.index.name = "date"

    # Create a continuous daily date range (includes weekends)
    full_range = pd.date_range(start=merged.index.min(), end=merged.index.max(), freq="D")
    merged = merged.reindex(full_range)
    merged.index.name = "date"

    # Forward-fill weekends and holidays (markets closed)
    merged = merged.ffill()

    # Backfill any remaining leading NaNs (e.g. Jan 1 holiday)
    merged = merged.bfill()

    # Drop any leading rows that are still NaN (before first trading day)
    merged = merged.dropna(subset=["usdinr"])

    # Sort chronologically
    merged = merged.sort_index()

    # Reset index so 'date' is a column
    merged = merged.reset_index()

    # Summary
    print(f"\nDataset shape: {merged.shape}")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"Columns: {merged.columns.tolist()}")
    print(f"\nRate stats:")
    print(f"  Min:  {merged['usdinr'].min():.4f}")
    print(f"  Max:  {merged['usdinr'].max():.4f}")
    print(f"  Mean: {merged['usdinr'].mean():.4f}")
    print(f"\nNull counts:\n{merged.isnull().sum()}")
    total_nulls = merged.isnull().sum().sum()
    print(f"\nTotal nulls: {total_nulls}")
    print(f"\nSample (last 5 rows):\n{merged.tail()}")

    return merged


def fetch_full_history() -> pd.DataFrame:
    """
    Download maximum available history from 2003-01-01.

    Handles missing early data gracefully:
      - USDINR=X: available from ~2003
      - BZ=F (Brent): unavailable before ~2007 → use CL=F (WTI) as proxy
      - DX-Y.NYB: available from 2003
      - ^VIX: available from 2003
      - ^TNX: available from 2003
    """
    start = FULL_HISTORY_START
    end = datetime.today().strftime("%Y-%m-%d")
    print(f"Fetching FULL HISTORY market data from {start} to {end}\n")

    # --- Download primary tickers ---
    series_dict = {}
    ticker_notes = {}

    for name, symbol in TICKERS.items():
        s = _download_ticker(symbol, start, end)
        if not s.empty:
            series_dict[name] = s
            ticker_notes[name] = f"{symbol}: {len(s)} trading days, {s.index.min().date()} to {s.index.max().date()}"
        else:
            ticker_notes[name] = f"{symbol}: NO DATA"

    if "usdinr" not in series_dict or series_dict["usdinr"].empty:
        raise RuntimeError("Failed to download USD/INR data — cannot proceed.")

    # --- Handle Brent oil gap: if BZ=F starts after USDINR, backfill with WTI ---
    usdinr_start = series_dict["usdinr"].index.min()
    if "oil" in series_dict:
        oil_start = series_dict["oil"].index.min()
        if oil_start > usdinr_start + pd.Timedelta(days=30):
            print(f"\n  Brent (BZ=F) starts at {oil_start.date()}, "
                  f"USDINR starts at {usdinr_start.date()}")
            print(f"  Fetching WTI (CL=F) as proxy for pre-{oil_start.date()} oil data...")
            wti = _download_ticker("CL=F", start, end)
            if not wti.empty:
                # Use WTI for dates before Brent is available
                wti_only = wti[wti.index < oil_start]
                combined_oil = pd.concat([wti_only, series_dict["oil"]]).sort_index()
                combined_oil = combined_oil[~combined_oil.index.duplicated(keep="last")]
                series_dict["oil"] = combined_oil
                ticker_notes["oil"] += f" (WTI proxy for {len(wti_only)} days before {oil_start.date()})"
                print(f"  Combined oil series: {len(combined_oil)} days")
    else:
        # No Brent at all — use WTI entirely
        print("  No Brent data — using WTI (CL=F) as full oil proxy...")
        wti = _download_ticker("CL=F", start, end)
        if not wti.empty:
            series_dict["oil"] = wti
            ticker_notes["oil"] = f"CL=F (WTI proxy): {len(wti)} days"

    # --- Fetch FRED series ---
    fred_data = _fetch_fred_series(start, end)
    series_dict.update(fred_data)

    # --- Merge all series ---
    merged = pd.DataFrame(series_dict)
    merged.index.name = "date"

    # Create continuous daily date range
    full_range = pd.date_range(start=merged.index.min(), end=merged.index.max(), freq="D")
    merged = merged.reindex(full_range)
    merged.index.name = "date"

    # Forward-fill weekends and holidays
    # Also forward-fills monthly FRED series (FEDFUNDS, CPIAUCSL) to daily
    merged = merged.ffill()

    # Backfill any remaining leading NaNs (first few days might be holiday)
    merged = merged.bfill()

    # Drop rows where USDINR is still NaN
    merged = merged.dropna(subset=["usdinr"])

    merged = merged.sort_index().reset_index()

    # --- Check for remaining nulls and handle ---
    null_counts = merged.isnull().sum()
    total_nulls = null_counts.sum()

    # --- Print summary ---
    print(f"\nDataset shape: {merged.shape}")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"\nRate stats:")
    print(f"  Min:  {merged['usdinr'].min():.4f}")
    print(f"  Max:  {merged['usdinr'].max():.4f}")
    print(f"  Mean: {merged['usdinr'].mean():.4f}")
    print(f"  Std:  {merged['usdinr'].std():.4f}")
    print(f"\nTicker coverage:")
    for name, note in ticker_notes.items():
        print(f"  {name}: {note}")
    print(f"\nNull counts:\n{null_counts}")
    print(f"Total nulls: {total_nulls}")

    # --- Regime coverage table ---
    print(f"\nREGIME COVERAGE")
    print(f"{'═' * 55}")
    print(f"{'Period':<13s} {'Rows':>5s}  {'Rate Range':<13s} {'Key Event'}")
    print(f"{'─' * 13} {'─' * 5}  {'─' * 13} {'─' * 20}")
    regimes = [
        ("2003-2007", "Pre-crisis"),
        ("2008-2009", "Financial crisis"),
        ("2010-2012", "Post-crisis"),
        ("2013-2016", "Taper tantrum"),
        ("2017-2019", "EM crisis 2018"),
        ("2020-2021", "COVID"),
        ("2022-2023", "Dollar surge"),
        ("2024-2025", "Current regime"),
    ]
    for period, event in regimes:
        y_start, y_end = period.split("-")
        mask = (merged["date"].dt.year >= int(y_start)) & (merged["date"].dt.year <= int(y_end))
        subset = merged.loc[mask, "usdinr"]
        if len(subset) > 0:
            rng = f"{subset.min():.1f}-{subset.max():.1f}"
            print(f"{period:<13s} {len(subset):>5d}  {rng:<13s} {event}")
        else:
            print(f"{period:<13s}     0  {'N/A':<13s} {event}")
    print(f"{'═' * 55}")

    return merged


def save(df: pd.DataFrame, path: str = OUTPUT_FILE) -> str:
    """Save dataframe to CSV. Returns the output path."""
    df.to_csv(path, index=False)
    print(f"\nSaved to {path} ({len(df)} rows)")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch market data from Yahoo Finance")
    parser.add_argument("--extended", action="store_true",
                        help="Fetch extended 10-year history (2015-present)")
    parser.add_argument("--full", action="store_true",
                        help="Fetch full history (2003-present)")
    args = parser.parse_args()

    if args.full:
        df = fetch_full_history()
        save(df, FULL_HISTORY_OUTPUT_FILE)
        print("\n--- Full history data fetch complete ---")
    elif args.extended:
        df = fetch_extended()
        save(df, EXTENDED_OUTPUT_FILE)
        print("\n--- Extended data fetch complete ---")
    else:
        df = fetch_all()
        save(df)
        print("\n--- Phase 1 complete ---")
