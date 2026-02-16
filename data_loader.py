"""
Data Loader (data_loader.py)
============================
All data download and loading functions in one place.

Data sources:
  1. BTC price (yfinance)
  2. Fear & Greed Index (alternative.me API)
  3. Binance derivatives - funding rate & OI (Binance Vision + Futures API fallback)
  4. Coinbase Premium proxy (yfinance: BTC-USD vs BTC=F)

Usage:
------
    # In engine.py or strategy_test.py:
    from data_loader import load_all_data

    df = load_all_data()  # Returns merged DataFrame with all raw columns

    # Update Binance derivatives CSV:
    python data_loader.py              # Incremental update
    python data_loader.py --full       # Full download from 2020-01-01
"""

import os
import io
import time
import zipfile
import concurrent.futures
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dateutil.relativedelta import relativedelta


# ============================================================================
# CONFIGURATION
# ============================================================================

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_PROJECT_DIR, "data", "binance")
BINANCE_CSV_PATH = os.path.join(_DATA_DIR, "binance_derivatives_daily.csv")

_SYMBOL = "BTCUSDT"
_BINANCE_START = datetime(2020, 1, 1)
_BASE_URL = "https://data.binance.vision/data/futures/um"
_FAPI_URL = "https://fapi.binance.com"
_RATE_LIMIT_SLEEP = 0.3


# ============================================================================
# 1. BTC PRICE (yfinance)
# ============================================================================

def load_btc_price(start: str = "2018-01-01") -> pd.DataFrame:
    """
    Download BTC-USD daily OHLCV from yfinance.

    Returns:
        DataFrame with columns: date, open, close, volume
    """
    print("Loading BTC price...")
    btc = yf.download("BTC-USD", start=start, progress=False)
    btc = btc.reset_index()
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [c[0] for c in btc.columns]
    btc = btc.rename(columns={"Date": "date", "Open": "open", "Close": "close", "Volume": "volume"})
    btc["date"] = pd.to_datetime(btc["date"]).dt.tz_localize(None).dt.normalize()
    return btc[["date", "open", "close", "volume"]].copy()


# ============================================================================
# 2. FEAR & GREED INDEX (alternative.me)
# ============================================================================

def load_fear_greed() -> pd.DataFrame:
    """
    Download Fear & Greed Index from alternative.me API.

    Returns:
        DataFrame with columns: date, fear_greed
        Empty DataFrame on failure.
    """
    print("Loading Fear & Greed Index...")
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=15)
        fg_data = resp.json()["data"]
        fg_df = pd.DataFrame(fg_data)
        fg_df["date"] = pd.to_datetime(fg_df["timestamp"].astype(int), unit="s").dt.normalize()
        fg_df["fear_greed"] = fg_df["value"].astype(int)
        print(f"  F&G: {fg_df['fear_greed'].notna().sum()} days loaded")
        return fg_df[["date", "fear_greed"]]
    except Exception as e:
        print(f"  F&G failed: {e}")
        return pd.DataFrame(columns=["date", "fear_greed"])


# ============================================================================
# 3. BINANCE DERIVATIVES (Vision archives + API fallback)
# ============================================================================

def _download_and_extract(url, file_label):
    """Download a ZIP from Binance Vision and return the inner CSV as DataFrame."""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                filename = z.namelist()[0]
                with z.open(filename) as f:
                    return pd.read_csv(f)
        elif response.status_code != 404:
            print(f"  Error {response.status_code}: {url}")
        return None
    except Exception as e:
        print(f"  Exception {file_label}: {e}")
        return None


def _fetch_monthly_funding(date):
    """Fetch one month of funding rate data from Binance Vision ZIP."""
    year, month = date.strftime("%Y"), date.strftime("%m")
    url = f"{_BASE_URL}/monthly/fundingRate/{_SYMBOL}/{_SYMBOL}-fundingRate-{year}-{month}.zip"
    df = _download_and_extract(url, f"Funding {year}-{month}")

    if df is not None:
        df = df.rename(columns={"last_funding_rate": "funding_rate"})
        if "calc_time" in df.columns and "funding_rate" in df.columns:
            df = df[["calc_time", "funding_rate"]]
            df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms")
            return df
        else:
            print(f"  Missing columns in {year}-{month}: {df.columns.tolist()}")
    return None


def _fetch_daily_metrics(date):
    """Fetch one day of OI metrics from Binance Vision ZIP."""
    date_str = date.strftime("%Y-%m-%d")
    url = f"{_BASE_URL}/daily/metrics/{_SYMBOL}/{_SYMBOL}-metrics-{date_str}.zip"
    df = _download_and_extract(url, f"Metrics {date_str}")

    if df is not None:
        if "create_time" in df.columns and "sum_open_interest_value" in df.columns:
            return {
                "date": date.replace(hour=0, minute=0, second=0, microsecond=0),
                "open_interest_usd": df["sum_open_interest_value"].mean(),
            }
        else:
            print(f"  Missing metrics columns in {date_str}: {df.columns.tolist()}")
    return None


def _fetch_funding_via_api(start_date, end_date):
    """Fallback: fetch recent funding rate via Binance Futures REST API."""
    print(f"\n--- Fetching recent Funding Rate via API ({start_date.date()} ~ {end_date.date()}) ---")
    records = []
    cursor_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    while cursor_ms < end_ms:
        resp = requests.get(
            f"{_FAPI_URL}/fapi/v1/fundingRate",
            params={"symbol": _SYMBOL, "startTime": cursor_ms, "endTime": end_ms, "limit": 1000},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        records.extend(data)
        cursor_ms = data[-1]["fundingTime"] + 1
        time.sleep(_RATE_LIMIT_SLEEP)

    if not records:
        return pd.DataFrame(columns=["date", "funding_rate"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.tz_localize(None).dt.normalize()
    df["funding_rate"] = df["fundingRate"].astype(float)
    daily = df.groupby("date")["funding_rate"].mean().reset_index()
    print(f"  API supplemented {len(daily)} days")
    return daily


def load_derivatives_csv() -> pd.DataFrame:
    """
    Load the local Binance derivatives CSV.

    Returns:
        DataFrame with columns: date, funding_rate, open_interest_usd
        Empty DataFrame if file doesn't exist.
    """
    if not os.path.exists(BINANCE_CSV_PATH):
        return pd.DataFrame(columns=["date", "funding_rate", "open_interest_usd"])
    df = pd.read_csv(BINANCE_CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce").fillna(0)
    df["open_interest_usd"] = pd.to_numeric(df["open_interest_usd"], errors="coerce").fillna(0)
    return df


def update_derivatives(full: bool = False):
    """
    Download / update Binance derivatives CSV (funding rate + OI).

    Args:
        full: If True, re-download everything from 2020-01-01.
              If False, incremental update from last valid date.
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    end_date = datetime.utcnow() - timedelta(days=2)

    existing = load_derivatives_csv()
    if not full and not existing.empty:
        valid = existing[existing["funding_rate"] != 0]
        actual_start = valid["date"].max().replace(day=1) if not valid.empty else _BINANCE_START
        print(f"Incremental update for {_SYMBOL}: {actual_start.date()} to {end_date.date()}")
    else:
        actual_start = _BINANCE_START
        print(f"Full download for {_SYMBOL}: {actual_start.date()} to {end_date.date()}")

    # --- Funding Rate (monthly archives) ---
    print("\n--- Downloading Funding Rates (Monthly) ---")
    months = []
    cur = actual_start
    while cur < end_date:
        months.append(cur)
        cur += relativedelta(months=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_fetch_monthly_funding, months))

    funding_frames = [r for r in results if r is not None]
    if funding_frames:
        all_funding = pd.concat(funding_frames)
        all_funding["date"] = all_funding["calc_time"].dt.normalize()
        daily_funding = all_funding.groupby("date")["funding_rate"].mean().reset_index()
        print(f"Successfully processed {len(daily_funding)} days of Funding Rates.")
    else:
        daily_funding = pd.DataFrame(columns=["date", "funding_rate"])

    # API fallback for recent un-archived data
    api_start = daily_funding["date"].max() + timedelta(days=1) if not daily_funding.empty else actual_start
    if api_start.date() < end_date.date():
        api_daily = _fetch_funding_via_api(api_start, end_date)
        if not api_daily.empty:
            daily_funding = pd.concat([daily_funding, api_daily], ignore_index=True)
            daily_funding = daily_funding.drop_duplicates(subset="date", keep="last").sort_values("date")

    if daily_funding.empty:
        print("Warning: No Funding Rate data found from any source.")

    # --- Open Interest (daily metrics archives) ---
    print("\n--- Downloading Open Interest (Daily Metrics) ---")
    days = []
    cur = actual_start
    while cur < end_date:
        days.append(cur)
        cur += timedelta(days=1)

    metrics_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_fetch_daily_metrics, d): d for d in days}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                metrics_data.append(res)
            completed += 1
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{len(days)} days checked")

    if metrics_data:
        daily_oi = pd.DataFrame(metrics_data).sort_values("date")
        print(f"Successfully processed {len(daily_oi)} days of Open Interest.")
    else:
        print("Warning: No Open Interest data found.")
        daily_oi = pd.DataFrame(columns=["date", "open_interest_usd"])

    # --- Merge & save ---
    print("\n--- Merging Data ---")
    if not daily_funding.empty and not daily_oi.empty:
        final_df = pd.merge(daily_funding, daily_oi, on="date", how="outer")
    elif not daily_funding.empty:
        final_df = daily_funding
        final_df["open_interest_usd"] = 0
    elif not daily_oi.empty:
        final_df = daily_oi
        final_df["funding_rate"] = 0
    else:
        print("Error: No data retrieved.")
        return

    final_df = final_df.sort_values("date").fillna(0)

    # Incremental: merge with existing
    if not full and not existing.empty:
        cutoff = final_df["date"].min()
        keep = existing[existing["date"] < cutoff]
        final_df = pd.concat([keep, final_df], ignore_index=True).sort_values("date").reset_index(drop=True)

    final_df.to_csv(BINANCE_CSV_PATH, index=False)
    print(f"\nSaved: {BINANCE_CSV_PATH}")
    print(f"Total: {len(final_df)} days ({final_df['date'].min().date()} ~ {final_df['date'].max().date()})")
    print(final_df.tail())


# ============================================================================
# 4. COINBASE PREMIUM (yfinance)
# ============================================================================

def load_coinbase_premium() -> pd.DataFrame:
    """
    Download Coinbase Premium proxy (BTC-USD spot vs CME BTC=F futures).

    Returns:
        DataFrame with columns: date, coinbase_premium
        Empty DataFrame on failure.
    """
    print("Loading Coinbase Premium (BTC-USD vs CME BTC=F)...")
    try:
        spot = yf.download("BTC-USD", period="max", progress=False).reset_index()
        futures = yf.download("BTC=F", period="max", progress=False).reset_index()
        for d in [spot, futures]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] for c in d.columns]
        spot = spot.rename(columns={"Date": "date", "Close": "spot_close"})
        futures = futures.rename(columns={"Date": "date", "Close": "futures_close"})
        spot["date"] = pd.to_datetime(spot["date"]).dt.tz_localize(None).dt.normalize()
        futures["date"] = pd.to_datetime(futures["date"]).dt.tz_localize(None).dt.normalize()
        m = pd.merge(spot[["date", "spot_close"]], futures[["date", "futures_close"]], on="date", how="inner")
        m["coinbase_premium"] = ((m["spot_close"] - m["futures_close"]) / m["futures_close"]) * 100
        print(f"  CB Premium: {len(m)} days loaded")
        return m[["date", "coinbase_premium"]]
    except Exception as e:
        print(f"  CB Premium failed: {e}")
        return pd.DataFrame(columns=["date", "coinbase_premium"])


# ============================================================================
# 5. UNIFIED LOADER (merges all sources, no indicators)
# ============================================================================

def load_all_data(start: str = "2018-01-01",
                  include_fg: bool = True,
                  include_derivatives: bool = True,
                  include_cb_premium: bool = False) -> pd.DataFrame:
    """
    Load and merge all raw data sources into a single DataFrame.
    Does NOT compute indicators (that's engine.py's job).

    Args:
        start: Start date for BTC price data.
        include_fg: Include Fear & Greed Index.
        include_derivatives: Include Binance Funding Rate and OI from local CSV.
        include_cb_premium: Include Coinbase Premium proxy.

    Returns:
        DataFrame with columns: date, open, close, volume,
        [fear_greed], [funding_rate, open_interest_usd], [coinbase_premium]
    """
    df = load_btc_price(start)

    if include_fg:
        fg_df = load_fear_greed()
        if not fg_df.empty:
            df = pd.merge(df, fg_df, on="date", how="left")
        else:
            df["fear_greed"] = np.nan

    if include_derivatives:
        deriv = load_derivatives_csv()
        if not deriv.empty:
            print(f"Loading Binance derivatives...")
            df = pd.merge(df, deriv[["date", "funding_rate", "open_interest_usd"]], on="date", how="left")
            print(f"  Derivatives: {len(deriv)} days loaded")
        else:
            print(f"  Derivatives CSV not found: {BINANCE_CSV_PATH}")
            df["funding_rate"] = np.nan
            df["open_interest_usd"] = np.nan

    if include_cb_premium:
        cb_df = load_coinbase_premium()
        if not cb_df.empty:
            df = pd.merge(df, cb_df, on="date", how="left")
        else:
            df["coinbase_premium"] = np.nan

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ============================================================================
# CLI: python data_loader.py [--full]
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update Binance derivatives data")
    parser.add_argument("--full", action="store_true", help="Full download (default: incremental)")
    args = parser.parse_args()
    update_derivatives(full=args.full)
