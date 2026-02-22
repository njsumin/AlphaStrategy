"""
Volume Bar Generation and Signal Mapping (volume_bars.py)
=========================================================
Converts 15-minute OHLCV data to constant-volume bars for signal generation,
then maps those signals back to 15-minute execution bars.

Architecture:
  15m bars (execution) ← signals ← Volume bars (signal generation)

Signal generation on volume bars:
  - Each volume bar covers a fixed amount of traded volume
  - S/R levels and momentum indicators are computed on volume bars
  - Signals are cleaner because each bar represents equal market activity

Execution on 15m bars:
  - The standard backtest engine runs on 15m bars (T+1 open execution)
  - Entry signals are populated only at the 15m bar where each volume bar closes
  - S/R levels propagate forward between volume bar updates
  - Distances to S/R are recomputed live from 15m close price every bar
  - Exit conditions (stop loss, near S/R) evaluated at every 15m bar

Usage:
------
    from volume_bars import (
        make_volume_bars,
        compute_volume_per_bar,
        add_vbar_sr_levels,
        map_vbar_signals_to_15m,
    )
"""

import numpy as np
import pandas as pd

from engine import _add_volume_profile
from sr_levels import add_sr_levels


# ============================================================================
# VOLUME BAR GENERATION
# ============================================================================

def make_volume_bars(df: pd.DataFrame,
                     volume_per_bar: float) -> pd.DataFrame:
    """
    Convert 15m OHLCV bars to constant-volume bars.

    Accumulates consecutive 15m bars until cumulative volume >= volume_per_bar.
    If a single 15m bar already exceeds volume_per_bar it becomes one bar.

    Args:
        df: 15m OHLCV DataFrame with columns:
            'date', 'open', 'high', 'low', 'close', 'volume'.
        volume_per_bar: Target cumulative volume per output bar.

    Returns:
        Volume bar DataFrame (same OHLCV structure).
        'date' = timestamp of the last 15m bar in each volume bar.
    """
    if volume_per_bar <= 0:
        raise ValueError(f"volume_per_bar must be > 0, got {volume_per_bar}")

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    volume = df["volume"].fillna(0).values.astype(float)
    open_  = df["open"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    close  = df["close"].values.astype(float)
    dates  = df["date"].values
    n      = len(df)

    if n == 0:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"])

    cumvol = np.cumsum(volume)
    total_vol = cumvol[-1]
    if total_vol <= 0:
        raise ValueError("Total volume is zero — cannot build volume bars.")

    # Thresholds: bar k closes when cumvol first crosses k * volume_per_bar
    n_bars_max = int(total_vol / volume_per_bar) + 2
    thresholds = np.arange(1, n_bars_max + 1, dtype=float) * volume_per_bar

    # For each threshold, find first index i where cumvol[i] >= threshold
    bar_end_idx = np.searchsorted(cumvol, thresholds, side="left")
    bar_end_idx = bar_end_idx[bar_end_idx < n]    # drop out-of-range
    bar_end_idx = np.unique(bar_end_idx)           # de-duplicate (single bar may span multiple thresholds)

    if len(bar_end_idx) == 0:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"])

    bar_start_idx = np.concatenate([[0], bar_end_idx[:-1] + 1])

    rows = []
    for s, e in zip(bar_start_idx, bar_end_idx):
        rows.append({
            "date":   dates[e],
            "open":   open_[s],
            "high":   float(np.max(high[s:e + 1])),
            "low":    float(np.min(low[s:e + 1])),
            "close":  close[e],
            "volume": float(np.sum(volume[s:e + 1])),
        })

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def compute_volume_per_bar(df_15m: pd.DataFrame,
                           target_bars_per_day: float,
                           train_start: str = "2021-01-01",
                           train_end: str = "2025-05-31") -> float:
    """
    Compute volume_per_bar from training-set median daily volume (15m data).

    Uses only training-set data to avoid lookahead bias.
    volume_per_bar = median_daily_volume / target_bars_per_day

    Args:
        df_15m: Full 15-minute OHLCV DataFrame.
        target_bars_per_day: Desired average volume bars per day during training.
        train_start / train_end: Training period boundaries.

    Returns:
        volume_per_bar (float).
    """
    mask = (df_15m["date"] >= train_start) & (df_15m["date"] <= train_end)
    df_tr = df_15m[mask].copy()
    if len(df_tr) == 0:
        raise ValueError("No 15m data in training period.")

    df_tr["_date_only"] = pd.to_datetime(df_tr["date"]).dt.date
    daily_vol = df_tr.groupby("_date_only")["volume"].sum()
    median_daily_vol = float(daily_vol.median())

    if median_daily_vol <= 0:
        raise ValueError("Median daily volume is zero in training period.")

    return median_daily_vol / target_bars_per_day


# ============================================================================
# S/R LEVEL DETECTION ON VOLUME BARS
# ============================================================================

def add_vbar_sr_levels(df_vbars: pd.DataFrame,
                       vp_windows: list = None,
                       cluster_gap_pct: float = 0.01) -> pd.DataFrame:
    """
    Add VP-based S/R levels to volume bar data.

    Looks back `w` volume bars to build each Volume Profile.
    The 'd' suffix in column names refers to number of volume bars, not days.

    Args:
        df_vbars: Volume bar DataFrame from make_volume_bars().
        vp_windows: Lookback sizes in volume bars. Default [30, 60, 120].
        cluster_gap_pct: Minimum gap fraction for level merging.

    Adds: nearest_resistance, nearest_support,
          dist_to_resistance, dist_to_support, sr_count.
    """
    if vp_windows is None:
        vp_windows = [30, 60, 120]

    df = df_vbars.copy()
    for w in vp_windows:
        df = _add_volume_profile(df, bars_per_day=1, window_days=w)

    df = add_sr_levels(df, vp_windows=vp_windows, cluster_gap_pct=cluster_gap_pct)
    return df


# ============================================================================
# SIGNAL MAPPING: VOLUME BARS → 15-MINUTE EXECUTION BARS
# ============================================================================

def map_vbar_signals_to_15m(df_15m: pd.DataFrame,
                             df_vbar: pd.DataFrame) -> pd.DataFrame:
    """
    Map volume bar indicators back to 15-minute execution bars.

    Mapping rules:
      Entry signals (norm_velocity, velocity_delta, etc.):
        - Non-NaN ONLY at the 15m bar where a volume bar closes.
        - NaN at all other 15m bars → entry conditions return False.
        - This ensures entries fire at most once per volume bar.

      S/R levels (nearest_resistance, nearest_support):
        - Set at volume bar close bars, then propagated forward (ffill).
        - Stable between volume bar updates.

      Distances (dist_to_resistance, dist_to_support):
        - Recomputed every 15m bar using the LIVE 15m close price.
        - This allows exit conditions to track price movement continuously.

    Args:
        df_15m: 15-minute OHLCV DataFrame (execution bars). Must be sorted
                by date ascending and cover at least the volume bar period.
        df_vbar: Volume bar DataFrame with computed momentum + S/R indicators.

    Returns:
        15m DataFrame enriched with volume-bar-derived signals.
        The caller should set .attrs["bars_per_day"] = 96 before backtest.
    """
    # Work on a copy of the 15m data
    result = df_15m[["date", "open", "high", "low", "close", "volume"]].copy()
    result["date"] = pd.to_datetime(result["date"])
    n_15m = len(result)

    # Columns sourced from volume bars
    # Entry cols: NaN except at vbar close bars (prevents mid-bar signals)
    entry_cols = [
        "norm_velocity", "velocity_delta",
        "price_velocity",
        "fast_norm_velocity", "fast_velocity_delta",
        "fast_price_velocity",
        "sr_count", "atr",
    ]
    # S/R level cols: set at vbar closes, then forward-filled
    sr_level_cols = ["nearest_resistance", "nearest_support"]

    all_vbar_cols = [c for c in entry_cols + sr_level_cols if c in df_vbar.columns]
    for col in all_vbar_cols:
        result[col] = np.nan

    # --- Find matching 15m bars for each volume bar close timestamp ---
    # For each vbar date, find the last 15m bar with date <= vbar date.
    # (A volume bar closes at the end of a 15m bar, so timestamps should align.)
    dates_15m = result["date"].values.astype("datetime64[ns]")
    dates_vb  = pd.to_datetime(df_vbar["date"]).values.astype("datetime64[ns]")

    # searchsorted(side="right") → first index where 15m date > vbar date
    # subtract 1 → last 15m bar with date <= vbar date
    idx_arr = np.searchsorted(dates_15m, dates_vb, side="right") - 1
    valid   = (idx_arr >= 0) & (idx_arr < n_15m)

    # Vectorised assignment: write vbar signals to matching 15m bar indices
    for col in all_vbar_cols:
        col_arr  = np.full(n_15m, np.nan, dtype=float)
        src_vals = df_vbar[col].values.astype(float)
        col_arr[idx_arr[valid]] = src_vals[valid]
        result[col] = col_arr

    # --- Forward-fill S/R levels across 15m bars ---
    for col in sr_level_cols:
        if col in result.columns:
            result[col] = result[col].ffill()

    # --- Recompute distances from LIVE 15m close price ---
    close = result["close"].values
    res   = result["nearest_resistance"].values if "nearest_resistance" in result.columns else np.full(n_15m, np.nan)
    sup   = result["nearest_support"].values    if "nearest_support"    in result.columns else np.full(n_15m, np.nan)

    result["dist_to_resistance"] = np.where(
        ~np.isnan(res) & (close > 0),
        (res - close) / close,
        np.nan,
    )
    result["dist_to_support"] = np.where(
        ~np.isnan(sup) & (close > 0),
        (close - sup) / close,
        np.nan,
    )

    return result
