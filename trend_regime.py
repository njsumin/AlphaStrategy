"""
trend_regime.py — Trend Regime Detection Interface
====================================================
Unified interface for computing bull/bear market regimes from daily OHLCV data.
Used to gate long/short entries: longs only in bull regime, shorts only in bear.

Supported methods:
  supertrend  — ATR-based dynamic support/resistance bands (default)
  ma200       — Price vs 200-day SMA
  ma_cross    — Fast/slow SMA golden-cross / death-cross
  momentum    — N-day return sign

No-lookahead guarantee:
  Daily regime computed from close[D] is mapped to 15m bars on day D+1 onwards
  (shift=1 day). Within-day bars never see the same-day regime flip.

Usage:
    from trend_regime import compute_regime, map_regime_to_bars
    from trend_regime import RegimeLongCond, RegimeShortCond

    regime = compute_regime(df_daily, method="supertrend", period=10, multiplier=3.0)
    df_15m["regime"] = map_regime_to_bars(regime, df_15m)

    long_cond  = RegimeLongCond(ReversalLongCond(prox, decel))
    short_cond = RegimeShortCond(ReversalShortCond(prox, decel))
"""

import numpy as np
import pandas as pd


# ============================================================================
# Core Supertrend Calculation
# ============================================================================

def _compute_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        period: int = 10, multiplier: float = 3.0):
    """
    Supertrend indicator using Wilder's ATR smoothing.

    Band update rules (no lookahead — uses close[i-1] for decisions):
      Upper band: tightens downward only when new basic_upper < prev final_upper
                  OR prev close > prev final_upper (trend flipped)
      Lower band: tightens upward only when new basic_lower > prev final_lower
                  OR prev close < prev final_lower (trend flipped)
      Direction : +1 (bull) when close > lower band; -1 (bear) when close < upper band

    Args:
        high, low, close: np arrays of daily OHLCV.
        period    : ATR smoothing window. Default=10.
        multiplier: Band width factor. Default=3.0.

    Returns:
        supertrend (np.ndarray): Support/resistance line value per bar.
        direction  (np.ndarray): +1 = bull, -1 = bear.
    """
    n = len(close)
    prev_close = np.concatenate([[close[0]], close[:-1]])

    # True Range
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    # Wilder ATR
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    direction   = np.ones(n, dtype=np.int8)
    supertrend  = np.zeros(n)

    # Initialise first valid bar
    if n >= period:
        final_upper[period - 1] = basic_upper[period - 1]
        final_lower[period - 1] = basic_lower[period - 1]
        supertrend[period - 1]  = final_lower[period - 1]

    for i in range(period, n):
        # Upper band: lock in lower value once price stays below
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band: lock in higher value once price stays above
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction flip
        if direction[i - 1] == -1:
            direction[i] = np.int8(1) if close[i] > final_upper[i] else np.int8(-1)
        else:
            direction[i] = np.int8(-1) if close[i] < final_lower[i] else np.int8(1)

        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return supertrend, direction


# ============================================================================
# Public Interface
# ============================================================================

def compute_regime(df_daily: pd.DataFrame,
                   method: str = "supertrend",
                   **kwargs) -> pd.Series:
    """
    Compute bull/bear market regime from daily OHLCV.

    Args:
        df_daily : DataFrame with columns [date, high, low, close].
                   'date' can be a column or the index.
        method   : Regime detection method. One of:
                     "supertrend" — ATR Supertrend (default; period=10, multiplier=3.0)
                     "ma200"      — Price vs 200d SMA
                     "ma_cross"   — SMA fast/slow crossover (fast=50, slow=200)
                     "momentum"   — Sign of N-day return (lookback=90)
        **kwargs : Method-specific parameters (see method descriptions above).

    Returns:
        pd.Series indexed by date, values +1 (bull) or -1 (bear).
        NaN at warmup bars before the indicator has enough history.
    """
    df = df_daily.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    if method == "supertrend":
        period     = int(kwargs.get("period",     10))
        multiplier = float(kwargs.get("multiplier", 3.0))
        hi = df["high"].values
        lo = df["low"].values
        cl = df["close"].values
        _, direction = _compute_supertrend(hi, lo, cl, period=period,
                                           multiplier=multiplier)
        regime = pd.Series(direction.astype(float), index=df.index, name="regime")
        # Warmup bars before ATR is valid → NaN
        if len(regime) >= period:
            regime.iloc[:period - 1] = np.nan

    elif method == "ma200":
        window = int(kwargs.get("window", 200))
        ma = df["close"].rolling(window).mean()
        regime = np.where(df["close"] > ma, 1.0, -1.0)
        regime = pd.Series(regime, index=df.index, name="regime")
        regime.iloc[:window - 1] = np.nan

    elif method == "ma_cross":
        fast = int(kwargs.get("fast", 50))
        slow = int(kwargs.get("slow", 200))
        ma_fast = df["close"].rolling(fast).mean()
        ma_slow = df["close"].rolling(slow).mean()
        regime = np.where(ma_fast > ma_slow, 1.0, -1.0)
        regime = pd.Series(regime, index=df.index, name="regime")
        regime.iloc[:slow - 1] = np.nan

    elif method == "momentum":
        lookback = int(kwargs.get("lookback", 90))
        ret = df["close"].pct_change(lookback)
        regime = np.where(ret > 0, 1.0, -1.0)
        regime = pd.Series(regime, index=df.index, name="regime")
        regime.iloc[:lookback] = np.nan

    else:
        raise ValueError(
            f"Unknown regime method: {method!r}. "
            "Choose from: supertrend, ma200, ma_cross, momentum"
        )

    return regime


def map_regime_to_bars(regime: pd.Series,
                       df_bars: pd.DataFrame,
                       date_col: str = "date",
                       shift_days: int = 1) -> np.ndarray:
    """
    Forward-fill daily regime onto any bar frequency (15m, 1h, daily).

    No-lookahead: regime[D] is applied to bars on day D+shift_days onwards.
    Default shift_days=1 means today's bars use yesterday's daily regime signal.

    Args:
        regime    : pd.Series from compute_regime(), indexed by date.
        df_bars   : Target DataFrame with date_col timestamp column.
        date_col  : Name of timestamp column in df_bars.
        shift_days: Days to shift regime forward (default 1 for no-lookahead).

    Returns:
        np.ndarray of int8 (+1 or -1), same length as df_bars.
        Bars before the first valid regime → default +1 (bull).
    """
    # Shift regime by N days (so bars on day D see regime from day D-N)
    if shift_days > 0:
        regime = regime.shift(shift_days)

    # Forward-fill NaN (warmup / gaps)
    regime_ffill = regime.ffill()

    # Build date-to-regime lookup (day-level keys)
    regime_dates = regime_ffill.index.normalize()
    regime_dict  = dict(zip(regime_dates, regime_ffill.values))

    bar_dates = pd.to_datetime(df_bars[date_col]).dt.normalize()
    result    = np.ones(len(df_bars), dtype=np.int8)  # default bull
    last      = np.int8(1)
    for i, d in enumerate(bar_dates):
        v = regime_dict.get(d, None)
        if v is not None and not np.isnan(v):
            last = np.int8(int(v))
        result[i] = last

    return result


def resample_to_daily(df_bars: pd.DataFrame,
                      date_col: str = "date") -> pd.DataFrame:
    """
    Resample 15m or 1h bar DataFrame to daily OHLCV.
    Needed when only intraday data is available and daily regime is required.

    Args:
        df_bars : DataFrame with date_col, open, high, low, close, volume.
        date_col: Name of timestamp column.

    Returns:
        Daily DataFrame with columns [date, open, high, low, close, volume].
    """
    df = df_bars.copy()
    df["_dt"] = pd.to_datetime(df[date_col])
    df = df.set_index("_dt")

    has_hl = "high" in df.columns and "low" in df.columns

    agg = {"close": "last", "open": "first"}
    if has_hl:
        agg["high"] = "max"
        agg["low"]  = "min"
    if "volume" in df.columns:
        agg["volume"] = "sum"

    daily = df.resample("D").agg(agg).dropna(subset=["close"])
    daily = daily.reset_index().rename(columns={"_dt": "date"})

    if not has_hl:
        daily["high"] = daily["close"]
        daily["low"]  = daily["close"]

    return daily[["date", "open", "high", "low", "close"]
                 + (["volume"] if "volume" in daily.columns else [])]


# ============================================================================
# Condition Wrappers
# ============================================================================

class RegimeLongCond:
    """
    Wrap any long condition: only fires when df regime column == +1 (bull).

    Args:
        inner_cond : Existing buy condition callable (row -> str/True/False).
        regime_col : Column name for regime in the DataFrame row. Default='regime'.

    Example:
        long = RegimeLongCond(ReversalLongCond(0.005, 1.0))
        # long(row) returns False when row['regime'] == -1 (bear market)
    """
    def __init__(self, inner_cond, regime_col: str = "regime"):
        self.inner     = inner_cond
        self.regime_col = regime_col
        inner_name = getattr(inner_cond, "name",
                             getattr(inner_cond, "__class__.__name__", "L"))
        self.name = f"Regime+({inner_name})"

    def __call__(self, row: pd.Series):
        regime_val = row.get(self.regime_col, 1)
        if pd.isna(regime_val) or int(regime_val) != 1:
            return False
        return self.inner(row)

    def __repr__(self):
        return f"RegimeLongCond({self.inner!r})"


class RegimeShortCond:
    """
    Wrap any short condition: only fires when df regime column == -1 (bear).

    Args:
        inner_cond : Existing short condition callable (row -> str/True/False).
        regime_col : Column name for regime in the DataFrame row. Default='regime'.

    Example:
        short = RegimeShortCond(ReversalShortCond(0.005, 1.0))
        # short(row) returns False when row['regime'] == +1 (bull market)
    """
    def __init__(self, inner_cond, regime_col: str = "regime"):
        self.inner      = inner_cond
        self.regime_col = regime_col
        inner_name = getattr(inner_cond, "name",
                             getattr(inner_cond, "__class__.__name__", "S"))
        self.name = f"Regime-({inner_name})"

    def __call__(self, row: pd.Series):
        regime_val = row.get(self.regime_col, -1)
        if pd.isna(regime_val) or int(regime_val) != -1:
            return False
        return self.inner(row)

    def __repr__(self):
        return f"RegimeShortCond({self.inner!r})"
