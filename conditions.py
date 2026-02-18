"""
Strategy Conditions Library (conditions.py)
============================================
Reusable buy/sell/close condition functions with configurable parameters.

Each condition function takes a DataFrame row (pd.Series) and returns:
  - True/False  for simple conditions
  - str label   for source-tracking (truthy = buy/sell triggered)

All thresholds are configurable via function parameters with sensible defaults
derived from historical backtesting (2018-2026).

Usage:
------
    from conditions import FearGreedBuyCond, RSIBuyCond, FundingBuyCond
    from conditions import FearGreedSellCond, RSISellCond, FundingSellCond
    from conditions import combine_conditions

    # Single condition
    buy = FearGreedBuyCond(threshold=10)
    signal = buy(row)  # True if fear_greed < 10

    # Combined conditions (AND)
    buy = combine_conditions(
        FearGreedBuyCond(threshold=10),
        RSIBuyCond(threshold=35),
        mode="AND"
    )

    # Combined conditions (OR)
    buy = combine_conditions(
        FearGreedBuyCond(threshold=10),
        FundingBuyCond(threshold=-0.0001),
        mode="OR"
    )
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Callable, List

# Type alias: condition functions return truthy (str label) or False
ConditionResult = Union[str, bool]
ConditionFn = Callable[[pd.Series], ConditionResult]


# ============================================================================
# BUY CONDITIONS
# ============================================================================

class FearGreedBuyCond:
    """
    Buy when Fear & Greed Index is below threshold (extreme fear).

    Args:
        threshold (int): Buy when F&G < threshold. Default=10 (extreme fear).
            - 10: Only the most extreme panic days (~3% of all days)
            - 15: Moderate fear filter (~5%)
            - 25: Broad fear zone (~15%)

    Required columns: fear_greed
    Best used with: RSIBuyCond for confirmation
    Historical performance: F&G<10 alone catches major bottoms (Feb 2018, Mar 2020, May 2022)
    """
    def __init__(self, threshold: int = 10):
        self.threshold = threshold
        self.name = f"FG<{threshold}"

    def __call__(self, row: pd.Series) -> ConditionResult:
        fg = row.get("fear_greed", np.nan)
        if pd.isna(fg):
            return False
        return self.name if fg < self.threshold else False

    def __repr__(self):
        return f"FearGreedBuyCond(threshold={self.threshold})"


class RSIBuyCond:
    """
    Buy when RSI-14 is below threshold (oversold).

    Args:
        threshold (float): Buy when RSI < threshold. Default=35.
            - 30: Strict oversold (fewer signals, higher conviction)
            - 35: Standard (proven with F&G combo)
            - 40: Relaxed (more signals, lower conviction)
        period (int): RSI lookback period. Default=14. Column name = rsi_{period}.

    Required columns: rsi_14 (or rsi_{period})
    Best used with: FearGreedBuyCond or FundingBuyCond
    """
    def __init__(self, threshold: float = 35.0, period: int = 14):
        self.threshold = threshold
        self.period = period
        self.col = f"rsi_{period}"
        self.name = f"RSI<{threshold:.0f}"

    def __call__(self, row: pd.Series) -> ConditionResult:
        rsi = row.get(self.col, np.nan)
        if pd.isna(rsi):
            return False
        return self.name if rsi < self.threshold else False

    def __repr__(self):
        return f"RSIBuyCond(threshold={self.threshold}, period={self.period})"


class FundingBuyCond:
    """
    Buy when Funding Rate SMA7 is below threshold (shorts dominant = mean-reversion).

    Args:
        threshold (float): Buy when funding_sma7 < threshold. Default=-0.0001 (-0.01%).
            - -0.0001: Standard (negative funding = shorts paying longs)
            - -0.0005: Strict (deeply negative funding, very rare)
            - 0: Any negative funding

    Required columns: funding_sma7
    Historical: Catches COVID crash (Mar 2020), China ban (Jul 2021), FTX (Nov 2022)
    """
    def __init__(self, threshold: float = -0.0001):
        self.threshold = threshold
        self.name = f"Fund<{threshold*100:.2f}%"

    def __call__(self, row: pd.Series) -> ConditionResult:
        f = row.get("funding_sma7", np.nan)
        if pd.isna(f):
            return False
        return self.name if f < self.threshold else False

    def __repr__(self):
        return f"FundingBuyCond(threshold={self.threshold})"


class OIDropBuyCond:
    """
    Buy when Open Interest drops significantly (leveraged positions liquidated).

    Args:
        threshold (float): Buy when oi_pct_7d < -threshold. Default=-0.15 (-15%).
            - -0.10: Moderate OI drop
            - -0.15: Standard (significant liquidation)
            - -0.25: Extreme flush

    Required columns: oi_pct_7d, oi_valid
    """
    def __init__(self, threshold: float = -0.15):
        self.threshold = threshold
        self.name = f"OI<{threshold*100:.0f}%"

    def __call__(self, row: pd.Series) -> ConditionResult:
        oi = row.get("oi_pct_7d", np.nan)
        valid = row.get("oi_valid", False)
        if not valid or pd.isna(oi):
            return False
        return self.name if oi < self.threshold else False

    def __repr__(self):
        return f"OIDropBuyCond(threshold={self.threshold})"


class CBPremiumBuyCond:
    """
    Buy when Coinbase Premium SMA7 is below threshold (US sellers panicking).

    Args:
        threshold (float): Buy when cb_prem_sma7 < threshold. Default=-0.5%.
            - -0.3: Mild US selling
            - -0.5: Standard (P19 of distribution)
            - -0.75: Tight (P10)
            - -1.0: Ultra tight (P3, extreme panic only)

    Required columns: cb_prem_sma7
    Note: This is a proxy (Spot vs CME Futures), not true Coinbase-Binance spread.
    """
    def __init__(self, threshold: float = -0.5):
        self.threshold = threshold
        self.name = f"CB<{threshold:.1f}%"

    def __call__(self, row: pd.Series) -> ConditionResult:
        p = row.get("cb_prem_sma7", np.nan)
        if pd.isna(p):
            return False
        return self.name if p < self.threshold else False

    def __repr__(self):
        return f"CBPremiumBuyCond(threshold={self.threshold})"


class PriceDropBuyCond:
    """
    Buy when price has dropped significantly from recent high (drawdown buy).

    Args:
        threshold (float): Buy when drawdown > threshold. Default=-0.20 (-20%).
        lookback (int): Days to look back for the high. Default=30.

    Required columns: close (and close_high_{lookback}d computed by engine)
    """
    def __init__(self, threshold: float = -0.20, lookback: int = 30):
        self.threshold = threshold
        self.lookback = lookback
        self.col = f"drawdown_{lookback}d"
        self.name = f"Drop>{threshold*100:.0f}%/{lookback}d"

    def __call__(self, row: pd.Series) -> ConditionResult:
        dd = row.get(self.col, np.nan)
        if pd.isna(dd):
            return False
        return self.name if dd < self.threshold else False

    def __repr__(self):
        return f"PriceDropBuyCond(threshold={self.threshold}, lookback={self.lookback})"


class TrendFilterCond:
    """
    Trend filter: True when close > SMA(N) (uptrend confirmed).

    Args:
        ma_period (int): Moving average period. Default=200.

    Required columns: sma_{ma_period}
    """
    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period
        self.col = f"sma_{ma_period}"
        self.name = f"Close>SMA{ma_period}"

    def __call__(self, row: pd.Series) -> ConditionResult:
        sma = row.get(self.col, np.nan)
        close = row.get("close", np.nan)
        if pd.isna(sma) or pd.isna(close):
            return False
        return self.name if close > sma else False

    def __repr__(self):
        return f"TrendFilterCond(ma_period={self.ma_period})"


class FundingNotOverheatedCond:
    """
    Buy filter: True when Funding Rate SMA7 is below max threshold (not overheated).

    Args:
        max_funding (float): Upper limit. Default=0.0003 (0.03%/8h).

    Required columns: funding_sma7
    Note: Returns True (pass-through) when data is missing.
    """
    def __init__(self, max_funding: float = 0.0003):
        self.max_funding = max_funding
        self.name = f"Fund<{max_funding*100:.2f}%"

    def __call__(self, row: pd.Series) -> ConditionResult:
        f = row.get("funding_sma7", np.nan)
        if pd.isna(f):
            return self.name  # No data = don't filter
        return self.name if f < self.max_funding else False

    def __repr__(self):
        return f"FundingNotOverheatedCond(max_funding={self.max_funding})"


class RSIRisingCond:
    """
    RSI direction filter: True when RSI has risen by at least min_delta over lookback days.

    Args:
        min_delta (float): Minimum RSI change. Default=0 (any rise).
        lookback (int): Days for diff. Default=3.
        rsi_period (int): RSI period. Default=14.

    Required columns: rsi_{rsi_period}_delta{lookback}
    """
    def __init__(self, min_delta: float = 0.0, lookback: int = 3, rsi_period: int = 14):
        self.min_delta = min_delta
        self.lookback = lookback
        self.col = f"rsi_{rsi_period}_delta{lookback}"
        self.name = f"RSI_d{lookback}>{min_delta:.0f}"

    def __call__(self, row: pd.Series) -> ConditionResult:
        d = row.get(self.col, np.nan)
        if pd.isna(d):
            return False
        return self.name if d > self.min_delta else False

    def __repr__(self):
        return f"RSIRisingCond(min_delta={self.min_delta}, lookback={self.lookback})"


class VPAbovePOCCond:
    """
    Signal when price is above VP Point of Control.

    Args:
        margin (float): Extra margin above POC. Default=0.0.
            close > POC * (1 + margin) triggers signal.
        window_days (int): VP lookback window in days. Default=7.

    Required columns: vp_poc_{window_days}d
    """
    def __init__(self, margin: float = 0.0, window_days: int = 7):
        self.margin = margin
        self.window_days = window_days
        self.col = f"vp_poc_{window_days}d"
        m_str = f"+{margin*100:.0f}%" if margin > 0 else ""
        self.name = f"VP>POC{m_str}({window_days}d)"

    def __call__(self, row: pd.Series) -> ConditionResult:
        poc = row.get(self.col, np.nan)
        close = row.get("close", np.nan)
        if pd.isna(poc) or pd.isna(close):
            return False
        return self.name if close > poc * (1 + self.margin) else False

    def __repr__(self):
        return f"VPAbovePOCCond(margin={self.margin}, window_days={self.window_days})"


class VPBelowPOCCond:
    """
    Signal when price is below VP Point of Control.

    Args:
        margin (float): Extra margin below POC. Default=0.0.
            close < POC * (1 - margin) triggers signal.
        window_days (int): VP lookback window in days. Default=7.

    Required columns: vp_poc_{window_days}d
    """
    def __init__(self, margin: float = 0.0, window_days: int = 7):
        self.margin = margin
        self.window_days = window_days
        self.col = f"vp_poc_{window_days}d"
        m_str = f"-{margin*100:.0f}%" if margin > 0 else ""
        self.name = f"VP<POC{m_str}({window_days}d)"

    def __call__(self, row: pd.Series) -> ConditionResult:
        poc = row.get(self.col, np.nan)
        close = row.get("close", np.nan)
        if pd.isna(poc) or pd.isna(close):
            return False
        return self.name if close < poc * (1 - self.margin) else False

    def __repr__(self):
        return f"VPBelowPOCCond(margin={self.margin}, window_days={self.window_days})"


class VPBelowVALCond:
    """
    Buy/Sell when price is below Value Area Low (VP support break).

    Args:
        margin (float): Extra margin below VAL. Default=0.0.
            close < VAL * (1 - margin) triggers signal.
        window_days (int): VP lookback window in days. Default=7.

    Required columns: vp_val_{window_days}d
    """
    def __init__(self, margin: float = 0.0, window_days: int = 7):
        self.margin = margin
        self.window_days = window_days
        self.col = f"vp_val_{window_days}d"
        m_str = f"-{margin*100:.0f}%" if margin > 0 else ""
        self.name = f"VP<VAL{m_str}({window_days}d)"

    def __call__(self, row: pd.Series) -> ConditionResult:
        val = row.get(self.col, np.nan)
        close = row.get("close", np.nan)
        if pd.isna(val) or pd.isna(close):
            return False
        return self.name if close < val * (1 - self.margin) else False

    def __repr__(self):
        return f"VPBelowVALCond(margin={self.margin}, window_days={self.window_days})"


class VPAboveVAHCond:
    """
    Buy/Sell when price is above Value Area High (VP resistance break).

    Args:
        margin (float): Extra margin above VAH. Default=0.0.
        window_days (int): VP lookback window in days. Default=7.

    Required columns: vp_vah_{window_days}d
    """
    def __init__(self, margin: float = 0.0, window_days: int = 7):
        self.margin = margin
        self.window_days = window_days
        self.col = f"vp_vah_{window_days}d"
        m_str = f"+{margin*100:.0f}%" if margin > 0 else ""
        self.name = f"VP>VAH{m_str}({window_days}d)"

    def __call__(self, row: pd.Series) -> ConditionResult:
        vah = row.get(self.col, np.nan)
        close = row.get("close", np.nan)
        if pd.isna(vah) or pd.isna(close):
            return False
        return self.name if close > vah * (1 + self.margin) else False

    def __repr__(self):
        return f"VPAboveVAHCond(margin={self.margin}, window_days={self.window_days})"


# ============================================================================
# SELL CONDITIONS
# ============================================================================

class FearGreedSellCond:
    """
    Sell when Fear & Greed Index is above threshold (extreme greed).

    Args:
        threshold (int): Sell when F&G > threshold. Default=92.
            - 85: Broad greed (more exits, captures smaller tops)
            - 90: Standard
            - 92: Strict (only extreme euphoria, proven in backtest)

    Required columns: fear_greed
    """
    def __init__(self, threshold: int = 92):
        self.threshold = threshold
        self.name = f"FG>{threshold}"

    def __call__(self, row: pd.Series) -> ConditionResult:
        fg = row.get("fear_greed", np.nan)
        if pd.isna(fg):
            return False
        return self.name if fg > self.threshold else False

    def __repr__(self):
        return f"FearGreedSellCond(threshold={self.threshold})"


class RSISellCond:
    """
    Sell when RSI-14 is above threshold (overbought).

    Args:
        threshold (float): Sell when RSI > threshold. Default=65.
            - 60: Relaxed (earlier exits)
            - 65: Standard (works well with F&G sell)
            - 70: Strict (only when strongly overbought)

    Required columns: rsi_14 (or rsi_{period})
    """
    def __init__(self, threshold: float = 65.0, period: int = 14):
        self.threshold = threshold
        self.period = period
        self.col = f"rsi_{period}"
        self.name = f"RSI>{threshold:.0f}"

    def __call__(self, row: pd.Series) -> ConditionResult:
        rsi = row.get(self.col, np.nan)
        if pd.isna(rsi):
            return False
        return self.name if rsi > self.threshold else False

    def __repr__(self):
        return f"RSISellCond(threshold={self.threshold}, period={self.period})"


class FundingSellCond:
    """
    Sell when Funding Rate SMA7 exceeds threshold (longs overleveraged).

    Args:
        threshold (float): Sell when funding_sma7 > threshold. Default=0.0003 (0.03%).
            - 0.0002: Earlier exit
            - 0.0003: Standard (proven in Funding MR strategy)
            - 0.0005: Late exit (more patient)

    Required columns: funding_sma7
    """
    def __init__(self, threshold: float = 0.0003):
        self.threshold = threshold
        self.name = f"Fund>{threshold*100:.2f}%"

    def __call__(self, row: pd.Series) -> ConditionResult:
        f = row.get("funding_sma7", np.nan)
        if pd.isna(f):
            return False
        return self.name if f > self.threshold else False

    def __repr__(self):
        return f"FundingSellCond(threshold={self.threshold})"


class OISurgeSellCond:
    """
    Sell when Open Interest surges significantly (new leverage piling in).

    Args:
        threshold (float): Sell when oi_pct_7d > threshold. Default=0.20 (+20%).

    Required columns: oi_pct_7d, oi_valid
    """
    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold
        self.name = f"OI>{threshold*100:.0f}%"

    def __call__(self, row: pd.Series) -> ConditionResult:
        oi = row.get("oi_pct_7d", np.nan)
        valid = row.get("oi_valid", False)
        if not valid or pd.isna(oi):
            return False
        return self.name if oi > self.threshold else False

    def __repr__(self):
        return f"OISurgeSellCond(threshold={self.threshold})"


class CBPremiumSellCond:
    """
    Sell when Coinbase Premium SMA7 exceeds threshold (US FOMO buying).

    Args:
        threshold (float): Sell when cb_prem_sma7 > threshold. Default=1.0%.
            - 0.5: Early exit on mild US buying
            - 1.0: Standard (P90 of distribution)
            - 1.5: Tight (P95, extreme FOMO only)

    Required columns: cb_prem_sma7
    """
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.name = f"CB>{threshold:.1f}%"

    def __call__(self, row: pd.Series) -> ConditionResult:
        p = row.get("cb_prem_sma7", np.nan)
        if pd.isna(p):
            return False
        return self.name if p > self.threshold else False

    def __repr__(self):
        return f"CBPremiumSellCond(threshold={self.threshold})"


# ============================================================================
# CLOSE CONDITIONS (Position Management / Risk)
# ============================================================================

class StopLossCond:
    """
    Close position if unrealized loss exceeds threshold.

    Args:
        threshold (float): Close if loss > threshold from entry. Default=-0.20 (-20%).

    Required: entry price tracked by the engine (passed via row context)
    Note: This is evaluated by the engine, not from row data directly.
          Use with engine's close_conditions parameter.
    """
    def __init__(self, threshold: float = -0.20):
        self.threshold = threshold
        self.name = f"SL{threshold*100:.0f}%"

    def check(self, current_price: float, entry_price: float) -> ConditionResult:
        pnl = (current_price - entry_price) / entry_price
        return self.name if pnl < self.threshold else False

    def __repr__(self):
        return f"StopLossCond(threshold={self.threshold})"


class TakeProfitCond:
    """
    Close position if unrealized profit exceeds threshold.

    Args:
        threshold (float): Close if profit > threshold from entry. Default=3.0 (300%).

    Required: entry price tracked by the engine
    """
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.name = f"TP{threshold*100:.0f}%"

    def check(self, current_price: float, entry_price: float) -> ConditionResult:
        pnl = (current_price - entry_price) / entry_price
        return self.name if pnl > self.threshold else False

    def __repr__(self):
        return f"TakeProfitCond(threshold={self.threshold})"


class TimeExitCond:
    """
    Close position after holding for max_days, but only if profitable.

    Args:
        max_days (int): Max holding period. Default=365.
        min_profit (float): Min profit required for time exit. Default=0.20 (20%).
    """
    def __init__(self, max_days: int = 365, min_profit: float = 0.20):
        self.max_days = max_days
        self.min_profit = min_profit
        self.name = f"Time{max_days}d"

    def check(self, current_price: float, entry_price: float, days_held: int) -> ConditionResult:
        if days_held <= self.max_days:
            return False
        pnl = (current_price - entry_price) / entry_price
        return self.name if pnl > self.min_profit else False

    def __repr__(self):
        return f"TimeExitCond(max_days={self.max_days}, min_profit={self.min_profit})"


class TrailingStopCond:
    """
    Close position if price drops from its peak since entry.

    Args:
        threshold (float): Close if price drops threshold from peak. Default=-0.25 (-25%).

    Required: peak price tracked by the engine
    """
    def __init__(self, threshold: float = -0.25):
        self.threshold = threshold
        self.name = f"Trail{threshold*100:.0f}%"

    def check(self, current_price: float, peak_price: float) -> ConditionResult:
        drop = (current_price - peak_price) / peak_price
        return self.name if drop < self.threshold else False

    def __repr__(self):
        return f"TrailingStopCond(threshold={self.threshold})"


class SRStopLossLong:
    """
    S/R dynamic stop-loss for LONG positions.
    Stop = nearest_support * (1 - buffer), clamped to [entry*(1-max_loss), entry*(1-min_loss)].

    Args:
        buffer_pct (float): Buffer below support level. Default=0.003 (0.3%).
        max_loss (float): Maximum allowed loss from entry. Default=0.03 (3%).
        min_loss (float): Minimum loss threshold (don't stop too tight). Default=0.01 (1%).

    Required columns in row: nearest_support, close
    """
    def __init__(self, buffer_pct: float = 0.003, max_loss: float = 0.03,
                 min_loss: float = 0.01):
        self.buffer_pct = buffer_pct
        self.max_loss = max_loss
        self.min_loss = min_loss
        self.name = f"SR_SL_L(b{buffer_pct*1000:.0f}_x{max_loss*100:.0f})"

    def check(self, current_price: float, entry_price: float, row) -> ConditionResult:
        support = row.get("nearest_support", np.nan)
        close = row.get("close", current_price)
        if pd.isna(support):
            return False
        stop = support * (1 - self.buffer_pct)
        # Clamp: at most max_loss, at least min_loss space
        stop = max(stop, entry_price * (1 - self.max_loss))
        stop = min(stop, entry_price * (1 - self.min_loss))
        return self.name if close < stop else False

    def __repr__(self):
        return (f"SRStopLossLong(buffer_pct={self.buffer_pct}, "
                f"max_loss={self.max_loss}, min_loss={self.min_loss})")


class SRStopLossShort:
    """
    S/R dynamic stop-loss for SHORT positions.
    Stop = nearest_resistance * (1 + buffer), clamped to [entry*(1+min_loss), entry*(1+max_loss)].

    Args:
        buffer_pct (float): Buffer above resistance level. Default=0.003 (0.3%).
        max_loss (float): Maximum allowed loss from entry. Default=0.03 (3%).
        min_loss (float): Minimum loss threshold. Default=0.01 (1%).

    Required columns in row: nearest_resistance, close
    """
    def __init__(self, buffer_pct: float = 0.003, max_loss: float = 0.03,
                 min_loss: float = 0.01):
        self.buffer_pct = buffer_pct
        self.max_loss = max_loss
        self.min_loss = min_loss
        self.name = f"SR_SL_S(b{buffer_pct*1000:.0f}_x{max_loss*100:.0f})"

    def check(self, current_price: float, entry_price: float, row) -> ConditionResult:
        resistance = row.get("nearest_resistance", np.nan)
        close = row.get("close", current_price)
        if pd.isna(resistance):
            return False
        stop = resistance * (1 + self.buffer_pct)
        # Clamp: at most max_loss, at least min_loss space
        stop = min(stop, entry_price * (1 + self.max_loss))
        stop = max(stop, entry_price * (1 + self.min_loss))
        return self.name if close > stop else False

    def __repr__(self):
        return (f"SRStopLossShort(buffer_pct={self.buffer_pct}, "
                f"max_loss={self.max_loss}, min_loss={self.min_loss})")


class ATRStopLossCond:
    """
    ATR-based volatility stop-loss. Stop distance = multiplier * ATR.
    Works for both long and short via engine's PnL price transformation.

    Args:
        multiplier (float): ATR multiplier for stop distance. Default=2.0.

    Required columns in row: atr
    For long: triggers when current_price < entry - multiplier * atr
    For short: engine passes short_pnl_price, equivalent math applies.
    """
    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier
        self.name = f"ATR{multiplier:.1f}x"

    def check(self, current_price: float, entry_price: float, row) -> ConditionResult:
        atr = row.get("atr", 0)
        if pd.isna(atr) or atr <= 0:
            return False
        stop = entry_price - self.multiplier * atr
        return self.name if current_price < stop else False

    def __repr__(self):
        return f"ATRStopLossCond(multiplier={self.multiplier})"


class TimeBarStopCond:
    """
    Time-based stop: exit if held > max_bars and position is not profitable.

    Args:
        max_bars (int): Maximum bars to hold before checking. Default=48.
        bars_per_day (int): Bars per calendar day (24 for 1h). Default=24.

    Uses days_held dispatch (no row needed).
    """
    def __init__(self, max_bars: int = 48, bars_per_day: int = 24):
        self.max_bars = max_bars
        self.bars_per_day = bars_per_day
        self.name = f"Time{max_bars}b"

    def check(self, current_price: float, entry_price: float,
              days_held: float) -> ConditionResult:
        bars_held = days_held * self.bars_per_day
        if bars_held < self.max_bars:
            return False
        pnl = (current_price - entry_price) / entry_price
        return self.name if pnl <= 0 else False

    def __repr__(self):
        return f"TimeBarStopCond(max_bars={self.max_bars})"


# ============================================================================
# COMBINATORS
# ============================================================================

def combine_conditions(*conditions: ConditionFn, mode: str = "AND") -> ConditionFn:
    """
    Combine multiple conditions into one using AND or OR logic.

    Args:
        *conditions: Condition instances (FearGreedBuyCond, RSIBuyCond, etc.)
        mode: "AND" = all must pass  |  "OR" = any can pass

    Returns:
        A combined condition function with the same signature.

    Examples:
        # Buy only when BOTH F&G < 10 AND RSI < 35:
        buy = combine_conditions(FearGreedBuyCond(10), RSIBuyCond(35), mode="AND")

        # Buy when EITHER F&G < 10 OR Funding < -0.01%:
        buy = combine_conditions(FearGreedBuyCond(10), FundingBuyCond(-0.0001), mode="OR")
    """
    names = [c.name if hasattr(c, 'name') else str(c) for c in conditions]
    joiner = "+" if mode == "AND" else "|"
    combined_name = joiner.join(names)

    if mode == "AND":
        def _and_fn(row):
            results = []
            for cond in conditions:
                r = cond(row)
                if not r:
                    return False
                results.append(r if isinstance(r, str) else "")
            return combined_name
        _and_fn.name = combined_name
        _and_fn.__repr__ = lambda: f"AND({', '.join(repr(c) for c in conditions)})"
        return _and_fn
    else:  # OR
        def _or_fn(row):
            for cond in conditions:
                r = cond(row)
                if r:
                    return r if isinstance(r, str) else cond.name if hasattr(cond, 'name') else "SIGNAL"
            return False
        _or_fn.name = combined_name
        _or_fn.__repr__ = lambda: f"OR({', '.join(repr(c) for c in conditions)})"
        return _or_fn


# ============================================================================
# PRESETS (Common proven strategies)
# ============================================================================

def preset_fg_rsi_buy(fg_thresh=10, rsi_thresh=35):
    """F&G + RSI buy (proven: +822% return). FG<10 AND RSI<35."""
    return combine_conditions(FearGreedBuyCond(fg_thresh), RSIBuyCond(rsi_thresh), mode="AND")

def preset_fg_rsi_sell(fg_thresh=92, rsi_thresh=65):
    """F&G + RSI sell. FG>92 AND RSI>65."""
    return combine_conditions(FearGreedSellCond(fg_thresh), RSISellCond(rsi_thresh), mode="AND")

def preset_funding_mr_buy(thresh=-0.0001):
    """Funding Mean-Reversion buy (proven: Sharpe 1.22). FundingSMA7 < -0.01%."""
    return FundingBuyCond(thresh)

def preset_funding_mr_sell(thresh=0.0003):
    """Funding Mean-Reversion sell. FundingSMA7 > 0.03%."""
    return FundingSellCond(thresh)

def preset_union_buy(fg_thresh=10, rsi_thresh=35, fund_thresh=-0.0001):
    """Union buy: F&G+RSI OR Funding (proven: +1523% with source-aware sell)."""
    return combine_conditions(
        combine_conditions(FearGreedBuyCond(fg_thresh), RSIBuyCond(rsi_thresh), mode="AND"),
        FundingBuyCond(fund_thresh),
        mode="OR"
    )


def preset_alpha_combo(rsi_buy=70, rsi_sell=30,
                       fg_buy=10, fg_rsi_buy=35, fg_sell=92, fg_rsi_sell=65,
                       fund_buy=-0.0001, fund_sell=0.0003):
    """
    Alpha Combo: Merges top-3 strategies into one config dict.

    Combines:
      1. Reverse RSI (30/70)  +1752% return, Sharpe 1.04, 32 trades
         -> Momentum: buy RSI>70 (entering uptrend), sell RSI<30 (trend exhaustion)
      2. F&G + RSI             +860% return, Sharpe 0.82, 4 trades
         -> Sentiment: buy on extreme fear (FG<10 & RSI<35), sell on euphoria
      3. Funding MR            +814% return, Sharpe 1.22, 3 trades
         -> Derivatives: buy when shorts dominate, sell when longs overleveraged

    Buy:  (RSI>70 momentum)  OR  (FG<10 AND RSI<35 sentiment)  OR  (FundingSMA7<-0.01%)
    Sell: Source-aware exit per entry signal type:
          - RSI momentum entry  -> sell when RSI<30
          - F&G+RSI entry       -> sell when FG>92 AND RSI>65
          - Funding entry       -> sell when FundingSMA7>0.03%

    Returns:
        dict with keys: name, buy_cond, sell_cond, source_aware_sell, sell_map
        Ready to append to strategies list in strategy_test.py.

    Usage:
        strategies.append(preset_alpha_combo())
        # Custom thresholds:
        strategies.append(preset_alpha_combo(rsi_buy=65, fund_buy=-0.0002))
    """
    # Entry conditions
    rsi_momentum_buy = RSISellCond(threshold=rsi_buy)   # RSI>70 = buy (momentum)
    fg_rsi_buy_cond = combine_conditions(
        FearGreedBuyCond(fg_buy), RSIBuyCond(fg_rsi_buy), mode="AND"
    )
    funding_buy_cond = FundingBuyCond(fund_buy)

    buy_cond = combine_conditions(
        rsi_momentum_buy, fg_rsi_buy_cond, funding_buy_cond, mode="OR"
    )

    # Exit conditions (source-aware)
    rsi_momentum_sell = RSIBuyCond(threshold=rsi_sell)   # RSI<30 = sell (momentum exit)
    fg_rsi_sell_cond = combine_conditions(
        FearGreedSellCond(fg_sell), RSISellCond(fg_rsi_sell), mode="AND"
    )
    funding_sell_cond = FundingSellCond(fund_sell)

    return {
        "name": "Alpha Combo",
        "buy_cond": buy_cond,
        "sell_cond": rsi_momentum_sell,  # default sell (most frequent entry is RSI)
        "source_aware_sell": True,
        "sell_map": {
            rsi_momentum_buy.name: rsi_momentum_sell,     # RSI>70 entry -> RSI<30 exit
            fg_rsi_buy_cond.name: fg_rsi_sell_cond,       # FG+RSI entry -> FG+RSI exit
            funding_buy_cond.name: funding_sell_cond,     # Funding entry -> Funding exit
        },
    }
