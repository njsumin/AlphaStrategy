"""
S/R Momentum Signals (sr_signals.py)
=====================================
Momentum indicators and entry/exit condition classes for the
VP S/R Level Strategy (momentum + mean-reversion variants).

Usage:
------
    from sr_signals import (
        add_momentum_indicators,
        MomentumLongCond, MomentumShortCond,
        ReversalLongCond, ReversalShortCond,
        NearResistanceCond, NearSupportCond,
    )
"""

import numpy as np
import pandas as pd
from typing import Union

ConditionResult = Union[str, bool]


def add_momentum_indicators(df: pd.DataFrame,
                            velocity_bars: int = 4,
                            atr_period: int = 14) -> pd.DataFrame:
    """
    Add momentum indicators for S/R strategy.

    Args:
        df: DataFrame with 'close', 'open' (and optionally 'high', 'low') columns.
        velocity_bars: Number of bars for velocity calculation.
        atr_period: ATR lookback period in bars.

    Adds columns:
        price_velocity: (close - close[N ago]) / close[N ago]
        atr: Average True Range over atr_period bars
        norm_velocity: price_velocity / (atr / close) — ATR-normalized speed
        velocity_delta: change in norm_velocity over velocity_bars
    """
    close = df["close"]

    # Price velocity: N-bar return
    df["price_velocity"] = close.pct_change(periods=velocity_bars)

    # ATR calculation
    if "high" in df.columns and "low" in df.columns:
        high = df["high"]
        low = df["low"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
    else:
        # Fallback: use absolute close-to-close change as proxy
        tr = close.diff().abs()

    df["atr"] = tr.rolling(atr_period).mean()

    # Normalized velocity: velocity / (atr/close)
    # > 1 means price moved faster than 1 ATR in velocity_bars
    atr_pct = df["atr"] / close
    df["norm_velocity"] = np.where(
        atr_pct > 0,
        df["price_velocity"] / atr_pct,
        0.0
    )

    # Velocity delta: change in norm_velocity over lookback
    # Positive = accelerating / decelerating less, Negative = decelerating
    df["velocity_delta"] = df["norm_velocity"] - df["norm_velocity"].shift(velocity_bars)

    return df


# ============================================================================
# ENTRY CONDITIONS — Momentum (V1)
# ============================================================================

class MomentumLongCond:
    """
    Long entry: price moving up fast with enough room to resistance.

    Triggers when:
        norm_velocity > min_norm_velocity AND dist_to_resistance > min_profit_pct

    Args:
        min_norm_velocity: Minimum ATR-normalized velocity. Default=1.5.
        min_profit_pct: Minimum distance to resistance (fraction). Default=0.005.
    """
    def __init__(self, min_norm_velocity: float = 1.5,
                 min_profit_pct: float = 0.005):
        self.min_norm_velocity = min_norm_velocity
        self.min_profit_pct = min_profit_pct
        self.name = f"MomL(v>{min_norm_velocity:.1f},p>{min_profit_pct*100:.1f}%)"

    def __call__(self, row: pd.Series) -> ConditionResult:
        nv = row.get("norm_velocity", np.nan)
        dist = row.get("dist_to_resistance", np.nan)
        if pd.isna(nv) or pd.isna(dist):
            return False
        if nv > self.min_norm_velocity and dist > self.min_profit_pct:
            return self.name
        return False

    def __repr__(self):
        return (f"MomentumLongCond(min_norm_velocity={self.min_norm_velocity}, "
                f"min_profit_pct={self.min_profit_pct})")


class MomentumShortCond:
    """
    Short entry: price moving down fast with enough room to support.

    Triggers when:
        norm_velocity < -min_norm_velocity AND dist_to_support > min_profit_pct

    Args:
        min_norm_velocity: Minimum ATR-normalized velocity (absolute). Default=1.5.
        min_profit_pct: Minimum distance to support (fraction). Default=0.005.
    """
    def __init__(self, min_norm_velocity: float = 1.5,
                 min_profit_pct: float = 0.005):
        self.min_norm_velocity = min_norm_velocity
        self.min_profit_pct = min_profit_pct
        self.name = f"MomS(v>{min_norm_velocity:.1f},p>{min_profit_pct*100:.1f}%)"

    def __call__(self, row: pd.Series) -> ConditionResult:
        nv = row.get("norm_velocity", np.nan)
        dist = row.get("dist_to_support", np.nan)
        if pd.isna(nv) or pd.isna(dist):
            return False
        if nv < -self.min_norm_velocity and dist > self.min_profit_pct:
            return self.name
        return False

    def __repr__(self):
        return (f"MomentumShortCond(min_norm_velocity={self.min_norm_velocity}, "
                f"min_profit_pct={self.min_profit_pct})")


# ============================================================================
# ENTRY CONDITIONS — Mean Reversion (V2)
# ============================================================================

class ReversalLongCond:
    """
    Mean-reversion long: price near support + downward momentum reversing.

    Triggers when:
        1. dist_to_support < proximity_pct  (price near support)
        2. norm_velocity < velocity_ceil     (default 0: was falling)
        3. velocity_delta > min_decel        (falling speed decelerating / reversing)

    Optional filters (None = disabled):
        4. trend_sma: require close > sma_{trend_sma} (trade with trend)
        5. rsi_max: require rsi_14 < rsi_max (oversold confirmation)
        6. sr_count_min: require sr_count >= sr_count_min (S/R density)
        7. max_abs_velocity: require norm_velocity > -max_abs_velocity
           (filter false reversals where price still falling fast)

    Args:
        proximity_pct: Max distance to support (fraction). Default=0.01.
        min_decel: Min velocity_delta for reversal signal. Default=0.5.
        velocity_ceil: Upper bound for norm_velocity. Default=0 (must be falling).
            Set to positive value (e.g. 0.5) to allow entries where price
            has already turned up slightly.
        velocity_floor: Lower bound for norm_velocity. Default=None (no floor).
            Set to e.g. -2.0 to filter out extreme crashes.
            Replaces max_abs_velocity with a more intuitive API.
        trend_sma: SMA period for trend filter (e.g. 200). None=disabled.
        rsi_max: Max RSI-14 for oversold confirmation. None=disabled.
        sr_count_min: Min S/R cluster count. None=disabled.
        max_abs_velocity: Deprecated, use velocity_floor=-X instead. None=disabled.
    """
    def __init__(self, proximity_pct: float = 0.01, min_decel: float = 0.5,
                 velocity_ceil: float = 0.0, velocity_floor: float = None,
                 trend_sma: int = None, rsi_max: float = None,
                 sr_count_min: int = None, max_abs_velocity: float = None):
        self.proximity_pct = proximity_pct
        self.min_decel = min_decel
        self.velocity_ceil = velocity_ceil
        # velocity_floor takes precedence; max_abs_velocity for backwards compat
        if velocity_floor is not None:
            self.velocity_floor = velocity_floor
        elif max_abs_velocity is not None:
            self.velocity_floor = -max_abs_velocity
        else:
            self.velocity_floor = None
        self.trend_sma = trend_sma
        self.rsi_max = rsi_max
        self.sr_count_min = sr_count_min
        self.max_abs_velocity = max_abs_velocity
        self.name = (f"RevL(px{proximity_pct*100:.1f}%,"
                     f"dc>{min_decel:.1f})")

    def __call__(self, row: pd.Series) -> ConditionResult:
        dist_sup = row.get("dist_to_support", np.nan)
        nv = row.get("norm_velocity", np.nan)
        vd = row.get("velocity_delta", np.nan)
        if pd.isna(dist_sup) or pd.isna(nv) or pd.isna(vd):
            return False
        if not (dist_sup < self.proximity_pct
                and nv < self.velocity_ceil
                and vd > self.min_decel):
            return False
        # Optional filters
        if self.trend_sma is not None:
            sma = row.get(f"sma_{self.trend_sma}", np.nan)
            close = row.get("close", np.nan)
            if pd.isna(sma) or pd.isna(close) or close <= sma:
                return False
        if self.rsi_max is not None:
            rsi = row.get("rsi_14", np.nan)
            if pd.isna(rsi) or rsi >= self.rsi_max:
                return False
        if self.sr_count_min is not None:
            sr_count = row.get("sr_count", 0)
            if sr_count < self.sr_count_min:
                return False
        if self.velocity_floor is not None:
            if nv < self.velocity_floor:
                return False
        return self.name

    def __repr__(self):
        return (f"ReversalLongCond(proximity_pct={self.proximity_pct}, "
                f"min_decel={self.min_decel})")


class ReversalShortCond:
    """
    Mean-reversion short: price near resistance + upward momentum reversing.

    Triggers when:
        1. dist_to_resistance < proximity_pct  (price near resistance)
        2. norm_velocity > velocity_floor       (default 0: was rising)
        3. velocity_delta < -min_decel          (rising speed decelerating / reversing)

    Optional filters (None = disabled):
        4. trend_sma: require close < sma_{trend_sma} (trade with trend)
        5. rsi_min: require rsi_14 > rsi_min (overbought confirmation)
        6. sr_count_min: require sr_count >= sr_count_min (S/R density)
        7. max_abs_velocity: require norm_velocity < max_abs_velocity
           (filter false reversals where price still rising fast)

    Args:
        proximity_pct: Max distance to resistance (fraction). Default=0.01.
        min_decel: Min |velocity_delta| for reversal signal. Default=0.5.
        velocity_floor: Lower bound for norm_velocity. Default=0 (must be rising).
            Set to negative value (e.g. -0.5) to allow entries where price
            has already turned down slightly.
        velocity_ceil: Upper bound for norm_velocity. Default=None (no ceiling).
            Set to e.g. 2.0 to filter out extreme rallies.
            Replaces max_abs_velocity with a more intuitive API.
        trend_sma: SMA period for trend filter (e.g. 200). None=disabled.
        rsi_min: Min RSI-14 for overbought confirmation. None=disabled.
        sr_count_min: Min S/R cluster count. None=disabled.
        max_abs_velocity: Deprecated, use velocity_ceil=X instead. None=disabled.
    """
    def __init__(self, proximity_pct: float = 0.01, min_decel: float = 0.5,
                 velocity_floor: float = 0.0, velocity_ceil: float = None,
                 trend_sma: int = None, rsi_min: float = None,
                 sr_count_min: int = None, max_abs_velocity: float = None):
        self.proximity_pct = proximity_pct
        self.min_decel = min_decel
        self.velocity_floor = velocity_floor
        # velocity_ceil takes precedence; max_abs_velocity for backwards compat
        if velocity_ceil is not None:
            self.velocity_ceil = velocity_ceil
        elif max_abs_velocity is not None:
            self.velocity_ceil = max_abs_velocity
        else:
            self.velocity_ceil = None
        self.trend_sma = trend_sma
        self.rsi_min = rsi_min
        self.sr_count_min = sr_count_min
        self.max_abs_velocity = max_abs_velocity
        self.name = (f"RevS(px{proximity_pct*100:.1f}%,"
                     f"dc>{min_decel:.1f})")

    def __call__(self, row: pd.Series) -> ConditionResult:
        dist_res = row.get("dist_to_resistance", np.nan)
        nv = row.get("norm_velocity", np.nan)
        vd = row.get("velocity_delta", np.nan)
        if pd.isna(dist_res) or pd.isna(nv) or pd.isna(vd):
            return False
        if not (dist_res < self.proximity_pct
                and nv > self.velocity_floor
                and vd < -self.min_decel):
            return False
        # Optional filters
        if self.trend_sma is not None:
            sma = row.get(f"sma_{self.trend_sma}", np.nan)
            close = row.get("close", np.nan)
            if pd.isna(sma) or pd.isna(close) or close >= sma:
                return False
        if self.rsi_min is not None:
            rsi = row.get("rsi_14", np.nan)
            if pd.isna(rsi) or rsi <= self.rsi_min:
                return False
        if self.sr_count_min is not None:
            sr_count = row.get("sr_count", 0)
            if sr_count < self.sr_count_min:
                return False
        if self.velocity_ceil is not None:
            if nv > self.velocity_ceil:
                return False
        return self.name

    def __repr__(self):
        return (f"ReversalShortCond(proximity_pct={self.proximity_pct}, "
                f"min_decel={self.min_decel})")


# ============================================================================
# EXIT CONDITIONS
# ============================================================================

class NearResistanceCond:
    """
    Sell/take-profit when close approaches nearest resistance.

    Triggers when: close >= nearest_resistance * (1 - proximity_pct)

    Args:
        proximity_pct: Fraction proximity to trigger exit. Default=0.002 (0.2%).
    """
    def __init__(self, proximity_pct: float = 0.002):
        self.proximity_pct = proximity_pct
        self.name = f"NearRes({proximity_pct*100:.1f}%)"

    def __call__(self, row: pd.Series) -> ConditionResult:
        close = row.get("close", np.nan)
        res = row.get("nearest_resistance", np.nan)
        if pd.isna(close) or pd.isna(res):
            return False
        if close >= res * (1 - self.proximity_pct):
            return self.name
        return False

    def __repr__(self):
        return f"NearResistanceCond(proximity_pct={self.proximity_pct})"


class NearSupportCond:
    """
    Cover/take-profit when close approaches nearest support.

    Triggers when: close <= nearest_support * (1 + proximity_pct)

    Args:
        proximity_pct: Fraction proximity to trigger exit. Default=0.002 (0.2%).
    """
    def __init__(self, proximity_pct: float = 0.002):
        self.proximity_pct = proximity_pct
        self.name = f"NearSup({proximity_pct*100:.1f}%)"

    def __call__(self, row: pd.Series) -> ConditionResult:
        close = row.get("close", np.nan)
        sup = row.get("nearest_support", np.nan)
        if pd.isna(close) or pd.isna(sup):
            return False
        if close <= sup * (1 + self.proximity_pct):
            return self.name
        return False

    def __repr__(self):
        return f"NearSupportCond(proximity_pct={self.proximity_pct})"
