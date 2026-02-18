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
        2. norm_velocity < 0               (was falling)
        3. velocity_delta > min_decel      (falling speed decelerating / reversing)

    Args:
        proximity_pct: Max distance to support (fraction). Default=0.01.
        min_decel: Min velocity_delta for reversal signal. Default=0.5.
    """
    def __init__(self, proximity_pct: float = 0.01, min_decel: float = 0.5):
        self.proximity_pct = proximity_pct
        self.min_decel = min_decel
        self.name = (f"RevL(px{proximity_pct*100:.1f}%,"
                     f"dc>{min_decel:.1f})")

    def __call__(self, row: pd.Series) -> ConditionResult:
        dist_sup = row.get("dist_to_support", np.nan)
        nv = row.get("norm_velocity", np.nan)
        vd = row.get("velocity_delta", np.nan)
        if pd.isna(dist_sup) or pd.isna(nv) or pd.isna(vd):
            return False
        if dist_sup < self.proximity_pct and nv < 0 and vd > self.min_decel:
            return self.name
        return False

    def __repr__(self):
        return (f"ReversalLongCond(proximity_pct={self.proximity_pct}, "
                f"min_decel={self.min_decel})")


class ReversalShortCond:
    """
    Mean-reversion short: price near resistance + upward momentum reversing.

    Triggers when:
        1. dist_to_resistance < proximity_pct  (price near resistance)
        2. norm_velocity > 0                   (was rising)
        3. velocity_delta < -min_decel         (rising speed decelerating / reversing)

    Args:
        proximity_pct: Max distance to resistance (fraction). Default=0.01.
        min_decel: Min |velocity_delta| for reversal signal. Default=0.5.
    """
    def __init__(self, proximity_pct: float = 0.01, min_decel: float = 0.5):
        self.proximity_pct = proximity_pct
        self.min_decel = min_decel
        self.name = (f"RevS(px{proximity_pct*100:.1f}%,"
                     f"dc>{min_decel:.1f})")

    def __call__(self, row: pd.Series) -> ConditionResult:
        dist_res = row.get("dist_to_resistance", np.nan)
        nv = row.get("norm_velocity", np.nan)
        vd = row.get("velocity_delta", np.nan)
        if pd.isna(dist_res) or pd.isna(nv) or pd.isna(vd):
            return False
        if dist_res < self.proximity_pct and nv > 0 and vd < -self.min_decel:
            return self.name
        return False

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
