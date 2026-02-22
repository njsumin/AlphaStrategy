"""
S/R Momentum Signals (sr_signals.py)
=====================================
Momentum indicators and entry/exit condition classes for the
VP S/R Level Strategy (momentum + mean-reversion variants).

Usage:
------
    from sr_signals import (
        add_momentum_indicators,
        add_volume_momentum_indicators,
        MomentumLongCond, MomentumShortCond,
        ReversalLongCond, ReversalShortCond,
        VolumeReversalLongCond, VolumeReversalShortCond,
        NearResistanceCond, NearSupportCond,
    )
"""

import numpy as np
import pandas as pd
from typing import Union

ConditionResult = Union[str, bool]


def add_momentum_indicators(df: pd.DataFrame,
                            velocity_bars: int = 4,
                            atr_period: int = 14,
                            fast_velocity_bars: int = 2) -> pd.DataFrame:
    """
    Add momentum indicators for S/R strategy.

    Args:
        df: DataFrame with 'close', 'open' (and optionally 'high', 'low') columns.
        velocity_bars: Number of bars for velocity calculation.
        atr_period: ATR lookback period in bars.
        fast_velocity_bars: Number of bars for fast velocity calculation.

    Adds columns:
        price_velocity: (close - close[N ago]) / close[N ago]
        atr: Average True Range over atr_period bars
        norm_velocity: price_velocity / (atr / close) — ATR-normalized speed
        velocity_delta: change in norm_velocity over velocity_bars
        fast_price_velocity: fast N-bar return
        fast_norm_velocity: ATR-normalized fast velocity
        fast_velocity_delta: change in fast_norm_velocity over fast_velocity_bars
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

    # Fast velocity: short-period momentum (reuses same ATR)
    df["fast_price_velocity"] = close.pct_change(periods=fast_velocity_bars)
    df["fast_norm_velocity"] = np.where(
        atr_pct > 0,
        df["fast_price_velocity"] / atr_pct,
        0.0
    )
    df["fast_velocity_delta"] = (
        df["fast_norm_velocity"] - df["fast_norm_velocity"].shift(fast_velocity_bars)
    )

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
        8. min_reward_pct: require dist_to_resistance >= min_reward_pct
           (filter entries with insufficient upside profit space)
        9. min_w_strength: require w_bottom_strength >= threshold
           (15m W-bottom microstructure confirmation)

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
        min_reward_pct: Min distance to resistance (fraction). None=disabled.
        min_w_strength: Min W-bottom strength for 15m confirmation. None=disabled.
    """
    def __init__(self, proximity_pct: float = 0.01, min_decel: float = 0.5,
                 velocity_ceil: float = 0.0, velocity_floor: float = None,
                 trend_sma: int = None, rsi_max: float = None,
                 sr_count_min: int = None, max_abs_velocity: float = None,
                 fast_velocity_ceil: float = None,
                 fast_velocity_floor: float = None,
                 fast_min_decel: float = None,
                 min_reward_pct: float = None,
                 min_w_strength: float = None):
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
        self.fast_velocity_ceil = fast_velocity_ceil
        self.fast_velocity_floor = fast_velocity_floor
        self.fast_min_decel = fast_min_decel
        self.min_reward_pct = min_reward_pct
        self.min_w_strength = min_w_strength
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
        # Fast velocity filters
        if self.fast_velocity_ceil is not None:
            fnv = row.get("fast_norm_velocity", np.nan)
            if pd.isna(fnv) or fnv > self.fast_velocity_ceil:
                return False
        if self.fast_velocity_floor is not None:
            fnv = row.get("fast_norm_velocity", np.nan)
            if pd.isna(fnv) or fnv < self.fast_velocity_floor:
                return False
        if self.fast_min_decel is not None:
            fvd = row.get("fast_velocity_delta", np.nan)
            if pd.isna(fvd) or fvd < self.fast_min_decel:
                return False
        # Reward filter: ensure enough upside to nearest resistance
        if self.min_reward_pct is not None:
            dist_res = row.get("dist_to_resistance", np.nan)
            # NaN = no resistance found → allow (conservative)
            if not pd.isna(dist_res) and dist_res < self.min_reward_pct:
                return False
        # W-bottom microstructure filter (15m)
        if self.min_w_strength is not None:
            ws = row.get("w_bottom_strength", 0)
            if pd.isna(ws) or ws < self.min_w_strength:
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
        8. min_reward_pct: require dist_to_support >= min_reward_pct
           (filter entries with insufficient downside profit space)

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
        min_reward_pct: Min distance to support (fraction). None=disabled.
    """
    def __init__(self, proximity_pct: float = 0.01, min_decel: float = 0.5,
                 velocity_floor: float = 0.0, velocity_ceil: float = None,
                 trend_sma: int = None, rsi_min: float = None,
                 sr_count_min: int = None, max_abs_velocity: float = None,
                 fast_velocity_floor: float = None,
                 fast_velocity_ceil: float = None,
                 fast_min_decel: float = None,
                 min_reward_pct: float = None,
                 min_m_strength: float = None):
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
        self.fast_velocity_floor = fast_velocity_floor
        self.fast_velocity_ceil = fast_velocity_ceil
        self.fast_min_decel = fast_min_decel
        self.min_reward_pct = min_reward_pct
        self.min_m_strength = min_m_strength
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
        # Fast velocity filters
        if self.fast_velocity_floor is not None:
            fnv = row.get("fast_norm_velocity", np.nan)
            if pd.isna(fnv) or fnv < self.fast_velocity_floor:
                return False
        if self.fast_velocity_ceil is not None:
            fnv = row.get("fast_norm_velocity", np.nan)
            if pd.isna(fnv) or fnv > self.fast_velocity_ceil:
                return False
        if self.fast_min_decel is not None:
            fvd = row.get("fast_velocity_delta", np.nan)
            if pd.isna(fvd) or fvd > -self.fast_min_decel:
                return False
        # Reward filter: ensure enough downside to nearest support
        if self.min_reward_pct is not None:
            dist_sup = row.get("dist_to_support", np.nan)
            # NaN = no support found → allow (conservative)
            if not pd.isna(dist_sup) and dist_sup < self.min_reward_pct:
                return False
        # M-top microstructure filter (15m)
        if self.min_m_strength is not None:
            ms = row.get("m_top_strength", 0)
            if pd.isna(ms) or ms < self.min_m_strength:
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


# ============================================================================
# VOLUME-CLOCK MOMENTUM INDICATORS
# ============================================================================

def _compute_vol_lookback(volume_arr: np.ndarray,
                          target_vol_arr: np.ndarray,
                          max_lookback: int = 500) -> np.ndarray:
    """
    For each bar i, find lookback L such that sum(vol[i-L:i+1]) >= target_vol[i].

    Uses cumulative sum + binary search (vectorized, O(n log n)).
    Returns array of lookback lengths (integers >= 0).

    Design:
        c[k] = sum(vol[0:k])  (c[0]=0, c[1]=vol[0], ...)
        sum(vol[j:i+1]) = c[i+1] - c[j]
        Want largest j in [max(0, i+1-max_lookback), i] where c[j] <= c[i+1] - tgt
        → lookback L = i - j
    """
    n = len(volume_arr)
    # Prepend 0 so c[k] = cumulative volume including bars 0..k-1
    c = np.concatenate([[0.0], np.cumsum(volume_arr.astype(float))])

    # For each bar i: threshold = c[i+1] - target_vol[i]
    thresholds = c[1:] - np.maximum(0.0, target_vol_arr)

    # searchsorted on sorted c array: find insertion point for each threshold
    # → position p = first k where c[k] > threshold
    # → largest j where c[j] <= threshold = p - 1
    j_indices = np.searchsorted(c, thresholds, side="right") - 1

    # Clamp j to [max(0, i+1-max_lookback), i]
    i_arr = np.arange(n)
    j_min = np.maximum(0, i_arr + 1 - max_lookback)
    j_indices = np.clip(j_indices, j_min, i_arr)

    return i_arr - j_indices  # lookback lengths


def add_volume_momentum_indicators(df: pd.DataFrame,
                                   velocity_vol_mult: float = 8.0,
                                   fast_vol_mult: float = 2.0,
                                   atr_period: int = 14,
                                   vol_ref_period: int = 168,
                                   max_lookback: int = 500) -> pd.DataFrame:
    """
    Add volume-clock momentum indicators for S/R strategy.

    Instead of measuring price change over N fixed time bars, measures over a
    rolling volume window where cumulative volume reaches
    velocity_vol_mult × median(volume over vol_ref_period bars).

    This makes the 'clock' tick faster in high-volume periods (e.g., panic
    selling) and slower in quiet periods, so velocity captures market activity
    intensity rather than elapsed calendar time.

    Args:
        df: DataFrame with 'close', 'volume' (and optionally 'high', 'low').
        velocity_vol_mult: Target volume multiplier for velocity window.
            e.g. 8.0 ≈ 8× median hourly volume ≈ 8 h at average activity.
        fast_vol_mult: Target volume multiplier for fast velocity window.
        atr_period: ATR lookback in bars (time-based, for price normalisation).
        vol_ref_period: Bars for rolling median volume reference.
            Default 168 = 1 week of 1 h bars.
        max_lookback: Maximum bars to search backward for volume accumulation.

    Adds columns:
        vol_velocity          : price return over vol_window of cumulative volume
        vol_norm_velocity     : ATR-normalised volume velocity
        vol_velocity_delta    : change in vol_norm_velocity over velocity lookback
        fast_vol_velocity     : fast vol-clock velocity
        fast_vol_norm_velocity: fast vol-clock normalised velocity
        fast_vol_velocity_delta: fast vol-clock acceleration
        vol_lookback_bars     : actual time bars in velocity window (diagnostic)
        volume_ratio          : current volume / rolling mean volume
    """
    if "volume" not in df.columns:
        raise ValueError("add_volume_momentum_indicators requires a 'volume' column")

    close = df["close"].values
    volume = df["volume"].fillna(0).values
    n = len(df)

    # ── Reference volume: rolling median over vol_ref_period bars ────────────
    vol_series = df["volume"].fillna(0)
    ref_vol = vol_series.rolling(vol_ref_period, min_periods=1).median().values

    target_vol = velocity_vol_mult * ref_vol
    fast_target_vol = fast_vol_mult * ref_vol

    # ── ATR (time-based, for price-level normalisation) ───────────────────────
    if "high" in df.columns and "low" in df.columns:
        high = df["high"].values
        low = df["low"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - prev_close),
                                   np.abs(low - prev_close)))
    else:
        diffs = np.abs(np.diff(close, prepend=close[0]))
        tr = diffs

    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values
    atr_pct = np.where(close > 0, atr / close, 0.0)

    # ── Volume ratio: current bar vs rolling mean ─────────────────────────────
    vol_mean = vol_series.rolling(atr_period, min_periods=1).mean().values
    volume_ratio = np.where(vol_mean > 0, volume / vol_mean, 1.0)

    # ── Volume-clock lookbacks ────────────────────────────────────────────────
    vel_lookbacks = _compute_vol_lookback(volume, target_vol, max_lookback)
    fast_lookbacks = _compute_vol_lookback(volume, fast_target_vol, max_lookback)

    # ── Volume velocities (price return over volume window) ───────────────────
    i_arr = np.arange(n)
    j_arr = np.maximum(0, i_arr - vel_lookbacks)      # starting bars (velocity)
    fj_arr = np.maximum(0, i_arr - fast_lookbacks)    # starting bars (fast)

    start_close = close[j_arr]
    vol_velocity = np.where(start_close > 0,
                            (close - start_close) / start_close, 0.0)

    fstart_close = close[fj_arr]
    fast_vol_velocity = np.where(fstart_close > 0,
                                 (close - fstart_close) / fstart_close, 0.0)

    # ── Normalised velocities (divide by ATR%) ────────────────────────────────
    vol_norm_velocity = np.where(atr_pct > 0, vol_velocity / atr_pct, 0.0)
    fast_vol_norm_velocity = np.where(atr_pct > 0,
                                      fast_vol_velocity / atr_pct, 0.0)

    # ── Acceleration: change in norm_velocity over the same volume window ─────
    # Compare current norm_velocity with the norm_velocity at the start of window
    vol_velocity_delta = vol_norm_velocity - vol_norm_velocity[j_arr]
    fast_vol_velocity_delta = (fast_vol_norm_velocity
                               - fast_vol_norm_velocity[fj_arr])

    # ── Store results ─────────────────────────────────────────────────────────
    df = df.copy()
    df["vol_velocity"] = vol_velocity
    df["vol_norm_velocity"] = vol_norm_velocity
    df["vol_velocity_delta"] = vol_velocity_delta
    df["fast_vol_velocity"] = fast_vol_velocity
    df["fast_vol_norm_velocity"] = fast_vol_norm_velocity
    df["fast_vol_velocity_delta"] = fast_vol_velocity_delta
    df["vol_lookback_bars"] = vel_lookbacks
    df["volume_ratio"] = volume_ratio

    return df


# ============================================================================
# VOLUME-CLOCK ENTRY CONDITIONS
# ============================================================================

class VolumeReversalLongCond:
    """
    Volume-clock mean-reversion long entry near support.

    Mirrors ReversalLongCond but uses vol_norm_velocity / vol_velocity_delta
    (volume-clock based) instead of time-bar-based norm_velocity / velocity_delta.

    Triggers when:
        1. dist_to_support < proximity_pct   (price near support)
        2. vol_norm_velocity < velocity_ceil  (was falling in volume-time)
        3. vol_velocity_delta > min_decel     (fall decelerating / reversing)

    Optional filters (None = disabled):
        velocity_floor   : lower bound on vol_norm_velocity
        fast_velocity_ceil / fast_velocity_floor / fast_min_decel
            : filters on fast_vol_norm_velocity / fast_vol_velocity_delta
        min_reward_pct   : minimum dist_to_resistance (profit space)
        min_vol_ratio    : minimum volume_ratio (confirm with above-avg volume)
        max_vol_ratio    : maximum volume_ratio (filter extreme spike bars)
        trend_sma        : require close > sma_{trend_sma}
        rsi_max          : require rsi_14 < rsi_max

    Args:
        proximity_pct: Max distance to support (fraction). Default=0.01.
        min_decel: Min vol_velocity_delta for reversal. Default=0.5.
        velocity_ceil: Upper bound for vol_norm_velocity. Default=0.
        velocity_floor: Lower bound for vol_norm_velocity. Default=None.
        fast_velocity_ceil: Upper bound for fast_vol_norm_velocity. Default=None.
        fast_velocity_floor: Lower bound for fast_vol_norm_velocity. Default=None.
        fast_min_decel: Min fast_vol_velocity_delta. Default=None.
        min_reward_pct: Min dist_to_resistance. Default=None.
        min_vol_ratio: Min volume_ratio to confirm entry. Default=None.
        max_vol_ratio: Max volume_ratio to reject spike bars. Default=None.
        trend_sma: SMA period for trend filter. Default=None.
        rsi_max: Max RSI-14. Default=None.
    """
    def __init__(self, proximity_pct: float = 0.01, min_decel: float = 0.5,
                 velocity_ceil: float = 0.0, velocity_floor: float = None,
                 fast_velocity_ceil: float = None,
                 fast_velocity_floor: float = None,
                 fast_min_decel: float = None,
                 min_reward_pct: float = None,
                 min_vol_ratio: float = None,
                 max_vol_ratio: float = None,
                 trend_sma: int = None,
                 rsi_max: float = None):
        self.proximity_pct = proximity_pct
        self.min_decel = min_decel
        self.velocity_ceil = velocity_ceil
        self.velocity_floor = velocity_floor
        self.fast_velocity_ceil = fast_velocity_ceil
        self.fast_velocity_floor = fast_velocity_floor
        self.fast_min_decel = fast_min_decel
        self.min_reward_pct = min_reward_pct
        self.min_vol_ratio = min_vol_ratio
        self.max_vol_ratio = max_vol_ratio
        self.trend_sma = trend_sma
        self.rsi_max = rsi_max
        self.name = (f"VolRevL(px{proximity_pct*100:.1f}%,"
                     f"dc>{min_decel:.1f})")

    def __call__(self, row: pd.Series) -> ConditionResult:
        dist_sup = row.get("dist_to_support", np.nan)
        nv = row.get("vol_norm_velocity", np.nan)
        vd = row.get("vol_velocity_delta", np.nan)
        if pd.isna(dist_sup) or pd.isna(nv) or pd.isna(vd):
            return False
        if not (dist_sup < self.proximity_pct
                and nv < self.velocity_ceil
                and vd > self.min_decel):
            return False
        # velocity_floor: filter extreme crash bars
        if self.velocity_floor is not None and nv < self.velocity_floor:
            return False
        # Fast velocity filters
        if self.fast_velocity_ceil is not None:
            fnv = row.get("fast_vol_norm_velocity", np.nan)
            if pd.isna(fnv) or fnv > self.fast_velocity_ceil:
                return False
        if self.fast_velocity_floor is not None:
            fnv = row.get("fast_vol_norm_velocity", np.nan)
            if pd.isna(fnv) or fnv < self.fast_velocity_floor:
                return False
        if self.fast_min_decel is not None:
            fvd = row.get("fast_vol_velocity_delta", np.nan)
            if pd.isna(fvd) or fvd < self.fast_min_decel:
                return False
        # Reward filter: ensure upside to resistance
        if self.min_reward_pct is not None:
            dist_res = row.get("dist_to_resistance", np.nan)
            if not pd.isna(dist_res) and dist_res < self.min_reward_pct:
                return False
        # Volume ratio filters
        if self.min_vol_ratio is not None or self.max_vol_ratio is not None:
            vr = row.get("volume_ratio", np.nan)
            if pd.isna(vr):
                return False
            if self.min_vol_ratio is not None and vr < self.min_vol_ratio:
                return False
            if self.max_vol_ratio is not None and vr > self.max_vol_ratio:
                return False
        # Trend filter
        if self.trend_sma is not None:
            sma = row.get(f"sma_{self.trend_sma}", np.nan)
            close = row.get("close", np.nan)
            if pd.isna(sma) or pd.isna(close) or close <= sma:
                return False
        # RSI filter
        if self.rsi_max is not None:
            rsi = row.get("rsi_14", np.nan)
            if pd.isna(rsi) or rsi >= self.rsi_max:
                return False
        return self.name

    def __repr__(self):
        return (f"VolumeReversalLongCond(proximity_pct={self.proximity_pct}, "
                f"min_decel={self.min_decel})")


class VolumeReversalShortCond:
    """
    Volume-clock mean-reversion short entry near resistance.

    Mirrors ReversalShortCond but uses vol_norm_velocity / vol_velocity_delta.

    Triggers when:
        1. dist_to_resistance < proximity_pct  (price near resistance)
        2. vol_norm_velocity > velocity_floor   (was rising in volume-time)
        3. vol_velocity_delta < -min_decel      (rise decelerating / reversing)

    Optional filters (None = disabled):
        velocity_ceil    : upper bound on vol_norm_velocity
        fast_velocity_ceil / fast_velocity_floor / fast_min_decel
            : filters on fast_vol_norm_velocity / fast_vol_velocity_delta
        min_reward_pct   : minimum dist_to_support (profit space)
        min_vol_ratio    : minimum volume_ratio
        max_vol_ratio    : maximum volume_ratio
        trend_sma        : require close < sma_{trend_sma}
        rsi_min          : require rsi_14 > rsi_min

    Args:
        proximity_pct: Max distance to resistance (fraction). Default=0.01.
        min_decel: Min |vol_velocity_delta| for reversal. Default=0.5.
        velocity_floor: Lower bound for vol_norm_velocity. Default=0.
        velocity_ceil: Upper bound for vol_norm_velocity. Default=None.
        fast_velocity_floor: Lower bound for fast_vol_norm_velocity. Default=None.
        fast_velocity_ceil: Upper bound for fast_vol_norm_velocity. Default=None.
        fast_min_decel: Min |fast_vol_velocity_delta|. Default=None.
        min_reward_pct: Min dist_to_support. Default=None.
        min_vol_ratio: Min volume_ratio. Default=None.
        max_vol_ratio: Max volume_ratio. Default=None.
        trend_sma: SMA period for trend filter. Default=None.
        rsi_min: Min RSI-14. Default=None.
    """
    def __init__(self, proximity_pct: float = 0.01, min_decel: float = 0.5,
                 velocity_floor: float = 0.0, velocity_ceil: float = None,
                 fast_velocity_floor: float = None,
                 fast_velocity_ceil: float = None,
                 fast_min_decel: float = None,
                 min_reward_pct: float = None,
                 min_vol_ratio: float = None,
                 max_vol_ratio: float = None,
                 trend_sma: int = None,
                 rsi_min: float = None):
        self.proximity_pct = proximity_pct
        self.min_decel = min_decel
        self.velocity_floor = velocity_floor
        self.velocity_ceil = velocity_ceil
        self.fast_velocity_floor = fast_velocity_floor
        self.fast_velocity_ceil = fast_velocity_ceil
        self.fast_min_decel = fast_min_decel
        self.min_reward_pct = min_reward_pct
        self.min_vol_ratio = min_vol_ratio
        self.max_vol_ratio = max_vol_ratio
        self.trend_sma = trend_sma
        self.rsi_min = rsi_min
        self.name = (f"VolRevS(px{proximity_pct*100:.1f}%,"
                     f"dc>{min_decel:.1f})")

    def __call__(self, row: pd.Series) -> ConditionResult:
        dist_res = row.get("dist_to_resistance", np.nan)
        nv = row.get("vol_norm_velocity", np.nan)
        vd = row.get("vol_velocity_delta", np.nan)
        if pd.isna(dist_res) or pd.isna(nv) or pd.isna(vd):
            return False
        if not (dist_res < self.proximity_pct
                and nv > self.velocity_floor
                and vd < -self.min_decel):
            return False
        # velocity_ceil: filter extreme rally bars
        if self.velocity_ceil is not None and nv > self.velocity_ceil:
            return False
        # Fast velocity filters
        if self.fast_velocity_floor is not None:
            fnv = row.get("fast_vol_norm_velocity", np.nan)
            if pd.isna(fnv) or fnv < self.fast_velocity_floor:
                return False
        if self.fast_velocity_ceil is not None:
            fnv = row.get("fast_vol_norm_velocity", np.nan)
            if pd.isna(fnv) or fnv > self.fast_velocity_ceil:
                return False
        if self.fast_min_decel is not None:
            fvd = row.get("fast_vol_velocity_delta", np.nan)
            if pd.isna(fvd) or fvd > -self.fast_min_decel:
                return False
        # Reward filter: ensure downside to support
        if self.min_reward_pct is not None:
            dist_sup = row.get("dist_to_support", np.nan)
            if not pd.isna(dist_sup) and dist_sup < self.min_reward_pct:
                return False
        # Volume ratio filters
        if self.min_vol_ratio is not None or self.max_vol_ratio is not None:
            vr = row.get("volume_ratio", np.nan)
            if pd.isna(vr):
                return False
            if self.min_vol_ratio is not None and vr < self.min_vol_ratio:
                return False
            if self.max_vol_ratio is not None and vr > self.max_vol_ratio:
                return False
        # Trend filter
        if self.trend_sma is not None:
            sma = row.get(f"sma_{self.trend_sma}", np.nan)
            close = row.get("close", np.nan)
            if pd.isna(sma) or pd.isna(close) or close >= sma:
                return False
        # RSI filter
        if self.rsi_min is not None:
            rsi = row.get("rsi_14", np.nan)
            if pd.isna(rsi) or rsi <= self.rsi_min:
                return False
        return self.name

    def __repr__(self):
        return (f"VolumeReversalShortCond(proximity_pct={self.proximity_pct}, "
                f"min_decel={self.min_decel})")
