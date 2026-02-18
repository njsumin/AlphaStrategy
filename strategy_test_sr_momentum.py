"""
VP S/R Mean-Reversion Strategy Test (strategy_test_sr_momentum.py)
===================================================================
V2: Mean-reversion entry at clustered S/R levels with momentum reversal
detection.  Uses 1h data with long-period VP (7d/14d/30d).

Baseline (proven best):
  D_nv<+0.5 — Dual-direction, velocity_bars=8, proximity=0.5%,
  min_decel=1.5, cluster_gap=1%, velocity_ceil=+0.5 (allow confirmed reversal),
  SL=-2% + Time48.  5-year: +599%, Sharpe 1.531, 0 loss years.

Grid parameters:
  - velocity_bars: [4, 8, 12]  (1h bars: 4h/8h/12h momentum window)
  - proximity_pct: [0.005, 0.01, 0.015]  (S/R proximity for entry)
  - min_decel: [0.3, 0.5, 1.0]  (minimum reversal momentum)
  - cluster_gap_pct: [0.01, 0.02]  (S/R clustering distance)

Long + Short + Dual direction for each combination.

15m mode (--15m):
  Explores 15-minute granularity with bar counts scaled ×4 from 1h.
  Backtests most recent 1 year only (>= 2025-02-18).

Usage:
    python strategy_test_sr_momentum.py               # 1h baseline only
    python strategy_test_sr_momentum.py --grid         # 1h baseline + full grid sweep
    python strategy_test_sr_momentum.py --sl           # 1h stop-loss mechanism grid
    python strategy_test_sr_momentum.py --sl --grid    # 1h stop-loss + parameter grid
    python strategy_test_sr_momentum.py --filter       # 1h entry filter grid (trend/RSI/sr_count/velocity)
    python strategy_test_sr_momentum.py --velocity     # 1h velocity range fine-grained grid
    python strategy_test_sr_momentum.py --15m          # 15m baseline (1 year)
    python strategy_test_sr_momentum.py --15m --grid   # 15m baseline + grid (1 year)
"""

import sys
import numpy as np
import pandas as pd
from conditions import (
    StopLossCond, combine_conditions,
    SRStopLossLong, SRStopLossShort, ATRStopLossCond, TimeBarStopCond,
)
from engine import load_data, backtest, print_results, save_results_to_files
from sr_levels import add_sr_levels
from sr_signals import (
    add_momentum_indicators,
    ReversalLongCond, ReversalShortCond,
    NearResistanceCond, NearSupportCond,
)


VP_WINDOWS = [7, 14, 30]

# Baseline: D_nv<+0.5 (velocity_ceil=0.5, SL=-2%+Time48)
BASELINE = {
    "velocity_bars": 8,
    "proximity_pct": 0.005,
    "min_decel": 1.5,
    "cluster_gap_pct": 0.01,
    "velocity_ceil_long": 0.5,    # allow nv up to +0.5 (confirmed reversal)
    "velocity_floor_short": -0.5, # symmetric for short
}
BASELINE_SL = [StopLossCond(-0.02), TimeBarStopCond(48)]

# 15m baseline: bar counts ×4 from 1h (8h window = 32×15m bars)
BASELINE_15M = {
    "velocity_bars": 32,
    "proximity_pct": 0.005,
    "min_decel": 1.0,
    "cluster_gap_pct": 0.01,
    "vp_windows": [7, 14, 30],
}

# Columns needed by backtest + conditions (slim df to speed up iterrows)
_KEEP_COLS = [
    "date", "open", "close",
    "nearest_resistance", "nearest_support",
    "dist_to_resistance", "dist_to_support",
    "norm_velocity", "price_velocity", "velocity_delta",
    "sr_count", "atr",
    "rsi_14", "sma_200",
]


def _slim_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns needed for backtest to speed up iterrows."""
    cols = [c for c in _KEEP_COLS if c in df.columns]
    out = df[cols].copy()
    out.attrs = df.attrs.copy()
    return out


def _prepare_data(df, cluster_gap_pct, velocity_bars, backtest_start="2021-01-01"):
    """Add S/R levels + momentum indicators, trim to backtest period."""
    df_sr = add_sr_levels(df.copy(), vp_windows=VP_WINDOWS,
                          cluster_gap_pct=cluster_gap_pct)
    df_bt = df_sr[df_sr["date"] >= backtest_start].reset_index(drop=True)
    df_bt.attrs = df_sr.attrs.copy()
    df_m = add_momentum_indicators(df_bt, velocity_bars=velocity_bars)
    return _slim_df(df_m)


def _compute_bh(df_bt):
    """Compute Buy & Hold total and 1-year returns."""
    bh_ret = (df_bt["close"].iloc[-1] / df_bt["close"].iloc[0] - 1) * 100
    one_year_ago = df_bt["date"].max() - pd.Timedelta(days=365)
    df_1y = df_bt[df_bt["date"] >= one_year_ago]
    bh_1y = None
    if len(df_1y) > 1:
        bh_1y = (df_1y["close"].iloc[-1] / df_1y["close"].iloc[0] - 1) * 100
    return bh_ret, bh_1y


def print_yearly_breakdown(result, bars_per_day=24):
    """Print per-year return and Sharpe for a backtest result."""
    pv = result["portfolio"]
    pv = pv.copy()
    pv["year"] = pv["date"].dt.year
    annualize = np.sqrt(365 * bars_per_day)

    print(f"\n  {result['name']} -- Yearly Breakdown:")
    print(f"  {'Year':<6} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print(f"  {'-'*42}")

    trades = result.get("trade_log", [])
    for year, grp in pv.groupby("year"):
        if len(grp) < 2:
            continue
        yr_ret = (grp["value"].iloc[-1] / grp["value"].iloc[0] - 1) * 100
        dr = grp["value"].pct_change().dropna()
        yr_sharpe = (dr.mean() / dr.std()) * annualize if dr.std() > 0 else 0
        yr_dd = ((grp["value"] - grp["value"].cummax()) / grp["value"].cummax()).min() * 100
        yr_trades = len([t for t in trades
                         if pd.Timestamp(t["date"]).year == year
                         and t["type"] in ("BUY", "SHORT")])
        print(f"  {year:<6} {yr_ret:>8.1f}% {yr_sharpe:>8.3f} {yr_dd:>7.1f}% {yr_trades:>7}")
    print()


def run_baseline(df):
    """Run the proven baseline strategy D_nv<+0.5."""
    b = BASELINE
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"])
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox = b["proximity_pct"]
    decel = b["min_decel"]
    v_ceil = b["velocity_ceil_long"]
    v_floor = b["velocity_floor_short"]

    results = []

    # Baseline Long
    results.append(backtest(
        df_slim, "Baseline_L",
        buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
        sell_cond=NearResistanceCond(0.002),
        close_conditions=BASELINE_SL,
    ))

    # Baseline Short
    results.append(backtest(
        df_slim, "Baseline_S",
        buy_cond=lambda row: False,
        sell_cond=lambda row: False,
        short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor),
        cover_cond=NearSupportCond(0.002),
        close_conditions=BASELINE_SL,
    ))

    # Baseline Dual (the proven best)
    results.append(backtest(
        df_slim, "Baseline_D",
        buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
        sell_cond=NearResistanceCond(0.002),
        short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor),
        cover_cond=NearSupportCond(0.002),
        close_conditions=BASELINE_SL,
    ))

    return results, bh_ret, bh_1y


def run_grid(df):
    """Run full parameter grid sweep."""
    velocity_bars_list = [4, 8, 12]
    proximity_pct_list = [0.005, 0.01, 0.015]
    min_decel_list = [0.3, 0.5, 1.0]
    cluster_gap_list = [0.01, 0.02]

    total = (len(velocity_bars_list) * len(proximity_pct_list)
             * len(min_decel_list) * len(cluster_gap_list) * 3)
    count = 0
    all_results = []
    bh_ret = bh_1y = 0

    for cgap in cluster_gap_list:
        df_sr = add_sr_levels(df.copy(), vp_windows=VP_WINDOWS,
                              cluster_gap_pct=cgap)
        df_bt = df_sr[df_sr["date"] >= "2021-01-01"].reset_index(drop=True)
        df_bt.attrs = df_sr.attrs.copy()

        if len(df_bt) == 0:
            print(f"No data after 2021-01-01, skipping cgap={cgap}")
            continue

        bh_ret, bh_1y = _compute_bh(df_bt)

        for vb in velocity_bars_list:
            df_m = add_momentum_indicators(df_bt.copy(), velocity_bars=vb)
            df_slim = _slim_df(df_m)

            for prox in proximity_pct_list:
                for decel in min_decel_list:
                    tag = (f"vb{vb}_px{prox*1000:.0f}_dc{decel*10:.0f}"
                           f"_cg{cgap*100:.0f}")

                    all_results.append(backtest(
                        df_slim, f"L_{tag}",
                        buy_cond=ReversalLongCond(prox, decel),
                        sell_cond=NearResistanceCond(0.002),
                        close_conditions=[StopLossCond(threshold=-0.02)],
                    ))
                    count += 1

                    all_results.append(backtest(
                        df_slim, f"S_{tag}",
                        buy_cond=lambda row: False,
                        sell_cond=lambda row: False,
                        short_cond=ReversalShortCond(prox, decel),
                        cover_cond=NearSupportCond(0.002),
                        close_conditions=[StopLossCond(threshold=-0.02)],
                    ))
                    count += 1

                    all_results.append(backtest(
                        df_slim, f"D_{tag}",
                        buy_cond=ReversalLongCond(prox, decel),
                        sell_cond=NearResistanceCond(0.002),
                        short_cond=ReversalShortCond(prox, decel),
                        cover_cond=NearSupportCond(0.002),
                        close_conditions=[StopLossCond(threshold=-0.02)],
                    ))
                    count += 1

                    if count % 27 == 0 or count == total:
                        print(f"  Progress: {count}/{total} combos done")

    return all_results, bh_ret, bh_1y


def run_baseline_15m(df):
    """Run baseline strategy on 15m data (most recent 1 year)."""
    b = BASELINE_15M
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"],
                            backtest_start="2025-02-18")
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox = b["proximity_pct"]
    decel = b["min_decel"]

    results = []

    results.append(backtest(
        df_slim, "Baseline_15m_L",
        buy_cond=ReversalLongCond(prox, decel),
        sell_cond=NearResistanceCond(0.002),
        close_conditions=[StopLossCond(threshold=-0.02)],
    ))

    results.append(backtest(
        df_slim, "Baseline_15m_S",
        buy_cond=lambda row: False,
        sell_cond=lambda row: False,
        short_cond=ReversalShortCond(prox, decel),
        cover_cond=NearSupportCond(0.002),
        close_conditions=[StopLossCond(threshold=-0.02)],
    ))

    results.append(backtest(
        df_slim, "Baseline_15m_D",
        buy_cond=ReversalLongCond(prox, decel),
        sell_cond=NearResistanceCond(0.002),
        short_cond=ReversalShortCond(prox, decel),
        cover_cond=NearSupportCond(0.002),
        close_conditions=[StopLossCond(threshold=-0.02)],
    ))

    return results, bh_ret, bh_1y


def run_grid_15m(df):
    """Run parameter grid sweep on 15m data (most recent 1 year)."""
    velocity_bars_list = [16, 32, 48]       # 4h / 8h / 12h in 15m bars
    proximity_pct_list = [0.003, 0.005, 0.01]
    min_decel_list = [0.5, 1.0, 1.5]
    cluster_gap_list = [0.01, 0.02]

    total = (len(velocity_bars_list) * len(proximity_pct_list)
             * len(min_decel_list) * len(cluster_gap_list) * 3)
    count = 0
    all_results = []
    bh_ret = bh_1y = 0

    for cgap in cluster_gap_list:
        df_sr = add_sr_levels(df.copy(), vp_windows=BASELINE_15M["vp_windows"],
                              cluster_gap_pct=cgap)
        df_bt = df_sr[df_sr["date"] >= "2025-02-18"].reset_index(drop=True)
        df_bt.attrs = df_sr.attrs.copy()

        if len(df_bt) == 0:
            print(f"No data after 2025-02-18, skipping cgap={cgap}")
            continue

        bh_ret, bh_1y = _compute_bh(df_bt)

        for vb in velocity_bars_list:
            df_m = add_momentum_indicators(df_bt.copy(), velocity_bars=vb)
            df_slim = _slim_df(df_m)

            for prox in proximity_pct_list:
                for decel in min_decel_list:
                    tag = (f"vb{vb}_px{prox*1000:.0f}_dc{decel*10:.0f}"
                           f"_cg{cgap*100:.0f}")

                    all_results.append(backtest(
                        df_slim, f"L_{tag}",
                        buy_cond=ReversalLongCond(prox, decel),
                        sell_cond=NearResistanceCond(0.002),
                        close_conditions=[StopLossCond(threshold=-0.02)],
                    ))
                    count += 1

                    all_results.append(backtest(
                        df_slim, f"S_{tag}",
                        buy_cond=lambda row: False,
                        sell_cond=lambda row: False,
                        short_cond=ReversalShortCond(prox, decel),
                        cover_cond=NearSupportCond(0.002),
                        close_conditions=[StopLossCond(threshold=-0.02)],
                    ))
                    count += 1

                    all_results.append(backtest(
                        df_slim, f"D_{tag}",
                        buy_cond=ReversalLongCond(prox, decel),
                        sell_cond=NearResistanceCond(0.002),
                        short_cond=ReversalShortCond(prox, decel),
                        cover_cond=NearSupportCond(0.002),
                        close_conditions=[StopLossCond(threshold=-0.02)],
                    ))
                    count += 1

                    if count % 27 == 0 or count == total:
                        print(f"  Progress: {count}/{total} combos done")

    return all_results, bh_ret, bh_1y


def run_filter_grid(df):
    """Run entry filter grid search with fixed baseline parameters + best stop-loss."""
    b = BASELINE
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"])
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox = b["proximity_pct"]
    decel = b["min_decel"]
    sl = [StopLossCond(-0.02), TimeBarStopCond(48)]  # best SL from prior grid

    # Filter configurations: each is a dict of optional kwargs for ReversalLongCond/ShortCond
    # None values are omitted (filter disabled)
    filter_configs = [
        # Baseline (no filters)
        {"name": "NoFilter",
         "long": {}, "short": {}},

        # --- 1. Trend filter (SMA200) ---
        {"name": "Trend200",
         "long": {"trend_sma": 200}, "short": {"trend_sma": 200}},

        # --- 2. RSI confirmation ---
        {"name": "RSI<45",
         "long": {"rsi_max": 45}, "short": {"rsi_min": 55}},
        {"name": "RSI<40",
         "long": {"rsi_max": 40}, "short": {"rsi_min": 60}},
        {"name": "RSI<50",
         "long": {"rsi_max": 50}, "short": {"rsi_min": 50}},

        # --- 3. S/R count filter ---
        {"name": "SR>=3",
         "long": {"sr_count_min": 3}, "short": {"sr_count_min": 3}},
        {"name": "SR>=4",
         "long": {"sr_count_min": 4}, "short": {"sr_count_min": 4}},

        # --- 4. Velocity cap (false reversal filter) ---
        {"name": "MaxV1.0",
         "long": {"max_abs_velocity": 1.0}, "short": {"max_abs_velocity": 1.0}},
        {"name": "MaxV1.5",
         "long": {"max_abs_velocity": 1.5}, "short": {"max_abs_velocity": 1.5}},
        {"name": "MaxV2.0",
         "long": {"max_abs_velocity": 2.0}, "short": {"max_abs_velocity": 2.0}},

        # --- 5. Combinations ---
        {"name": "Trend+RSI45",
         "long": {"trend_sma": 200, "rsi_max": 45},
         "short": {"trend_sma": 200, "rsi_min": 55}},
        {"name": "Trend+SR3",
         "long": {"trend_sma": 200, "sr_count_min": 3},
         "short": {"trend_sma": 200, "sr_count_min": 3}},
        {"name": "Trend+MaxV1.5",
         "long": {"trend_sma": 200, "max_abs_velocity": 1.5},
         "short": {"trend_sma": 200, "max_abs_velocity": 1.5}},
        {"name": "RSI45+SR3",
         "long": {"rsi_max": 45, "sr_count_min": 3},
         "short": {"rsi_min": 55, "sr_count_min": 3}},
        {"name": "RSI45+MaxV1.5",
         "long": {"rsi_max": 45, "max_abs_velocity": 1.5},
         "short": {"rsi_min": 55, "max_abs_velocity": 1.5}},
        {"name": "SR3+MaxV1.5",
         "long": {"sr_count_min": 3, "max_abs_velocity": 1.5},
         "short": {"sr_count_min": 3, "max_abs_velocity": 1.5}},

        # --- 6. Triple combos ---
        {"name": "Trend+RSI45+SR3",
         "long": {"trend_sma": 200, "rsi_max": 45, "sr_count_min": 3},
         "short": {"trend_sma": 200, "rsi_min": 55, "sr_count_min": 3}},
        {"name": "Trend+RSI45+MaxV1.5",
         "long": {"trend_sma": 200, "rsi_max": 45, "max_abs_velocity": 1.5},
         "short": {"trend_sma": 200, "rsi_min": 55, "max_abs_velocity": 1.5}},
        {"name": "RSI45+SR3+MaxV1.5",
         "long": {"rsi_max": 45, "sr_count_min": 3, "max_abs_velocity": 1.5},
         "short": {"rsi_min": 55, "sr_count_min": 3, "max_abs_velocity": 1.5}},

        # --- 7. All four filters ---
        {"name": "All4",
         "long": {"trend_sma": 200, "rsi_max": 45, "sr_count_min": 3,
                  "max_abs_velocity": 1.5},
         "short": {"trend_sma": 200, "rsi_min": 55, "sr_count_min": 3,
                   "max_abs_velocity": 1.5}},
    ]

    all_results = []
    total = len(filter_configs) * 3
    count = 0

    for cfg in filter_configs:
        tag = cfg["name"]
        long_kw = cfg["long"]
        short_kw = cfg["short"]

        # Long only
        all_results.append(backtest(
            df_slim, f"L_{tag}",
            buy_cond=ReversalLongCond(prox, decel, **long_kw),
            sell_cond=NearResistanceCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        # Short only
        all_results.append(backtest(
            df_slim, f"S_{tag}",
            buy_cond=lambda row: False,
            sell_cond=lambda row: False,
            short_cond=ReversalShortCond(prox, decel, **short_kw),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        # Dual
        all_results.append(backtest(
            df_slim, f"D_{tag}",
            buy_cond=ReversalLongCond(prox, decel, **long_kw),
            sell_cond=NearResistanceCond(0.002),
            short_cond=ReversalShortCond(prox, decel, **short_kw),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        if count % 9 == 0 or count == total:
            print(f"  Progress: {count}/{total} combos done")

    return all_results, bh_ret, bh_1y


def run_velocity_grid(df):
    """Fine-grained velocity range grid: test different nv bounds for entry."""
    b = BASELINE
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"])
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox = b["proximity_pct"]
    decel = b["min_decel"]
    sl = [StopLossCond(-0.02), TimeBarStopCond(48)]

    # velocity_ceil for Long (default 0 = nv<0 = still falling)
    # velocity_floor for Short (default 0 = nv>0 = still rising)
    # Positive ceil = allow nv to have turned positive (confirmed reversal)
    # max_abs_velocity = lower bound cap (filter extreme speed)
    #
    # Tested ranges form a 2D grid: [velocity_ceil/floor] x [max_abs_velocity]
    #
    # Key scenarios for Long:
    #   nv in (-inf, 0)       baseline: still falling, any speed
    #   nv in (-2.0, 0)       MaxV2.0: still falling, not too fast
    #   nv in (-1.0, 0)       MaxV1.0: almost stopped falling
    #   nv in (-0.5, 0)       MaxV0.5: nearly at inflection
    #   nv in (-inf, +0.3)    ceil+0.3: allow just-turned-positive
    #   nv in (-inf, +0.5)    ceil+0.5: allow slightly positive
    #   nv in (-inf, +1.0)    ceil+1.0: allow positive momentum
    #   nv in (-2.0, +0.3)    band: turned positive but wasn't crashing
    #   nv in (-1.0, +0.5)    tight band: clean reversal zone

    # For Long:  velocity_floor < nv < velocity_ceil
    # For Short: velocity_floor < nv < velocity_ceil
    # Long default:  nv < 0 (still falling), no floor
    # Short default: nv > 0 (still rising), no ceil
    configs = [
        # --- A. Baseline reference ---
        {"name": "base_nv<0",
         "long": {},
         "short": {}},

        # --- B. Tighter floor only (filter extreme crashes) ---
        {"name": "nv(-2,0)",
         "long": {"velocity_floor": -2.0},
         "short": {"velocity_ceil": 2.0}},
        {"name": "nv(-1.5,0)",
         "long": {"velocity_floor": -1.5},
         "short": {"velocity_ceil": 1.5}},
        {"name": "nv(-1,0)",
         "long": {"velocity_floor": -1.0},
         "short": {"velocity_ceil": 1.0}},
        {"name": "nv(-0.5,0)",
         "long": {"velocity_floor": -0.5},
         "short": {"velocity_ceil": 0.5}},

        # --- C. Raise ceiling: allow velocity to have turned positive ---
        {"name": "nv<+0.3",
         "long": {"velocity_ceil": 0.3},
         "short": {"velocity_floor": -0.3}},
        {"name": "nv<+0.5",
         "long": {"velocity_ceil": 0.5},
         "short": {"velocity_floor": -0.5}},
        {"name": "nv<+1.0",
         "long": {"velocity_ceil": 1.0},
         "short": {"velocity_floor": -1.0}},

        # --- D. Band: raise ceiling + floor cap ---
        {"name": "nv(-2,+0.3)",
         "long": {"velocity_ceil": 0.3, "velocity_floor": -2.0},
         "short": {"velocity_floor": -0.3, "velocity_ceil": 2.0}},
        {"name": "nv(-2,+0.5)",
         "long": {"velocity_ceil": 0.5, "velocity_floor": -2.0},
         "short": {"velocity_floor": -0.5, "velocity_ceil": 2.0}},
        {"name": "nv(-1.5,+0.3)",
         "long": {"velocity_ceil": 0.3, "velocity_floor": -1.5},
         "short": {"velocity_floor": -0.3, "velocity_ceil": 1.5}},
        {"name": "nv(-1.5,+0.5)",
         "long": {"velocity_ceil": 0.5, "velocity_floor": -1.5},
         "short": {"velocity_floor": -0.5, "velocity_ceil": 1.5}},
        {"name": "nv(-1,+0.3)",
         "long": {"velocity_ceil": 0.3, "velocity_floor": -1.0},
         "short": {"velocity_floor": -0.3, "velocity_ceil": 1.0}},
        {"name": "nv(-1,+0.5)",
         "long": {"velocity_ceil": 0.5, "velocity_floor": -1.0},
         "short": {"velocity_floor": -0.5, "velocity_ceil": 1.0}},
        {"name": "nv(-1,+1)",
         "long": {"velocity_ceil": 1.0, "velocity_floor": -1.0},
         "short": {"velocity_floor": -1.0, "velocity_ceil": 1.0}},

        # --- E. Require confirmed reversal: nv crossed zero ---
        {"name": "nv(0,+0.5)",
         "long": {"velocity_ceil": 0.5, "velocity_floor": 0.0},
         "short": {"velocity_floor": -0.5, "velocity_ceil": 0.0}},
        {"name": "nv(0,+1.0)",
         "long": {"velocity_ceil": 1.0, "velocity_floor": 0.0},
         "short": {"velocity_floor": -1.0, "velocity_ceil": 0.0}},
        {"name": "nv(0,+2.0)",
         "long": {"velocity_ceil": 2.0, "velocity_floor": 0.0},
         "short": {"velocity_floor": -2.0, "velocity_ceil": 0.0}},
    ]

    all_results = []
    total = len(configs) * 3
    count = 0

    for cfg in configs:
        tag = cfg["name"]
        long_kw = cfg["long"]
        short_kw = cfg["short"]

        all_results.append(backtest(
            df_slim, f"L_{tag}",
            buy_cond=ReversalLongCond(prox, decel, **long_kw),
            sell_cond=NearResistanceCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        all_results.append(backtest(
            df_slim, f"S_{tag}",
            buy_cond=lambda row: False,
            sell_cond=lambda row: False,
            short_cond=ReversalShortCond(prox, decel, **short_kw),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        all_results.append(backtest(
            df_slim, f"D_{tag}",
            buy_cond=ReversalLongCond(prox, decel, **long_kw),
            sell_cond=NearResistanceCond(0.002),
            short_cond=ReversalShortCond(prox, decel, **short_kw),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        if count % 9 == 0 or count == total:
            print(f"  Progress: {count}/{total} combos done")

    return all_results, bh_ret, bh_1y


def run_stoploss_grid(df):
    """Run stop-loss mechanism grid search with fixed baseline entry parameters."""
    b = BASELINE
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"])
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox = b["proximity_pct"]
    decel = b["min_decel"]

    stoploss_configs = [
        # Current baseline
        {"name": "Fixed2%",
         "long": [StopLossCond(-0.02)],
         "short": [StopLossCond(-0.02)]},
        # S/R dynamic
        {"name": "SR_b3_x3",
         "long": [SRStopLossLong(0.003, 0.03)],
         "short": [SRStopLossShort(0.003, 0.03)]},
        {"name": "SR_b5_x5",
         "long": [SRStopLossLong(0.005, 0.05)],
         "short": [SRStopLossShort(0.005, 0.05)]},
        # ATR
        {"name": "ATR1.5x",
         "long": [ATRStopLossCond(1.5)],
         "short": [ATRStopLossCond(1.5)]},
        {"name": "ATR2.0x",
         "long": [ATRStopLossCond(2.0)],
         "short": [ATRStopLossCond(2.0)]},
        {"name": "ATR2.5x",
         "long": [ATRStopLossCond(2.5)],
         "short": [ATRStopLossCond(2.5)]},
        # Time stop (with fixed 2% SL as safety net)
        {"name": "Time48",
         "long": [StopLossCond(-0.02), TimeBarStopCond(48)],
         "short": [StopLossCond(-0.02), TimeBarStopCond(48)]},
        {"name": "Time72",
         "long": [StopLossCond(-0.02), TimeBarStopCond(72)],
         "short": [StopLossCond(-0.02), TimeBarStopCond(72)]},
        # Combo: S/R + Time
        {"name": "SR3+T48",
         "long": [SRStopLossLong(0.003, 0.03), TimeBarStopCond(48)],
         "short": [SRStopLossShort(0.003, 0.03), TimeBarStopCond(48)]},
        {"name": "SR3+T72",
         "long": [SRStopLossLong(0.003, 0.03), TimeBarStopCond(72)],
         "short": [SRStopLossShort(0.003, 0.03), TimeBarStopCond(72)]},
        # Combo: ATR + Time
        {"name": "ATR2+T48",
         "long": [ATRStopLossCond(2.0), TimeBarStopCond(48)],
         "short": [ATRStopLossCond(2.0), TimeBarStopCond(48)]},
        {"name": "ATR2+T72",
         "long": [ATRStopLossCond(2.0), TimeBarStopCond(72)],
         "short": [ATRStopLossCond(2.0), TimeBarStopCond(72)]},
    ]

    all_results = []
    total = len(stoploss_configs) * 3
    count = 0

    for cfg in stoploss_configs:
        sl_name = cfg["name"]

        # Long
        all_results.append(backtest(
            df_slim, f"L_{sl_name}",
            buy_cond=ReversalLongCond(prox, decel),
            sell_cond=NearResistanceCond(0.002),
            close_conditions=cfg["long"],
        ))
        count += 1

        # Short
        all_results.append(backtest(
            df_slim, f"S_{sl_name}",
            buy_cond=lambda row: False,
            sell_cond=lambda row: False,
            short_cond=ReversalShortCond(prox, decel),
            cover_cond=NearSupportCond(0.002),
            close_conditions=cfg["short"],
        ))
        count += 1

        # Dual
        all_results.append(backtest(
            df_slim, f"D_{sl_name}",
            buy_cond=ReversalLongCond(prox, decel),
            sell_cond=NearResistanceCond(0.002),
            short_cond=ReversalShortCond(prox, decel),
            cover_cond=NearSupportCond(0.002),
            close_conditions=cfg["long"],  # long conditions for dual
        ))
        count += 1

        if count % 9 == 0 or count == total:
            print(f"  Progress: {count}/{total} combos done")

    return all_results, bh_ret, bh_1y


def main():
    grid_mode = "--grid" in sys.argv
    mode_15m = "--15m" in sys.argv
    sl_mode = "--sl" in sys.argv
    filter_mode = "--filter" in sys.argv
    velocity_mode = "--velocity" in sys.argv

    if mode_15m:
        # 15m mode: load 15m data with 30d warmup before 1-year backtest
        df = load_data(
            start="2024-02-18",
            include_fg=False,
            include_derivatives=False,
            include_cb_premium=False,
            interval="15m",
        )

        baseline_results, bh_ret, bh_1y = run_baseline_15m(df)
        all_results = list(baseline_results)

        print("\n===== 15m BASELINE (vb32_px5_dc10_cg1) =====")
        print_results(baseline_results, buy_and_hold_ret=bh_ret,
                      buy_and_hold_1y=bh_1y)

        if grid_mode:
            print("\n===== 15m GRID SWEEP =====")
            grid_results, bh_ret, bh_1y = run_grid_15m(df)
            all_results.extend(grid_results)

        active = [r for r in all_results if r["trades"] > 0]
        active.sort(key=lambda x: x["sharpe"], reverse=True)

        if grid_mode:
            top = active[:30]
            print_results(top, buy_and_hold_ret=bh_ret, buy_and_hold_1y=bh_1y)

        save_results_to_files(
            active,
            summary_path="results_sr_momentum_15m_summary.txt",
            trade_log_path="trade_logs_sr_momentum_15m.txt",
            buy_and_hold_ret=bh_ret,
            buy_and_hold_1y=bh_1y,
        )

        print(f"\nTotal strategies: {len(active)} active (15m)")

    else:
        # 1h mode (original behavior)
        df = load_data(
            start="2020-01-01",
            include_fg=False,
            include_derivatives=False,
            include_cb_premium=False,
            interval="1h",
        )

        baseline_results, bh_ret, bh_1y = run_baseline(df)
        all_results = list(baseline_results)

        print("\n===== BASELINE (D_nv<+0.5) =====")
        print_results(baseline_results, buy_and_hold_ret=bh_ret,
                      buy_and_hold_1y=bh_1y)

        for res in baseline_results:
            print_yearly_breakdown(res, bars_per_day=24)

        if filter_mode:
            print("\n===== ENTRY FILTER GRID SWEEP =====")
            filter_results, bh_ret, bh_1y = run_filter_grid(df)
            all_results.extend(filter_results)

        if velocity_mode:
            print("\n===== VELOCITY RANGE GRID SWEEP =====")
            vel_results, bh_ret, bh_1y = run_velocity_grid(df)
            all_results.extend(vel_results)

        if sl_mode:
            print("\n===== STOP-LOSS GRID SWEEP =====")
            sl_results, bh_ret, bh_1y = run_stoploss_grid(df)
            all_results.extend(sl_results)

        if grid_mode:
            print("\n===== GRID SWEEP =====")
            grid_results, bh_ret, bh_1y = run_grid(df)
            all_results.extend(grid_results)

        active = [r for r in all_results if r["trades"] > 0]
        active.sort(key=lambda x: x["sharpe"], reverse=True)

        if grid_mode or sl_mode or filter_mode or velocity_mode:
            top = active[:30]
            print_results(top, buy_and_hold_ret=bh_ret, buy_and_hold_1y=bh_1y)

            # Yearly breakdown for top-3
            for res in top[:3]:
                print_yearly_breakdown(res, bars_per_day=24)

        save_results_to_files(
            active,
            summary_path="results_sr_momentum_summary.txt",
            trade_log_path="trade_logs_sr_momentum.txt",
            buy_and_hold_ret=bh_ret,
            buy_and_hold_1y=bh_1y,
        )

        print(f"\nTotal strategies: {len(active)} active")


if __name__ == "__main__":
    main()
