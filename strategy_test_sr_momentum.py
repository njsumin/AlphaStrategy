"""
VP S/R Mean-Reversion Strategy Test (strategy_test_sr_momentum.py)
===================================================================
V2: Mean-reversion entry at clustered S/R levels with momentum reversal
detection.  Uses 1h data with long-period VP (7d/14d/30d).

Baseline (proven best):
  D_vb8_px5_dc10_cg1 — Dual-direction, velocity_bars=8, proximity=0.5%,
  min_decel=1.0, cluster_gap=1%.  5-year: +256%, Sharpe 1.09, 0 loss years.

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
    python strategy_test_sr_momentum.py --15m          # 15m baseline (1 year)
    python strategy_test_sr_momentum.py --15m --grid   # 15m baseline + grid (1 year)
"""

import sys
import numpy as np
import pandas as pd
from conditions import StopLossCond, combine_conditions
from engine import load_data, backtest, print_results, save_results_to_files
from sr_levels import add_sr_levels
from sr_signals import (
    add_momentum_indicators,
    ReversalLongCond, ReversalShortCond,
    NearResistanceCond, NearSupportCond,
)


VP_WINDOWS = [7, 14, 30]

# Baseline: D_vb8_px5_dc15_cg1
BASELINE = {
    "velocity_bars": 8,
    "proximity_pct": 0.005,
    "min_decel": 1.5,
    "cluster_gap_pct": 0.01,
}

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
    "sr_count",
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
    """Run the proven baseline strategy D_vb8_px5_dc10_cg1."""
    b = BASELINE
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"])
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox = b["proximity_pct"]
    decel = b["min_decel"]

    results = []

    # Baseline Long
    results.append(backtest(
        df_slim, "Baseline_L",
        buy_cond=ReversalLongCond(prox, decel),
        sell_cond=NearResistanceCond(0.002),
        close_conditions=[StopLossCond(threshold=-0.02)],
    ))

    # Baseline Short
    results.append(backtest(
        df_slim, "Baseline_S",
        buy_cond=lambda row: False,
        sell_cond=lambda row: False,
        short_cond=ReversalShortCond(prox, decel),
        cover_cond=NearSupportCond(0.002),
        close_conditions=[StopLossCond(threshold=-0.02)],
    ))

    # Baseline Dual (the proven best)
    results.append(backtest(
        df_slim, "Baseline_D",
        buy_cond=ReversalLongCond(prox, decel),
        sell_cond=NearResistanceCond(0.002),
        short_cond=ReversalShortCond(prox, decel),
        cover_cond=NearSupportCond(0.002),
        close_conditions=[StopLossCond(threshold=-0.02)],
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


def main():
    grid_mode = "--grid" in sys.argv
    mode_15m = "--15m" in sys.argv

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

        print("\n===== BASELINE (D_vb8_px5_dc15_cg1) =====")
        print_results(baseline_results, buy_and_hold_ret=bh_ret,
                      buy_and_hold_1y=bh_1y)

        for res in baseline_results:
            print_yearly_breakdown(res, bars_per_day=24)

        if grid_mode:
            print("\n===== GRID SWEEP =====")
            grid_results, bh_ret, bh_1y = run_grid(df)
            all_results.extend(grid_results)

        active = [r for r in all_results if r["trades"] > 0]
        active.sort(key=lambda x: x["sharpe"], reverse=True)

        if grid_mode:
            top = active[:30]
            print_results(top, buy_and_hold_ret=bh_ret, buy_and_hold_1y=bh_1y)

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
