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
  Full-history 15m baseline mirroring 1h baseline with bar counts scaled ×4.
  velocity_bars=32 (8h), fast_velocity_bars=8 (2h), TimeBarStop=192b (48h).
  Backtests full history from 2021-01-01 (same period as 1h baseline).

Usage:
    python strategy_test_sr_momentum.py               # 1h baseline only
    python strategy_test_sr_momentum.py --grid         # 1h baseline + full grid sweep
    python strategy_test_sr_momentum.py --sl           # 1h stop-loss mechanism grid
    python strategy_test_sr_momentum.py --sl --grid    # 1h stop-loss + parameter grid
    python strategy_test_sr_momentum.py --filter       # 1h entry filter grid (trend/RSI/sr_count/velocity)
    python strategy_test_sr_momentum.py --velocity     # 1h velocity range fine-grained grid
    python strategy_test_sr_momentum.py --fast          # 1h fast velocity filter grid
    python strategy_test_sr_momentum.py --vp            # 1h VP window combination grid
    python strategy_test_sr_momentum.py --wbottom      # 1h W-bottom/M-top microstructure filter grid
    python strategy_test_sr_momentum.py --sensitivity  # 1h one-at-a-time parameter sensitivity analysis
    python strategy_test_sr_momentum.py --15m           # 15m baseline (full history)
    python strategy_test_sr_momentum.py --15m --grid    # 15m baseline + grid
    python strategy_test_sr_momentum.py --15m --optimize # 15m walk-forward optimization
    python strategy_test_sr_momentum.py --volbaseline   # volume-clock S/R strategy
                                                        #   baseline (2021-present)
                                                        #   + walk-forward opt (train 2021-2025.6 / val 2025.6+)
"""

import sys
import numpy as np
import pandas as pd
from conditions import (
    StopLossCond, TakeProfitCond, TrailingStopCond, RatchetStopCond,
    combine_conditions,
    SRStopLossLong, SRStopLossShort, ATRStopLossCond, TimeBarStopCond,
)
from engine import load_data, backtest, print_results, save_results_to_files
from sr_levels import add_sr_levels
from sr_signals import (
    add_momentum_indicators,
    add_volume_momentum_indicators,
    ReversalLongCond, ReversalShortCond,
    VolumeReversalLongCond, VolumeReversalShortCond,
    NearResistanceCond, NearSupportCond,
)
from trend_regime import (
    compute_regime, map_regime_to_bars, resample_to_daily,
    RegimeLongCond, RegimeShortCond,
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
# Parameters mirror the 1h baseline; time-based values scaled by ×4.
BASELINE_15M = {
    "velocity_bars": 32,          # 8h × 4 = 32 bars in 15m
    "fast_velocity_bars": 8,      # 2h × 4 = 8 bars in 15m
    "proximity_pct": 0.005,
    "min_decel": 1.5,             # same as 1h baseline
    "cluster_gap_pct": 0.01,
    "velocity_ceil_long": 0.5,    # same as 1h baseline
    "velocity_floor_short": -0.5, # same as 1h baseline
    "vp_windows": [7, 14, 30],
}
# 48 1h-bars = 2 calendar days → 192 15m-bars
BASELINE_15M_SL = [StopLossCond(-0.02), TimeBarStopCond(192, bars_per_day=96)]

# Columns needed by backtest + conditions (slim df to speed up iterrows)
_KEEP_COLS = [
    "date", "open", "close",
    "nearest_resistance", "nearest_support",
    "dist_to_resistance", "dist_to_support",
    "norm_velocity", "price_velocity", "velocity_delta",
    "fast_norm_velocity", "fast_price_velocity", "fast_velocity_delta",
    # volume-clock indicators
    "vol_norm_velocity", "vol_velocity", "vol_velocity_delta",
    "fast_vol_norm_velocity", "fast_vol_velocity", "fast_vol_velocity_delta",
    "vol_lookback_bars", "volume_ratio",
    "sr_count", "atr",
    "rsi_14", "sma_200",
    "w_bottom_strength", "m_top_strength",
]


def _slim_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns needed for backtest to speed up iterrows."""
    cols = [c for c in _KEEP_COLS if c in df.columns]
    out = df[cols].copy()
    out.attrs = df.attrs.copy()
    return out


def _prepare_data(df, cluster_gap_pct, velocity_bars, backtest_start="2021-01-01",
                   vp_windows=None, fast_velocity_bars=2):
    """Add S/R levels + momentum indicators, trim to backtest period."""
    df_sr = add_sr_levels(df.copy(), vp_windows=vp_windows or VP_WINDOWS,
                          cluster_gap_pct=cluster_gap_pct)
    df_bt = df_sr[df_sr["date"] >= backtest_start].reset_index(drop=True)
    df_bt.attrs = df_sr.attrs.copy()
    df_m = add_momentum_indicators(df_bt, velocity_bars=velocity_bars,
                                   fast_velocity_bars=fast_velocity_bars)
    return _slim_df(df_m)


def _prepare_vol_data(df, cluster_gap_pct, velocity_vol_mult,
                      fast_vol_mult=2.0, vol_ref_period=168,
                      backtest_start="2021-01-01", vp_windows=None):
    """Add S/R levels + volume-clock momentum indicators, trim to period.

    Note: S/R level computation and volume indicators use ALL data before
    backtest_start for warm-up (VP windows, rolling volume reference).
    Only the trimmed slice is returned for backtesting.
    """
    df_sr = add_sr_levels(df.copy(), vp_windows=vp_windows or VP_WINDOWS,
                          cluster_gap_pct=cluster_gap_pct)
    df_bt = df_sr[df_sr["date"] >= backtest_start].reset_index(drop=True)
    df_bt.attrs = df_sr.attrs.copy()
    df_m = add_volume_momentum_indicators(
        df_bt,
        velocity_vol_mult=velocity_vol_mult,
        fast_vol_mult=fast_vol_mult,
        vol_ref_period=vol_ref_period,
    )
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
    """Run baseline strategy on 15m data (full history from 2021-01-01).

    Parameters mirror the proven 1h baseline (D_nv<+0.5) with all time-based
    bar counts scaled ×4 for 15m granularity:
      velocity_bars=32  (8h), fast_velocity_bars=8  (2h),
      TimeBarStop=192 bars (48h = 2 days), SL=-2%.
    """
    b = BASELINE_15M
    df_slim = _prepare_data(
        df, b["cluster_gap_pct"], b["velocity_bars"],
        backtest_start="2021-01-01",
        vp_windows=b.get("vp_windows"),
        fast_velocity_bars=b["fast_velocity_bars"],
    )
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox    = b["proximity_pct"]
    decel   = b["min_decel"]
    v_ceil  = b["velocity_ceil_long"]
    v_floor = b["velocity_floor_short"]

    results = []

    results.append(backtest(
        df_slim, "Baseline_15m_L",
        buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
        sell_cond=NearResistanceCond(0.002),
        close_conditions=BASELINE_15M_SL,
    ))

    results.append(backtest(
        df_slim, "Baseline_15m_S",
        buy_cond=lambda row: False,
        sell_cond=lambda row: False,
        short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor),
        cover_cond=NearSupportCond(0.002),
        close_conditions=BASELINE_15M_SL,
    ))

    results.append(backtest(
        df_slim, "Baseline_15m_D",
        buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
        sell_cond=NearResistanceCond(0.002),
        short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor),
        cover_cond=NearSupportCond(0.002),
        close_conditions=BASELINE_15M_SL,
    ))

    return results, bh_ret, bh_1y


def run_optimization_15m(df):
    """15m parameter walk-forward optimization.

    Strategy:
      Phase 1 — Dual-direction grid on train set  (~48 combos, ~7 min)
      Phase 2 — Validate top-15 D configs on val  (~15 backtests)
      Phase 3 — L / S breakdown for best-5 configs (~10 backtests)

    Train  : 2022-01-01 ~ 2025-05-31  (~119k 15m bars)
    Validate: 2025-06-01 ~ latest      (~25k 15m bars)

    Fixed : cluster_gap_pct=0.01, vp_windows=[7,14,30], SL=-2%+Time192b
    Grid  (coarse but covers key axes):
        velocity_bars      [16, 32, 48]   (4h / 8h / 12h)
        fast_velocity_bars [2, 8]         (30m / 2h)  — must be < vb
        proximity_pct      [0.003, 0.007] (tight / loose proximity)
        min_decel          [0.5, 1.5, 3.0]
        velocity_ceil_long [0.0, 0.5]     (wait for fall / allow slight bounce)
    Phase-1 total (D only): 3×2×2×3×2 = 72 combos  (~10 min on train)
    """
    TRAIN_START = "2022-01-01"
    TRAIN_END   = "2025-05-31"
    VAL_START   = "2025-06-01"
    TOP_D       = 15   # top Dual candidates to validate

    b    = BASELINE_15M
    cgap = b["cluster_gap_pct"]
    vp_w = b["vp_windows"]
    sl   = BASELINE_15M_SL

    vb_list    = [16, 32, 48]
    fvb_list   = [2, 8]           # fast_velocity_bars — skipped if >= vb
    prox_list  = [0.003, 0.007]
    decel_list = [0.5, 1.5, 3.0]
    vceil_list = [0.0, 0.5]

    # ── 1. S/R levels (uses pre-computed VP columns from load_data) ───────────
    print("  Computing S/R levels...", flush=True)
    # Trim first so the Python for-loop runs over fewer rows
    df_base = df[df["date"] >= TRAIN_START].reset_index(drop=True).copy()
    df_base.attrs = df.attrs.copy()
    df_sr   = add_sr_levels(df_base, vp_windows=vp_w, cluster_gap_pct=cgap)
    df_sr.attrs = df_base.attrs.copy()

    df_train_base = df_sr[df_sr["date"] <= TRAIN_END].reset_index(drop=True)
    df_train_base.attrs = df_sr.attrs.copy()
    df_val_base   = df_sr[df_sr["date"] >= VAL_START].reset_index(drop=True)
    df_val_base.attrs = df_sr.attrs.copy()

    bh_train = (df_train_base["close"].iloc[-1] / df_train_base["close"].iloc[0] - 1) * 100
    bh_val   = (df_val_base["close"].iloc[-1]   / df_val_base["close"].iloc[0]   - 1) * 100

    # Count valid (vb, fvb) pairs
    valid_pairs = [(vb, fvb) for vb in vb_list for fvb in fvb_list if fvb < vb]
    phase1_total = len(valid_pairs) * len(prox_list) * len(decel_list) * len(vceil_list)
    print(f"  Phase 1: {phase1_total} Dual combos on train "
          f"{TRAIN_START}~{TRAIN_END} ({len(df_train_base)} bars)")
    print(f"  Val period: {VAL_START}~latest ({len(df_val_base)} bars)")
    print(f"  BH train={bh_train:+.1f}%  val={bh_val:+.1f}%", flush=True)

    # ── 2. Momentum cache: compute once per (vb, fvb) pair ───────────────────
    momentum_cache = {}  # (vb, fvb) -> (df_train_slim, df_val_slim)
    for vb, fvb in valid_pairs:
        df_tm = _slim_df(add_momentum_indicators(
            df_train_base.copy(), velocity_bars=vb, fast_velocity_bars=fvb))
        df_vm = _slim_df(add_momentum_indicators(
            df_val_base.copy(),   velocity_bars=vb, fast_velocity_bars=fvb))
        momentum_cache[(vb, fvb)] = (df_tm, df_vm)

    # ── 3. Phase 1: Dual-direction grid on training set ───────────────────────
    d_results = []   # (tag, vb, fvb, prox, decel, vceil, result_train)
    count = 0
    for vb, fvb in valid_pairs:
        df_tm, _ = momentum_cache[(vb, fvb)]
        for prox in prox_list:
            for decel in decel_list:
                for vceil in vceil_list:
                    vfloor = -vceil if vceil > 0 else 0.0
                    tag = (f"vb{vb}_fvb{fvb}"
                           f"_px{prox*1000:.0f}"
                           f"_dc{decel:.1f}"
                           f"_vc{vceil:.1f}")
                    r = backtest(
                        df_tm, f"D_{tag}",
                        buy_cond=ReversalLongCond(prox, decel, velocity_ceil=vceil),
                        sell_cond=NearResistanceCond(0.002),
                        short_cond=ReversalShortCond(prox, decel, velocity_floor=vfloor),
                        cover_cond=NearSupportCond(0.002),
                        close_conditions=sl,
                    )
                    d_results.append((tag, vb, fvb, prox, decel, vceil, r))
                    count += 1
                    if count % 36 == 0 or count == phase1_total:
                        print(f"  Phase 1 progress: {count}/{phase1_total}", flush=True)

    # Sort by train Sharpe
    d_results.sort(key=lambda x: x[6]["sharpe"], reverse=True)
    top_d = [x for x in d_results if x[6]["trades"] > 0][:TOP_D]

    # ── 4. Phase 2: Validate top Dual configs on val set ─────────────────────
    print(f"\n  Phase 2: validating top {len(top_d)} Dual configs on val set...",
          flush=True)

    def _run_dual(df_slim, tag, prox, decel, vceil):
        vfloor = -vceil if vceil > 0 else 0.0
        return backtest(
            df_slim, f"D_{tag}",
            buy_cond=ReversalLongCond(prox, decel, velocity_ceil=vceil),
            sell_cond=NearResistanceCond(0.002),
            short_cond=ReversalShortCond(prox, decel, velocity_floor=vfloor),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        )

    val_d = []
    for tag, vb, fvb, prox, decel, vceil, tr in top_d:
        _, df_vm = momentum_cache[(vb, fvb)]
        vr = _run_dual(df_vm, tag, prox, decel, vceil)
        val_d.append((tag, vb, fvb, prox, decel, vceil, tr, vr))

    # ── 5. Phase 3: L / S breakdown for best 5 Dual configs ──────────────────
    # Rank val_d by combined score: val_sharpe × sign(val_ret)
    val_d_sorted = sorted(
        val_d,
        key=lambda x: x[7]["sharpe"] if x[7]["total_return"] > 0 else -99,
        reverse=True,
    )
    best5 = val_d_sorted[:5]

    print(f"\n  Phase 3: L/S breakdown for top-5 configs...", flush=True)
    ls_results = []  # (tag, vb, fvb, prox, decel, vceil, dir, tr, vr)
    for tag, vb, fvb, prox, decel, vceil, _, _ in best5:
        vfloor = -vceil if vceil > 0 else 0.0
        df_tm, df_vm = momentum_cache[(vb, fvb)]
        for direction in ["L", "S"]:
            if direction == "L":
                tr = backtest(df_tm, f"L_{tag}",
                              buy_cond=ReversalLongCond(prox, decel, velocity_ceil=vceil),
                              sell_cond=NearResistanceCond(0.002),
                              close_conditions=sl)
                vr = backtest(df_vm, f"L_{tag}",
                              buy_cond=ReversalLongCond(prox, decel, velocity_ceil=vceil),
                              sell_cond=NearResistanceCond(0.002),
                              close_conditions=sl)
            else:
                tr = backtest(df_tm, f"S_{tag}",
                              buy_cond=lambda row: False,
                              sell_cond=lambda row: False,
                              short_cond=ReversalShortCond(prox, decel, velocity_floor=vfloor),
                              cover_cond=NearSupportCond(0.002),
                              close_conditions=sl)
                vr = backtest(df_vm, f"S_{tag}",
                              buy_cond=lambda row: False,
                              sell_cond=lambda row: False,
                              short_cond=ReversalShortCond(prox, decel, velocity_floor=vfloor),
                              cover_cond=NearSupportCond(0.002),
                              close_conditions=sl)
            ls_results.append((tag, vb, fvb, prox, decel, vceil, direction, tr, vr))

    # ── 6. Print results ──────────────────────────────────────────────────────
    W = 114
    def _header(title, bh_t, bh_v):
        print(f"\n{'='*W}")
        print(f"  {title}   BH train={bh_t:+.1f}%  val={bh_v:+.1f}%")
        print(f"{'='*W}")
        print(f"  {'':2}{'vb':>4} {'fvb':>4} {'px':>5} {'dc':>5} {'vc':>5}"
              f"  {'TrRet':>7} {'TrShr':>6} {'TrMDD':>7} {'TrN':>5}"
              f"  {'VaRet':>7} {'VaShr':>6} {'VaMDD':>7} {'VaN':>5}")
        print("  " + "-" * (W - 2))

    def _row(tag, vb, fvb, prox, decel, vceil, tr, vr, prefix=""):
        ok = "*" if vr["sharpe"] > 0.3 and vr["total_return"] > 0 else " "
        print(f"  {ok}{prefix:1}{vb:>4} {fvb:>4} {prox*1000:>5.0f}"
              f" {decel:>5.1f} {vceil:>5.1f}"
              f"  {tr['total_return']:>6.1f}% {tr['sharpe']:>6.3f}"
              f" {tr['max_drawdown']:>6.1f}% {tr['trades']:>5}"
              f"  {vr['total_return']:>6.1f}% {vr['sharpe']:>6.3f}"
              f" {vr['max_drawdown']:>6.1f}% {vr['trades']:>5}")

    # Dual top-15
    _header("DUAL — Top-15 by train Sharpe (validated on val set)",
            bh_train, bh_val)
    for tag, vb, fvb, prox, decel, vceil, tr, vr in val_d:
        _row(tag, vb, fvb, prox, decel, vceil, tr, vr)

    # Re-sort by val Sharpe for the "best in val" view
    val_d_by_val = sorted(val_d,
                          key=lambda x: x[7]["sharpe"], reverse=True)
    _header("DUAL — Same configs sorted by VAL Sharpe",
            bh_train, bh_val)
    for tag, vb, fvb, prox, decel, vceil, tr, vr in val_d_by_val:
        _row(tag, vb, fvb, prox, decel, vceil, tr, vr)

    # L/S breakdown for best 5
    _header("LONG / SHORT breakdown for best-5 Dual configs",
            bh_train, bh_val)
    for tag, vb, fvb, prox, decel, vceil, direction, tr, vr in ls_results:
        _row(tag, vb, fvb, prox, decel, vceil, tr, vr, prefix=direction)

    return bh_train, bh_val


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


def run_fast_grid(df):
    """Grid search for fast norm_velocity filters on top of baseline."""
    b = BASELINE
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"])
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox = b["proximity_pct"]
    decel = b["min_decel"]
    v_ceil = b["velocity_ceil_long"]
    v_floor = b["velocity_floor_short"]
    sl = BASELINE_SL

    # Grid dimensions for fast velocity filters
    fast_ceil_long_list = [None, -0.5, 0, 0.5, 1.0]
    fast_floor_long_list = [None, -2.0, -1.0, -0.5]
    fast_decel_list = [None, 0.3, 0.5, 1.0]

    configs = []
    for fc in fast_ceil_long_list:
        for ff in fast_floor_long_list:
            for fd in fast_decel_list:
                # Skip all-None (already covered by baseline)
                if fc is None and ff is None and fd is None:
                    continue
                parts = []
                if fc is not None:
                    parts.append(f"fc{fc:+.1f}")
                if ff is not None:
                    parts.append(f"ff{ff:+.1f}")
                if fd is not None:
                    parts.append(f"fd{fd:.1f}")
                tag = "_".join(parts)

                long_kw = {"velocity_ceil": v_ceil}
                short_kw = {"velocity_floor": v_floor}

                if fc is not None:
                    long_kw["fast_velocity_ceil"] = fc
                    short_kw["fast_velocity_floor"] = -fc  # symmetric
                if ff is not None:
                    long_kw["fast_velocity_floor"] = ff
                    short_kw["fast_velocity_ceil"] = -ff  # symmetric
                if fd is not None:
                    long_kw["fast_min_decel"] = fd
                    short_kw["fast_min_decel"] = fd

                configs.append({"name": tag, "long": long_kw, "short": short_kw})

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


def run_vp_grid(df):
    """Grid search over VP window combinations with baseline + fast velocity params."""
    b = BASELINE
    bh_ret = bh_1y = 0

    prox = b["proximity_pct"]
    decel = b["min_decel"]
    v_ceil = b["velocity_ceil_long"]
    v_floor = b["velocity_floor_short"]
    sl = BASELINE_SL

    # VP window combinations to test
    vp_combos = [
        # Single windows
        [5], [7], [10], [14], [21], [30], [45], [60], [90],
        # Dual windows
        [5, 14], [7, 14], [7, 21], [7, 30], [10, 30], [14, 30], [14, 45],
        # Triple windows (baseline + variants)
        [3, 7, 14], [5, 10, 21], [5, 14, 30], [7, 14, 30],
        [7, 21, 45], [3, 7, 30], [10, 14, 30],
        # Triple with long-period windows
        [7, 14, 60], [7, 30, 90], [14, 30, 60], [7, 14, 90],
        # Quad windows
        [3, 7, 14, 30], [5, 7, 14, 30], [7, 10, 14, 30], [5, 10, 21, 45],
        # Quad with long-period windows
        [7, 14, 30, 60], [7, 14, 30, 90], [7, 14, 60, 90], [7, 30, 60, 90],
        # Five windows
        [7, 14, 30, 60, 90],
    ]

    # Param sets: baseline, fast, and reward-filtered variants
    param_sets = [
        {"name": "base",
         "long": {"velocity_ceil": v_ceil},
         "short": {"velocity_floor": v_floor}},
        {"name": "fast",
         "long": {"velocity_ceil": v_ceil,
                  "fast_velocity_ceil": 0.5, "fast_velocity_floor": -1.0,
                  "fast_min_decel": 0.5},
         "short": {"velocity_floor": v_floor,
                   "fast_velocity_floor": -0.5, "fast_velocity_ceil": 1.0,
                   "fast_min_decel": 0.5}},
        {"name": "base_rw1",
         "long": {"velocity_ceil": v_ceil, "min_reward_pct": 0.01},
         "short": {"velocity_floor": v_floor, "min_reward_pct": 0.01}},
        {"name": "base_rw2",
         "long": {"velocity_ceil": v_ceil, "min_reward_pct": 0.02},
         "short": {"velocity_floor": v_floor, "min_reward_pct": 0.02}},
        {"name": "fast_rw1",
         "long": {"velocity_ceil": v_ceil,
                  "fast_velocity_ceil": 0.5, "fast_velocity_floor": -1.0,
                  "fast_min_decel": 0.5, "min_reward_pct": 0.01},
         "short": {"velocity_floor": v_floor,
                   "fast_velocity_floor": -0.5, "fast_velocity_ceil": 1.0,
                   "fast_min_decel": 0.5, "min_reward_pct": 0.01}},
        {"name": "fast_rw2",
         "long": {"velocity_ceil": v_ceil,
                  "fast_velocity_ceil": 0.5, "fast_velocity_floor": -1.0,
                  "fast_min_decel": 0.5, "min_reward_pct": 0.02},
         "short": {"velocity_floor": v_floor,
                   "fast_velocity_floor": -0.5, "fast_velocity_ceil": 1.0,
                   "fast_min_decel": 0.5, "min_reward_pct": 0.02}},
    ]

    all_results = []
    total = len(vp_combos) * len(param_sets) * 3
    count = 0

    for vp_wins in vp_combos:
        vp_tag = "vp" + "_".join(str(w) for w in vp_wins)
        df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"],
                                vp_windows=vp_wins)
        bh_ret, bh_1y = _compute_bh(df_slim)

        for ps in param_sets:
            tag = f"{vp_tag}_{ps['name']}"
            long_kw = ps["long"]
            short_kw = ps["short"]

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


def run_sl_optimization(df):
    """
    Walk-forward stop-loss optimization for 1h Baseline_D.

    固定入场参数（baseline: prox=0.5%, decel=1.5, vb=8, vc=0.5）
    只在止损机制上做网格搜索，walk-forward验证防止过拟合。

    Train: 2021-01-01 ~ 2025-05-31
    Val  : 2025-06-01 ~ present

    网格维度：
      Phase A - fixed_sl  : 无 / -1% / -1.5% / -2% / -3% / -4% / -5%  × time_stop
      Phase B - sr_sl     : SR(buffer, max_loss) 多个组合  × time_stop（多空分离）
      Phase C - atr_sl    : ATR(1.5x/2x/3x)  × time_stop
      Phase D - trailing  : Trail(3/5/8/10%)  × T24h/noT（多头追踪，空头等幅固定）
      Phase E - tp+sl     : TP(1.5/2/3/5%) + SL4%  × T24h/noT
      time_stop : 无 / 24h / 48h / 72h / 96h

    对每个组合统计：收益/夏普/最大回撤/止损触发率(SL%)
    """
    TRAIN_START = "2021-01-01"
    TRAIN_END   = "2025-05-31"
    VAL_START   = "2025-06-01"

    b    = BASELINE
    prox   = b["proximity_pct"]
    decel  = b["min_decel"]
    v_ceil = b["velocity_ceil_long"]
    v_floor= b["velocity_floor_short"]

    print("  Building S/R + momentum dataset...", flush=True)
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"])

    df_train = df_slim[df_slim["date"] <= TRAIN_END].reset_index(drop=True)
    df_train.attrs = df_slim.attrs.copy()
    df_val   = df_slim[df_slim["date"] >= VAL_START].reset_index(drop=True)
    df_val.attrs = df_slim.attrs.copy()

    bh_train = (df_train["close"].iloc[-1] / df_train["close"].iloc[0] - 1) * 100
    bh_val   = (df_val["close"].iloc[-1]   / df_val["close"].iloc[0]   - 1) * 100

    buy_cond   = ReversalLongCond(prox, decel, velocity_ceil=v_ceil)
    short_cond = ReversalShortCond(prox, decel, velocity_floor=v_floor)
    sell_cond  = NearResistanceCond(0.002)
    cover_cond = NearSupportCond(0.002)

    # ── 止损配置网格 ──────────────────────────────────────────────────────────
    # 每个 config: (name, long_sl_list, short_sl_list)
    sl_configs = []

    fixed_sls   = [None, -0.010, -0.015, -0.020, -0.030, -0.040, -0.050]
    time_stops  = [None, 24, 48, 72, 96]
    sr_configs  = [
        None,
        (0.002, 0.02),   # buffer=0.2%, max=2%
        (0.003, 0.03),   # buffer=0.3%, max=3%
        (0.005, 0.04),   # buffer=0.5%, max=4%
    ]
    atr_mults   = [None, 1.5, 2.0, 3.0]
    trail_pcts  = [0.03, 0.05, 0.08, 0.10]   # Phase D: trailing stop thresholds
    tp_pcts     = [0.015, 0.02, 0.03, 0.05]  # Phase E: take-profit targets

    # Phase A: fixed SL × time stop  (no SR/ATR)
    for fsl in fixed_sls:
        for ts in time_stops:
            long_conds  = []
            short_conds = []
            if fsl is not None:
                long_conds.append(StopLossCond(fsl))
                short_conds.append(StopLossCond(fsl))
            if ts is not None:
                long_conds.append(TimeBarStopCond(ts))
                short_conds.append(TimeBarStopCond(ts))
            fsl_str = f"SL{abs(fsl)*100:.0f}%" if fsl else "noSL"
            ts_str  = f"T{ts}h" if ts else "noT"
            sl_configs.append((f"{fsl_str}+{ts_str}", long_conds, short_conds))

    # Phase B: SR stop × time stop  (long/short分离，修复空头SR止损)
    for sr in sr_configs[1:]:   # skip None
        for ts in time_stops:
            buf, mx = sr
            long_conds  = [SRStopLossLong(buf, mx)]
            short_conds = [SRStopLossShort(buf, mx)]
            if ts is not None:
                long_conds.append(TimeBarStopCond(ts))
                short_conds.append(TimeBarStopCond(ts))
            sr_str = f"SR(b{buf*1000:.0f}_x{mx*100:.0f}%)"
            ts_str = f"T{ts}h" if ts else "noT"
            sl_configs.append((f"{sr_str}+{ts_str}", long_conds, short_conds))

    # Phase C: ATR stop × time stop  (no fixed SL)
    for atr in atr_mults[1:]:
        for ts in time_stops:
            long_conds  = [ATRStopLossCond(atr)]
            short_conds = [ATRStopLossCond(atr)]
            if ts is not None:
                long_conds.append(TimeBarStopCond(ts))
                short_conds.append(TimeBarStopCond(ts))
            atr_str = f"ATR{atr:.1f}x"
            ts_str  = f"T{ts}h" if ts else "noT"
            sl_configs.append((f"{atr_str}+{ts_str}", long_conds, short_conds))

    # Phase D: 追踪止损 × T24h/noT（仅多头有效，空头引擎返回False）
    for tr in trail_pcts:
        for ts in [None, 24]:
            long_conds  = [TrailingStopCond(-tr)]
            short_conds = [StopLossCond(-tr)]   # 空头用等幅固定止损替代
            if ts is not None:
                long_conds.append(TimeBarStopCond(ts))
                short_conds.append(TimeBarStopCond(ts))
            tr_str = f"Trail{tr*100:.0f}%"
            ts_str = f"T{ts}h" if ts else "noT"
            sl_configs.append((f"{tr_str}+{ts_str}", long_conds, short_conds))

    # Phase E: 止盈 + SL4% + T24h/noT 组合
    best_sl = -0.04   # 依据Phase A最优
    for tp in tp_pcts:
        for ts in [None, 24]:
            long_conds  = [TakeProfitCond(tp), StopLossCond(best_sl)]
            short_conds = [TakeProfitCond(tp), StopLossCond(best_sl)]
            if ts is not None:
                long_conds.append(TimeBarStopCond(ts))
                short_conds.append(TimeBarStopCond(ts))
            tp_str = f"TP{tp*100:.1f}%+SL4%"
            ts_str = f"T{ts}h" if ts else "noT"
            sl_configs.append((f"{tp_str}+{ts_str}", long_conds, short_conds))

    # Phase F: 棘轮止损（两阶段：初始硬止损 + 激活后追踪）
    # 目标：减少每笔损失幅度，同时不截断赢家
    # hard_sl × activate_pct × trail_pct × T24h/noT
    ratchet_params = [
        (-0.010, 0.005, 0.025),  # SL1%，激活+0.5%，追踪2.5%
        (-0.015, 0.005, 0.030),  # SL1.5%，激活+0.5%，追踪3%
        (-0.015, 0.010, 0.030),  # SL1.5%，激活+1%，追踪3%
        (-0.020, 0.005, 0.030),  # SL2%，激活+0.5%，追踪3%
        (-0.020, 0.010, 0.040),  # SL2%，激活+1%，追踪4%
        (-0.025, 0.005, 0.040),  # SL2.5%，激活+0.5%，追踪4%
        (-0.030, 0.005, 0.040),  # SL3%，激活+0.5%，追踪4%
        (-0.030, 0.010, 0.040),  # SL3%，激活+1%，追踪4%
    ]
    for hsl, act, tr in ratchet_params:
        for ts in [None, 24]:
            rc = RatchetStopCond(hard_sl=hsl, activate_pct=act, trail_pct=tr)
            long_conds  = [rc]
            short_conds = [rc]
            if ts is not None:
                long_conds  = [rc, TimeBarStopCond(ts)]
                short_conds = [rc, TimeBarStopCond(ts)]
            name = (f"Ratchet(sl{abs(hsl)*100:.0f}%"
                    f"_a{act*100:.1f}%_tr{tr*100:.0f}%)+{('T24h' if ts else 'noT')}")
            sl_configs.append((name, long_conds, short_conds))

    total = len(sl_configs)
    print(f"  Phase 1: {total} Dual combos  train {TRAIN_START}~{TRAIN_END}"
          f"  /  val {VAL_START}~present", flush=True)
    print(f"  BH train={bh_train:+.1f}%  val={bh_val:+.1f}%")

    def _sl_rate(result):
        """止损触发率 = SL类型退出 / 总退出次数（source字段）"""
        trades = result.get("trade_log", [])
        exits  = [t for t in trades if t["type"] in ("SELL", "COVER")]
        if not exits:
            return 0.0
        sl_exits = [t for t in exits
                    if any(kw in str(t.get("source", ""))
                           for kw in ("SL", "ATR", "SR_SL", "Time"))]
        return len(sl_exits) / len(exits)

    # ── 训练集扫描 ────────────────────────────────────────────────────────────
    train_results = []
    for i, (name, long_sl, short_sl) in enumerate(sl_configs, 1):
        res = backtest(
            df_train, f"D_{name}",
            buy_cond=buy_cond,   sell_cond=sell_cond,
            short_cond=short_cond, cover_cond=cover_cond,
            close_conditions=long_sl or None,
            short_close_conditions=short_sl or None,
        )
        train_results.append((name, long_sl, short_sl, res))
        if i % max(1, total // 5) == 0 or i == total:
            print(f"  Phase 1: {i}/{total}", flush=True)

    # 按训练夏普排序，取 top-20
    train_results.sort(key=lambda x: x[3]["sharpe"], reverse=True)
    top20 = train_results[:20]

    # ── 验证集 ────────────────────────────────────────────────────────────────
    print(f"\n  Phase 2: validating top-20 configs...", flush=True)
    val_results = []
    for name, long_sl, short_sl, res_t in top20:
        res_v = backtest(
            df_val, f"D_{name}",
            buy_cond=buy_cond,   sell_cond=sell_cond,
            short_cond=short_cond, cover_cond=cover_cond,
            close_conditions=long_sl or None,
            short_close_conditions=short_sl or None,
        )
        val_results.append((name, res_t, res_v))

    # ── 打印结果表 ────────────────────────────────────────────────────────────
    hdr = (f"\n{'':=<130}\n"
           f"  SL Optimization — Top-20 by Train Sharpe (validated)"
           f"   BH train={bh_train:+.1f}%  val={bh_val:+.1f}%\n"
           f"{'':=<130}")
    print(hdr)
    col = (f"  {'config':<28} "
           f"{'TrRet':>7} {'TrShr':>6} {'TrMDD':>7} {'TrN':>5} {'TrSL%':>6}  "
           f"{'VaRet':>7} {'VaShr':>6} {'VaMDD':>7} {'VaN':>5} {'VaSL%':>6}")
    print(col)
    print(f"  {'-'*128}")

    for name, res_t, res_v in val_results:
        sl_t = _sl_rate(res_t) * 100
        sl_v = _sl_rate(res_v) * 100
        star = "*" if res_v["sharpe"] > res_t["sharpe"] * 0.7 else " "
        print(
            f"  {star} {name:<28} "
            f"{res_t['total_return']:>7.1f}% {res_t['sharpe']:>6.3f} "
            f"{res_t['max_drawdown']:>7.1f}% {res_t['trades']:>5} {sl_t:>5.1f}%  "
            f"{res_v['total_return']:>7.1f}% {res_v['sharpe']:>6.3f} "
            f"{res_v['max_drawdown']:>7.1f}% {res_v['trades']:>5} {sl_v:>5.1f}%"
        )

    # val Sharpe 排序
    val_results_sorted = sorted(val_results, key=lambda x: x[2]["sharpe"], reverse=True)
    print(f"\n{'':=<130}")
    print(f"  SL Optimization — Same configs sorted by Val Sharpe")
    print(f"{'':=<130}")
    print(col)
    print(f"  {'-'*128}")
    for name, res_t, res_v in val_results_sorted:
        sl_t = _sl_rate(res_t) * 100
        sl_v = _sl_rate(res_v) * 100
        star = "*" if res_v["sharpe"] > res_t["sharpe"] * 0.7 else " "
        print(
            f"  {star} {name:<28} "
            f"{res_t['total_return']:>7.1f}% {res_t['sharpe']:>6.3f} "
            f"{res_t['max_drawdown']:>7.1f}% {res_t['trades']:>5} {sl_t:>5.1f}%  "
            f"{res_v['total_return']:>7.1f}% {res_v['sharpe']:>6.3f} "
            f"{res_v['max_drawdown']:>7.1f}% {res_v['trades']:>5} {sl_v:>5.1f}%"
        )

    # ── 最优配置年度明细 ──────────────────────────────────────────────────────
    best_name, best_res_t, best_res_v = val_results_sorted[0]
    # 在全集上重跑最优配置
    best_cfg = next((x for x in sl_configs if x[0] == best_name), None)
    if best_cfg:
        _, long_sl, _ = best_cfg
        res_full = backtest(
            df_slim, f"BestSL_{best_name}",
            buy_cond=buy_cond,   sell_cond=sell_cond,
            short_cond=short_cond, cover_cond=cover_cond,
            close_conditions=long_sl,
        )
        print(f"\n  Best Val config: {best_name}")
        print(f"  Full period: {res_full['total_return']:.1f}%  "
              f"Sharpe={res_full['sharpe']:.3f}  MaxDD={res_full['max_drawdown']:.1f}%  "
              f"Trades={res_full['trades']}")
        print_yearly_breakdown(res_full, bars_per_day=24)

    # 与原 baseline 对比
    print(f"\n  --- Baseline_D reference ---")
    ref = backtest(
        df_slim, "Baseline_D_ref",
        buy_cond=buy_cond,   sell_cond=sell_cond,
        short_cond=short_cond, cover_cond=cover_cond,
        close_conditions=BASELINE_SL,
    )
    sl_ref = _sl_rate(ref) * 100
    print(f"  Full period: {ref['total_return']:.1f}%  "
          f"Sharpe={ref['sharpe']:.3f}  MaxDD={ref['max_drawdown']:.1f}%  "
          f"Trades={ref['trades']}  SL%={sl_ref:.1f}%")
    print_yearly_breakdown(ref, bars_per_day=24)


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


def run_wbottom_grid(df):
    """Grid search for W-bottom/M-top microstructure filter thresholds."""
    b = BASELINE
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"])
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox = b["proximity_pct"]
    decel = b["min_decel"]
    v_ceil = b["velocity_ceil_long"]
    v_floor = b["velocity_floor_short"]
    sl = BASELINE_SL

    strength_thresholds = [0.3, 0.5, 0.7, 1.0]
    all_results = []
    count = 0

    # Baseline (no W/M filter) as reference
    for direction in ["L", "S", "D"]:
        tag = "NoWM"
        if direction == "L":
            all_results.append(backtest(
                df_slim, f"L_{tag}",
                buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
                sell_cond=NearResistanceCond(0.002),
                close_conditions=sl,
            ))
        elif direction == "S":
            all_results.append(backtest(
                df_slim, f"S_{tag}",
                buy_cond=lambda row: False,
                sell_cond=lambda row: False,
                short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor),
                cover_cond=NearSupportCond(0.002),
                close_conditions=sl,
            ))
        else:
            all_results.append(backtest(
                df_slim, f"D_{tag}",
                buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
                sell_cond=NearResistanceCond(0.002),
                short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor),
                cover_cond=NearSupportCond(0.002),
                close_conditions=sl,
            ))
        count += 1

    # Grid: W-bottom strength thresholds for Long, M-top for Short
    total = 3 + len(strength_thresholds) * 3 * 2  # baseline + thresholds × directions × (w/m)
    for thresh in strength_thresholds:
        tag_w = f"W{thresh:.1f}"
        tag_m = f"M{thresh:.1f}"
        tag_wm = f"WM{thresh:.1f}"

        # Long with W-bottom filter
        all_results.append(backtest(
            df_slim, f"L_{tag_w}",
            buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil,
                                      min_w_strength=thresh),
            sell_cond=NearResistanceCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        # Short with M-top filter
        all_results.append(backtest(
            df_slim, f"S_{tag_m}",
            buy_cond=lambda row: False,
            sell_cond=lambda row: False,
            short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor,
                                         min_m_strength=thresh),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        # Dual with both filters
        all_results.append(backtest(
            df_slim, f"D_{tag_wm}",
            buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil,
                                      min_w_strength=thresh),
            sell_cond=NearResistanceCond(0.002),
            short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor,
                                         min_m_strength=thresh),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        # Long with W filter only (no M on short side) in Dual
        all_results.append(backtest(
            df_slim, f"D_Wonly{thresh:.1f}",
            buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil,
                                      min_w_strength=thresh),
            sell_cond=NearResistanceCond(0.002),
            short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        # Short with M filter only (no W on long side) in Dual
        all_results.append(backtest(
            df_slim, f"D_Monly{thresh:.1f}",
            buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
            sell_cond=NearResistanceCond(0.002),
            short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor,
                                         min_m_strength=thresh),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        ))
        count += 1

        # Asymmetric: different thresholds for W vs M
        for thresh2 in strength_thresholds:
            if thresh2 == thresh:
                continue
            all_results.append(backtest(
                df_slim, f"D_W{thresh:.1f}M{thresh2:.1f}",
                buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil,
                                          min_w_strength=thresh),
                sell_cond=NearResistanceCond(0.002),
                short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor,
                                             min_m_strength=thresh2),
                cover_cond=NearSupportCond(0.002),
                close_conditions=sl,
            ))
            count += 1

        if count % 10 == 0:
            print(f"  Progress: {count} combos done")

    print(f"  Total: {count} combos")
    return all_results, bh_ret, bh_1y


def _win_rate_from_log(trade_log: list) -> float:
    """Compute overall win rate (%) from a trade log list."""
    pairs = []
    for i, t in enumerate(trade_log):
        if t["type"] == "SELL":
            for j in range(i - 1, -1, -1):
                if trade_log[j]["type"] == "BUY":
                    pnl = (t["price"] / trade_log[j]["price"] - 1) * 100
                    pairs.append(pnl)
                    break
        elif t["type"] == "COVER":
            for j in range(i - 1, -1, -1):
                if trade_log[j]["type"] == "SHORT":
                    pnl = (trade_log[j]["price"] / t["price"] - 1) * 100
                    pairs.append(pnl)
                    break
    if not pairs:
        return 0.0
    return len([p for p in pairs if p > 0]) / len(pairs) * 100


def run_sensitivity(df):
    """One-at-a-time parameter sensitivity analysis for the baseline strategy.

    Varies each parameter individually while holding all others at baseline values.
    Reports Dual direction results only (the proven best).
    Output: one formatted table per parameter showing Return, Sharpe, MaxDD,
    1Y-Sharpe, Trades, WinRate. Baseline value marked with *.
    """
    b = BASELINE
    BASELINE_EXIT_PROX = 0.002  # NearResistanceCond / NearSupportCond proximity

    def _m(r):
        """Extract key metrics dict from a backtest result.
        Note: total_return / max_drawdown / return_1y are already in % form."""
        return {
            "ret":       r["total_return"],
            "sharpe":    r["sharpe"],
            "maxdd":     r["max_drawdown"],
            "ret_1y":    r.get("return_1y") or 0,
            "sharpe_1y": r.get("sharpe_1y") or 0,
            "trades":    r["trades"],
            "wr":        _win_rate_from_log(r.get("trade_log", [])),
        }

    def _print_table(param_name, baseline_val, rows):
        SEP = "=" * 80
        print(f"\n{SEP}")
        print(f"  SENSITIVITY: {param_name}  (baseline = {baseline_val})")
        print(SEP)
        hdr = (f"  {'Value':<14} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8}"
               f" {'1Y Ret':>9} {'1Y Shrp':>8} {'Trades':>7} {'WinRate':>8}")
        print(hdr)
        print("  " + "-" * 78)
        for val, m in rows:
            mark = "* " if val == baseline_val else "  "
            val_s = f"{val}" if not isinstance(val, float) else f"{val:.4g}"
            print(f"{mark}{val_s:<14} {m['ret']:>8.1f}% {m['sharpe']:>7.3f}"
                  f" {m['maxdd']:>7.1f}% {m['ret_1y']:>8.1f}% {m['sharpe_1y']:>7.3f}"
                  f" {m['trades']:>7} {m['wr']:>7.1f}%")

    def _dual(df_slim, prox, decel, v_ceil, ep, sl_conds):
        return backtest(
            df_slim, "D",
            buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
            sell_cond=NearResistanceCond(ep),
            short_cond=ReversalShortCond(prox, decel, velocity_floor=-v_ceil),
            cover_cond=NearSupportCond(ep),
            close_conditions=sl_conds,
        )

    # Pre-compute S/R levels once (reused for all params except cluster_gap_pct)
    df_sr = add_sr_levels(df.copy(), vp_windows=VP_WINDOWS,
                          cluster_gap_pct=b["cluster_gap_pct"])
    df_bt = df_sr[df_sr["date"] >= "2021-01-01"].reset_index(drop=True)
    df_bt.attrs = df_sr.attrs.copy()
    bh_ret, bh_1y = _compute_bh(df_bt)

    # Pre-compute momentum indicators at baseline velocity_bars
    df_slim_base = _slim_df(add_momentum_indicators(df_bt.copy(),
                                                     velocity_bars=b["velocity_bars"]))

    prox0  = b["proximity_pct"]
    decel0 = b["min_decel"]
    vceil0 = b["velocity_ceil_long"]
    ep0    = BASELINE_EXIT_PROX
    sl0    = BASELINE_SL

    # ------------------------------------------------------------------
    # 1. velocity_bars  (requires re-computing momentum indicators)
    # ------------------------------------------------------------------
    print("\n[Sensitivity 1/8] velocity_bars ...")
    vb_list = [2, 4, 6, 8, 10, 12, 16, 20, 24]
    vb_rows = []
    for vb in vb_list:
        df_m = _slim_df(add_momentum_indicators(df_bt.copy(), velocity_bars=vb))
        vb_rows.append((vb, _m(_dual(df_m, prox0, decel0, vceil0, ep0, sl0))))
    _print_table("velocity_bars  (1h bars)", b["velocity_bars"], vb_rows)

    # ------------------------------------------------------------------
    # 2. proximity_pct  (reuse base slim df)
    # ------------------------------------------------------------------
    print("\n[Sensitivity 2/8] proximity_pct ...")
    prox_list = [0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]
    prox_rows = []
    for prox in prox_list:
        prox_rows.append((prox, _m(_dual(df_slim_base, prox, decel0, vceil0, ep0, sl0))))
    _print_table("proximity_pct  (entry S/R distance)", prox0, prox_rows)

    # ------------------------------------------------------------------
    # 3. min_decel
    # ------------------------------------------------------------------
    print("\n[Sensitivity 3/8] min_decel ...")
    decel_list = [0.30, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]
    decel_rows = []
    for dc in decel_list:
        decel_rows.append((dc, _m(_dual(df_slim_base, prox0, dc, vceil0, ep0, sl0))))
    _print_table("min_decel  (velocity_delta threshold)", decel0, decel_rows)

    # ------------------------------------------------------------------
    # 4. velocity_ceil_long  (and symmetric velocity_floor_short = -ceil)
    # ------------------------------------------------------------------
    print("\n[Sensitivity 4/8] velocity_ceil_long ...")
    vceil_list = [-1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0]
    vceil_rows = []
    for vc in vceil_list:
        vceil_rows.append((vc, _m(_dual(df_slim_base, prox0, decel0, vc, ep0, sl0))))
    _print_table("velocity_ceil_long  (+-symmetric, entry nv bound)", vceil0, vceil_rows)

    # ------------------------------------------------------------------
    # 5. stop_loss_threshold  (keep TimeBarStop=48 as partner)
    # ------------------------------------------------------------------
    print("\n[Sensitivity 5/8] stop_loss_threshold ...")
    sl_list = [-0.005, -0.010, -0.015, -0.020, -0.025, -0.030, -0.040, -0.050]
    sl_rows = []
    for sl_thresh in sl_list:
        sl_conds = [StopLossCond(sl_thresh), TimeBarStopCond(48)]
        sl_rows.append((sl_thresh, _m(_dual(df_slim_base, prox0, decel0, vceil0, ep0, sl_conds))))
    _print_table("stop_loss_threshold  (with Time48)", -0.02, sl_rows)

    # ------------------------------------------------------------------
    # 6. time_stop_bars  (paired with -2% fixed SL)
    # ------------------------------------------------------------------
    print("\n[Sensitivity 6/8] time_stop_bars ...")
    ts_list = [None, 12, 24, 36, 48, 72, 96, 120, 168]
    ts_rows = []
    for ts in ts_list:
        sl_conds = ([StopLossCond(-0.02), TimeBarStopCond(ts)]
                    if ts is not None else [StopLossCond(-0.02)])
        ts_rows.append((ts, _m(_dual(df_slim_base, prox0, decel0, vceil0, ep0, sl_conds))))
    _print_table("time_stop_bars  (with -2% SL; None = disabled)", 48, ts_rows)

    # ------------------------------------------------------------------
    # 7. exit_proximity  (NearResistanceCond / NearSupportCond)
    # ------------------------------------------------------------------
    print("\n[Sensitivity 7/8] exit_proximity ...")
    ep_list = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]
    ep_rows = []
    for ep in ep_list:
        ep_rows.append((ep, _m(_dual(df_slim_base, prox0, decel0, vceil0, ep, sl0))))
    _print_table("exit_proximity  (NearRes/Sup trigger distance)", ep0, ep_rows)

    # ------------------------------------------------------------------
    # 8. cluster_gap_pct  (requires re-computing S/R levels)
    # ------------------------------------------------------------------
    print("\n[Sensitivity 8/8] cluster_gap_pct ...")
    cgap_list = [0.005, 0.007, 0.010, 0.015, 0.020, 0.030]
    cgap_rows = []
    for cgap in cgap_list:
        df_slim = _prepare_data(df, cgap, b["velocity_bars"])
        cgap_rows.append((cgap, _m(_dual(df_slim, prox0, decel0, vceil0, ep0, sl0))))
    _print_table("cluster_gap_pct  (S/R level merging gap)", b["cluster_gap_pct"], cgap_rows)

    return bh_ret, bh_1y


# ============================================================================
# VOLUME-CLOCK BASELINE CONFIG
# ============================================================================

# Volume-clock baseline: velocity_vol_mult=8 ≈ 8× median hourly volume ≈ 8 h
# at normal activity.  SL=-2% + Time48 same as time-based baseline.
VOL_BASELINE = {
    "velocity_vol_mult": 8.0,     # ≈ 8h of typical hourly volume
    "fast_vol_mult": 2.0,          # ≈ 2h of typical hourly volume
    "vol_ref_period": 168,         # 1-week rolling median for reference vol
    "proximity_pct": 0.005,
    "min_decel": 1.5,
    "cluster_gap_pct": 0.01,
    "velocity_ceil_long": 0.5,
    "velocity_floor_short": -0.5,
}
VOL_BASELINE_SL = [StopLossCond(-0.02), TimeBarStopCond(48)]

TRAIN_START_VOL = "2021-01-01"
TRAIN_END_VOL   = "2025-05-31"
VAL_START_VOL   = "2025-06-01"


# ============================================================================
# TREND REGIME BASELINE (conditional long/short switching)
# ============================================================================

REGIME_METHODS = {
    "supertrend": {"period": 10, "multiplier": 3.0},
    # Add more methods here as needed
}


def run_regime_baseline(df, method: str = "supertrend", **regime_kwargs):
    """
    Run S/R Baseline (1h) with daily trend regime gating.

    Long entries  → only when daily regime == +1 (bull)
    Short entries → only when daily regime == -1 (bear)

    Regime computed from daily OHLCV resampled from the 1h data.
    shift_days=1: no lookahead (yesterday's close → today's entries).

    Prints comparison:
      - Baseline_D       (original, unrestricted)
      - Regime_{method}_D (regime-gated dual)
      - Regime_{method}_L (regime-gated long)
      - Regime_{method}_S (regime-gated short)

    Args:
        df     : 1h DataFrame from load_data(interval='1h').
        method : Regime method ('supertrend', 'ma200', 'ma_cross', 'momentum').
    """
    defaults = REGIME_METHODS.get(method, {})
    params   = {**defaults, **regime_kwargs}

    b       = BASELINE
    df_slim = _prepare_data(df, b["cluster_gap_pct"], b["velocity_bars"])
    bh_ret, bh_1y = _compute_bh(df_slim)

    # Compute daily regime from resampled 1h data
    daily_ohlcv   = resample_to_daily(df_slim, date_col="date")
    regime_series = compute_regime(daily_ohlcv, method=method, **params)
    regime_arr    = map_regime_to_bars(regime_series, df_slim,
                                       date_col="date", shift_days=1)
    df_slim = df_slim.copy()
    df_slim["regime"] = regime_arr

    bull_pct = (regime_arr == 1).mean() * 100
    bear_pct = (regime_arr == -1).mean() * 100
    print(f"  Regime ({method} {params}): "
          f"bull={bull_pct:.1f}%  bear={bear_pct:.1f}%")

    prox    = b["proximity_pct"]
    decel   = b["min_decel"]
    v_ceil  = b["velocity_ceil_long"]
    v_floor = b["velocity_floor_short"]
    sl      = BASELINE_SL

    results = []

    # Original baseline (no regime filter)
    results.append(backtest(
        df_slim, "Baseline_D",
        buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
        sell_cond=NearResistanceCond(0.002),
        short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor),
        cover_cond=NearSupportCond(0.002),
        close_conditions=sl,
    ))

    # Regime-gated Dual
    results.append(backtest(
        df_slim, f"Regime_{method}_D",
        buy_cond=RegimeLongCond(
            ReversalLongCond(prox, decel, velocity_ceil=v_ceil)),
        sell_cond=NearResistanceCond(0.002),
        short_cond=RegimeShortCond(
            ReversalShortCond(prox, decel, velocity_floor=v_floor)),
        cover_cond=NearSupportCond(0.002),
        close_conditions=sl,
    ))

    # Regime-gated Long only
    results.append(backtest(
        df_slim, f"Regime_{method}_L",
        buy_cond=RegimeLongCond(
            ReversalLongCond(prox, decel, velocity_ceil=v_ceil)),
        sell_cond=NearResistanceCond(0.002),
        close_conditions=sl,
    ))

    # Regime-gated Short only
    results.append(backtest(
        df_slim, f"Regime_{method}_S",
        buy_cond=lambda row: False,
        sell_cond=lambda row: False,
        short_cond=RegimeShortCond(
            ReversalShortCond(prox, decel, velocity_floor=v_floor)),
        cover_cond=NearSupportCond(0.002),
        close_conditions=sl,
    ))

    return results, bh_ret, bh_1y


def run_regime_baseline_15m(df, method: str = "supertrend", **regime_kwargs):
    """
    Run S/R Baseline_15m with daily trend regime gating.

    Long entries  → only when daily regime == +1 (bull)
    Short entries → only when daily regime == -1 (bear)

    Regime is computed from daily OHLCV resampled from the 15m data.
    Uses shift_days=1: today's bars see yesterday's daily close regime
    (no lookahead across day boundary).

    Prints comparison table:
      - Original Baseline_15m_D   (unrestricted dual)
      - Regime_15m_D              (regime-gated dual)
      - Regime_15m_L only
      - Regime_15m_S only

    Args:
        df         : 15m DataFrame from load_data(interval='15m').
        method     : Regime method ('supertrend', 'ma200', 'ma_cross', 'momentum').
        **regime_kwargs: Override default params for the chosen method.
    """
    defaults = REGIME_METHODS.get(method, {})
    params   = {**defaults, **regime_kwargs}

    b       = BASELINE_15M
    df_slim = _prepare_data(
        df,
        b["cluster_gap_pct"],
        b["velocity_bars"],
        backtest_start="2021-01-01",
        vp_windows=b.get("vp_windows"),
        fast_velocity_bars=b["fast_velocity_bars"],
    )
    bh_ret, bh_1y = _compute_bh(df_slim)

    # ── Compute daily regime from resampled 15m data ─────────────────────────
    daily_ohlcv = resample_to_daily(df_slim, date_col="date")
    regime_series = compute_regime(daily_ohlcv, method=method, **params)

    # shift_days=1: no lookahead (yesterday's daily close → today's bars)
    regime_arr = map_regime_to_bars(regime_series, df_slim,
                                    date_col="date", shift_days=1)
    df_slim = df_slim.copy()
    df_slim["regime"] = regime_arr

    # Report regime distribution
    bull_pct = (regime_arr == 1).mean() * 100
    bear_pct = (regime_arr == -1).mean() * 100
    print(f"  Regime ({method} {params}): "
          f"bull={bull_pct:.1f}%  bear={bear_pct:.1f}%")

    prox    = b["proximity_pct"]
    decel   = b["min_decel"]
    v_ceil  = b["velocity_ceil_long"]
    v_floor = b["velocity_floor_short"]
    sl      = BASELINE_15M_SL

    results = []

    # Original baseline (no regime filter) for comparison
    results.append(backtest(
        df_slim, "Baseline_15m_D",
        buy_cond=ReversalLongCond(prox, decel, velocity_ceil=v_ceil),
        sell_cond=NearResistanceCond(0.002),
        short_cond=ReversalShortCond(prox, decel, velocity_floor=v_floor),
        cover_cond=NearSupportCond(0.002),
        close_conditions=sl,
    ))

    # Regime-gated Dual
    results.append(backtest(
        df_slim, f"Regime_{method}_D",
        buy_cond=RegimeLongCond(
            ReversalLongCond(prox, decel, velocity_ceil=v_ceil)),
        sell_cond=NearResistanceCond(0.002),
        short_cond=RegimeShortCond(
            ReversalShortCond(prox, decel, velocity_floor=v_floor)),
        cover_cond=NearSupportCond(0.002),
        close_conditions=sl,
    ))

    # Regime-gated Long only
    results.append(backtest(
        df_slim, f"Regime_{method}_L",
        buy_cond=RegimeLongCond(
            ReversalLongCond(prox, decel, velocity_ceil=v_ceil)),
        sell_cond=NearResistanceCond(0.002),
        close_conditions=sl,
    ))

    # Regime-gated Short only
    results.append(backtest(
        df_slim, f"Regime_{method}_S",
        buy_cond=lambda row: False,
        sell_cond=lambda row: False,
        short_cond=RegimeShortCond(
            ReversalShortCond(prox, decel, velocity_floor=v_floor)),
        cover_cond=NearSupportCond(0.002),
        close_conditions=sl,
    ))

    return results, bh_ret, bh_1y


def run_regime_optimization_15m(df, method: str = "supertrend"):
    """
    Walk-forward optimisation of regime-gated S/R Baseline_15m.

    Grid: regime params × S/R params
      supertrend: period=[7,10,14], multiplier=[2.0,3.0,4.0]
      S/R       : proximity=[0.003,0.007], decel=[0.5,1.5]

    Train: 2022-01-01~2025-05-31  Val: 2025-06-01~present
    """
    TRAIN_START = "2022-01-01"
    TRAIN_END   = "2025-05-31"
    VAL_START   = "2025-06-01"

    b    = BASELINE_15M
    cgap = b["cluster_gap_pct"]
    vp_w = b.get("vp_windows")
    vb   = b["velocity_bars"]
    fvb  = b["fast_velocity_bars"]
    sl   = BASELINE_15M_SL

    # Regime param grids per method
    regime_grids = {
        "supertrend": [
            {"period": p, "multiplier": m}
            for p in [7, 10, 14]
            for m in [2.0, 3.0, 4.0]
        ],
        "ma200":     [{"window": w} for w in [100, 200]],
        "ma_cross":  [{"fast": f, "slow": s}
                      for f, s in [(20, 100), (50, 200)]],
        "momentum":  [{"lookback": lb} for lb in [30, 60, 90]],
    }
    rg = regime_grids.get(method, [{}])

    prox_list  = [0.003, 0.005, 0.007]
    decel_list = [0.5, 1.0, 1.5]

    print(f"  Building S/R levels...", flush=True)
    df_base = df[df["date"] >= TRAIN_START].reset_index(drop=True).copy()
    df_base.attrs = df.attrs.copy()
    df_sr = add_sr_levels(df_base, vp_windows=vp_w, cluster_gap_pct=cgap)
    df_sr.attrs = df_base.attrs.copy()

    df_train_sr = df_sr[df_sr["date"] <= TRAIN_END].reset_index(drop=True)
    df_train_sr.attrs = df_sr.attrs.copy()
    df_val_sr   = df_sr[df_sr["date"] >= VAL_START].reset_index(drop=True)
    df_val_sr.attrs = df_sr.attrs.copy()

    bh_train = (df_train_sr["close"].iloc[-1] / df_train_sr["close"].iloc[0] - 1) * 100
    bh_val   = (df_val_sr["close"].iloc[-1]   / df_val_sr["close"].iloc[0]   - 1) * 100

    df_m      = add_momentum_indicators(df_sr.copy(), velocity_bars=vb,
                                        fast_velocity_bars=fvb)
    df_slim   = _slim_df(df_m)
    df_slim_t = df_slim[df_slim["date"] <= TRAIN_END].reset_index(drop=True)
    df_slim_t.attrs = df_slim.attrs.copy()
    df_slim_v = df_slim[df_slim["date"] >= VAL_START].reset_index(drop=True)
    df_slim_v.attrs = df_slim.attrs.copy()

    total = len(rg) * len(prox_list) * len(decel_list)
    print(f"  Phase 1: {total} Dual combos  "
          f"train {TRAIN_START}~{TRAIN_END} / val {VAL_START}~present",
          flush=True)
    print(f"  BH train={bh_train:+.1f}%  val={bh_val:+.1f}%")

    train_results = []
    count = 0

    # Pre-compute daily regimes for each regime param set (on full df)
    daily_ohlcv = resample_to_daily(df_slim, date_col="date")
    regime_cache = {}
    for rp in rg:
        key = tuple(sorted(rp.items()))
        if key not in regime_cache:
            regime_cache[key] = compute_regime(daily_ohlcv, method=method, **rp)

    for rp in rg:
        key = tuple(sorted(rp.items()))
        regime_series = regime_cache[key]
        regime_t = map_regime_to_bars(regime_series, df_slim_t,
                                      date_col="date", shift_days=1)
        regime_v = map_regime_to_bars(regime_series, df_slim_v,
                                      date_col="date", shift_days=1)
        df_t = df_slim_t.copy()
        df_t["regime"] = regime_t
        df_v = df_slim_v.copy()
        df_v["regime"] = regime_v

        rp_str = ",".join(f"{k}={v}" for k, v in rp.items())

        for prox in prox_list:
            for decel in decel_list:
                tag = f"rg({rp_str})_px{prox*1000:.0f}_dc{decel}"
                res_t = backtest(
                    df_t, f"D_{tag}",
                    buy_cond=RegimeLongCond(
                        ReversalLongCond(prox, decel,
                                        velocity_ceil=b["velocity_ceil_long"])),
                    sell_cond=NearResistanceCond(0.002),
                    short_cond=RegimeShortCond(
                        ReversalShortCond(prox, decel,
                                         velocity_floor=b["velocity_floor_short"])),
                    cover_cond=NearSupportCond(0.002),
                    close_conditions=sl,
                )
                train_results.append((rp, prox, decel, res_t))
                count += 1
                if count % max(1, total // 4) == 0 or count == total:
                    print(f"  Phase 1: {count}/{total}", flush=True)

    # Sort by train Sharpe
    train_results.sort(key=lambda x: x[3]["sharpe"], reverse=True)
    top15 = train_results[:15]

    print(f"\n  Phase 2: validating top-15 Dual configs...", flush=True)
    val_results = []
    for rp, prox, decel, res_t in top15:
        key = tuple(sorted(rp.items()))
        regime_v = map_regime_to_bars(regime_cache[key], df_slim_v,
                                      date_col="date", shift_days=1)
        df_v = df_slim_v.copy()
        df_v["regime"] = regime_v

        rp_str = ",".join(f"{k}={v}" for k, v in rp.items())
        tag    = f"rg({rp_str})_px{prox*1000:.0f}_dc{decel}"
        res_v  = backtest(
            df_v, f"D_{tag}",
            buy_cond=RegimeLongCond(
                ReversalLongCond(prox, decel,
                                 velocity_ceil=b["velocity_ceil_long"])),
            sell_cond=NearResistanceCond(0.002),
            short_cond=RegimeShortCond(
                ReversalShortCond(prox, decel,
                                  velocity_floor=b["velocity_floor_short"])),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        )
        val_results.append((rp, prox, decel, res_t, res_v))

    # Print results table
    _hdr = (f"\n{'':=<120}\n"
            f"  REGIME-{method.upper()} Dual — Top-15 by train Sharpe (validated)"
            f"   BH train={bh_train:+.1f}%  val={bh_val:+.1f}%\n"
            f"{'':=<120}")
    print(_hdr)
    _col = (f"  {'regime_params':<30} {'px':>4} {'dc':>4}  "
            f"{'TrRet':>7} {'TrShr':>6} {'TrMDD':>7} {'TrN':>5}  "
            f"{'VaRet':>7} {'VaShr':>6} {'VaMDD':>7} {'VaN':>5}")
    print(_col)
    print(f"  {'-'*118}")

    val_results_sorted_val = sorted(val_results, key=lambda x: x[4]["sharpe"], reverse=True)
    for rp, prox, decel, res_t, res_v in val_results:
        star = "*" if res_v["sharpe"] > 0.5 else " "
        rp_str = ",".join(f"{k}={v}" for k, v in rp.items())
        print(
            f"  {star} {rp_str:<30} {prox*1000:>4.0f} {decel:>4.1f}  "
            f"{res_t['total_return']:>7.1f}% {res_t['sharpe']:>6.3f} "
            f"{res_t['max_drawdown']:>7.1f}% {res_t['trades']:>5}  "
            f"{res_v['total_return']:>7.1f}% {res_v['sharpe']:>6.3f} "
            f"{res_v['max_drawdown']:>7.1f}% {res_v['trades']:>5}"
        )

    print(f"\n{'':=<120}")
    print(f"  REGIME-{method.upper()} — Same configs sorted by VAL Sharpe"
          f"   BH train={bh_train:+.1f}%  val={bh_val:+.1f}%")
    print(f"{'':=<120}")
    print(_col)
    print(f"  {'-'*118}")
    for rp, prox, decel, res_t, res_v in val_results_sorted_val:
        star = "*" if res_v["sharpe"] > 0.5 else " "
        rp_str = ",".join(f"{k}={v}" for k, v in rp.items())
        print(
            f"  {star} {rp_str:<30} {prox*1000:>4.0f} {decel:>4.1f}  "
            f"{res_t['total_return']:>7.1f}% {res_t['sharpe']:>6.3f} "
            f"{res_t['max_drawdown']:>7.1f}% {res_t['trades']:>5}  "
            f"{res_v['total_return']:>7.1f}% {res_v['sharpe']:>6.3f} "
            f"{res_v['max_drawdown']:>7.1f}% {res_v['trades']:>5}"
        )


def run_vol_baseline(df):
    """Run volume-clock baseline on full history (2021-present)."""
    b = VOL_BASELINE
    df_slim = _prepare_vol_data(
        df,
        cluster_gap_pct=b["cluster_gap_pct"],
        velocity_vol_mult=b["velocity_vol_mult"],
        fast_vol_mult=b["fast_vol_mult"],
        vol_ref_period=b["vol_ref_period"],
        backtest_start=TRAIN_START_VOL,
    )
    bh_ret, bh_1y = _compute_bh(df_slim)

    prox   = b["proximity_pct"]
    decel  = b["min_decel"]
    v_ceil = b["velocity_ceil_long"]
    v_floor = b["velocity_floor_short"]

    results = []

    results.append(backtest(
        df_slim, "VolBaseline_L",
        buy_cond=VolumeReversalLongCond(prox, decel, velocity_ceil=v_ceil),
        sell_cond=NearResistanceCond(0.002),
        close_conditions=VOL_BASELINE_SL,
    ))

    results.append(backtest(
        df_slim, "VolBaseline_S",
        buy_cond=lambda row: False,
        sell_cond=lambda row: False,
        short_cond=VolumeReversalShortCond(prox, decel, velocity_floor=v_floor),
        cover_cond=NearSupportCond(0.002),
        close_conditions=VOL_BASELINE_SL,
    ))

    results.append(backtest(
        df_slim, "VolBaseline_D",
        buy_cond=VolumeReversalLongCond(prox, decel, velocity_ceil=v_ceil),
        sell_cond=NearResistanceCond(0.002),
        short_cond=VolumeReversalShortCond(prox, decel, velocity_floor=v_floor),
        cover_cond=NearSupportCond(0.002),
        close_conditions=VOL_BASELINE_SL,
    ))

    return results, bh_ret, bh_1y


def run_vol_optimization(df):
    """
    Volume-clock walk-forward optimisation.

    Train  : 2021-01-01 ~ 2025-05-31
    Validate: 2025-06-01 ~ latest

    Strategy: volume-clock S/R mean-reversion (dual direction).
    Volume clock adapts lookback to cumulative traded volume instead of
    fixed bar counts, so high-activity periods have compressed lookbacks.

    Phase 1 — Dual grid on train (~54 combos)
    Phase 2 — Validate top-15 D configs on val set
    Phase 3 — L/S breakdown for best-5 Dual configs

    Grid axes:
        velocity_vol_mult  [4, 8, 12, 16]   (≈ 4h / 8h / 12h / 16h avg vol)
        fast_vol_mult      [1, 2, 4]        (≈ 1h / 2h / 4h avg vol)
        proximity_pct      [0.003, 0.005, 0.010]
        min_decel          [0.5, 1.0, 1.5, 2.0]
        velocity_ceil_long [0.0, 0.5]
    Fixed: cluster_gap_pct=0.01, vol_ref_period=168, SL=-2%+Time48
    """
    cgap    = VOL_BASELINE["cluster_gap_pct"]
    vp_w    = VP_WINDOWS
    sl      = VOL_BASELINE_SL
    ref_per = VOL_BASELINE["vol_ref_period"]
    TOP_D   = 15

    vvm_list   = [4.0, 8.0, 12.0, 16.0]
    fvm_list   = [1.0, 2.0, 4.0]
    prox_list  = [0.003, 0.005, 0.010]
    decel_list = [0.5, 1.0, 1.5, 2.0]
    vceil_list = [0.0, 0.5]

    # ── 1. S/R levels on full data ────────────────────────────────────────────
    print("  Computing S/R levels...", flush=True)
    df_base = df[df["date"] >= TRAIN_START_VOL].reset_index(drop=True).copy()
    df_base.attrs = df.attrs.copy()
    df_sr = add_sr_levels(df_base, vp_windows=vp_w, cluster_gap_pct=cgap)
    df_sr.attrs = df_base.attrs.copy()

    df_train_base = df_sr[df_sr["date"] <= TRAIN_END_VOL].reset_index(drop=True)
    df_train_base.attrs = df_sr.attrs.copy()
    df_val_base   = df_sr[df_sr["date"] >= VAL_START_VOL].reset_index(drop=True)
    df_val_base.attrs = df_sr.attrs.copy()

    bh_train = (df_train_base["close"].iloc[-1]
                / df_train_base["close"].iloc[0] - 1) * 100
    bh_val   = (df_val_base["close"].iloc[-1]
                / df_val_base["close"].iloc[0] - 1) * 100

    # Valid (vvm, fvm) pairs: fast window must be strictly smaller
    valid_pairs = [(vvm, fvm) for vvm in vvm_list for fvm in fvm_list if fvm < vvm]
    phase1_total = (len(valid_pairs) * len(prox_list)
                    * len(decel_list) * len(vceil_list))
    print(f"  Phase 1: {phase1_total} Dual combos on train "
          f"{TRAIN_START_VOL}~{TRAIN_END_VOL} ({len(df_train_base)} bars)")
    print(f"  Val period: {VAL_START_VOL}~latest ({len(df_val_base)} bars)")
    print(f"  BH train={bh_train:+.1f}%  val={bh_val:+.1f}%", flush=True)

    # ── 2. Volume indicator cache (compute once per (vvm, fvm) pair) ──────────
    mom_cache = {}   # (vvm, fvm) -> (df_train_slim, df_val_slim)
    for vvm, fvm in valid_pairs:
        df_tm = _slim_df(add_volume_momentum_indicators(
            df_train_base.copy(), velocity_vol_mult=vvm,
            fast_vol_mult=fvm, vol_ref_period=ref_per))
        df_vm = _slim_df(add_volume_momentum_indicators(
            df_val_base.copy(), velocity_vol_mult=vvm,
            fast_vol_mult=fvm, vol_ref_period=ref_per))
        mom_cache[(vvm, fvm)] = (df_tm, df_vm)

    # ── 3. Phase 1: Dual-direction grid on training set ───────────────────────
    d_results = []   # (tag, vvm, fvm, prox, decel, vceil, result_train)
    count = 0
    for vvm, fvm in valid_pairs:
        df_tm, _ = mom_cache[(vvm, fvm)]
        for prox in prox_list:
            for decel in decel_list:
                for vceil in vceil_list:
                    vfloor = -vceil if vceil > 0 else 0.0
                    tag = (f"vvm{vvm:.0f}_fvm{fvm:.0f}"
                           f"_px{prox*1000:.0f}"
                           f"_dc{decel:.1f}"
                           f"_vc{vceil:.1f}")
                    r = backtest(
                        df_tm, f"D_{tag}",
                        buy_cond=VolumeReversalLongCond(prox, decel,
                                                        velocity_ceil=vceil),
                        sell_cond=NearResistanceCond(0.002),
                        short_cond=VolumeReversalShortCond(prox, decel,
                                                           velocity_floor=vfloor),
                        cover_cond=NearSupportCond(0.002),
                        close_conditions=sl,
                    )
                    d_results.append((tag, vvm, fvm, prox, decel, vceil, r))
                    count += 1
                    if count % 36 == 0 or count == phase1_total:
                        print(f"  Phase 1 progress: {count}/{phase1_total}",
                              flush=True)

    # Sort by train Sharpe
    d_results.sort(key=lambda x: x[6]["sharpe"], reverse=True)
    top_d = [x for x in d_results if x[6]["trades"] > 0][:TOP_D]

    # ── 4. Phase 2: Validate top Dual configs on val set ─────────────────────
    print(f"\n  Phase 2: validating top {len(top_d)} Dual configs on val set...",
          flush=True)

    def _run_dual_vol(df_slim, tag, prox, decel, vceil):
        vfloor = -vceil if vceil > 0 else 0.0
        return backtest(
            df_slim, f"D_{tag}",
            buy_cond=VolumeReversalLongCond(prox, decel, velocity_ceil=vceil),
            sell_cond=NearResistanceCond(0.002),
            short_cond=VolumeReversalShortCond(prox, decel, velocity_floor=vfloor),
            cover_cond=NearSupportCond(0.002),
            close_conditions=sl,
        )

    val_d = []
    for tag, vvm, fvm, prox, decel, vceil, tr in top_d:
        _, df_vm = mom_cache[(vvm, fvm)]
        vr = _run_dual_vol(df_vm, tag, prox, decel, vceil)
        val_d.append((tag, vvm, fvm, prox, decel, vceil, tr, vr))

    # ── 5. Phase 3: L/S breakdown for best-5 Dual configs ─────────────────────
    val_d_sorted = sorted(
        val_d,
        key=lambda x: x[7]["sharpe"] if x[7]["total_return"] > 0 else -99,
        reverse=True,
    )
    best5 = val_d_sorted[:5]

    print(f"\n  Phase 3: L/S breakdown for top-5 configs...", flush=True)
    ls_results = []
    for tag, vvm, fvm, prox, decel, vceil, _, _ in best5:
        vfloor = -vceil if vceil > 0 else 0.0
        df_tm, df_vm = mom_cache[(vvm, fvm)]
        for direction in ["L", "S"]:
            if direction == "L":
                tr = backtest(df_tm, f"L_{tag}",
                              buy_cond=VolumeReversalLongCond(prox, decel,
                                                              velocity_ceil=vceil),
                              sell_cond=NearResistanceCond(0.002),
                              close_conditions=sl)
                vr = backtest(df_vm, f"L_{tag}",
                              buy_cond=VolumeReversalLongCond(prox, decel,
                                                              velocity_ceil=vceil),
                              sell_cond=NearResistanceCond(0.002),
                              close_conditions=sl)
            else:
                tr = backtest(df_tm, f"S_{tag}",
                              buy_cond=lambda row: False,
                              sell_cond=lambda row: False,
                              short_cond=VolumeReversalShortCond(prox, decel,
                                                                 velocity_floor=vfloor),
                              cover_cond=NearSupportCond(0.002),
                              close_conditions=sl)
                vr = backtest(df_vm, f"S_{tag}",
                              buy_cond=lambda row: False,
                              sell_cond=lambda row: False,
                              short_cond=VolumeReversalShortCond(prox, decel,
                                                                 velocity_floor=vfloor),
                              cover_cond=NearSupportCond(0.002),
                              close_conditions=sl)
            ls_results.append((tag, vvm, fvm, prox, decel, vceil, direction, tr, vr))

    # ── 6. Print results ──────────────────────────────────────────────────────
    W = 120
    def _hdr(title):
        print(f"\n{'='*W}")
        print(f"  {title}   BH train={bh_train:+.1f}%  val={bh_val:+.1f}%")
        print(f"{'='*W}")
        print(f"  {'':2}{'vvm':>5} {'fvm':>4} {'px':>5} {'dc':>5} {'vc':>5}"
              f"  {'TrRet':>7} {'TrShr':>6} {'TrMDD':>7} {'TrN':>5}"
              f"  {'VaRet':>7} {'VaShr':>6} {'VaMDD':>7} {'VaN':>5}")
        print("  " + "-" * (W - 2))

    def _row(tag, vvm, fvm, prox, decel, vceil, tr, vr, prefix=""):
        ok = "*" if vr["sharpe"] > 0.3 and vr["total_return"] > 0 else " "
        print(f"  {ok}{prefix:1}{vvm:>5.0f} {fvm:>4.0f} {prox*1000:>5.0f}"
              f" {decel:>5.1f} {vceil:>5.1f}"
              f"  {tr['total_return']:>6.1f}% {tr['sharpe']:>6.3f}"
              f" {tr['max_drawdown']:>6.1f}% {tr['trades']:>5}"
              f"  {vr['total_return']:>6.1f}% {vr['sharpe']:>6.3f}"
              f" {vr['max_drawdown']:>6.1f}% {vr['trades']:>5}")

    _hdr("DUAL — Top-15 by train Sharpe (validated)")
    for tag, vvm, fvm, prox, decel, vceil, tr, vr in val_d:
        _row(tag, vvm, fvm, prox, decel, vceil, tr, vr)

    val_d_by_val = sorted(val_d,
                          key=lambda x: x[7]["sharpe"], reverse=True)
    _hdr("DUAL — Same configs sorted by VAL Sharpe")
    for tag, vvm, fvm, prox, decel, vceil, tr, vr in val_d_by_val:
        _row(tag, vvm, fvm, prox, decel, vceil, tr, vr)

    _hdr("LONG / SHORT breakdown for best-5 Dual configs")
    for tag, vvm, fvm, prox, decel, vceil, direction, tr, vr in ls_results:
        _row(tag, vvm, fvm, prox, decel, vceil, tr, vr, prefix=direction)

    return bh_train, bh_val


def main():
    grid_mode = "--grid" in sys.argv
    mode_15m = "--15m" in sys.argv
    optimize_mode = "--optimize" in sys.argv
    sl_mode = "--sl" in sys.argv
    sl_opt_mode = "--sl-opt" in sys.argv
    filter_mode = "--filter" in sys.argv
    velocity_mode = "--velocity" in sys.argv
    fast_mode = "--fast" in sys.argv
    vp_mode = "--vp" in sys.argv
    wbottom_mode = "--wbottom" in sys.argv
    sensitivity_mode = "--sensitivity" in sys.argv
    volbaseline_mode = "--volbaseline" in sys.argv
    regime_mode = "--regime" in sys.argv

    if sl_opt_mode:
        print("\n===== STOP-LOSS WALK-FORWARD OPTIMISATION (1h Baseline) =====")
        print(f"  Train: 2021-01-01 ~ 2025-05-31  /  Val: 2025-06-01 ~ present")
        df = load_data(
            start="2020-01-01",
            include_fg=False,
            include_derivatives=False,
            include_cb_premium=False,
            interval="1h",
        )
        run_sl_optimization(df)
        return

    if regime_mode:
        # Regime-filtered S/R Baseline (15m), daily Supertrend gating
        # Usage: python strategy_test_sr_momentum.py --regime [--optimize] [method=supertrend]
        method = "supertrend"
        for arg in sys.argv:
            if arg.startswith("method="):
                method = arg.split("=", 1)[1]

        interval = "15m" if mode_15m else "1h"
        print(f"\n===== TREND REGIME BASELINE ({interval}, method={method}) =====")
        df = load_data(
            start="2020-01-01",
            include_fg=False,
            include_derivatives=False,
            include_cb_premium=False,
            interval=interval,
        )

        # Default: 1h regime baseline
        # Use --15m to run 15m version instead
        if mode_15m:
            print("  (15m mode)")
            regime_results, bh_ret, bh_1y = run_regime_baseline_15m(df, method=method)
            bpd = 96
        else:
            print("  (1h mode)")
            # Reload with 1h interval
            df1h = load_data(
                start="2020-01-01",
                include_fg=False,
                include_derivatives=False,
                include_cb_premium=False,
                interval="1h",
            )
            regime_results, bh_ret, bh_1y = run_regime_baseline(df1h, method=method)
            bpd = 24

        print_results(regime_results, buy_and_hold_ret=bh_ret,
                      buy_and_hold_1y=bh_1y)
        for res in regime_results:
            print_yearly_breakdown(res, bars_per_day=bpd)

        if optimize_mode:
            print(f"\n===== REGIME WALK-FORWARD OPTIMISATION =====")
            if mode_15m:
                run_regime_optimization_15m(df, method=method)

        save_results_to_files(
            [r for r in regime_results if r["trades"] > 0],
            summary_path="results_regime_summary.txt",
            trade_log_path="trade_logs_regime.txt",
            buy_and_hold_ret=bh_ret,
            buy_and_hold_1y=bh_1y,
        )
        return

    if volbaseline_mode:
        # Volume-clock S/R strategy: 1h data, train 2021-2025.6, val 2025.6+
        print("===== VOLUME-CLOCK S/R STRATEGY =====")
        print(f"  Train: {TRAIN_START_VOL} ~ {TRAIN_END_VOL}")
        print(f"  Val  : {VAL_START_VOL} ~ present")
        df = load_data(
            start="2020-01-01",   # extra year for VP + vol_ref_period warmup
            include_fg=False,
            include_derivatives=False,
            include_cb_premium=False,
            interval="1h",
        )

        # ── Volume-clock baseline (full period 2021-present) ─────────────────
        print("\n----- Volume-clock Baseline (full history 2021-present) -----")
        vol_results, bh_ret, bh_1y = run_vol_baseline(df)
        print_results(vol_results, buy_and_hold_ret=bh_ret,
                      buy_and_hold_1y=bh_1y)
        for res in vol_results:
            print_yearly_breakdown(res, bars_per_day=24)

        # ── Walk-forward optimisation: train 2021-2025.6 / val 2025.6+ ───────
        print("\n===== VOLUME-CLOCK WALK-FORWARD OPTIMISATION =====")
        run_vol_optimization(df)

        # ── Save all results ──────────────────────────────────────────────────
        active = [r for r in vol_results if r["trades"] > 0]
        active.sort(key=lambda x: x["sharpe"], reverse=True)
        save_results_to_files(
            active,
            summary_path="results_vol_sr_summary.txt",
            trade_log_path="trade_logs_vol_sr.txt",
            buy_and_hold_ret=bh_ret,
            buy_and_hold_1y=bh_1y,
        )
        print(f"\nTotal strategies: {len(active)} active (vol-clock)")
        return

    if mode_15m:
        # 15m mode: load from 2020-01-01 to cover full 2021-2025 backtest
        # (VP windows need up to 90d warmup; 1y extra buffer is sufficient)
        df = load_data(
            start="2020-01-01",
            include_fg=False,
            include_derivatives=False,
            include_cb_premium=False,
            interval="15m",
        )

        baseline_results, bh_ret, bh_1y = run_baseline_15m(df)
        all_results = list(baseline_results)

        print("\n===== 15m BASELINE (vb32/8h, px0.5%, dc>1.5, SL-2%+Time192b) =====")
        print_results(baseline_results, buy_and_hold_ret=bh_ret,
                      buy_and_hold_1y=bh_1y)

        for res in baseline_results:
            print_yearly_breakdown(res, bars_per_day=96)

        if optimize_mode:
            print("\n===== 15m WALK-FORWARD OPTIMIZATION =====")
            run_optimization_15m(df)

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
        w_params = None
        if wbottom_mode:
            w_params = {"lookback_bars": 32, "swing_window": 3,
                        "second_low_tol": 0.005}
        df = load_data(
            start="2020-01-01",
            include_fg=False,
            include_derivatives=False,
            include_cb_premium=False,
            interval="1h",
            w_bottom_params=w_params,
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

        if fast_mode:
            print("\n===== FAST VELOCITY GRID SWEEP =====")
            fast_results, bh_ret, bh_1y = run_fast_grid(df)
            all_results.extend(fast_results)

        if vp_mode:
            print("\n===== VP WINDOW GRID SWEEP =====")
            vp_results, bh_ret, bh_1y = run_vp_grid(df)
            all_results.extend(vp_results)

        if wbottom_mode:
            print("\n===== W-BOTTOM/M-TOP MICROSTRUCTURE GRID =====")
            wb_results, bh_ret, bh_1y = run_wbottom_grid(df)
            all_results.extend(wb_results)

        if sl_mode:
            print("\n===== STOP-LOSS GRID SWEEP =====")
            sl_results, bh_ret, bh_1y = run_stoploss_grid(df)
            all_results.extend(sl_results)

        if grid_mode:
            print("\n===== GRID SWEEP =====")
            grid_results, bh_ret, bh_1y = run_grid(df)
            all_results.extend(grid_results)

        if sensitivity_mode:
            print("\n===== ONE-AT-A-TIME PARAMETER SENSITIVITY =====")
            bh_ret, bh_1y = run_sensitivity(df)

        active = [r for r in all_results if r["trades"] > 0]
        active.sort(key=lambda x: x["sharpe"], reverse=True)

        if grid_mode or sl_mode or filter_mode or velocity_mode or fast_mode or vp_mode or wbottom_mode:
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
