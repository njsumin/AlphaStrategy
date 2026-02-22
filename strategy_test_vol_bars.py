"""
Volume Bar S/R Strategy (strategy_test_vol_bars.py)
====================================================
Generates constant-volume bars from 15-minute data for signal computation,
then maps those signals back to the 15-minute bars for execution.

Architecture:
  15m bars (execution, ~96/day)
      ← entry signals ←  Volume bars (signal, e.g. 12/day)
                          S/R levels + momentum reversal detection

Entry signals fire only at 15m bars where a volume bar closes.
S/R levels propagate forward between updates; distances recomputed live.
Execution: T-bar signal → T+1 bar 15m open price (no look-ahead).
Exit: StopLoss + TimeBarStop(192 15m-bars = 48 h) checked every 15m bar.

Walk-forward structure:
  Train  : 2021-01-01 ~ 2025-05-31  (parameter search)
  Validate: 2025-06-01 ~ present    (out-of-sample)

Grid axes (Phase 1, Dual direction):
  target_bars_per_day [6, 12, 24]      → volume_per_bar from train median
  velocity_bars       [4, 8, 12]       (volume bars)
  proximity_pct       [0.003, 0.005, 0.010]
  min_decel           [0.5, 1.0, 1.5]
  velocity_ceil_long  [0.0, 0.5]
Fixed: vp_windows=[30, 60, 120] vbars, cluster_gap=0.01,
       SL=-2%, TimeBarStop=192 (48 h of 15m bars)

Usage:
  python strategy_test_vol_bars.py            # baseline + full optimisation
  python strategy_test_vol_bars.py --baseline # baseline only (fast)
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

from conditions import StopLossCond, TimeBarStopCond
from engine import backtest, print_results, save_results_to_files
from data_loader import load_btc_price_15m
from sr_signals import (
    add_momentum_indicators,
    ReversalLongCond, ReversalShortCond,
    NearResistanceCond, NearSupportCond,
)
from volume_bars import (
    make_volume_bars,
    compute_volume_per_bar,
    add_vbar_sr_levels,
    map_vbar_signals_to_15m,
)


# ============================================================================
# CONSTANTS
# ============================================================================

TRAIN_START = "2021-01-01"
TRAIN_END   = "2025-05-31"
VAL_START   = "2025-06-01"

VP_WINDOWS_VB   = [30, 60, 120]   # VP lookback in volume bars
CLUSTER_GAP_PCT = 0.01

# Fixed stop-loss: -2% price stop + 192 × 15m-bars time stop (≈ 48 h)
STOP_LOSS_PCT  = -0.02
TIME_STOP_15M  = 192   # 15m bars = 48 h
BARS_PER_DAY   = 96    # 15m execution bars per calendar day

# Columns needed by backtest + conditions
_KEEP_COLS = [
    "date", "open", "close",
    "nearest_resistance", "nearest_support",
    "dist_to_resistance", "dist_to_support",
    "norm_velocity", "price_velocity", "velocity_delta",
    "fast_norm_velocity", "fast_price_velocity", "fast_velocity_delta",
    "sr_count", "atr",
]


def _slim(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _KEEP_COLS if c in df.columns]
    out = df[cols].copy()
    out.attrs = df.attrs.copy()
    return out


# ============================================================================
# DATA PREPARATION
# ============================================================================

def build_vbar_dataset(df_15m: pd.DataFrame,
                        target_bars_per_day: int,
                        velocity_bars: int,
                        fast_velocity_bars: int = 2,
                        vp_windows: list = None,
                        cluster_gap_pct: float = CLUSTER_GAP_PCT) -> dict:
    """
    Build 15m execution DataFrame enriched with volume-bar signals.

    Steps:
      1. Compute volume_per_bar from training-set 15m data (no lookahead).
      2. Convert ALL 15m data to volume bars.
      3. Add VP + S/R levels on volume bars.
      4. Add momentum indicators on volume bars.
      5. Map volume-bar signals back to 15m bars:
           - Entry signals at vbar close bars only.
           - S/R levels forward-filled; distances recomputed from live close.
      6. Set bars_per_day=96 (15m execution) in DataFrame attrs.

    Returns dict:
      df_train   : slim 15m+signals DataFrame, date <= TRAIN_END
      df_val     : slim 15m+signals DataFrame, date >= VAL_START
      vpb        : volume_per_bar used
      n_vbars    : total volume bars generated
      bh_train   : buy-and-hold return in training period (%)
      bh_val     : buy-and-hold return in validation period (%)
    """
    if vp_windows is None:
        vp_windows = VP_WINDOWS_VB

    # 1. Volume per bar — training data only
    vpb = compute_volume_per_bar(df_15m, target_bars_per_day,
                                  train_start=TRAIN_START, train_end=TRAIN_END)

    # 2. Convert full 15m dataset to volume bars
    df_vb = make_volume_bars(df_15m, vpb)

    # 3. VP + S/R on volume bars (rolling, no lookahead)
    df_vb = add_vbar_sr_levels(df_vb, vp_windows=vp_windows,
                                cluster_gap_pct=cluster_gap_pct)

    # 4. Momentum indicators on volume bars
    df_vb = add_momentum_indicators(df_vb,
                                    velocity_bars=velocity_bars,
                                    fast_velocity_bars=fast_velocity_bars)

    # 5. Map signals to 15m bars
    df_mapped = map_vbar_signals_to_15m(df_15m, df_vb)
    df_mapped.attrs["bars_per_day"] = BARS_PER_DAY   # 15m execution

    # 6. Split and slim
    df_slim = _slim(df_mapped)
    df_slim.attrs["bars_per_day"] = BARS_PER_DAY

    mask_tr = df_slim["date"] <= TRAIN_END
    mask_va = df_slim["date"] >= VAL_START
    df_train = df_slim[mask_tr].reset_index(drop=True)
    df_val   = df_slim[mask_va].reset_index(drop=True)
    df_train.attrs = {"bars_per_day": BARS_PER_DAY}
    df_val.attrs   = {"bars_per_day": BARS_PER_DAY}

    bh_train = bh_val = None
    if len(df_train) > 1:
        bh_train = (df_train["close"].iloc[-1]
                    / df_train["close"].iloc[0] - 1) * 100
    if len(df_val) > 1:
        bh_val = (df_val["close"].iloc[-1]
                  / df_val["close"].iloc[0] - 1) * 100

    return {
        "df_train":  df_train,
        "df_val":    df_val,
        "vpb":       vpb,
        "n_vbars":   len(df_vb),
        "bh_train":  bh_train,
        "bh_val":    bh_val,
    }


# ============================================================================
# YEARLY BREAKDOWN
# ============================================================================

def print_yearly_breakdown(result: dict):
    pv = result["portfolio"].copy()
    pv["year"] = pd.to_datetime(pv["date"]).dt.year
    annualize  = np.sqrt(365 * BARS_PER_DAY)
    trades     = result.get("trade_log", [])

    print(f"\n  {result['name']} -- Yearly Breakdown:")
    print(f"  {'Year':<6} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print(f"  {'-'*42}")
    for year, grp in pv.groupby("year"):
        if len(grp) < 2:
            continue
        yr_ret = (grp["value"].iloc[-1] / grp["value"].iloc[0] - 1) * 100
        dr = grp["value"].pct_change().dropna()
        yr_sharpe = (dr.mean() / dr.std()) * annualize if dr.std() > 0 else 0
        yr_dd = (
            (grp["value"] - grp["value"].cummax()) / grp["value"].cummax()
        ).min() * 100
        yr_trades = len([t for t in trades
                         if pd.Timestamp(t["date"]).year == year
                         and t["type"] in ("BUY", "SHORT")])
        print(f"  {year:<6} {yr_ret:>8.1f}% {yr_sharpe:>8.3f}"
              f" {yr_dd:>7.1f}% {yr_trades:>7}")
    print()


# ============================================================================
# BASELINE
# ============================================================================

def run_baseline(df_15m: pd.DataFrame):
    """
    Volume-bar baseline mirroring the proven 1h S/R Dual parameters:
      target_bars_per_day=12, velocity_bars=8, proximity=0.5%,
      min_decel=1.5, velocity_ceil=0.5, SL=-2%+Time192(48h).

    Runs on full period 2021-present.
    """
    TARGET_BPD = 12
    VB_BARS    = 8
    PROX       = 0.005
    DECEL      = 1.5
    V_CEIL     = 0.5
    V_FLOOR    = -0.5
    sl = [StopLossCond(STOP_LOSS_PCT),
          TimeBarStopCond(TIME_STOP_15M, bars_per_day=BARS_PER_DAY)]

    print(f"  Building dataset: {TARGET_BPD} vbars/day, "
          f"velocity={VB_BARS} vbars ...", flush=True)
    data = build_vbar_dataset(df_15m, TARGET_BPD, VB_BARS)
    print(f"  {data['n_vbars']} volume bars  (vpb={data['vpb']:,.0f} BTC)  "
          f"Train: {len(data['df_train'])} 15m bars  "
          f"Val: {len(data['df_val'])} 15m bars", flush=True)

    df_full = pd.concat([data["df_train"], data["df_val"]], ignore_index=True)
    df_full.attrs = {"bars_per_day": BARS_PER_DAY}

    bh_ret = bh_1y = None
    if len(df_full) > 1:
        bh_ret = (df_full["close"].iloc[-1]
                  / df_full["close"].iloc[0] - 1) * 100
        one_year_ago = df_full["date"].max() - pd.Timedelta(days=365)
        df_1y = df_full[df_full["date"] >= one_year_ago]
        if len(df_1y) > 1:
            bh_1y = (df_1y["close"].iloc[-1]
                     / df_1y["close"].iloc[0] - 1) * 100

    results = []
    results.append(backtest(
        df_full, "VB_Baseline_L",
        buy_cond=ReversalLongCond(PROX, DECEL, velocity_ceil=V_CEIL),
        sell_cond=NearResistanceCond(0.002),
        close_conditions=sl,
    ))
    results.append(backtest(
        df_full, "VB_Baseline_S",
        buy_cond=lambda row: False,
        sell_cond=lambda row: False,
        short_cond=ReversalShortCond(PROX, DECEL, velocity_floor=V_FLOOR),
        cover_cond=NearSupportCond(0.002),
        close_conditions=sl,
    ))
    results.append(backtest(
        df_full, "VB_Baseline_D",
        buy_cond=ReversalLongCond(PROX, DECEL, velocity_ceil=V_CEIL),
        sell_cond=NearResistanceCond(0.002),
        short_cond=ReversalShortCond(PROX, DECEL, velocity_floor=V_FLOOR),
        cover_cond=NearSupportCond(0.002),
        close_conditions=sl,
    ))
    return results, bh_ret, bh_1y


# ============================================================================
# WALK-FORWARD OPTIMISATION
# ============================================================================

def run_optimization(df_15m: pd.DataFrame):
    """
    Walk-forward grid search: signal from volume bars, execution on 15m bars.

    Phase 1 — Dual grid on train set (2021-2025.5)
    Phase 2 — Validate top-15 Dual configs on val (2025.6+)
    Phase 3 — L/S breakdown for top-5 Dual configs

    Grid axes:
      target_bars_per_day  [6, 12, 24]
      velocity_bars        [4, 8, 12]  (volume bars)
      proximity_pct        [0.003, 0.005, 0.010]
      min_decel            [0.5, 1.0, 1.5]
      velocity_ceil_long   [0.0, 0.5]
    Fixed:
      vp_windows=[30,60,120] vbars, cluster_gap=0.01
      SL=-2%, TimeBarStop=192 (48 h)
    """
    bpd_list   = [6, 12, 24]
    vb_list    = [4, 8, 12]
    prox_list  = [0.003, 0.005, 0.010]
    decel_list = [0.5, 1.0, 1.5]
    vceil_list = [0.0, 0.5]
    sl = [StopLossCond(STOP_LOSS_PCT),
          TimeBarStopCond(TIME_STOP_15M, bars_per_day=BARS_PER_DAY)]
    TOP_D = 15

    combos_per_bpd = len(vb_list) * len(prox_list) * len(decel_list) * len(vceil_list)
    total_train    = len(bpd_list) * combos_per_bpd
    print(f"  Phase 1: {total_train} Dual combos  "
          f"train {TRAIN_START}~{TRAIN_END} / val {VAL_START}~present",
          flush=True)

    all_d_results = []
    bh_train_last = bh_val_last = None
    count = 0

    for bpd_val in bpd_list:
        t0 = time.time()
        print(f"\n  [target={bpd_val} vbars/day] "
              f"Building S/R volume bar dataset...", flush=True)

        # Compute vpb + S/R once per bpd (expensive VP loops)
        vpb = compute_volume_per_bar(df_15m, bpd_val,
                                      train_start=TRAIN_START,
                                      train_end=TRAIN_END)
        df_vb_raw = make_volume_bars(df_15m, vpb)
        df_vb_sr  = add_vbar_sr_levels(df_vb_raw,
                                        vp_windows=VP_WINDOWS_VB,
                                        cluster_gap_pct=CLUSTER_GAP_PCT)

        # Cache momentum + mapped 15m slices per velocity_bars value
        mom_cache = {}   # vb -> (df_train_slim, df_val_slim)
        for vb in vb_list:
            fvb = max(2, vb // 4)
            df_vb_m  = add_momentum_indicators(df_vb_sr.copy(),
                                                velocity_bars=vb,
                                                fast_velocity_bars=fvb)
            df_mapped = map_vbar_signals_to_15m(df_15m, df_vb_m)
            df_mapped.attrs["bars_per_day"] = BARS_PER_DAY
            df_slim = _slim(df_mapped)
            df_slim.attrs["bars_per_day"] = BARS_PER_DAY

            df_tr = df_slim[df_slim["date"] <= TRAIN_END].reset_index(drop=True)
            df_va = df_slim[df_slim["date"] >= VAL_START].reset_index(drop=True)
            df_tr.attrs = {"bars_per_day": BARS_PER_DAY}
            df_va.attrs = {"bars_per_day": BARS_PER_DAY}
            mom_cache[vb] = (df_tr, df_va)

        # BH stats from first vb entry
        df_tr0, df_va0 = mom_cache[vb_list[0]]
        bh_train_last = (df_tr0["close"].iloc[-1] / df_tr0["close"].iloc[0] - 1) * 100 if len(df_tr0) > 1 else 0
        bh_val_last   = (df_va0["close"].iloc[-1] / df_va0["close"].iloc[0] - 1) * 100 if len(df_va0) > 1 else 0

        print(f"    vpb={vpb:,.0f} BTC  "
              f"vbars={len(df_vb_raw)}  "
              f"train={len(df_tr0)} 15m-bars  val={len(df_va0)} 15m-bars  "
              f"BH train={bh_train_last:+.1f}%  val={bh_val_last:+.1f}%  "
              f"({time.time()-t0:.1f}s)", flush=True)

        # Grid search
        for vb in vb_list:
            df_tr, _ = mom_cache[vb]
            for prox in prox_list:
                for decel in decel_list:
                    for vceil in vceil_list:
                        vfloor = -vceil if vceil > 0 else 0.0
                        tag = (f"bpd{bpd_val}_vb{vb}"
                               f"_px{prox*1000:.0f}"
                               f"_dc{decel:.1f}"
                               f"_vc{vceil:.1f}")
                        r = backtest(
                            df_tr, f"D_{tag}",
                            buy_cond=ReversalLongCond(prox, decel,
                                                      velocity_ceil=vceil),
                            sell_cond=NearResistanceCond(0.002),
                            short_cond=ReversalShortCond(prox, decel,
                                                         velocity_floor=vfloor),
                            cover_cond=NearSupportCond(0.002),
                            close_conditions=sl,
                        )
                        all_d_results.append(
                            (tag, bpd_val, vb, prox, decel, vceil, r,
                             mom_cache))
                        count += 1
                        if count % 27 == 0 or count == total_train:
                            print(f"  Phase 1: {count}/{total_train}", flush=True)

    # Sort by train Sharpe
    all_d_results.sort(key=lambda x: x[6]["sharpe"], reverse=True)
    top_d = [x for x in all_d_results if x[6]["trades"] > 0][:TOP_D]

    # ── Phase 2: Validate top Dual configs ────────────────────────────────────
    print(f"\n  Phase 2: validating top {len(top_d)} Dual configs...", flush=True)

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
    for tag, bpd_val, vb, prox, decel, vceil, tr, cache in top_d:
        _, df_va = cache[vb]
        vr = _run_dual(df_va, tag, prox, decel, vceil)
        val_d.append((tag, bpd_val, vb, prox, decel, vceil, tr, vr, cache))

    # ── Phase 3: L/S breakdown for best-5 ────────────────────────────────────
    val_d_sorted = sorted(
        val_d,
        key=lambda x: x[7]["sharpe"] if x[7]["total_return"] > 0 else -99,
        reverse=True,
    )
    best5 = val_d_sorted[:5]
    print(f"\n  Phase 3: L/S breakdown for top-5...", flush=True)

    ls_results = []
    for tag, bpd_val, vb, prox, decel, vceil, _, _, cache in best5:
        vfloor = -vceil if vceil > 0 else 0.0
        df_tr, df_va = cache[vb]
        for direction in ["L", "S"]:
            if direction == "L":
                tr = backtest(df_tr, f"L_{tag}",
                              buy_cond=ReversalLongCond(prox, decel,
                                                        velocity_ceil=vceil),
                              sell_cond=NearResistanceCond(0.002),
                              close_conditions=sl)
                vr = backtest(df_va, f"L_{tag}",
                              buy_cond=ReversalLongCond(prox, decel,
                                                        velocity_ceil=vceil),
                              sell_cond=NearResistanceCond(0.002),
                              close_conditions=sl)
            else:
                tr = backtest(df_tr, f"S_{tag}",
                              buy_cond=lambda row: False,
                              sell_cond=lambda row: False,
                              short_cond=ReversalShortCond(prox, decel,
                                                           velocity_floor=vfloor),
                              cover_cond=NearSupportCond(0.002),
                              close_conditions=sl)
                vr = backtest(df_va, f"S_{tag}",
                              buy_cond=lambda row: False,
                              sell_cond=lambda row: False,
                              short_cond=ReversalShortCond(prox, decel,
                                                           velocity_floor=vfloor),
                              cover_cond=NearSupportCond(0.002),
                              close_conditions=sl)
            ls_results.append(
                (tag, bpd_val, vb, prox, decel, vceil, direction, tr, vr))

    # ── Print tables ──────────────────────────────────────────────────────────
    W = 125

    def _hdr(title):
        bht = bh_train_last or 0.0
        bhv = bh_val_last   or 0.0
        print(f"\n{'='*W}")
        print(f"  {title}   "
              f"BH train={bht:+.1f}%  val={bhv:+.1f}%")
        print(f"{'='*W}")
        print(f"  {'':2}{'bpd':>4} {'vb':>4} {'px':>5} {'dc':>5} {'vc':>5}"
              f"  {'TrRet':>7} {'TrShr':>6} {'TrMDD':>7} {'TrN':>5}"
              f"  {'VaRet':>7} {'VaShr':>6} {'VaMDD':>7} {'VaN':>5}")
        print("  " + "-" * (W - 2))

    def _row(tag, bpd_val, vb, prox, decel, vceil, tr, vr, prefix=""):
        ok = "*" if vr["sharpe"] > 0.3 and vr["total_return"] > 0 else " "
        print(f"  {ok}{prefix:1}{bpd_val:>4} {vb:>4} {prox*1000:>5.0f}"
              f" {decel:>5.1f} {vceil:>5.1f}"
              f"  {tr['total_return']:>6.1f}% {tr['sharpe']:>6.3f}"
              f" {tr['max_drawdown']:>6.1f}% {tr['trades']:>5}"
              f"  {vr['total_return']:>6.1f}% {vr['sharpe']:>6.3f}"
              f" {vr['max_drawdown']:>6.1f}% {vr['trades']:>5}")

    _hdr("DUAL — Top-15 by train Sharpe (validated)")
    for tag, bpd_val, vb, prox, decel, vceil, tr, vr, _ in val_d:
        _row(tag, bpd_val, vb, prox, decel, vceil, tr, vr)

    val_d_by_val = sorted(val_d, key=lambda x: x[7]["sharpe"], reverse=True)
    _hdr("DUAL — Same configs sorted by VAL Sharpe")
    for tag, bpd_val, vb, prox, decel, vceil, tr, vr, _ in val_d_by_val:
        _row(tag, bpd_val, vb, prox, decel, vceil, tr, vr)

    _hdr("LONG / SHORT breakdown for best-5 Dual configs")
    for tag, bpd_val, vb, prox, decel, vceil, direction, tr, vr in ls_results:
        _row(tag, bpd_val, vb, prox, decel, vceil, tr, vr, prefix=direction)

    return val_d, ls_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    baseline_only = "--baseline" in sys.argv

    print("=" * 70)
    print("  Volume Bar S/R Strategy  (15m execution, volume-bar signals)")
    print(f"  Train : {TRAIN_START} ~ {TRAIN_END}")
    print(f"  Val   : {VAL_START} ~ present")
    print(f"  VP    : {VP_WINDOWS_VB} volume bars")
    print(f"  Stop  : {STOP_LOSS_PCT*100:.0f}% SL + {TIME_STOP_15M} 15m-bars (48 h)")
    print("=" * 70, flush=True)

    print("\nLoading 15-minute OHLCV data...", flush=True)
    t0 = time.time()
    df_15m = load_btc_price_15m(start="2020-01-01")
    df_15m = df_15m[df_15m["date"] >= "2020-01-01"].reset_index(drop=True)
    print(f"  {len(df_15m)} 15m bars  "
          f"({df_15m['date'].min().date()} ~ {df_15m['date'].max().date()})  "
          f"{time.time()-t0:.1f}s", flush=True)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n===== VOLUME BAR BASELINE (target=12 vbars/day) =====", flush=True)
    t0 = time.time()
    baseline_results, bh_ret, bh_1y = run_baseline(df_15m)
    print(f"  Baseline done in {time.time()-t0:.1f}s", flush=True)

    print_results(baseline_results, buy_and_hold_ret=bh_ret,
                  buy_and_hold_1y=bh_1y)
    for r in baseline_results:
        print_yearly_breakdown(r)

    if baseline_only:
        save_results_to_files(
            [r for r in baseline_results if r["trades"] > 0],
            summary_path="results_vol_bars_summary.txt",
            trade_log_path="trade_logs_vol_bars.txt",
            buy_and_hold_ret=bh_ret,
            buy_and_hold_1y=bh_1y,
        )
        return

    # ── Walk-forward optimisation ─────────────────────────────────────────────
    print("\n===== WALK-FORWARD OPTIMISATION =====", flush=True)
    t0 = time.time()
    run_optimization(df_15m)
    print(f"\nOptimisation done in {time.time()-t0:.0f}s", flush=True)

    # Save baseline results
    save_results_to_files(
        [r for r in baseline_results if r["trades"] > 0],
        summary_path="results_vol_bars_summary.txt",
        trade_log_path="trade_logs_vol_bars.txt",
        buy_and_hold_ret=bh_ret,
        buy_and_hold_1y=bh_1y,
    )
    print("Results saved to results_vol_bars_summary.txt")


if __name__ == "__main__":
    main()
