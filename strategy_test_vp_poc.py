"""
VP POC + RevRSI Strategy Test (strategy_test_vp_poc.py)
========================================================
Test VP POC as entry timing filter for Reverse RSI strategy.

Grid:
  - POC direction: Above POC (trend) / Below POC (mean-reversion)
  - POC window:    3d, 7d, 14d
  - POC margin:    0%, 1%, 2%
  - Combo mode:    AND (RSI+POC both required) / OR (either triggers)

Usage:
    python strategy_test_vp_poc.py
"""

from conditions import (
    VPAbovePOCCond, VPBelowPOCCond,
    VPBelowVALCond,
    RSIBuyCond, RSISellCond,
    combine_conditions,
)
from engine import load_data, backtest, print_results, save_results_to_files
import pandas as pd


def main():
    df = load_data(
        start="2018-01-01",
        include_fg=False,
        include_derivatives=False,
        include_cb_premium=False,
        interval="1h",
    )

    P = 21   # RSI period (hours)
    BT = 75  # buy threshold
    ST = 30  # sell threshold

    rsi_buy = RSISellCond(threshold=BT, period=P)
    rsi_sell = RSIBuyCond(threshold=ST, period=P)

    strategies = []

    # ================================================================
    # BASELINE
    # ================================================================
    strategies.append({
        "name": "RevRSI Base",
        "buy_cond": rsi_buy,
        "sell_cond": rsi_sell,
    })

    # ================================================================
    # GRID: RSI AND/OR POC for entry, RSI sell
    # ================================================================
    for window in [3, 7, 14]:
        for margin in [0.0, 0.01, 0.02]:
            m_label = f"m{margin*100:.0f}%" if margin > 0 else ""

            # --- AND: RSI>75 AND price>POC entry ---
            strategies.append({
                "name": f"RSI+>POC{m_label} {window}d",
                "buy_cond": combine_conditions(
                    rsi_buy,
                    VPAbovePOCCond(margin=margin, window_days=window),
                    mode="AND"),
                "sell_cond": rsi_sell,
            })

            # --- AND: RSI>75 AND price<POC entry (MR timing) ---
            strategies.append({
                "name": f"RSI+<POC{m_label} {window}d",
                "buy_cond": combine_conditions(
                    rsi_buy,
                    VPBelowPOCCond(margin=margin, window_days=window),
                    mode="AND"),
                "sell_cond": rsi_sell,
            })

            # --- OR: RSI>75 OR price>POC entry ---
            strategies.append({
                "name": f"RSI|>POC{m_label} {window}d",
                "buy_cond": combine_conditions(
                    rsi_buy,
                    VPAbovePOCCond(margin=margin, window_days=window),
                    mode="OR"),
                "sell_cond": rsi_sell,
            })

    # ================================================================
    # BEST-OF: POC for both entry and exit
    # ================================================================
    for window in [3, 7, 14]:
        for margin in [0.0, 0.01]:
            m_label = f"m{margin*100:.0f}%" if margin > 0 else ""

            # AND entry + OR exit (RSI<30 OR price<POC)
            strategies.append({
                "name": f"RSI+>POC{m_label}|<POC {window}d",
                "buy_cond": combine_conditions(
                    rsi_buy,
                    VPAbovePOCCond(margin=margin, window_days=window),
                    mode="AND"),
                "sell_cond": combine_conditions(
                    rsi_sell,
                    VPBelowPOCCond(margin=margin, window_days=window),
                    mode="OR"),
            })

    # ================================================================
    # LONG/SHORT: RSI>75 AND price<POC = BUY, RSI<30 AND price>POC = SHORT
    # ================================================================
    for window in [3, 7, 14]:
        for margin in [0.0, 0.01, 0.02]:
            m_label = f"m{margin*100:.0f}%" if margin > 0 else ""

            # Long: RSI>75 AND price<POC (mean-reversion buy)
            long_buy = combine_conditions(
                rsi_buy,
                VPBelowPOCCond(margin=margin, window_days=window),
                mode="AND")
            long_sell = rsi_sell  # RSI<30 = exit long

            # Short: RSI<30 AND price>POC (weak momentum above fair value)
            short_entry = combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                VPAbovePOCCond(margin=margin, window_days=window),
                mode="AND")
            short_exit = RSISellCond(threshold=BT, period=P)  # RSI>75 = exit short

            strategies.append({
                "name": f"L/S POC{m_label} {window}d",
                "buy_cond": long_buy,
                "sell_cond": long_sell,
                "short_cond": short_entry,
                "cover_cond": short_exit,
            })

    # ================================================================
    # RUN
    # ================================================================
    results = []
    for strat in strategies:
        res = backtest(
            df,
            name=strat["name"],
            buy_cond=strat["buy_cond"],
            sell_cond=strat["sell_cond"],
            short_cond=strat.get("short_cond"),
            cover_cond=strat.get("cover_cond"),
        )
        results.append(res)

    # ================================================================
    # DISPLAY
    # ================================================================
    bh_ret = ((df["close"].iloc[-1] / df["close"].iloc[0]) - 1) * 100
    one_year_ago = df["date"].max() - pd.Timedelta(days=365)
    df_1y = df[df["date"] >= one_year_ago]
    bh_1y = ((df_1y["close"].iloc[-1] / df_1y["close"].iloc[0]) - 1) * 100 if len(df_1y) > 1 else 0
    print_results(results, buy_and_hold_ret=bh_ret, buy_and_hold_1y=bh_1y)

    save_results_to_files(
        results,
        summary_path="d:/work/alpha_strategy/results_vp_poc_summary.txt",
        trade_log_path="d:/work/alpha_strategy/trade_logs_vp_poc.txt",
        buy_and_hold_ret=bh_ret,
        buy_and_hold_1y=bh_1y,
    )


if __name__ == "__main__":
    main()
