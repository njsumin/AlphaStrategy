"""
Volume Profile Strategy Test (strategy_test_vp.py)
===================================================
Test Volume Profile indicators on hourly data.

Phase 1: VP standalone — best representatives per category
Phase 2: VP-Trend 7d + Reverse RSI(21) — best combo per type

Usage:
    python strategy_test_vp.py
"""

from conditions import (
    VPBelowVALCond, VPAboveVAHCond,
    RSIBuyCond, RSISellCond,
    combine_conditions,
)
from engine import load_data, backtest, print_results, save_results_to_files
import pandas as pd


def main():
    # ================================================================
    # LOAD HOURLY DATA
    # ================================================================
    df = load_data(
        start="2018-01-01",
        include_fg=False,
        include_derivatives=False,
        include_cb_premium=False,
        interval="1h",
    )

    # RevRSI hourly baseline params
    P = 21   # RSI period (hours)
    BT = 75  # buy threshold (RSI > BT)
    ST = 30  # sell threshold (RSI < ST)

    strategies = [
        # ============================================================
        # VP STANDALONE — best per category
        # ============================================================

        # MR best: 3d no margin (203%, Sharpe 0.525)
        {
            "name": "VP-MR 3d",
            "buy_cond": VPBelowVALCond(margin=0, window_days=3),
            "sell_cond": VPAboveVAHCond(margin=0, window_days=3),
        },
        # Trend best: 7d m2% (854%, Sharpe 0.831)
        {
            "name": "VP-Trend 7d m2%",
            "buy_cond": VPAboveVAHCond(margin=0.02, window_days=7),
            "sell_cond": VPBelowVALCond(margin=0.02, window_days=7),
        },

        # ============================================================
        # BASELINE — RevRSI(21, 75/30)
        # ============================================================
        {
            "name": "RevRSI Base",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
        },

        # ============================================================
        # VP + RevRSI COMBOS — best per type
        # ============================================================

        # AND sell: RSI buy, sell = RSI<30 AND price<VAL-1% (5687%, 1Y +4.4%)
        {
            "name": "RSI+VP-sell m1%",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                VPBelowVALCond(margin=0.01, window_days=7),
                mode="AND"),
        },
        # AND both: RSI>75 AND price>VAH-1% buy, RSI<30 AND price<VAL-1% sell (3502%, 1Y +0.7%)
        {
            "name": "RSI+VP-both m1%",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                VPAboveVAHCond(margin=0.01, window_days=7),
                mode="AND"),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                VPBelowVALCond(margin=0.01, window_days=7),
                mode="AND"),
        },
        # OR buy: RSI>75 OR price>VAH+1% buy, RSI sell (5217%, 1Y +3.7%)
        {
            "name": "RSI|VP-buy m1%",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                VPAboveVAHCond(margin=0.01, window_days=7),
                mode="OR"),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
        },
        # OR sell: RSI buy, sell = RSI<30 OR price<VAL-2% (2783%, 1Y +2.5%, MDD -14.8%)
        {
            "name": "RSI|VP-sell m2%",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                VPBelowVALCond(margin=0.02, window_days=7),
                mode="OR"),
        },
    ]

    # ================================================================
    # RUN ALL STRATEGIES
    # ================================================================
    results = []
    for strat in strategies:
        res = backtest(
            df,
            name=strat["name"],
            buy_cond=strat["buy_cond"],
            sell_cond=strat["sell_cond"],
            close_conditions=strat.get("close_conditions"),
        )
        results.append(res)

    # ================================================================
    # DISPLAY RESULTS
    # ================================================================
    bh_ret = ((df["close"].iloc[-1] / df["close"].iloc[0]) - 1) * 100
    one_year_ago = df["date"].max() - pd.Timedelta(days=365)
    df_1y = df[df["date"] >= one_year_ago]
    bh_1y = ((df_1y["close"].iloc[-1] / df_1y["close"].iloc[0]) - 1) * 100 if len(df_1y) > 1 else 0
    print_results(results, buy_and_hold_ret=bh_ret, buy_and_hold_1y=bh_1y)

    # Save results
    save_results_to_files(
        results,
        summary_path="d:/work/alpha_strategy/results_vp_summary.txt",
        trade_log_path="d:/work/alpha_strategy/trade_logs_vp.txt",
        buy_and_hold_ret=bh_ret,
        buy_and_hold_1y=bh_1y,
    )


if __name__ == "__main__":
    main()
