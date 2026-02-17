"""
RSI + POC Strategy Test (strategy_test_rsi_poc.py)
===================================================
Test classic RSI (buy<30, sell>70) alone and combined with VP POC.

Grid:
  1. Baseline: RSI<30 buy, RSI>70 sell
  2. Long only: RSI<30 AND price<POC buy (oversold + below fair value)
  3. Long only: RSI<30 AND price>POC buy (oversold + above fair value)
  4. Long/Short:
     - Long:  RSI<30 AND price<POC  /  exit RSI>70
     - Short: RSI>70 AND price>POC  /  exit RSI<30
  5. Sweep: window=[3,7,14], margin=[0%,1%,2%]

Usage:
    python strategy_test_rsi_poc.py
"""

from conditions import (
    VPAbovePOCCond, VPBelowPOCCond,
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
    BT = 30  # buy threshold  (oversold)
    ST = 70  # sell threshold (overbought)

    rsi_buy = RSIBuyCond(threshold=BT, period=P)    # RSI<30 = buy
    rsi_sell = RSISellCond(threshold=ST, period=P)   # RSI>70 = sell

    strategies = []

    # ================================================================
    # BASELINE: classic RSI
    # ================================================================
    strategies.append({
        "name": "RSI Classic",
        "buy_cond": rsi_buy,
        "sell_cond": rsi_sell,
    })

    # ================================================================
    # LONG ONLY: RSI + POC filter
    # ================================================================
    for window in [3, 7, 14]:
        for margin in [0.0, 0.01, 0.02]:
            m_label = f"m{margin*100:.0f}%" if margin > 0 else ""

            # RSI<30 AND price<POC (oversold + below fair value = deep value)
            strategies.append({
                "name": f"RSI+<POC{m_label} {window}d",
                "buy_cond": combine_conditions(
                    rsi_buy,
                    VPBelowPOCCond(margin=margin, window_days=window),
                    mode="AND"),
                "sell_cond": rsi_sell,
            })

            # RSI<30 AND price>POC (oversold but above fair value = bounce)
            strategies.append({
                "name": f"RSI+>POC{m_label} {window}d",
                "buy_cond": combine_conditions(
                    rsi_buy,
                    VPAbovePOCCond(margin=margin, window_days=window),
                    mode="AND"),
                "sell_cond": rsi_sell,
            })

    # ================================================================
    # LONG/SHORT: RSI + POC
    #   Long:  RSI<30 AND price<POC  -> exit RSI>70
    #   Short: RSI>70 AND price>POC  -> exit RSI<30
    # ================================================================
    for window in [3, 7, 14]:
        for margin in [0.0, 0.01, 0.02]:
            m_label = f"m{margin*100:.0f}%" if margin > 0 else ""

            long_buy = combine_conditions(
                RSIBuyCond(threshold=BT, period=P),
                VPBelowPOCCond(margin=margin, window_days=window),
                mode="AND")
            long_sell = RSISellCond(threshold=ST, period=P)

            short_entry = combine_conditions(
                RSISellCond(threshold=ST, period=P),
                VPAbovePOCCond(margin=margin, window_days=window),
                mode="AND")
            short_exit = RSIBuyCond(threshold=BT, period=P)

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
        summary_path="d:/work/alpha_strategy/results_rsi_poc_summary.txt",
        trade_log_path="d:/work/alpha_strategy/trade_logs_rsi_poc.txt",
        buy_and_hold_ret=bh_ret,
        buy_and_hold_1y=bh_1y,
    )


if __name__ == "__main__":
    main()
