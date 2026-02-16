"""
Hourly Strategy Test (strategy_test_hourly.py)
===============================================
Same strategies as strategy_test.py but on 1-hour data.

BTC price: Binance BTCUSDT 1h klines (full history with local cache).
Other sources (F&G, derivatives): daily data expanded to hourly
  - hour 0-22: previous day's value (no intraday look-ahead)
  - hour 23: current day's value

Indicator periods are auto-scaled by engine (e.g. RSI 14 -> 336 bars).

Usage:
    python strategy_test_hourly.py
"""

from conditions import (
    # Buy conditions
    FearGreedBuyCond, RSIBuyCond, FundingBuyCond, OIDropBuyCond,
    CBPremiumBuyCond, PriceDropBuyCond,
    # Filters
    TrendFilterCond, FundingNotOverheatedCond, RSIRisingCond,
    # Sell conditions
    FearGreedSellCond, RSISellCond, FundingSellCond, OISurgeSellCond,
    CBPremiumSellCond,
    # Close conditions
    StopLossCond, TakeProfitCond, TimeExitCond, TrailingStopCond,
    # Combinators
    combine_conditions,
    # Presets
    preset_fg_rsi_buy, preset_fg_rsi_sell,
    preset_funding_mr_buy, preset_funding_mr_sell,
    preset_union_buy, preset_alpha_combo,
)
from engine import load_data, backtest, print_results, print_trade_log, plot_results
import pandas as pd


def main():
    # ================================================================
    # LOAD HOURLY DATA
    # ================================================================
    df = load_data(
        start="2018-01-01",
        include_fg=True,
        include_derivatives=True,
        include_cb_premium=False,
        interval="1h",
    )

    # ================================================================
    # DEFINE STRATEGIES (same as daily, periods auto-scaled by engine)
    # ================================================================

    strategies = [
        # ------ BASELINES ------
        {
            "name": "F&G + RSI",
            "buy_cond": preset_fg_rsi_buy(fg_thresh=10, rsi_thresh=35),
            "sell_cond": preset_fg_rsi_sell(fg_thresh=92, rsi_thresh=65),
        },
        {
            "name": "Funding MR",
            "buy_cond": preset_funding_mr_buy(thresh=-0.0001),
            "sell_cond": preset_funding_mr_sell(thresh=0.0003),
        },

        # ------ REVERSE RSI ------
        {
            "name": "RevRSI Base",
            "buy_cond": RSISellCond(threshold=70),
            "sell_cond": RSIBuyCond(threshold=30),
        },
        {
            "name": "RevRSI+SL10%",
            "buy_cond": RSISellCond(threshold=70),
            "sell_cond": RSIBuyCond(threshold=30),
            "close_conditions": [StopLossCond(-0.10)],
        },
        {
            "name": "RevRSI+Trail20%",
            "buy_cond": RSISellCond(threshold=70),
            "sell_cond": RSIBuyCond(threshold=30),
            "close_conditions": [TrailingStopCond(-0.20)],
        },
        {
            "name": "RevRSI+Trail20+TP200",
            "buy_cond": RSISellCond(threshold=70),
            "sell_cond": RSIBuyCond(threshold=30),
            "close_conditions": [TrailingStopCond(-0.20), TakeProfitCond(2.0)],
        },

        # ------ F&G sell enhancement ------
        {
            "name": "RevRSI+FG90",
            "buy_cond": RSISellCond(threshold=70),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=30),
                FearGreedSellCond(threshold=90),
                mode="OR"),
        },

        # ------ Trend filter ------
        {
            "name": "RevRSI+SMA200",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=70),
                TrendFilterCond(ma_period=200),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=30),
        },

        # ------ Optimized combo ------
        {
            "name": "RevRSI Optimized",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=70),
                TrendFilterCond(ma_period=200),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=30),
            "close_conditions": [StopLossCond(-0.10)],
        },
    ]

    strategies.append(preset_alpha_combo())

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
            source_aware_sell=strat.get("source_aware_sell", False),
            sell_map=strat.get("sell_map"),
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

    # Print trade logs for top 3
    sorted_results = sorted(results, key=lambda x: x["total_return"], reverse=True)
    for res in sorted_results[:3]:
        print_trade_log(res)

    # ================================================================
    # PLOT
    # ================================================================
    plot_results(results, df, output_path="d:/work/AlphaGPT/strategy_test_hourly_results.png")


if __name__ == "__main__":
    main()
