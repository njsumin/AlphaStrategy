"""
Hourly Strategy Test (strategy_test_hourly.py)
===============================================
Same strategies as strategy_test.py but on 1-hour data.

BTC price: Binance BTCUSDT 1h klines (full history with local cache).
Other sources (F&G, derivatives): daily data expanded to hourly
  - hour 0-22: previous day's value (no intraday look-ahead)
  - hour 23: current day's value

RSI keeps bar-level periods (e.g. RSI 14 = 14 hours).
Other indicators (SMA, drawdown, funding) are day-scaled (e.g. SMA200 = 4800 bars).

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
from engine import load_data, backtest, print_results, plot_results, save_results_to_files
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
    # DEFINE STRATEGIES
    # ================================================================
    # Phase 1 结论: RSI(21,75/30) 是小时级最优基线
    #   4772.6% return, 1.395 Sharpe, -28.7% MaxDD, 269 trades
    # Phase 2: 围绕此基线应用 Step 1-7 优化
    # ================================================================

    # RSI(21) 基线参数
    P = 21   # RSI period (hours)
    BT = 75  # buy threshold (RSI > BT)
    ST = 30  # sell threshold (RSI < ST)

    strategies = [
        # ------ 对照组 ------
        {
            "name": "Funding MR",
            "buy_cond": preset_funding_mr_buy(thresh=-0.0001),
            "sell_cond": preset_funding_mr_sell(thresh=0.0003),
        },

        # ------ 基线 ------
        {
            "name": "H-RevRSI Base",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
        },

        # ------ Step 1: 止损 & Trailing Stop ------
        {
            "name": "H-Rev+SL10%",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [StopLossCond(-0.10)],
        },
        {
            "name": "H-Rev+SL15%",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [StopLossCond(-0.15)],
        },
        {
            "name": "H-Rev+Trail10%",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [TrailingStopCond(-0.10)],
        },
        {
            "name": "H-Rev+Trail15%",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [TrailingStopCond(-0.15)],
        },
        {
            "name": "H-Rev+Trail20%",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [TrailingStopCond(-0.20)],
        },

        # ------ Step 2: 止盈 & 时间退出 ------
        {
            "name": "H-Rev+TP200%",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [TakeProfitCond(2.0)],
        },
        {
            "name": "H-Rev+TP100%",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [TakeProfitCond(1.0)],
        },
        {
            "name": "H-Rev+Time90d",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [TimeExitCond(max_days=90, min_profit=0.10)],
        },
        {
            "name": "H-Rev+Time180d",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [TimeExitCond(max_days=180, min_profit=0.10)],
        },

        # ------ Step 3: F&G 卖出增强 ------
        {
            "name": "H-Rev+FG85",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                FearGreedSellCond(threshold=85),
                mode="OR"),
        },
        {
            "name": "H-Rev+FG90",
            "buy_cond": RSISellCond(threshold=BT, period=P),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                FearGreedSellCond(threshold=90),
                mode="OR"),
        },

        # ------ Step 4: 趋势过滤 ------
        {
            "name": "H-Rev+SMA50",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                TrendFilterCond(ma_period=50),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
        },
        {
            "name": "H-Rev+SMA100",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                TrendFilterCond(ma_period=100),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
        },
        {
            "name": "H-Rev+SMA200",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                TrendFilterCond(ma_period=200),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
        },

        # ------ Step 5: 衍生品过滤 ------
        {
            "name": "H-Rev+FundFilt",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                FundingNotOverheatedCond(max_funding=0.0003),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
        },
        {
            "name": "H-Rev+FundSell",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                FundingNotOverheatedCond(max_funding=0.0003),
                mode="AND"),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                FundingSellCond(threshold=0.0003),
                mode="OR"),
        },

        # ------ Step 6: RSI 动量方向 ------
        {
            "name": "H-Rev+Rising",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                RSIRisingCond(min_delta=0, lookback=3),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
        },

        # ------ Step 7: 组合优化 ------
        {
            "name": "H-Rev+SMA100+SL10",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                TrendFilterCond(ma_period=100),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [StopLossCond(-0.10)],
        },
        {
            "name": "H-Rev+SMA100+Trail15",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                TrendFilterCond(ma_period=100),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
            "close_conditions": [TrailingStopCond(-0.15)],
        },
        {
            "name": "H-Rev+SMA100+FG90",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                TrendFilterCond(ma_period=100),
                mode="AND"),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                FearGreedSellCond(threshold=90),
                mode="OR"),
        },
        {
            "name": "H-Rev+Fund+SMA100",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                TrendFilterCond(ma_period=100),
                FundingNotOverheatedCond(max_funding=0.0003),
                mode="AND"),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                FundingSellCond(threshold=0.0003),
                mode="OR"),
        },
        {
            "name": "H-Rev+Rising+SMA100",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                TrendFilterCond(ma_period=100),
                RSIRisingCond(min_delta=0, lookback=3),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=ST, period=P),
        },
        {
            "name": "H-Rev Full Combo",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=BT, period=P),
                TrendFilterCond(ma_period=100),
                FundingNotOverheatedCond(max_funding=0.0003),
                mode="AND"),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=ST, period=P),
                FundingSellCond(threshold=0.0003),
                mode="OR"),
            "close_conditions": [StopLossCond(-0.10)],
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

    # Save detailed results and trade logs to files
    save_results_to_files(
        results,
        summary_path="d:/work/alpha_strategy/results_hourly_summary.txt",
        trade_log_path="d:/work/alpha_strategy/trade_logs_hourly.txt",
        buy_and_hold_ret=bh_ret,
        buy_and_hold_1y=bh_1y,
    )

    # ================================================================
    # PLOT
    # ================================================================
    plot_results(results, df, output_path="d:/work/AlphaGPT/strategy_test_hourly_results.png")


if __name__ == "__main__":
    main()
