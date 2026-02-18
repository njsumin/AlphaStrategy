"""
Strategy Test Harness (strategy_test.py)
=========================================
Main entry point to test different strategy combinations.
Mix-and-match BuyCond, SellCond, and CloseCond to find optimal strategies.

Usage Examples:
    # Run all preset strategies:
    python strategy_test.py

    # To create custom strategies, import and use:
    from conditions import *
    from engine import load_data, backtest, print_results, plot_results

    df = load_data()

    # Custom: Buy on extreme fear + oversold, sell on extreme greed + overbought
    result = backtest(df, "Custom",
        buy_cond=combine_conditions(FearGreedBuyCond(15), RSIBuyCond(40), mode="AND"),
        sell_cond=combine_conditions(FearGreedSellCond(85), RSISellCond(60), mode="AND"))

    # With stop-loss and trailing stop:
    from conditions import StopLossCond, TrailingStopCond
    result = backtest(df, "With Risk Mgmt",
        buy_cond=FearGreedBuyCond(10),
        sell_cond=FearGreedSellCond(92),
        close_conditions=[StopLossCond(-0.20), TrailingStopCond(-0.25)])

==========================================================================
FUNCTION REFERENCE
==========================================================================

BUY CONDITIONS (conditions.py):
  FearGreedBuyCond(threshold=10)     Buy when F&G Index < threshold (extreme fear)
  RSIBuyCond(threshold=35, period=14) Buy when RSI < threshold (oversold)
  FundingBuyCond(threshold=-0.0001)  Buy when Funding Rate SMA7 < threshold (shorts dominant)
  OIDropBuyCond(threshold=-0.15)     Buy when OI 7d change < threshold (liquidation flush)
  CBPremiumBuyCond(threshold=-0.5)   Buy when Coinbase Premium SMA7 < threshold (US selling)
  PriceDropBuyCond(threshold=-0.20)  Buy when price drawdown from recent high > threshold

SELL CONDITIONS (conditions.py):
  FearGreedSellCond(threshold=92)    Sell when F&G Index > threshold (extreme greed)
  RSISellCond(threshold=65, period=14) Sell when RSI > threshold (overbought)
  FundingSellCond(threshold=0.0003)  Sell when Funding Rate SMA7 > threshold (longs overleveraged)
  OISurgeSellCond(threshold=0.20)    Sell when OI 7d change > threshold (leverage piling in)
  CBPremiumSellCond(threshold=1.0)   Sell when Coinbase Premium SMA7 > threshold (US FOMO)

CLOSE CONDITIONS (conditions.py):
  StopLossCond(threshold=-0.20)      Close if unrealized loss > threshold
  TakeProfitCond(threshold=3.0)      Close if unrealized profit > threshold
  TimeExitCond(max_days=365, min_profit=0.20)  Close after N days if profitable
  TrailingStopCond(threshold=-0.25)  Close if price drops > threshold from peak

COMBINATORS (conditions.py):
  combine_conditions(*conds, mode="AND")  All conditions must pass
  combine_conditions(*conds, mode="OR")   Any condition triggers

PRESETS (conditions.py):
  preset_fg_rsi_buy(fg_thresh=10, rsi_thresh=35)   Proven F&G+RSI buy combo
  preset_fg_rsi_sell(fg_thresh=92, rsi_thresh=65)   Proven F&G+RSI sell combo
  preset_funding_mr_buy(thresh=-0.0001)              Funding mean-reversion buy
  preset_funding_mr_sell(thresh=0.0003)              Funding mean-reversion sell
  preset_union_buy(fg_thresh=10, rsi_thresh=35, fund_thresh=-0.0001)  Buy on either signal

ENGINE (engine.py):
  load_data(start, include_fg, include_derivatives, include_cb_premium)  Load merged dataset
  backtest(df, name, buy_cond, sell_cond, close_conditions, source_aware_sell, sell_map)
  print_results(results, buy_and_hold_ret)  Print comparison table
  print_trade_log(result)                   Print detailed trades
  plot_results(results, df, highlight, output_path)  Save chart PNG
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
    # LOAD DATA
    # ================================================================
    df = load_data(
        start="2018-01-01",
        include_fg=True,
        include_derivatives=True,
        include_cb_premium=False,  # Set True to include CB Premium
    )

    # ================================================================
    # DEFINE STRATEGIES
    # Each strategy is a dict with name, buy_cond, sell_cond, and
    # optional close_conditions, source_aware_sell, sell_map.
    #
    # To test YOUR OWN strategy, simply add another entry here!
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

        # # ------ HYBRID: UNION + SOURCE-AWARE SELL (Best: +1523%) ------
        # {
        #     "name": "Union+SourceSell",
        #     "buy_cond": preset_union_buy(),
        #     "sell_cond": preset_fg_rsi_sell(),  # default sell
        #     "source_aware_sell": True,
        #     "sell_map": {
        #         "FG<10+RSI<35": preset_fg_rsi_sell(),
        #         "Fund<-0.01%": preset_funding_mr_sell(),
        #     },
        # },

        # # ------ UNION (OR) buy / OR sell ------
        # {
        #     "name": "Union (OR/OR)",
        #     "buy_cond": preset_union_buy(),
        #     "sell_cond": combine_conditions(
        #         preset_fg_rsi_sell(),
        #         preset_funding_mr_sell(),
        #         mode="OR"
        #     ),
        # },

        # ------ RELAXED F&G thresholds ------
        # {
        #     "name": "F&G(15)+RSI(40)",
        #     "buy_cond": preset_fg_rsi_buy(fg_thresh=15, rsi_thresh=60),
        #     "sell_cond": preset_fg_rsi_sell(fg_thresh=85, rsi_thresh=40),
        # },

        # # ------ F&G + RSI with RISK MANAGEMENT ------
        # {
        #     "name": "F&G+RSI+StopLoss",
        #     "buy_cond": preset_fg_rsi_buy(),
        #     "sell_cond": preset_fg_rsi_sell(),
        #     "close_conditions": [StopLossCond(-0.30)],
        # },
        # {
        #     "name": "F&G+RSI+Trailing",
        #     "buy_cond": preset_fg_rsi_buy(),
        #     "sell_cond": preset_fg_rsi_sell(),
        #     "close_conditions": [TrailingStopCond(-0.30)],
        # },

        # # ------ UNION + TIME EXIT ------
        # {
        #     "name": "Union+TimeExit(365d)",
        #     "buy_cond": preset_union_buy(),
        #     "sell_cond": preset_fg_rsi_sell(),
        #     "source_aware_sell": True,
        #     "sell_map": {
        #         "FG<10+RSI<35": preset_fg_rsi_sell(),
        #         "Fund<-0.01%": preset_funding_mr_sell(),
        #     },
        #     "close_conditions": [TimeExitCond(max_days=365, min_profit=0.20)],
        # },

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

        # ------ Step 1: Trailing Stop 系列 ------
        {
            "name": "RevRSI+Trail15%",
            "buy_cond": RSISellCond(threshold=70),
            "sell_cond": RSIBuyCond(threshold=30),
            "close_conditions": [TrailingStopCond(-0.15)],
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
        {
            "name": "RevRSI+Time180d",
            "buy_cond": RSISellCond(threshold=70),
            "sell_cond": RSIBuyCond(threshold=30),
            "close_conditions": [TimeExitCond(max_days=180, min_profit=0.10)],
        },

        # ------ Step 2: F&G 卖出增强 ------
        {
            "name": "RevRSI+FG85",
            "buy_cond": RSISellCond(threshold=70),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=30),
                FearGreedSellCond(threshold=85),
                mode="OR"),
        },
        {
            "name": "RevRSI+FG90",
            "buy_cond": RSISellCond(threshold=70),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=30),
                FearGreedSellCond(threshold=90),
                mode="OR"),
        },
        # ------ Step 3: 趋势过滤 ------
        {
            "name": "RevRSI+SMA200",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=70),
                TrendFilterCond(ma_period=200),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=30),
        },
        {
            "name": "RevRSI+SMA100",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=70),
                TrendFilterCond(ma_period=100),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=30),
        },

        # ------ Step 5: 衍生品过滤 ------
        {
            "name": "RevRSI+FundFilter",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=70),
                FundingNotOverheatedCond(max_funding=0.0003),
                mode="AND"),
            "sell_cond": combine_conditions(
                RSIBuyCond(threshold=30),
                FundingSellCond(threshold=0.0003),
                mode="OR"),
        },

        # ------ Step 6: RSI 动量方向过滤 ------
        {
            "name": "RevRSI+Rising",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=70),
                RSIRisingCond(min_delta=0, lookback=3),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=30),
        },
        # ------ Step 7: 最优组合整合 ------
        {
            "name": "RevRSI Optimized",
            "buy_cond": combine_conditions(
                RSISellCond(threshold=70),
                TrendFilterCond(ma_period=200),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=30),
            "close_conditions": [StopLossCond(-0.10)],
        },

        # ------ FG + RevRSI + SMA 三重过滤 ------
        # FG<50（排除贪婪）AND RSI>70 AND close>SMA200：最低回撤-27.2%，近1Y正收益
        {
            "name": "FG50&RevRSI&SMA200",
            "buy_cond": combine_conditions(
                FearGreedBuyCond(threshold=50),
                RSISellCond(threshold=70),
                TrendFilterCond(ma_period=200),
                mode="AND"),
            "sell_cond": RSIBuyCond(threshold=30),
        },
    ]

    strategies.append(preset_alpha_combo())

    # ------ Step 4: RSI 参数扫描 ------
    for period in [7, 10, 14, 21]:
        for buy_th in [65, 70, 75]:
            for sell_th in [25, 30, 35]:
                if period == 14 and buy_th == 70 and sell_th == 30:
                    continue  # 跳过已有的 Base 配置
                strategies.append({
                    "name": f"RevRSI({period},{buy_th}/{sell_th})",
                    "buy_cond": RSISellCond(threshold=buy_th, period=period),
                    "sell_cond": RSIBuyCond(threshold=sell_th, period=period),
                })

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
    # B&H last year
    one_year_ago = df["date"].max() - pd.Timedelta(days=365)
    df_1y = df[df["date"] >= one_year_ago]
    bh_1y = ((df_1y["close"].iloc[-1] / df_1y["close"].iloc[0]) - 1) * 100 if len(df_1y) > 1 else 0
    print_results(results, buy_and_hold_ret=bh_ret, buy_and_hold_1y=bh_1y)

    # Save detailed results and trade logs to files
    save_results_to_files(
        results,
        summary_path="d:/work/alpha_strategy/results_summary.txt",
        trade_log_path="d:/work/alpha_strategy/trade_logs.txt",
        buy_and_hold_ret=bh_ret,
        buy_and_hold_1y=bh_1y,
    )

    # ================================================================
    # PLOT
    # ================================================================
    plot_results(results, df, output_path="d:/work/AlphaGPT/strategy_test_results.png")


if __name__ == "__main__":
    main()
