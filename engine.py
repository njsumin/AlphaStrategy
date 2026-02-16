"""
Strategy Backtest Engine (engine.py)
=====================================
Indicator computation, backtest execution, and result plotting.
Data downloading is handled by data_loader.py.

Usage:
------
    from engine import load_data, backtest, print_results, plot_results

    df = load_data()
    result = backtest(df, "My Strategy", buy_cond, sell_cond)
    print_results([result])
    plot_results([result], df)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, List, Callable, Union
from data_loader import load_all_data


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(start: str = "2018-01-01",
              include_fg: bool = True,
              include_derivatives: bool = True,
              include_cb_premium: bool = False,
              ) -> pd.DataFrame:
    """
    Load all data sources and compute indicators.

    Data downloading is delegated to data_loader.py.
    This function adds technical indicators on top of the raw data.

    Args:
        start: Start date for BTC price data.
        include_fg: Include Fear & Greed Index.
        include_derivatives: Include Binance Funding Rate and OI.
        include_cb_premium: Include Coinbase Premium proxy.

    Returns:
        DataFrame with columns: date, open, close, fear_greed, funding_rate,
        open_interest_usd, coinbase_premium, rsi_14, funding_sma7,
        cb_prem_sma7, oi_pct_7d, oi_valid, drawdown_30d, etc.
    """
    df = load_all_data(
        start=start,
        include_fg=include_fg,
        include_derivatives=include_derivatives,
        include_cb_premium=include_cb_premium,
    )

    # Compute indicators
    df = _add_indicators(df)
    df = df.dropna(subset=["rsi_14"]).reset_index(drop=True)

    print(f"\nDataset ready: {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")
    _print_coverage(df)
    return df


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators from raw data columns."""
    # RSI-14
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Coinbase Premium SMA7
    if "coinbase_premium" in df.columns:
        df["coinbase_premium"] = df["coinbase_premium"].ffill(limit=3)
        df["cb_prem_sma7"] = df["coinbase_premium"].rolling(7, min_periods=5).mean()

    # Funding SMA7
    if "funding_rate" in df.columns:
        df["funding_rate"] = df["funding_rate"].ffill(limit=3)
        df["funding_sma7"] = df["funding_rate"].rolling(7, min_periods=5).mean()

    # OI metrics
    if "open_interest_usd" in df.columns:
        df["oi_valid"] = df["open_interest_usd"] > 0
        df["oi_pct_7d"] = df["open_interest_usd"].pct_change(periods=7, fill_method=None)

    # Drawdowns from recent high
    for lookback in [30, 60, 90]:
        high = df["close"].rolling(lookback).max()
        df[f"drawdown_{lookback}d"] = (df["close"] - high) / high

    return df


def _print_coverage(df: pd.DataFrame):
    """Print data coverage summary."""
    for col, label in [("fear_greed", "F&G"), ("funding_rate", "Funding"),
                       ("open_interest_usd", "OI"), ("coinbase_premium", "CB Premium")]:
        if col in df.columns:
            valid = df[col].notna().sum()
            if valid > 0:
                print(f"  {label}: {valid} days")


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def backtest(df: pd.DataFrame,
             name: str,
             buy_cond,
             sell_cond,
             close_conditions: Optional[list] = None,
             source_aware_sell: bool = False,
             sell_map: Optional[dict] = None,
             initial_capital: float = 10000.0) -> dict:
    """
    Run a backtest with configurable buy/sell/close conditions.

    Args:
        df: DataFrame from load_data().
        name: Strategy name for results.
        buy_cond: Buy condition function (row -> str/True/False).
        sell_cond: Sell condition function (row -> str/True/False).
            If source_aware_sell=True, this is the default sell condition.
        close_conditions: Optional list of close condition objects (StopLoss, TakeProfit, etc.)
            These are checked every bar when in position. Each must have a check() method.
        source_aware_sell: If True, uses sell_map to route exits by entry source.
        sell_map: Dict mapping entry source label -> sell condition function.
            Example: {"FG<10+RSI<35": fg_rsi_sell, "Fund<-0.01%": funding_sell}
        initial_capital: Starting capital in USD.

    Returns:
        dict with keys: name, total_return, sharpe, max_drawdown, trades,
        trade_log, portfolio (DataFrame)

    Execution model (no look-ahead bias):
        - Signals are evaluated using day T's close-based indicators.
        - Trades execute at day T+1's open price.
        - Close conditions (stop-loss, etc.) use day T's close for evaluation,
          but also execute at T+1 open.
    """
    capital = initial_capital
    position = 0  # 0=cash, 1=holding
    btc_held = 0.0
    trades = []
    portfolio = []
    entry_price = 0.0
    entry_date = None
    entry_source = None
    peak_price = 0.0

    # Pending signal from previous day (T-day signal -> T+1 execution)
    pending_buy = None    # str source label, or None
    pending_sell = None   # str source label, or None

    for _, row in df.iterrows():
        date = row["date"]
        close = row["close"]
        exec_price = row.get("open", close)  # T+1 open; fallback to close if open unavailable

        # === PHASE 1: Execute pending signals at today's open ===
        if pending_buy and position == 0:
            btc_held = capital / exec_price
            trades.append({"type": "BUY", "date": date, "price": exec_price,
                          "capital": capital, "source": pending_buy})
            capital = 0
            position = 1
            entry_price = exec_price
            entry_date = date
            entry_source = pending_buy
            peak_price = exec_price
            pending_buy = None

        elif pending_sell and position == 1:
            capital = btc_held * exec_price
            trades.append({"type": "SELL", "date": date, "price": exec_price,
                          "capital": capital, "source": pending_sell})
            btc_held = 0
            position = 0
            entry_source = None
            entry_date = None
            pending_sell = None

        # Clear stale pending signals
        pending_buy = None
        pending_sell = None

        # === PHASE 2: Evaluate signals using today's close (for tomorrow's execution) ===
        if position == 0:
            signal = buy_cond(row)
            if signal:
                pending_buy = signal if isinstance(signal, str) else "BUY"

        elif position == 1:
            peak_price = max(peak_price, close)
            should_sell = False
            sell_source = ""

            # Check sell conditions
            if source_aware_sell and sell_map and entry_source in sell_map:
                s = sell_map[entry_source](row)
                if s:
                    should_sell = True
                    sell_source = s if isinstance(s, str) else "SELL"
            else:
                s = sell_cond(row)
                if s:
                    should_sell = True
                    sell_source = s if isinstance(s, str) else "SELL"

            # Check close conditions
            if not should_sell and close_conditions:
                days_held = (date - entry_date).days if entry_date else 0
                for cc in close_conditions:
                    if hasattr(cc, 'check'):
                        import inspect
                        sig = inspect.signature(cc.check)
                        params = list(sig.parameters.keys())
                        if 'days_held' in params:
                            r = cc.check(close, entry_price, days_held)
                        elif 'peak_price' in params:
                            r = cc.check(close, peak_price)
                        else:
                            r = cc.check(close, entry_price)
                        if r:
                            should_sell = True
                            sell_source = r if isinstance(r, str) else cc.name
                            break

            if should_sell:
                pending_sell = sell_source

        # Track portfolio value (mark-to-market at close)
        value = capital if position == 0 else btc_held * close
        portfolio.append({"date": date, "value": value, "position": position})

    # If still holding, mark-to-market
    if position == 1:
        capital = btc_held * df.iloc[-1]["close"]

    pv = pd.DataFrame(portfolio)
    ret = ((capital - initial_capital) / initial_capital) * 100
    daily_ret = pv["value"].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() > 0 else 0
    max_dd = ((pv["value"] - pv["value"].cummax()) / pv["value"].cummax()).min() * 100

    # Last 1 year metrics
    last_date = pv["date"].max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    pv_1y = pv[pv["date"] >= one_year_ago].copy()
    if len(pv_1y) > 1:
        ret_1y = ((pv_1y["value"].iloc[-1] / pv_1y["value"].iloc[0]) - 1) * 100
        dr_1y = pv_1y["value"].pct_change().dropna()
        sharpe_1y = (dr_1y.mean() / dr_1y.std()) * np.sqrt(365) if dr_1y.std() > 0 else 0
        dd_1y = ((pv_1y["value"] - pv_1y["value"].cummax()) / pv_1y["value"].cummax()).min() * 100
    else:
        ret_1y, sharpe_1y, dd_1y = 0.0, 0.0, 0.0

    return {
        "name": name,
        "total_return": ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "return_1y": ret_1y,
        "sharpe_1y": sharpe_1y,
        "max_drawdown_1y": dd_1y,
        "trades": len([t for t in trades if t["type"] == "BUY"]),
        "trade_log": trades,
        "portfolio": pv,
    }


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def print_results(results: list, buy_and_hold_ret: Optional[float] = None,
                  buy_and_hold_1y: Optional[float] = None):
    """
    Print a formatted comparison table of strategy results.

    Args:
        results: List of result dicts from backtest().
        buy_and_hold_ret: Optional B&H total return for comparison.
        buy_and_hold_1y: Optional B&H last-year return for comparison.
    """
    w = 130
    print("\n" + "=" * w)
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}"
          f"  |  {'1Y Ret':>9} {'1Y Shrp':>8} {'1Y MDD':>8}")
    print("-" * w)
    for res in results:
        print(f"{res['name']:<25} {res['total_return']:>9.1f}% {res['sharpe']:>7.3f} "
              f"{res['max_drawdown']:>7.1f}% {res['trades']:>6}"
              f"  |  {res.get('return_1y', 0):>8.1f}% {res.get('sharpe_1y', 0):>7.3f} "
              f"{res.get('max_drawdown_1y', 0):>7.1f}%")
    if buy_and_hold_ret is not None:
        bh_1y_str = f"{buy_and_hold_1y:>8.1f}%" if buy_and_hold_1y is not None else f"{'N/A':>9}"
        print(f"{'Buy & Hold':<25} {buy_and_hold_ret:>9.1f}%{'':>17}"
              f"  |  {bh_1y_str}")
    print("=" * w)

    if results:
        best_s = max(results, key=lambda x: x["sharpe"])
        best_r = max(results, key=lambda x: x["total_return"])
        print(f"\nBest Sharpe:  {best_s['name']} ({best_s['sharpe']:.3f})")
        print(f"Best Return:  {best_r['name']} ({best_r['total_return']:.1f}%)")


def print_trade_log(result: dict):
    """Print detailed trade log for a single strategy result."""
    print(f"\n--- {result['name']} ({result['trades']} entries, {result['total_return']:.0f}%) ---")
    for t in result["trade_log"]:
        src = t.get("source", "")
        print(f"  {t['type']:>4} | {t['date'].strftime('%Y-%m-%d')} | "
              f"${t['price']:>10,.0f} | Cap: ${t['capital']:>10,.0f} | {src}")


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(results: list, df: pd.DataFrame,
                 highlight: Optional[str] = None,
                 output_path: str = "d:/work/AlphaGPT/strategy_results.png"):
    """
    Plot strategy comparison with indicator panels.

    Args:
        results: List of result dicts from backtest().
        df: DataFrame from load_data().
        highlight: Name of strategy to highlight (defaults to best Sharpe).
        output_path: Path to save the chart PNG.
    """
    if not results:
        return

    best = None
    if highlight:
        best = next((r for r in results if r["name"] == highlight), None)
    if not best:
        best = max(results, key=lambda x: x["sharpe"])

    # Determine panels
    panels = ["portfolio", "price"]
    if "fear_greed" in df.columns and df["fear_greed"].notna().any():
        panels.append("fg")
    if "funding_sma7" in df.columns and df["funding_sma7"].notna().any():
        panels.append("funding")
    if "cb_prem_sma7" in df.columns and df["cb_prem_sma7"].notna().any():
        panels.append("cb_premium")

    n = len(panels)
    ratios = [3 if p == "portfolio" else 2 for p in panels]
    fig, axes = plt.subplots(n, 1, figsize=(18, 4 * n), gridspec_kw={"height_ratios": ratios})
    if n == 1:
        axes = [axes]
    fig.suptitle("Strategy Comparison", fontsize=16, fontweight="bold")

    for i, panel in enumerate(panels):
        ax = axes[i]
        if panel == "portfolio":
            for res in results:
                pv = res["portfolio"]
                lw = 2.5 if res["name"] == best["name"] else 1.0
                alpha = 1.0 if res["name"] == best["name"] else 0.6
                ax.plot(pv["date"], pv["value"],
                        label=f'{res["name"]} ({res["total_return"]:.0f}%)',
                        linewidth=lw, alpha=alpha)
            p0 = df["close"].iloc[0]
            bh = (df["close"] / p0) * 10000
            ax.plot(df["date"], bh, label=f'B&H ({((df["close"].iloc[-1]/p0-1)*100):.0f}%)',
                    color="gray", ls=":", alpha=0.5)
            for t in best["trade_log"]:
                c = "green" if t["type"] == "BUY" else "red"
                ax.axvline(t["date"], color=c, alpha=0.15, lw=0.8)
            ax.set_ylabel("Portfolio ($)")
            ax.legend(loc="upper left", fontsize=7)
            ax.set_yscale("log")
            ax.set_title("Portfolio (Log Scale)")

        elif panel == "price":
            ax.plot(df["date"], df["close"], color="orange", lw=1)
            for t in best["trade_log"]:
                m = "^" if t["type"] == "BUY" else "v"
                c = "green" if t["type"] == "BUY" else "red"
                ax.scatter(t["date"], t["price"], color=c, marker=m, s=100, zorder=5)
                src = t.get("source", "")
                ax.annotate(src, (t["date"], t["price"]),
                           textcoords="offset points",
                           xytext=(5, 10 if t["type"] == "BUY" else -15),
                           fontsize=7, color=c, fontweight="bold")
            ax.set_ylabel("BTC ($)")
            ax.set_title(f"BTC + {best['name']} Signals")
            ax.set_yscale("log")

        elif panel == "fg":
            v = df.dropna(subset=["fear_greed"])
            ax.fill_between(v["date"], v["fear_greed"], alpha=0.3, color="orange")
            ax.plot(v["date"], v["fear_greed"], color="darkorange", lw=0.8)
            ax.axhline(10, color="green", ls="--", alpha=0.5, label="Buy <10")
            ax.axhline(92, color="red", ls="--", alpha=0.5, label="Sell >92")
            ax.set_ylim(0, 100)
            ax.set_ylabel("F&G")
            ax.set_title("Fear & Greed Index")
            ax.legend(loc="upper left", fontsize=8)

        elif panel == "funding":
            v = df.dropna(subset=["funding_sma7"])
            ax.plot(v["date"], v["funding_sma7"], color="darkviolet", lw=1.2, label="SMA7")
            ax.axhline(0, color="black", lw=0.5)
            ax.axhline(-0.0001, color="green", ls="--", alpha=0.5, label="Buy")
            ax.axhline(0.0003, color="red", ls="--", alpha=0.5, label="Sell")
            ax.set_ylabel("Funding Rate")
            ax.set_title("Funding Rate (8h)")
            ax.legend(loc="upper left", fontsize=8)

        elif panel == "cb_premium":
            v = df.dropna(subset=["cb_prem_sma7"])
            ax.plot(v["date"], v["cb_prem_sma7"], color="navy", lw=1.2, label="SMA7")
            ax.axhline(0, color="black", lw=0.5)
            ax.axhline(-0.5, color="green", ls="--", alpha=0.6, label="Buy <-0.5%")
            ax.axhline(1.0, color="red", ls="--", alpha=0.6, label="Sell >1.0%")
            ax.set_ylabel("CB Premium (%)")
            ax.set_title("Coinbase Premium")
            ax.legend(loc="upper left", fontsize=8)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to {output_path}")
