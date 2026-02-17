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
from data_loader import load_all_data, load_all_data_hourly, load_btc_price_15m


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(start: str = "2018-01-01",
              include_fg: bool = True,
              include_derivatives: bool = True,
              include_cb_premium: bool = False,
              interval: str = "1d",
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
        interval: "1d" for daily, "1h" for hourly.

    Returns:
        DataFrame with columns: date, open, close, fear_greed, funding_rate,
        open_interest_usd, coinbase_premium, rsi_14, funding_sma7,
        cb_prem_sma7, oi_pct_7d, oi_valid, drawdown_30d, etc.
    """
    if interval == "1h":
        df = load_all_data_hourly(
            start=start,
            include_fg=include_fg,
            include_derivatives=include_derivatives,
            include_cb_premium=include_cb_premium,
        )
        bars_per_day = 24
    else:
        df = load_all_data(
            start=start,
            include_fg=include_fg,
            include_derivatives=include_derivatives,
            include_cb_premium=include_cb_premium,
        )
        bars_per_day = 1

    # Compute indicators
    df = _add_indicators(df, bars_per_day=bars_per_day)
    df = df.dropna(subset=["rsi_14"]).reset_index(drop=True)

    # Store bars_per_day in DataFrame metadata so backtest can use it
    df.attrs["bars_per_day"] = bars_per_day

    unit = "bars" if interval == "1h" else "days"
    print(f"Dataset: {len(df)} {unit} ({df['date'].min()} ~ {df['date'].max()})")
    return df


def _add_indicators(df: pd.DataFrame, bars_per_day: int = 1) -> pd.DataFrame:
    """
    Compute all technical indicators from raw data columns.

    Args:
        bars_per_day: 1 for daily data, 24 for hourly. All day-based periods
                      are multiplied by this factor so indicators retain
                      the same calendar-time meaning.
    """
    bpd = bars_per_day

    # RSI multi-period (keep bar-level periods, not calendar-day scaled)
    for period in [7, 10, 14, 21]:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # Coinbase Premium SMA7
    if "coinbase_premium" in df.columns:
        df["coinbase_premium"] = df["coinbase_premium"].ffill(limit=3 * bpd)
        df["cb_prem_sma7"] = df["coinbase_premium"].rolling(
            7 * bpd, min_periods=5 * bpd).mean()

    # Funding SMA7
    if "funding_rate" in df.columns:
        df["funding_rate"] = df["funding_rate"].ffill(limit=3 * bpd)
        df["funding_sma7"] = df["funding_rate"].rolling(
            7 * bpd, min_periods=5 * bpd).mean()

    # OI metrics
    if "open_interest_usd" in df.columns:
        df["oi_valid"] = df["open_interest_usd"] > 0
        df["oi_pct_7d"] = df["open_interest_usd"].pct_change(
            periods=7 * bpd, fill_method=None)

    # Drawdowns from recent high
    for lookback in [30, 60, 90]:
        high = df["close"].rolling(lookback * bpd).max()
        df[f"drawdown_{lookback}d"] = (df["close"] - high) / high

    # Trend SMAs
    for ma_period in [50, 100, 200]:
        df[f"sma_{ma_period}"] = df["close"].rolling(ma_period * bpd).mean()

    # RSI momentum direction (3-bar change, matches RSI bar-level granularity)
    df["rsi_14_delta3"] = df["rsi_14"].diff(3)

    # Volume Profile (rolling window, multiple lookbacks)
    if "volume" in df.columns and df["volume"].notna().any():
        if bpd == 24:
            # Use 15-min data for higher-resolution VP, resample back to hourly
            df = _compute_vp_from_15m(df)
        else:
            for vp_days in [3, 7, 14]:
                df = _add_volume_profile(df, bars_per_day=bpd, window_days=vp_days)

    return df


def _add_volume_profile(df: pd.DataFrame, bars_per_day: int = 24,
                         window_days: int = 7, n_bins: int = 50,
                         value_area_pct: float = 0.70) -> pd.DataFrame:
    """
    Compute rolling Volume Profile indicators: POC, VAH, VAL.

    For each bar, looks back `window_days * bars_per_day` bars, distributes
    volume into price bins, finds POC (max volume bin), then expands outward
    from POC until value_area_pct of total volume is covered.

    Args:
        df: DataFrame with 'close' and 'volume' columns.
        bars_per_day: 1 for daily, 24 for hourly.
        window_days: Lookback window in calendar days.
        n_bins: Number of price bins for the profile.
        value_area_pct: Fraction of volume for value area (default 70%).

    Adds columns: vp_poc_{window_days}d, vp_vah_{window_days}d, vp_val_{window_days}d
    """
    window = window_days * bars_per_day
    closes = df["close"].values
    volumes = df["volume"].values
    n = len(df)

    vp_poc = np.full(n, np.nan)
    vp_vah = np.full(n, np.nan)
    vp_val = np.full(n, np.nan)

    for i in range(window, n):
        c = closes[i - window:i]
        v = volumes[i - window:i]

        # Skip if insufficient data
        valid = ~np.isnan(c) & ~np.isnan(v) & (v > 0)
        if valid.sum() < window * 0.5:
            continue

        c_valid = c[valid]
        v_valid = v[valid]

        lo, hi = c_valid.min(), c_valid.max()
        if hi == lo:
            vp_poc[i] = lo
            vp_vah[i] = hi
            vp_val[i] = lo
            continue

        # Build volume profile: distribute volume into price bins
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_vol = np.zeros(n_bins)

        indices = np.clip(
            ((c_valid - lo) / (hi - lo) * (n_bins - 1)).astype(int),
            0, n_bins - 1
        )
        for idx, vol in zip(indices, v_valid):
            bin_vol[idx] += vol

        total_vol = bin_vol.sum()
        if total_vol == 0:
            continue

        # POC = bin with max volume
        poc_idx = np.argmax(bin_vol)
        vp_poc[i] = bin_centers[poc_idx]

        # Value Area: expand from POC until covering value_area_pct
        va_vol = bin_vol[poc_idx]
        lo_idx = poc_idx
        hi_idx = poc_idx
        target = total_vol * value_area_pct

        while va_vol < target:
            look_lo = bin_vol[lo_idx - 1] if lo_idx > 0 else 0
            look_hi = bin_vol[hi_idx + 1] if hi_idx < n_bins - 1 else 0

            if look_lo == 0 and look_hi == 0:
                break

            if look_lo >= look_hi:
                lo_idx -= 1
                va_vol += bin_vol[lo_idx]
            else:
                hi_idx += 1
                va_vol += bin_vol[hi_idx]

        vp_val[i] = bin_edges[lo_idx]       # lower edge of lowest VA bin
        vp_vah[i] = bin_edges[hi_idx + 1]   # upper edge of highest VA bin

    suffix = f"{window_days}d"
    df[f"vp_poc_{suffix}"] = vp_poc
    df[f"vp_vah_{suffix}"] = vp_vah
    df[f"vp_val_{suffix}"] = vp_val

    return df


def _compute_vp_from_15m(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VP indicators from 15-min data and merge into hourly DataFrame.

    1. Load 15-min OHLCV data
    2. Compute VP (POC/VAH/VAL) on 15-min bars (bars_per_day=96)
    3. Resample to hourly (take last 15-min bar per hour)
    4. Merge VP columns into df_hourly
    """
    start_date = df_hourly["date"].min().strftime("%Y-%m-%d")
    df_15m = load_btc_price_15m(start=start_date)

    if df_15m.empty or "volume" not in df_15m.columns:
        # Fallback: compute VP from hourly data
        for vp_days in [3, 7, 14]:
            df_hourly = _add_volume_profile(
                df_hourly, bars_per_day=24, window_days=vp_days)
        return df_hourly

    print(f"Computing VP from 15m data ({len(df_15m)} bars)...")

    # Compute VP on 15-min data
    for vp_days in [3, 7, 14]:
        df_15m = _add_volume_profile(
            df_15m, bars_per_day=96, window_days=vp_days)

    # Resample to hourly: take last 15-min bar per hour
    vp_cols = [c for c in df_15m.columns if c.startswith("vp_")]
    df_15m["hour"] = df_15m["date"].dt.floor("h")
    vp_hourly = df_15m.groupby("hour")[vp_cols].last().reset_index()
    vp_hourly = vp_hourly.rename(columns={"hour": "date"})

    # Drop existing VP columns from hourly df if any
    existing_vp = [c for c in df_hourly.columns if c.startswith("vp_")]
    if existing_vp:
        df_hourly = df_hourly.drop(columns=existing_vp)

    # Merge
    df_hourly = df_hourly.merge(vp_hourly, on="date", how="left")
    return df_hourly


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
             short_cond=None,
             cover_cond=None,
             close_conditions: Optional[list] = None,
             source_aware_sell: bool = False,
             sell_map: Optional[dict] = None,
             initial_capital: float = 10000.0,
             bars_per_day: int = 0) -> dict:
    """
    Run a backtest with configurable buy/sell/short/cover conditions.

    Args:
        df: DataFrame from load_data().
        name: Strategy name for results.
        buy_cond: Buy condition function (row -> str/True/False).
        sell_cond: Sell condition function (row -> str/True/False).
            If source_aware_sell=True, this is the default sell condition.
        short_cond: Optional short condition (row -> str/True/False). None = no shorting.
        cover_cond: Optional cover condition (row -> str/True/False). Required if short_cond set.
        close_conditions: Optional list of close condition objects (StopLoss, TakeProfit, etc.)
            These are checked every bar when in position. Each must have a check() method.
        source_aware_sell: If True, uses sell_map to route exits by entry source.
        sell_map: Dict mapping entry source label -> sell condition function.
            Example: {"FG<10+RSI<35": fg_rsi_sell, "Fund<-0.01%": funding_sell}
        initial_capital: Starting capital in USD.
        bars_per_day: 1 for daily, 24 for hourly. 0 = auto-detect from df.attrs.

    Returns:
        dict with keys: name, total_return, sharpe, max_drawdown, trades,
        long_trades, short_trades, trade_log, portfolio (DataFrame)

    Execution model (no look-ahead bias):
        - Signals are evaluated using bar T's close-based indicators.
        - Trades execute at bar T+1's open price.
        - Close conditions (stop-loss, etc.) use bar T's close for evaluation,
          but also execute at T+1 open.

    Short mechanics (1x, no leverage):
        - SHORT: record short_capital and entry_price
        - Mark-to-market: value = short_capital * (2 - close / entry_price)
        - COVER: capital = short_capital * (2 - exec_price / entry_price)
    """
    if bars_per_day == 0:
        bars_per_day = df.attrs.get("bars_per_day", 1)
    capital = initial_capital
    position = 0  # 0=cash, 1=long, -1=short
    btc_held = 0.0
    short_capital = 0.0  # capital at time of short entry
    trades = []
    portfolio = []
    entry_price = 0.0
    entry_date = None
    entry_source = None
    peak_price = 0.0

    # Pending signal from previous day (T-day signal -> T+1 execution)
    pending_buy = None    # str source label, or None
    pending_sell = None   # str source label, or None
    pending_short = None  # str source label, or None
    pending_cover = None  # str source label, or None

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

        elif pending_short and position == 0 and short_cond is not None:
            short_capital = capital
            trades.append({"type": "SHORT", "date": date, "price": exec_price,
                          "capital": capital, "source": pending_short})
            capital = 0
            position = -1
            entry_price = exec_price
            entry_date = date
            entry_source = pending_short
            pending_short = None

        elif pending_cover and position == -1:
            capital = short_capital * (2 - exec_price / entry_price)
            capital = max(capital, 0)  # floor at 0 (can't go negative without leverage)
            trades.append({"type": "COVER", "date": date, "price": exec_price,
                          "capital": capital, "source": pending_cover})
            short_capital = 0
            position = 0
            entry_source = None
            entry_date = None
            pending_cover = None

        # Clear stale pending signals
        pending_buy = None
        pending_sell = None
        pending_short = None
        pending_cover = None

        # === PHASE 2: Evaluate signals using today's close (for tomorrow's execution) ===
        if position == 0:
            signal = buy_cond(row)
            if signal:
                pending_buy = signal if isinstance(signal, str) else "BUY"
            elif short_cond is not None:
                signal = short_cond(row)
                if signal:
                    pending_short = signal if isinstance(signal, str) else "SHORT"

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
                days_held = (date - entry_date).total_seconds() / 86400 if entry_date else 0
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

        elif position == -1 and cover_cond is not None:
            s = cover_cond(row)
            if s:
                pending_cover = s if isinstance(s, str) else "COVER"

        # Track portfolio value (mark-to-market at close)
        if position == 0:
            value = capital
        elif position == 1:
            value = btc_held * close
        else:  # position == -1
            value = short_capital * (2 - close / entry_price)
            value = max(value, 0)
        portfolio.append({"date": date, "value": value, "position": position})

    # If still holding, mark-to-market
    if position == 1:
        capital = btc_held * df.iloc[-1]["close"]
    elif position == -1:
        last_close = df.iloc[-1]["close"]
        capital = short_capital * (2 - last_close / entry_price)
        capital = max(capital, 0)

    pv = pd.DataFrame(portfolio)
    ret = ((capital - initial_capital) / initial_capital) * 100
    daily_ret = pv["value"].pct_change().dropna()
    annualize = np.sqrt(365 * bars_per_day)
    sharpe = (daily_ret.mean() / daily_ret.std()) * annualize if daily_ret.std() > 0 else 0
    max_dd = ((pv["value"] - pv["value"].cummax()) / pv["value"].cummax()).min() * 100

    # Last 1 year metrics
    last_date = pv["date"].max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    pv_1y = pv[pv["date"] >= one_year_ago].copy()
    if len(pv_1y) > 1:
        ret_1y = ((pv_1y["value"].iloc[-1] / pv_1y["value"].iloc[0]) - 1) * 100
        dr_1y = pv_1y["value"].pct_change().dropna()
        sharpe_1y = (dr_1y.mean() / dr_1y.std()) * annualize if dr_1y.std() > 0 else 0
        dd_1y = ((pv_1y["value"] - pv_1y["value"].cummax()) / pv_1y["value"].cummax()).min() * 100
    else:
        ret_1y, sharpe_1y, dd_1y = 0.0, 0.0, 0.0

    long_trades = len([t for t in trades if t["type"] == "BUY"])
    short_trades = len([t for t in trades if t["type"] == "SHORT"])

    return {
        "name": name,
        "total_return": ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "return_1y": ret_1y,
        "sharpe_1y": sharpe_1y,
        "max_drawdown_1y": dd_1y,
        "trades": long_trades + short_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "trade_log": trades,
        "portfolio": pv,
    }


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def _format_results_table(results: list, buy_and_hold_ret: Optional[float] = None,
                          buy_and_hold_1y: Optional[float] = None) -> str:
    """Format strategy results into a table string."""
    w = 130
    lines = []
    lines.append("=" * w)
    lines.append(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>9}"
                 f"  |  {'1Y Ret':>9} {'1Y Shrp':>8} {'1Y MDD':>8}")
    lines.append("-" * w)
    for res in results:
        long_t = res.get('long_trades', res['trades'])
        short_t = res.get('short_trades', 0)
        if short_t > 0:
            trades_str = f"{long_t}L/{short_t}S"
        else:
            trades_str = str(long_t)
        lines.append(f"{res['name']:<25} {res['total_return']:>9.1f}% {res['sharpe']:>7.3f} "
                     f"{res['max_drawdown']:>7.1f}% {trades_str:>9}"
                     f"  |  {res.get('return_1y', 0):>8.1f}% {res.get('sharpe_1y', 0):>7.3f} "
                     f"{res.get('max_drawdown_1y', 0):>7.1f}%")
    if buy_and_hold_ret is not None:
        bh_1y_str = f"{buy_and_hold_1y:>8.1f}%" if buy_and_hold_1y is not None else f"{'N/A':>9}"
        lines.append(f"{'Buy & Hold':<25} {buy_and_hold_ret:>9.1f}%{'':>17}"
                     f"  |  {bh_1y_str}")
    lines.append("=" * w)

    if results:
        best_s = max(results, key=lambda x: x["sharpe"])
        best_r = max(results, key=lambda x: x["total_return"])
        lines.append(f"\nBest Sharpe:  {best_s['name']} ({best_s['sharpe']:.3f})")
        lines.append(f"Best Return:  {best_r['name']} ({best_r['total_return']:.1f}%)")
    return "\n".join(lines)


def _format_trade_log(result: dict) -> str:
    """Format trade log for a single strategy."""
    lines = []
    lines.append(f"\n--- {result['name']} ({result['trades']} entries, {result['total_return']:.0f}%) ---")
    for t in result["trade_log"]:
        src = t.get("source", "")
        date_str = t['date'].strftime('%Y-%m-%d %H:%M') if hasattr(t['date'], 'hour') and t['date'].hour != 0 else t['date'].strftime('%Y-%m-%d')
        lines.append(f"  {t['type']:>4} | {date_str} | "
                     f"${t['price']:>10,.0f} | Cap: ${t['capital']:>10,.0f} | {src}")

    # Last 1-year trade summary
    lines.append(_format_1y_trade_summary(result))
    return "\n".join(lines)


def _format_1y_trade_summary(result: dict) -> str:
    """Format a summary of the last 1-year trades for a strategy."""
    pv = result.get("portfolio")
    trade_log = result.get("trade_log", [])
    if pv is None or pv.empty or not trade_log:
        return ""

    last_date = pv["date"].max()
    one_year_ago = last_date - pd.Timedelta(days=365)

    # Filter trades in last year
    trades_1y = [t for t in trade_log if t["date"] >= one_year_ago]
    entries_1y = [t for t in trades_1y if t["type"] in ("BUY", "SHORT")]

    if not trades_1y:
        return f"\n  [Last 1Y] No trades since {one_year_ago.strftime('%Y-%m-%d')}"

    lines = []
    lines.append(f"\n  {'='*70}")
    lines.append(f"  [Last 1Y Summary] {one_year_ago.strftime('%Y-%m-%d')} ~ {last_date.strftime('%Y-%m-%d')}")
    lines.append(f"  Return: {result.get('return_1y', 0):.1f}%  |  "
                 f"Sharpe: {result.get('sharpe_1y', 0):.3f}  |  "
                 f"MaxDD: {result.get('max_drawdown_1y', 0):.1f}%  |  "
                 f"Trades: {len(entries_1y)}")

    # Win/loss breakdown from paired trades (long: BUY->SELL, short: SHORT->COVER)
    pairs = []
    for i, t in enumerate(trades_1y):
        if t["type"] == "SELL":
            idx_in_full = trade_log.index(t)
            entry_t = None
            for j in range(idx_in_full - 1, -1, -1):
                if trade_log[j]["type"] == "BUY":
                    entry_t = trade_log[j]
                    break
            if entry_t:
                pnl_pct = (t["price"] / entry_t["price"] - 1) * 100
                pairs.append(pnl_pct)
        elif t["type"] == "COVER":
            idx_in_full = trade_log.index(t)
            entry_t = None
            for j in range(idx_in_full - 1, -1, -1):
                if trade_log[j]["type"] == "SHORT":
                    entry_t = trade_log[j]
                    break
            if entry_t:
                # Short profit: entry sold high, covered low
                pnl_pct = (entry_t["price"] / t["price"] - 1) * 100
                pairs.append(pnl_pct)

    if pairs:
        wins = [p for p in pairs if p > 0]
        losses = [p for p in pairs if p <= 0]
        win_rate = len(wins) / len(pairs) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        lines.append(f"  Win rate: {win_rate:.0f}% ({len(wins)}W/{len(losses)}L)  |  "
                     f"Avg win: +{avg_win:.1f}%  |  Avg loss: {avg_loss:.1f}%")

    # Last year trade list
    lines.append(f"  {'-'*70}")
    for t in trades_1y:
        src = t.get("source", "")
        date_str = t['date'].strftime('%Y-%m-%d %H:%M') if hasattr(t['date'], 'hour') and t['date'].hour != 0 else t['date'].strftime('%Y-%m-%d')
        lines.append(f"  {t['type']:>4} | {date_str} | "
                     f"${t['price']:>10,.0f} | Cap: ${t['capital']:>10,.0f} | {src}")
    lines.append(f"  {'='*70}")

    return "\n".join(lines)


def print_results(results: list, buy_and_hold_ret: Optional[float] = None,
                  buy_and_hold_1y: Optional[float] = None):
    """Print a formatted comparison table of strategy results to console."""
    print("\n" + _format_results_table(results, buy_and_hold_ret, buy_and_hold_1y))


def print_trade_log(result: dict):
    """Print detailed trade log for a single strategy result."""
    print(_format_trade_log(result))


def save_results_to_files(results: list,
                          summary_path: str,
                          trade_log_path: str,
                          buy_and_hold_ret: Optional[float] = None,
                          buy_and_hold_1y: Optional[float] = None):
    """
    Save strategy results summary and trade logs to separate files.

    Args:
        results: List of result dicts from backtest().
        summary_path: Path to save the results summary table.
        trade_log_path: Path to save the trade logs.
        buy_and_hold_ret: Optional B&H total return.
        buy_and_hold_1y: Optional B&H last-year return.
    """
    # Summary file
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(_format_results_table(results, buy_and_hold_ret, buy_and_hold_1y))
        f.write("\n")
    print(f"Results saved to {summary_path}")

    # Trade log file — all strategies sorted by return
    sorted_results = sorted(results, key=lambda x: x["total_return"], reverse=True)
    with open(trade_log_path, "w", encoding="utf-8") as f:
        for res in sorted_results:
            f.write(_format_trade_log(res))
            f.write("\n")
    print(f"Trade logs saved to {trade_log_path}")


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
