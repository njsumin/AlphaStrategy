"""
analyze_regime_returns.py
==========================
统计1h Baseline_D 各笔交易的收益率分布，按牛市/熊市分类。

牛熊判定：使用全量历史数据（允许未来函数），方法：
  - 以日线价格计算「居中移动平均」(centered MA, 前后各N天)
  - close > centered_MA → 牛市；否则 → 熊市
  该分类仅用于事后统计分析，不作为交易信号。

运行：python analyze_regime_returns.py
输出：控制台统计表 + 分布直方图（可选）
"""

import re
import os
import sys
import numpy as np
import pandas as pd

TRADE_LOG = "trade_logs_sr_momentum.txt"
PRICE_CSV  = "data/binance/btc_hourly.csv"   # 1h价格数据
CENTERED_WINDOW = 90   # 居中MA前后各90天（共180天），较平滑的周期级别判断


# ─── 1. 解析交易日志 ────────────────────────────────────────────────────────

def parse_trade_log(path: str, strategy: str = "Baseline_D") -> pd.DataFrame:
    """
    解析交易日志，提取交易对的进出场价格、时间、方向。
    返回 DataFrame，每行为一笔完整交易（进出场配对）。
    """
    trades = []
    in_strategy = False
    pending = None  # 待匹配的开仓记录

    entry_re = re.compile(
        r"\s*(BUY|SELL|SHORT|COVER)\s*\|\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2})"
        r"\s*\|\s*\$\s*([\d,]+)\s*\|\s*Cap:\s*\$\s*([\d,]+)"
    )

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            # 检测策略区块开始
            if line.startswith("---"):
                in_strategy = strategy in line
                pending = None
                continue
            if not in_strategy:
                continue

            m = entry_re.search(line)
            if not m:
                continue

            action = m.group(1)
            ts     = pd.Timestamp(m.group(2))
            price  = float(m.group(3).replace(",", ""))
            cap    = float(m.group(4).replace(",", ""))

            if action in ("BUY", "SHORT"):
                pending = {"direction": "long" if action == "BUY" else "short",
                           "entry_time": ts, "entry_price": price, "cap_before": cap}
            elif action in ("SELL", "COVER") and pending is not None:
                if action == "SELL" and pending["direction"] == "long":
                    ret = price / pending["entry_price"] - 1
                elif action == "COVER" and pending["direction"] == "short":
                    ret = pending["entry_price"] / price - 1
                else:
                    pending = None
                    continue

                exit_line = line.split("|")
                exit_reason = exit_line[-1].strip() if len(exit_line) > 3 else ""

                trades.append({
                    "direction"  : pending["direction"],
                    "entry_time" : pending["entry_time"],
                    "exit_time"  : ts,
                    "entry_price": pending["entry_price"],
                    "exit_price" : price,
                    "return_pct" : ret * 100,
                    "exit_reason": exit_reason,
                })
                pending = None

    return pd.DataFrame(trades)


# ─── 2. 牛熊分类（允许未来函数） ────────────────────────────────────────────

def compute_bull_bear_regime(price_csv: str, window_days: int = CENTERED_WINDOW
                             ) -> pd.DataFrame:
    """
    基于全量历史数据用居中MA（前后各 window_days 天）判断牛熊。
    close > centered_MA → bull (+1)
    close <= centered_MA → bear (-1)

    居中MA = (rolling(window, min_periods=1).mean() on forward+backward)
    等价于: centered_ma[i] = mean(close[i-W..i+W])

    返回 daily DataFrame: [date, close, centered_ma, regime]
    """
    df = pd.read_csv(price_csv)
    df["date"] = pd.to_datetime(df["date"])
    # 日线重采样
    daily = (df.resample("D", on="date")["close"].last()
               .dropna().reset_index())
    daily.columns = ["date", "close"]

    cl = daily["close"].values
    n  = len(cl)
    centered_ma = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window_days)
        hi = min(n, i + window_days + 1)
        centered_ma[i] = cl[lo:hi].mean()

    daily["centered_ma"] = centered_ma
    daily["regime"]      = np.where(daily["close"] > daily["centered_ma"],
                                    "bull", "bear")
    return daily


# ─── 3. 主分析 ──────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(TRADE_LOG):
        print(f"[ERROR] 未找到 {TRADE_LOG}，请先运行 strategy_test_sr_momentum.py")
        sys.exit(1)
    if not os.path.exists(PRICE_CSV):
        print(f"[ERROR] 未找到 {PRICE_CSV}")
        sys.exit(1)

    print("解析交易日志...", flush=True)
    trades = parse_trade_log(TRADE_LOG, strategy="Baseline_D")
    print(f"  总笔数: {len(trades)} (多:{len(trades[trades.direction=='long'])}"
          f" 空:{len(trades[trades.direction=='short'])})")

    print(f"计算牛熊制度（居中MA {CENTERED_WINDOW}天）...", flush=True)
    regime_df = compute_bull_bear_regime(PRICE_CSV, window_days=CENTERED_WINDOW)
    regime_map = dict(zip(regime_df["date"].dt.normalize(), regime_df["regime"]))

    # 为每笔交易打上牛熊标签（按入场日期）
    trades["regime"] = (trades["entry_time"].dt.normalize()
                                            .map(regime_map)
                                            .fillna("unknown"))

    bull_pct = (regime_df["regime"] == "bull").mean() * 100
    bear_pct = (regime_df["regime"] == "bear").mean() * 100
    print(f"  历史期间: 牛市={bull_pct:.1f}%  熊市={bear_pct:.1f}%")

    # ── 分组统计 ──────────────────────────────────────────────────────────────
    def stats(series: pd.Series, label: str):
        if len(series) == 0:
            return {"label": label, "n": 0}
        return {
            "label"     : label,
            "n"         : len(series),
            "win_rate"  : (series > 0).mean() * 100,
            "mean_ret"  : series.mean(),
            "median_ret": series.median(),
            "std_ret"   : series.std(),
            "p10"       : series.quantile(0.10),
            "p25"       : series.quantile(0.25),
            "p75"       : series.quantile(0.75),
            "p90"       : series.quantile(0.90),
            "max_win"   : series.max(),
            "max_loss"  : series.min(),
        }

    groups = [
        ("多单-牛市", "long",  "bull"),
        ("多单-熊市", "long",  "bear"),
        ("空单-熊市", "short", "bear"),
        ("空单-牛市", "short", "bull"),
    ]

    rows = []
    for label, direction, regime in groups:
        mask = (trades["direction"] == direction) & (trades["regime"] == regime)
        rows.append(stats(trades.loc[mask, "return_pct"], label))

    # 全体汇总
    rows.append(stats(trades.loc[trades.direction=="long", "return_pct"],  "多单-全部"))
    rows.append(stats(trades.loc[trades.direction=="short", "return_pct"], "空单-全部"))
    rows.append(stats(trades["return_pct"], "全部交易"))

    # ── 打印表格 ──────────────────────────────────────────────────────────────
    hdr = (f"\n{'':=<100}\n"
           f"  1h Baseline_D 交易收益分布  (牛熊制度: 居中MA{CENTERED_WINDOW}天, 含未来函数)\n"
           f"{'':=<100}")
    print(hdr)
    col = (f"  {'类别':<12} {'笔数':>5} {'胜率':>7} {'均值':>7} {'中位':>7} "
           f"{'标准差':>7} {'P10':>7} {'P25':>7} {'P75':>7} {'P90':>7} "
           f"{'最大盈':>8} {'最大亏':>8}")
    print(col)
    print(f"  {'-'*98}")
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['label']:<12} {'0':>5}")
            continue
        sep = "  " if r["label"].endswith("全部") else ""
        print(
            f"{sep}  {r['label']:<12} {r['n']:>5} {r['win_rate']:>6.1f}% "
            f"{r['mean_ret']:>6.2f}% {r['median_ret']:>6.2f}% "
            f"{r['std_ret']:>6.2f}% {r['p10']:>6.2f}% {r['p25']:>6.2f}% "
            f"{r['p75']:>6.2f}% {r['p90']:>6.2f}% "
            f"{r['max_win']:>7.2f}% {r['max_loss']:>7.2f}%"
        )

    # ── 退出原因分析 ──────────────────────────────────────────────────────────
    print(f"\n  {'退出原因分析':}")
    print(f"  {'-'*60}")
    for (direction, regime), grp in trades.groupby(["direction", "regime"]):
        label = f"{'多' if direction=='long' else '空'}单-{'牛' if regime=='bull' else '熊'}市"
        reason_stats = grp.groupby("exit_reason")["return_pct"].agg(["count","mean"])
        reason_stats = reason_stats.sort_values("count", ascending=False)
        print(f"\n  [{label}]  n={len(grp)}")
        for reason, row2 in reason_stats.iterrows():
            print(f"    {reason:<25} 笔数={int(row2['count']):>4}  均收益={row2['mean']:>6.2f}%")

    # ── 简单分位分布（文字版直方图）──────────────────────────────────────────
    print(f"\n  {'收益率区间分布':}")
    print(f"  {'-'*60}")
    bins  = [-100, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 100]
    blabs = ["<-3%", "-3~-2%", "-2~-1%", "-1~-0.5%", "-0.5~0%",
             "0~0.5%", "0.5~1%", "1~2%", "2~3%", ">3%"]

    for (direction, regime), grp in trades.groupby(["direction", "regime"]):
        label = f"{'多' if direction=='long' else '空'}单-{'牛' if regime=='bull' else '熊'}市"
        ret   = grp["return_pct"]
        counts, _ = np.histogram(ret, bins=bins)
        total = len(ret)
        print(f"\n  [{label}]  n={total}  mean={ret.mean():.2f}%  win={((ret>0).sum()/total*100):.1f}%")
        bar_max = counts.max()
        for lbl, cnt in zip(blabs, counts):
            bar_len = int(cnt / max(bar_max, 1) * 30)
            pct = cnt / total * 100
            print(f"    {lbl:>10}  {'█'*bar_len:<30}  {cnt:>4}笔 ({pct:4.1f}%)")

    # ── 可选：生成图表 ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"1h Baseline_D 交易收益分布  (牛熊制度: 居中MA{CENTERED_WINDOW}天)",
                     fontsize=13)

        plot_groups = [
            ("多单-牛市", "long",  "bull", axes[0, 0], "steelblue"),
            ("多单-熊市", "long",  "bear", axes[0, 1], "tomato"),
            ("空单-熊市", "short", "bear", axes[1, 0], "tomato"),
            ("空单-牛市", "short", "bull", axes[1, 1], "steelblue"),
        ]

        for label, direction, regime, ax, color in plot_groups:
            mask = (trades["direction"] == direction) & (trades["regime"] == regime)
            ret  = trades.loc[mask, "return_pct"]
            if len(ret) == 0:
                ax.set_title(f"{label} (n=0)")
                continue
            ax.hist(ret, bins=40, color=color, alpha=0.7, edgecolor="white",
                    linewidth=0.5)
            ax.axvline(0, color="black", linewidth=1, linestyle="--")
            ax.axvline(ret.mean(), color="red", linewidth=1.5, linestyle="-",
                       label=f"均值={ret.mean():.2f}%")
            ax.set_title(f"{label}  n={len(ret)}  胜率={((ret>0).mean()*100):.1f}%")
            ax.set_xlabel("收益率 (%)")
            ax.set_ylabel("笔数")
            ax.legend(fontsize=9)
            ax.set_xlim(-4, 4)

        plt.tight_layout()
        out_path = "regime_returns_dist.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n图表已保存: {out_path}")
    except Exception as e:
        print(f"\n[图表跳过] {e}")


if __name__ == "__main__":
    main()
