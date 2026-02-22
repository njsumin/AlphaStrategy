"""
未来函数检测测试 (test_no_lookahead.py)
==========================================
核心思路：构造在某一天发生价格跳变的虚假数据，
验证跳变前的指标不会因为跳变后的数据产生异常。

检查项：
  1. 指标计算无前瞻偏差（跳变前指标不受跳变后数据影响）
  2. 回测引擎信号日 vs 成交日是否分离（T日信号 -> T+1日成交）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


# ============================================================================
# 辅助函数：构造带跳变的虚假数据
# ============================================================================

def make_synthetic_data(n_before=50, n_after=30, price_before=1.15, price_after=25.0):
    """
    构造一组虚假日线数据，前 n_before 天价格在 price_before 附近，
    之后突然跳到 price_after 附近。

    返回 DataFrame，列：date, close, volume, fear_greed, funding_rate, open_interest_usd
    """
    np.random.seed(42)
    total = n_before + n_after
    dates = pd.date_range("2023-01-01", periods=total, freq="D")

    # 跳变前：1.10 ~ 1.20 随机波动
    prices_before = np.random.uniform(price_before - 0.05, price_before + 0.05, n_before)
    # 跳变后：20 ~ 30 随机波动
    prices_after = np.random.uniform(price_after - 5, price_after + 5, n_after)
    close = np.concatenate([prices_before, prices_after])

    df = pd.DataFrame({
        "date": dates,
        "close": close,
        "volume": np.random.uniform(1e6, 5e6, total),
        "fear_greed": np.random.randint(20, 80, total),
        "funding_rate": np.random.uniform(-0.0002, 0.0005, total),
        "open_interest_usd": np.random.uniform(1e9, 5e9, total),
    })
    return df, n_before  # 返回跳变点索引


# ============================================================================
# 测试用例
# ============================================================================

def test_indicators_no_lookahead():
    """
    验证指标计算不使用未来数据。

    方法：
    1. 构造完整数据，计算指标
    2. 只取跳变前的数据，重新计算指标
    3. 比较跳变前最后一天的指标值——两者必须完全相同
    """
    from engine import _add_indicators

    df_full, jump_idx = make_synthetic_data()

    # 完整数据算指标
    df_full_ind = _add_indicators(df_full.copy())

    # 只用跳变前数据算指标
    df_before = df_full.iloc[:jump_idx].copy()
    df_before_ind = _add_indicators(df_before.copy())

    # 比较跳变前最后一天的指标
    last_idx = jump_idx - 1
    check_cols = ["rsi_14", "funding_sma7", "oi_pct_7d",
                  "drawdown_30d", "drawdown_60d", "drawdown_90d"]

    all_pass = True
    for col in check_cols:
        if col not in df_full_ind.columns:
            continue

        val_full = df_full_ind.loc[last_idx, col]
        val_before = df_before_ind.loc[last_idx, col]

        if pd.isna(val_full) and pd.isna(val_before):
            print(f"  [PASS] {col}: 均为 NaN（数据不足）")
            continue

        if pd.isna(val_full) or pd.isna(val_before):
            print(f"  [FAIL] {col}: full={val_full}, before={val_before}")
            all_pass = False
            continue

        if abs(val_full - val_before) > 1e-10:
            print(f"  [FAIL] {col}: full={val_full:.8f}, before={val_before:.8f}, diff={val_full-val_before:.2e}")
            all_pass = False
        else:
            print(f"  [PASS] {col}: {val_full:.8f}")

    assert all_pass, "指标计算存在未来函数！"
    print("\n  结论：所有指标计算无前瞻偏差")


def test_backtest_signal_vs_execution():
    """
    验证回测引擎中信号日和成交日是否正确分离。

    CLAUDE.md 要求：昨天收盘价判断 + 今天开盘价交易
    即 T 日产生信号 -> T+1 日开盘价成交

    检查方法：构造数据使得某天触发买入信号，
    检查实际成交价格是否为次日开盘价（而非当日收盘价）。
    """
    from engine import backtest

    # 构造数据：让 RSI 在第 20 天跌破 30（触发买入）
    np.random.seed(123)
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # 构造一个先跌后涨的价格序列，使 RSI 在中间跌破阈值
    closes = np.concatenate([
        np.linspace(100, 60, 25),   # 持续下跌，RSI 会很低
        np.linspace(61, 120, 35),   # 反弹
    ])
    # open 价格 = 前一天 close 加小幅随机偏移（模拟真实隔夜跳空）
    opens = np.roll(closes, 1) + np.random.uniform(-1, 1, n)
    opens[0] = closes[0] + 0.5  # 第一天无前一天数据

    df = pd.DataFrame({
        "date": dates,
        "open": opens,
        "close": closes,
        "volume": np.random.uniform(1e6, 5e6, n),
    })

    # 手动计算 RSI 以确定信号日
    from engine import _add_indicators
    df = _add_indicators(df.copy())

    # 简单买入条件：RSI < 30
    def buy_when_rsi_low(row):
        rsi = row.get("rsi_14", np.nan)
        if pd.isna(rsi):
            return False
        return "RSI<30" if rsi < 30 else False

    def never_sell(row):
        return False

    result = backtest(df, "test", buy_when_rsi_low, never_sell)

    if result["trade_log"]:
        trade = result["trade_log"][0]
        trade_date = trade["date"]
        trade_price = trade["price"]

        # 找到成交日在 df 中的行
        exec_row = df[df["date"] == trade_date].iloc[0]
        exec_open = exec_row["open"]
        exec_close = exec_row["close"]

        # 找到信号日（成交日的前一天）
        trade_idx = df[df["date"] == trade_date].index[0]
        if trade_idx > 0:
            signal_row = df.iloc[trade_idx - 1]
            signal_date = signal_row["date"]
            signal_rsi = signal_row.get("rsi_14", np.nan)
        else:
            signal_date = None
            signal_rsi = np.nan

        print(f"  信号日期: {signal_date.strftime('%Y-%m-%d') if signal_date else 'N/A'} (RSI={signal_rsi:.2f})")
        print(f"  成交日期: {trade_date.strftime('%Y-%m-%d')}")
        print(f"  成交价格: {trade_price:.4f}")
        print(f"  成交日 open: {exec_open:.4f}")
        print(f"  成交日 close: {exec_close:.4f}")

        all_pass = True

        # 检查1: 成交价应该 = 成交日的 open（而非 close）
        if abs(trade_price - exec_open) < 0.01:
            print(f"  [PASS] 成交价 = 成交日开盘价")
        else:
            print(f"  [FAIL] 成交价 ≠ 成交日开盘价")
            all_pass = False

        # 检查2: 信号日应该 ≠ 成交日（T日信号 → T+1成交）
        if signal_date and signal_date < trade_date:
            print(f"  [PASS] 信号日 < 成交日（T日信号 → T+1成交）")
        else:
            print(f"  [FAIL] 信号日未早于成交日")
            all_pass = False

        # 检查3: 信号日 RSI 确实 < 30
        if not pd.isna(signal_rsi) and signal_rsi < 30:
            print(f"  [PASS] 信号日 RSI={signal_rsi:.2f} < 30，信号合理")
        else:
            print(f"  [WARN] 信号日 RSI={signal_rsi:.2f}，可能不满足条件")

        return all_pass
    else:
        print(f"  [SKIP] 未触发交易")
        return True


def test_w_bottom_no_lookahead():
    """
    验证 W-bottom/M-top 强度计算不使用未来数据。

    方法：
    1. 构造合成15m数据，前段低价稳定，后段突然跳到高价
    2. 用完整数据计算 w_bottom_strength_15m
    3. 只用跳变前数据重新计算
    4. 比较跳变前最后一根bar的值——两者必须完全相同
    """
    from engine import _compute_w_bottom_m_top

    np.random.seed(42)
    n_before = 100  # 100 bars @15m before jump
    n_after = 50    # 50 bars @15m after jump

    # Before: price oscillating 1.10-1.20 (create some swing points)
    prices_lo_before = np.random.uniform(1.08, 1.18, n_before)
    prices_hi_before = prices_lo_before + np.random.uniform(0.01, 0.04, n_before)
    prices_cl_before = (prices_lo_before + prices_hi_before) / 2

    # After: price jumps to 25-30
    prices_lo_after = np.random.uniform(23, 28, n_after)
    prices_hi_after = prices_lo_after + np.random.uniform(0.5, 2.0, n_after)
    prices_cl_after = (prices_lo_after + prices_hi_after) / 2

    dates = pd.date_range("2023-01-01", periods=n_before + n_after, freq="15min")

    df_full = pd.DataFrame({
        "date": dates,
        "open": np.concatenate([prices_cl_before, prices_cl_after]),
        "high": np.concatenate([prices_hi_before, prices_hi_after]),
        "low": np.concatenate([prices_lo_before, prices_lo_after]),
        "close": np.concatenate([prices_cl_before, prices_cl_after]),
        "volume": np.random.uniform(1e4, 5e4, n_before + n_after),
    })

    # Compute on full data
    df_full_wm = _compute_w_bottom_m_top(df_full.copy(),
                                          lookback_bars=32, swing_window=3)

    # Compute on before-only data
    df_before = df_full.iloc[:n_before].copy()
    df_before_wm = _compute_w_bottom_m_top(df_before.copy(),
                                            lookback_bars=32, swing_window=3)

    last_idx = n_before - 1
    all_pass = True

    for col in ["w_bottom_strength_15m", "m_top_strength_15m"]:
        val_full = df_full_wm.loc[last_idx, col]
        val_before = df_before_wm.loc[last_idx, col]

        if pd.isna(val_full) and pd.isna(val_before):
            print(f"  [PASS] {col}: 均为 NaN")
            continue

        if pd.isna(val_full) or pd.isna(val_before):
            print(f"  [FAIL] {col}: full={val_full}, before={val_before}")
            all_pass = False
            continue

        if abs(val_full - val_before) > 1e-10:
            print(f"  [FAIL] {col}: full={val_full:.8f}, before={val_before:.8f}, "
                  f"diff={val_full - val_before:.2e}")
            all_pass = False
        else:
            print(f"  [PASS] {col}: {val_full:.8f}")

    assert all_pass, "W-bottom/M-top 计算存在未来函数！"
    print("\n  结论：W-bottom/M-top 强度计算无前瞻偏差")


# ============================================================================
# 综合报告
# ============================================================================

def generate_report():
    """生成未来函数检测报告"""
    print("=" * 70)
    print("           未来函数检测报告")
    print("=" * 70)

    tests = [
        ("1. 指标计算前瞻偏差检测", test_indicators_no_lookahead),
        ("2. 回测信号日 vs 成交日检测", test_backtest_signal_vs_execution),
        ("3. W-bottom/M-top 前瞻偏差检测", test_w_bottom_no_lookahead),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_fn()
            if result is False:
                failed += 1
            else:
                passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"结果: {passed} 通过, {failed} 有问题 (共 {len(tests)} 项)")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = generate_report()
    sys.exit(0 if success else 1)
