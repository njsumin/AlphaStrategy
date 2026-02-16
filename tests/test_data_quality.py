"""
数据质量检查测试 (test_data_quality.py)
========================================
检查项：
  1. 缺失值检测
  2. 重复时间戳检测
  3. 价格异常值检测 (3σ)
  4. 日期连续性检测（含周末补全合理性）
  5. 数值范围合理性
  6. 衍生品数据特殊问题（零值段、尾部异常）
  7. 生成数据质量报告
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# ============================================================================
# 数据加载（本地 CSV，不依赖网络）
# ============================================================================

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "binance", "binance_derivatives_daily.csv")


def load_derivatives_csv():
    """加载衍生品 CSV 并做基本类型转换"""
    assert os.path.exists(CSV_PATH), f"CSV 文件不存在: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df["open_interest_usd"] = pd.to_numeric(df["open_interest_usd"], errors="coerce")
    return df


# ============================================================================
# 测试用例
# ============================================================================

def test_csv_exists_and_not_empty():
    """CSV 文件存在且非空"""
    df = load_derivatives_csv()
    assert len(df) > 0, "CSV 为空"
    print(f"  [PASS] CSV 共 {len(df)} 行")


def test_required_columns():
    """必需列存在"""
    df = load_derivatives_csv()
    required = ["date", "funding_rate", "open_interest_usd"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"缺少必需列: {missing}"
    print(f"  [PASS] 必需列齐全: {required}")


def test_no_duplicate_dates():
    """无重复时间戳"""
    df = load_derivatives_csv()
    dup_count = df["date"].duplicated().sum()
    if dup_count > 0:
        dups = df[df["date"].duplicated(keep=False)]["date"].unique()
        print(f"  [FAIL] 重复日期 {dup_count} 个: {dups[:5]}")
    assert dup_count == 0, f"发现 {dup_count} 个重复时间戳"
    print(f"  [PASS] 无重复时间戳")


def test_missing_values():
    """检测缺失值"""
    df = load_derivatives_csv()
    report = {}
    for col in ["funding_rate", "open_interest_usd"]:
        nan_count = df[col].isna().sum()
        report[col] = nan_count

    print(f"  缺失值统计: {report}")
    for col, cnt in report.items():
        if cnt > 0:
            pct = cnt / len(df) * 100
            print(f"  [WARN] {col} 缺失 {cnt} 行 ({pct:.1f}%)")
        else:
            print(f"  [PASS] {col} 无缺失")


def test_date_continuity():
    """日期连续性检测：找出缺失的日期（允许合理间隔）"""
    df = load_derivatives_csv()
    df_sorted = df.sort_values("date").reset_index(drop=True)
    date_range = pd.date_range(df_sorted["date"].min(), df_sorted["date"].max(), freq="D")
    existing_dates = set(df_sorted["date"].dt.normalize())
    missing_dates = sorted(set(date_range) - existing_dates)

    if missing_dates:
        print(f"  [WARN] 缺失 {len(missing_dates)} 个日期:")
        for d in missing_dates[:10]:
            print(f"         {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})")
        if len(missing_dates) > 10:
            print(f"         ... 共 {len(missing_dates)} 个")
    else:
        print(f"  [PASS] 日期完全连续 ({df_sorted['date'].min().date()} ~ {df_sorted['date'].max().date()})")

    # 不做断言失败，仅报告（加密货币数据允许少量缺失）
    return missing_dates


def test_funding_rate_range():
    """Funding rate 范围合理性检测"""
    df = load_derivatives_csv()
    fr = df["funding_rate"].dropna()

    # 正常范围: 一般在 -0.01 ~ 0.01 之间
    extreme_low = (fr < -0.01).sum()
    extreme_high = (fr > 0.01).sum()

    print(f"  Funding Rate 统计:")
    print(f"    范围: [{fr.min():.6f}, {fr.max():.6f}]")
    print(f"    均值: {fr.mean():.6f}, 标准差: {fr.std():.6f}")
    print(f"    极端值 (<-0.01): {extreme_low}, (>0.01): {extreme_high}")

    # 3σ 异常值检测
    mean, std = fr.mean(), fr.std()
    outliers = df[(fr < mean - 3 * std) | (fr > mean + 3 * std)]
    if len(outliers) > 0:
        print(f"  [WARN] Funding Rate 3σ 异常值: {len(outliers)} 个")
        for _, row in outliers.head(5).iterrows():
            print(f"         {row['date'].strftime('%Y-%m-%d')}: {row['funding_rate']:.6f}")
    else:
        print(f"  [PASS] Funding Rate 无 3σ 异常值")

    # 检查尾部全零问题
    tail = df.tail(10)
    zero_tail = (tail["funding_rate"] == 0).sum()
    if zero_tail >= 3:
        zero_start = df[df["funding_rate"] == 0].tail(zero_tail)["date"].iloc[0]
        print(f"  [WARN] Funding Rate 尾部连续零值: 从 {zero_start.strftime('%Y-%m-%d')} 开始有 {zero_tail} 个零值，可能数据源中断")

    assert fr.min() > -0.1, f"Funding rate 异常低: {fr.min()}"
    assert fr.max() < 0.1, f"Funding rate 异常高: {fr.max()}"


def test_open_interest_range():
    """Open Interest 范围合理性检测"""
    df = load_derivatives_csv()
    oi = df["open_interest_usd"]

    # 检查零值段
    zero_count = (oi == 0).sum()
    nonzero = oi[oi > 0]

    print(f"  Open Interest 统计:")
    print(f"    总行数: {len(oi)}, 零值: {zero_count}, 有效值: {len(nonzero)}")

    if len(nonzero) > 0:
        print(f"    有效范围: [{nonzero.min():,.0f}, {nonzero.max():,.0f}]")
        print(f"    有效均值: {nonzero.mean():,.0f}")

        # 3σ 异常值检测（仅对非零值）
        mean, std = nonzero.mean(), nonzero.std()
        outlier_mask = (nonzero < mean - 3 * std) | (nonzero > mean + 3 * std)
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            print(f"  [WARN] OI 3σ 异常值: {outlier_count} 个")
        else:
            print(f"  [PASS] OI 无 3σ 异常值")

    # 检查零值到非零值的转换点
    if zero_count > 0 and len(nonzero) > 0:
        first_nonzero_idx = (oi > 0).idxmax()
        first_nonzero_date = df.loc[first_nonzero_idx, "date"]
        print(f"  [INFO] OI 零值段: 前 {first_nonzero_idx} 行为零，首个非零日期: {first_nonzero_date.strftime('%Y-%m-%d')}")


def test_funding_rate_sudden_change():
    """Funding rate 突变检测（日间变化过大）"""
    df = load_derivatives_csv()
    fr = df["funding_rate"].dropna()
    fr_diff = fr.diff().abs()

    # 找出日间变化超过 3σ 的点
    mean_diff = fr_diff.mean()
    std_diff = fr_diff.std()
    threshold = mean_diff + 3 * std_diff

    spikes = df[fr_diff > threshold].copy()
    if len(spikes) > 0:
        print(f"  [WARN] Funding Rate 突变点 (>3σ={threshold:.6f}): {len(spikes)} 个")
        for _, row in spikes.head(5).iterrows():
            idx = row.name
            prev_val = df.loc[idx - 1, "funding_rate"] if idx > 0 else None
            curr_val = row["funding_rate"]
            print(f"         {row['date'].strftime('%Y-%m-%d')}: {prev_val:.6f} -> {curr_val:.6f}")
    else:
        print(f"  [PASS] Funding Rate 无异常突变")


def test_oi_sudden_change():
    """Open Interest 突变检测（日间百分比变化过大）"""
    df = load_derivatives_csv()
    oi = df["open_interest_usd"].copy()

    # 仅检查非零段
    valid_mask = oi > 0
    oi_valid = oi[valid_mask]
    if len(oi_valid) < 10:
        print(f"  [SKIP] OI 有效数据不足")
        return

    pct_change = oi_valid.pct_change().abs()
    threshold = 0.5  # 单日变化超过 50%

    spikes = df.loc[pct_change[pct_change > threshold].index]
    if len(spikes) > 0:
        print(f"  [WARN] OI 单日变化 >50%: {len(spikes)} 个")
        for _, row in spikes.head(5).iterrows():
            idx = row.name
            prev_val = df.loc[idx - 1, "open_interest_usd"] if idx > 0 else 0
            curr_val = row["open_interest_usd"]
            change = (curr_val - prev_val) / prev_val * 100 if prev_val > 0 else float('inf')
            print(f"         {row['date'].strftime('%Y-%m-%d')}: {prev_val:,.0f} -> {curr_val:,.0f} ({change:+.1f}%)")
    else:
        print(f"  [PASS] OI 无异常突变 (阈值: ±50%)")


def test_date_sorted():
    """日期按升序排列"""
    df = load_derivatives_csv()
    is_sorted = df["date"].is_monotonic_increasing
    assert is_sorted, "日期未按升序排列"
    print(f"  [PASS] 日期按升序排列")


def test_data_freshness():
    """数据新鲜度检查"""
    df = load_derivatives_csv()
    latest = df["date"].max()
    today = pd.Timestamp.now().normalize()
    days_old = (today - latest).days

    print(f"  最新数据日期: {latest.strftime('%Y-%m-%d')}")
    print(f"  距今: {days_old} 天")

    if days_old > 7:
        print(f"  [WARN] 数据超过 7 天未更新")
    else:
        print(f"  [PASS] 数据较新")


# ============================================================================
# 综合报告
# ============================================================================

def generate_quality_report():
    """生成完整的数据质量报告"""
    print("=" * 70)
    print("           数据质量检查报告")
    print("=" * 70)

    df = load_derivatives_csv()
    print(f"\n文件: {CSV_PATH}")
    print(f"行数: {len(df)}")
    print(f"列数: {list(df.columns)}")
    print(f"日期范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")

    tests = [
        ("1. 文件与列检查", test_csv_exists_and_not_empty),
        ("1b. 必需列", test_required_columns),
        ("2. 重复时间戳", test_no_duplicate_dates),
        ("3. 缺失值", test_missing_values),
        ("4. 日期连续性", test_date_continuity),
        ("5. 日期排序", test_date_sorted),
        ("6. Funding Rate 范围", test_funding_rate_range),
        ("7. Open Interest 范围", test_open_interest_range),
        ("8. Funding Rate 突变", test_funding_rate_sudden_change),
        ("9. OI 突变", test_oi_sudden_change),
        ("10. 数据新鲜度", test_data_freshness),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"结果: {passed} 通过, {failed} 失败 (共 {len(tests)} 项)")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = generate_quality_report()
    sys.exit(0 if success else 1)
