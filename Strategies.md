# 策略实现文档

> 本文档描述框架中四大核心策略的详细实现，包括信号逻辑、参数含义、数学公式及执行模型。
> 数据粒度：日线 (`1d`)、小时线 (`1h`)、15分钟线 (`15m`)；回测区间默认 2021-01-01 ~ 最新。

---

## 目录

1. [回测引擎基础](#1-回测引擎基础)
2. [S/R Baseline 策略](#2-sr-baseline-策略)
3. [F&G + RSI 策略](#3-fg--rsi-策略)
4. [Reverse RSI 策略](#4-reverse-rsi-策略)
5. [S/R with Fast Velocity 策略](#5-sr-with-fast-velocity-策略)
6. [共用指标公式](#6-共用指标公式)
7. [止损 Walk-Forward 优化](#7-止损-walk-forward-优化)

---

## 1. 回测引擎基础

### 1.1 执行模型（T+1）

框架采用严格 T+1 成交，消除未来函数（lookahead bias）：

```
T 日收盘后  → 计算所有指标
            → 评估买入/卖出条件
            → 产生 pending_buy / pending_sell 信号

T+1 日开盘价 → 执行成交
```

### 1.2 收益与风险指标

| 指标 | 计算公式 |
|------|----------|
| 总收益 | `(final_capital - 10000) / 10000 × 100%` |
| Sharpe | `mean(daily_ret) / std(daily_ret) × sqrt(365 × bars_per_day)` |
| 最大回撤 | `min((pv - pv.cummax()) / pv.cummax()) × 100%` |
| 胜率 | `赢利交易对数 / 总交易对数 × 100%` |

`bars_per_day`：日线=1，小时线=24，15m=96

### 1.3 多空头寸计算

**多头**
```
入场: btc_held = capital / entry_price
持仓价值: value = btc_held × close
平仓: capital = btc_held × exit_price
```

**空头（1x，无借贷成本）**
```
入场: short_capital = capital
持仓价值: value = short_capital × (2 - close / entry_price)
平仓: capital = short_capital × (2 - exit_price / entry_price)

示例: 入场 50000，平仓 45000
  收益 = (50000 - 45000) / 50000 = +10%
  出金 = 10000 × (2 - 45000/50000) = 11000 ✓
```

---

## 2. S/R Baseline 策略

### 2.1 策略概述

基于 Volume Profile 生成动态支撑阻力（S/R）水平，结合价格动量（`norm_velocity`）和动量加速度（`velocity_delta`），在价格触及支撑/阻力且动量反转时入场。

**性能（1h，2021-2026）**

| 配置 | 总收益 | Sharpe | 最大回撤 | 交易次数 |
|------|--------|--------|----------|----------|
| Baseline_D（SL−2%+T48h） | +599.7% | 1.530 | −34.9% | 924 |
| **优化版（Trail3%+T24h）** | **+660.7%** | **1.649** | **−34.1%** | 936 |
| Buy & Hold（对比）| +135% | — | — | — |

### 2.2 参数配置

```python
# strategy_test_sr_momentum.py

BASELINE = {
    "velocity_bars":       8,      # 动量窗口：8根1h K线 = 8小时
    "proximity_pct":       0.005,  # 入场距离：价格距 S/R 水平 ≤ 0.5%
    "min_decel":           1.5,    # 反转强度阈值：velocity_delta ≥ 1.5
    "cluster_gap_pct":     0.01,   # S/R 聚类间距：1%
    "velocity_ceil_long":  0.5,    # 多头入场上限：norm_velocity ≤ +0.5
    "velocity_floor_short":-0.5,   # 空头入场下限：norm_velocity ≥ -0.5
}

BASELINE_SL = [
    StopLossCond(-0.02),           # 固定止损：-2%（原始版本）
    TimeBarStopCond(48),           # 时间止损：48根1h K线 = 2日历天
]

# 优化版止损（Walk-Forward 2021-2025.5 训练 / 2025.6+ 验证，见第7节）
BASELINE_SL_OPT = [
    TrailingStopCond(-0.03),       # 追踪止损：从峰值回撤 3%
    TimeBarStopCond(24),           # 时间止损：24根1h K线 = 1日历天
]
```

**15m 版本参数（等效时间窗口）**

```python
BASELINE_15M = {
    "velocity_bars":       32,     # 32根15m K线 = 8小时（同1h版本）
    "fast_velocity_bars":  8,      # 8根15m K线 = 2小时（快速动量）
    "proximity_pct":       0.005,
    "min_decel":           1.5,
    "cluster_gap_pct":     0.01,
    "velocity_ceil_long":  0.5,
    "velocity_floor_short":-0.5,
    "vp_windows":          [7, 14, 30],  # VP 窗口（天数）
}

BASELINE_15M_SL = [
    StopLossCond(-0.02),
    TimeBarStopCond(192, bars_per_day=96),  # 192根15m = 48小时 = 2日历天
]
```

### 2.3 S/R 水平生成

**Step 1 — Volume Profile 原始水平**

`load_data` 预计算多窗口 VP，每行包含：

```
vp_poc_7d   vp_vah_7d   vp_val_7d
vp_poc_14d  vp_vah_14d  vp_val_14d
vp_poc_30d  vp_vah_30d  vp_val_30d
```

- **POC**（Point of Control）：该窗口成交量最大价格
- **VAH**（Value Area High）：价值区上界（成交量70%集中区间上沿）
- **VAL**（Value Area Low）：价值区下界

**Step 2 — 聚类去重**

```python
def _cluster_levels(levels, close, min_gap_pct=0.01):
    # 按价格排序
    sorted_lvls = sorted(levels)
    min_gap = min_gap_pct × close      # 例: close=50000, gap=500

    clusters = [[sorted_lvls[0]]]
    for lvl in sorted_lvls[1:]:
        if lvl - mean(last_cluster) < min_gap:
            last_cluster.append(lvl)   # 合并到上一簇
        else:
            clusters.append([lvl])     # 开新簇

    return [mean(c) for c in clusters]  # 每簇取均值
```

**Step 3 — 计算距离指标**

```
nearest_resistance = min(clustered_levels > close)
nearest_support    = max(clustered_levels < close)

dist_to_resistance = (nearest_resistance - close) / close
dist_to_support    = (close - nearest_support) / close
```

### 2.4 入场条件

#### 多头入场（Near Support → Reversal Long）

```
条件1: dist_to_support ≤ proximity_pct          (≤ 0.5%，价格紧贴支撑)
条件2: velocity_delta ≥ min_decel               (≥ 1.5，下跌动量减速)
条件3: norm_velocity ≤ velocity_ceil_long        (≤ +0.5，未过度反弹)
```

三条件同时成立（AND）→ T+1 开盘买入

**直觉解释**：价格跌到支撑位附近，下跌动量已经显著减弱（`velocity_delta` 由负转正或负值变小），但尚未大幅反弹（`norm_velocity` 仍然偏低）— 典型的超卖反转形态。

#### 空头入场（Near Resistance → Reversal Short）

```
条件1: dist_to_resistance ≤ proximity_pct        (≤ 0.5%，价格紧贴阻力)
条件2: velocity_delta ≤ -min_decel              (≤ -1.5，上涨动量减速)
条件3: norm_velocity ≥ velocity_floor_short      (≥ -0.5，未过度回撤)
```

三条件同时成立（AND）→ T+1 开盘做空

### 2.5 出场条件

| 出场类型 | 条件 |
|----------|------|
| 多头止盈 | `dist_to_resistance ≤ 0.2%`（触及阻力）|
| 空头止盈 | `dist_to_support ≤ 0.2%`（触及支撑）|
| 固定止损 | 持仓亏损 ≥ 2% |
| 时间止损 | 持仓超过 48 根1h K线（2天）且未盈利 |

### 2.6 Walk-Forward 优化（15m）

训练集：2022-01-01 ~ 2025-05-31 | 验证集：2025-06-01 ~ 最新

**参数网格（共 72 个 Dual 组合）**

```
velocity_bars:       [16, 32, 48]        → 4h / 8h / 12h
fast_velocity_bars:  [2, 8]              → 30m / 2h
proximity_pct:       [0.003, 0.007]      → 紧/松接近度
min_decel:           [0.5, 1.5, 3.0]    → 弱/中/强反转阈值
velocity_ceil_long:  [0.0, 0.5]         → 等待回调 / 允许小幅反弹
```

**最优配置（验证集 Sharpe 排序）**

| vb | fvb | px | dc | 验证收益 | 验证 Sharpe |
|----|-----|----|----|---------|------------|
| 16 | 2 | 0.7% | 0.5 | +14.5% | 0.847 |
| 48 | 2 | 0.7% | 0.5 | +13.4% | 0.830 |
| 48 | 2 | 0.7% | 1.5 | +10.0% | 0.701 |

---

## 3. F&G + RSI 策略

### 3.1 策略概述

利用「极度恐慌」情绪（Fear & Greed Index < 10）与技术面超卖（RSI < 35）双重确认抄底，在「极度贪婪」（F&G > 92）时离场。纯日线策略，交易频率极低（历史约 4 笔），但每笔收益巨大。

**性能（日线，2018-2026）**

| 指标 | 数值 |
|------|------|
| 总收益 | +822% |
| Sharpe | 0.82 |
| 交易次数 | ~4 笔 |
| 历史捕获 | 2019底、2020 COVID、2022 FTX后 |

### 3.2 参数配置

```python
# conditions.py / strategy_test.py

FG_BUY_THRESH   = 10    # 极度恐慌阈值（约 3% 的交易日触发）
RSI_BUY_THRESH  = 35    # RSI 超卖阈值
FG_SELL_THRESH  = 92    # 极度贪婪阈值
RSI_SELL_THRESH = 65    # RSI 超买阈值
```

### 3.3 入场条件

```
买入触发（AND）:
  1. fear_greed < 10    （当日 F&G 指数处于极度恐慌区）
  2. rsi_14    < 35     （14日 RSI 处于超卖区）
```

**F&G 指数说明**

| 值 | 含义 | 频率 |
|----|------|------|
| 0-10 | 极度恐慌 | ~3% 的日期 |
| 10-25 | 恐慌 | ~12% |
| 25-45 | 偏空 | ~20% |
| 55-75 | 偏多 | ~20% |
| 75-90 | 贪婪 | ~12% |
| 90-100 | 极度贪婪 | ~3% |

**RSI 说明**

```
RSI_14[i] = 100 - 100 / (1 + RS)
RS = avg_gain_14 / avg_loss_14

RSI < 35 → 过去14天以下跌为主，且幅度显著
RSI > 65 → 过去14天以上涨为主，且幅度显著
```

### 3.4 出场条件

```
卖出触发（AND）:
  1. fear_greed > 92   （极度贪婪）
  2. rsi_14    > 65    （RSI 超买区）
```

双重确认减少虚假出场；两条件都满足才出场，避免单一指标的噪音。

### 3.5 扩展变体

```python
# F&G 单独版
FG_ONLY:
  买入: fear_greed < 10
  卖出: fear_greed > 85

# F&G + 资金费率增强版
FG_RSI_FUNDING:
  买入: FG < 10 AND RSI < 35 AND funding_sma7 < -0.0001
  卖出: FG > 92 AND RSI > 65

# 参数扫描范围
FG 阈值:    [5, 10, 15, 20, 25]
RSI 阈值:   [25, 30, 35, 40]
```

---

## 4. Reverse RSI 策略

### 4.1 策略概述

**反向使用 RSI**：RSI > 70（超买）作为**买入信号**，RSI < 30（超卖）作为**卖出信号**。核心思想是动量跟随（Momentum Following）而非均值回归：RSI 高说明当前处于强势上升趋势中，是入场时机；RSI 跌回低位说明趋势结束，是离场时机。

**性能（日线，2018-2026）**

| 指标 | 数值 |
|------|------|
| 总收益 | +1752% |
| Sharpe | 1.04 |
| 交易次数 | ~32 笔 |
| 平均持仓 | ~60-90 天 |

### 4.2 参数配置

```python
RSI_BUY_THRESH  = 70    # 买入：RSI 突破超买区（动量强劲）
RSI_SELL_THRESH = 30    # 卖出：RSI 跌入超卖区（趋势结束）
RSI_PERIOD      = 14    # RSI 计算周期（天）
```

### 4.3 入场条件

```
买入触发:
  rsi_14 > 70    （RSI 进入超买区，上升动量充足）
```

**逻辑说明**：传统用法认为 RSI > 70 是卖出信号（超买回调）。Reverse RSI 反其道而行，认为 RSI 进入超买区说明当前处于牛市加速阶段，此时入场可以跟随主升浪。BTC 的历史数据显示，RSI > 70 后往往有 30-100% 的额外涨幅。

### 4.4 出场条件

```
卖出触发:
  rsi_14 < 30    （RSI 跌入超卖区，上升动量耗尽）
```

**逻辑说明**：RSI < 30 意味着价格在近14天内持续下跌，上涨趋势已经结束，出场锁定收益。

### 4.5 止损增强版本

```python
# 版本 1：固定止损
RevRSI + SL10%:
  close_conditions = [StopLossCond(-0.10)]  # -10% 止损

# 版本 2：追踪止损
RevRSI + Trail15%:
  close_conditions = [TrailingStopCond(-0.15)]  # 从峰值回撤 -15%

RevRSI + Trail20%:
  close_conditions = [TrailingStopCond(-0.20)]

# 版本 3：追踪止损 + 止盈
RevRSI + Trail20 + TP200%:
  close_conditions = [
      TrailingStopCond(-0.20),
      TakeProfitCond(2.0),    # 达到 +200% 止盈
  ]

# 版本 4：时间止损
RevRSI + Time180d:
  close_conditions = [TimeExitCond(max_days=180, min_profit=0.10)]
  # 持仓超过 180 天且盈利 > 10% 才出场
```

### 4.6 信号增强变体

```python
# 增强 1：F&G 卖出辅助
RevRSI + FG85:
  sell_cond = RSIBuyCond(30) OR FearGreedSellCond(85)
  # RSI < 30 或 F&G > 85，任一触发卖出

# 增强 2：趋势过滤（只在上升趋势中入场）
RevRSI + SMA200:
  buy_cond = RSISellCond(70) AND TrendFilterCond(200)
  # 还需要 close > SMA200

# 增强 3：资金费率过热过滤
RevRSI + FundFilter:
  buy_cond  = RSISellCond(70) AND FundingNotOverheatedCond(0.0003)
  sell_cond = RSIBuyCond(30)  OR  FundingSellCond(0.0003)
  # 资金费率 > 0.03%/8h 时不入场（多头过度杠杆）

# 增强 4：RSI 动量方向确认
RevRSI + Rising:
  buy_cond = RSISellCond(70) AND RSIRisingCond(min_delta=0, lookback=3)
  # RSI > 70 且过去3天 RSI 在上升
```

### 4.7 参数扫描范围

```
RSI 周期:   [7, 10, 14, 21]
买入阈值:   [65, 70, 75]         # 动量入场早晚
卖出阈值:   [25, 30, 35]         # 出场宽严
总组合:     4 × 3 × 3 = 36 个策略
```

---

## 5. S/R with Fast Velocity 策略

### 5.1 策略概述

在 S/R Baseline 基础上，增加「快速动量」（`fast_norm_velocity` / `fast_velocity_delta`）作为第二层过滤器。快速动量基于更短时间窗口（1h 版本：2根K线 = 2小时；15m 版本：8根K线 = 2小时），用于确认短周期动量转向，减少基线策略在趋势加速阶段的误入场。

### 5.2 参数配置

```python
# strategy_test_sr_momentum.py — run_fast_grid()

BASE_PARAMS = {
    # 继承 Baseline 全部参数
    "velocity_bars":      8,
    "proximity_pct":      0.005,
    "min_decel":          1.5,
    "cluster_gap_pct":    0.01,
    "velocity_ceil_long": 0.5,
}

FAST_FILTER = {
    "fast_velocity_bars":  2,    # 快速窗口：2根1h K线 = 2小时
    # 多头快速过滤:
    "fast_velocity_ceil":  0.5,  # fast_norm_velocity ≤ +0.5（仍处于下跌或微反弹）
    # 空头快速过滤:
    "fast_velocity_floor": -1.0, # fast_norm_velocity ≥ -1.0（仍处于上涨或微回调）
    # 快速加速度:
    "fast_min_decel":      0.5,  # fast_velocity_delta 的反转强度（可选）
}
```

**15m 版本快速参数**

```python
"fast_velocity_bars": 8     # 8根15m = 2小时（与1h版保持等效时间窗口）
```

### 5.3 入场条件

#### 多头入场（Fast Enhanced Long）

```
慢速层（Baseline 同）:
  dist_to_support    ≤ 0.5%
  velocity_delta     ≥ 1.5
  norm_velocity      ≤ +0.5

快速层（新增）:
  fast_norm_velocity ≤ fast_velocity_ceil        （短周期仍未大涨）
  fast_velocity_delta ≥ fast_min_decel           （短周期减速已启动，可选）
```

全部条件 AND → T+1 开盘买入

**快速层含义**：即使慢速动量（8h）已经显示反转迹象，若短期2小时动量仍然强烈下跌（`fast_norm_velocity` 极负），可能尚未到最佳入场点；若2小时动量减速（`fast_velocity_delta` 转正），则信号更可靠。

#### 空头入场（Fast Enhanced Short）

```
慢速层（Baseline 同）:
  dist_to_resistance  ≤ 0.5%
  velocity_delta      ≤ -1.5
  norm_velocity       ≥ -0.5

快速层（新增）:
  fast_norm_velocity  ≥ fast_velocity_floor      （短周期仍未大跌）
  fast_velocity_delta ≤ -fast_min_decel          （短周期加速下行，可选）
```

### 5.4 快速动量指标公式

```
fast_norm_velocity[i] =
    (close[i] - close[i - fast_velocity_bars]) / close[i - fast_velocity_bars]
    / (atr_14[i] / close[i])

fast_velocity_delta[i] =
    fast_norm_velocity[i] - fast_norm_velocity[i - fast_velocity_bars]
```

与慢速版本（`norm_velocity` / `velocity_delta`）相同的公式，区别仅在 `fast_velocity_bars`（2 vs 8）更小，对价格变化更敏感。

### 5.5 快速过滤器网格

| 参数 | 候选值 |
|------|--------|
| `fast_velocity_ceil` | `None, -0.5, 0, 0.5, 1.0` |
| `fast_velocity_floor` | `None, -2.0, -1.0, -0.5` |
| `fast_min_decel` | `None, 0.3, 0.5, 1.0` |

`None` 表示不启用该过滤器。总计 5×4×4 = 80 个配置（含 Baseline 无过滤基线）。

### 5.6 典型有效配置

| 名称 | fast_ceil | fast_floor | fast_dc | 适用场景 |
|------|-----------|------------|---------|----------|
| NoFast（Baseline）| — | — | — | 基线对照 |
| FastCeil0.5 | 0.5 | — | — | 避免在反弹已发生后入多 |
| FastFloor-1.0 | — | -1.0 | — | 避免在回调已发生后入空 |
| FastFull | 0.5 | -1.0 | 0.5 | 慢+快双重反转确认 |

### 5.7 与 Baseline 的对比

| 维度 | S/R Baseline | S/R with Fast |
|------|-------------|---------------|
| 信号时间窗口 | 8h | 8h（慢）+ 2h（快）|
| 入场条件数 | 3 | 4~5 |
| 交易频率 | 标准 | 略低（过滤更严）|
| 目标 | 支撑/阻力反转 | 支撑/阻力反转 + 短期确认 |
| 优势 | 信号充足 | 减少趋势中途入场的假信号 |
| 风险 | 趋势行情中误入 | 过滤过严，可能错过早期入场 |

---

## 6. 共用指标公式

### 6.1 norm_velocity（ATR 归一化动量）

```
price_velocity[i] = (close[i] - close[i-N]) / close[i-N]
    N = velocity_bars (慢速窗口，默认 8)

atr_14[i] = SMA( max(H-L, |H-C[-1]|, |L-C[-1]|), 14 )

norm_velocity[i] = price_velocity[i] / (atr_14[i] / close[i])
```

**含义**：N 根K线内的价格变动相对于当前日波幅的倍数。

| 值 | 解释 |
|----|------|
| > +2.0 | 快速上涨（> 2倍日波幅）|
| 0 ~ +0.5 | 缓慢上涨或横盘 |
| 0 | 价格不变 |
| -0.5 ~ 0 | 缓慢下跌 |
| < -2.0 | 快速下跌（跌幅 > 2倍日波幅）|

### 6.2 velocity_delta（动量加速度）

```
velocity_delta[i] = norm_velocity[i] - norm_velocity[i-N]
    N = velocity_bars
```

**含义**：速度的变化量（二阶导数），用于判断动量是否在减弱（反转信号）。

| 值 | 解释 |
|----|------|
| > +1.5 | 下跌动量快速减弱 → 多头反转信号 |
| 0 ~ +1.5 | 下跌轻微减速 |
| 0 | 动量不变 |
| -1.5 ~ 0 | 上涨轻微减速 |
| < -1.5 | 上涨动量快速减弱 → 空头反转信号 |

### 6.3 VP（Volume Profile）窗口参数

| 参数 | 说明 |
|------|------|
| `vp_poc_Xd` | X天窗口内的成交量峰值价格（Point of Control）|
| `vp_vah_Xd` | X天窗口价值区上界（Value Area High）|
| `vp_val_Xd` | X天窗口价值区下界（Value Area Low）|
| `vp_windows` | 使用的窗口列表，默认 `[7, 14, 30]` |

多窗口的 POC/VAH/VAL 经过聚类（`cluster_gap_pct=1%`）后形成最终 S/R 水平。

### 6.4 止损条件说明

| 条件类 | 参数 | 触发逻辑 |
|--------|------|----------|
| `StopLossCond(t)` | `t=-0.02`（-2%）| `(close - entry) / entry < t` |
| `TimeBarStopCond(n, bpd)` | `n=48, bpd=24` | 持仓 bar 数 > n 且 PnL ≤ 0 |
| `TrailingStopCond(t)` | `t=-0.15`（-15%）| `(close - peak) / peak < t` |
| `SRStopLossLong(buf, max)` | `buf=0.3%, max=3%` | `close < support × (1-buf)`，且亏损不超 `max` |
| `ATRStopLossCond(m)` | `m=2.0` | `close < entry - m × ATR` |
| `TakeProfitCond(t)` | `t=2.0`（+200%）| `(close - entry) / entry > t` |

---

---

## 7. 止损 Walk-Forward 优化

### 7.1 优化框架

针对 S/R Baseline（1h Dual 多空）的止损机制做 Walk-Forward 网格搜索，消除过拟合。

```
训练集：2021-01-01 ~ 2025-05-31（约 4.5 年）
验证集：2025-06-01 ~ 2026-02-24（约 9 个月）
入场参数固定（同 Baseline），仅搜索止损配置
```

**网格维度（97 个配置）**

| Phase | 类型 | 参数 |
|-------|------|------|
| A | 固定止损 × 时间止损 | SL 1/1.5/2/3/4/5% × noT/24h/48h/72h/96h |
| B | S/R 动态止损 × 时间止损 | buffer 0.2/0.3/0.5% × max 2/3/4% × noT/24h… |
| C | ATR 止损 × 时间止损 | ATR 1.5x/2x/3x × noT/24h… |
| D | 追踪止损 × T24h/noT | Trail 3/5/8/10% |
| E | 止盈+SL4% × T24h/noT | TP 1.5/2/3/5% + SL4% |
| F | 棘轮止损 × T24h/noT | hard_sl 1~3% × activate 0.5~1% × trail 2.5~4% |

训练集按 Sharpe 取 Top-20，进入验证集评估，最终以验证集 Sharpe 选优。

---

### 7.2 最优配置：Trail3%+T24h

**止损逻辑**

```python
close_conditions = [
    TrailingStopCond(-0.03),   # 从峰值回撤超过 3% 时止损
    TimeBarStopCond(24),       # 持仓超过 24h 且未盈利时退出
]
```

**追踪止损机制**

```
从入场开始，引擎记录持仓期间的 peak_price（最高收盘价）

触发条件：(close - peak_price) / peak_price < -0.03

示例（入场 100）：
  峰值涨至 105 → 追踪止损线 = 105 × 0.97 = 101.85
  若随后跌破 101.85 → 止损退出（锁定约 +1.85%）
  若继续涨至 110 → 追踪线 = 106.7，进一步锁住利润
```

---

### 7.3 全周期表现（2021-02-24 至今）

| 指标 | Trail3%+T24h | Baseline（SL−2%+T48h）| 改善 |
|------|-------------|----------------------|------|
| 总收益 | **+660.7%** | +599.7% | +61pp |
| Sharpe | **1.649** | 1.530 | +0.119 |
| 最大回撤 | **−34.1%** | −34.9% | +0.8pp |
| 交易次数 | 936 | 924 | +12 |
| SL 触发率（训练） | 10.3% | 14.1% | −3.8pp |

**年度拆分**

| 年份 | 收益 | Sharpe | 最大回撤 | 交易次数 |
|------|------|--------|----------|----------|
| 2021 | +84.4% | 2.014 | −30.3% | 159 |
| 2022 | +18.3% | 0.771 | −24.1% | 158 |
| 2023 | +39.4% | 1.654 | −11.0% | 186 |
| 2024 | +59.7% | 2.029 | −14.8% | 190 |
| 2025 | +50.7% | 1.949 | −14.9% | 221 |
| 2026 YTD | +3.9% | 1.084 | −12.4% | 22 |

**验证集（2025-06 ~ 2026-02，BH=−37%）**

| 指标 | Trail3%+T24h | Baseline |
|------|-------------|----------|
| 收益 | +33.8% | +26.5% |
| Sharpe | **1.921** | ~1.78 |
| 最大回撤 | −14.9% | −14.0% |
| SL 触发率 | 10.7% | ~14% |

---

### 7.4 各止损方式综合对比

| 配置 | 全周期收益 | 全周期 Sharpe | 验证 Sharpe | 验证 MaxDD | 训练 SL% |
|------|-----------|-------------|------------|-----------|---------|
| **Trail3%+T24h ★** | +660.7% | 1.649 | **1.921** | −14.9% | 10.3% |
| Trail3%+noT | +557.4%\* | 1.642 | 1.917 | **−14.0%** | 4.8% |
| SL4%+noT | +671.6%\* | 1.658 | 1.853 | −16.8% | 4.4% |
| SL4%+T24h | +761.8%\* | 1.599 | 2.097† | −16.0% | 11.2% |
| Ratchet(sl3%_a1%_tr4%)+T24h | — | 1.680 | 1.599 | **−13.2%** | 6.2% |
| Baseline（SL−2%+T48h） | +599.7% | 1.530 | ~1.78 | −14.0% | 14.1% |
| SL1%+T48h | +526.1%\* | 1.735 | 1.134 | −12.7% | 23.9% |

\* 仅训练期/全期数字，验证期排除在外
† SL4%+T24h 验证 Sharpe 在不同时段数据下有小幅波动（约 2.0~2.1）

**关键规律**

- **SL=1%（过窄）**：训练 Sharpe 最高（1.735），验证最差（1.134）→ 频繁止损截断均值回归利润
- **追踪止损** 在趋势年（2021/2024/2025）显著优于固定止损；震荡年（2022/2023）略弱
- **棘轮止损** 验证期 MaxDD 最低（−13.2%），适合以控制每笔亏损幅度为首要目标的场景
- **止盈（TP）完全无效**：策略每笔均值收益约 +0.2%，提前止盈严重截断长尾盈利

---

### 7.5 交易损益分布（Baseline 参考）

基于全历史 1033 笔交易（多空合计）：

```
胜率：35.8%   平均盈利：+2.04%   平均亏损：−0.79%

亏损分布：
  <−4%       ██                      6笔  ( 0.9%)
  −4~−3%     █                      16笔  ( 2.4%)   ← 棘轮止损主要拦截区
  −3~−2%     ████████               98笔  (14.8%)   ← 主要损失来源
  −2~−1.5%   █                      14笔  ( 2.1%)
  −1.5~−1%   █                      19笔  ( 2.9%)
  −1~−0.5%   ██████                 95笔  (14.3%)
  −0.5~0%    ██████████████████    415笔  (62.6%)   ← 信号驱动退出，止损无法干预

盈利交易持仓越长收益越高：
  <4h    均盈 +0.81%
  4−8h   均盈 +1.38%
  8−12h  均盈 +1.73%
  12−24h 均盈 +1.78%
  24−48h 均盈 +2.54%
  >48h   均盈 +5.46%   ← 不能过早截断赢家
```

---

### 7.6 止损条件速查（含新增类型）

| 条件类 | 参数示例 | 触发逻辑 | 适用场景 |
|--------|---------|----------|----------|
| `StopLossCond(t)` | `t=−0.04`（−4%）| `pnl < t` | 固定保护上限 |
| `TimeBarStopCond(n)` | `n=24`（24h）| 持仓 >n bar 且未盈利 | 清除长期卡住仓位 |
| `TrailingStopCond(t)` | `t=−0.03`（−3%）| `(close−peak)/peak < t` | 锁住已有利润 |
| `SRStopLossLong(buf,max)` | `buf=0.3%, max=3%` | `close < support×(1−buf)` | 跌破支撑位 |
| `ATRStopLossCond(m)` | `m=2.0` | `close < entry − m×ATR` | 波动率自适应 |
| `TakeProfitCond(t)` | `t=0.03`（+3%）| `pnl > t` | 强制止盈（本策略无效）|
| `RatchetStopCond(sl,act,tr)` | `sl=−3%, act=+1%, tr=4%` | Phase1: 硬止损；Phase2（peak>act后）: 从峰值追踪 | 减少每笔亏损幅度 |

**注**：`TrailingStopCond` 和 `RatchetStopCond` 对空头通过 `short_pnl_peak`（反射峰值）完全对称支持。

---

*文件路径：`Strategies.md` | 代码入口：`strategy_test_sr_momentum.py`, `strategy_test.py`, `conditions.py`, `sr_signals.py`*
