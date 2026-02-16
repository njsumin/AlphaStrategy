# CLAUDE.md

## 语言要求
中文交互，代码变量名英文。

## 项目概述
加密货币交易策略回测框架。

## 运行方式
```bash
python strategy_test.py          # 运行回测
python data_loader.py            # 增量更新 Binance 衍生品数据
python data_loader.py --full     # 全量重新下载
```

## 依赖
pip install pandas numpy matplotlib yfinance requests python-dateutil

## 架构
- data_loader.py — 数据下载与加载（BTC价格、F&G、Binance衍生品、CB Premium）
- conditions.py — 条件库（买入/卖出/风控条件 + 组合器）
- engine.py — 回测引擎（指标计算 + backtest + print_results + plot_results）
- strategy_test.py — 入口文件
- data/binance/binance_derivatives_daily.csv — Binance 衍生品本地缓存
- tests/test_data_quality.py — 数据质量检查（缺失值、重复、异常值、连续性）
- tests/test_no_lookahead.py — 未来函数检测（指标前瞻偏差 + 信号/成交分离）

## 重要约定
- 新策略参考 strategy_test.py 中的现有配置
- 尽可能保持现有结构，保持代码的可重用性
- 所有数据下载/网络请求逻辑集中在 data_loader.py，engine.py 不做网络调用

## 核心规则（违反即 bug）

### 禁止未来函数
信号计算和交易决策只能使用当前及历史数据。回测执行模型：T日收盘指标产生信号 → T+1日开盘价成交。
```python
# ❌ df['signal'] = df['close'].shift(-1)
# ❌ 当天收盘价判断 + 当天收盘价交易
# ✅ 昨天收盘价判断 + 今天开盘价交易（engine.py pending_buy/pending_sell 机制）
```
生成虚假的历史数据作为测试案例来验证是否用到未来函数,例如：
之前三十天的close价格是1.1-1.2之间，之后给出20-30之间的数据，在数据跳变前的最后一个日期的指标不应该有大异常

### 回测必须包含
关键指标（夏普比率、最大回撤、胜率）

## 开发工作流
1. **数据下载后** → 立即验证：缺失值、重复时间戳、价格异常值（3σ），生成数据质量报告，对于合理的缺失例如周末等自动补全数据
2. **指标生成后** → 立即验证：无未来函数、无 inf/nan、时间索引对齐、输出分布统计
3. **回测完成后** → 验证信号无未来函数、成交价合理性，报告含风险提示
4. 验证失败必须抛异常或生成警告，不得静默跳过
5. 每个模块完成后运行 tests/ 下对应测试
```bash
python tests/test_data_quality.py   # 数据质量
python tests/test_no_lookahead.py   # 未来函数检测
```

