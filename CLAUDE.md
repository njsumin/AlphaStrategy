# CLAUDE.md

## 语言要求
中文交互，代码变量名英文。

## 项目概述
加密货币交易策略回测框架，支持日线/小时线多时间粒度回测，集成 Volume Profile、RSI、资金费率等多类指标。

## 运行方式
```bash
python strategy_test.py              # 日线回测（主入口）
python strategy_test_hourly.py       # 小时线回测
python strategy_test_vp.py           # Volume Profile 策略回测
python strategy_test_vp_poc.py       # VP POC + Reverse RSI 回测
python strategy_test_rsi_poc.py      # RSI + VP POC 组合回测
python data_loader.py                # 增量更新 Binance 衍生品数据
python data_loader.py --full         # 全量重新下载
```

## 依赖
pip install pandas numpy matplotlib yfinance requests python-dateutil

## 架构

### 核心模块
- data_loader.py — 数据下载与加载（BTC价格、F&G、Binance衍生品、CB Premium、15m/1h K线）
- conditions.py — 条件库（买入/卖出/风控条件 + VP条件 + 组合器 + 预设组合）
- engine.py — 回测引擎（指标计算含VP、backtest支持多空、print_results + save_results_to_files + plot_results）

### 策略入口
- strategy_test.py — 日线策略（F&G+RSI、Reverse RSI、资金费率、Alpha Combo等）
- strategy_test_hourly.py — 小时线策略（同日线策略框架，RSI周期按小时缩放）
- strategy_test_vp.py — VP独立策略（VP-MR均值回归、VP-Trend趋势、VP+RevRSI组合）
- strategy_test_vp_poc.py — VP POC作为RevRSI入场时机过滤
- strategy_test_rsi_poc.py — 经典RSI + VP POC组合，含多空策略

### 数据文件
- data/btc_daily.csv — BTC日线OHLCV缓存
- data/fear_greed.csv — 恐惧贪婪指数缓存
- data/cb_premium.csv — Coinbase Premium缓存
- data/binance/binance_derivatives_daily.csv — Binance衍生品（资金费率+OI）
- data/binance/btc_hourly.csv — BTC小时K线缓存
- data/binance/btc_15m.csv — BTC 15分钟K线缓存（用于VP计算）
- data/binance/funding/ — 资金费率月度归档
- data/binance/metrics/ — OI月度归档

### 输出文件
- results_*_summary.txt — 各策略回测结果汇总
- trade_logs_*.txt — 各策略交易日志

### 测试
- tests/test_data_quality.py — 数据质量检查（缺失值、重复、异常值、连续性、新鲜度）
- tests/test_no_lookahead.py — 未来函数检测（合成跳变数据验证指标 + 信号/成交分离）

## 重要约定
- 新策略参考 strategy_test.py 中的现有配置
- 尽可能保持现有结构，保持代码的可重用性
- 所有数据下载/网络请求逻辑集中在 data_loader.py，engine.py 不做网络调用
- 所有程序console只输出最重要的结果，交易信息/日志放入单独的两个文件里
- 查日志/交易数据等用grep/find/head/tail，不要一次打开大文件
- 条件通过 combine_conditions(mode="AND"/"OR") 组合，支持 source-aware 退出逻辑
- 回测引擎支持多空（long/short），1x杠杆

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
关键指标（夏普比率、最大回撤、胜率, profit R Square, 最近一年的收益，夏普比率、最大回撤、胜率）

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
