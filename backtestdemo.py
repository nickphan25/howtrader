from howtrader.app.cta_strategy.backtesting import BacktestingEngine, OptimizationSetting
from howtrader.trader.object import Interval
from datetime import datetime
# from examples.strategies.atr_rsi_strategy import  AtrRsiStrategy  # 要导入你回测的策略，你自己开发的。
from examples.strategies.testsmc import SMCBasic

engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol="BTCUSDT.BINANCE",
    interval=Interval.MINUTE,
    start=datetime(2025, 1, 1),
    end=datetime(2025, 1, 3),
    rate=4/10000,
    slippage=0,
    size=1,
    pricetick=0.01,
    capital=1000000,
)

engine.add_strategy(SMCBasic, {})
# engine.add_strategy(AtrRsi15MinStrategy, {})

engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
# engine.show_chart()
# engine.show_widget()
# setting = OptimizationSetting()
# setting.set_target("sharpe_ratio")
# setting.add_parameter("atr_length", 3, 39, 1)
# setting.add_parameter("atr_ma_length", 10, 30, 1)
#
# result = engine.run_ga_optimization(setting)  # 优化策略参数
# print(result)  # 打印回测的结果，结果中会有比较好的结果值。
