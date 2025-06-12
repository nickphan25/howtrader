from howtrader.app.cta_strategy import (
    CtaTemplate,
    StopOrder
)
from howtrader.trader.object import TickData, BarData, TradeData, OrderData
from howtrader.trader.utility import BarGenerator, ArrayManager
from howtrader.trader.constant import Interval
from howtrader.app.cta_strategy.engine import CtaEngine
from decimal import Decimal


class FixedTradeTimeStrategy(CtaTemplate):
    """
    基于价格的定投 + 固定止损止盈
    """
    author = "51bitquant"

    # Parameters
    fixed_trade_money = 1000
    stop_loss_points = 2000  # 固定止损点数
    take_profit_points = 2000  # 固定止盈点数

    parameters = ["fixed_trade_money", "stop_loss_points", "take_profit_points"]

    # Variables to track positions
    entry_price = 0.0
    position_opened = False

    variables = ["entry_price", "position_opened"]

    def __init__(self, cta_engine: CtaEngine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg_1hour = BarGenerator(self.on_bar, 1, self.on_1hour_bar, Interval.HOUR)
        self.am = ArrayManager(size=100)

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(1)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log(f"我的策略启动")
        self.put_event()

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")
        self.put_event()

    def on_tick(self, tick: TickData):
        self.bg_1hour.update_tick(tick)

        # 实时检查止损止盈
        if self.position_opened and self.pos > 0:
            current_price = tick.last_price

            # 检查止损
            if current_price <= (self.entry_price - self.stop_loss_points):
                self.sell(Decimal(current_price), Decimal(abs(self.pos)))
                self.write_log(
                    f"触发止损: 入场价{self.entry_price}, 当前价{current_price}, 亏损{self.stop_loss_points}点")
                self.position_opened = False
                self.entry_price = 0.0

            # 检查止盈
            elif current_price >= (self.entry_price + self.take_profit_points):
                self.sell(Decimal(current_price), Decimal(abs(self.pos)))
                self.write_log(
                    f"触发止盈: 入场价{self.entry_price}, 当前价{current_price}, 盈利{self.take_profit_points}点")
                self.position_opened = False
                self.entry_price = 0.0

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.bg_1hour.update_bar(bar)
        self.put_event()

    def on_1hour_bar(self, bar: BarData):
        """
        1小时的K线数据.
        """
        self.cancel_all()
        self.am.update_bar(bar)

        if not self.am.inited:
            return

        # 如果已有持仓，先检查止损止盈
        if self.position_opened and self.pos > 0:
            current_price = bar.close_price

            # 检查止损
            if current_price <= (self.entry_price - self.stop_loss_points):
                price = current_price * 0.999  # 稍微低一点确保成交
                self.sell(Decimal(price), Decimal(abs(self.pos)))
                self.write_log(f"K线止损: 入场价{self.entry_price}, 当前价{current_price}")
                self.position_opened = False
                self.entry_price = 0.0
                return

            # 检查止盈
            elif current_price >= (self.entry_price + self.take_profit_points):
                price = current_price * 0.999  # 稍微低一点确保成交
                self.sell(Decimal(price), Decimal(abs(self.pos)))
                self.write_log(f"K线止盈: 入场价{self.entry_price}, 当前价{current_price}")
                self.position_opened = False
                self.entry_price = 0.0
                return

        # 定投逻辑: 周四下午三点定投， 周五下午四点定投
        # 只有在没有持仓的时候才开新仓
        if not self.position_opened and self.pos == 0:
            should_buy = False

            if bar.datetime.isoweekday() == 5 and bar.datetime.hour == 16:
                should_buy = True
            elif bar.datetime.isoweekday() == 4 and bar.datetime.hour == 15:
                should_buy = True

            if should_buy:
                price = bar.close_price * 1.001
                volume = self.fixed_trade_money / price

                self.buy(Decimal(price), Decimal(volume))
                self.entry_price = price
                self.position_opened = True

                self.write_log(f"开仓: 价格{price}, 数量{volume:.4f}")
                self.write_log(f"止损价: {self.entry_price - self.stop_loss_points}")
                self.write_log(f"止盈价: {self.entry_price + self.take_profit_points}")

        self.put_event()

    def on_order(self, order: OrderData):
        """
        订单的回调方法: 订单状态更新的时候，会调用这个方法。
        """
        self.write_log(f"订单更新: {order.orderid}, 状态: {order.status}, 价格: {order.price}, 数量: {order.volume}")
        self.put_event()

    def on_trade(self, trade: TradeData):
        """
        成交回调
        """
        self.write_log(f"成交: 时间{trade.datetime}, 价格{trade.price}, 数量{trade.volume}, 方向{trade.direction}")
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        停止单回调
        """
        pass