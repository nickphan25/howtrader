from howtrader.app.cta_strategy import (
    CtaTemplate,
    StopOrder
)
from howtrader.trader.object import TickData, BarData, TradeData, OrderData
from howtrader.trader.utility import BarGenerator, ArrayManager
from decimal import Decimal


class LongShortBalancedStrategy(CtaTemplate):
    """
    📊 Balanced Long/Short Strategy for Widget Testing

    This strategy alternates between long and short positions based on:
    - Moving Average Crossover for direction
    - RSI for entry timing
    - Fixed stop-loss and take-profit levels

    Perfect for testing the Advanced Trading Widget with both position types!
    """

    author = "Trading Platform Demo"

    # Strategy parameters
    fast_ma_period = 10      # Fast moving average
    slow_ma_period = 20      # Slow moving average
    rsi_period = 14          # RSI period
    rsi_oversold = 30        # RSI oversold level (buy signal)
    rsi_overbought = 70      # RSI overbought level (sell signal)

    # Risk management
    stop_loss_percent = 2.0   # 2% stop loss
    take_profit_percent = 4.0 # 4% take profit (2:1 R/R)
    position_size = 1.0       # Fixed position size

    # Variables
    fast_ma = 0.0
    slow_ma = 0.0
    rsi_value = 0.0
    entry_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0
    current_trend = ""        # "bullish" or "bearish"

    parameters = [
        "fast_ma_period", "slow_ma_period", "rsi_period",
        "rsi_oversold", "rsi_overbought", "stop_loss_percent",
        "take_profit_percent", "position_size"
    ]

    variables = [
        "fast_ma", "slow_ma", "rsi_value", "entry_price",
        "stop_loss_price", "take_profit_price", "current_trend"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # Use 15-minute bars for more frequent signals
        self.bg = BarGenerator(self.on_bar, 15, self.on_15min_bar)
        self.am = ArrayManager(size=100)

    def on_init(self):
        """Initialize strategy"""
        self.write_log("🚀 Long-Short Balanced Strategy initialized")
        self.load_bar(30)  # Load enough bars for indicators

    def on_start(self):
        """Start strategy"""
        self.write_log("▶️ Long-Short Balanced Strategy started")

    def on_stop(self):
        """Stop strategy"""
        self.write_log("⏸️ Long-Short Balanced Strategy stopped")

    def on_tick(self, tick: TickData):
        """Process tick data"""
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """Process 1-minute bar data"""
        self.bg.update_bar(bar)

    def on_15min_bar(self, bar: BarData):
        """Process 15-minute bar data - main trading logic"""
        # Cancel any pending orders
        self.cancel_all()

        # Update array manager
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # Calculate indicators
        self.fast_ma = self.am.sma(self.fast_ma_period)
        self.slow_ma = self.am.sma(self.slow_ma_period)
        self.rsi_value = self.am.rsi(self.rsi_period)

        current_price = bar.close_price

        # Determine market trend
        if self.fast_ma > self.slow_ma:
            self.current_trend = "bullish"
        elif self.fast_ma < self.slow_ma:
            self.current_trend = "bearish"
        else:
            self.current_trend = "neutral"

        # 🎯 RISK MANAGEMENT: Check stop-loss and take-profit first
        if self.pos != 0:
            self._check_exit_conditions(current_price)
            return  # Don't enter new positions if already in one

        # 📈 ENTRY LOGIC: Enter positions based on trend + RSI
        if self.pos == 0:  # Only enter if no position

            # 🟢 LONG ENTRY CONDITIONS
            if (self.current_trend == "bullish" and
                    self.rsi_value < self.rsi_oversold and
                    self.rsi_value > 20):  # Avoid extreme oversold

                self._enter_long_position(current_price)

            # 🔴 SHORT ENTRY CONDITIONS
            elif (self.current_trend == "bearish" and
                  self.rsi_value > self.rsi_overbought and
                  self.rsi_value < 80):  # Avoid extreme overbought

                self._enter_short_position(current_price)

        self.put_event()

    def _enter_long_position(self, price):
        """Enter long position with risk management"""
        entry_price = price * 1.001  # Slight slippage

        # Calculate stop-loss and take-profit
        self.entry_price = entry_price
        self.stop_loss_price = entry_price * (1 - self.stop_loss_percent / 100)
        self.take_profit_price = entry_price * (1 + self.take_profit_percent / 100)

        # Place buy order
        self.buy(Decimal(entry_price), Decimal(self.position_size))

        self.write_log(
            f"🟢 LONG ENTRY: Price={entry_price:.4f}, "
            f"SL={self.stop_loss_price:.4f}, TP={self.take_profit_price:.4f}, "
            f"RSI={self.rsi_value:.1f}, Trend={self.current_trend}"
        )

    def _enter_short_position(self, price):
        """Enter short position with risk management"""
        entry_price = price * 0.999  # Slight slippage

        # Calculate stop-loss and take-profit
        self.entry_price = entry_price
        self.stop_loss_price = entry_price * (1 + self.stop_loss_percent / 100)
        self.take_profit_price = entry_price * (1 - self.take_profit_percent / 100)

        # Place short order
        self.short(Decimal(entry_price), Decimal(self.position_size))

        self.write_log(
            f"🔴 SHORT ENTRY: Price={entry_price:.4f}, "
            f"SL={self.stop_loss_price:.4f}, TP={self.take_profit_price:.4f}, "
            f"RSI={self.rsi_value:.1f}, Trend={self.current_trend}"
        )

    def _check_exit_conditions(self, current_price):
        """Check stop-loss and take-profit conditions"""

        if self.pos > 0:  # Long position
            if current_price <= self.stop_loss_price:
                # Stop-loss hit
                self.sell(Decimal(current_price * 0.999), Decimal(abs(self.pos)))
                self.write_log(f"🛑 LONG STOP-LOSS: Price={current_price:.4f}, Loss={(current_price - self.entry_price) / self.entry_price * 100:.2f}%")

            elif current_price >= self.take_profit_price:
                # Take-profit hit
                self.sell(Decimal(current_price * 0.999), Decimal(abs(self.pos)))
                self.write_log(f"🎯 LONG TAKE-PROFIT: Price={current_price:.4f}, Profit={(current_price - self.entry_price) / self.entry_price * 100:.2f}%")

        elif self.pos < 0:  # Short position
            if current_price >= self.stop_loss_price:
                # Stop-loss hit
                self.cover(Decimal(current_price * 1.001), Decimal(abs(self.pos)))
                self.write_log(f"🛑 SHORT STOP-LOSS: Price={current_price:.4f}, Loss={(self.entry_price - current_price) / self.entry_price * 100:.2f}%")

            elif current_price <= self.take_profit_price:
                # Take-profit hit
                self.cover(Decimal(current_price * 1.001), Decimal(abs(self.pos)))
                self.write_log(f"🎯 SHORT TAKE-PROFIT: Price={current_price:.4f}, Profit={(self.entry_price - current_price) / self.entry_price * 100:.2f}%")

    def on_order(self, order: OrderData):
        """Handle order updates"""
        self.write_log(f"📋 Order update: {order.orderid} - {order.status}")

    def on_trade(self, trade: TradeData):
        """Handle trade execution"""
        direction_symbol = "🟢" if trade.direction.value == "LONG" else "🔴"
        action_text = "OPEN" if trade.offset.value == "OPEN" else "CLOSE"

        self.write_log(
            f"{direction_symbol} Trade {action_text}: "
            f"Price={trade.price:.4f}, Volume={trade.volume:.4f}, "
            f"Time={trade.datetime.strftime('%H:%M:%S')}"
        )

    def on_stop_order(self, stop_order: StopOrder):
        """Handle stop order updates"""
        pass