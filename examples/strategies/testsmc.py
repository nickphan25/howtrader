import talib
from howtrader.app.cta_strategy import (
    CtaTemplate,
    StopOrder
)

from howtrader.trader.object import TickData, BarData, TradeData, OrderData
from howtrader.trader.constant import Interval, Direction, Offset
from howtrader.trader.utility import BarGenerator, ArrayManager

from decimal import Decimal
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime

try:
    from smartmoneyconcepts import smc
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False


class SMCBasic(CtaTemplate):
    author = "Pan"

    # Strategy parameters
    fixed_size = 1
    swing_length = 5  # Reduced for more sensitivity
    min_bars_for_structure = 20  # Reduced requirement

    # Variables
    entry_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0
    long_entered = False
    short_entered = False

    # Structure variables
    swing_high = 0.0
    swing_low = 0.0
    trend_direction = ""
    structure_confirmed = False

    parameters = ["fixed_size", "swing_length", "min_bars_for_structure"]
    variables = [
        "entry_price", "stop_loss_price", "take_profit_price",
        "long_entered", "short_entered", "swing_high", "swing_low",
        "trend_direction", "structure_confirmed"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # Simplified: Only use 15min bars
        self.bg_15m = BarGenerator(self.on_bar, 15, self.on_15m_bar)
        self.am_15m = ArrayManager(size=100)

    def on_init(self):
        self.write_log("SMCBasic strategy initialized.")
        self.load_bar(20)

    def on_start(self):
        self.write_log("SMCBasic strategy started.")

    def on_stop(self):
        self.write_log("SMCBasic strategy stopped.")

    def on_tick(self, tick: TickData):
        self.bg_15m.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.bg_15m.update_bar(bar)

    def on_15m_bar(self, bar: BarData):
        self.cancel_all()

        self.am_15m.update_bar(bar)
        if not self.am_15m.inited or self.am_15m.count < self.min_bars_for_structure:
            return

        current_price = bar.close_price

        # Check for stop loss and take profit first
        if self.long_entered:
            if current_price <= self.stop_loss_price:
                self.sell(Decimal(current_price), Decimal(abs(self.pos)))
                self.long_entered = False
                self.write_log(f"Long position stopped out at {current_price}")
                return
            elif current_price >= self.take_profit_price:
                self.sell(Decimal(current_price), Decimal(abs(self.pos)))
                self.long_entered = False
                self.write_log(f"Long position profit taken at {current_price}")
                return

        if self.short_entered:
            if current_price >= self.stop_loss_price:
                self.cover(Decimal(current_price), Decimal(abs(self.pos)))
                self.short_entered = False
                self.write_log(f"Short position stopped out at {current_price}")
                return
            elif current_price <= self.take_profit_price:
                self.cover(Decimal(current_price), Decimal(abs(self.pos)))
                self.short_entered = False
                self.write_log(f"Short position profit taken at {current_price}")
                return

        # Update market structure
        self.update_market_structure()

        # Entry logic
        if self.structure_confirmed and self.pos == 0:
            fib_zone = self.calculate_fibonacci_zone()

            if fib_zone:
                fib_low, fib_high, fib_786 = fib_zone
                in_fib_zone = fib_low <= current_price <= fib_high

                self.write_log(f"Structure: {self.trend_direction}, Price: {current_price:.4f}, Fib Zone: {fib_low:.4f}-{fib_high:.4f}")

                # Long entry
                if (self.trend_direction == "uptrend" and in_fib_zone and
                        not self.long_entered):

                    self.buy(Decimal(current_price), Decimal(self.fixed_size))
                    self.entry_price = current_price
                    self.stop_loss_price = fib_786

                    # Take profit at 2x risk
                    risk = abs(self.entry_price - self.stop_loss_price)
                    self.take_profit_price = self.entry_price + (2 * risk)

                    self.long_entered = True
                    self.write_log(f"LONG ENTRY: Price={current_price:.4f}, SL={self.stop_loss_price:.4f}, TP={self.take_profit_price:.4f}")

                # Short entry
                elif (self.trend_direction == "downtrend" and in_fib_zone and
                      not self.short_entered):

                    self.short(Decimal(current_price), Decimal(self.fixed_size))
                    self.entry_price = current_price
                    self.stop_loss_price = fib_786

                    # Take profit at 2x risk
                    risk = abs(self.entry_price - self.stop_loss_price)
                    self.take_profit_price = self.entry_price - (2 * risk)

                    self.short_entered = True
                    self.write_log(f"SHORT ENTRY: Price={current_price:.4f}, SL={self.stop_loss_price:.4f}, TP={self.take_profit_price:.4f}")

        self.put_event()

    def update_market_structure(self):
        """Simplified structure detection using swing highs/lows"""
        am = self.am_15m

        if am.count < self.min_bars_for_structure:
            return

        # Get recent price data
        highs = am.high_array[-30:]  # Last 30 bars
        lows = am.low_array[-30:]

        # Find swing highs and lows using simple logic
        swing_highs = []
        swing_lows = []

        # Look for swing points
        for i in range(self.swing_length, len(highs) - self.swing_length):
            # Swing high: highest point in local window
            if highs[i] == max(highs[i-self.swing_length:i+self.swing_length+1]):
                swing_highs.append((i, highs[i]))

            # Swing low: lowest point in local window
            if lows[i] == min(lows[i-self.swing_length:i+self.swing_length+1]):
                swing_lows.append((i, lows[i]))

        # Need at least 2 swing highs and 2 swing lows
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Get most recent swings
            recent_highs = swing_highs[-2:]
            recent_lows = swing_lows[-2:]

            high1_price = recent_highs[0][1]
            high2_price = recent_highs[1][1]
            low1_price = recent_lows[0][1]
            low2_price = recent_lows[1][1]

            # Determine trend
            if high2_price > high1_price and low2_price > low1_price:
                # Higher Highs + Higher Lows = Uptrend
                self.trend_direction = "uptrend"
                self.swing_high = high2_price
                self.swing_low = low2_price
                self.structure_confirmed = True
                self.write_log(f"UPTREND confirmed: HH={high2_price:.4f}, HL={low2_price:.4f}")

            elif high2_price < high1_price and low2_price < low1_price:
                # Lower Highs + Lower Lows = Downtrend
                self.trend_direction = "downtrend"
                self.swing_high = high2_price
                self.swing_low = low2_price
                self.structure_confirmed = True
                self.write_log(f"DOWNTREND confirmed: LH={high2_price:.4f}, LL={low2_price:.4f}")

            else:
                self.structure_confirmed = False
        else:
            self.structure_confirmed = False

    def calculate_fibonacci_zone(self):
        """Calculate Fibonacci retracement levels"""
        if not self.structure_confirmed or self.swing_high == 0 or self.swing_low == 0:
            return None

        swing_range = abs(self.swing_high - self.swing_low)

        if self.trend_direction == "uptrend":
            # Retracement from swing high
            fib_50 = self.swing_high - (swing_range * 0.5)   # 50%
            fib_618 = self.swing_high - (swing_range * 0.618) # 61.8%
            fib_786 = self.swing_high - (swing_range * 0.786) # 78.6%

            # Wider zone for more trades: 50%-78.6%
            fib_low = fib_786
            fib_high = fib_50
            stop_level = fib_786 - (swing_range * 0.1)  # Stop below 78.6%

        else:  # downtrend
            # Retracement from swing low
            fib_50 = self.swing_low + (swing_range * 0.5)   # 50%
            fib_618 = self.swing_low + (swing_range * 0.618) # 61.8%
            fib_786 = self.swing_low + (swing_range * 0.786) # 78.6%

            # Wider zone for more trades: 50%-78.6%
            fib_low = fib_50
            fib_high = fib_786
            stop_level = fib_786 + (swing_range * 0.1)  # Stop above 78.6%

        return fib_low, fib_high, stop_level

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.write_log(f"Trade executed: {trade.datetime}, Price: {trade.price}, Volume: {trade.volume}")
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass