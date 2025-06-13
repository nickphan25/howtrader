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
    raise ImportError("SMC library is required for this strategy")


class PureSMCStrategy(CtaTemplate):
    """
    Pure SMC Strategy - Uses ONLY smc.swing_highs_lows() function

    This strategy completely relies on the SMC library for:
    1. Swing point detection
    2. Structure analysis
    3. Entry/exit decisions

    No manual calculations whatsoever.
    """

    author = "Pure SMC Library Strategy"

    # Strategy parameters
    fixed_size = 1
    swing_length = 5  # Parameter for SMC swing detection
    min_bars_required = 50
    lookback_bars = 100  # How many bars to analyze with SMC

    # Risk management
    risk_reward_ratio = 2.0
    fib_entry_zone_start = 0.50   # 50% retracement entry start
    fib_entry_zone_end = 0.786    # 78.6% retracement entry end
    structure_buffer = 0.05       # 5% buffer for stop loss

    # Position tracking
    entry_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0
    long_entered = False
    short_entered = False

    # SMC Analysis Results
    smc_swing_data = None
    current_structure = None  # 'bullish', 'bearish', or None
    entry_setup = None        # Current entry setup details

    parameters = [
        "fixed_size", "swing_length", "min_bars_required", "lookback_bars",
        "risk_reward_ratio", "fib_entry_zone_start", "fib_entry_zone_end", "structure_buffer"
    ]

    variables = [
        "entry_price", "stop_loss_price", "take_profit_price",
        "long_entered", "short_entered", "current_structure"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        if not SMC_AVAILABLE:
            raise ImportError("SMC library is required for this strategy")

        # Use 1-hour timeframe for better swing detection
        self.bg_1h = BarGenerator(self.on_bar, 60, self.on_1h_bar)
        self.am_1h = ArrayManager(size=200)

    def on_init(self):
        self.write_log("Pure SMC Strategy initialized.")
        self.load_bar(30)

    def on_start(self):
        self.write_log("Pure SMC Strategy started.")

    def on_stop(self):
        self.write_log("Pure SMC Strategy stopped.")

    def on_tick(self, tick: TickData):
        self.bg_1h.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.bg_1h.update_bar(bar)

    def on_1h_bar(self, bar: BarData):
        """Main strategy logic using only SMC library"""
        self.cancel_all()

        self.am_1h.update_bar(bar)
        if not self.am_1h.inited or self.am_1h.count < self.min_bars_required:
            return

        current_price = bar.close_price

        # 1. Handle existing positions first
        if self.pos != 0:
            self.manage_position(current_price)
            return

        # 2. Get SMC swing analysis
        if not self.analyze_with_smc():
            return

        # 3. Look for entry opportunities based on pure SMC analysis
        if self.current_structure and self.entry_setup:
            self.look_for_smc_entry(current_price)

        self.put_event()

    def analyze_with_smc(self) -> bool:
        """
        Complete market analysis using ONLY SMC library
        Returns True if analysis is successful
        """
        try:
            # Prepare OHLC data for SMC
            bars_to_analyze = min(self.lookback_bars, self.am_1h.count)

            ohlc_df = pd.DataFrame({
                'open': self.am_1h.open_array[-bars_to_analyze:],
                'high': self.am_1h.high_array[-bars_to_analyze:],
                'low': self.am_1h.low_array[-bars_to_analyze:],
                'close': self.am_1h.close_array[-bars_to_analyze:]
            })

            # Get swing points using SMC library
            self.smc_swing_data = smc.swing_highs_lows(ohlc_df, swing_length=self.swing_length)

            if self.smc_swing_data is None or self.smc_swing_data.empty:
                return False

            # Analyze structure using SMC results
            self.analyze_smc_structure()

            return True

        except Exception as e:
            self.write_log(f"SMC analysis failed: {e}")
            return False

    def analyze_smc_structure(self):
        """
        Analyze market structure using pure SMC swing data
        """
        # Extract valid swing points from SMC results
        swing_points = []

        for idx in self.smc_swing_data.index:
            high_low_val = self.smc_swing_data.loc[idx, 'HighLow']
            level_val = self.smc_swing_data.loc[idx, 'Level']

            if pd.notna(high_low_val) and pd.notna(level_val):
                swing_points.append({
                    'index': idx,
                    'type': 'High' if high_low_val == 1 else 'Low',
                    'price': level_val,
                    'smc_value': high_low_val
                })

        if len(swing_points) < 4:
            self.current_structure = None
            self.entry_setup = None
            return

        # Sort by index to ensure chronological order
        swing_points.sort(key=lambda x: x['index'])

        # Analyze the most recent swing pattern
        self.identify_smc_structure_pattern(swing_points)

    def identify_smc_structure_pattern(self, swing_points: List[Dict]):
        """
        Identify bullish/bearish structure from SMC swing points
        """
        # Reset current analysis
        self.current_structure = None
        self.entry_setup = None

        # Need at least 4 swing points for pattern analysis
        if len(swing_points) < 4:
            return

        # Analyze recent swing points for patterns
        recent_points = swing_points[-6:]  # Look at last 6 swing points

        # Look for bullish structure: Low -> High -> Low (Higher Low)
        self.check_bullish_structure(recent_points)

        # Look for bearish structure: High -> Low -> High (Lower High)
        if not self.current_structure:  # Only check if no bullish structure found
            self.check_bearish_structure(recent_points)

    def check_bullish_structure(self, swing_points: List[Dict]):
        """
        Check for bullish structure: Low(1) -> High(2) -> Low(3) where Low(3) > Low(1)
        """
        for i in range(len(swing_points) - 2):
            for j in range(i + 1, len(swing_points) - 1):
                for k in range(j + 1, len(swing_points)):

                    point1 = swing_points[i]  # Should be Low
                    point2 = swing_points[j]  # Should be High
                    point3 = swing_points[k]  # Should be Low (Higher Low)

                    # Check if pattern matches: Low -> High -> Low
                    if (point1['type'] == 'Low' and
                            point2['type'] == 'High' and
                            point3['type'] == 'Low'):

                        # Validate bullish structure rules
                        if (point2['price'] > point1['price'] and      # Higher High
                                point3['price'] > point1['price'] and      # Higher Low
                                point3['price'] < point2['price']):        # Valid retracement

                            # Calculate structure metrics
                            structure_range = point2['price'] - point1['price']
                            retracement = point2['price'] - point3['price']
                            retracement_pct = retracement / structure_range if structure_range > 0 else 0

                            # Valid retracement (30% to 85%)
                            if 0.30 <= retracement_pct <= 0.85:
                                self.current_structure = 'bullish'
                                self.entry_setup = {
                                    'type': 'bullish',
                                    'lowest_low': point1,
                                    'higher_high': point2,
                                    'higher_low': point3,
                                    'structure_range': structure_range,
                                    'retracement_pct': retracement_pct
                                }

                                self.write_log(f"SMC BULLISH STRUCTURE DETECTED:")
                                self.write_log(f"Lowest Low: {point1['price']:.4f}")
                                self.write_log(f"Higher High: {point2['price']:.4f}")
                                self.write_log(f"Higher Low: {point3['price']:.4f}")
                                self.write_log(f"Retracement: {retracement_pct:.1%}")
                                return

    def check_bearish_structure(self, swing_points: List[Dict]):
        """
        Check for bearish structure: High(1) -> Low(2) -> High(3) where High(3) < High(1)
        """
        for i in range(len(swing_points) - 2):
            for j in range(i + 1, len(swing_points) - 1):
                for k in range(j + 1, len(swing_points)):

                    point1 = swing_points[i]  # Should be High
                    point2 = swing_points[j]  # Should be Low
                    point3 = swing_points[k]  # Should be High (Lower High)

                    # Check if pattern matches: High -> Low -> High
                    if (point1['type'] == 'High' and
                            point2['type'] == 'Low' and
                            point3['type'] == 'High'):

                        # Validate bearish structure rules
                        if (point2['price'] < point1['price'] and      # Lower Low
                                point3['price'] < point1['price'] and      # Lower High
                                point3['price'] > point2['price']):        # Valid retracement

                            # Calculate structure metrics
                            structure_range = point1['price'] - point2['price']
                            retracement = point3['price'] - point2['price']
                            retracement_pct = retracement / structure_range if structure_range > 0 else 0

                            # Valid retracement (30% to 85%)
                            if 0.30 <= retracement_pct <= 0.85:
                                self.current_structure = 'bearish'
                                self.entry_setup = {
                                    'type': 'bearish',
                                    'highest_high': point1,
                                    'lower_low': point2,
                                    'lower_high': point3,
                                    'structure_range': structure_range,
                                    'retracement_pct': retracement_pct
                                }

                                self.write_log(f"SMC BEARISH STRUCTURE DETECTED:")
                                self.write_log(f"Highest High: {point1['price']:.4f}")
                                self.write_log(f"Lower Low: {point2['price']:.4f}")
                                self.write_log(f"Lower High: {point3['price']:.4f}")
                                self.write_log(f"Retracement: {retracement_pct:.1%}")
                                return

    def look_for_smc_entry(self, current_price: float):
        """
        Look for entry opportunities based on SMC structure analysis
        """
        if not self.entry_setup:
            return

        if self.current_structure == 'bullish':
            self.check_bullish_entry(current_price)
        elif self.current_structure == 'bearish':
            self.check_bearish_entry(current_price)

    def check_bullish_entry(self, current_price: float):
        """
        Check for bullish entry using SMC-identified structure
        """
        setup = self.entry_setup
        lowest_low = setup['lowest_low']['price']
        higher_high = setup['higher_high']['price']
        higher_low = setup['higher_low']['price']
        structure_range = setup['structure_range']

        # Calculate Fibonacci retracement levels from SMC swing points
        fib_start = higher_high - (structure_range * self.fib_entry_zone_start)  # 50%
        fib_end = higher_high - (structure_range * self.fib_entry_zone_end)      # 78.6%

        # Entry zone
        entry_zone_high = fib_start
        entry_zone_low = fib_end

        # Check if price is in the entry zone
        if entry_zone_low <= current_price <= entry_zone_high:

            # Additional SMC confirmation: price above higher low
            if current_price > higher_low:

                # Execute long trade
                self.execute_smc_long_trade(current_price, setup)

                self.write_log(f"SMC LONG ENTRY:")
                self.write_log(f"Price: {current_price:.4f}")
                self.write_log(f"Entry Zone: {entry_zone_low:.4f} - {entry_zone_high:.4f}")
                self.write_log(f"Stop: {self.stop_loss_price:.4f}")
                self.write_log(f"Target: {self.take_profit_price:.4f}")

    def check_bearish_entry(self, current_price: float):
        """
        Check for bearish entry using SMC-identified structure
        """
        setup = self.entry_setup
        highest_high = setup['highest_high']['price']
        lower_low = setup['lower_low']['price']
        lower_high = setup['lower_high']['price']
        structure_range = setup['structure_range']

        # Calculate Fibonacci retracement levels from SMC swing points
        fib_start = lower_low + (structure_range * self.fib_entry_zone_start)  # 50%
        fib_end = lower_low + (structure_range * self.fib_entry_zone_end)      # 78.6%

        # Entry zone
        entry_zone_low = fib_start
        entry_zone_high = fib_end

        # Check if price is in the entry zone
        if entry_zone_low <= current_price <= entry_zone_high:

            # Additional SMC confirmation: price below lower high
            if current_price < lower_high:

                # Execute short trade
                self.execute_smc_short_trade(current_price, setup)

                self.write_log(f"SMC SHORT ENTRY:")
                self.write_log(f"Price: {current_price:.4f}")
                self.write_log(f"Entry Zone: {entry_zone_low:.4f} - {entry_zone_high:.4f}")
                self.write_log(f"Stop: {self.stop_loss_price:.4f}")
                self.write_log(f"Target: {self.take_profit_price:.4f}")

    def execute_smc_long_trade(self, entry_price: float, setup: Dict):
        """
        Execute long trade based on SMC analysis
        """
        self.entry_price = entry_price

        # Stop loss below higher low with buffer
        higher_low_price = setup['higher_low']['price']
        self.stop_loss_price = higher_low_price * (1 - self.structure_buffer)

        # Take profit based on risk-reward ratio
        risk = abs(self.entry_price - self.stop_loss_price)
        self.take_profit_price = self.entry_price + (risk * self.risk_reward_ratio)

        # Execute trade
        self.buy(Decimal(entry_price), Decimal(self.fixed_size))
        self.long_entered = True

        # Reset structure analysis after entry
        self.current_structure = None
        self.entry_setup = None

    def execute_smc_short_trade(self, entry_price: float, setup: Dict):
        """
        Execute short trade based on SMC analysis
        """
        self.entry_price = entry_price

        # Stop loss above lower high with buffer
        lower_high_price = setup['lower_high']['price']
        self.stop_loss_price = lower_high_price * (1 + self.structure_buffer)

        # Take profit based on risk-reward ratio
        risk = abs(self.entry_price - self.stop_loss_price)
        self.take_profit_price = self.entry_price - (risk * self.risk_reward_ratio)

        # Execute trade
        self.short(Decimal(entry_price), Decimal(self.fixed_size))
        self.short_entered = True

        # Reset structure analysis after entry
        self.current_structure = None
        self.entry_setup = None

    def manage_position(self, current_price: float):
        """
        Manage existing positions with stop loss and take profit
        """
        if self.long_entered:
            if current_price <= self.stop_loss_price:
                self.sell(Decimal(current_price), Decimal(abs(self.pos)))
                self.write_log(f"SMC LONG STOPPED OUT at {current_price:.4f}")
                self.reset_position_tracking()
            elif current_price >= self.take_profit_price:
                self.sell(Decimal(current_price), Decimal(abs(self.pos)))
                self.write_log(f"SMC LONG PROFIT TAKEN at {current_price:.4f}")
                self.reset_position_tracking()

        elif self.short_entered:
            if current_price >= self.stop_loss_price:
                self.cover(Decimal(current_price), Decimal(abs(self.pos)))
                self.write_log(f"SMC SHORT STOPPED OUT at {current_price:.4f}")
                self.reset_position_tracking()
            elif current_price <= self.take_profit_price:
                self.cover(Decimal(current_price), Decimal(abs(self.pos)))
                self.write_log(f"SMC SHORT PROFIT TAKEN at {current_price:.4f}")
                self.reset_position_tracking()

    def reset_position_tracking(self):
        """
        Reset all position tracking variables
        """
        self.long_entered = False
        self.short_entered = False
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.write_log(f"SMC Trade executed: {trade.datetime}, Price: {trade.price}, Volume: {trade.volume}")
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass