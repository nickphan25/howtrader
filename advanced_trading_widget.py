"""
🎯 Advanced Trading Chart Widget for Howtrader Backtesting
Comprehensive charting solution with technical indicators and Smart Money Concepts
"""

import sys
import pyqtgraph as pg
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QCheckBox, QSpinBox, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSplitter,
    QDateEdit, QTabWidget, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QDate, QRectF, QPointF
from PySide6.QtGui import QFont, QColor, QPainter, QPicture

# Try to import required packages
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available - using basic indicators only")

try:
    import smartmoneyconcepts as smc
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    print("⚠️ Smart Money Concepts package not available")

# Configure pyqtgraph
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', '#1e1e1e')
pg.setConfigOption('foreground', '#ffffff')


@dataclass
class TradeData:
    """Trade information for plotting"""
    datetime: datetime
    price: float
    volume: float
    direction: str  # 'long' or 'short'
    action: str  # 'open' or 'close'
    pnl: float = 0.0


class TimeframeConverter:
    """🔥 Convert OHLCV data between different timeframes"""

    @staticmethod
    def convert_timeframe(bars_data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Convert bars data to target timeframe"""
        if bars_data.empty:
            return bars_data

        try:
            # Pre-computed timeframe mapping for faster lookup
            timeframe_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '1H', '4h': '4H', '1D': '1D', '1W': '1W', '1M': '1M'
            }

            freq = timeframe_map.get(target_timeframe)
            if not freq:
                print(f"❌ Unsupported timeframe: {target_timeframe}")
                return bars_data

            # Efficient data preparation
            df = bars_data.copy()
            if 'datetime' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df['datetime'] = df.index
                else:
                    print("❌ No datetime column found")
                    return bars_data

            # Fast resampling with optimized aggregation
            df = df.set_index('datetime')
            agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            resampled = df.resample(freq).agg(agg_dict).dropna()
            resampled.reset_index(inplace=True)

            print(f"🔄 Converted {len(bars_data)} bars to {len(resampled)} bars ({target_timeframe})")
            return resampled

        except Exception as e:
            print(f"❌ Error converting timeframe: {e}")
            return bars_data


class DataManager:
    """🔥 Data loading and caching management"""

    def __init__(self):
        self.original_bars_data: pd.DataFrame = None
        self.bars_data: pd.DataFrame = None
        self.trades_data: List[TradeData] = []
        self.symbol: str = ""
        self.interval: str = ""
        self.timestamps: List[datetime] = []
        self.converter = TimeframeConverter()
        self._cached_ohlcv_array = None  # Cache for performance

    def load_data(self, bars_data: pd.DataFrame, trades_data: List = None,
                  symbol: str = "", interval: str = ""):
        """Load and prepare data for charting"""
        self.original_bars_data = bars_data.copy() if bars_data is not None else pd.DataFrame()
        self.bars_data = self.original_bars_data.copy()
        self.trades_data = trades_data or []
        self.symbol = symbol
        self.interval = interval
        self._cached_ohlcv_array = None  # Clear cache

        # Ensure required columns exist
        if not self.bars_data.empty:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in self.bars_data.columns]
            if missing_cols:
                for col in missing_cols:
                    self.bars_data[col] = 0

            self._extract_timestamps()

        print(f"📊 Data loaded: {len(self.bars_data)} bars, {len(self.trades_data)} trades")

    def _extract_timestamps(self):
        """Extract timestamps from data efficiently"""
        if 'datetime' in self.bars_data.columns:
            self.timestamps = pd.to_datetime(self.bars_data['datetime']).tolist()
        elif hasattr(self.bars_data.index, 'name') and self.bars_data.index.name == 'datetime':
            self.timestamps = pd.to_datetime(self.bars_data.index).tolist()
        elif isinstance(self.bars_data.index, pd.DatetimeIndex):
            self.timestamps = self.bars_data.index.tolist()
        else:
            # Generate dummy timestamps efficiently
            base_time = datetime.now()
            self.timestamps = [base_time + timedelta(hours=i) for i in range(len(self.bars_data))]

    def convert_timeframe(self, target_timeframe: str):
        """Convert current data to target timeframe"""
        if self.original_bars_data.empty:
            return

        if target_timeframe == self.interval:
            self.bars_data = self.original_bars_data.copy()
        else:
            self.bars_data = self.converter.convert_timeframe(self.original_bars_data, target_timeframe)

        self.interval = target_timeframe
        self._extract_timestamps()
        self._cached_ohlcv_array = None  # Clear cache
        print(f"🔄 Timeframe converted to {target_timeframe}: {len(self.bars_data)} bars")

    def get_ohlcv_data(self) -> np.ndarray:
        """Get OHLCV data as numpy array with caching"""
        if self.bars_data.empty:
            return np.array([])

        if self._cached_ohlcv_array is None:
            self._cached_ohlcv_array = self.bars_data[['open', 'high', 'low', 'close', 'volume']].values

        return self._cached_ohlcv_array


class ImprovedTimeAxis(pg.AxisItem):
    """Improved time axis with proper formatting"""

    def __init__(self, timestamps: List[datetime] = None, interval: str = "1h", **kwargs):
        super().__init__(**kwargs)
        self.timestamps = timestamps or []
        self.interval = interval
        self.timestamps_len = len(self.timestamps)

    def update_timestamps(self, timestamps: List[datetime], interval: str):
        """Update timestamps and interval"""
        self.timestamps = timestamps
        self.interval = interval
        self.timestamps_len = len(self.timestamps)

    def tickStrings(self, values, scale, spacing):
        """Generate formatted tick strings"""
        if not self.timestamps:
            return [str(int(v)) for v in values]

        strings = []
        for value in values:
            try:
                index = int(round(value))
                if 0 <= index < self.timestamps_len:
                    timestamp = self.timestamps[index]
                    strings.append(self._format_timestamp(timestamp))
                else:
                    strings.append("")
            except (ValueError, IndexError):
                strings.append("")

        return strings

    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp based on interval"""
        format_map = {
            ('1m', '5m', '15m', '30m'): '%H:%M',
            ('1h', '4h'): '%m-%d %H:%M',
            ('1D',): '%m-%d',
            ('1W', '1M'): '%Y-%m'
        }

        for intervals, fmt in format_map.items():
            if self.interval in intervals:
                return timestamp.strftime(fmt)

        return timestamp.strftime('%m-%d %H:%M')


class IndicatorCalculator:
    """🔥 Technical indicator calculations"""

    @staticmethod
    def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA efficiently"""
        if TALIB_AVAILABLE:
            return talib.EMA(data, timeperiod=period)

        # Optimized EMA calculation
        alpha = 2.0 / (period + 1.0)
        ema = np.empty_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI efficiently"""
        if TALIB_AVAILABLE:
            return talib.RSI(data, timeperiod=period)

        # Optimized RSI calculation
        delta = np.diff(data, prepend=data[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = np.full(len(data), 50.0)
        if period < len(data):
            rsi[period:] = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      k_period: int = 9, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate KDJ indicator efficiently"""
        if TALIB_AVAILABLE:
            k, d = talib.STOCH(high, low, close,
                               fastk_period=k_period,
                               slowk_period=d_period,
                               slowd_period=d_period)
            j = 3 * k - 2 * d
            return k, d, j

        # Optimized KDJ calculation
        data_len = len(close)
        lowest_low = np.full(data_len, np.nan)
        highest_high = np.full(data_len, np.nan)

        for i in range(k_period - 1, data_len):
            start_idx = i - k_period + 1
            lowest_low[i] = np.min(low[start_idx:i + 1])
            highest_high[i] = np.max(high[start_idx:i + 1])

        range_val = highest_high - lowest_low
        k = np.where(range_val != 0, 100 * (close - lowest_low) / range_val, 50.0)
        d = np.full(data_len, 50.0)
        j = np.full(data_len, 50.0)

        return k, d, j


class SMCCalculator:
    """🔥 Smart Money Concepts calculations"""

    @staticmethod
    def calculate_fvg(ohlc_data: pd.DataFrame) -> List[Dict]:
        """Calculate Fair Value Gaps"""
        if not SMC_AVAILABLE or ohlc_data.empty:
            return []

        try:
            df = ohlc_data[['open', 'high', 'low', 'close']].copy()
            fvg_result = smc.smc.fvg(df, join_consecutive=False)

            if fvg_result is not None and not fvg_result.empty:
                fvg_list = []
                for idx in fvg_result.index:
                    fvg_value = fvg_result.loc[idx, 'FVG']
                    if pd.notna(fvg_value) and fvg_value != 0:
                        top_value = fvg_result.loc[idx, 'Top']
                        bottom_value = fvg_result.loc[idx, 'Bottom']

                        if pd.notna(top_value) and pd.notna(bottom_value):
                            fvg_type = 'BullishFVG' if fvg_value == 1 else 'BearishFVG'
                            fvg_list.append({
                                'index': idx,
                                'type': fvg_type,
                                'top': float(top_value),
                                'bottom': float(bottom_value)
                            })

                print(f"💰 Found {len(fvg_list)} Fair Value Gaps")
                return fvg_list

            print("💰 No Fair Value Gaps found")
            return []

        except Exception as e:
            print(f"❌ Error calculating FVG: {e}")
            return []

    @staticmethod
    def calculate_swing_highs_lows(ohlc_data: pd.DataFrame, swing_length: int = 5) -> Dict:
        """Calculate swing highs and lows"""
        if not SMC_AVAILABLE or ohlc_data.empty:
            return {'swing_highs': [], 'swing_lows': []}

        try:
            df = ohlc_data[['open', 'high', 'low', 'close']].copy()
            swing_result = smc.smc.swing_highs_lows(df, swing_length=swing_length)

            swing_highs = []
            swing_lows = []

            if swing_result is not None and not swing_result.empty:
                # The SMC library returns DataFrame with columns: 'HighLow' and 'Level'
                for idx in swing_result.index:
                    high_low_value = swing_result.loc[idx, 'HighLow']
                    level_value = swing_result.loc[idx, 'Level']

                    if pd.notna(high_low_value) and pd.notna(level_value):
                        if high_low_value == 1:  # Swing High
                            swing_highs.append({
                                'index': idx,
                                'price': float(level_value),
                                'type': 'swing_high'
                            })
                        elif high_low_value == -1:  # Swing Low
                            swing_lows.append({
                                'index': idx,
                                'price': float(level_value),
                                'type': 'swing_low'
                            })

            print(f"💰 Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
            return {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'total_points': len(swing_highs) + len(swing_lows)
            }

        except Exception as e:
            print(f"❌ Error calculating swing H/L: {e}")
            return {'swing_highs': [], 'swing_lows': []}

    @staticmethod
    def calculate_bos_choch(ohlc_data: pd.DataFrame) -> List[Dict]:
        """Calculate BOS/CHOCH with REAL break points - Enhanced List Format"""
        if not SMC_AVAILABLE or ohlc_data.empty:
            return []

        try:
            df = ohlc_data[['open', 'high', 'low', 'close']].copy()
            swing_result = smc.smc.swing_highs_lows(df, swing_length=5)
            bos_choch_result = smc.smc.bos_choch(df, swing_result, close_break=True)

            bos_choch_list = []
            if bos_choch_result is not None and not bos_choch_result.empty:
                for idx in bos_choch_result.index:
                    for col in ['BOS', 'CHOCH']:
                        if col in bos_choch_result.columns:
                            value = bos_choch_result.loc[idx, col]
                            level = bos_choch_result.loc[idx, 'Level']
                            broken_idx = bos_choch_result.loc[idx, 'BrokenIndex']

                            if pd.notna(value) and value != 0 and pd.notna(level) and pd.notna(broken_idx):
                                bos_choch_list.append({
                                    'index': idx,
                                    'type': f'{col}_{value}',
                                    'level': level,  # 🆕 Real structure level
                                    'broken_index': int(broken_idx),  # 🆕 Real break point
                                    'price': df.iloc[idx]['close'] if idx < len(df) else 0,
                                    'direction': int(value)  # 🆕 Direction for styling
                                })

            print(f"💰 Found {len(bos_choch_list)} REAL BOS/CHOCH signals")
            return bos_choch_list

        except Exception as e:
            print(f"❌ Error calculating BOS/CHOCH: {e}")
            return []

    @staticmethod
    def calculate_order_blocks(ohlc_data: pd.DataFrame) -> List[Dict]:
        """Calculate Order Blocks with realistic market data structure"""
        if not SMC_AVAILABLE or ohlc_data.empty:
            return []

        try:
            # Check if volume exists - SMC library requires it
            if 'volume' not in ohlc_data.columns:
                print("❌ Order Blocks require volume data - not available in dataset")
                return []

            df = ohlc_data[['open', 'high', 'low', 'close', 'volume']].copy()

            # First calculate swing highs/lows (required for order blocks)
            swing_hl_result = smc.smc.swing_highs_lows(df)

            if swing_hl_result is None or swing_hl_result.empty:
                print("📦 No swing highs/lows found for Order Blocks calculation")
                return []

            # Now calculate order blocks with swing highs/lows
            ob_result = smc.smc.ob(df, swing_hl_result)

            if ob_result is not None and not ob_result.empty:
                # Create realistic order block data structure similar to testa.py
                ob_list = []

                for idx in ob_result.index:
                    ob_value = ob_result.loc[idx, 'OB']
                    if pd.notna(ob_value) and ob_value != 0:
                        top_value = ob_result.loc[idx, 'Top']
                        bottom_value = ob_result.loc[idx, 'Bottom']

                        if pd.notna(top_value) and pd.notna(bottom_value):
                            # Get volume at this bar for realistic analysis
                            volume_at_ob = df.loc[idx, 'volume'] if idx < len(df) else 0

                            # Calculate mitigation index (when OB becomes invalid)
                            mitigation_idx = None
                            ob_price_level = float(top_value) if ob_value == 1 else float(bottom_value)

                            # Look ahead to find when order block is mitigated
                            for future_idx in range(idx + 1, len(df)):
                                future_bar = df.iloc[future_idx]

                                # Bullish OB mitigated when price closes below bottom
                                if ob_value == 1 and future_bar['close'] < bottom_value:
                                    mitigation_idx = future_idx
                                    break
                                # Bearish OB mitigated when price closes above top
                                elif ob_value == -1 and future_bar['close'] > top_value:
                                    mitigation_idx = future_idx
                                    break

                            ob_type = 'BullishOB' if ob_value == 1 else 'BearishOB'

                            # Create realistic order block structure
                            ob_data = {
                                'index': idx,
                                'type': ob_type,
                                'top': float(top_value),
                                'bottom': float(bottom_value),
                                'volume': float(volume_at_ob),
                                'mitigation_index': mitigation_idx,
                                'is_active': mitigation_idx is None,  # Still active if not mitigated
                                'ob_value': ob_value,
                                'strength': 'high' if volume_at_ob > df['volume'].median() else 'low'
                            }

                            ob_list.append(ob_data)

                print(f"📦 Found {len(ob_list)} Order Blocks with realistic market data")
                return ob_list

            print("📦 No Order Blocks found")
            return []

        except Exception as e:
            print(f"❌ Error calculating Order Blocks: {e}")
            return []

    @staticmethod
    def calculate_liquidity_levels(ohlc_data: pd.DataFrame) -> List[Dict]:
        """Calculate liquidity levels using professional SMC approach"""
        if not SMC_AVAILABLE or ohlc_data.empty:
            return []

        try:
            # Step 1: Calculate swing highs/lows first
            swing_result = smc.smc.swing_highs_lows(ohlc_data, swing_length=50)

            if swing_result is None or swing_result.empty:
                print("⚠️ No swing points found for liquidity calculation")
                return []

            # Step 2: Calculate liquidity using SMC library
            liquidity_result = smc.smc.liquidity(ohlc_data, swing_result, range_percent=0.01)

            if liquidity_result is None or liquidity_result.empty:
                print("⚠️ No liquidity levels found")
                return []

            # Step 3: Convert to our format
            liquidity_levels = []
            for i in range(len(liquidity_result)):
                if not pd.isna(liquidity_result['Liquidity'].iloc[i]):
                    level_data = {
                        'index': i,
                        'price': float(liquidity_result['Level'].iloc[i]),
                        'type': 'BSL' if liquidity_result['Liquidity'].iloc[i] == 1 else 'SSL',
                        'end_index': int(liquidity_result['End'].iloc[i]) if not pd.isna(
                            liquidity_result['End'].iloc[i]) else i + 15,
                        'swept_index': int(liquidity_result['Swept'].iloc[i]) if not pd.isna(
                            liquidity_result['Swept'].iloc[i]) else 0
                    }
                    liquidity_levels.append(level_data)

            print(f"💰 Found {len(liquidity_levels)} professional liquidity levels")
            return liquidity_levels

        except Exception as e:
            print(f"❌ Error calculating liquidity levels: {e}")
            return []

    @staticmethod
    def calculate_previous_hl(ohlc_data: pd.DataFrame) -> List[Dict]:
        """Calculate Previous Highs/Lows with market-realistic structure"""
        if ohlc_data.empty:
            return []

        try:
            # First get swing highs and lows as foundation
            swing_data = SMCCalculator.calculate_swing_highs_lows(ohlc_data, swing_length=20)

            prev_hl_lines = []

            # Process Previous Highs
            if swing_data['swing_highs']:
                high_levels = []
                for swing_high in swing_data['swing_highs']:
                    idx = swing_high['index']
                    price = swing_high['price']

                    # Only add if it's a new level (avoid duplicates)
                    if not high_levels or abs(price - high_levels[-1]['price']) > price * 0.001:  # 0.1% threshold
                        high_levels.append({'index': idx, 'price': price})

                # Create lines between consecutive high levels
                for i in range(len(high_levels) - 1):
                    current = high_levels[i]
                    next_level = high_levels[i + 1]

                    prev_hl_lines.append({
                        'start_index': current['index'],
                        'end_index': next_level['index'],
                        'price': current['price'],
                        'type': 'PreviousHigh',
                        'level_id': f"PH_{i}",
                        'strength': 1.0  # Could be enhanced with volume/touch analysis
                    })

            # Process Previous Lows
            if swing_data['swing_lows']:
                low_levels = []
                for swing_low in swing_data['swing_lows']:
                    idx = swing_low['index']
                    price = swing_low['price']

                    # Only add if it's a new level (avoid duplicates)
                    if not low_levels or abs(price - low_levels[-1]['price']) > price * 0.001:  # 0.1% threshold
                        low_levels.append({'index': idx, 'price': price})

                # Create lines between consecutive low levels
                for i in range(len(low_levels) - 1):
                    current = low_levels[i]
                    next_level = low_levels[i + 1]

                    prev_hl_lines.append({
                        'start_index': current['index'],
                        'end_index': next_level['index'],
                        'price': current['price'],
                        'type': 'PreviousLow',
                        'level_id': f"PL_{i}",
                        'strength': 1.0  # Could be enhanced with volume/touch analysis
                    })

            print(f"💰 Found {len(prev_hl_lines)} Previous H/L lines")
            return prev_hl_lines

        except Exception as e:
            print(f"❌ Error calculating Previous H/L: {e}")
            return []

    @staticmethod
    def calculate_sessions(ohlc_data: pd.DataFrame) -> List[Dict]:
        """
        Calculate trading sessions using SMC library

        Args:
            ohlc_data: DataFrame with OHLC data

        Returns:
            List[Dict]: Session data
        """
        try:
            if ohlc_data.empty:
                print("⚠️ No OHLC data provided for session calculation")
                return []

            # Prepare DataFrame for SMC library
            df = ohlc_data.copy()

            # Rename columns if needed for SMC compatibility
            column_mapping = {
                'open_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'close_price': 'close'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})

            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Missing required columns: {missing_cols}")
                return []

            # Ensure datetime index for SMC
            if 'datetime' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('datetime')
            elif not isinstance(df.index, pd.DatetimeIndex):
                print("⚠️ DataFrame must have datetime index for sessions calculation")
                return []

            sessions_data = []

            # Standard session configurations (matching SMC library)
            session_configs = ['Tokyo', 'London', 'New York']

            print("🎯 Calculating trading sessions using SMC library...")

            # Use SMC library with correct class method calling
            try:
                if SMC_AVAILABLE and smc is not None:
                    for session_name in session_configs:
                        try:
                            print(f"📊 Processing {session_name} session...")

                            # Call sessions method - CORRECT WAY: smc.smc.sessions()
                            session_result = smc.smc.sessions(df, session=session_name)

                            # Check if session_result has data
                            if session_result is None or session_result.empty:
                                print(f"⚠️ No {session_name} session data returned")
                                continue

                            print(f"✅ {session_name} session result shape: {session_result.shape}")
                            print(f"📋 {session_name} columns: {list(session_result.columns)}")

                            # Process session results - find session boundaries
                            # Get active periods where Active == 1
                            active_mask = session_result['Active'] == 1
                            active_indices = session_result[active_mask].index.tolist()

                            if len(active_indices) == 0:
                                print(f"⚠️ No active periods found for {session_name}")
                                continue

                            print(f"🔍 Found {len(active_indices)} active periods for {session_name}")

                            # Group consecutive active periods into sessions
                            session_groups = []
                            current_group = []

                            for i, idx in enumerate(active_indices):
                                if i == 0:
                                    current_group = [idx]
                                else:
                                    # Check if this index is consecutive to the previous
                                    # Since these are integer indices, we can compare directly
                                    prev_idx = active_indices[i - 1]

                                    if idx - prev_idx == 1:
                                        current_group.append(idx)
                                    else:
                                        # Start new group - save current group first
                                        if current_group:
                                            session_groups.append(current_group)
                                        current_group = [idx]

                            # Add the last group
                            if current_group:
                                session_groups.append(current_group)

                            # Create session data from groups
                            for group_idx, group in enumerate(session_groups):
                                try:
                                    start_idx = group[0]
                                    end_idx = group[-1]

                                    # Get the actual datetime from the original DataFrame
                                    start_time = df.index[start_idx]
                                    end_time = df.index[end_idx]

                                    # Get session high/low from the SMC result
                                    session_high = session_result.loc[group, 'High'].max()
                                    session_low = session_result.loc[group, 'Low'].min()

                                    # Filter out zero values - use actual OHLC data instead
                                    if session_low == 0 or pd.isna(session_low):
                                        session_low = df.iloc[group]['low'].min()
                                    if session_high == 0 or pd.isna(session_high):
                                        session_high = df.iloc[group]['high'].max()

                                    # Create session data entry
                                    session_data = {
                                        'session': session_name,
                                        'datetime': start_time,
                                        'high': float(session_high),
                                        'low': float(session_low),
                                        'index': start_idx,  # Use integer index for display
                                        'price': float((session_high + session_low) / 2),
                                        'duration': len(group),
                                        'start_time': start_time,
                                        'end_time': end_time,
                                        'start_index': start_idx,
                                        'end_index': end_idx
                                    }

                                    sessions_data.append(session_data)

                                except Exception as group_error:
                                    print(f"⚠️ Error processing group {group_idx} for {session_name}: {group_error}")
                                    continue

                            print(f"✅ {session_name}: Found {len(session_groups)} session periods")

                        except Exception as smc_error:
                            print(f"⚠️ SMC {session_name} session error: {smc_error}")
                            import traceback
                            traceback.print_exc()
                            continue

                else:
                    print("⚠️ SMC library not available")
                    return []

            except Exception as smc_lib_error:
                print(f"❌ SMC library error: {smc_lib_error}")
                import traceback
                traceback.print_exc()
                return []

            print(f"✅ Sessions calculated: {len(sessions_data)} session periods")
            return sessions_data

        except Exception as e:
            print(f"❌ Error in calculate_sessions: {e}")
            import traceback
            traceback.print_exc()
            return []


class CandlestickItem(pg.GraphicsObject):
    """🔥 Custom candlestick chart item"""

    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()

    def generatePicture(self):
        """Generate the candlestick picture efficiently"""
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)

        candlestick_width = 0.6
        green_color = pg.QtGui.QColor(34, 197, 94)
        red_color = pg.QtGui.QColor(239, 68, 68)

        for i, (open_price, high, low, close, volume) in enumerate(self.data):
            color = green_color if close >= open_price else red_color
            p.setPen(pg.mkPen(color))
            p.setBrush(pg.mkBrush(color))

            # Draw wick
            p.drawLine(pg.QtCore.QPointF(i, low), pg.QtCore.QPointF(i, high))

            # Draw body
            body_height = abs(close - open_price)
            body_top = max(close, open_price)

            if body_height > 0:
                p.drawRect(pg.QtCore.QRectF(i - candlestick_width / 2, body_top, candlestick_width, -body_height))
            else:
                p.drawLine(pg.QtCore.QPointF(i - candlestick_width / 2, close),
                          pg.QtCore.QPointF(i + candlestick_width / 2, close))

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class SMCRectangleItem(pg.GraphicsObject):
    """SMC rectangle visualization (FVG, Order Blocks) - Compatible with tuple format and PySide6"""

    def __init__(self, rectangles, color, opacity=0.3, text_overlay=None):
        super().__init__()
        self.rectangles = rectangles
        self.color = color
        self.opacity = opacity
        self.text_overlay = text_overlay
        self.picture = None  # Lazy init

    def generatePicture(self):
        self.picture = QPicture()
        p = QPainter(self.picture)

        fill_color = QColor(self.color)
        fill_color.setAlphaF(self.opacity)

        border_color = QColor(self.color)
        border_color.setAlphaF(0.8)

        p.setPen(pg.mkPen(border_color, width=1))
        p.setBrush(pg.mkBrush(fill_color))

        for rect_data in self.rectangles:
            if isinstance(rect_data, tuple) and len(rect_data) >= 4:
                x0, y0, x1, y1 = rect_data[:4]
                text = rect_data[4] if len(rect_data) > 4 else None
                width = x1 - x0
                height = y1 - y0
            elif isinstance(rect_data, dict):
                x0 = rect_data.get('x', 0)
                y0 = rect_data.get('y', 0)
                width = rect_data.get('width', 1)
                height = rect_data.get('height', 1)
                text = rect_data.get('text', None)
            else:
                continue

            rect = QRectF(x0, y0, width, height)
            p.drawRect(rect)

            if text:
                center = QPointF(x0 + width / 2, y0 + height / 2)
                p.setPen(pg.mkPen('#ffffff', width=1))
                p.drawText(center, str(text))

        p.end()

    def paint(self, p, *args):
        if self.picture is None:
            self.generatePicture()
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if not self.rectangles:
            return QRectF()

        x_coords = []
        y_coords = []

        for rect_data in self.rectangles:
            if isinstance(rect_data, tuple) and len(rect_data) >= 4:
                x0, y0, x1, y1 = rect_data[:4]
            elif isinstance(rect_data, dict):
                x0 = rect_data.get('x', 0)
                y0 = rect_data.get('y', 0)
                width = rect_data.get('width', 1)
                height = rect_data.get('height', 1)
                x1 = x0 + width
                y1 = y0 + height
            else:
                continue

            x_coords.extend([x0, x1])
            y_coords.extend([y0, y1])

        if not x_coords or not y_coords:
            return QRectF()

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)


class SMCLineItem(pg.GraphicsObject):
    """Custom line item for SMC features matching testa.py style"""

    def __init__(self, lines, color='blue', width=2, style='solid', text_labels=None):
        super().__init__()
        self.lines = lines
        self.color = color
        self.width = width
        self.style = style
        self.text_labels = text_labels or []
        self.picture = None

    def generatePicture(self):
        """Generate picture for drawing lines"""
        self.picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self.picture)

        # Set pen style
        pen = pg.mkPen(color=self.color, width=self.width)
        if hasattr(self, 'style') and self.style == 'dashed':
            pen.setStyle(pg.QtCore.Qt.DashLine)

        painter.setPen(pen)

        # Draw lines
        for line in self.lines:
            # Handle tuple format: (x1, y1, x2, y2)
            if isinstance(line, tuple) and len(line) >= 4:
                x1, y1, x2, y2 = line
            # Handle dictionary format: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            elif isinstance(line, dict):
                x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
            else:
                print(f"⚠️ Unknown line format in generatePicture: {type(line)} - {line}")
                continue

            painter.drawLine(pg.QtCore.QPointF(x1, y1), pg.QtCore.QPointF(x2, y2))

        # Draw text labels if any
        if hasattr(self, 'text_labels') and self.text_labels:
            font = pg.QtGui.QFont()
            font.setPixelSize(10)
            painter.setFont(font)

            for label in self.text_labels:
                if isinstance(label, tuple) and len(label) >= 3:
                    x, y, text = label[0], label[1], label[2]
                    painter.drawText(pg.QtCore.QPointF(x, y), str(text))

        painter.end()

    def paint(self, painter, option, widget):
        if self.picture is None:
            self.generatePicture()
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        """Calculate bounding rectangle for all lines"""
        if not self.lines:
            return pg.QtCore.QRectF(0, 0, 1, 1)

        all_x = []
        all_y = []

        for line in self.lines:
            # Handle tuple format: (x1, y1, x2, y2)
            if isinstance(line, tuple) and len(line) >= 4:
                x1, y1, x2, y2 = line
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])
            # Handle dictionary format: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            elif isinstance(line, dict):
                all_x.extend([line['x1'], line['x2']])
                all_y.extend([line['y1'], line['y2']])
            else:
                print(f"⚠️ Unknown line format in boundingRect: {type(line)} - {line}")
                continue

        if not all_x or not all_y:
            return pg.QtCore.QRectF(0, 0, 1, 1)

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        width = max_x - min_x if max_x != min_x else 1
        height = max_y - min_y if max_y != min_y else 1

        return pg.QtCore.QRectF(min_x, min_y, width, height)


class TechnicalIndicatorPanel(QWidget):
    """🔥 Technical indicator control panel"""

    indicator_changed = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.indicators = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Title
        title = QLabel("📈 Technical Indicators")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #ffffff; padding: 6px; background-color: #2d3748;")
        layout.addWidget(title)

        # Indicator table
        self.indicator_table = QTableWidget()
        self.indicator_table.setColumnCount(4)
        self.indicator_table.setHorizontalHeaderLabels(['Indicator', 'Period', 'Param2', 'Show'])
        self.indicator_table.verticalHeader().setVisible(False)  # 👈 Ẩn mảng trắng góc trái
        self.indicator_table.setFont(QFont("Arial", 10))         # 👈 Font chữ đồng bộ

        self.indicator_table.setStyleSheet("""
            QTableWidget {
                background-color: #2d3748;
                color: #ffffff;
                gridline-color: #4a5568;
                border: 1px solid #4a5568;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #1a202c;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #4a5568;
            }
        """)
        self.indicator_table.verticalHeader().setDefaultSectionSize(34)  # 👈 Tăng chiều cao dòng

        # Set column widths
        header = self.indicator_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.indicator_table.setColumnWidth(1, 60)
        self.indicator_table.setColumnWidth(2, 60)
        self.indicator_table.setColumnWidth(3, 50)

        # Default indicators
        default_indicators = [
            ("EMA", 20, "", True),
            ("EMA", 50, "", False),
            ("EMA", 100, "", False),
            ("RSI", 14, "", False),
            ("KDJ", 9, "3", False)
        ]

        for name, period, param2, show in default_indicators:
            self.add_indicator_row(name, period, param2, show)

        layout.addWidget(self.indicator_table)

        # Update button
        update_btn = QPushButton("Update Indicators")
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #3182ce;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2c5aa0; }
        """)
        update_btn.clicked.connect(self.emit_indicator_change)
        layout.addWidget(update_btn)

        self.setLayout(layout)

    def add_indicator_row(self, name: str, period: int, param2: str, show: bool):
        """Add indicator row to table"""
        row = self.indicator_table.rowCount()
        self.indicator_table.insertRow(row)

        indicator_combo = QComboBox()
        indicators = ['EMA', 'SMA', 'RSI', 'MACD', 'KDJ', 'Bollinger']
        if TALIB_AVAILABLE:
            indicators.extend(['ATR', 'ADX', 'CCI', 'Williams'])
        indicator_combo.addItems(indicators)
        indicator_combo.setCurrentText(name)
        self.indicator_table.setCellWidget(row, 0, indicator_combo)

        period_spin = QSpinBox()
        period_spin.setRange(1, 200)
        period_spin.setValue(period)
        self.indicator_table.setCellWidget(row, 1, period_spin)

        param2_item = QTableWidgetItem(param2)
        self.indicator_table.setItem(row, 2, param2_item)

        show_check = QCheckBox()
        show_check.setChecked(show)
        self.indicator_table.setCellWidget(row, 3, show_check)

    def get_active_indicators(self) -> List[Dict]:
        """Get list of active indicators"""
        active = []
        for row in range(self.indicator_table.rowCount()):
            show_widget = self.indicator_table.cellWidget(row, 3)
            if show_widget and show_widget.isChecked():
                indicator_widget = self.indicator_table.cellWidget(row, 0)
                period_widget = self.indicator_table.cellWidget(row, 1)
                param2_item = self.indicator_table.item(row, 2)

                active.append({
                    'name': indicator_widget.currentText(),
                    'period': period_widget.value(),
                    'param2': param2_item.text() if param2_item else "",
                })
        return active

    def emit_indicator_change(self):
        """Emit indicator change signal"""
        indicators = self.get_active_indicators()
        self.indicator_changed.emit("update", {"indicators": indicators})



class SMCPanel(QWidget):
    """🔥 Smart Money Concepts control panel"""

    smc_changed = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("💰 Smart Money Concepts")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #ffffff; padding: 5px; background-color: #2d3748;")
        layout.addWidget(title)

        # SMC status
        status_text = "✅ SMC Package Available" if SMC_AVAILABLE else "❌ SMC Package Not Available"
        status_color = "#68d391" if SMC_AVAILABLE else "#fc8181"
        status_label = QLabel(status_text)
        status_label.setStyleSheet(f"color: {status_color}; padding: 3px;")
        layout.addWidget(status_label)

        # SMC features
        self.smc_features = {
            'FVG': QCheckBox("Fair Value Gaps"),
            'SwingHL': QCheckBox("Swing Highs/Lows"),
            'BOS_CHOCH': QCheckBox("BOS/CHOCH"),
            'OrderBlocks': QCheckBox("Order Blocks"),
            'Liquidity': QCheckBox("Liquidity Levels"),
            'Pre-HL': QCheckBox("Previous H/L"),
            'Sessions': QCheckBox("Trading Sessions")
        }

        for checkbox in self.smc_features.values():
            checkbox.setStyleSheet("color: #ffffff; padding: 3px;")
            checkbox.setEnabled(SMC_AVAILABLE)
            layout.addWidget(checkbox)

        # Update button
        update_btn = QPushButton("Update SMC")
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #805ad5;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #6b46c1; }
        """)
        update_btn.setEnabled(SMC_AVAILABLE)
        update_btn.clicked.connect(self.emit_smc_change)
        layout.addWidget(update_btn)

        layout.addStretch()
        self.setLayout(layout)

    def get_active_smc_features(self) -> List[str]:
        """Get list of active SMC features"""
        return [name for name, checkbox in self.smc_features.items() if checkbox.isChecked()]

    def emit_smc_change(self):
        """Emit SMC change signal"""
        features = self.get_active_smc_features()
        self.smc_changed.emit("update", {"features": features})


class DataControlPanel(QWidget):
    """🔥 Data control panel with symbol, timeframe, date range"""

    data_changed = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("📊 Data Controls")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #ffffff; padding: 5px; background-color: #2d3748;")
        layout.addWidget(title)

        # Symbol
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['BTCUSDT.BINANCE', 'ETHUSDT.BINANCE', 'ADAUSDT.BINANCE'])
        self.symbol_combo.setEditable(False)
        self.symbol_combo.currentTextChanged.connect(self.emit_data_change)
        symbol_layout.addWidget(self.symbol_combo)
        layout.addLayout(symbol_layout)

        # Timeframe
        timeframe_layout = QHBoxLayout()
        timeframe_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1h', '4h', '1D', '1W', '1M'])
        self.timeframe_combo.setCurrentText('5m')
        self.timeframe_combo.currentTextChanged.connect(self.emit_data_change)
        timeframe_layout.addWidget(self.timeframe_combo)
        layout.addLayout(timeframe_layout)

        # Date range
        date_layout = QVBoxLayout()
        date_layout.addWidget(QLabel("Date Range:"))

        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-30))
        self.start_date.setCalendarPopup(True)
        self.start_date.dateChanged.connect(self.emit_data_change)
        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        self.end_date.dateChanged.connect(self.emit_data_change)
        date_layout.addWidget(QLabel("To:"))
        date_layout.addWidget(self.end_date)

        layout.addLayout(date_layout)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #38a169;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2f855a; }
        """)
        refresh_btn.clicked.connect(self.emit_data_change)
        layout.addWidget(refresh_btn)

        layout.addStretch()
        self.setLayout(layout)

    def emit_data_change(self):
        """Emit data change signal"""
        self.data_changed.emit("refresh", {
            "symbol": self.symbol_combo.currentText(),
            "timeframe": self.timeframe_combo.currentText(),
            "start_date": self.start_date.date().toPython(),
            "end_date": self.end_date.date().toPython()
        })


class TradingChartView(QWidget):
    """🔥 Main multi-panel chart display with fixed time axis"""

    def __init__(self):
        super().__init__()
        self.data_manager = DataManager()
        self.indicator_calculator = IndicatorCalculator()
        self.smc_calculator = SMCCalculator()
        self.trades_data = []
        self.time_axis = None
        self.volume_time_axis = None
        self.oscillator_time_axis = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create chart widget with multiple plots
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground('#1e1e1e')

        # Initialize improved time axes
        self.time_axis = ImprovedTimeAxis(orientation='bottom')
        self.volume_time_axis = ImprovedTimeAxis(orientation='bottom')
        self.oscillator_time_axis = ImprovedTimeAxis(orientation='bottom')

        # Main price chart
        self.price_plot = self.graphics_widget.addPlot(row=0, col=0, title="Price Chart",
                                                       axisItems={'bottom': self.time_axis})
        self.price_plot.addLegend(offset=(10, 10))
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setLabel('left', 'Price')

        # Volume chart
        self.volume_plot = self.graphics_widget.addPlot(row=1, col=0, title="Volume",
                                                        axisItems={'bottom': self.volume_time_axis})
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setLabel('left', 'Volume')
        self.volume_plot.setXLink(self.price_plot)

        # Oscillator chart
        self.oscillator_plot = self.graphics_widget.addPlot(row=2, col=0, title="Oscillators",
                                                            axisItems={'bottom': self.oscillator_time_axis})
        self.oscillator_plot.showGrid(x=True, y=True, alpha=0.3)
        self.oscillator_plot.setLabel('left', 'Value')
        self.oscillator_plot.setXLink(self.price_plot)

        self.graphics_widget.ci.layout.setRowStretchFactor(0, 7)  # Price chart - 70%
        self.graphics_widget.ci.layout.setRowStretchFactor(1, 1.5)  # Volume - 15%
        self.graphics_widget.ci.layout.setRowStretchFactor(2, 1.5)  # Oscillators - 15%

        layout.addWidget(self.graphics_widget)
        self.setLayout(layout)

    def update_chart_data(self, bars_data: pd.DataFrame, trades_data: List = None):
        """Update chart with new data"""
        self.data_manager.load_data(bars_data, trades_data)
        if trades_data:
            self.trades_data = trades_data

        self._update_time_axes()
        self.render_charts()

    def _update_time_axes(self):
        """Update all time axis instances"""
        if self.data_manager.timestamps:
            axes = [self.time_axis, self.volume_time_axis, self.oscillator_time_axis]
            for axis in axes:
                if axis:
                    axis.update_timestamps(self.data_manager.timestamps, self.data_manager.interval)

    def convert_timeframe(self, target_timeframe: str):
        """Convert chart data to new timeframe"""
        self.data_manager.convert_timeframe(target_timeframe)
        self._update_time_axes()
        self.render_charts()

    def render_charts(self):
        """Render all charts"""
        if self.data_manager.bars_data.empty:
            return

        # Clear existing plots
        self.price_plot.clear()
        self.volume_plot.clear()
        self.oscillator_plot.clear()

        data = self.data_manager.get_ohlcv_data()
        if len(data) == 0:
            return

        x_axis = np.arange(len(data))

        # Render candlesticks
        candlestick_item = CandlestickItem(data)
        self.price_plot.addItem(candlestick_item)

        # Render volume
        volume_data = data[:, 4]  # Volume column
        open_prices = data[:, 0]  # Open column
        close_prices = data[:, 3]  # Close column

        # Create separate bars for green (bullish) and red (bearish) volumes
        green_volumes = []
        red_volumes = []
        green_x = []
        red_x = []

        for i in range(len(volume_data)):
            if close_prices[i] >= open_prices[i]:  # Bullish candle
                green_volumes.append(volume_data[i])
                green_x.append(i)
                red_volumes.append(0)
                red_x.append(i)
            else:  # Bearish candle
                red_volumes.append(volume_data[i])
                red_x.append(i)
                green_volumes.append(0)
                green_x.append(i)

        # Add green volume bars (bullish)
        if green_volumes:
            green_bars = pg.BarGraphItem(x=green_x, height=green_volumes, width=0.8,
                                         brush=pg.mkBrush(34, 197, 94), pen=None)  # Green
            self.volume_plot.addItem(green_bars)

        # Add red volume bars (bearish)
        if red_volumes:
            red_bars = pg.BarGraphItem(x=red_x, height=red_volumes, width=0.8,
                                       brush=pg.mkBrush(239, 68, 68), pen=None)# Red
            self.volume_plot.addItem(red_bars)

        # Add trade marks AFTER rendering candlesticks
        self.add_trade_marks(x_axis)

        print(f"📊 Chart rendered with {len(data)} bars")

    def add_trade_marks(self, x_axis):
        """Add entry/exit trade marks with connecting lines - SIMPLIFIED VERSION"""
        if not self.trades_data:
            print("⚠️ No trades data available")
            return

        print(f"🔍 Processing {len(self.trades_data)} trades...")

        # Separate entry and exit trades
        entry_trades = []
        exit_trades = []

        # Process trades data efficiently
        for i, trade in enumerate(self.trades_data):
            try:
                # Handle different trade data formats
                if isinstance(trade, dict):
                    action = trade.get('action', '')
                    price = trade.get('price', 0)
                    volume = trade.get('volume', 0)
                    datetime_val = trade.get('datetime', None)
                    direction = trade.get('direction', 'long')
                    pnl = trade.get('pnl', 0.0)
                    trade_id = trade.get('trade_id', f"trade_{i}")
                    position_type = direction  # Use the direction directly from backtestwidget.py

                else:
                    # Raw trade objects (backup handling)
                    if hasattr(trade, 'datetime') and hasattr(trade, 'price'):
                        direction = getattr(trade, 'direction', 'long')
                        action = getattr(trade, 'action', 'open')
                        position_type = direction
                        price = float(trade.price)
                        volume = float(trade.volume)
                        datetime_val = trade.datetime
                        pnl = 0.0
                        trade_id = getattr(trade, 'tradeid', f"trade_{i}")
                    else:
                        print(f"⚠️ Skipping invalid trade object: {type(trade)}")
                        continue

                # Validate essential trade data
                if not datetime_val or not price or not action or not volume:
                    print(f"⚠️ Skipping trade: missing data")
                    continue

                # Find closest timestamp index
                if datetime_val and self.data_manager.timestamps:
                    closest_index = self._find_closest_timestamp_index(datetime_val, self.data_manager.timestamps)
                    if closest_index is not None:
                        trade_data = {
                            'index': closest_index,
                            'price': float(price),
                            'volume': float(volume),
                            'datetime': datetime_val,
                            'direction': direction,
                            'action': action,
                            'pnl': float(pnl) if pnl else 0.0,
                            'trade_id': trade_id,
                            'position_type': position_type
                        }

                        if action == 'open':
                            entry_trades.append(trade_data)
                        elif action == 'close':
                            exit_trades.append(trade_data)

            except Exception as e:
                print(f"❌ Error processing trade: {e}")
                continue

        # Group trades into entry-exit pairs with volume matching
        trade_pairs, unmatched_entries, unmatched_exits = self._group_trades_into_pairs(entry_trades, exit_trades)
        print(f"🔗 Created {len(trade_pairs)} trade pairs")

        # Draw marks and lines for matched pairs
        for pair in trade_pairs:
            entry = pair['entry']
            exit_trade = pair['exit']
            position_type = pair['position_type']

            # Direction-based coloring for entry marks
            if position_type == 'long':
                entry_color = pg.mkBrush(34, 197, 94)  # Green for long entry
                exit_color = pg.mkBrush(239, 68, 68)  # Red for long exit
                entry_symbol = 't'  # Triangle up for long entry
                exit_symbol = 't1'  # Triangle down for long exit
            else:  # short
                entry_color = pg.mkBrush(239, 68, 68)  # Red for short entry
                exit_color = pg.mkBrush(34, 197, 94)  # Green for short exit
                entry_symbol = 't1'  # Triangle down for short entry
                exit_symbol = 't'  # Triangle up for short exit

            # Add entry mark
            entry_scatter = pg.ScatterPlotItem(
                x=[entry['index']], y=[entry['price']],
                symbol=entry_symbol, size=15, brush=entry_color, pen=pg.mkPen('white', width=2)
            )
            self.price_plot.addItem(entry_scatter)

            # Add exit mark
            exit_scatter = pg.ScatterPlotItem(
                x=[exit_trade['index']], y=[exit_trade['price']],
                symbol=exit_symbol, size=15, brush=exit_color, pen=pg.mkPen('white', width=2)
            )
            self.price_plot.addItem(exit_scatter)

            # Draw connecting line with PnL information
            self._draw_trade_connection_line(entry, exit_trade, pair)

        # Draw marks for unmatched trades
        self._draw_unmatched_trades(unmatched_entries, unmatched_exits)

    def _group_trades_into_pairs(self, entry_trades, exit_trades):
        """
        🔥 FINAL CORRECTED: Match trades by same direction AND chronological order
        """
        from decimal import Decimal

        def validate_trade(trade, trade_type):
            """Validate trade data and return True if valid."""
            required_fields = ['direction', 'price', 'volume', 'datetime', 'index']

            for field in required_fields:
                if field not in trade or trade[field] is None:
                    print(f"⚠️ Skipping {trade_type} trade: missing {field}")
                    return False

            try:
                price = Decimal(str(trade['price']))
                volume = Decimal(str(trade['volume']))

                if price <= 0 or volume <= 0:
                    print(f"⚠️ Skipping {trade_type} trade: invalid price({price}) or volume({volume})")
                    return False

                if trade['direction'] not in ['long', 'short']:
                    print(f"⚠️ Skipping {trade_type} trade: invalid direction({trade['direction']})")
                    return False

            except (ValueError, TypeError) as e:
                print(f"⚠️ Skipping {trade_type} trade: data conversion error - {e}")
                return False

            return True

        def calculate_pnl(entry_price, exit_price, volume, position_type):
            """Calculate PnL based on position type."""
            if position_type == 'long':
                return (exit_price - entry_price) * volume
            else:  # short
                return (entry_price - exit_price) * volume

        def can_match_trades(entry, exit_trade):
            """🎯 FINAL FIX: Match trades with SAME direction (same position)."""
            # Entry and exit should have same direction for same position
            return entry['direction'] == exit_trade['direction']

        # Filter and validate trades
        valid_entries = []
        valid_exits = []

        for i, trade in enumerate(entry_trades):
            if validate_trade(trade, f"entry[{i}]"):
                trade_copy = trade.copy()
                trade_copy['remaining_volume'] = Decimal(str(trade['volume']))
                trade_copy['original_volume'] = Decimal(str(trade['volume']))
                valid_entries.append(trade_copy)

        for i, trade in enumerate(exit_trades):
            if validate_trade(trade, f"exit[{i}]"):
                trade_copy = trade.copy()
                trade_copy['remaining_volume'] = Decimal(str(trade['volume']))
                trade_copy['original_volume'] = Decimal(str(trade['volume']))
                valid_exits.append(trade_copy)

        print(f"✅ Valid trades: {len(valid_entries)} entries, {len(valid_exits)} exits")

        # Sort by datetime for chronological matching
        valid_entries.sort(key=lambda x: x['datetime'])
        valid_exits.sort(key=lambda x: x['datetime'])

        # Initialize result containers
        trade_pairs = []
        matched_exit_indices = set()


        for entry_idx, entry in enumerate(valid_entries):
            entry_remaining = entry['remaining_volume']
            position_type = entry.get('position_type', entry['direction'])

            # Find matching exits for this entry
            for exit_idx, exit_trade in enumerate(valid_exits):
                # Skip already fully matched exits
                if exit_idx in matched_exit_indices:
                    continue

                # Check chronological order (exit must be after entry)
                if exit_trade['datetime'] <= entry['datetime']:
                    continue

                # 🎯 NEW LOGIC: Check if trades can be matched (same direction)
                if not can_match_trades(entry, exit_trade):
                    continue

                # Check if exit has remaining volume
                if exit_trade['remaining_volume'] <= 0:
                    continue

                # Calculate volumes to match
                exit_remaining = exit_trade['remaining_volume']
                volume_to_match = min(entry_remaining, exit_remaining)


                # Create trade pair
                entry_price = Decimal(str(entry['price']))
                exit_price = Decimal(str(exit_trade['price']))

                pnl = calculate_pnl(entry_price, exit_price, volume_to_match, position_type)
                duration = exit_trade['datetime'] - entry['datetime']

                trade_pair = {
                    'entry': {
                        'datetime': entry['datetime'],
                        'price': float(entry_price),
                        'volume': float(volume_to_match),
                        'direction': entry['direction'],
                        'index': entry['index'],
                        'trade_id': entry.get('trade_id', 'unknown')
                    },
                    'exit': {
                        'datetime': exit_trade['datetime'],
                        'price': float(exit_price),
                        'volume': float(volume_to_match),
                        'direction': exit_trade['direction'],
                        'index': exit_trade['index'],
                        'trade_id': exit_trade.get('trade_id', 'unknown')
                    },
                    'pnl': float(pnl),
                    'position_type': position_type,
                    'duration': duration,
                    'volume_matched': float(volume_to_match)
                }

                trade_pairs.append(trade_pair)

                # Update remaining volumes
                entry['remaining_volume'] -= volume_to_match
                exit_trade['remaining_volume'] -= volume_to_match
                entry_remaining -= volume_to_match

                # Mark exit as fully matched if no remaining volume
                if exit_trade['remaining_volume'] <= 0:
                    matched_exit_indices.add(exit_idx)

                # Break if entry is fully matched
                if entry_remaining <= 0:
                    break

        # Collect unmatched trades
        unmatched_entries = []
        unmatched_exits = []

        for entry in valid_entries:
            if entry['remaining_volume'] > 0:
                unmatched_entry = {
                    'datetime': entry['datetime'],
                    'price': float(entry['price']),
                    'volume': float(entry['remaining_volume']),
                    'original_volume': float(entry['original_volume']),
                    'direction': entry['direction'],
                    'index': entry['index'],
                    'status': 'partial' if entry['remaining_volume'] < entry['original_volume'] else 'unmatched'
                }
                unmatched_entries.append(unmatched_entry)

        for exit_trade in valid_exits:
            if exit_trade['remaining_volume'] > 0:
                unmatched_exit = {
                    'datetime': exit_trade['datetime'],
                    'price': float(exit_trade['price']),
                    'volume': float(exit_trade['remaining_volume']),
                    'original_volume': float(exit_trade['original_volume']),
                    'direction': exit_trade['direction'],
                    'index': exit_trade['index'],
                    'status': 'partial' if exit_trade['remaining_volume'] < exit_trade[
                        'original_volume'] else 'unmatched'
                }
                unmatched_exits.append(unmatched_exit)

        # Summary statistics
        total_matched_volume = sum(pair['volume_matched'] for pair in trade_pairs)
        total_pnl = sum(pair['pnl'] for pair in trade_pairs)
        profitable_pairs = sum(1 for pair in trade_pairs if pair['pnl'] > 0)

        print(f"\n📊 === PAIRING SUMMARY ===")
        print(f"✅ Total pairs created: {len(trade_pairs)}")
        if trade_pairs:
            print(
                f"📈 Profitable pairs: {profitable_pairs}/{len(trade_pairs)} ({profitable_pairs / len(trade_pairs) * 100:.1f}%)")
            print(f"💰 Total PnL: {total_pnl:+.4f}")
            print(f"📊 Total matched volume: {total_matched_volume}")
        print(f"⚠️ Unmatched entries: {len(unmatched_entries)}")
        print(f"⚠️ Unmatched exits: {len(unmatched_exits)}")

        return trade_pairs, unmatched_entries, unmatched_exits

    def _draw_trade_connection_line(self, entry, exit_trade, pair):
        """Draw connection line between entry and exit with PnL display and position info"""
        try:
            entry_x = entry['index']
            entry_y = entry['price']
            exit_x = exit_trade['index']
            exit_y = exit_trade['price']

            pnl = pair['pnl']
            position_type = pair['position_type']
            volume = pair['volume_matched']
            duration = pair['duration']

            # 🔧 FIXED: Use Qt.PenStyle instead of integer for pen style
            from PySide6.QtCore import Qt

            # PnL-based line styling
            if pnl > 0:
                line_color = pg.mkPen(color=(34, 197, 94), width=3, style=Qt.PenStyle.SolidLine)  # Green for profit
                pnl_color = (34, 197, 94)
            elif pnl < 0:
                line_color = pg.mkPen(color=(239, 68, 68), width=3, style=Qt.PenStyle.SolidLine)  # Red for loss
                pnl_color = (239, 68, 68)
            else:
                line_color = pg.mkPen(color=(156, 163, 175), width=2, style=Qt.PenStyle.DashLine)  # Gray for breakeven
                pnl_color = (156, 163, 175)

            # Draw the connection line
            line = pg.PlotDataItem(
                x=[entry_x, exit_x],
                y=[entry_y, exit_y],
                pen=line_color,
                connect='all'
            )
            self.price_plot.addItem(line)

            # Calculate midpoint for text placement
            mid_x = (entry_x + exit_x) / 2
            mid_y = (entry_y + exit_y) / 2

            # Create comprehensive PnL text with position info
            position_symbol = "🔵" if position_type == 'long' else "🔴"
            duration_str = f"{duration.total_seconds() / 3600:.1f}h" if duration.total_seconds() < 86400 else f"{duration.days}d"

            # Multi-line text with position info
            pnl_text = f"{position_symbol} {position_type.upper()}\n"
            pnl_text += f"PnL: {pnl:+.4f}\n"
            pnl_text += f"Vol: {volume:.4f}\n"
            pnl_text += f"⏱ {duration_str}"

            # Add text item for PnL display
            text_item = pg.TextItem(
                text=pnl_text,
                color=pnl_color,
                anchor=(0.5, 0.5),
                border=pg.mkPen(color=(255, 255, 255), width=1),
                fill=pg.mkBrush(color=(0, 0, 0, 180))  # Semi-transparent background
            )
            text_item.setPos(mid_x, mid_y)
            self.price_plot.addItem(text_item)

            # Add volume indicators at entry and exit points
            self._add_volume_indicator(entry_x, entry_y, volume, position_type, "entry")
            self._add_volume_indicator(exit_x, exit_y, volume, position_type, "exit")

        except Exception as e:
            print(f"❌ Error drawing trade connection line: {e}")
            import traceback
            traceback.print_exc()

    def _add_volume_indicator(self, x, y, volume, position_type, trade_type):
        """Add small volume indicator near trade marks"""
        try:
            # Offset for volume text placement
            y_offset = 5 if trade_type == "entry" else -5

            volume_text = f"V:{volume:.2f}"
            volume_color = (34, 197, 94) if position_type == 'long' else (239, 68, 68)

            volume_item = pg.TextItem(
                text=volume_text,
                color=volume_color,
                anchor=(0.5, 0.5 if trade_type == "entry" else 0.5)
            )
            volume_item.setPos(x, y + y_offset)
            self.price_plot.addItem(volume_item)

        except Exception as e:
            print(f"⚠️ Warning: Could not add volume indicator - {e}")

    def _draw_unmatched_trades(self, unmatched_entries, unmatched_exits):
        """Draw markers for unmatched trades with different styling"""
        try:
            # Draw unmatched entries
            for entry in unmatched_entries:
                color = pg.mkBrush(255, 165, 0)  # Orange for unmatched
                symbol = 'x' if entry['status'] == 'unmatched' else 'star'

                scatter = pg.ScatterPlotItem(
                    x=[entry['index']], y=[entry['price']],
                    symbol=symbol, size=12, brush=color, pen=pg.mkPen('white', width=2)
                )
                self.price_plot.addItem(scatter)

                # Add warning text
                warning_text = f"⚠️ {entry['status'].upper()}\nEntry: {entry['direction']}\nVol: {entry['volume']:.4f}"
                text_item = pg.TextItem(
                    text=warning_text,
                    color=(255, 165, 0),
                    anchor=(0.5, 0.5)
                )
                text_item.setPos(entry['index'], entry['price'] + 5)
                self.price_plot.addItem(text_item)

            # Draw unmatched exits
            for exit_trade in unmatched_exits:
                color = pg.mkBrush(255, 165, 0)  # Orange for unmatched
                symbol = 'x' if exit_trade['status'] == 'unmatched' else 'star'

                scatter = pg.ScatterPlotItem(
                    x=[exit_trade['index']], y=[exit_trade['price']],
                    symbol=symbol, size=12, brush=color, pen=pg.mkPen('white', width=2)
                )
                self.price_plot.addItem(scatter)

                # Add warning text
                warning_text = f"⚠️ {exit_trade['status'].upper()}\nExit: {exit_trade['direction']}\nVol: {exit_trade['volume']:.4f}"
                text_item = pg.TextItem(
                    text=warning_text,
                    color=(255, 165, 0),
                    anchor=(0.5, 0.5)
                )
                text_item.setPos(exit_trade['index'], exit_trade['price'] - 5)
                self.price_plot.addItem(text_item)

        except Exception as e:
            print(f"❌ Error drawing unmatched trades: {e}")

    def _find_closest_timestamp_index(self, target_datetime, timestamps_list):
        """Find the closest timestamp index for trade marking"""
        if not timestamps_list:
            return None

        try:
            target_dt = pd.to_datetime(target_datetime)
            timestamps_dt = pd.to_datetime(timestamps_list)
            time_diffs = np.abs(timestamps_dt - target_dt)
            return int(time_diffs.argmin())
        except Exception:
            return None

    def add_indicator(self, indicator_data: Dict):
        """Add technical indicator to chart"""
        name = indicator_data.get('name', '')
        period = indicator_data.get('period', 14)

        if self.data_manager.bars_data.empty:
            return

        close_data = self.data_manager.bars_data['close'].values
        high_data = self.data_manager.bars_data['high'].values
        low_data = self.data_manager.bars_data['low'].values
        x_axis = np.arange(len(close_data))

        try:
            if name == 'EMA':
                ema_data = self.indicator_calculator.calculate_ema(close_data, period)
                if not np.all(np.isnan(ema_data)):
                    self.price_plot.plot(x_axis, ema_data, pen=pg.mkPen(color='yellow', width=2),
                                         name=f'EMA({period})')

            elif name == 'RSI':
                rsi_data = self.indicator_calculator.calculate_rsi(close_data, period)
                if not np.all(np.isnan(rsi_data)):
                    self.oscillator_plot.plot(x_axis, rsi_data, pen=pg.mkPen(color='orange', width=2),
                                              name=f'RSI({period})')
                    # Add reference lines
                    self.oscillator_plot.addLine(y=70, pen=pg.mkPen(color='red', style=pg.QtCore.Qt.DashLine))
                    self.oscillator_plot.addLine(y=30, pen=pg.mkPen(color='green', style=pg.QtCore.Qt.DashLine))

            elif name == 'KDJ':
                k, d, j = self.indicator_calculator.calculate_kdj(high_data, low_data, close_data, period)
                if not np.all(np.isnan(k)):
                    self.oscillator_plot.plot(x_axis, k, pen=pg.mkPen(color='blue', width=2), name=f'K({period})')
                    self.oscillator_plot.plot(x_axis, d, pen=pg.mkPen(color='red', width=2), name=f'D({period})')
                    self.oscillator_plot.plot(x_axis, j, pen=pg.mkPen(color='green', width=2), name=f'J({period})')

            print(f"📈 Added indicator: {name}({period})")
        except Exception as e:
            print(f"❌ Error adding indicator {name}: {e}")

    def add_smc_features(self, features: List[str]):
        """Add Smart Money Concepts features"""
        if not SMC_AVAILABLE or self.data_manager.bars_data.empty:
            print("❌ SMC not available or no data")
            return

        feature_methods = {
            'FVG': self.add_fair_value_gaps,
            'SwingHL': self.add_swing_highs_lows,
            'BOS_CHOCH': self.add_bos_choch,
            'OrderBlocks': self.add_order_blocks,
            'Liquidity': self.add_liquidity_levels,
            'Pre-HL': self.add_previous_hl,
            'Sessions': self.add_sessions
        }

        for feature in features:
            try:
                if feature in feature_methods:
                    feature_methods[feature]()
                    print(f"💰 Added SMC feature: {feature}")
            except Exception as e:
                print(f"❌ Error adding SMC feature {feature}: {e}")

    def add_fair_value_gaps(self):
        """Add Fair Value Gaps using professional display from testa.py"""
        try:
            fvg_data = self.smc_calculator.calculate_fvg(self.data_manager.bars_data)

            if not fvg_data:
                print("⚠️ No FVG data available")
                return

            data_len = len(self.data_manager.bars_data)
            self.display_fvg(fvg_data, data_len)

        except Exception as e:
            print(f"❌ Error adding Fair Value Gaps: {e}")

    def display_fvg(self, fvg_data, data_len):
        """Professional FVG display matching testa.py style"""
        rectangles = []
        for fvg in fvg_data:
            if isinstance(fvg, dict):
                idx = fvg.get('index', 0)
                top = fvg.get('top', 0)
                bottom = fvg.get('bottom', 0)
                fvg_type = fvg.get('type', 'bullish')

                # Create rectangle data
                x_start = max(0, idx - 2)
                x_end = min(data_len - 1, idx + 10)
                width = x_end - x_start
                height = abs(top - bottom)

                rectangles.append({
                    'x': x_start,
                    'y': min(top, bottom),
                    'width': width,
                    'height': height,
                    'type': fvg_type
                })

        if rectangles:
            # Use custom SMCRectangleItem for professional display
            color = '#FFD700' if 'bullish' in str(rectangles[0].get('type', '')) else '#FF6347'
            smc_item = SMCRectangleItem(rectangles, color=color, opacity=0.3, text_overlay="FVG")
            self.price_plot.addItem(smc_item)
            print(f"💰 Added {len(rectangles)} FVG rectangles")

    def add_swing_highs_lows(self):
        """Add Swing Highs/Lows using professional display from testa.py"""
        try:
            swing_data = self.smc_calculator.calculate_swing_highs_lows(self.data_manager.bars_data)

            if not swing_data:
                print("⚠️ No Swing H/L data available")
                return

            data_len = len(self.data_manager.bars_data)
            self.display_swing_highs_lows(swing_data, data_len)

        except Exception as e:
            print(f"❌ Error adding Swing H/L: {e}")

    def display_swing_highs_lows(self, swing_data, data_len):
        """Display swing highs and lows using SMCLineItem - lines only, no dots"""
        try:
            swing_highs = swing_data.get('swing_highs', [])
            swing_lows = swing_data.get('swing_lows', [])

            print(f"🎯 Creating swing lines for {len(swing_highs)} highs and {len(swing_lows)} lows")

            if not swing_highs and not swing_lows:
                print("⚠️ No swing points to display")
                return

            # Prepare all swing points and sort chronologically
            all_swings = []

            # Add swing highs
            for swing in swing_highs:
                if 0 <= swing['index'] < data_len:
                    all_swings.append({
                        'index': swing['index'],
                        'price': swing['price'],
                        'type': 'high'
                    })

            # Add swing lows
            for swing in swing_lows:
                if 0 <= swing['index'] < data_len:
                    all_swings.append({
                        'index': swing['index'],
                        'price': swing['price'],
                        'type': 'low'
                    })

            # Sort by index (chronological order)
            all_swings.sort(key=lambda x: x['index'])

            if len(all_swings) < 2:
                print("⚠️ Need at least 2 swing points to draw lines")
                return

            # Create lines for zigzag pattern (connecting all swings chronologically)
            zigzag_lines = []
            for i in range(len(all_swings) - 1):
                current = all_swings[i]
                next_swing = all_swings[i + 1]

                zigzag_lines.append({
                    'x1': current['index'],
                    'y1': current['price'],
                    'x2': next_swing['index'],
                    'y2': next_swing['price']
                })

            # Create zigzag line item
            if zigzag_lines:
                zigzag_item = SMCLineItem(
                    lines=zigzag_lines,
                    color='yellow',
                    width=2,
                    style='dash'
                )
                self.price_plot.addItem(zigzag_item)
                print(f"✅ Added zigzag line with {len(zigzag_lines)} segments")

            # Store items for management
            if not hasattr(self, 'smc_items'):
                self.smc_items = []

            # Store all created line items
            if 'zigzag_item' in locals():
                self.smc_items.append(zigzag_item)

            total_items = len([item for item in ['zigzag_item'] if item in locals()])
            print(f"💰 Successfully created {total_items} swing line items")

        except Exception as e:
            print(f"❌ Error displaying swing lines: {e}")
            import traceback
            traceback.print_exc()

    def add_bos_choch(self):
        """Add BOS/CHOCH using professional display from testa.py"""
        try:
            bos_choch_data = self.smc_calculator.calculate_bos_choch(self.data_manager.bars_data)

            if not bos_choch_data:
                print("⚠️ No BOS/CHOCH data available")
                return

            data_len = len(self.data_manager.bars_data)
            self.display_bos_choch(bos_choch_data, data_len)

        except Exception as e:
            print(f"❌ Error adding BOS/CHOCH: {e}")

    def display_bos_choch(self, bos_choch_data, data_len):
        """Display REAL BOS/CHOCH with accurate break points"""
        try:
            if not bos_choch_data:
                print("⚠️ No BOS/CHOCH data to display")
                return

            print(f"🎯 Creating REAL BOS/CHOCH display")

            bos_lines = []
            choch_lines = []
            bos_labels = []
            choch_labels = []

            for signal in bos_choch_data:
                try:
                    start_idx = signal['index']
                    signal_type = signal['type']
                    level = signal.get('level', signal['price'])  # Use structure level if available
                    broken_idx = signal.get('broken_index', start_idx + 20)  # Use real break or fallback
                    direction = signal.get('direction', 1)

                    # Ensure indices are within bounds
                    if start_idx >= data_len or broken_idx >= data_len:
                        continue

                    # Create line from signal to ACTUAL break point
                    line_data = {
                        'x1': start_idx,
                        'y1': level,  # 🆕 Use structure level, not close price
                        'x2': broken_idx,  # 🆕 Use real break point
                        'y2': level
                    }

                    # Process different signal types
                    if signal_type.startswith('BOS_'):
                        bos_lines.append(line_data)

                        # Enhanced label with break info
                        label_text = f"BOS {'↗' if direction == 1 else '↘'}"
                        label_offset = abs(level) * 0.008 * direction

                        bos_labels.append({
                            'x': start_idx + (broken_idx - start_idx) * 0.3,  # 30% along the line
                            'y': level + label_offset,
                            'text': label_text
                        })

                    elif signal_type.startswith('CHOCH_'):
                        choch_lines.append(line_data)

                        # Enhanced label with break info
                        label_text = f"CHoCH {'↗' if direction == 1 else '↘'}"
                        label_offset = abs(level) * 0.008 * direction

                        choch_labels.append({
                            'x': start_idx + (broken_idx - start_idx) * 0.3,  # 30% along the line
                            'y': level + label_offset,
                            'text': label_text
                        })

                except Exception as e:
                    print(f"⚠️ Error processing signal {signal}: {e}")
                    continue

            # Create BOS line items
            if bos_lines:
                bos_item = SMCLineItem(
                    lines=bos_lines,
                    color='#FF6B35',  # Vibrant Orange
                    width=4,
                    style='solid',
                    text_labels=bos_labels
                )
                self.price_plot.addItem(bos_item)
                print(f"✅ Added {len(bos_lines)} REAL BOS signals")

            # Create CHOCH line items
            if choch_lines:
                choch_item = SMCLineItem(
                    lines=choch_lines,
                    color='#8A2BE2',  # Blue Violet
                    width=3,
                    style='dash',
                    text_labels=choch_labels
                )
                self.price_plot.addItem(choch_item)
                print(f"✅ Added {len(choch_lines)} REAL CHOCH signals")

            # Store items for management
            if not hasattr(self, 'smc_items'):
                self.smc_items = []

            if 'bos_item' in locals():
                self.smc_items.append(bos_item)
            if 'choch_item' in locals():
                self.smc_items.append(choch_item)

            total_signals = len(bos_lines) + len(choch_lines)
            print(f"💰 Successfully displayed {total_signals} REAL BOS/CHOCH signals")

        except Exception as e:
            print(f"❌ Error displaying BOS/CHOCH: {e}")
            import traceback
            traceback.print_exc()

    def add_order_blocks(self):
        """Add Order Blocks using professional display from testa.py"""
        try:
            ob_data = self.smc_calculator.calculate_order_blocks(self.data_manager.bars_data)

            if not ob_data:
                print("⚠️ No Order Blocks data available")
                return

            data_len = len(self.data_manager.bars_data)
            self.display_order_blocks(ob_data, data_len)

        except Exception as e:
            print(f"❌ Error adding Order Blocks: {e}")

    def display_order_blocks(self, ob_data, data_len):
        """Realistic Order Blocks display with dynamic mitigation and volume analysis using SMCRectangleItem"""
        if not ob_data:
            return

        # Separate active and mitigated order blocks
        active_bullish = []
        active_bearish = []
        mitigated_blocks = []

        for i, ob in enumerate(ob_data):
            try:
                if isinstance(ob, dict):
                    idx = ob.get('index', 0)
                    top = ob.get('top', 0)
                    bottom = ob.get('bottom', 0)
                    ob_type = ob.get('type', 'BullishOB')
                    volume = ob.get('volume', 0)
                    mitigation_idx = ob.get('mitigation_index')
                    is_active = ob.get('is_active', True)

                    # Validate data
                    if not isinstance(idx, (int, float)) or idx < 0:
                        print(f"⚠️ Invalid index in order block {i}: {idx}")
                        continue

                    # Dynamic end calculation (realistic approach like testa.py)
                    if mitigation_idx is not None:
                        # Order block ends when mitigated
                        end_idx = min(int(mitigation_idx), data_len - 1)
                    else:
                        # Still active, extends to current time
                        end_idx = data_len - 1

                    # Create realistic volume-based text
                    if volume >= 1e9:
                        volume_text = f"{volume / 1e9:.1f}B"
                    elif volume >= 1e6:
                        volume_text = f"{volume / 1e6:.1f}M"
                    elif volume >= 1e3:
                        volume_text = f"{volume / 1e3:.1f}K"
                    else:
                        volume_text = f"{volume:.0f}"

                    # Add activity status to text
                    status = "Active" if is_active else "Mitigated"
                    text_overlay = f"OB: {volume_text} ({status})"

                    # Create rectangle tuple for SMCRectangleItem: (x0, y0, x1, y1, text)
                    rect_tuple = (
                        int(idx),  # x0 - start index (ensure integer)
                        float(min(top, bottom)),  # y0 - bottom price
                        int(end_idx),  # x1 - end index (ensure integer)
                        float(max(top, bottom)),  # y1 - top price
                        text_overlay  # text overlay
                    )

                    # Categorize by type and status
                    if is_active:
                        if 'Bullish' in ob_type:
                            active_bullish.append(rect_tuple)
                        else:
                            active_bearish.append(rect_tuple)
                    else:
                        mitigated_blocks.append(rect_tuple)
                else:
                    print(f"⚠️ Unexpected order block format at index {i}: {type(ob)} - {ob}")
                    continue

            except Exception as e:
                print(f"❌ Error processing order block {i}: {e}")
                print(f"   Data: {ob}")
                continue

        # Colors for different order block states
        active_bullish_color = '#4169E1'  # Royal Blue for active bullish
        active_bearish_color = '#DC143C'  # Crimson for active bearish
        mitigated_color = '#808080'  # Gray for mitigated blocks

        try:
            # Display mitigated blocks first (lower layer, lower opacity)
            if mitigated_blocks:
                mitigated_item = SMCRectangleItem(
                    mitigated_blocks,
                    color=mitigated_color,
                    opacity=0.15,
                    text_overlay=None  # Text is in individual rect_data[4]
                )
                self.price_plot.addItem(mitigated_item)
                if hasattr(self, 'smc_items'):
                    self.smc_items['MitigatedOrderBlocks'] = mitigated_item

            # Display active bullish order blocks
            if active_bullish:
                bullish_item = SMCRectangleItem(
                    active_bullish,
                    color=active_bullish_color,
                    opacity=0.25,
                    text_overlay=None  # Text is in individual rect_data[4]
                )
                self.price_plot.addItem(bullish_item)
                if hasattr(self, 'smc_items'):
                    self.smc_items['BullishOrderBlocks'] = bullish_item

            # Display active bearish order blocks
            if active_bearish:
                bearish_item = SMCRectangleItem(
                    active_bearish,
                    color=active_bearish_color,
                    opacity=0.25,
                    text_overlay=None  # Text is in individual rect_data[4]
                )
                self.price_plot.addItem(bearish_item)
                if hasattr(self, 'smc_items'):
                    self.smc_items['BearishOrderBlocks'] = bearish_item

            # Summary
            total_active = len(active_bullish) + len(active_bearish)
            total_mitigated = len(mitigated_blocks)
            print(f"💰 Added {total_active} Active + {total_mitigated} Mitigated Order Blocks using SMCRectangleItem")

        except Exception as e:
            print(f"❌ Error creating SMCRectangleItem: {e}")
            print(f"   Active bullish: {len(active_bullish) if active_bullish else 0}")
            print(f"   Active bearish: {len(active_bearish) if active_bearish else 0}")
            print(f"   Mitigated: {len(mitigated_blocks) if mitigated_blocks else 0}")

    def add_liquidity_levels(self):
        """Add Liquidity Levels using professional display from testa.py"""
        try:
            liquidity_data = self.smc_calculator.calculate_liquidity_levels(self.data_manager.bars_data)

            if not liquidity_data:
                print("⚠️ No Liquidity data available")
                return

            data_len = len(self.data_manager.bars_data)
            self.display_liquidity(liquidity_data, data_len)

        except Exception as e:
            print(f"❌ Error adding Liquidity Levels: {e}")

    def display_liquidity(self, liquidity_data, data_len):
        """Professional Liquidity display with realistic market representation"""
        lines = []
        text_labels = []

        for liq in liquidity_data:
            if isinstance(liq, dict):
                start_idx = liq.get('index', 0)
                price = liq.get('price', 0)
                liq_type = liq.get('type', 'BSL')
                end_idx = liq.get('end_index', start_idx + 15)
                swept_idx = liq.get('swept_index', 0)

                # ✅ REALISTIC: Draw line from liquidity point to actual end
                lines.append({
                    'x1': start_idx,
                    'y1': price,
                    'x2': end_idx,
                    'y2': price,
                    'type': liq_type
                })

                # Add liquidity label
                mid_x = (start_idx + end_idx) / 2
                text_labels.append({
                    'x': mid_x,
                    'y': price,
                    'text': liq_type
                })

                # ✅ REALISTIC: Show sweep line if liquidity was taken
                if swept_idx > 0 and swept_idx < data_len:
                    # Get the price where liquidity was swept
                    try:
                        if liq_type == 'BSL':  # Bullish liquidity swept above
                            swept_price = self.data_manager.bars_data.iloc[swept_idx]['high']
                        else:  # Bearish liquidity swept below
                            swept_price = self.data_manager.bars_data.iloc[swept_idx]['low']

                        # Draw sweep line from end of liquidity to sweep point
                        lines.append({
                            'x1': end_idx,
                            'y1': price,
                            'x2': swept_idx,
                            'y2': swept_price,
                            'type': 'swept'
                        })

                        # Add "SWEPT" label
                        sweep_mid_x = (end_idx + swept_idx) / 2
                        sweep_mid_y = (price + swept_price) / 2
                        text_labels.append({
                            'x': sweep_mid_x,
                            'y': sweep_mid_y,
                            'text': 'SWEPT'
                        })
                    except:
                        pass

        if lines:
            # ✅ REALISTIC: Use professional colors and styles
            color = '#ffa500'  # Orange like testa.py
            smc_item = SMCLineItem(lines, color=color, width=2, style='dash', text_labels=text_labels)
            self.price_plot.addItem(smc_item)
            print(
                f"💧 Added {len([l for l in lines if l.get('type') != 'swept'])} Liquidity levels with {len([l for l in lines if l.get('type') == 'swept'])} sweep lines")

    def add_previous_hl(self):
        """Add Previous High/Low using professional display from testa.py"""
        try:
            prev_hl_data = self.smc_calculator.calculate_previous_hl(self.data_manager.bars_data)

            if not prev_hl_data:
                print("⚠️ No Previous H/L data available")
                return

            data_len = len(self.data_manager.bars_data)
            self.display_prev_hl(prev_hl_data, data_len)

        except Exception as e:
            print(f"❌ Error adding Previous H/L: {e}")

    def display_prev_hl(self, prev_hl_data, data_len):
        """Market-realistic Previous H/L display using SMCLineItem"""
        if not prev_hl_data:
            return

        # Separate lines by type for better organization
        high_lines = []
        low_lines = []
        high_labels = []
        low_labels = []

        for line_data in prev_hl_data:
            start_idx = line_data['start_index']
            end_idx = line_data['end_index']
            price = line_data['price']
            line_type = line_data['type']

            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, data_len - 1))
            end_idx = max(0, min(end_idx, data_len - 1))

            # Create line tuple in SMCLineItem format: (x1, y1, x2, y2)
            line_tuple = (start_idx, price, end_idx, price)

            if line_type == 'PreviousHigh':
                high_lines.append(line_tuple)
                # Add text label at the end of the line
                high_labels.append((end_idx, price, "PH"))
            elif line_type == 'PreviousLow':
                low_lines.append(line_tuple)
                # Add text label at the end of the line
                low_labels.append((end_idx, price, "PL"))

        # Create separate SMCLineItem for highs and lows with different colors
        if high_lines:
            high_item = SMCLineItem(
                lines=high_lines,
                color='#FF6B6B',  # Red for previous highs
                width=2,
                style='solid',
                text_labels=high_labels
            )
            self.price_plot.addItem(high_item)
            self.smc_items.append(high_item)
            print(f"💰 Added {len(high_lines)} Previous High lines")

        if low_lines:
            low_item = SMCLineItem(
                lines=low_lines,
                color='#4ECDC4',  # Teal for previous lows
                width=2,
                style='solid',
                text_labels=low_labels
            )
            self.price_plot.addItem(low_item)
            self.smc_items.append(low_item)
            print(f"💰 Added {len(low_lines)} Previous Low lines")


    def add_sessions(self):
        """Add Trading Sessions using professional display from testa.py"""
        try:
            sessions_data = self.smc_calculator.calculate_sessions(self.data_manager.bars_data)

            if not sessions_data:
                print("⚠️ No Sessions data available")
                return

            data_len = len(self.data_manager.bars_data)
            self.display_sessions(sessions_data, data_len)

        except Exception as e:
            print(f"❌ Error adding Sessions: {e}")

    def display_sessions(self, session_data, data_len):
        """🎨 Display trading sessions with enhanced visual style (PySide6-compatible)"""
        try:
            print(f"🎨 Displaying {len(session_data)} enhanced sessions...")

            if not session_data:
                print("⚠️ No session data to display")
                return

            # Create rectangles for each session
            rectangles = []
            for session in session_data:
                if isinstance(session, dict):
                    start_idx = session.get('start_index', 0)
                    end_idx = session.get('end_index', 0)
                    high = session.get('high', 0)
                    low = session.get('low', 0)
                    session_name = session.get('session', 'Unknown')

                    width = max(1, end_idx - start_idx)
                    height = abs(high - low)

                    rectangles.append({
                        'x': start_idx,
                        'y': low,
                        'width': width,
                        'height': height,
                        'text': session_name
                    })

            if rectangles:
                # Define session colors (RGBA)
                color_map = {
                    'Tokyo': (255, 193, 7, 60),  # Amber
                    'London': (33, 150, 243, 60),  # Blue
                    'New York': (76, 175, 80, 60)  # Green
                }

                # Duyệt từng phiên, gom lại theo session_name nếu cần sau này
                for session_name in ['Tokyo', 'London', 'New York']:
                    # Lọc các hình chữ nhật thuộc phiên này
                    session_rects = [r for r in rectangles if r.get('text') == session_name]
                    if not session_rects:
                        continue

                    rgba = color_map.get(session_name, (255, 255, 255, 60))  # Default: white semi-transparent
                    color = QColor(*rgba[:3])
                    color.setAlpha(rgba[3])

                    smc_item = SMCRectangleItem(
                        rectangles=session_rects,
                        color=color,
                        opacity=rgba[3] / 255.0,
                        text_overlay=True
                    )

                    self.price_plot.addItem(smc_item)

                print(f"✅ Enhanced sessions displayed: {len(rectangles)} session periods")

        except Exception as e:
            print(f"❌ Error in enhanced display_sessions: {e}")
            import traceback
            traceback.print_exc()


class AdvancedTradingWidget(QWidget):
    """🎯 Main Advanced Trading Chart Widget"""

    def __init__(self, bars_data: pd.DataFrame = None, trades_data: List = None,
                 symbol: str = "BTCUSDT.BINANCE", interval: str = "1h"):
        super().__init__()

        self.bars_data = bars_data if bars_data is not None else pd.DataFrame()
        self.trades_data = trades_data or []
        self.symbol = symbol
        self.interval = interval

        self.init_ui()
        self.connect_signals()

        # Initialize with data if provided
        if not self.bars_data.empty:
            self.chart_view.update_chart_data(self.bars_data, self.trades_data)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🎯 Advanced Trading Chart Widget - Howtrader Backtesting")
        self.setGeometry(100, 100, 1600, 1000)

        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1a202c;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4a5568;
                border-radius: 8px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QComboBox, QSpinBox, QDateEdit {
                background-color: #2d3748;
                border: 1px solid #4a5568;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #ffffff;
            }
        """)

        # Main layout
        main_layout = QHBoxLayout()

        # Left control panel
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_panel.setStyleSheet("background-color: #2d3748; border-right: 2px solid #4a5568;")

        left_layout = QVBoxLayout()

        # Data controls
        self.data_panel = DataControlPanel()
        left_layout.addWidget(self.data_panel)

        # Technical indicators
        self.indicator_panel = TechnicalIndicatorPanel()
        left_layout.addWidget(self.indicator_panel)

        # Smart Money Concepts
        self.smc_panel = SMCPanel()
        left_layout.addWidget(self.smc_panel)

        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # Right chart area
        self.chart_view = TradingChartView()
        main_layout.addWidget(self.chart_view)

        # Set layout proportions
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        self.setLayout(main_layout)

    def connect_signals(self):
        """Connect signals between components"""
        self.data_panel.data_changed.connect(self.on_data_changed)
        self.indicator_panel.indicator_changed.connect(self.on_indicator_changed)
        self.smc_panel.smc_changed.connect(self.on_smc_changed)

    def on_data_changed(self, action: str, data: Dict):
        """Handle data control changes"""
        print(f"📊 Data changed: {action}, {data}")

        new_timeframe = data.get('timeframe', self.interval)
        if new_timeframe != self.interval:
            self.interval = new_timeframe
            print(f"🔄 Converting chart to timeframe: {new_timeframe}")

            self.chart_view.convert_timeframe(new_timeframe)
            self.data_panel.timeframe_combo.setCurrentText(new_timeframe)

    def on_indicator_changed(self, action: str, data: Dict):
        """Handle indicator changes"""
        print(f"📈 Indicators changed: {action}")
        indicators = data.get('indicators', [])

        # Clear and re-render with new indicators
        self.chart_view.render_charts()

        for indicator in indicators:
            self.chart_view.add_indicator(indicator)

    def on_smc_changed(self, action: str, data: Dict):
        """Handle SMC feature changes"""
        print(f"💰 SMC changed: {action}")
        features = data.get('features', [])

        if features:
            print(f"💰 Applying SMC features: {features}")
            self.chart_view.add_smc_features(features)
        else:
            print("💰 No SMC features selected")

    def update_data(self, bars_data: pd.DataFrame, trades_data: List = None,
                    symbol: str = None, interval: str = None):
        """Update widget with new data"""
        self.bars_data = bars_data if bars_data is not None else self.bars_data
        self.trades_data = trades_data if trades_data is not None else self.trades_data
        self.symbol = symbol if symbol is not None else self.symbol
        self.interval = interval if interval is not None else self.interval

        # Update chart
        self.chart_view.update_chart_data(self.bars_data, self.trades_data)

        # Update data panel
        if symbol:
            self.data_panel.symbol_combo.setCurrentText(symbol)
        if interval:
            self.data_panel.timeframe_combo.setCurrentText(interval)


def create_sample_data() -> Tuple[pd.DataFrame, List[Dict]]:
    """Create sample OHLCV data and trades for testing"""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')

    # Generate realistic price data efficiently
    np.random.seed(42)
    initial_price = 50000.0
    changes = np.random.normal(0, 200, 999)
    changes = np.insert(changes, 0, 0)  # No change for first price

    prices = np.empty(1000)
    prices[0] = initial_price

    for i in range(1, 1000):
        prices[i] = max(prices[i-1] + changes[i], 1000)  # Minimum price constraint

    # Vectorized OHLCV data creation
    close_changes = np.random.normal(0, 50, 1000)
    high_changes = np.abs(np.random.normal(0, 30, 1000))
    low_changes = np.abs(np.random.normal(0, 30, 1000))
    volumes = np.random.uniform(100, 10000, 1000)

    opens = prices
    closes = prices + close_changes
    highs = np.maximum(opens, closes) + high_changes
    lows = np.minimum(opens, closes) - low_changes

    # Create DataFrame efficiently
    df = pd.DataFrame({
        'datetime': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

    # Create sample trades efficiently
    trade_indices = np.arange(0, len(dates), 50)
    sample_trades = []

    for i in trade_indices:
        if i + 25 < len(dates):
            # Entry trade
            sample_trades.append({
                'datetime': dates[i],
                'price': prices[i],
                'volume': 1.0,
                'direction': 'long',
                'action': 'open',
                'pnl': 0.0
            })
            # Exit trade
            sample_trades.append({
                'datetime': dates[i + 25],
                'price': prices[i + 25],
                'volume': 1.0,
                'direction': 'long',
                'action': 'close',
                'pnl': prices[i + 25] - prices[i]
            })

    return df, sample_trades


def main():
    """Main function for testing the widget"""
    app = QApplication(sys.argv)

    # Create sample data
    sample_data, sample_trades = create_sample_data()

    # Create and show widget
    widget = AdvancedTradingWidget(
        bars_data=sample_data,
        trades_data=sample_trades,
        symbol="BTCUSDT.BINANCE",
        interval="1h"
    )

    widget.show()

    print("🎯 Advanced Trading Widget launched successfully!")
    print("📊 Features available:")
    print("  ✅ Multi-panel charting (Price/Volume/Oscillators)")
    print("  ✅ Technical indicators (EMA, RSI, KDJ)")
    print(f"  {'✅' if TALIB_AVAILABLE else '❌'} TA-Lib enhanced indicators")
    print(f"  {'✅' if SMC_AVAILABLE else '❌'} Smart Money Concepts")
    print("  ✅ Interactive controls")
    print("  ✅ Improved time axis")
    print("  ✅ Trade markers overlaid on price chart")
    print("  ✅ Working timeframe conversion")
    print("  ✅ Functional SMC features")
    print("  ✅ Dark theme")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()