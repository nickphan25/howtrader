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
    def calculate_swing_highs_lows(ohlc_data: pd.DataFrame, swing_length: int = 50) -> Dict:
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
            swing_result = smc.smc.swing_highs_lows(df, swing_length=20)
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
        """Calculate Previous High/Low - No volume required"""
        if not SMC_AVAILABLE or ohlc_data.empty:
            return []

        try:
            # Previous H/L only needs OHLC data
            df = ohlc_data[['open', 'high', 'low', 'close']].copy()
            prev_hl_result = smc.smc.previous_high_low(df)

            if prev_hl_result is not None and not prev_hl_result.empty:
                prev_hl_list = []
                for idx in prev_hl_result.index:
                    for col in ['PrevHigh', 'PrevLow']:
                        if col in prev_hl_result.columns:
                            hl_value = prev_hl_result.loc[idx, col]
                            if pd.notna(hl_value) and hl_value != 0:
                                prev_hl_list.append({
                                    'index': idx,
                                    'type': col,
                                    'price': float(hl_value),
                                    'level': 'high' if col == 'PrevHigh' else 'low'
                                })

                print(f"📈 Found {len(prev_hl_list)} Previous High/Low levels")
                return prev_hl_list

            print("📈 No Previous High/Low found")
            return []

        except Exception as e:
            print(f"❌ Error calculating Previous High/Low: {e}")
            return []

    @staticmethod
    def calculate_sessions(ohlc_data: pd.DataFrame) -> List[Dict]:
        """Calculate Trading Sessions - No volume required"""
        if not SMC_AVAILABLE or ohlc_data.empty:
            return []

        try:
            # Sessions only need OHLC data
            df = ohlc_data[['open', 'high', 'low', 'close']].copy()
            sessions_result = smc.smc.sessions(df)

            if sessions_result is not None and not sessions_result.empty:
                sessions_list = []
                for idx in sessions_result.index:
                    for col in ['Session', 'Asia', 'London', 'NewYork']:
                        if col in sessions_result.columns:
                            session_value = sessions_result.loc[idx, col]
                            if pd.notna(session_value) and session_value != 0:
                                sessions_list.append({
                                    'index': idx,
                                    'type': col,
                                    'session': col,
                                    'active': bool(session_value)
                                })

                print(f"🌍 Found {len(sessions_list)} Session markers")
                return sessions_list

            print("🌍 No Sessions found")
            return []

        except Exception as e:
            print(f"❌ Error calculating Sessions: {e}")
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
        self.picture = QPicture()
        painter = QPainter(self.picture)

        # Set up pen style
        pen_style = Qt.SolidLine
        if self.style == 'dash':
            pen_style = Qt.DashLine
        elif self.style == 'dot':
            pen_style = Qt.DotLine
        elif self.style == 'dashdot':
            pen_style = Qt.DashDotLine

        pen = pg.mkPen(color=self.color, width=self.width, style=pen_style)
        painter.setPen(pen)

        # Draw lines
        for line in self.lines:
            x1, y1 = line['x1'], line['y1']
            x2, y2 = line['x2'], line['y2']
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # Draw text labels
        if self.text_labels:
            painter.setPen(pg.mkPen(color='white', width=1))
            for label in self.text_labels:
                painter.drawText(QPointF(label['x'], label['y']), label['text'])

        painter.end()

    def paint(self, painter, option, widget):
        if self.picture is None:
            self.generatePicture()
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if not self.lines:
            return QRectF()

        all_x = []
        all_y = []
        for line in self.lines:
            all_x.extend([line['x1'], line['x2']])
            all_y.extend([line['y1'], line['y2']])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)


class TechnicalIndicatorPanel(QWidget):
    """🔥 Technical indicator control panel"""

    indicator_changed = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.indicators = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("📈 Technical Indicators")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #ffffff; padding: 5px; background-color: #2d3748;")
        layout.addWidget(title)

        # Indicator table
        self.indicator_table = QTableWidget()
        self.indicator_table.setColumnCount(4)
        self.indicator_table.setHorizontalHeaderLabels(['Indicator', 'Period', 'Param2', 'Show'])
        self.indicator_table.setStyleSheet("""
            QTableWidget {
                background-color: #2d3748;
                color: #ffffff;
                gridline-color: #4a5568;
                border: 1px solid #4a5568;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #4a5568;
            }
            QHeaderView::section {
                background-color: #1a202c;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #4a5568;
            }
        """)

        # Set column widths
        header = self.indicator_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.indicator_table.setColumnWidth(1, 60)
        self.indicator_table.setColumnWidth(2, 60)
        self.indicator_table.setColumnWidth(3, 50)

        # Add default indicators
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

        # Indicator name
        indicator_combo = QComboBox()
        indicators = ['EMA', 'SMA', 'RSI', 'MACD', 'KDJ', 'Bollinger']
        if TALIB_AVAILABLE:
            indicators.extend(['ATR', 'ADX', 'CCI', 'Williams'])
        indicator_combo.addItems(indicators)
        indicator_combo.setCurrentText(name)
        self.indicator_table.setCellWidget(row, 0, indicator_combo)

        # Period
        period_spin = QSpinBox()
        period_spin.setRange(1, 200)
        period_spin.setValue(period)
        self.indicator_table.setCellWidget(row, 1, period_spin)

        # Param2
        param2_item = QTableWidgetItem(param2)
        self.indicator_table.setItem(row, 2, param2_item)

        # Show checkbox
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
        self.symbol_combo.setEditable(True)
        self.symbol_combo.currentTextChanged.connect(self.emit_data_change)
        symbol_layout.addWidget(self.symbol_combo)
        layout.addLayout(symbol_layout)

        # Timeframe
        timeframe_layout = QHBoxLayout()
        timeframe_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1h', '4h', '1D', '1W', '1M'])
        self.timeframe_combo.setCurrentText('1h')
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
        volume_bars = pg.BarGraphItem(x=x_axis, height=volume_data, width=0.8,
                                      brush=pg.mkBrush(100, 149, 237, 100))
        self.volume_plot.addItem(volume_bars)

        # Add trade marks AFTER rendering candlesticks
        self.add_trade_marks(x_axis)

        print(f"📊 Chart rendered with {len(data)} bars")

    def add_trade_marks(self, x_axis):
        """Add entry/exit trade marks overlaid on price chart"""
        if not self.trades_data:
            return

        entry_x, entry_y = [], []
        exit_x, exit_y = [], []

        # Process trades data efficiently
        for trade in self.trades_data:
            try:
                if isinstance(trade, dict):
                    action = trade.get('action', '')
                    price = trade.get('price', 0)
                    datetime_val = trade.get('datetime', None)
                elif hasattr(trade, 'action'):
                    action = trade.action
                    price = trade.price
                    datetime_val = trade.datetime
                else:
                    continue

                if datetime_val and self.data_manager.timestamps:
                    closest_index = self._find_closest_timestamp_index(datetime_val, self.data_manager.timestamps)
                    if closest_index is not None:
                        if action == 'open':
                            entry_x.append(closest_index)
                            entry_y.append(price)
                        elif action == 'close':
                            exit_x.append(closest_index)
                            exit_y.append(price)

            except Exception as e:
                print(f"❌ Error processing trade: {e}")
                continue

        # Add entry marks (green triangles pointing up)
        if entry_x:
            entry_scatter = pg.ScatterPlotItem(
                x=entry_x, y=entry_y,
                symbol='t', size=20, brush=pg.mkBrush(34, 197, 94), pen=pg.mkPen('white', width=2)
            )
            self.price_plot.addItem(entry_scatter)

        # Add exit marks (red triangles pointing down)
        if exit_x:
            exit_scatter = pg.ScatterPlotItem(
                x=exit_x, y=exit_y,
                symbol='t1', size=20, brush=pg.mkBrush(239, 68, 68), pen=pg.mkPen('white', width=2)
            )
            self.price_plot.addItem(exit_scatter)

        print(f"📈 Added {len(entry_x)} entry marks and {len(exit_x)} exit marks to price chart")

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
        """Professional Previous H/L display matching testa.py style"""
        lines = []

        for prev_hl in prev_hl_data:
            if isinstance(prev_hl, dict):
                idx = prev_hl.get('index', 0)
                price = prev_hl.get('price', 0)
                hl_type = prev_hl.get('type', 'PrevHL')

                # Create horizontal line spanning multiple bars
                x_start = max(0, idx - 10)
                x_end = min(data_len - 1, idx + 30)

                lines.append({
                    'x1': x_start,
                    'y1': price,
                    'x2': x_end,
                    'y2': price,
                    'type': hl_type
                })

        if lines:
            # Use custom SMCLineItem with appropriate colors
            color = '#9932CC' if 'High' in str(lines[0].get('type', '')) else '#32CD32'
            smc_item = SMCLineItem(
                lines,
                color=color,
                width=2,
                style='dashed',
                text_labels=[f"P{line.get('type', '')}" for line in lines]
            )
            self.price_plot.addItem(smc_item)
            self.smc_items.append(smc_item)
            print(f"💰 Added {len(lines)} Previous H/L lines")


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

    def display_sessions(self, sessions_data, data_len):
        """Professional Sessions display matching testa.py style"""
        rectangles = []

        # Group sessions by type for better visualization
        session_groups = {}
        for session in sessions_data:
            if isinstance(session, dict):
                session_type = session.get('session', 'Unknown')
                if session_type not in session_groups:
                    session_groups[session_type] = []
                session_groups[session_type].append(session)

        # Create rectangles for each session group
        for session_type, sessions in session_groups.items():
            if not sessions:
                continue

            # Find session boundaries
            start_idx = min(s.get('index', 0) for s in sessions)
            end_idx = max(s.get('index', 0) for s in sessions)

            if end_idx - start_idx < 5:  # Minimum session width
                end_idx = start_idx + 10

            # Get price range for the session
            prices = [s.get('price', 0) for s in sessions if s.get('price', 0) > 0]
            if prices:
                session_high = max(prices) * 1.002  # Slight padding
                session_low = min(prices) * 0.998
            else:
                # Fallback to a default range
                session_high = 100
                session_low = 90

            rectangles.append({
                'x': start_idx,
                'y': session_low,
                'width': end_idx - start_idx,
                'height': session_high - session_low,
                'type': session_type
            })

        if rectangles:
            # Use different colors for different session types
            session_colors = {
                'AsianSession': '#4169E1',
                'LondonSession': '#DC143C',
                'NewYorkSession': '#228B22',
                'Session': '#9932CC'
            }

            # Get color based on session type
            first_type = rectangles[0].get('type', 'Session')
            color = session_colors.get(first_type, '#808080')

            smc_item = SMCRectangleItem(
                rectangles,
                color=color,
                opacity=0.15,
                text_overlay=f"SESSION"
            )
            self.price_plot.addItem(smc_item)
            self.smc_items.append(smc_item)
            print(f"💰 Added {len(rectangles)} Session rectangles")


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