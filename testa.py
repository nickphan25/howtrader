from datetime import datetime, timedelta
import sys
import os

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1.0"

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                               QHBoxLayout, QPushButton, QComboBox, QCheckBox,
                               QGroupBox, QGridLayout, QSpacerItem, QSizePolicy,
                               QMessageBox, QProgressBar, QDateTimeEdit, QSpinBox,
                               QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget)
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QDateTime
from PySide6.QtGui import QFont, QColor
import pyqtgraph as pg
import numpy as np
import pandas as pd

from howtrader.trader.constant import Exchange, Interval
from howtrader.trader.database import get_database
from howtrader.trader.object import BarData
from howtrader.trader.utility import BarGenerator, ArrayManager

# 🔥 ENHANCED: Import additional howtrader components
try:
    import talib

    TALIB_AVAILABLE = True
    print("✅ TA-Lib available for advanced indicators")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available - using basic indicators")

try:
    from smartmoneyconcepts.smc import smc

    SMC_AVAILABLE = True
    print("✅ SMC library imported successfully")
except ImportError as e:
    SMC_AVAILABLE = False
    print(f"❌ SMC library not found: {e}")
    print("SMC features will be disabled")

pg.setConfigOption('useOpenGL', False)
pg.setConfigOption('antialias', True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


# ==================== ORIGINAL CHART COMPONENTS (UNCHANGED) ====================

class TradingViewGrid(pg.GraphicsObject):
    """Dense TradingView style grid background"""

    def __init__(self, x_range, y_range, interval):
        pg.GraphicsObject.__init__(self)
        self.x_range = x_range
        self.y_range = y_range
        self.interval = interval
        self.generatePicture()

    def generatePicture(self):
        """Generate dense TradingView style grid"""
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        p.setRenderHint(pg.QtGui.QPainter.Antialiasing, False)

        # TradingView grid colors
        light_grid_color = pg.QtGui.QColor(240, 243, 250, 80)
        medium_grid_color = pg.QtGui.QColor(226, 232, 240, 120)
        major_grid_color = pg.QtGui.QColor(203, 213, 225, 180)

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        if self.interval in [Interval.MINUTE, Interval.MINUTE_5, Interval.MINUTE_15]:
            x_minor_step = max(1, int((x_max - x_min) / 50))
            x_major_step = max(5, int((x_max - x_min) / 10))
            y_lines = 20
        elif self.interval in [Interval.MINUTE_30, Interval.HOUR]:
            x_minor_step = max(1, int((x_max - x_min) / 40))
            x_major_step = max(4, int((x_max - x_min) / 8))
            y_lines = 15
        else:
            x_minor_step = max(1, int((x_max - x_min) / 30))
            x_major_step = max(3, int((x_max - x_min) / 6))
            y_lines = 12

        # Draw vertical grid lines
        for i in range(int(x_min), int(x_max) + 1, x_minor_step):
            if i % x_major_step == 0:
                pen = pg.mkPen(major_grid_color, width=1.2, style=pg.QtCore.Qt.SolidLine)
            else:
                pen = pg.mkPen(light_grid_color, width=0.8, style=pg.QtCore.Qt.SolidLine)

            p.setPen(pen)
            p.drawLine(pg.QtCore.QPointF(i, y_min), pg.QtCore.QPointF(i, y_max))

        # Draw horizontal grid lines
        y_step = (y_max - y_min) / y_lines
        for i in range(y_lines + 1):
            y = y_min + i * y_step
            if i % 5 == 0:
                pen = pg.mkPen(major_grid_color, width=1.2, style=pg.QtCore.Qt.SolidLine)
            else:
                pen = pg.mkPen(medium_grid_color, width=0.8, style=pg.QtCore.Qt.SolidLine)

            p.setPen(pen)
            p.drawLine(pg.QtCore.QPointF(x_min, y), pg.QtCore.QPointF(x_max, y))

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick item with TradingView colors"""

    def __init__(self, data, timestamps):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.timestamps = timestamps
        self.generatePicture()

    def generatePicture(self):
        """Generate candlestick chart with TradingView colors"""
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        p.setRenderHint(pg.QtGui.QPainter.Antialiasing, True)

        for i, (timestamp, open_price, high_price, low_price, close_price, volume) in enumerate(self.data):
            is_bullish = close_price >= open_price

            if is_bullish:
                pen_color = pg.QtGui.QColor(34, 171, 148)
                fill_color = pg.QtGui.QColor(34, 171, 148)
            else:
                pen_color = pg.QtGui.QColor(240, 67, 56)
                fill_color = pg.QtGui.QColor(240, 67, 56)

            wick_pen = pg.mkPen(pen_color, width=1.5, style=pg.QtCore.Qt.SolidLine)

            if high_price != low_price:
                p.setPen(wick_pen)
                p.drawLine(pg.QtCore.QPointF(i, max(open_price, close_price)),
                           pg.QtCore.QPointF(i, high_price))
                p.drawLine(pg.QtCore.QPointF(i, min(open_price, close_price)),
                           pg.QtCore.QPointF(i, low_price))

            body_height = abs(close_price - open_price)

            if body_height > 0:
                body_top = max(open_price, close_price)
                body_bottom = min(open_price, close_price)

                candle_width = 0.6
                rect = pg.QtCore.QRectF(i - candle_width / 2, body_bottom, candle_width, body_height)

                body_pen = pg.mkPen(pen_color, width=1.8, style=pg.QtCore.Qt.SolidLine)
                p.setPen(body_pen)
                p.setBrush(pg.mkBrush(fill_color))
                p.drawRect(rect)
            else:
                doji_pen = pg.mkPen(pen_color, width=2.5, style=pg.QtCore.Qt.SolidLine)
                p.setPen(doji_pen)
                p.drawLine(pg.QtCore.QPointF(i - 0.3, open_price),
                           pg.QtCore.QPointF(i + 0.3, open_price))

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class VolumeBarItem(pg.GraphicsObject):
    """Volume bars with TradingView colors"""

    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        p.setRenderHint(pg.QtGui.QPainter.Antialiasing, False)

        for i, (timestamp, open_price, high_price, low_price, close_price, volume) in enumerate(self.data):
            if volume <= 0:
                continue

            if close_price >= open_price:
                color = pg.QtGui.QColor(34, 171, 148, 100)
            else:
                color = pg.QtGui.QColor(240, 67, 56, 100)

            p.setPen(pg.mkPen(color, width=0.5))
            p.setBrush(pg.mkBrush(color))

            bar_width = 0.8
            rect = pg.QtCore.QRectF(i - bar_width / 2, 0, bar_width, volume)
            p.drawRect(rect)

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class ImprovedTimeAxis(pg.AxisItem):
    """Improved time axis"""

    def __init__(self, timestamps, interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamps = timestamps
        self.interval = interval
        self.setPen(color='#e1e8ed', width=1)
        self.setTextPen(color='#1e293b')

    def tickStrings(self, values, scale, spacing):
        strings = []

        for v in values:
            try:
                index = int(v)
                if 0 <= index < len(self.timestamps):
                    dt = self.timestamps[index]

                    if self.interval == Interval.MINUTE:
                        if index > 0:
                            prev_dt = self.timestamps[max(0, index - 20)]
                            if dt.date() == prev_dt.date():
                                time_str = dt.strftime('%H:%M')
                            else:
                                time_str = dt.strftime('%m/%d\n%H:%M')
                        else:
                            time_str = dt.strftime('%H:%M')

                    elif self.interval in [Interval.MINUTE_5, Interval.MINUTE_15, Interval.MINUTE_30]:
                        if index > 0:
                            prev_index = max(0, index - 10)
                            prev_dt = self.timestamps[prev_index]
                            if dt.date() != prev_dt.date():
                                time_str = dt.strftime('%m/%d\n%H:%M')
                            else:
                                time_str = dt.strftime('%H:%M')
                        else:
                            time_str = dt.strftime('%H:%M')

                    elif self.interval in [Interval.HOUR, Interval.HOUR_4]:
                        time_str = dt.strftime('%m/%d\n%H:%M')
                    elif self.interval == Interval.DAILY:
                        time_str = dt.strftime('%m/%d')
                    elif self.interval == Interval.WEEKLY:
                        time_str = dt.strftime('%m/%d')
                    elif self.interval == Interval.MONTH:
                        time_str = dt.strftime('%m/%Y')
                    else:
                        time_str = dt.strftime('%m/%d')

                    strings.append(time_str)
                else:
                    strings.append('')
            except (ValueError, IndexError, AttributeError):
                strings.append('')

        return strings


class IndicatorItem(pg.GraphicsObject):
    """Technical indicator display item"""

    def __init__(self, data, color, width=2):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.color = color
        self.width = width
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)

        if len(self.data) < 2:
            p.end()
            return

        pen = pg.mkPen(self.color, width=self.width)
        p.setPen(pen)

        for i in range(len(self.data) - 1):
            if not np.isnan(self.data[i]) and not np.isnan(self.data[i + 1]):
                p.drawLine(
                    pg.QtCore.QPointF(i, self.data[i]),
                    pg.QtCore.QPointF(i + 1, self.data[i + 1])
                )

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


# ==================== SMC COMPONENTS (UNCHANGED) ====================

class SMCRectangleItem(pg.GraphicsObject):
    """SMC rectangle visualization (FVG, Order Blocks)"""

    def __init__(self, rectangles, color, opacity=0.3, text_overlay=None):
        pg.GraphicsObject.__init__(self)
        self.rectangles = rectangles
        self.color = color
        self.opacity = opacity
        self.text_overlay = text_overlay
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)

        fill_color = pg.QtGui.QColor(self.color)
        fill_color.setAlphaF(self.opacity)

        border_color = pg.QtGui.QColor(self.color)
        border_color.setAlphaF(0.8)

        p.setPen(pg.mkPen(border_color, width=1))
        p.setBrush(pg.mkBrush(fill_color))

        for rect_data in self.rectangles:
            if len(rect_data) >= 4:
                x0, y0, x1, y1 = rect_data[:4]
                rect = pg.QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
                p.drawRect(rect)

                if len(rect_data) > 4 and rect_data[4]:
                    text = rect_data[4]
                    center_x = (x0 + x1) / 2
                    center_y = (y0 + y1) / 2

                    p.setPen(pg.mkPen('#ffffff', width=1))
                    p.drawText(pg.QtCore.QPointF(center_x, center_y), text)

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class SMCLineItem(pg.GraphicsObject):
    """SMC line visualization (Liquidity, BOS/CHOCH, Previous H/L)"""

    def __init__(self, lines, color, width=2, style='solid', text_labels=None):
        pg.GraphicsObject.__init__(self)
        self.lines = lines
        self.color = color
        self.width = width
        self.style = style
        self.text_labels = text_labels or []
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)

        pen_style = Qt.PenStyle.SolidLine if self.style == 'solid' else Qt.PenStyle.DashLine
        pen = pg.mkPen(self.color, width=self.width, style=pen_style)
        p.setPen(pen)

        for line_data in self.lines:
            if len(line_data) >= 4:
                x0, y0, x1, y1 = line_data
                p.drawLine(pg.QtCore.QPointF(x0, y0), pg.QtCore.QPointF(x1, y1))

        for label_data in self.text_labels:
            if len(label_data) >= 3:
                x, y, text = label_data
                p.setPen(pg.mkPen(self.color, width=1))
                p.drawText(pg.QtCore.QPointF(x, y), text)

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


# ==================== ENHANCED HOWTRADER COMPONENTS ====================

class HowtraderTimeframeConverter:
    """🔥 ENHANCED: Use BarGenerator for timeframe conversion"""

    def __init__(self):
        self.bar_generators = {}
        self.converted_bars = {}
        self.conversion_callbacks = {}

    def convert_bars(self, bars, target_interval):
        """Convert bars using BarGenerator"""
        if not bars:
            return []

        try:
            # If target is same as source, return original
            if hasattr(bars[0], 'interval') and bars[0].interval == target_interval:
                return bars

            print(f"🔄 Converting {len(bars)} bars to {target_interval.value}")

            # Get conversion parameters
            window_size, use_bar_generator = self._get_conversion_params(target_interval)

            if use_bar_generator:
                return self._convert_with_bar_generator(bars, target_interval, window_size)
            else:
                return self._convert_with_aggregation(bars, target_interval)

        except Exception as e:
            print(f"❌ Conversion error: {e}")
            return bars

    def _get_conversion_params(self, target_interval):
        """Get conversion parameters for target interval"""
        conversion_map = {
            Interval.MINUTE_3: (3, True),
            Interval.MINUTE_5: (5, True),
            Interval.MINUTE_15: (15, True),
            Interval.MINUTE_30: (30, True),
            Interval.HOUR: (60, True),
            Interval.HOUR_2: (120, True),
            Interval.HOUR_4: (240, True),
            Interval.HOUR_6: (360, True),
            Interval.HOUR_8: (480, True),
            Interval.HOUR_12: (720, True),
            Interval.DAILY: (1440, False),
            Interval.WEEKLY: (10080, False),
            Interval.MONTH: (43200, False)
        }

        return conversion_map.get(target_interval, (1, False))

    def _convert_with_bar_generator(self, minute_bars, target_interval, window_size):
        """Convert using BarGenerator for intraday timeframes"""
        converted_bars = []

        def on_bar(bar):
            converted_bars.append(bar)

        def on_window_bar(bar):
            converted_bars.append(bar)

        # Create BarGenerator
        bar_generator = BarGenerator(
            on_bar=on_bar,
            window=window_size,
            on_window_bar=on_window_bar,
            interval=target_interval
        )

        # Process all minute bars
        for bar in minute_bars:
            bar_generator.update_bar(bar)

        print(f"✅ BarGenerator converted to {len(converted_bars)} {target_interval.value} bars")
        return converted_bars

    def _convert_with_aggregation(self, bars, target_interval):
        """Convert using aggregation for daily/weekly/monthly"""
        if target_interval == Interval.DAILY:
            return self._convert_to_daily(bars)
        elif target_interval == Interval.WEEKLY:
            return self._convert_to_weekly(bars)
        elif target_interval == Interval.MONTH:
            return self._convert_to_monthly(bars)
        else:
            return self._simple_aggregation(bars, target_interval)

    def _convert_to_daily(self, bars):
        """Convert to daily bars"""
        if not bars:
            return []

        daily_bars = []
        current_bar = None

        for bar in bars:
            bar_date = bar.datetime.date()

            if current_bar is None or current_bar.datetime.date() != bar_date:
                if current_bar is not None:
                    daily_bars.append(current_bar)
                current_bar = self._create_new_bar(bar, Interval.DAILY)
            else:
                self._update_bar(current_bar, bar)

        if current_bar is not None:
            daily_bars.append(current_bar)

        return daily_bars

    def _convert_to_weekly(self, bars):
        """Convert to weekly bars"""
        if not bars:
            return []

        weekly_bars = []
        current_bar = None

        for bar in bars:
            year, week, _ = bar.datetime.isocalendar()
            week_key = (year, week)

            if current_bar is None or getattr(current_bar, 'week_key', None) != week_key:
                if current_bar:
                    weekly_bars.append(current_bar)

                current_bar = self._create_new_bar(bar, Interval.WEEKLY)
                current_bar.week_key = week_key
            else:
                self._update_bar(current_bar, bar)

        if current_bar:
            weekly_bars.append(current_bar)

        return weekly_bars

    def _convert_to_monthly(self, bars):
        """Convert to monthly bars"""
        if not bars:
            return []

        monthly_bars = []
        current_bar = None

        for bar in bars:
            month_key = (bar.datetime.year, bar.datetime.month)

            if current_bar is None or getattr(current_bar, 'month_key', None) != month_key:
                if current_bar:
                    monthly_bars.append(current_bar)

                current_bar = self._create_new_bar(bar, Interval.MONTH)
                current_bar.month_key = month_key
            else:
                self._update_bar(current_bar, bar)

        if current_bar:
            monthly_bars.append(current_bar)

        return monthly_bars

    def _create_new_bar(self, source_bar, interval):
        """Create new bar from source"""
        new_bar = BarData(
            symbol=source_bar.symbol,
            exchange=source_bar.exchange,
            datetime=source_bar.datetime,
            interval=interval,
            volume=source_bar.volume,
            turnover=getattr(source_bar, 'turnover', 0),
            open_interest=getattr(source_bar, 'open_interest', 0),
            open_price=source_bar.open_price,
            high_price=source_bar.high_price,
            low_price=source_bar.low_price,
            close_price=source_bar.close_price,
            gateway_name=getattr(source_bar, 'gateway_name', '')
        )
        return new_bar

    def _update_bar(self, current_bar, new_bar):
        """Update current bar with new bar data"""
        current_bar.high_price = max(current_bar.high_price, new_bar.high_price)
        current_bar.low_price = min(current_bar.low_price, new_bar.low_price)
        current_bar.close_price = new_bar.close_price
        current_bar.volume += new_bar.volume
        current_bar.turnover += getattr(new_bar, 'turnover', 0)
        current_bar.datetime = new_bar.datetime
        current_bar.open_interest = getattr(new_bar, 'open_interest', 0)


class HowtraderIndicatorCalculator:
    """🔥 ENHANCED: Use ArrayManager + TA-Lib for indicator calculations"""

    def __init__(self, max_size=1000):
        self.array_manager = ArrayManager(size=max_size)
        self.initialized = False

    def update_bars(self, bars):
        """Update ArrayManager with bar data"""
        try:
            # Reset ArrayManager
            self.array_manager = ArrayManager(size=len(bars) + 100)

            # Feed all bars to ArrayManager
            for bar in bars:
                self.array_manager.update_bar(bar)

            self.initialized = self.array_manager.inited
            print(f"✅ ArrayManager updated with {len(bars)} bars, initialized: {self.initialized}")

        except Exception as e:
            print(f"❌ Error updating ArrayManager: {e}")
            self.initialized = False

    def calculate_indicator(self, name, params, data_length):
        """Calculate indicator using ArrayManager + TA-Lib"""
        if not self.initialized:
            print(f"⚠️ ArrayManager not initialized for {name}")
            return np.full(data_length, np.nan)

        try:
            period = params.get('period', 20)

            if name == 'EMA':
                return self._calculate_ema_full(period, data_length)
            elif name == 'SMA':
                return self._calculate_sma_full(period, data_length)
            elif name == 'RSI':
                return self._calculate_rsi_full(period, data_length)
            elif name == 'MACD':
                return self._calculate_macd_full(data_length)
            elif name == 'KDJ':
                return self._calculate_kdj_full(period, data_length)
            elif name == 'Bollinger':
                return self._calculate_bollinger_full(period, data_length)
            elif name == 'ATR':
                return self._calculate_atr_full(period, data_length)
            elif name == 'ADX':
                return self._calculate_adx_full(period, data_length)
            elif name == 'CCI':
                return self._calculate_cci_full(period, data_length)
            elif name == 'Williams':
                return self._calculate_williams_full(period, data_length)
            else:
                print(f"❌ Unknown indicator: {name}")
                return np.full(data_length, np.nan)

        except Exception as e:
            print(f"❌ Error calculating {name}: {e}")
            return np.full(data_length, np.nan)

    def _calculate_ema_full(self, period, data_length):
        """Calculate EMA using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                closes = self.array_manager.close_array[:data_length]
                ema_values = talib.EMA(closes, timeperiod=period)
                return ema_values
            else:
                # Fallback calculation
                return self._manual_ema(period, data_length)
        except:
            return self._manual_ema(period, data_length)

    def _calculate_sma_full(self, period, data_length):
        """Calculate SMA using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                closes = self.array_manager.close_array[:data_length]
                sma_values = talib.SMA(closes, timeperiod=period)
                return sma_values
            else:
                return self._manual_sma(period, data_length)
        except:
            return self._manual_sma(period, data_length)

    def _calculate_rsi_full(self, period, data_length):
        """Calculate RSI using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                closes = self.array_manager.close_array[:data_length]
                rsi_values = talib.RSI(closes, timeperiod=period)
                return rsi_values
            else:
                return self._manual_rsi(period, data_length)
        except:
            return self._manual_rsi(period, data_length)

    def _calculate_macd_full(self, data_length):
        """Calculate MACD using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                closes = self.array_manager.close_array[:data_length]
                macd_line, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
                return macd_line
            else:
                return self._manual_macd(data_length)
        except:
            return self._manual_macd(data_length)

    def _calculate_kdj_full(self, period, data_length):
        """Calculate KDJ using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                highs = self.array_manager.high_array[:data_length]
                lows = self.array_manager.low_array[:data_length]
                closes = self.array_manager.close_array[:data_length]
                k, d = talib.STOCH(highs, lows, closes, fastk_period=period, slowk_period=3, slowd_period=3)
                return k
            else:
                return self._manual_kdj(period, data_length)
        except:
            return self._manual_kdj(period, data_length)

    def _calculate_bollinger_full(self, period, data_length):
        """Calculate Bollinger Bands using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                closes = self.array_manager.close_array[:data_length]
                upper, middle, lower = talib.BBANDS(closes, timeperiod=period, nbdevup=2, nbdevdn=2)
                return upper
            else:
                return self._manual_bollinger(period, data_length)
        except:
            return self._manual_bollinger(period, data_length)

    def _calculate_atr_full(self, period, data_length):
        """Calculate ATR using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                highs = self.array_manager.high_array[:data_length]
                lows = self.array_manager.low_array[:data_length]
                closes = self.array_manager.close_array[:data_length]
                atr_values = talib.ATR(highs, lows, closes, timeperiod=period)
                return atr_values
            else:
                # Use ArrayManager's built-in ATR
                atr_values = np.full(data_length, np.nan)
                for i in range(period - 1, data_length):
                    atr_values[i] = self.array_manager.atr(period)
                return atr_values
        except:
            return np.full(data_length, np.nan)

    def _calculate_adx_full(self, period, data_length):
        """Calculate ADX using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                highs = self.array_manager.high_array[:data_length]
                lows = self.array_manager.low_array[:data_length]
                closes = self.array_manager.close_array[:data_length]
                adx_values = talib.ADX(highs, lows, closes, timeperiod=period)
                return adx_values
            else:
                return np.full(data_length, np.nan)
        except:
            return np.full(data_length, np.nan)

    def _calculate_cci_full(self, period, data_length):
        """Calculate CCI using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                highs = self.array_manager.high_array[:data_length]
                lows = self.array_manager.low_array[:data_length]
                closes = self.array_manager.close_array[:data_length]
                cci_values = talib.CCI(highs, lows, closes, timeperiod=period)
                return cci_values
            else:
                return np.full(data_length, np.nan)
        except:
            return np.full(data_length, np.nan)

    def _calculate_williams_full(self, period, data_length):
        """Calculate Williams %R using ArrayManager"""
        try:
            if TALIB_AVAILABLE:
                highs = self.array_manager.high_array[:data_length]
                lows = self.array_manager.low_array[:data_length]
                closes = self.array_manager.close_array[:data_length]
                williams_values = talib.WILLR(highs, lows, closes, timeperiod=period)
                return williams_values
            else:
                return np.full(data_length, np.nan)
        except:
            return np.full(data_length, np.nan)

    # Fallback manual calculations (simplified versions)
    def _manual_ema(self, period, data_length):
        """Manual EMA calculation"""
        closes = self.array_manager.close_array[:data_length]
        ema_values = np.full(data_length, np.nan)

        if data_length >= period:
            alpha = 2.0 / (period + 1)
            ema_values[period - 1] = np.mean(closes[:period])

            for i in range(period, data_length):
                ema_values[i] = alpha * closes[i] + (1 - alpha) * ema_values[i - 1]

        return ema_values

    def _manual_sma(self, period, data_length):
        """Manual SMA calculation"""
        closes = self.array_manager.close_array[:data_length]
        sma_values = np.full(data_length, np.nan)

        for i in range(period - 1, data_length):
            sma_values[i] = np.mean(closes[i - period + 1:i + 1])

        return sma_values

    def _manual_rsi(self, period, data_length):
        """Manual RSI calculation"""
        closes = self.array_manager.close_array[:data_length]
        rsi_values = np.full(data_length, np.nan)

        if data_length >= period + 1:
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values[period] = 100 - (100 / (1 + rs))

            for i in range(period + 1, data_length):
                avg_gain = ((avg_gain * (period - 1)) + gains[i - 1]) / period
                avg_loss = ((avg_loss * (period - 1)) + losses[i - 1]) / period

                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi_values[i] = 100 - (100 / (1 + rs))

        return rsi_values

    def _manual_macd(self, data_length):
        """Manual MACD calculation"""
        closes = self.array_manager.close_array[:data_length]

        ema12 = np.full(data_length, np.nan)
        ema26 = np.full(data_length, np.nan)

        alpha12 = 2.0 / 13
        alpha26 = 2.0 / 27

        if data_length >= 12:
            ema12[11] = np.mean(closes[:12])
            for i in range(12, data_length):
                ema12[i] = alpha12 * closes[i] + (1 - alpha12) * ema12[i - 1]

        if data_length >= 26:
            ema26[25] = np.mean(closes[:26])
            for i in range(26, data_length):
                ema26[i] = alpha26 * closes[i] + (1 - alpha26) * ema26[i - 1]

        return ema12 - ema26

    def _manual_kdj(self, period, data_length):
        """Manual KDJ calculation"""
        highs = self.array_manager.high_array[:data_length]
        lows = self.array_manager.low_array[:data_length]
        closes = self.array_manager.close_array[:data_length]

        k_values = np.full(data_length, np.nan)

        for i in range(period - 1, data_length):
            highest_high = np.max(highs[i - period + 1:i + 1])
            lowest_low = np.min(lows[i - period + 1:i + 1])

            if highest_high != lowest_low:
                k_values[i] = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k_values[i] = 50

        return k_values

    def _manual_bollinger(self, period, data_length):
        """Manual Bollinger Bands calculation"""
        closes = self.array_manager.close_array[:data_length]
        upper_band = np.full(data_length, np.nan)

        for i in range(period - 1, data_length):
            window = closes[i - period + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window)
            upper_band[i] = sma + (2 * std)

        return upper_band


class HowtraderDataLoader(QThread):
    """🔥 ENHANCED: Optimized data loader with howtrader database optimization"""

    data_loaded = Signal(list)
    error_occurred = Signal(str)
    progress_updated = Signal(int)

    def __init__(self, symbol, exchange, interval, start_time, end_time):
        super().__init__()
        self.symbol = symbol
        self.exchange = exchange
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.converter = HowtraderTimeframeConverter()

    def run(self):
        try:
            self.progress_updated.emit(10)

            # 🔥 ENHANCED: Optimized database connection
            database = get_database()
            self.progress_updated.emit(30)

            print(f"🔍 Loading data for {self.symbol} from {self.exchange}")
            print(f"   Timeframe: {self.interval}")
            print(f"   Period: {self.start_time} to {self.end_time}")

            # Always load minute data first for maximum conversion flexibility
            bars = database.load_bar_data(
                self.symbol,
                self.exchange,
                Interval.MINUTE,
                start=self.start_time,
                end=self.end_time
            )
            self.progress_updated.emit(50)

            if bars and len(bars) > 0:
                print(f"✅ Found {len(bars)} minute bars")

                # 🔥 ENHANCED: Use HowtraderTimeframeConverter
                if self.interval != Interval.MINUTE:
                    converted_bars = self.converter.convert_bars(bars, self.interval)
                    if converted_bars:
                        bars = converted_bars
                        print(f"✅ Converted to {len(bars)} {self.interval.value} bars")
                    else:
                        print("⚠️ Conversion failed, using original minute data")

            self.progress_updated.emit(90)

            if bars and len(bars) > 0:
                print(f"✅ Final result: {len(bars)} bars")
                print(f"   Time range: {bars[0].datetime} to {bars[-1].datetime}")
                self.data_loaded.emit(bars)
            else:
                error_msg = self._generate_detailed_error_message()
                self.error_occurred.emit(error_msg)

            self.progress_updated.emit(100)

        except Exception as e:
            error_msg = f"Database error: {str(e)}\n\nPlease check your database connection and data availability."
            self.error_occurred.emit(error_msg)

    def _generate_detailed_error_message(self):
        """Generate detailed error message with troubleshooting tips"""
        return (
            f"❌ No data found for {self.symbol} on {self.exchange}\n\n"
            "🔧 Troubleshooting steps:\n"
            "• Verify symbol exists in your database\n"
            "• Check if data collection is running for this exchange\n"
            "• Try a different date range\n"
            "• Ensure exchange name matches your database\n"
            "• Check database connection status\n\n"
            "📊 Supported exchanges in howtrader:\n"
            "• BINANCE, HUOBI, OKX, BYBIT, etc.\n\n"
            "💡 Pro tip: Start data collection first, then analyze!"
        )


# ==================== ENHANCED UI PANELS ====================

class TechnicalIndicatorPanel(QWidget):
    """🔥 ENHANCED: Technical indicator panel with more howtrader indicators"""

    indicator_changed = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.indicators = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("📈 Technical Indicators")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        title.setStyleSheet("color: #1e293b; padding: 5px;")
        layout.addWidget(title)

        # 🔥 ENHANCED: Show TA-Lib status
        if TALIB_AVAILABLE:
            talib_status = QLabel("✅ TA-Lib Enhanced Mode")
            talib_status.setStyleSheet("color: #059669; font-size: 10px; padding: 2px;")
        else:
            talib_status = QLabel("⚠️ Basic Mode (TA-Lib not available)")
            talib_status.setStyleSheet("color: #dc2626; font-size: 10px; padding: 2px;")

        layout.addWidget(talib_status)

        self.indicator_table = QTableWidget()
        self.indicator_table.setColumnCount(4)
        self.indicator_table.setHorizontalHeaderLabels(['Indicator', 'Period', 'Param2', 'Show'])

        header = self.indicator_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.indicator_table.setColumnWidth(1, 60)
        self.indicator_table.setColumnWidth(2, 60)
        self.indicator_table.setColumnWidth(3, 50)

        # 🔥 ENHANCED: Add more indicators with TA-Lib support
        self.add_indicator_row("EMA", 20, "", True)
        self.add_indicator_row("RSI", 14, "", False)
        self.add_indicator_row("KDJ", 9, "3", False)

        # Add new howtrader indicators
        if TALIB_AVAILABLE:
            self.add_indicator_row("ATR", 14, "", False)
            self.add_indicator_row("ADX", 14, "", False)
            self.add_indicator_row("CCI", 20, "", False)
            self.add_indicator_row("Williams", 14, "", False)

        layout.addWidget(self.indicator_table)

        button_layout = QHBoxLayout()

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_custom_indicator)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #1d4ed8; }
        """)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_indicator)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #b91c1c; }
        """)

        button_layout.addWidget(add_btn)
        button_layout.addWidget(remove_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.indicator_table.itemChanged.connect(self.on_item_changed)

    def add_indicator_row(self, name, period, param2, show):
        row = self.indicator_table.rowCount()
        self.indicator_table.insertRow(row)

        indicator_combo = QComboBox()

        # 🔥 ENHANCED: More indicators
        standard_indicators = ['EMA', 'SMA', 'RSI', 'MACD', 'KDJ', 'Bollinger']

        if TALIB_AVAILABLE:
            talib_indicators = ['ATR', 'ADX', 'CCI', 'Williams']
            standard_indicators.extend(talib_indicators)

        # Add SMC indicators if available
        if SMC_AVAILABLE:
            smc_indicators = ['FVG', 'SwingHL', 'BOS_CHOCH', 'OrderBlocks', 'Liquidity', 'PrevHL', 'Sessions',
                              'Retracements']
            indicator_combo.addItems(standard_indicators + smc_indicators)
        else:
            indicator_combo.addItems(standard_indicators)

        indicator_combo.setCurrentText(name)
        indicator_combo.currentTextChanged.connect(lambda: self.emit_indicator_change())
        self.indicator_table.setCellWidget(row, 0, indicator_combo)

        period_spin = QSpinBox()
        period_spin.setRange(1, 200)
        period_spin.setValue(period)
        period_spin.valueChanged.connect(lambda: self.emit_indicator_change())
        self.indicator_table.setCellWidget(row, 1, period_spin)

        if param2:
            param2_spin = QSpinBox()
            param2_spin.setRange(1, 50)
            param2_spin.setValue(int(param2))
            param2_spin.valueChanged.connect(lambda: self.emit_indicator_change())
            self.indicator_table.setCellWidget(row, 2, param2_spin)
        else:
            self.indicator_table.setItem(row, 2, QTableWidgetItem(""))

        show_check = QCheckBox()
        show_check.setChecked(show)
        show_check.stateChanged.connect(lambda: self.emit_indicator_change())
        self.indicator_table.setCellWidget(row, 3, show_check)

    def add_custom_indicator(self):
        default_indicator = "ATR" if TALIB_AVAILABLE else "EMA"
        self.add_indicator_row(default_indicator, 20, "", False)

    def remove_indicator(self):
        current_row = self.indicator_table.currentRow()
        if current_row >= 0:
            self.indicator_table.removeRow(current_row)
            self.emit_indicator_change()

    def on_item_changed(self, item):
        self.emit_indicator_change()

    def emit_indicator_change(self):
        self.get_active_indicators()

    def get_active_indicators(self):
        active_indicators = []

        for row in range(self.indicator_table.rowCount()):
            indicator_combo = self.indicator_table.cellWidget(row, 0)
            period_spin = self.indicator_table.cellWidget(row, 1)
            param2_widget = self.indicator_table.cellWidget(row, 2)
            show_check = self.indicator_table.cellWidget(row, 3)

            if show_check.isChecked():
                params = {
                    'period': period_spin.value()
                }

                if param2_widget and hasattr(param2_widget, 'value'):
                    params['param2'] = param2_widget.value()

                active_indicators.append({
                    'name': indicator_combo.currentText(),
                    'params': params
                })

                self.indicator_changed.emit(indicator_combo.currentText(), params)

        return active_indicators


class SMCFeaturePanel(QWidget):
    """SMC Feature configuration panel (UNCHANGED)"""

    smc_feature_changed = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.smc_features = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("💡 Smart Money Concepts")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        title.setStyleSheet("color: #1e293b; padding: 5px;")
        layout.addWidget(title)

        if not SMC_AVAILABLE:
            warning = QLabel("⚠️ SMC library not available\nPlease install smartmoneyconcepts")
            warning.setStyleSheet(
                "color: #dc2626; font-size: 11px; padding: 10px; background: #fef2f2; border-radius: 4px;")
            layout.addWidget(warning)
            self.setLayout(layout)
            return

        # SMC Feature table
        self.smc_table = QTableWidget()
        self.smc_table.setColumnCount(4)
        self.smc_table.setHorizontalHeaderLabels(['Feature', 'Param1', 'Param2', 'Show'])

        header = self.smc_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.smc_table.setColumnWidth(1, 60)
        self.smc_table.setColumnWidth(2, 60)
        self.smc_table.setColumnWidth(3, 50)

        # Add SMC features
        self.add_smc_row("FVG", "", "", False)
        self.add_smc_row("SwingHL", "50", "", False)
        self.add_smc_row("BOS_CHOCH", "", "", False)
        self.add_smc_row("OrderBlocks", "", "", False)
        self.add_smc_row("Liquidity", "1", "", False)

        layout.addWidget(self.smc_table)

        # Buttons
        button_layout = QHBoxLayout()

        add_btn = QPushButton("Add SMC")
        add_btn.clicked.connect(self.add_custom_smc)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #7c3aed;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #6d28d9; }
        """)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_smc)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #b91c1c; }
        """)

        button_layout.addWidget(add_btn)
        button_layout.addWidget(remove_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.smc_table.itemChanged.connect(self.on_smc_changed)

    def add_smc_row(self, name, param1, param2, show):
        row = self.smc_table.rowCount()
        self.smc_table.insertRow(row)

        # Feature combo
        feature_combo = QComboBox()
        smc_features = ['FVG', 'SwingHL', 'BOS_CHOCH', 'OrderBlocks', 'Liquidity', 'PrevHL', 'Sessions', 'Retracements']
        feature_combo.addItems(smc_features)
        feature_combo.setCurrentText(name)
        feature_combo.currentTextChanged.connect(lambda: self.emit_smc_change())
        self.smc_table.setCellWidget(row, 0, feature_combo)

        # Parameters
        if param1:
            param1_spin = QSpinBox()
            param1_spin.setRange(1, 200)
            param1_spin.setValue(int(float(param1)))
            param1_spin.valueChanged.connect(lambda: self.emit_smc_change())
            self.smc_table.setCellWidget(row, 1, param1_spin)
        else:
            self.smc_table.setItem(row, 1, QTableWidgetItem(""))

        if param2:
            param2_spin = QSpinBox()
            param2_spin.setRange(1, 50)
            param2_spin.setValue(int(param2))
            param2_spin.valueChanged.connect(lambda: self.emit_smc_change())
            self.smc_table.setCellWidget(row, 2, param2_spin)
        else:
            self.smc_table.setItem(row, 2, QTableWidgetItem(""))

        # Show checkbox
        show_check = QCheckBox()
        show_check.setChecked(show)
        show_check.stateChanged.connect(lambda: self.emit_smc_change())
        self.smc_table.setCellWidget(row, 3, show_check)

    def add_custom_smc(self):
        self.add_smc_row("FVG", "", "", False)

    def remove_smc(self):
        current_row = self.smc_table.currentRow()
        if current_row >= 0:
            self.smc_table.removeRow(current_row)
            self.emit_smc_change()

    def on_smc_changed(self, item):
        self.emit_smc_change()

    def emit_smc_change(self):
        self.get_active_smc_features()

    def get_active_smc_features(self):
        active_features = []

        if not SMC_AVAILABLE:
            return active_features

        for row in range(self.smc_table.rowCount()):
            feature_combo = self.smc_table.cellWidget(row, 0)
            param1_widget = self.smc_table.cellWidget(row, 1)
            param2_widget = self.smc_table.cellWidget(row, 2)
            show_check = self.smc_table.cellWidget(row, 3)

            if show_check.isChecked():
                params = {}

                if param1_widget and hasattr(param1_widget, 'value'):
                    params['param1'] = param1_widget.value()

                if param2_widget and hasattr(param2_widget, 'value'):
                    params['param2'] = param2_widget.value()

                active_features.append({
                    'name': feature_combo.currentText(),
                    'params': params
                })

                self.smc_feature_changed.emit(feature_combo.currentText(), params)

        return active_features


# ==================== ENHANCED MAIN CHART CLASS ====================

class AdvancedHowtraderChart(QWidget):
    """🔥 ENHANCED: Professional trading platform with howtrader integration"""

    def __init__(self):
        super().__init__()
        self.bars = None
        self.data_loader = None

        # 🔥 ENHANCED: Initialize howtrader components
        self.indicator_calculator = HowtraderIndicatorCalculator()
        self.timeframe_converter = HowtraderTimeframeConverter()

        # Keep original components
        self.price_indicator_items = {}
        self.oscillator_indicator_items = {}
        self.smc_items = {}
        self.df_ohlc = None
        self.timestamps = []

        # Enhanced properties
        self.backtest_mode = False
        self.backtest_data_ready = False

        self.initUI()

    def initUI(self):
        """Keep original UI layout (UNCHANGED)"""
        self.setWindowTitle("📊 Professional Trading Platform with Smart Money Concepts")
        self.setGeometry(50, 50, 1900, 1000)

        main_layout = QHBoxLayout()

        # Left panel (UNCHANGED)
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_panel.setStyleSheet("""
            background-color: #f8fafc; 
            border-right: 2px solid #e2e8f0;
            border-radius: 8px;
        """)

        left_layout = QVBoxLayout()

        # Title (UNCHANGED)
        title_widget = QWidget()
        title_widget.setFixedHeight(60)
        title_widget.setStyleSheet("""
            background-color: #1e293b;
            border-radius: 8px;
            margin-bottom: 10px;
        """)
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(15, 10, 15, 10)

        title = QLabel("📊 Pro Trading Platform")
        title.setFont(QFont("Arial", 15, QFont.Bold))
        title.setStyleSheet("color: white; text-align: center;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(title)

        title_widget.setLayout(title_layout)
        left_layout.addWidget(title_widget)

        # Exchange selection (UNCHANGED)
        exchange_group = QGroupBox("Exchange")
        exchange_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                color: #1e293b; 
                font-size: 13px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        exchange_layout = QVBoxLayout()

        self.exchange_combo = QComboBox()
        self.exchange_combo.addItem("Binance", Exchange.BINANCE)
        self.exchange_combo.addItem("Huobi", Exchange.HUOBI)
        self.exchange_combo.addItem("OKX", Exchange.OKX)

        exchange_layout.addWidget(self.exchange_combo)
        exchange_group.setLayout(exchange_layout)
        left_layout.addWidget(exchange_group)

        # Symbol and timeframe (UNCHANGED)
        symbol_group = QGroupBox("Symbol & Timeframe")
        symbol_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                color: #1e293b; 
                font-size: 13px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        symbol_layout = QGridLayout()

        symbol_layout.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT"])
        symbol_layout.addWidget(self.symbol_combo, 0, 1)

        symbol_layout.addWidget(QLabel("Timeframe:"), 1, 0)
        self.timeframe_combo = QComboBox()

        # 🔥 ENHANCED: More timeframe options
        timeframe_options = [
            ("1 Minute", Interval.MINUTE),
            ("3 Minutes", Interval.MINUTE_3),
            ("5 Minutes", Interval.MINUTE_5),
            ("15 Minutes", Interval.MINUTE_15),
            ("30 Minutes", Interval.MINUTE_30),
            ("1 Hour", Interval.HOUR),
            ("2 Hours", Interval.HOUR_2),
            ("4 Hours", Interval.HOUR_4),
            ("6 Hours", Interval.HOUR_6),
            ("8 Hours", Interval.HOUR_8),
            ("12 Hours", Interval.HOUR_12),
            ("1 Day", Interval.DAILY),
            ("1 Week", Interval.WEEKLY),
            ("1 Month", Interval.MONTH)
        ]

        for text, interval in timeframe_options:
            self.timeframe_combo.addItem(text, interval)

        symbol_layout.addWidget(self.timeframe_combo, 1, 1)
        symbol_group.setLayout(symbol_layout)
        left_layout.addWidget(symbol_group)

        # Date range (UNCHANGED)
        date_group = QGroupBox("Date Range")
        date_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                color: #1e293b; 
                font-size: 13px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        date_layout = QGridLayout()

        date_layout.addWidget(QLabel("Start:"), 0, 0)
        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setDateTime(QDateTime.currentDateTime().addDays(-30))
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.setStyleSheet("font-size: 11px;")
        date_layout.addWidget(self.start_datetime, 0, 1)

        date_layout.addWidget(QLabel("End:"), 1, 0)
        self.end_datetime = QDateTimeEdit()
        self.end_datetime.setDateTime(QDateTime.currentDateTime())
        self.end_datetime.setCalendarPopup(True)
        self.end_datetime.setStyleSheet("font-size: 11px;")
        date_layout.addWidget(self.end_datetime, 1, 1)

        date_group.setLayout(date_layout)
        left_layout.addWidget(date_group)

        # Load button (UNCHANGED)
        self.load_btn = QPushButton("📊 Load Chart Data")
        self.load_btn.clicked.connect(self.load_data_async)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #047857; }
            QPushButton:disabled { background-color: #94a3b8; }
        """)
        left_layout.addWidget(self.load_btn)

        # Progress bar (UNCHANGED)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Chart info (UNCHANGED)
        info_group = QGroupBox("Chart Information")
        info_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                color: #1e293b; 
                font-size: 13px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        info_layout = QVBoxLayout()

        self.symbol_label = QLabel("Symbol: --")
        self.timeframe_label = QLabel("Timeframe: --")
        self.data_range_label = QLabel("Data Range: --")
        self.current_price_label = QLabel("Current Price: --")

        for label in [self.symbol_label, self.timeframe_label, self.data_range_label, self.current_price_label]:
            label.setStyleSheet("padding: 2px; font-size: 11px; color: #475569;")
            info_layout.addWidget(label)

        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        # 🔥 ENHANCED: Tabbed panels with status
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #f1f5f9;
                color: #475569;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #2563eb;
                color: white;
            }
        """)

        # Technical indicators tab (ENHANCED)
        self.indicator_panel = TechnicalIndicatorPanel()
        self.indicator_panel.indicator_changed.connect(self.update_indicators)
        tab_widget.addTab(self.indicator_panel, "📈 Indicators")

        # SMC features tab (UNCHANGED)
        self.smc_panel = SMCFeaturePanel()
        self.smc_panel.smc_feature_changed.connect(self.update_smc_features)
        tab_widget.addTab(self.smc_panel, "💡 SMC")

        left_layout.addWidget(tab_widget)

        # Status (ENHANCED)
        self.status_label = QLabel("🚀 Ready to load data")
        self.status_label.setStyleSheet("color: #059669; padding: 5px; font-weight: bold; font-size: 12px;")
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # Right panel for charts (UNCHANGED)
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        self.create_professional_charts(right_layout)

        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel, 1)

        self.setLayout(main_layout)

    def create_professional_charts(self, layout):
        """Keep original chart layout (UNCHANGED)"""
        # Price chart
        self.price_widget = pg.PlotWidget()
        self.price_widget.setLabel('left', 'Price (USDT)', color='#1e293b', size='14pt')
        self.price_widget.setMinimumHeight(400)
        self.price_widget.setBackground('#ffffff')

        self.price_widget.getAxis('left').setPen('#e1e8ed')
        self.price_widget.getAxis('left').setTextPen('#1e293b')
        self.price_widget.getAxis('bottom').setPen('#e1e8ed')
        self.price_widget.getAxis('bottom').setTextPen('#1e293b')

        self.price_widget.showGrid(x=False, y=False)

        layout.addWidget(self.price_widget)

        # Oscillator panel
        self.oscillator_widget = pg.PlotWidget()
        self.oscillator_widget.setLabel('left', 'Oscillators', color='#1e293b', size='12pt')
        self.oscillator_widget.setFixedHeight(150)
        self.oscillator_widget.setBackground('#ffffff')

        self.oscillator_widget.getAxis('left').setPen('#e1e8ed')
        self.oscillator_widget.getAxis('left').setTextPen('#1e293b')
        self.oscillator_widget.getAxis('bottom').setPen('#e1e8ed')
        self.oscillator_widget.getAxis('bottom').setTextPen('#1e293b')

        self.oscillator_widget.showGrid(x=False, y=False)
        self.oscillator_widget.setXLink(self.price_widget)

        # Reference lines
        self.oscillator_widget.addLine(y=50, pen=pg.mkPen('#94a3b8', width=1, style=Qt.PenStyle.DashLine))
        self.oscillator_widget.addLine(y=30, pen=pg.mkPen('#dc2626', width=1, style=Qt.PenStyle.DashLine))
        self.oscillator_widget.addLine(y=70, pen=pg.mkPen('#dc2626', width=1, style=Qt.PenStyle.DashLine))

        layout.addWidget(self.oscillator_widget)

        # Volume chart
        self.volume_widget = pg.PlotWidget()
        self.volume_widget.setLabel('left', 'Volume', color='#1e293b', size='12pt')
        self.volume_widget.setFixedHeight(120)
        self.volume_widget.setBackground('#ffffff')

        self.volume_widget.getAxis('left').setPen('#e1e8ed')
        self.volume_widget.getAxis('left').setTextPen('#1e293b')
        self.volume_widget.getAxis('bottom').setPen('#e1e8ed')
        self.volume_widget.getAxis('bottom').setTextPen('#1e293b')

        self.volume_widget.showGrid(x=False, y=False)
        self.volume_widget.setXLink(self.price_widget)

        layout.addWidget(self.volume_widget)

    def setup_backtest_mode(self):
        """Setup widget for backtest mode (ENHANCED)"""
        try:
            print("🔧 Setting up backtest mode...")

            # Keep original backtest setup logic
            if hasattr(self, 'symbol_combo'):
                self.symbol_combo.setEnabled(False)
                symbol = self.backtest_params['symbol'].split('.')[0]
                self.symbol_combo.setCurrentText(symbol)

            if hasattr(self, 'exchange_combo'):
                self.exchange_combo.setEnabled(False)
                exchange = self.backtest_params['symbol'].split('.')[1]
                self.exchange_combo.setCurrentText(exchange)

            if hasattr(self, 'start_datetime'):
                self.start_datetime.setEnabled(False)
                self.start_datetime.setDateTime(self.backtest_params['start'])

            if hasattr(self, 'end_datetime'):
                self.end_datetime.setEnabled(False)
                self.end_datetime.setDateTime(self.backtest_params['end'])

            if hasattr(self, 'load_btn'):
                self.load_btn.setVisible(False)

            # Update timeframe combo with supported intervals
            if hasattr(self, 'timeframe_combo'):
                self.timeframe_combo.clear()
                supported_timeframes = [
                    (Interval.MINUTE, "1m"),
                    (Interval.MINUTE_3, "3m"),
                    (Interval.MINUTE_5, "5m"),
                    (Interval.MINUTE_15, "15m"),
                    (Interval.MINUTE_30, "30m"),
                    (Interval.HOUR, "1h"),
                    (Interval.HOUR_2, "2h"),
                    (Interval.HOUR_4, "4h"),
                    (Interval.HOUR_6, "6h"),
                    (Interval.HOUR_8, "8h"),
                    (Interval.HOUR_12, "12h"),
                    (Interval.DAILY, "1d"),
                    (Interval.WEEKLY, "1w"),
                    (Interval.MONTH, "1M")
                ]

                for interval, label in supported_timeframes:
                    self.timeframe_combo.addItem(label, interval)

                # Set current timeframe
                current_label = self.backtest_params['interval'].value
                index = self.timeframe_combo.findText(current_label)
                if index >= 0:
                    self.timeframe_combo.setCurrentIndex(index)

                # Connect timeframe change
                self.timeframe_combo.currentIndexChanged.connect(self.on_backtest_timeframe_changed)

            # Update window title
            symbol = self.backtest_params['symbol']
            start_str = self.backtest_params['start'].strftime('%Y-%m-%d')
            end_str = self.backtest_params['end'].strftime('%Y-%m-%d')
            self.setWindowTitle(f"🎯 Backtest Results - {symbol} ({start_str} to {end_str})")

            print("✅ Backtest mode setup completed")

        except Exception as e:
            print(f"❌ Error setting up backtest mode: {e}")

    def on_backtest_timeframe_changed(self):
        """🔥 ENHANCED: Handle timeframe change using HowtraderTimeframeConverter"""
        try:
            if not hasattr(self, 'base_bars') or not self.base_bars:
                print("❌ No base bars available")
                return

            new_interval = self.timeframe_combo.currentData()
            if not new_interval:
                print("❌ No interval selected")
                return

            print(f"🔄 Converting timeframe to: {new_interval.value}")

            # 🔥 ENHANCED: Use HowtraderTimeframeConverter
            converted_bars = self.timeframe_converter.convert_bars(self.base_bars, new_interval)

            if converted_bars:
                # Update current data
                self.bars = converted_bars
                self.timestamps = [bar.datetime for bar in converted_bars]
                self.current_timeframe = new_interval

                # 🔥 ENHANCED: Update indicator calculator with new bars
                self.indicator_calculator.update_bars(converted_bars)

                # Convert bars to OHLC format for display
                self._update_ohlc_data()

                # Recalculate trade markers for new timeframe
                self._update_trade_markers()

                # Clear and refresh chart completely
                self.clear_chart()
                self.display_professional_chart()

                print(f"✅ Timeframe converted: {len(converted_bars)} bars")
            else:
                print("❌ Failed to convert timeframe")

        except Exception as e:
            print(f"❌ Error changing timeframe: {e}")
            import traceback
            traceback.print_exc()

    def load_data_async(self):
        """🔥 ENHANCED: Load data with HowtraderDataLoader"""
        try:
            # Check if we're in backtest mode
            if hasattr(self, 'backtest_mode') and self.backtest_mode:
                print("⚠️ Skipping data load - in backtest mode")
                return

            if hasattr(self, 'backtest_data_ready') and self.backtest_data_ready:
                print("⚠️ Skipping data load - backtest data already loaded")
                return

            # Check if data loader is already running
            if self.data_loader and self.data_loader.isRunning():
                return

            symbol = self.symbol_combo.currentText()
            interval = self.timeframe_combo.currentData()
            exchange = self.exchange_combo.currentData()
            start_time = self.start_datetime.dateTime().toPython()
            end_time = self.end_datetime.dateTime().toPython()

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.load_btn.setEnabled(False)
            self.status_label.setText(f"🔄 Loading {symbol} from {self.exchange_combo.currentText()}...")

            # 🔥 ENHANCED: Use HowtraderDataLoader
            self.data_loader = HowtraderDataLoader(symbol, exchange, interval, start_time, end_time)
            self.data_loader.data_loaded.connect(self.on_data_loaded)
            self.data_loader.error_occurred.connect(self.on_data_error)
            self.data_loader.progress_updated.connect(self.progress_bar.setValue)
            self.data_loader.start()

        except Exception as e:
            print(f"❌ Error in load_data_async: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"❌ Error loading data: {e}")
            if hasattr(self, 'load_btn'):
                self.load_btn.setEnabled(True)
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(False)

    def on_data_loaded(self, bars):
        """🔥 ENHANCED: Handle loaded data with HowtraderIndicatorCalculator"""
        try:
            self.bars = bars
            print(f"📊 Displaying {len(bars)} bars")

            # Store timestamps
            self.timestamps = [bar.datetime for bar in bars]

            # 🔥 ENHANCED: Update HowtraderIndicatorCalculator with ALL data
            self.indicator_calculator.update_bars(bars)

            # Prepare DataFrame for SMC calculations
            self.prepare_smc_dataframe()

            self.display_professional_chart()
            self.update_chart_info()
            self.update_all_indicators()
            self.update_all_smc_features()

            self.status_label.setText(f"✅ Loaded {len(bars)} bars successfully")

        except Exception as e:
            self.on_data_error(f"Display error: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.load_btn.setEnabled(True)

    def update_indicators(self, indicator_name, params):
        """🔥 ENHANCED: Update indicators using HowtraderIndicatorCalculator"""
        if not self.bars:
            return

        try:
            data_len = len(self.bars)
            print(f"🔄 Calculating {indicator_name} for {data_len} bars...")

            # 🔥 ENHANCED: Use HowtraderIndicatorCalculator
            indicator_data = self.indicator_calculator.calculate_indicator(indicator_name, params, data_len)

            if indicator_data is not None:
                self.display_indicator(indicator_name, indicator_data, params)
                print(f"✅ Updated {indicator_name} with {len(indicator_data)} values")
            else:
                print(f"⚠️ Failed to calculate {indicator_name}")

        except Exception as e:
            print(f"❌ Error updating indicator {indicator_name}: {e}")

    def display_indicator(self, name, data, params):
        """Keep original indicator display logic (UNCHANGED)"""
        oscillator_indicators = ['RSI', 'KDJ', 'MACD', 'CCI', 'Williams', 'ADX']

        if name in oscillator_indicators:
            if name in self.oscillator_indicator_items:
                self.oscillator_widget.removeItem(self.oscillator_indicator_items[name])

            # 🔥 ENHANCED: More color options
            colors = {
                'RSI': '#7c3aed',
                'KDJ': '#ea580c',
                'MACD': '#059669',
                'CCI': '#dc2626',
                'Williams': '#8b5cf6',
                'ADX': '#f59e0b'
            }
            color = colors.get(name, '#1e293b')

            indicator_item = IndicatorItem(data, color, width=2)
            self.oscillator_widget.addItem(indicator_item)
            self.oscillator_indicator_items[name] = indicator_item

        else:
            if name in self.price_indicator_items:
                self.price_widget.removeItem(self.price_indicator_items[name])

            colors = {
                'EMA': '#2563eb',
                'SMA': '#dc2626',
                'Bollinger': '#8b5cf6',
                'ATR': '#f59e0b'
            }

            color = colors.get(name, '#1e293b')

            indicator_item = IndicatorItem(data, color, width=2)
            self.price_widget.addItem(indicator_item)
            self.price_indicator_items[name] = indicator_item

    def update_all_indicators(self):
        """Keep original update logic (UNCHANGED)"""
        active_indicators = self.indicator_panel.get_active_indicators()
        for indicator in active_indicators:
            self.update_indicators(indicator['name'], indicator['params'])

    # ==================== KEEP ALL ORIGINAL METHODS ====================

    def _update_ohlc_data(self):
        """Keep original OHLC update method (UNCHANGED)"""
        try:
            if not self.bars:
                return

            # Extract OHLC data
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            for bar in self.bars:
                timestamps.append(bar.datetime.timestamp())
                opens.append(float(bar.open_price))
                highs.append(float(bar.high_price))
                lows.append(float(bar.low_price))
                closes.append(float(bar.close_price))
                volumes.append(float(bar.volume))

            # Update data arrays
            self.timestamps = timestamps
            self.ohlc_data = {
                'timestamp': np.array(timestamps),
                'open': np.array(opens),
                'high': np.array(highs),
                'low': np.array(lows),
                'close': np.array(closes),
                'volume': np.array(volumes)
            }

            print(f"✅ OHLC data updated: {len(timestamps)} points")

        except Exception as e:
            print(f"❌ Error updating OHLC data: {e}")

    def _update_trade_markers(self):
        """Keep original trade marker update method (UNCHANGED)"""
        try:
            if not hasattr(self, 'backtest_trades') or not self.backtest_trades:
                print("⚠️ No backtest trades available")
                return

            if not self.bars:
                print("⚠️ No bars available for trade mapping")
                return

            start_time = self.bars[0].datetime
            end_time = self.bars[-1].datetime

            self.trade_markers = []

            for trade in self.backtest_trades:
                trade_time = trade.datetime

                # Check if trade is within current timeframe range
                if start_time <= trade_time <= end_time:
                    # Find closest bar for this trade
                    closest_bar_idx = self._find_closest_bar_index(trade_time)
                    if closest_bar_idx is not None:
                        # Determine offset type
                        offset_str = str(trade.offset).upper()

                        marker = {
                            'timestamp': self.bars[closest_bar_idx].datetime.timestamp(),
                            'price': float(trade.price),
                            'direction': 'long' if trade.direction.value == 1 else 'short',
                            'offset': offset_str,  # 'OPEN' or 'CLOSE'
                            'volume': float(trade.volume),
                            'datetime': trade.datetime
                        }
                        self.trade_markers.append(marker)

            print(f"✅ Trade markers updated: {len(self.trade_markers)} trades")

            # Debug: Print first few markers
            for i, marker in enumerate(self.trade_markers[:5]):
                print(f"   Trade {i + 1}: {marker['offset']} {marker['direction']} at {marker['price']}")

        except Exception as e:
            print(f"❌ Error updating trade markers: {e}")

    def _find_closest_bar_index(self, target_time):
        """Keep original method (UNCHANGED)"""
        try:
            min_diff = float('inf')
            closest_idx = None

            for i, bar in enumerate(self.bars):
                diff = abs((bar.datetime - target_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i

            return closest_idx

        except Exception as e:
            print(f"❌ Error finding closest bar: {e}")
            return None

    def clear_chart(self):
        """Keep original method (UNCHANGED)"""
        try:
            # Clear main chart
            if hasattr(self, 'price_chart') and self.price_chart:
                self.price_chart.clear()

            # Clear volume chart
            if hasattr(self, 'volume_chart') and self.volume_chart:
                self.volume_chart.clear()

            # Clear oscillator charts
            for chart_name in ['rsi_chart', 'macd_chart', 'stoch_chart']:
                if hasattr(self, chart_name):
                    chart = getattr(self, chart_name)
                    if chart:
                        chart.clear()

            print("✅ Chart cleared")

        except Exception as e:
            print(f"❌ Error clearing chart: {e}")

    def prepare_smc_dataframe(self):
        """Keep original SMC DataFrame preparation (UNCHANGED)"""
        if not self.bars:
            return

        try:
            # Create DataFrame with proper column names for SMC
            data = []

            for bar in self.bars:
                data.append({
                    'open': bar.open_price,
                    'high': bar.high_price,
                    'low': bar.low_price,
                    'close': bar.close_price,
                    'volume': bar.volume
                })

            self.df_ohlc = pd.DataFrame(data)
            self.df_ohlc.index = self.timestamps
            print(f"✅ SMC DataFrame prepared with {len(self.df_ohlc)} rows")

        except Exception as e:
            print(f"❌ Error preparing SMC DataFrame: {e}")
            self.df_ohlc = None

    def on_data_error(self, error_msg):
        """Keep original error handling (UNCHANGED)"""
        self.status_label.setText("❌ Error occurred")
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        QMessageBox.warning(self, "Data Loading Error", error_msg)

    def update_chart_info(self):
        """Keep original chart info update (UNCHANGED)"""
        if not self.bars:
            return

        try:
            symbol = self.symbol_combo.currentText()
            exchange = self.exchange_combo.currentText()
            timeframe = self.timeframe_combo.currentText()

            self.symbol_label.setText(f"Symbol: {symbol} ({exchange})")
            self.timeframe_label.setText(f"Timeframe: {timeframe}")

            start_date = self.bars[0].datetime.strftime('%m/%d %H:%M')
            end_date = self.bars[-1].datetime.strftime('%m/%d %H:%M')
            self.data_range_label.setText(f"Range: {start_date} - {end_date}")

            current_price = f"${self.bars[-1].close_price:,.2f}"
            self.current_price_label.setText(f"Price: {current_price}")

        except Exception as e:
            print(f"Error updating chart info: {e}")

    def display_professional_chart(self):
        """Keep original professional chart display (UNCHANGED)"""
        try:
            # Ensure OHLC data is available
            if not hasattr(self, 'ohlc_data') or not self.ohlc_data:
                print("⚠️ No OHLC data - generating from bars...")
                self._update_ohlc_data()

            if not hasattr(self, 'ohlc_data') or not self.ohlc_data:
                print("❌ No OHLC data available")
                return

            # Clear existing items
            self.price_widget.clear()
            self.oscillator_widget.clear()
            self.volume_widget.clear()
            self.price_indicator_items.clear()
            self.oscillator_indicator_items.clear()
            self.smc_items.clear()

            if not self.bars:
                return

            chart_data = []

            for i, bar in enumerate(self.bars):
                chart_data.append((
                    i,  # INDEX, not timestamp!
                    bar.open_price,
                    bar.high_price,
                    bar.low_price,
                    bar.close_price,
                    bar.volume
                ))

            interval = self.bars[0].interval if hasattr(self.bars[0], 'interval') else Interval.MINUTE

            prices = [bar.high_price for bar in self.bars] + [bar.low_price for bar in self.bars]
            volumes = [bar.volume for bar in self.bars]

            price_min, price_max = min(prices), max(prices)
            volume_max = max(volumes) if volumes else 1

            # Add grids
            price_grid = TradingViewGrid((0, len(self.bars)), (price_min, price_max), interval)
            self.price_widget.addItem(price_grid)

            oscillator_grid = TradingViewGrid((0, len(self.bars)), (0, 100), interval)
            self.oscillator_widget.addItem(oscillator_grid)

            volume_grid = TradingViewGrid((0, len(self.bars)), (0, volume_max), interval)
            self.volume_widget.addItem(volume_grid)

            # Time axes
            price_time_axis = ImprovedTimeAxis(self.timestamps, interval, orientation='bottom')
            self.price_widget.setAxisItems({'bottom': price_time_axis})

            oscillator_time_axis = ImprovedTimeAxis(self.timestamps, interval, orientation='bottom')
            self.oscillator_widget.setAxisItems({'bottom': oscillator_time_axis})

            volume_time_axis = ImprovedTimeAxis(self.timestamps, interval, orientation='bottom')
            self.volume_widget.setAxisItems({'bottom': volume_time_axis})

            # Chart items
            candle_item = CandlestickItem(chart_data, self.timestamps)
            self.price_widget.addItem(candle_item)

            volume_item = VolumeBarItem(chart_data)
            self.volume_widget.addItem(volume_item)

            # Set ranges
            data_len = len(self.bars)
            start_ix = max(0, data_len - 150)

            price_padding = (price_max - price_min) * 0.02
            self.price_widget.setRange(
                xRange=[start_ix, data_len],
                yRange=[price_min - price_padding, price_max + price_padding],
                padding=0
            )

            self.oscillator_widget.setRange(
                xRange=[start_ix, data_len],
                yRange=[0, 100],
                padding=0
            )

            self.volume_widget.setRange(
                xRange=[start_ix, data_len],
                yRange=[0, volume_max * 1.1],
                padding=0
            )

            # Update indicators and SMC features if not in backtest mode
            if not (hasattr(self, 'backtest_mode') and self.backtest_mode):
                self.update_all_indicators()
                self.update_all_smc_features()

            # ADD TRADE MARKERS
            print("🎯 About to add trade markers...")
            if hasattr(self, 'price_widget') and self.price_widget:
                print(f"✅ Price widget exists, adding markers...")
                self._display_trade_markers_simple()
            else:
                print("❌ No price widget found for markers")

            print("✅ Professional chart displayed successfully!")

        except Exception as e:
            print(f"❌ Error displaying chart: {e}")
            import traceback
            traceback.print_exc()

    # ==================== KEEP ALL ORIGINAL TRADE MARKER METHODS ====================

    def _display_trade_markers_simple(self):
        """Keep original trade marker display (UNCHANGED)"""
        try:
            if not hasattr(self, 'trade_markers') or not self.trade_markers:
                print("⚠️ No trade markers to display")
                return

            if not hasattr(self, 'price_widget') or not self.price_widget:
                print("❌ No price widget available for markers")
                return

            print(f"🎯 Adding {len(self.trade_markers)} trade markers...")

            # Convert timestamps to chart indices
            marker_indices = self._convert_markers_to_indices()

            if not marker_indices:
                print("❌ No valid marker indices found")
                return

            # Group trades into entry-exit pairs
            trade_pairs = self._group_trades_into_pairs(marker_indices)

            # Add trade marker pairs
            for pair in trade_pairs:
                self._add_trade_marker_pair(pair)

            print(f"✅ Trade marker pairs added: {len(trade_pairs)}")

        except Exception as e:
            print(f"❌ Error in trade marker display: {e}")
            import traceback
            traceback.print_exc()

    def _convert_markers_to_indices(self):
        """Keep original method (UNCHANGED)"""
        try:
            if not self.bars or not self.trade_markers:
                return []

            marker_indices = []

            for marker in self.trade_markers:
                # Find closest bar index for this trade timestamp
                marker_time = marker['datetime']  # Use datetime object

                closest_index = None
                min_diff = float('inf')

                for i, bar in enumerate(self.bars):
                    diff = abs((bar.datetime - marker_time).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_index = i

                if closest_index is not None:
                    marker_indices.append({
                        'index': closest_index,
                        'price': marker['price'],
                        'offset': marker['offset'],
                        'direction': marker['direction']
                    })

            print(f"✅ Converted {len(marker_indices)} markers to chart indices")
            return marker_indices

        except Exception as e:
            print(f"❌ Error converting markers to indices: {e}")
            return []

    def _group_trades_into_pairs(self, marker_indices):
        """Keep original method (UNCHANGED)"""
        try:
            trade_pairs = []
            entry_marker = None

            for marker in marker_indices:
                if marker['offset'] == 'OFFSET.OPEN':
                    entry_marker = marker
                elif marker['offset'] == 'OFFSET.CLOSE' and entry_marker:
                    # Create trade pair
                    trade_pairs.append({
                        'entry': entry_marker,
                        'exit': marker,
                        'direction': entry_marker['direction']
                    })
                    entry_marker = None

            print(f"✅ Grouped into {len(trade_pairs)} trade pairs")
            return trade_pairs

        except Exception as e:
            print(f"❌ Error grouping trades: {e}")
            return []

    def _add_trade_marker_pair(self, pair):
        """Keep original method (UNCHANGED)"""
        try:
            entry = pair['entry']
            exit_trade = pair['exit']
            direction = pair['direction']

            # Colors based on direction
            if direction == 'long':
                entry_color = 'green'
                exit_color = 'red'
                entry_symbol = 't'  # Triangle up
                exit_symbol = 't1'  # Triangle down
            else:
                entry_color = 'red'
                exit_color = 'green'
                entry_symbol = 't1'  # Triangle down
                exit_symbol = 't'  # Triangle up

            # Add entry marker
            entry_scatter = pg.ScatterPlotItem(
                [entry['index']],
                [entry['price']],
                size=15,
                pen=pg.mkPen(entry_color, width=2),
                brush=pg.mkBrush(entry_color),
                symbol=entry_symbol
            )
            self.price_widget.addItem(entry_scatter)

            # Add exit marker
            exit_scatter = pg.ScatterPlotItem(
                [exit_trade['index']],
                [exit_trade['price']],
                size=15,
                pen=pg.mkPen(exit_color, width=2),
                brush=pg.mkBrush(exit_color),
                symbol=exit_symbol
            )
            self.price_widget.addItem(exit_scatter)

            # Add connecting line
            line_color = 'green' if exit_trade['price'] > entry['price'] else 'red'
            trade_line = pg.PlotDataItem(
                [entry['index'], exit_trade['index']],
                [entry['price'], exit_trade['price']],
                pen=pg.mkPen(line_color, width=1, style=pg.QtCore.Qt.DashLine)
            )
            self.price_widget.addItem(trade_line)

            # Add profit/loss text
            pnl = exit_trade['price'] - entry['price']
            if direction == 'short':
                pnl = -pnl

            pnl_text = f"P/L: {pnl:.2f}"
            text_item = pg.TextItem(
                pnl_text,
                color=line_color,
                anchor=(0.5, 1.5)
            )
            text_item.setPos(exit_trade['index'], exit_trade['price'])
            self.price_widget.addItem(text_item)

        except Exception as e:
            print(f"❌ Error adding trade marker pair: {e}")

    # ==================== KEEP ALL ORIGINAL SMC METHODS ====================

    def update_smc_features(self, feature_name, params):
        """Keep original SMC update method (UNCHANGED)"""
        if not SMC_AVAILABLE or self.df_ohlc is None:
            return

        try:
            print(f"🔄 Calculating SMC feature: {feature_name}")
            smc_data = self.calculate_smc_feature(feature_name, params)
            if smc_data is not None:
                self.display_smc_feature_professional(feature_name, smc_data, params)
                print(f"✅ Updated SMC {feature_name}")
        except Exception as e:
            print(f"❌ Error updating SMC {feature_name}: {e}")

    def update_all_smc_features(self):
        """Keep original method (UNCHANGED)"""
        if SMC_AVAILABLE:
            active_features = self.smc_panel.get_active_smc_features()
            for feature in active_features:
                self.update_smc_features(feature['name'], feature['params'])

    def calculate_smc_feature(self, name, params):
        """Keep original SMC calculation method (UNCHANGED)"""
        if not SMC_AVAILABLE or self.df_ohlc is None:
            return None

        try:
            if name == 'FVG':
                return smc.fvg(self.df_ohlc, join_consecutive=True)
            elif name == 'SwingHL':
                swing_length = params.get('param1', 50)
                return smc.swing_highs_lows(self.df_ohlc, swing_length=swing_length)
            elif name == 'BOS_CHOCH':
                swing_data = smc.swing_highs_lows(self.df_ohlc, swing_length=50)
                return smc.bos_choch(self.df_ohlc, swing_data)
            elif name == 'OrderBlocks':
                swing_data = smc.swing_highs_lows(self.df_ohlc, swing_length=50)
                return smc.ob(self.df_ohlc, swing_data)
            elif name == 'Liquidity':
                swing_data = smc.swing_highs_lows(self.df_ohlc, swing_length=50)
                range_percent = params.get('param1', 1) / 100.0
                return smc.liquidity(self.df_ohlc, swing_data, range_percent=range_percent)
            elif name == 'PrevHL':
                return smc.previous_high_low(self.df_ohlc, time_frame="1D")
            elif name == 'Sessions':
                return smc.sessions(self.df_ohlc, session="London")
            elif name == 'Retracements':
                swing_data = smc.swing_highs_lows(self.df_ohlc, swing_length=50)
                return smc.retracements(self.df_ohlc, swing_data)
        except Exception as e:
            print(f"❌ SMC calculation error for {name}: {e}")
            return None

        return None

    def display_smc_feature_professional(self, name, data, params):
        """Keep original SMC display method (UNCHANGED)"""
        try:
            # Remove existing SMC item
            if name in self.smc_items:
                if isinstance(self.smc_items[name], list):
                    for item in self.smc_items[name]:
                        self.price_widget.removeItem(item)
                else:
                    self.price_widget.removeItem(self.smc_items[name])

            data_len = len(self.bars)

            if name == 'FVG':
                self.display_fvg_professional(data, data_len)
            elif name == 'SwingHL':
                self.display_swing_hl_professional(data, data_len)
            elif name == 'BOS_CHOCH':
                self.display_bos_choch_professional(data, data_len)
            elif name == 'OrderBlocks':
                self.display_order_blocks_professional(data, data_len)
            elif name == 'Liquidity':
                self.display_liquidity_professional(data, data_len)
            elif name == 'PrevHL':
                self.display_prev_hl_professional(data, data_len)
            elif name == 'Sessions':
                self.display_sessions_professional(data, data_len)

        except Exception as e:
            print(f"❌ Error displaying SMC {name}: {e}")

    def display_fvg_professional(self, fvg_data, data_len):
        """Keep original method (UNCHANGED)"""
        rectangles = []

        for i in range(len(fvg_data)):
            if 'FVG' in fvg_data.columns and not pd.isna(fvg_data['FVG'].iloc[i]):
                if 'Top' in fvg_data.columns and 'Bottom' in fvg_data.columns:
                    top = fvg_data['Top'].iloc[i]
                    bottom = fvg_data['Bottom'].iloc[i]

                    if not pd.isna(top) and not pd.isna(bottom):
                        # Determine end index
                        end_idx = data_len - 1
                        if 'MitigatedIndex' in fvg_data.columns:
                            mitigated = fvg_data['MitigatedIndex'].iloc[i]
                            if not pd.isna(mitigated) and mitigated > 0:
                                end_idx = int(mitigated)

                        rectangles.append((i, bottom, end_idx, top, "FVG"))

        if rectangles:
            fvg_item = SMCRectangleItem(rectangles, '#ffff00', opacity=0.2)
            self.price_widget.addItem(fvg_item)
            self.smc_items['FVG'] = fvg_item

    def display_swing_hl_professional(self, swing_data, data_len):
        """Keep original method (UNCHANGED)"""
        lines = []
        text_labels = []

        if 'HighLow' in swing_data.columns and 'Level' in swing_data.columns:
            swing_points = []

            for i in range(len(swing_data)):
                if not pd.isna(swing_data['HighLow'].iloc[i]):
                    swing_points.append((i, swing_data['Level'].iloc[i], swing_data['HighLow'].iloc[i]))

            # Connect swing points
            for j in range(len(swing_points) - 1):
                i1, level1, type1 = swing_points[j]
                i2, level2, type2 = swing_points[j + 1]

                color = '#00ff00' if type1 == -1 else '#ff0000'  # Green for lows, red for highs
                lines.append((i1, level1, i2, level2))

        if lines:
            swing_item = SMCLineItem(lines, '#ffa500', width=2)  # Orange lines
            self.price_widget.addItem(swing_item)
            self.smc_items['SwingHL'] = swing_item

    def display_bos_choch_professional(self, bos_data, data_len):
        """Keep original method (UNCHANGED)"""
        lines = []
        text_labels = []

        for i in range(len(bos_data)):
            # BOS
            if 'BOS' in bos_data.columns and not pd.isna(bos_data['BOS'].iloc[i]):
                if 'BrokenIndex' in bos_data.columns and 'Level' in bos_data.columns:
                    broken_idx = bos_data['BrokenIndex'].iloc[i]
                    level = bos_data['Level'].iloc[i]

                    if not pd.isna(broken_idx) and not pd.isna(level):
                        lines.append((i, level, int(broken_idx), level))
                        mid_x = (i + int(broken_idx)) / 2
                        text_labels.append((mid_x, level, "BOS"))

            # CHOCH
            if 'CHOCH' in bos_data.columns and not pd.isna(bos_data['CHOCH'].iloc[i]):
                if 'BrokenIndex' in bos_data.columns and 'Level' in bos_data.columns:
                    broken_idx = bos_data['BrokenIndex'].iloc[i]
                    level = bos_data['Level'].iloc[i]

                    if not pd.isna(broken_idx) and not pd.isna(level):
                        lines.append((i, level, int(broken_idx), level))
                        mid_x = (i + int(broken_idx)) / 2
                        text_labels.append((mid_x, level, "CHOCH"))

        if lines:
            bos_item = SMCLineItem(lines, '#ffa500', width=2, text_labels=text_labels)
            self.price_widget.addItem(bos_item)
            self.smc_items['BOS_CHOCH'] = bos_item

    def display_order_blocks_professional(self, ob_data, data_len):
        """Keep original method (UNCHANGED)"""
        rectangles = []

        for i in range(len(ob_data)):
            if 'OB' in ob_data.columns and not pd.isna(ob_data['OB'].iloc[i]):
                if 'Top' in ob_data.columns and 'Bottom' in ob_data.columns:
                    top = ob_data['Top'].iloc[i]
                    bottom = ob_data['Bottom'].iloc[i]

                    if not pd.isna(top) and not pd.isna(bottom):
                        # Determine end index
                        end_idx = data_len - 1
                        if 'MitigatedIndex' in ob_data.columns:
                            mitigated = ob_data['MitigatedIndex'].iloc[i]
                            if not pd.isna(mitigated) and mitigated > 0:
                                end_idx = int(mitigated)

                        # Format text
                        text = "OB"
                        if 'OBVolume' in ob_data.columns:
                            volume = ob_data['OBVolume'].iloc[i]
                            if not pd.isna(volume):
                                if volume >= 1e9:
                                    text = f"OB: {volume / 1e9:.1f}B"
                                elif volume >= 1e6:
                                    text = f"OB: {volume / 1e6:.1f}M"
                                elif volume >= 1e3:
                                    text = f"OB: {volume / 1e3:.1f}K"

                        rectangles.append((i, bottom, end_idx, top, text))

        if rectangles:
            ob_item = SMCRectangleItem(rectangles, '#800080', opacity=0.2)  # Purple
            self.price_widget.addItem(ob_item)
            self.smc_items['OrderBlocks'] = ob_item

    def display_liquidity_professional(self, liq_data, data_len):
        """Keep original method (UNCHANGED)"""
        lines = []
        text_labels = []

        for i in range(len(liq_data)):
            if 'Liquidity' in liq_data.columns and not pd.isna(liq_data['Liquidity'].iloc[i]):
                if 'Level' in liq_data.columns and 'End' in liq_data.columns:
                    level = liq_data['Level'].iloc[i]
                    end_idx = liq_data['End'].iloc[i]

                    if not pd.isna(level) and not pd.isna(end_idx):
                        lines.append((i, level, int(end_idx), level))
                        mid_x = (i + int(end_idx)) / 2
                        text_labels.append((mid_x, level, "Liquidity"))

                        # Check for swept liquidity
                        if 'Swept' in liq_data.columns:
                            swept_idx = liq_data['Swept'].iloc[i]
                            if not pd.isna(swept_idx) and swept_idx > 0:
                                # Add sweep line
                                swept_idx = int(swept_idx)
                                if swept_idx < len(self.bars):
                                    swept_price = self.bars[swept_idx].high_price if liq_data['Liquidity'].iloc[
                                                                                         i] == 1 else self.bars[
                                        swept_idx].low_price
                                    lines.append((int(end_idx), level, swept_idx, swept_price))
                                    text_labels.append(
                                        ((int(end_idx) + swept_idx) / 2, (level + swept_price) / 2, "Swept"))

        if lines:
            liq_item = SMCLineItem(lines, '#ffa500', width=2, style='dash', text_labels=text_labels)
            self.price_widget.addItem(liq_item)
            self.smc_items['Liquidity'] = liq_item

    def display_prev_hl_professional(self, prev_data, data_len):
        """Keep original method (UNCHANGED)"""
        lines = []
        text_labels = []

        # Process Previous Highs
        if 'PreviousHigh' in prev_data.columns:
            high_levels = []
            high_indexes = []

            for i in range(len(prev_data)):
                high_val = prev_data['PreviousHigh'].iloc[i]
                if not pd.isna(high_val):
                    if not high_levels or high_val != high_levels[-1]:
                        high_levels.append(high_val)
                        high_indexes.append(i)

            # Draw lines between consecutive high levels
            for j in range(len(high_indexes) - 1):
                lines.append((high_indexes[j], high_levels[j], high_indexes[j + 1], high_levels[j]))
                text_labels.append((high_indexes[j + 1], high_levels[j], "PH"))

        # Process Previous Lows
        if 'PreviousLow' in prev_data.columns:
            low_levels = []
            low_indexes = []

            for i in range(len(prev_data)):
                low_val = prev_data['PreviousLow'].iloc[i]
                if not pd.isna(low_val):
                    if not low_levels or low_val != low_levels[-1]:
                        low_levels.append(low_val)
                        low_indexes.append(i)

            # Draw lines between consecutive low levels
            for j in range(len(low_indexes) - 1):
                lines.append((low_indexes[j], low_levels[j], low_indexes[j + 1], low_levels[j]))
                text_labels.append((low_indexes[j + 1], low_levels[j], "PL"))

        if lines:
            prev_item = SMCLineItem(lines, '#ffffff', width=2, text_labels=text_labels)
            self.price_widget.addItem(prev_item)
            self.smc_items['PrevHL'] = prev_item

    def display_sessions_professional(self, sessions_data, data_len):
        """Keep original method (UNCHANGED)"""
        rectangles = []

        for i in range(len(sessions_data) - 1):
            if 'Active' in sessions_data.columns and sessions_data['Active'].iloc[i] == 1:
                if 'High' in sessions_data.columns and 'Low' in sessions_data.columns:
                    high = sessions_data['High'].iloc[i]
                    low = sessions_data['Low'].iloc[i]

                    if not pd.isna(high) and not pd.isna(low):
                        rectangles.append((i, low, i + 1, high, "Session"))

        if rectangles:
            session_item = SMCRectangleItem(rectangles, '#16866E', opacity=0.2)
            self.price_widget.addItem(session_item)
            self.smc_items['Sessions'] = session_item


def main():
    """🔥 ENHANCED: Main function with better startup messages"""
    try:
        print("🚀 Starting Professional Trading Platform with Howtrader Integration")
        print("=" * 70)
        print("🔧 Framework Status:")
        print(f"   • Howtrader: ✅ Integrated")
        print(f"   • TA-Lib: {'✅ Available' if TALIB_AVAILABLE else '⚠️ Not available'}")
        print(f"   • SMC: {'✅ Available' if SMC_AVAILABLE else '⚠️ Not available'}")
        print("=" * 70)

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        chart = AdvancedHowtraderChart()
        chart.show()

        print("✅ Platform launched successfully!")

        return app.exec()

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())