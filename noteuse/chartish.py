from datetime import datetime, timedelta
import sys
import os

# Set Qt environment
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

# Import howtrader components
from howtrader.trader.constant import Exchange, Interval
from howtrader.trader.database import get_database
from howtrader.trader.object import BarData
from howtrader.trader.utility import BarGenerator, ArrayManager

# Import SMC library
try:
    from smartmoneyconcepts.smc import smc

    SMC_AVAILABLE = True
    print("✅ SMC library imported successfully")
except ImportError as e:
    SMC_AVAILABLE = False
    print(f"❌ SMC library not found: {e}")
    print("SMC features will be disabled")

# Configure pyqtgraph
pg.setConfigOption('useOpenGL', False)
pg.setConfigOption('antialias', True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


# =============================================================================
# CONFIGURATION CONSTANTS - Clean and centralized
# =============================================================================

class ChartConfig:
    """Chart configuration constants"""

    # Colors - TradingView Style
    class Colors:
        BACKGROUND = '#ffffff'
        GRID_LIGHT = '#f8fafc'
        GRID_MEDIUM = '#f1f5f9'
        GRID_MAJOR = '#e2e8f0'
        TEXT_PRIMARY = '#1e293b'
        TEXT_SECONDARY = '#475569'

        # Candlestick colors
        CANDLE_UP = '#26a69a'
        CANDLE_DOWN = '#ef5350'

        # Indicator colors
        EMA_COLOR = '#2563eb'
        SMA_COLOR = '#dc2626'
        RSI_COLOR = '#7c3aed'
        MACD_COLOR = '#059669'
        BOLLINGER_COLOR = '#8b5cf6'

        # SMC colors
        FVG_COLOR = '#fbbf24'  # Yellow
        ORDER_BLOCK_COLOR = '#8b5cf6'  # Purple
        BOS_COLOR = '#f97316'  # Orange
        CHOCH_COLOR = '#3b82f6'  # Blue
        LIQUIDITY_COLOR = '#f59e0b'  # Amber
        SWING_HL_COLOR = '#10b981'  # Emerald
        PREV_HL_COLOR = '#6b7280'  # Gray
        SESSION_COLOR = '#16a34a'  # Green

    # Chart dimensions
    class Dimensions:
        WINDOW_WIDTH = 1900
        WINDOW_HEIGHT = 1000
        LEFT_PANEL_WIDTH = 350
        TITLE_HEIGHT = 60
        PRICE_CHART_MIN_HEIGHT = 400
        OSCILLATOR_HEIGHT = 150
        VOLUME_HEIGHT = 120

    # Grid settings
    class Grid:
        OPACITY = 0.3
        MAJOR_GRID_INTERVAL = 5
        MINOR_GRID_LINES = 20

    # Text settings
    class Text:
        TITLE_SIZE = 15
        LABEL_SIZE = 13
        INFO_SIZE = 11
        SMC_LABEL_SIZE = 10
        OPACITY = 0.7


class SMCConfig:
    """SMC feature configuration"""
    # Default parameters
    SWING_LENGTH = 50
    LIQUIDITY_RANGE_PERCENT = 1.0
    FVG_JOIN_CONSECUTIVE = True
    DEFAULT_SESSION = "London"
    DEFAULT_TIMEFRAME = "4h"

    # Visual settings
    RECTANGLE_OPACITY = 0.25
    LINE_WIDTH = 2
    LABEL_OFFSET = 5


# =============================================================================
# IMPROVED CHART COMPONENTS
# =============================================================================

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

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        # Adaptive grid density
        if self.interval in [Interval.MINUTE, Interval.MINUTE_5, Interval.MINUTE_15]:
            x_minor_step = max(1, int((x_max - x_min) / 50))
            x_major_step = max(5, int((x_max - x_min) / 10))
        else:
            x_minor_step = max(1, int((x_max - x_min) / 30))
            x_major_step = max(3, int((x_max - x_min) / 6))

        # Vertical grid lines
        for i in range(int(x_min), int(x_max) + 1, x_minor_step):
            color = ChartConfig.Colors.GRID_MAJOR if i % x_major_step == 0 else ChartConfig.Colors.GRID_LIGHT
            p.setPen(pg.mkPen(color, width=1))
            p.drawLine(pg.QtCore.QPointF(i, y_min), pg.QtCore.QPointF(i, y_max))

        # Horizontal grid lines
        y_step = (y_max - y_min) / ChartConfig.Grid.MINOR_GRID_LINES
        for i in range(ChartConfig.Grid.MINOR_GRID_LINES + 1):
            y = y_min + i * y_step
            color = ChartConfig.Colors.GRID_MAJOR if i % ChartConfig.Grid.MAJOR_GRID_INTERVAL == 0 else ChartConfig.Colors.GRID_MEDIUM
            p.setPen(pg.mkPen(color, width=1))
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
        """Generate candlestick chart"""
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)

        for i, (timestamp, open_price, high_price, low_price, close_price, volume) in enumerate(self.data):
            # Color selection
            if close_price >= open_price:
                pen_color = pg.QtGui.QColor(ChartConfig.Colors.CANDLE_UP)
                brush_color = pg.QtGui.QColor(ChartConfig.Colors.CANDLE_UP)
                brush_color.setAlpha(180)
            else:
                pen_color = pg.QtGui.QColor(ChartConfig.Colors.CANDLE_DOWN)
                brush_color = pg.QtGui.QColor(ChartConfig.Colors.CANDLE_DOWN)
                brush_color.setAlpha(180)

            p.setPen(pg.mkPen(pen_color, width=1))
            p.setBrush(pg.mkBrush(brush_color))

            # Draw wick
            if high_price != low_price:
                p.drawLine(pg.QtCore.QPointF(i, low_price), pg.QtCore.QPointF(i, high_price))

            # Draw body
            if abs(open_price - close_price) > 0:
                body_top = max(open_price, close_price)
                body_bottom = min(open_price, close_price)
                body_height = body_top - body_bottom

                rect = pg.QtCore.QRectF(i - 0.4, body_bottom, 0.8, body_height)
                p.drawRect(rect)
            else:
                # Doji
                p.setPen(pg.mkPen(pen_color, width=2))
                p.drawLine(pg.QtCore.QPointF(i - 0.4, open_price), pg.QtCore.QPointF(i + 0.4, open_price))

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

        for i, (timestamp, open_price, high_price, low_price, close_price, volume) in enumerate(self.data):
            if close_price >= open_price:
                color = pg.QtGui.QColor(ChartConfig.Colors.CANDLE_UP)
            else:
                color = pg.QtGui.QColor(ChartConfig.Colors.CANDLE_DOWN)

            color.setAlpha(100)
            p.setPen(pg.mkPen(color, width=1))
            p.setBrush(pg.mkBrush(color))

            if volume > 0:
                rect = pg.QtCore.QRectF(i - 0.4, 0, 0.8, volume)
                p.drawRect(rect)

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class ImprovedTimeAxis(pg.AxisItem):
    """Improved time axis with proper formatting"""

    def __init__(self, timestamps, interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamps = timestamps
        self.interval = interval
        self.setPen(color=ChartConfig.Colors.GRID_MAJOR, width=1)
        self.setTextPen(color=ChartConfig.Colors.TEXT_PRIMARY)

    def tickStrings(self, values, scale, spacing):
        strings = []

        for v in values:
            try:
                index = int(v)
                if 0 <= index < len(self.timestamps):
                    dt = self.timestamps[index]

                    # Smart formatting based on timeframe
                    if self.interval in [Interval.MINUTE, Interval.MINUTE_5, Interval.MINUTE_15, Interval.MINUTE_30]:
                        if index > 0:
                            prev_dt = self.timestamps[max(0, index - 20)]
                            if dt.date() == prev_dt.date():
                                time_str = dt.strftime('%H:%M')
                            else:
                                time_str = dt.strftime('%m/%d\n%H:%M')
                        else:
                            time_str = dt.strftime('%H:%M')
                    elif self.interval in [Interval.HOUR, Interval.HOUR_4]:
                        time_str = dt.strftime('%m/%d\n%H:%M')
                    elif self.interval == Interval.DAILY:
                        time_str = dt.strftime('%m/%d')
                    elif self.interval == Interval.WEEKLY:
                        time_str = dt.strftime('%m/%d')
                    else:
                        time_str = dt.strftime('%m/%d')

                    strings.append(time_str)
                else:
                    strings.append('')
            except (ValueError, IndexError, AttributeError):
                strings.append('')

        return strings


class IndicatorCalculator:
    """Professional indicator calculator using howtrader methods"""

    @staticmethod
    def calculate_ema(closes, period):
        """Calculate EMA using proper exponential smoothing"""
        if len(closes) < period:
            return np.full(len(closes), np.nan)

        ema_values = np.full(len(closes), np.nan)
        alpha = 2.0 / (period + 1)

        # Initialize with SMA
        ema_values[period - 1] = np.mean(closes[:period])

        # Calculate subsequent EMA values
        for i in range(period, len(closes)):
            ema_values[i] = alpha * closes[i] + (1 - alpha) * ema_values[i - 1]

        return ema_values

    @staticmethod
    def calculate_sma(closes, period):
        """Calculate Simple Moving Average"""
        sma_values = np.full(len(closes), np.nan)

        for i in range(period - 1, len(closes)):
            sma_values[i] = np.mean(closes[i - period + 1:i + 1])

        return sma_values

    @staticmethod
    def calculate_rsi(closes, period=14):
        """Calculate RSI"""
        if len(closes) < period + 1:
            return np.full(len(closes), np.nan)

        rsi_values = np.full(len(closes), np.nan)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_values[period] = 100 - (100 / (1 + rs))

        # Calculate subsequent RSI values
        for i in range(period + 1, len(closes)):
            avg_gain = ((avg_gain * (period - 1)) + gains[i - 1]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i - 1]) / period

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))

        return rsi_values

    @staticmethod
    def calculate_macd(closes, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if len(closes) < slow:
            return np.full(len(closes), np.nan)

        ema_fast = IndicatorCalculator.calculate_ema(closes, fast)
        ema_slow = IndicatorCalculator.calculate_ema(closes, slow)

        macd_line = ema_fast - ema_slow
        return macd_line

    @staticmethod
    def calculate_kdj(highs, lows, closes, period=9):
        """Calculate KDJ"""
        k_values = np.full(len(closes), np.nan)

        for i in range(period - 1, len(closes)):
            highest_high = np.max(highs[i - period + 1:i + 1])
            lowest_low = np.min(lows[i - period + 1:i + 1])

            if highest_high != lowest_low:
                k_values[i] = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k_values[i] = 50

        return k_values

    @staticmethod
    def calculate_bollinger_upper(closes, period=20, std_dev=2):
        """Calculate Bollinger Upper Band"""
        upper_band = np.full(len(closes), np.nan)

        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window)
            upper_band[i] = sma + (std_dev * std)

        return upper_band


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


class SMCRectangleItem(pg.GraphicsObject):
    """SMC rectangle visualization with improved label positioning"""

    def __init__(self, rectangles, color, opacity=SMCConfig.RECTANGLE_OPACITY, text_overlay=None):
        pg.GraphicsObject.__init__(self)
        self.rectangles = rectangles
        self.color = color
        self.opacity = opacity
        self.text_overlay = text_overlay
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)

        # Set rectangle properties
        fill_color = pg.QtGui.QColor(self.color)
        fill_color.setAlphaF(self.opacity)

        border_color = pg.QtGui.QColor(self.color)
        border_color.setAlphaF(0.8)

        p.setPen(pg.mkPen(border_color, width=1))
        p.setBrush(pg.mkBrush(fill_color))

        # Draw rectangles with improved text positioning
        for rect_data in self.rectangles:
            if len(rect_data) >= 4:
                x0, y0, x1, y1 = rect_data[:4]
                rect = pg.QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
                p.drawRect(rect)

                # Improved text positioning
                if len(rect_data) > 4 and rect_data[4]:
                    text = rect_data[4]
                    center_x = (x0 + x1) / 2
                    center_y = (y0 + y1) / 2

                    # Set text properties
                    font = pg.QtGui.QFont()
                    font.setPointSize(ChartConfig.Text.SMC_LABEL_SIZE)
                    p.setFont(font)

                    text_color = pg.QtGui.QColor('#ffffff')
                    text_color.setAlphaF(ChartConfig.Text.OPACITY)
                    p.setPen(pg.mkPen(text_color, width=1))

                    # Calculate text bounds for better positioning
                    text_rect = p.fontMetrics().boundingRect(text)
                    text_x = center_x - text_rect.width() / 2
                    text_y = center_y + text_rect.height() / 4

                    p.drawText(pg.QtCore.QPointF(text_x, text_y), text)

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class SMCLineItem(pg.GraphicsObject):
    """SMC line visualization with fixed label positioning"""

    def __init__(self, lines, color, width=SMCConfig.LINE_WIDTH, style='solid', text_labels=None):
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

        # Set line properties
        pen_style = Qt.PenStyle.SolidLine if self.style == 'solid' else Qt.PenStyle.DashLine
        pen = pg.mkPen(self.color, width=self.width, style=pen_style)
        p.setPen(pen)

        # Draw lines
        for line_data in self.lines:
            if len(line_data) >= 4:
                x0, y0, x1, y1 = line_data
                p.drawLine(pg.QtCore.QPointF(x0, y0), pg.QtCore.QPointF(x1, y1))

        # Draw text labels with improved positioning
        for label_data in self.text_labels:
            if len(label_data) >= 3:
                x, y, text = label_data

                # Set text properties
                font = pg.QtGui.QFont()
                font.setPointSize(ChartConfig.Text.SMC_LABEL_SIZE)
                p.setFont(font)

                text_color = pg.QtGui.QColor(self.color)
                text_color.setAlphaF(ChartConfig.Text.OPACITY)
                p.setPen(pg.mkPen(text_color, width=1))

                # Calculate proper text position
                text_rect = p.fontMetrics().boundingRect(text)
                text_x = x - text_rect.width() / 2
                text_y = y + SMCConfig.LABEL_OFFSET

                # Ensure text doesn't flip by maintaining consistent positioning
                p.drawText(pg.QtCore.QPointF(text_x, text_y), text)

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


# =============================================================================
# MAIN TRADING PLATFORM CLASS
# =============================================================================

class ExtendedTimeframeConverter:
    """Convert minute bars to extended timeframes"""

    def __init__(self):
        self.converted_bars = []

    def convert_bars(self, minute_bars, target_interval):
        self.converted_bars = []

        if target_interval == Interval.MINUTE:
            return minute_bars

        if target_interval in [Interval.MINUTE_5, Interval.MINUTE_15,
                               Interval.MINUTE_30, Interval.HOUR,
                               Interval.HOUR_4, Interval.DAILY]:
            return self._convert_standard_intervals(minute_bars, target_interval)

        if target_interval == Interval.WEEKLY:
            return self._convert_to_weekly(minute_bars)
        elif target_interval == Interval.MONTH:
            return self._convert_to_monthly(minute_bars)

        return minute_bars

    def _convert_standard_intervals(self, minute_bars, target_interval):
        interval_windows = {
            Interval.MINUTE_5: 5,
            Interval.MINUTE_15: 15,
            Interval.MINUTE_30: 30,
            Interval.HOUR: 60,
            Interval.HOUR_4: 240,
            Interval.DAILY: 1440
        }

        window = interval_windows.get(target_interval, 1)

        if target_interval in [Interval.MINUTE_5, Interval.MINUTE_15, Interval.MINUTE_30]:
            generator = BarGenerator(
                on_bar=lambda x: None,
                window=window,
                on_window_bar=self._on_window_bar,
                interval=Interval.MINUTE
            )
        else:
            hours = window // 60 if window >= 60 else 1
            generator = BarGenerator(
                on_bar=lambda x: None,
                window=hours,
                on_window_bar=self._on_window_bar,
                interval=target_interval
            )

        for bar in minute_bars:
            generator.update_bar(bar)

        if hasattr(generator, 'window_bar') and generator.window_bar:
            self._on_window_bar(generator.window_bar)
        elif hasattr(generator, 'hour_bar') and generator.hour_bar:
            self._on_window_bar(generator.hour_bar)

        return self.converted_bars

    def _convert_to_weekly(self, minute_bars):
        if not minute_bars:
            return []

        result_bars = []
        current_bar = None

        for bar in minute_bars:
            year, week, _ = bar.datetime.isocalendar()
            week_key = (year, week)

            if current_bar is None or getattr(current_bar, 'week_key', None) != week_key:
                if current_bar:
                    result_bars.append(current_bar)

                current_bar = self._create_new_bar(bar, Interval.WEEKLY)
                current_bar.week_key = week_key
            else:
                self._update_bar(current_bar, bar)

        if current_bar:
            result_bars.append(current_bar)

        return result_bars

    def _convert_to_monthly(self, minute_bars):
        if not minute_bars:
            return []

        result_bars = []
        current_bar = None

        for bar in minute_bars:
            month_key = (bar.datetime.year, bar.datetime.month)

            if current_bar is None or getattr(current_bar, 'month_key', None) != month_key:
                if current_bar:
                    result_bars.append(current_bar)

                current_bar = self._create_new_bar(bar, Interval.MONTH)
                current_bar.month_key = month_key
            else:
                self._update_bar(current_bar, bar)

        if current_bar:
            result_bars.append(current_bar)

        return result_bars

    def _create_new_bar(self, source_bar, interval):
        new_bar = BarData(
            symbol=source_bar.symbol,
            exchange=source_bar.exchange,
            datetime=source_bar.datetime,
            interval=interval,
            volume=source_bar.volume,
            turnover=source_bar.turnover,
            open_interest=source_bar.open_interest,
            open_price=source_bar.open_price,
            high_price=source_bar.high_price,
            low_price=source_bar.low_price,
            close_price=source_bar.close_price,
            gateway_name=source_bar.gateway_name
        )
        return new_bar

    def _update_bar(self, current_bar, new_bar):
        current_bar.high_price = max(current_bar.high_price, new_bar.high_price)
        current_bar.low_price = min(current_bar.low_price, new_bar.low_price)
        current_bar.close_price = new_bar.close_price
        current_bar.volume += new_bar.volume
        current_bar.turnover += new_bar.turnover
        current_bar.datetime = new_bar.datetime
        current_bar.open_interest = new_bar.open_interest

    def _on_window_bar(self, bar):
        self.converted_bars.append(bar)


class DataLoader(QThread):
    """Background data loader thread"""
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
        self.converter = ExtendedTimeframeConverter()

    def run(self):
        try:
            self.progress_updated.emit(10)

            database = get_database()
            self.progress_updated.emit(30)

            print(f"🔍 Loading data for {self.symbol} from {self.exchange}")
            print(f"   Timeframe: {self.interval}")
            print(f"   Period: {self.start_time} to {self.end_time}")

            # Always load minute data for conversion flexibility
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

                # Convert to target timeframe
                converted_bars = self.converter.convert_bars(bars, self.interval)

                if converted_bars:
                    bars = converted_bars
                    print(f"✅ Converted to {len(bars)} {self.interval} bars")
                else:
                    print("⚠️ Conversion failed, using original minute data")

            self.progress_updated.emit(90)

            if bars and len(bars) > 0:
                print(f"✅ Final result: {len(bars)} bars")
                print(f"   Time range: {bars[0].datetime} to {bars[-1].datetime}")
                self.data_loaded.emit(bars)
            else:
                error_msg = (
                    f"❌ No data found for {self.symbol} on {self.exchange}\n\n"
                    "Possible solutions:\n"
                    "• Check if the symbol exists in your database\n"
                    "• Verify data collection is running for this exchange\n"
                    "• Try a different date range\n"
                    "• Check if the exchange name matches your database\n\n"
                    "Available exchanges in howtrader:\n"
                    "• BINANCE, HUOBI, OKX, etc."
                )
                self.error_occurred.emit(error_msg)

            self.progress_updated.emit(100)

        except Exception as e:
            error_msg = f"Database error: {str(e)}\n\nPlease check your database connection and data availability."
            self.error_occurred.emit(error_msg)


class TechnicalIndicatorPanel(QWidget):
    """Technical indicator configuration panel"""

    indicator_changed = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.indicators = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Technical Indicators")
        title.setFont(QFont("Arial", ChartConfig.Text.LABEL_SIZE, QFont.Bold))
        title.setStyleSheet(f"color: {ChartConfig.Colors.TEXT_PRIMARY}; padding: 5px;")
        layout.addWidget(title)

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

        # Add predefined indicators
        self.add_indicator_row("EMA", 20, "", True)
        self.add_indicator_row("RSI", 14, "", False)
        self.add_indicator_row("KDJ", 9, "3", False)

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

        # Standard indicators
        standard_indicators = ['EMA', 'SMA', 'RSI', 'MACD', 'KDJ', 'Bollinger']
        indicator_combo.addItems(standard_indicators)

        # Add SMC indicators if available
        if SMC_AVAILABLE:
            smc_indicators = ['FVG', 'SwingHL', 'BOS_CHOCH', 'OrderBlocks', 'Liquidity', 'PrevHL', 'Sessions',
                              'Retracements']
            indicator_combo.addItems(smc_indicators)

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
        self.add_indicator_row("EMA", 20, "", False)

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
    """SMC Feature configuration panel"""

    smc_feature_changed = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.smc_features = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Smart Money Concepts")
        title.setFont(QFont("Arial", ChartConfig.Text.LABEL_SIZE, QFont.Bold))
        title.setStyleSheet(f"color: {ChartConfig.Colors.TEXT_PRIMARY}; padding: 5px;")
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
        self.add_smc_row("SwingHL", str(SMCConfig.SWING_LENGTH), "", False)
        self.add_smc_row("BOS_CHOCH", "", "", False)
        self.add_smc_row("OrderBlocks", "", "", False)
        self.add_smc_row("Liquidity", str(int(SMCConfig.LIQUIDITY_RANGE_PERCENT)), "", False)

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


class AdvancedHowtraderChart(QWidget):
    """Professional trading platform with SMC integration - FIXED VERSION"""

    def __init__(self):
        super().__init__()
        self.bars = None
        self.data_loader = None
        self.array_manager = ArrayManager()
        self.price_indicator_items = {}
        self.oscillator_indicator_items = {}
        self.smc_items = {}
        self.df_ohlc = None
        self.timestamps = []
        self.initUI()

    def initUI(self):
        # Fixed title - removed extra parenthesis
        self.setWindowTitle("📊 Professional Trading Platform")
        self.setGeometry(50, 50, ChartConfig.Dimensions.WINDOW_WIDTH, ChartConfig.Dimensions.WINDOW_HEIGHT)

        main_layout = QHBoxLayout()

        # Left panel
        left_panel = QWidget()
        left_panel.setMaximumWidth(ChartConfig.Dimensions.LEFT_PANEL_WIDTH)
        left_panel.setStyleSheet(f"""
            background-color: {ChartConfig.Colors.BACKGROUND}; 
            border-right: 2px solid {ChartConfig.Colors.GRID_MAJOR};
            border-radius: 8px;
        """)

        left_layout = QVBoxLayout()

        # Fixed title widget
        title_widget = QWidget()
        title_widget.setFixedHeight(ChartConfig.Dimensions.TITLE_HEIGHT)
        title_widget.setStyleSheet(f"""
            background-color: {ChartConfig.Colors.TEXT_PRIMARY};
            border-radius: 8px;
            margin-bottom: 10px;
        """)
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(15, 10, 15, 10)

        # Fixed title text
        title = QLabel("📊 Pro Trading Platform")
        title.setFont(QFont("Arial", ChartConfig.Text.TITLE_SIZE, QFont.Bold))
        title.setStyleSheet("color: white; text-align: center;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(title)

        title_widget.setLayout(title_layout)
        left_layout.addWidget(title_widget)

        # Exchange selection
        exchange_group = QGroupBox("Exchange")
        exchange_group.setStyleSheet(f"""
            QGroupBox {{ 
                font-weight: bold; 
                color: {ChartConfig.Colors.TEXT_PRIMARY}; 
                font-size: {ChartConfig.Text.LABEL_SIZE}px;
                border: 1px solid {ChartConfig.Colors.GRID_MAJOR};
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        exchange_layout = QVBoxLayout()

        self.exchange_combo = QComboBox()
        self.exchange_combo.addItem("Binance", Exchange.BINANCE)
        self.exchange_combo.addItem("Huobi", Exchange.HUOBI)
        self.exchange_combo.addItem("OKX", Exchange.OKX)

        exchange_layout.addWidget(self.exchange_combo)
        exchange_group.setLayout(exchange_layout)
        left_layout.addWidget(exchange_group)

        # Symbol and timeframe
        symbol_group = QGroupBox("Symbol & Timeframe")
        symbol_group.setStyleSheet(f"""
            QGroupBox {{ 
                font-weight: bold; 
                color: {ChartConfig.Colors.TEXT_PRIMARY}; 
                font-size: {ChartConfig.Text.LABEL_SIZE}px;
                border: 1px solid {ChartConfig.Colors.GRID_MAJOR};
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        symbol_layout = QGridLayout()

        symbol_layout.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT"])
        symbol_layout.addWidget(self.symbol_combo, 0, 1)

        symbol_layout.addWidget(QLabel("Timeframe:"), 1, 0)
        self.timeframe_combo = QComboBox()

        timeframe_options = [
            ("1 Minute", Interval.MINUTE),
            ("5 Minutes", Interval.MINUTE_5),
            ("15 Minutes", Interval.MINUTE_15),
            ("30 Minutes", Interval.MINUTE_30),
            ("1 Hour", Interval.HOUR),
            ("4 Hours", Interval.HOUR_4),
            ("1 Day", Interval.DAILY),
            ("1 Week", Interval.WEEKLY),
            ("1 Month", Interval.MONTH)
        ]

        for text, interval in timeframe_options:
            self.timeframe_combo.addItem(text, interval)

        symbol_layout.addWidget(self.timeframe_combo, 1, 1)
        symbol_group.setLayout(symbol_layout)
        left_layout.addWidget(symbol_group)

        # Date range
        date_group = QGroupBox("Date Range")
        date_group.setStyleSheet(f"""
            QGroupBox {{ 
                font-weight: bold; 
                color: {ChartConfig.Colors.TEXT_PRIMARY}; 
                font-size: {ChartConfig.Text.LABEL_SIZE}px;
                border: 1px solid {ChartConfig.Colors.GRID_MAJOR};
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
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

        # Load button
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

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Chart info
        info_group = QGroupBox("Chart Information")
        info_group.setStyleSheet(f"""
            QGroupBox {{ 
                font-weight: bold; 
                color: {ChartConfig.Colors.TEXT_PRIMARY}; 
                font-size: {ChartConfig.Text.LABEL_SIZE}px;
                border: 1px solid {ChartConfig.Colors.GRID_MAJOR};
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        info_layout = QVBoxLayout()

        self.symbol_label = QLabel("Symbol: --")
        self.timeframe_label = QLabel("Timeframe: --")
        self.data_range_label = QLabel("Data Range: --")
        self.current_price_label = QLabel("Current Price: --")

        for label in [self.symbol_label, self.timeframe_label, self.data_range_label, self.current_price_label]:
            label.setStyleSheet(
                f"padding: 2px; font-size: {ChartConfig.Text.INFO_SIZE}px; color: {ChartConfig.Colors.TEXT_SECONDARY};")
            info_layout.addWidget(label)

        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        # Tabbed panels for indicators and SMC
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {ChartConfig.Colors.GRID_MAJOR};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background: {ChartConfig.Colors.GRID_LIGHT};
                color: {ChartConfig.Colors.TEXT_SECONDARY};
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: #2563eb;
                color: white;
            }}
        """)

        # Technical indicators tab
        self.indicator_panel = TechnicalIndicatorPanel()
        self.indicator_panel.indicator_changed.connect(self.update_indicators)
        tab_widget.addTab(self.indicator_panel, "Indicators")

        # SMC features tab
        self.smc_panel = SMCFeaturePanel()
        self.smc_panel.smc_feature_changed.connect(self.update_smc_features)
        tab_widget.addTab(self.smc_panel, "SMC")

        left_layout.addWidget(tab_widget)

        # Status
        self.status_label = QLabel("Ready to load data")
        self.status_label.setStyleSheet("color: #059669; padding: 5px; font-weight: bold; font-size: 12px;")
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # Right panel for charts
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        self.create_professional_charts(right_layout)

        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel, 1)

        self.setLayout(main_layout)

    def create_professional_charts(self, layout):
        # Price chart
        self.price_widget = pg.PlotWidget()
        self.price_widget.setLabel('left', 'Price (USDT)', color=ChartConfig.Colors.TEXT_PRIMARY, size='14pt')
        self.price_widget.setMinimumHeight(ChartConfig.Dimensions.PRICE_CHART_MIN_HEIGHT)
        self.price_widget.setBackground(ChartConfig.Colors.BACKGROUND)

        self.price_widget.getAxis('left').setPen(ChartConfig.Colors.GRID_MAJOR)
        self.price_widget.getAxis('left').setTextPen(ChartConfig.Colors.TEXT_PRIMARY)
        self.price_widget.getAxis('bottom').setPen(ChartConfig.Colors.GRID_MAJOR)
        self.price_widget.getAxis('bottom').setTextPen(ChartConfig.Colors.TEXT_PRIMARY)

        self.price_widget.showGrid(x=False, y=False)

        layout.addWidget(self.price_widget)

        # Oscillator panel
        self.oscillator_widget = pg.PlotWidget()
        self.oscillator_widget.setLabel('left', 'Oscillators', color=ChartConfig.Colors.TEXT_PRIMARY, size='12pt')
        self.oscillator_widget.setFixedHeight(ChartConfig.Dimensions.OSCILLATOR_HEIGHT)
        self.oscillator_widget.setBackground(ChartConfig.Colors.BACKGROUND)

        self.oscillator_widget.getAxis('left').setPen(ChartConfig.Colors.GRID_MAJOR)
        self.oscillator_widget.getAxis('left').setTextPen(ChartConfig.Colors.TEXT_PRIMARY)
        self.oscillator_widget.getAxis('bottom').setPen(ChartConfig.Colors.GRID_MAJOR)
        self.oscillator_widget.getAxis('bottom').setTextPen(ChartConfig.Colors.TEXT_PRIMARY)

        self.oscillator_widget.showGrid(x=False, y=False)
        self.oscillator_widget.setXLink(self.price_widget)

        # Reference lines
        self.oscillator_widget.addLine(y=50, pen=pg.mkPen('#94a3b8', width=1, style=Qt.PenStyle.DashLine))
        self.oscillator_widget.addLine(y=30, pen=pg.mkPen('#dc2626', width=1, style=Qt.PenStyle.DashLine))
        self.oscillator_widget.addLine(y=70, pen=pg.mkPen('#dc2626', width=1, style=Qt.PenStyle.DashLine))

        layout.addWidget(self.oscillator_widget)

        # Volume chart
        self.volume_widget = pg.PlotWidget()
        self.volume_widget.setLabel('left', 'Volume', color=ChartConfig.Colors.TEXT_PRIMARY, size='12pt')
        self.volume_widget.setFixedHeight(ChartConfig.Dimensions.VOLUME_HEIGHT)
        self.volume_widget.setBackground(ChartConfig.Colors.BACKGROUND)

        self.volume_widget.getAxis('left').setPen(ChartConfig.Colors.GRID_MAJOR)
        self.volume_widget.getAxis('left').setTextPen(ChartConfig.Colors.TEXT_PRIMARY)
        self.volume_widget.getAxis('bottom').setPen(ChartConfig.Colors.GRID_MAJOR)
        self.volume_widget.getAxis('bottom').setTextPen(ChartConfig.Colors.TEXT_PRIMARY)

        self.volume_widget.showGrid(x=False, y=False)
        self.volume_widget.setXLink(self.price_widget)

        layout.addWidget(self.volume_widget)

    def load_data_async(self):
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
        self.status_label.setText(f"Loading {symbol} from {self.exchange_combo.currentText()}...")

        self.data_loader = DataLoader(symbol, exchange, interval, start_time, end_time)
        self.data_loader.data_loaded.connect(self.on_data_loaded)
        self.data_loader.error_occurred.connect(self.on_data_error)
        self.data_loader.progress_updated.connect(self.progress_bar.setValue)
        self.data_loader.start()

    def on_data_loaded(self, bars):
        try:
            self.bars = bars
            print(f"📊 Displaying {len(bars)} bars")

            # Store timestamps
            self.timestamps = [bar.datetime for bar in bars]

            # Update array manager with ALL data - FIXED
            self.array_manager = ArrayManager(size=len(bars) + 100)
            for bar in bars:
                self.array_manager.update_bar(bar)

            # Prepare DataFrame for SMC calculations
            self.prepare_smc_dataframe()

            self.display_professional_chart()
            self.update_chart_info()
            self.update_all_indicators()  # This is now fixed
            self.update_all_smc_features()

            self.status_label.setText(f"✅ Loaded {len(bars)} bars successfully")

        except Exception as e:
            self.on_data_error(f"Display error: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.load_btn.setEnabled(True)

    def prepare_smc_dataframe(self):
        """Prepare DataFrame for SMC calculations"""
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
        self.status_label.setText("❌ Error occurred")
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        QMessageBox.warning(self, "Data Loading Error", error_msg)

    def update_chart_info(self):
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
        try:
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
                    i,
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

            print("✅ Professional chart displayed successfully!")

        except Exception as e:
            print(f"❌ Display error: {e}")
            import traceback
            traceback.print_exc()

    def update_indicators(self, indicator_name, params):
        """FIXED: Update traditional indicators using proper calculations"""
        if not self.bars or not self.array_manager.inited:
            return

        try:
            print(f"🔄 Calculating indicator: {indicator_name}")

            # Extract price data
            closes = np.array([bar.close_price for bar in self.bars])
            highs = np.array([bar.high_price for bar in self.bars])
            lows = np.array([bar.low_price for bar in self.bars])

            period = params.get('period', 20)

            # Calculate indicator using proper methods
            if indicator_name == 'EMA':
                indicator_data = IndicatorCalculator.calculate_ema(closes, period)
            elif indicator_name == 'SMA':
                indicator_data = IndicatorCalculator.calculate_sma(closes, period)
            elif indicator_name == 'RSI':
                indicator_data = IndicatorCalculator.calculate_rsi(closes, period)
            elif indicator_name == 'MACD':
                indicator_data = IndicatorCalculator.calculate_macd(closes)
            elif indicator_name == 'KDJ':
                indicator_data = IndicatorCalculator.calculate_kdj(highs, lows, closes, period)
            elif indicator_name == 'Bollinger':
                indicator_data = IndicatorCalculator.calculate_bollinger_upper(closes, period)
            else:
                return

            if indicator_data is not None:
                self.display_indicator(indicator_name, indicator_data, params)
                print(f"✅ Updated {indicator_name} with {len(indicator_data)} values")

        except Exception as e:
            print(f"❌ Error updating indicator {indicator_name}: {e}")
            import traceback
            traceback.print_exc()

    def display_indicator(self, name, data, params):
        """Display indicators on appropriate charts"""
        oscillator_indicators = ['RSI', 'KDJ', 'MACD']

        if name in oscillator_indicators:
            if name in self.oscillator_indicator_items:
                self.oscillator_widget.removeItem(self.oscillator_indicator_items[name])

            # Use configuration colors
            colors = {
                'RSI': ChartConfig.Colors.RSI_COLOR,
                'KDJ': ChartConfig.Colors.MACD_COLOR,
                'MACD': ChartConfig.Colors.MACD_COLOR
            }
            color = colors.get(name, ChartConfig.Colors.RSI_COLOR)

            indicator_item = IndicatorItem(data, color, width=2)
            self.oscillator_widget.addItem(indicator_item)
            self.oscillator_indicator_items[name] = indicator_item

        else:
            if name in self.price_indicator_items:
                self.price_widget.removeItem(self.price_indicator_items[name])

            colors = {
                'EMA': ChartConfig.Colors.EMA_COLOR,
                'SMA': ChartConfig.Colors.SMA_COLOR,
                'Bollinger': ChartConfig.Colors.BOLLINGER_COLOR
            }

            color = colors.get(name, ChartConfig.Colors.EMA_COLOR)

            indicator_item = IndicatorItem(data, color, width=2)
            self.price_widget.addItem(indicator_item)
            self.price_indicator_items[name] = indicator_item

    def update_all_indicators(self):
        """Update all active indicators"""
        active_indicators = self.indicator_panel.get_active_indicators()
        for indicator in active_indicators:
            self.update_indicators(indicator['name'], indicator['params'])

    def update_smc_features(self, feature_name, params):
        """Update SMC features with improved visualization"""
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
        """Update all active SMC features"""
        if SMC_AVAILABLE:
            active_features = self.smc_panel.get_active_smc_features()
            for feature in active_features:
                self.update_smc_features(feature['name'], feature['params'])

    def calculate_smc_feature(self, name, params):
        """Calculate SMC features using configuration"""
        if not SMC_AVAILABLE or self.df_ohlc is None:
            return None

        try:
            if name == 'FVG':
                return smc.fvg(self.df_ohlc, join_consecutive=SMCConfig.FVG_JOIN_CONSECUTIVE)
            elif name == 'SwingHL':
                swing_length = params.get('param1', SMCConfig.SWING_LENGTH)
                return smc.swing_highs_lows(self.df_ohlc, swing_length=swing_length)
            elif name == 'BOS_CHOCH':
                swing_data = smc.swing_highs_lows(self.df_ohlc, swing_length=SMCConfig.SWING_LENGTH)
                return smc.bos_choch(self.df_ohlc, swing_data)
            elif name == 'OrderBlocks':
                swing_data = smc.swing_highs_lows(self.df_ohlc, swing_length=SMCConfig.SWING_LENGTH)
                return smc.ob(self.df_ohlc, swing_data)
            elif name == 'Liquidity':
                swing_data = smc.swing_highs_lows(self.df_ohlc, swing_length=SMCConfig.SWING_LENGTH)
                range_percent = params.get('param1', SMCConfig.LIQUIDITY_RANGE_PERCENT) / 100.0
                return smc.liquidity(self.df_ohlc, swing_data, range_percent=range_percent)
            elif name == 'PrevHL':
                return smc.previous_high_low(self.df_ohlc, time_frame=SMCConfig.DEFAULT_TIMEFRAME)
            elif name == 'Sessions':
                return smc.sessions(self.df_ohlc, session=SMCConfig.DEFAULT_SESSION)
            elif name == 'Retracements':
                swing_data = smc.swing_highs_lows(self.df_ohlc, swing_length=SMCConfig.SWING_LENGTH)
                return smc.retracements(self.df_ohlc, swing_data)
        except Exception as e:
            print(f"❌ SMC calculation error for {name}: {e}")
            return None

        return None

    def display_smc_feature_professional(self, name, data, params):
        """Display SMC features with FIXED label positioning"""
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
                self.display_fvg_improved(data, data_len)
            elif name == 'SwingHL':
                self.display_swing_hl_improved(data, data_len)
            elif name == 'BOS_CHOCH':
                self.display_bos_choch_improved(data, data_len)
            elif name == 'OrderBlocks':
                self.display_order_blocks_improved(data, data_len)
            elif name == 'Liquidity':
                self.display_liquidity_improved(data, data_len)
            elif name == 'PrevHL':
                self.display_prev_hl_improved(data, data_len)
            elif name == 'Sessions':
                self.display_sessions_improved(data, data_len)

        except Exception as e:
            print(f"❌ Error displaying SMC {name}: {e}")

    def display_fvg_improved(self, fvg_data, data_len):
        """Display Fair Value Gaps with improved visualization"""
        rectangles = []

        for i in range(len(fvg_data)):
            if 'FVG' in fvg_data.columns and not pd.isna(fvg_data['FVG'].iloc[i]):
                if 'Top' in fvg_data.columns and 'Bottom' in fvg_data.columns:
                    top = fvg_data['Top'].iloc[i]
                    bottom = fvg_data['Bottom'].iloc[i]

                    if not pd.isna(top) and not pd.isna(bottom):
                        end_idx = data_len - 1
                        if 'MitigatedIndex' in fvg_data.columns:
                            mitigated = fvg_data['MitigatedIndex'].iloc[i]
                            if not pd.isna(mitigated) and mitigated > 0:
                                end_idx = int(mitigated)

                        rectangles.append((i, bottom, end_idx, top, "FVG"))

        if rectangles:
            fvg_item = SMCRectangleItem(rectangles, ChartConfig.Colors.FVG_COLOR, SMCConfig.RECTANGLE_OPACITY)
            self.price_widget.addItem(fvg_item)
            self.smc_items['FVG'] = fvg_item

    def display_swing_hl_improved(self, swing_data, data_len):
        """Display Swing Highs and Lows with fixed labels"""
        lines = []
        text_labels = []

        if 'HighLow' in swing_data.columns and 'Level' in swing_data.columns:
            swing_points = []

            for i in range(len(swing_data)):
                if not pd.isna(swing_data['HighLow'].iloc[i]):
                    swing_points.append((i, swing_data['Level'].iloc[i], swing_data['HighLow'].iloc[i]))

            # Connect swing points with proper labels
            for j in range(len(swing_points) - 1):
                i1, level1, type1 = swing_points[j]
                i2, level2, type2 = swing_points[j + 1]

                lines.append((i1, level1, i2, level2))

                # Fixed label positioning - always above/below based on type
                mid_x = (i1 + i2) / 2
                label_y = level1 + (level2 - level1) / 2
                if type1 == 1:  # High
                    label_y += abs(level2 - level1) * 0.1
                    text_labels.append((mid_x, label_y, "SH"))
                else:  # Low
                    label_y -= abs(level2 - level1) * 0.1
                    text_labels.append((mid_x, label_y, "SL"))

        if lines:
            swing_item = SMCLineItem(lines, ChartConfig.Colors.SWING_HL_COLOR, SMCConfig.LINE_WIDTH,
                                     text_labels=text_labels)
            self.price_widget.addItem(swing_item)
            self.smc_items['SwingHL'] = swing_item

    def display_bos_choch_improved(self, bos_data, data_len):
        """Display BOS/CHOCH with FIXED label positioning"""
        lines = []
        text_labels = []

        for i in range(len(bos_data)):
            # BOS
            if 'BOS' in bos_data.columns and not pd.isna(bos_data['BOS'].iloc[i]):
                if 'BrokenIndex' in bos_data.columns and 'Level' in bos_data.columns:
                    broken_idx = bos_data['BrokenIndex'].iloc[i]
                    level = bos_data['Level'].iloc[i]
                    bos_direction = bos_data['BOS'].iloc[i]

                    if not pd.isna(broken_idx) and not pd.isna(level):
                        lines.append((i, level, int(broken_idx), level))

                        # Fixed positioning based on direction
                        mid_x = (i + int(broken_idx)) / 2
                        label_y = level
                        if bos_direction == 1:  # Bullish
                            label_y += abs(level) * 0.002
                        else:  # Bearish
                            label_y -= abs(level) * 0.002

                        text_labels.append((mid_x, label_y, "BOS"))

            # CHOCH
            if 'CHOCH' in bos_data.columns and not pd.isna(bos_data['CHOCH'].iloc[i]):
                if 'BrokenIndex' in bos_data.columns and 'Level' in bos_data.columns:
                    broken_idx = bos_data['BrokenIndex'].iloc[i]
                    level = bos_data['Level'].iloc[i]
                    choch_direction = bos_data['CHOCH'].iloc[i]

                    if not pd.isna(broken_idx) and not pd.isna(level):
                        lines.append((i, level, int(broken_idx), level))

                        # Fixed positioning based on direction
                        mid_x = (i + int(broken_idx)) / 2
                        label_y = level
                        if choch_direction == 1:  # Bullish
                            label_y += abs(level) * 0.002
                        else:  # Bearish
                            label_y -= abs(level) * 0.002

                        text_labels.append((mid_x, label_y, "CHOCH"))

        if lines:
            # Use different colors for BOS and CHOCH
            bos_item = SMCLineItem(lines, ChartConfig.Colors.BOS_COLOR, SMCConfig.LINE_WIDTH, text_labels=text_labels)
            self.price_widget.addItem(bos_item)
            self.smc_items['BOS_CHOCH'] = bos_item

    def display_order_blocks_improved(self, ob_data, data_len):
        """Display Order Blocks with volume information"""
        rectangles = []

        for i in range(len(ob_data)):
            if 'OB' in ob_data.columns and not pd.isna(ob_data['OB'].iloc[i]):
                if 'Top' in ob_data.columns and 'Bottom' in ob_data.columns:
                    top = ob_data['Top'].iloc[i]
                    bottom = ob_data['Bottom'].iloc[i]

                    if not pd.isna(top) and not pd.isna(bottom):
                        end_idx = data_len - 1
                        if 'MitigatedIndex' in ob_data.columns:
                            mitigated = ob_data['MitigatedIndex'].iloc[i]
                            if not pd.isna(mitigated) and mitigated > 0:
                                end_idx = int(mitigated)

                        # Format volume text
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
            ob_item = SMCRectangleItem(rectangles, ChartConfig.Colors.ORDER_BLOCK_COLOR, SMCConfig.RECTANGLE_OPACITY)
            self.price_widget.addItem(ob_item)
            self.smc_items['OrderBlocks'] = ob_item

    def display_liquidity_improved(self, liq_data, data_len):
        """Display Liquidity with improved label positioning"""
        lines = []
        text_labels = []

        for i in range(len(liq_data)):
            if 'Liquidity' in liq_data.columns and not pd.isna(liq_data['Liquidity'].iloc[i]):
                if 'Level' in liq_data.columns and 'End' in liq_data.columns:
                    level = liq_data['Level'].iloc[i]
                    end_idx = liq_data['End'].iloc[i]
                    liq_direction = liq_data['Liquidity'].iloc[i]

                    if not pd.isna(level) and not pd.isna(end_idx):
                        lines.append((i, level, int(end_idx), level))

                        # Fixed label positioning
                        mid_x = (i + int(end_idx)) / 2
                        label_y = level
                        if liq_direction == 1:  # High liquidity
                            label_y += abs(level) * 0.001
                        else:  # Low liquidity
                            label_y -= abs(level) * 0.001

                        text_labels.append((mid_x, label_y, "LIQ"))

                        # Check for swept liquidity
                        if 'Swept' in liq_data.columns:
                            swept_idx = liq_data['Swept'].iloc[i]
                            if not pd.isna(swept_idx) and swept_idx > 0:
                                swept_idx = int(swept_idx)
                                if swept_idx < len(self.bars):
                                    swept_price = self.bars[swept_idx].high_price if liq_direction == 1 else self.bars[
                                        swept_idx].low_price
                                    lines.append((int(end_idx), level, swept_idx, swept_price))

                                    # Swept label
                                    swept_mid_x = (int(end_idx) + swept_idx) / 2
                                    swept_mid_y = (level + swept_price) / 2
                                    text_labels.append((swept_mid_x, swept_mid_y, "SWEPT"))

        if lines:
            liq_item = SMCLineItem(lines, ChartConfig.Colors.LIQUIDITY_COLOR, SMCConfig.LINE_WIDTH, style='dash',
                                   text_labels=text_labels)
            self.price_widget.addItem(liq_item)
            self.smc_items['Liquidity'] = liq_item

    def display_prev_hl_improved(self, prev_data, data_len):
        """Display Previous High/Low with fixed positioning"""
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

            for j in range(len(high_indexes) - 1):
                lines.append((high_indexes[j], high_levels[j], high_indexes[j + 1], high_levels[j]))
                # Fixed label positioning
                label_y = high_levels[j] + abs(high_levels[j]) * 0.001
                text_labels.append((high_indexes[j + 1], label_y, "PH"))

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

            for j in range(len(low_indexes) - 1):
                lines.append((low_indexes[j], low_levels[j], low_indexes[j + 1], low_levels[j]))
                # Fixed label positioning
                label_y = low_levels[j] - abs(low_levels[j]) * 0.001
                text_labels.append((low_indexes[j + 1], label_y, "PL"))

        if lines:
            prev_item = SMCLineItem(lines, ChartConfig.Colors.PREV_HL_COLOR, SMCConfig.LINE_WIDTH,
                                    text_labels=text_labels)
            self.price_widget.addItem(prev_item)
            self.smc_items['PrevHL'] = prev_item

    def display_sessions_improved(self, sessions_data, data_len):
        """Display Sessions with improved visualization"""
        rectangles = []

        for i in range(len(sessions_data) - 1):
            if 'Active' in sessions_data.columns and sessions_data['Active'].iloc[i] == 1:
                if 'High' in sessions_data.columns and 'Low' in sessions_data.columns:
                    high = sessions_data['High'].iloc[i]
                    low = sessions_data['Low'].iloc[i]

                    if not pd.isna(high) and not pd.isna(low):
                        rectangles.append((i, low, i + 1, high, "SESSION"))

        if rectangles:
            session_item = SMCRectangleItem(rectangles, ChartConfig.Colors.SESSION_COLOR, SMCConfig.RECTANGLE_OPACITY)
            self.price_widget.addItem(session_item)
            self.smc_items['Sessions'] = session_item


def main():
    """Main function"""
    try:
        print("📊 Starting Professional Trading Platform with SMC")
        print("=" * 60)

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        chart = AdvancedHowtraderChart()
        chart.show()

        if SMC_AVAILABLE:
            print("✅ SMC features enabled")
        else:
            print("⚠️ SMC features disabled - library not found")

        return app.exec()

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())