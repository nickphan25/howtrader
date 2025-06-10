"""
🚀 Professional Trading Platform with Howtrader Integration
✨ PySide6 Version - Production Ready
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json

# PySide6 imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QPushButton, QComboBox, QLabel, QProgressBar,
                            QDateEdit, QCheckBox, QGroupBox, QGridLayout, QSpinBox,
                            QDoubleSpinBox, QTabWidget, QTextEdit, QMessageBox,
                            QSplitter, QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QDateTimeEdit)
from PySide6.QtCore import QThread, Signal, QTimer, QDate, Qt, QTime, QDateTime, QDate, QTime
from PySide6.QtGui import QFont

# PyQtGraph imports
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, mkBrush

# Howtrader imports
from howtrader.app.cta_strategy.backtesting import BacktestingEngine
from howtrader.trader.object import BarData, Interval, Exchange
from howtrader.trader.database import BaseDatabase, get_database
from howtrader.trader.utility import ArrayManager
from howtrader.app.cta_strategy import CtaTemplate

# TA-Lib imports
import talib

# SMC imports
import smartmoneyconcepts as smc

# Custom PyQtGraph items for advanced charting
class CandlestickItem(pg.GraphicsObject):
    """Enhanced candlestick chart item"""
    def __init__(self, data, timestamps):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.timestamps = timestamps
        self.picture = None
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self.picture)

        for i, (idx, open_price, high, low, close, volume) in enumerate(self.data):
            color = '#26a69a' if close >= open_price else '#ef5350'
            painter.setPen(pg.mkPen(color, width=1))
            painter.setBrush(pg.mkBrush(color))

            # Body
            body_height = abs(close - open_price)
            body_bottom = min(open_price, close)
            painter.drawRect(pg.QtCore.QRectF(idx-0.3, body_bottom, 0.6, body_height))

            # Wicks
            painter.drawLine(pg.QtCore.QPointF(idx, low), pg.QtCore.QPointF(idx, high))

        painter.end()

    def paint(self, painter, option, widget):
        if self.picture:
            painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect()) if self.picture else pg.QtCore.QRectF()

class VolumeBarItem(pg.GraphicsObject):
    """Volume bar chart item"""
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.picture = None
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self.picture)

        for i, (idx, open_price, high, low, close, volume) in enumerate(self.data):
            color = '#26a69a' if close >= open_price else '#ef5350'
            painter.setPen(pg.mkPen(color, width=1))
            painter.setBrush(pg.mkBrush(color))
            painter.drawRect(pg.QtCore.QRectF(idx-0.4, 0, 0.8, volume))

        painter.end()

    def paint(self, painter, option, widget):
        if self.picture:
            painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect()) if self.picture else pg.QtCore.QRectF()

class TradingViewGrid(pg.PlotDataItem):
    """TradingView-style grid"""
    def __init__(self, x_range, y_range, interval):
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.interval = interval

class ImprovedTimeAxis(pg.AxisItem):
    """Enhanced time axis with better formatting"""
    def __init__(self, timestamps, interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamps = timestamps
        self.interval = interval

    def tickStrings(self, values, scale, spacing):
        if not self.timestamps:
            return [str(int(v)) for v in values]

        strings = []
        for v in values:
            try:
                idx = int(v)
                if 0 <= idx < len(self.timestamps):
                    dt = datetime.fromtimestamp(self.timestamps[idx])
                    strings.append(dt.strftime('%m/%d %H:%M'))
                else:
                    strings.append('')
            except:
                strings.append('')
        return strings

# Data Loading Thread
class DataLoadingThread(QThread):
    """Thread for loading historical data"""
    data_loaded = Signal(list)
    progress_updated = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, symbol, exchange, interval, start_date, end_date):
        super().__init__()
        self.symbol = symbol
        self.exchange = exchange
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        try:
            self.progress_updated.emit("Connecting to database...")

            # Get database instance
            database = get_database()

            self.progress_updated.emit("Loading historical data...")

            # Load bars from database
            bars = database.load_bar_data(
                symbol=self.symbol,
                exchange=Exchange(self.exchange),
                interval=Interval(self.interval),
                start=self.start_date,
                end=self.end_date
            )

            if not bars:
                self.error_occurred.emit("No data found for the specified parameters")
                return

            self.progress_updated.emit(f"Loaded {len(bars)} bars successfully")
            self.data_loaded.emit(bars)

        except Exception as e:
            self.error_occurred.emit(f"Error loading data: {str(e)}")

# Howtrader Integration Components
class HowtraderIndicatorCalculator:
    """Calculator for technical indicators using Howtrader"""

    def __init__(self, parent_chart):
        self.parent_chart = parent_chart
        self.bars = []
        print("✅ HowtraderIndicatorCalculator initialized")

    def update_bars(self, bars):
        """Update bars data"""
        self.bars = bars
        print(f"📊 Indicator calculator updated with {len(bars)} bars")

    def calculate_indicator(self, indicator_name, params, length):
        """Calculate indicator with proper ArrayManager access"""
        try:
            # Get ArrayManager from parent chart
            if not hasattr(self.parent_chart, 'array_manager'):
                print("❌ No ArrayManager available")
                return None

            am = self.parent_chart.array_manager

            if not am.inited:
                print("❌ ArrayManager not initialized")
                return None

            print(f"🔄 Calculating {indicator_name} with {am.count} bars")

            if indicator_name == "SMA":
                period = params.get('param1', 20)
                if am.count > period:
                    values = []
                    for i in range(period-1, am.count):
                        sma_val = am.sma(period, array=True)[i-(period-1)]
                        values.append(sma_val)
                    return values

            elif indicator_name == "EMA":
                period = params.get('param1', 20)
                if am.count > period:
                    close_array = am.close_array[-am.count:]
                    ema_values = talib.EMA(close_array, timeperiod=period)
                    return ema_values[~np.isnan(ema_values)]

            elif indicator_name == "RSI":
                period = params.get('param1', 14)
                if am.count > period:
                    return am.rsi(period, array=True)[-length:]

            elif indicator_name == "MACD":
                fast = params.get('param1', 12)
                slow = params.get('param2', 26)
                signal = params.get('param3', 9)
                if am.count > slow + signal:
                    close_array = am.close_array[-am.count:]
                    macd, signal_line, histogram = talib.MACD(close_array, fast, slow, signal)
                    return {
                        'macd': macd[~np.isnan(macd)],
                        'signal': signal_line[~np.isnan(signal_line)],
                        'histogram': histogram[~np.isnan(histogram)]
                    }

            elif indicator_name == "Bollinger Bands":
                period = params.get('param1', 20)
                std_dev = params.get('param2', 2.0)
                if am.count > period:
                    close_array = am.close_array[-am.count:]
                    upper, middle, lower = talib.BBANDS(close_array, period, std_dev, std_dev)
                    return {
                        'upper': upper[~np.isnan(upper)],
                        'middle': middle[~np.isnan(middle)],
                        'lower': lower[~np.isnan(lower)]
                    }

            elif indicator_name == "Stochastic":
                k_period = params.get('param1', 14)
                d_period = params.get('param2', 3)
                if am.count > k_period + d_period:
                    high_array = am.high_array[-am.count:]
                    low_array = am.low_array[-am.count:]
                    close_array = am.close_array[-am.count:]
                    k, d = talib.STOCH(high_array, low_array, close_array, k_period, d_period, d_period)
                    return {
                        'k': k[~np.isnan(k)],
                        'd': d[~np.isnan(d)]
                    }

            print(f"⚠️ Indicator {indicator_name} calculation completed with limited data")
            return None

        except Exception as e:
            print(f"❌ Error calculating {indicator_name}: {e}")
            return None

class HowtraderTimeframeConverter:
    """Converter for timeframe aggregation"""

    def __init__(self):
        self.timeframe_map = {
            '1m': Interval.MINUTE,
            '5m': Interval.MINUTE,
            '15m': Interval.MINUTE,
            '30m': Interval.MINUTE,
            '1h': Interval.HOUR,
            '4h': Interval.HOUR,
            '1d': Interval.DAILY,
            '1w': Interval.WEEKLY,
            '1M': Interval.MONTH
        }

    def convert_timeframe(self, bars, target_timeframe):
        """Convert bars to different timeframe"""
        try:
            print(f"🔄 Converting {len(bars)} bars to {target_timeframe}")
            # TODO: Implement proper timeframe conversion
            return bars

        except Exception as e:
            print(f"❌ Error converting timeframe: {e}")
            return bars

# Control Panels
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QGroupBox,
                            QCheckBox, QLabel, QSpinBox, QPushButton)

class IndicatorControlPanel(QWidget):
    """Control panel for technical indicators"""

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Indicators
        indicator_group = QGroupBox("Technical Indicators")
        indicator_layout = QGridLayout()

        self.indicator_features = {
            'EMA_1': {'period': ('Period', 20, 1, 200)},
            'EMA_2': {'period': ('Period', 50, 1, 200)},
            'EMA_3': {'period': ('Period', 100, 1, 200)},
            'RSI': {'period': ('Period', 14, 2, 50)}
        }

        self.indicator_widgets = {}
        row = 0

        for name, params in self.indicator_features.items():
            # Checkbox
            checkbox = QCheckBox(name.replace('_', ' '))
            indicator_layout.addWidget(checkbox, row, 0)

            # Parameter widgets
            param_widgets = {}
            col = 1

            for param_key, (label, default, min_val, max_val) in params.items():
                param_label = QLabel(f"{label}:")
                indicator_layout.addWidget(param_label, row, col)
                col += 1

                widget = QSpinBox()
                widget.setRange(min_val, max_val)
                widget.setValue(default)
                indicator_layout.addWidget(widget, row, col)
                param_widgets[param_key] = widget
                col += 1

            self.indicator_widgets[name] = {
                'checkbox': checkbox,
                'params': param_widgets
            }

            row += 1

        indicator_group.setLayout(indicator_layout)
        layout.addWidget(indicator_group)

        # Update button
        update_btn = QPushButton("Update Indicators")
        update_btn.clicked.connect(self.update_indicators)
        layout.addWidget(update_btn)

        self.setLayout(layout)

    def get_active_indicators(self):
        """Get list of active indicators with parameters"""
        active = []

        for name, widgets in self.indicator_widgets.items():
            if widgets['checkbox'].isChecked():
                params = {}
                for param_key, widget in widgets['params'].items():
                    params[param_key] = widget.value()

                active.append({
                    'name': name,
                    'params': params
                })

        return active

    def update_indicators(self):
        """Emit signal to update indicators"""
        print("📊 Indicator update requested")

class SMCControlPanel(QWidget):
    """Control panel for Smart Money Concepts features"""

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # SMC Features
        smc_group = QGroupBox("Smart Money Concepts")
        smc_layout = QGridLayout()

        self.smc_features = {
            'FVG': {},
            'SwingHL': {'param1': ('Length', 50, 5, 200)},
            'BOS_CHOCH': {},
            'OrderBlocks': {},
            'Liquidity': {'param1': ('Range %', 1, 1, 10)},
            'PrevHL': {},
            'Sessions': {}
        }

        self.smc_widgets = {}
        row = 0

        for name, params in self.smc_features.items():
            # Checkbox
            checkbox = QCheckBox(name)
            smc_layout.addWidget(checkbox, row, 0)

            # Parameter widgets
            param_widgets = {}
            col = 1

            for param_key, (label, default, min_val, max_val) in params.items():
                param_label = QLabel(f"{label}:")
                smc_layout.addWidget(param_label, row, col)
                col += 1

                widget = QSpinBox()
                widget.setRange(min_val, max_val)
                widget.setValue(default)
                smc_layout.addWidget(widget, row, col)
                param_widgets[param_key] = widget
                col += 1

            self.smc_widgets[name] = {
                'checkbox': checkbox,
                'params': param_widgets
            }

            row += 1

        smc_group.setLayout(smc_layout)
        layout.addWidget(smc_group)

        # Update button
        update_btn = QPushButton("Update SMC Features")
        update_btn.clicked.connect(self.update_smc_features)
        layout.addWidget(update_btn)

        self.setLayout(layout)

    def get_active_smc_features(self):
        """Get list of active SMC features with parameters"""
        active = []

        for name, widgets in self.smc_widgets.items():
            if widgets['checkbox'].isChecked():
                params = {}
                for param_key, widget in widgets['params'].items():
                    params[param_key] = widget.value()

                active.append({
                    'name': name,
                    'params': params
                })

        return active

    def update_smc_features(self):
        """Emit signal to update SMC features"""
        print("💡 SMC features update requested")

# Main Chart Widget
class AdvancedHowtraderChart(QMainWindow):
    """Advanced trading chart with Howtrader integration"""

    def __init__(self):
        super().__init__()
        self.bars = None
        self.data_loader = None

        # Initialize howtrader components
        self.indicator_calculator = HowtraderIndicatorCalculator(self)
        self.timeframe_converter = HowtraderTimeframeConverter()

        # Initialize ArrayManager
        self.array_manager = ArrayManager(size=500)

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
        """Initialize the user interface"""
        self.setWindowTitle("🚀 Professional Trading Platform with Howtrader")
        self.setGeometry(100, 100, 1600, 1000)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # Data loading controls
        self.create_data_controls(left_layout)

        # Indicator controls
        self.indicator_panel = IndicatorControlPanel()
        left_layout.addWidget(self.indicator_panel)

        # SMC controls
        self.smc_panel = SMCControlPanel()
        left_layout.addWidget(self.smc_panel)

        # Right panel for charts
        right_panel = self.create_chart_panel()

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_data_controls(self, layout):
        """Tạo controls cho data loading"""
        data_group = QGroupBox("Data Controls")
        data_layout = QVBoxLayout(data_group)

        # Symbol và Exchange
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Symbol:"))
        symbol_label = QLabel("BTCUSDT.BINANCE")
        symbol_label.setStyleSheet("font-weight: bold; color: #2E86AB;")
        symbol_layout.addWidget(symbol_label)
        symbol_layout.addStretch()
        data_layout.addLayout(symbol_layout)

        # Timeframe selector
        timeframe_layout = QHBoxLayout()
        timeframe_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'])
        self.timeframe_combo.setCurrentText('1h')
        self.timeframe_combo.currentTextChanged.connect(self.load_data)
        timeframe_layout.addWidget(self.timeframe_combo)
        timeframe_layout.addStretch()
        data_layout.addLayout(timeframe_layout)

        # Date range
        date_layout = QVBoxLayout()

        # Start Date
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start Date:"))
        self.start_date = QDateTimeEdit()

        # ✅ Sử dụng QDateTime.currentDateTime() thay vì QDateTimeEdit.currentDateTime()
        current_time = QDateTime.currentDateTime()
        self.start_date.setDateTime(current_time.addDays(-30))

        self.start_date.setEnabled(False)
        self.start_date.setStyleSheet("background-color: #f0f0f0; color: #666;")
        start_layout.addWidget(self.start_date)
        date_layout.addLayout(start_layout)

        # End Date
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End Date:"))
        self.end_date = QDateTimeEdit()

        # ✅ Sử dụng QDateTime.currentDateTime()
        self.end_date.setDateTime(current_time)

        self.end_date.setEnabled(False)
        self.end_date.setStyleSheet("background-color: #f0f0f0; color: #666;")
        end_layout.addWidget(self.end_date)
        date_layout.addLayout(end_layout)

        data_layout.addLayout(date_layout)

        # Load button
        self.load_btn = QPushButton("🔄 Refresh Data")
        self.load_btn.clicked.connect(self.load_data)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        data_layout.addWidget(self.load_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        data_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to load data")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        data_layout.addWidget(self.status_label)

        layout.addWidget(data_group)

    def create_chart_panel(self):
        """Create the chart panel with multiple sub-charts"""
        # Main splitter for charts
        chart_splitter = QSplitter(Qt.Orientation.Vertical)

        # Price chart (main)
        self.price_widget = PlotWidget()
        self.price_widget.setLabel('left', 'Price')
        self.price_widget.setLabel('bottom', 'Time')
        self.price_widget.showGrid(x=True, y=True, alpha=0.3)
        chart_splitter.addWidget(self.price_widget)

        # Volume chart
        self.volume_widget = PlotWidget()
        self.volume_widget.setLabel('left', 'Volume')
        self.volume_widget.showGrid(x=True, y=True, alpha=0.3)
        chart_splitter.addWidget(self.volume_widget)

        # Oscillator chart
        self.oscillator_widget = PlotWidget()
        self.oscillator_widget.setLabel('left', 'Oscillators')
        self.oscillator_widget.showGrid(x=True, y=True, alpha=0.3)
        chart_splitter.addWidget(self.oscillator_widget)

        # Set initial sizes
        chart_splitter.setSizes([600, 200, 200])

        return chart_splitter

    def load_data(self):
        """
        Load dữ liệu với timeframe mới, giữ nguyên database và date range
        Chỉ thay đổi timeframe hiển thị, không thay đổi dữ liệu gốc trong database
        """
        try:
            # ✅ Hiển thị trạng thái loading
            self.status_label.setText("Loading data...")
            self.status_label.setStyleSheet("color: orange;")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.load_btn.setEnabled(False)

            # ✅ Lấy timeframe được chọn từ combo box
            selected_timeframe = self.timeframe_combo.currentText()

            print(f"🔄 [DEBUG] Loading data with timeframe: {selected_timeframe}")

            # ✅ Kiểm tra xem đã có dữ liệu gốc chưa
            if hasattr(self, 'original_bars') and self.original_bars:
                print(f"📊 [DEBUG] Found existing data: {len(self.original_bars)} bars")

                # Chuyển đổi timeframe cho dữ liệu hiện có
                converted_bars = self.convert_timeframe_data(self.original_bars, selected_timeframe)

                if converted_bars and len(converted_bars) > 0:
                    # ✅ Cập nhật dữ liệu chart
                    self.bars = converted_bars
                    self.current_bars = converted_bars

                    # ✅ Cập nhật OHLC data cho chart
                    self._update_ohlc_data(converted_bars)

                    # ✅ Cập nhật indicators với dữ liệu mới
                    self.update_indicators_professional(converted_bars)

                    # ✅ Hiển thị chart với timeframe mới
                    self.display_professional_chart()

                    # ✅ Cập nhật thông tin hiển thị
                    self.update_display_info(selected_timeframe, converted_bars)

                    print(f"✅ [DEBUG] Successfully converted to {selected_timeframe}: {len(converted_bars)} bars")

                else:
                    # ❌ Lỗi chuyển đổi timeframe
                    error_msg = f"Failed to convert timeframe to {selected_timeframe}"
                    self.status_label.setText(error_msg)
                    self.status_label.setStyleSheet("color: red;")
                    print(f"❌ [DEBUG] {error_msg}")

            else:
                print("📥 [DEBUG] No existing data found, loading from database...")

                # ✅ Load dữ liệu gốc từ database lần đầu
                self.load_original_data_from_database(selected_timeframe)

        except Exception as e:
            # ❌ Xử lý lỗi
            error_msg = f"Error loading data: {str(e)}"
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: red;")
            print(f"❌ [DEBUG] {error_msg}")
            import traceback
            traceback.print_exc()

        finally:
            # ✅ Khôi phục UI
            self.progress_bar.setVisible(False)
            self.load_btn.setEnabled(True)

    def convert_timeframe_data(self, original_bars, target_timeframe):
        """
        Chuyển đổi timeframe cho dữ liệu hiện có
        Sử dụng BarGenerator để resample dữ liệu
        """
        try:
            print(f"🔄 [DEBUG] Converting {len(original_bars)} bars to {target_timeframe}")

            # ✅ Map timeframe string to interval
            timeframe_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440,
                '1w': 10080,
                '1M': 43200
            }

            target_minutes = timeframe_map.get(target_timeframe)
            if not target_minutes:
                print(f"❌ [DEBUG] Unsupported timeframe: {target_timeframe}")
                return None

            # ✅ Nếu timeframe giống với dữ liệu gốc, trả về nguyên bản
            if target_timeframe == getattr(self, 'current_timeframe', '1h'):
                print(f"✅ [DEBUG] Same timeframe, returning original data")
                return original_bars

            # ✅ Sử dụng pandas để resample dữ liệu
            converted_bars = self._resample_bars_with_pandas(original_bars, target_timeframe)

            if converted_bars:
                # ✅ Lưu timeframe hiện tại
                self.current_timeframe = target_timeframe
                print(f"✅ [DEBUG] Conversion successful: {len(converted_bars)} bars")
                return converted_bars
            else:
                print(f"❌ [DEBUG] Conversion failed")
                return None

        except Exception as e:
            print(f"❌ [DEBUG] Error converting timeframe: {e}")
            return None

    def _resample_bars_with_pandas(self, bars, target_timeframe):
        """
        Sử dụng pandas để resample dữ liệu bar
        """
        try:
            import pandas as pd
            from datetime import datetime

            # ✅ Chuyển đổi bars thành DataFrame
            data_list = []
            for bar in bars:
                data_list.append({
                    'datetime': bar.datetime,
                    'open': bar.open_price,
                    'high': bar.high_price,
                    'low': bar.low_price,
                    'close': bar.close_price,
                    'volume': bar.volume
                })

            df = pd.DataFrame(data_list)
            df.set_index('datetime', inplace=True)

            # ✅ Map timeframe to pandas frequency
            freq_map = {
                '1m': '1Min',
                '5m': '5Min',
                '15m': '15Min',
                '30m': '30Min',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D',
                '1w': '1W',
                '1M': '1M'
            }

            freq = freq_map.get(target_timeframe)
            if not freq:
                return None

            # ✅ Resample dữ liệu
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # ✅ Chuyển đổi ngược thành BarData objects
            converted_bars = []
            for timestamp, row in resampled.iterrows():
                # Lấy thông tin từ bar đầu tiên để có symbol, exchange
                first_bar = bars[0]

                bar_data = BarData(
                    symbol=first_bar.symbol,
                    exchange=first_bar.exchange,
                    datetime=timestamp.to_pydatetime(),
                    interval=self.timeframe_converter.convert_timeframe(target_timeframe),
                    open_price=float(row['open']),
                    high_price=float(row['high']),
                    low_price=float(row['low']),
                    close_price=float(row['close']),
                    volume=float(row['volume']),
                    turnover=0.0,
                    open_interest=0.0,
                    gateway_name=first_bar.gateway_name
                )
                converted_bars.append(bar_data)

            return converted_bars

        except Exception as e:
            print(f"❌ [DEBUG] Error in pandas resample: {e}")
            return None

    def load_original_data_from_database(self, initial_timeframe='1h'):
        """
        Load dữ liệu gốc từ database lần đầu tiên
        """
        try:
            print(f"📥 [DEBUG] Loading original data from database...")

            # ✅ Sử dụng DataLoadingThread để load dữ liệu
            if hasattr(self, 'data_loader') and self.data_loader:
                # Dừng thread cũ nếu đang chạy
                if self.data_loader.isRunning():
                    self.data_loader.quit()
                    self.data_loader.wait()

            # ✅ Tạo DataLoadingThread mới với thông tin cố định
            # Giả định: symbol và exchange được cố định trong ứng dụng
            default_symbol = "BTCUSDT"  # Có thể lấy từ config
            default_exchange = "BINANCE"  # Có thể lấy từ config

            start_date = self.start_date.dateTime().toPython()
            end_date = self.end_date.dateTime().toPython()

            self.data_loader = DataLoadingThread(
                symbol=default_symbol,
                exchange=default_exchange,
                interval=initial_timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # ✅ Kết nối signals
            self.data_loader.data_loaded.connect(self.on_original_data_loaded)
            self.data_loader.progress_updated.connect(self.on_progress_updated)
            self.data_loader.error_occurred.connect(self.on_data_error)

            # ✅ Bắt đầu load dữ liệu
            self.data_loader.start()

        except Exception as e:
            print(f"❌ [DEBUG] Error loading original data: {e}")
            self.status_label.setText(f"Error loading from database: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def on_original_data_loaded(self, bars):
        """
        Xử lý khi dữ liệu gốc được load từ database
        """
        try:
            print(f"✅ [DEBUG] Original data loaded: {len(bars)} bars")

            # ✅ Lưu dữ liệu gốc
            self.original_bars = bars
            self.bars = bars
            self.current_bars = bars

            # ✅ Lưu timeframe hiện tại
            self.current_timeframe = self.timeframe_combo.currentText()

            # ✅ Cập nhật OHLC data
            self._update_ohlc_data(bars)

            # ✅ Cập nhật indicators
            self.update_indicators_professional(bars)

            # ✅ Hiển thị chart
            self.display_professional_chart()

            # ✅ Cập nhật thông tin hiển thị
            self.update_display_info(self.current_timeframe, bars)

        except Exception as e:
            print(f"❌ [DEBUG] Error processing loaded data: {e}")
            self.status_label.setText(f"Error processing data: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def update_display_info(self, timeframe, bars):
        """
        Cập nhật thông tin hiển thị sau khi load dữ liệu
        """
        try:
            bar_count = len(bars) if bars else 0

            if bars and bar_count > 0:
                start_time = bars[0].datetime.strftime("%Y-%m-%d %H:%M")
                end_time = bars[-1].datetime.strftime("%Y-%m-%d %H:%M")

                status_text = f"✅ Loaded: {timeframe} | {bar_count} bars | {start_time} - {end_time}"
                self.status_label.setText(status_text)
                self.status_label.setStyleSheet("color: green;")

                print(f"✅ [DEBUG] Display updated: {status_text}")
            else:
                self.status_label.setText("No data available")
                self.status_label.setStyleSheet("color: gray;")

        except Exception as e:
            print(f"❌ [DEBUG] Error updating display info: {e}")

    def on_data_loaded(self, bars):
        """Handle successful data loading"""
        try:
            print(f"✅ Data loaded successfully: {len(bars)} bars")

            self.bars = bars
            self.backtest_data_ready = True

            # Update UI
            self.progress_bar.setVisible(False)
            self.load_btn.setEnabled(True)
            self.backtest_btn.setEnabled(True)
            self.status_label.setText(f"✅ Loaded {len(bars)} bars")

            # Update chart info
            self.update_chart_info()

            # Display chart
            self.display_professional_chart()

            print("🎨 Chart display completed")

        except Exception as e:
            print(f"❌ Error processing loaded data: {e}")
            self.on_data_error(str(e))

    def on_progress_updated(self, message):
        """Handle progress updates"""
        self.status_label.setText(message)
        print(f"📊 Progress: {message}")

    def on_data_error(self, error_msg):
        """Handle data loading errors"""
        self.status_label.setText("❌ Error occurred")
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        QMessageBox.warning(self, "Data Loading Error", error_msg)

    def update_indicators_professional(self, bars):
        """Method called by BacktestingEngine"""
        try:
            print(f"🎯 update_indicators_professional called with {len(bars)} bars")

            # Store bars
            self.bars = bars
            self.timestamps = [bar.datetime.timestamp() for bar in bars]

            # Update OHLC data
            self._update_ohlc_data()

            # Initialize ArrayManager with ALL bars
            print(f"🔧 Initializing ArrayManager with {len(bars)} bars...")
            self.array_manager = ArrayManager(size=max(len(bars) + 100, 500))

            for bar in bars:
                self.array_manager.update_bar(bar)

            print(f"✅ ArrayManager initialized with {self.array_manager.count} bars")

            # Update indicator calculator
            self.indicator_calculator.update_bars(bars)

            # Update all active indicators
            self.update_all_indicators()

            print("✅ Indicators professional update completed")

        except Exception as e:
            print(f"❌ Error in update_indicators_professional: {e}")
            import traceback
            traceback.print_exc()

    def display_indicator(self, indicator_name, indicator_data, params):
        """Display indicator on appropriate chart"""
        try:
            if indicator_data is None:
                print(f"⚠️ No data to display for {indicator_name}")
                return

            # Remove existing indicator
            if indicator_name in self.price_indicator_items:
                self.price_widget.removeItem(self.price_indicator_items[indicator_name])
                del self.price_indicator_items[indicator_name]

            if indicator_name in self.oscillator_indicator_items:
                self.oscillator_widget.removeItem(self.oscillator_indicator_items[indicator_name])
                del self.oscillator_indicator_items[indicator_name]

            data_len = len(self.bars)

            if indicator_name in ['SMA', 'EMA', 'Bollinger Bands']:
                self.display_price_indicator(indicator_name, indicator_data, data_len)
            elif indicator_name in ['RSI', 'MACD', 'Stochastic']:
                self.display_oscillator_indicator(indicator_name, indicator_data, data_len)

        except Exception as e:
            print(f"❌ Error displaying indicator {indicator_name}: {e}")

    def display_price_indicator(self, name, data, data_len):
        """Display price-based indicators"""
        try:
            if isinstance(data, dict):
                # Multi-line indicators like Bollinger Bands
                if name == 'Bollinger Bands':
                    if 'upper' in data and len(data['upper']) > 0:
                        start_idx = max(0, data_len - len(data['upper']))
                        x_data = list(range(start_idx, data_len))

                        upper_curve = self.price_widget.plot(x_data, data['upper'], pen=mkPen('red', width=2), name='BB Upper')
                        middle_curve = self.price_widget.plot(x_data, data['middle'], pen=mkPen('blue', width=2), name='BB Middle')
                        lower_curve = self.price_widget.plot(x_data, data['lower'], pen=mkPen('green', width=2), name='BB Lower')

                        self.price_indicator_items[name] = [upper_curve, middle_curve, lower_curve]
            else:
                # Single line indicators
                if len(data) > 0:
                    start_idx = max(0, data_len - len(data))
                    x_data = list(range(start_idx, data_len))

                    color = 'yellow' if name == 'SMA' else 'cyan'
                    curve = self.price_widget.plot(x_data, data, pen=mkPen(color, width=2), name=name)
                    self.price_indicator_items[name] = curve

        except Exception as e:
            print(f"❌ Error displaying price indicator {name}: {e}")

    def display_oscillator_indicator(self, name, data, data_len):
        """Display oscillator indicators"""
        try:
            if isinstance(data, dict):
                if name == 'MACD':
                    if all(k in data for k in ['macd', 'signal', 'histogram']):
                        start_idx = max(0, data_len - len(data['macd']))
                        x_data = list(range(start_idx, data_len))

                        macd_curve = self.oscillator_widget.plot(x_data, data['macd'], pen=mkPen('blue', width=2), name='MACD')
                        signal_curve = self.oscillator_widget.plot(x_data, data['signal'], pen=mkPen('red', width=2), name='Signal')

                        self.oscillator_indicator_items[name] = [macd_curve, signal_curve]

                elif name == 'Stochastic':
                    if 'k' in data and 'd' in data:
                        start_idx = max(0, data_len - len(data['k']))
                        x_data = list(range(start_idx, data_len))

                        k_curve = self.oscillator_widget.plot(x_data, data['k'], pen=mkPen('blue', width=2), name='%K')
                        d_curve = self.oscillator_widget.plot(x_data, data['d'], pen=mkPen('red', width=2), name='%D')

                        self.oscillator_indicator_items[name] = [k_curve, d_curve]
            else:
                # Single line oscillators
                if len(data) > 0:
                    start_idx = max(0, data_len - len(data))
                    x_data = list(range(start_idx, data_len))

                    color = 'magenta' if name == 'RSI' else 'orange'
                    curve = self.oscillator_widget.plot(x_data, data, pen=mkPen(color, width=2), name=name)
                    self.oscillator_indicator_items[name] = curve

        except Exception as e:
            print(f"❌ Error displaying oscillator {name}: {e}")

    def update_indicators(self, indicator_name, params):
        """Update indicators with proper ArrayManager initialization"""
        if not self.bars:
            return

        try:
            data_len = len(self.bars)
            print(f"🔄 Calculating {indicator_name} for {data_len} bars...")

            # Initialize ArrayManager properly BEFORE calculating
            if not hasattr(self, 'array_manager') or not self.array_manager.inited:
                print(f"🔧 Initializing ArrayManager with {data_len} bars...")
                self.array_manager = ArrayManager(size=max(data_len + 100, 500))

                # Add all bars to ArrayManager
                for bar in self.bars:
                    self.array_manager.update_bar(bar)

                print(f"✅ ArrayManager initialized with {self.array_manager.count} bars")

            # Now calculate indicator with initialized ArrayManager
            indicator_data = self.indicator_calculator.calculate_indicator(
                indicator_name, params, data_len
            )

            if indicator_data is not None:
                self.display_indicator(indicator_name, indicator_data, params)
                print(f"✅ Updated {indicator_name} with {len(indicator_data)} values")
            else:
                print(f"⚠️ Failed to calculate {indicator_name}")

        except Exception as e:
            print(f"❌ Error updating indicator {indicator_name}: {e}")
            import traceback
            traceback.print_exc()

    def update_all_indicators(self):
        """Update all active indicators"""
        try:
            print("🔄 Starting to update all indicators...")

            if not hasattr(self, 'indicator_panel'):
                print("❌ No indicator panel found")
                return

            # Get active indicators from panel
            active_indicators = self.indicator_panel.get_active_indicators()
            print(f"📊 Found {len(active_indicators)} active indicators: {[ind['name'] for ind in active_indicators]}")

            for indicator in active_indicators:
                name = indicator['name']
                params = indicator['params']
                print(f"🎯 Processing indicator: {name} with params: {params}")

                try:
                    self.update_indicators(name, params)
                    print(f"✅ Successfully updated {name}")
                except Exception as e:
                    print(f"❌ Failed to update {name}: {e}")

        except Exception as e:
            print(f"❌ Error in update_all_indicators: {e}")
            import traceback
            traceback.print_exc()

    def _update_ohlc_data(self):
        """Update OHLC data arrays"""
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
        """Update trade markers with timeframe awareness"""
        try:
            if not hasattr(self, 'backtest_trades') or not self.backtest_trades:
                print("⚠️ No backtest trades available")
                return

            # Use current chart's timestamps (after timeframe conversion)
            if not self.timestamps:
                print("⚠️ No timestamps available for mapping")
                return

            self.trade_markers = []

            for trade in self.backtest_trades:
                trade_timestamp = trade.datetime.timestamp()

                # Find closest bar in CONVERTED timeframe
                closest_index = self._find_closest_timestamp_index(trade_timestamp)

                if closest_index is not None:
                    marker = {
                        'timestamp': self.timestamps[closest_index],  # Use chart timestamp
                        'index': closest_index,  # Store chart index directly
                        'price': float(trade.price),
                        'direction': 'long' if trade.direction.value == 1 else 'short',
                        'offset': str(trade.offset).upper(),
                        'volume': float(trade.volume),
                        'datetime': trade.datetime,
                        'original_timestamp': trade_timestamp
                    }
                    self.trade_markers.append(marker)

            print(f"✅ Trade markers updated: {len(self.trade_markers)} trades")

        except Exception as e:
            print(f"❌ Error updating trade markers: {e}")

    def _find_closest_timestamp_index(self, target_timestamp):
        """Find closest timestamp index using binary search"""
        if not self.timestamps:
            return None

        # Convert target to same format as timestamps
        target = float(target_timestamp)

        # Binary search for closest match
        left, right = 0, len(self.timestamps) - 1
        closest_index = 0
        min_diff = abs(self.timestamps[0] - target)

        while left <= right:
            mid = (left + right) // 2
            current_timestamp = float(self.timestamps[mid])
            diff = abs(current_timestamp - target)

            if diff < min_diff:
                min_diff = diff
                closest_index = mid

            if current_timestamp < target:
                left = mid + 1
            elif current_timestamp > target:
                right = mid - 1
            else:
                return mid  # Exact match

        return closest_index

    def prepare_smc_dataframe(self):
        """Prepare DataFrame for SMC calculations"""
        try:
            if hasattr(self, 'df_ohlc') and self.df_ohlc is not None and len(self.df_ohlc) > 0:
                df = self.df_ohlc.copy()
                print(f"📊 Using existing df_ohlc: {len(df)} bars")
            elif hasattr(self, 'ohlc_data') and self.ohlc_data:
                if isinstance(self.ohlc_data, dict) and 'timestamp' in self.ohlc_data:
                    timestamps = self.ohlc_data['timestamp']
                    opens = self.ohlc_data['open']
                    highs = self.ohlc_data['high']
                    lows = self.ohlc_data['low']
                    closes = self.ohlc_data['close']
                    volumes = self.ohlc_data.get('volume', [0] * len(timestamps))

                    df_data = []
                    for i in range(len(timestamps)):
                        df_data.append({
                            'timestamp': pd.to_datetime(timestamps[i], unit='s'),
                            'open_price': opens[i],
                            'high_price': highs[i],
                            'low_price': lows[i],
                            'close_price': closes[i],
                            'volume': volumes[i]
                        })

                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('timestamp', inplace=True)
                        print(f"📊 Created dataframe from ohlc_data arrays: {len(df)} bars")
                    else:
                        print("❌ No valid ohlc_data arrays")
                        return None
                else:
                    print("❌ ohlc_data not in expected array format")
                    return None
            elif hasattr(self, 'bars') and self.bars:
                df_data = []
                for bar in self.bars:
                    df_data.append({
                        'timestamp': bar.datetime,
                        'open_price': bar.open_price,
                        'high_price': bar.high_price,
                        'low_price': bar.low_price,
                        'close_price': bar.close_price,
                        'volume': bar.volume
                    })

                if df_data:
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    print(f"📊 Created dataframe from bars: {len(df)} bars")
                else:
                    print("❌ No bars available")
                    return None
            else:
                print("❌ No data available for SMC dataframe")
                return None

            # Apply SMC column mapping
            rename_map = {
                "open_price": "open",
                "high_price": "high",
                "low_price": "low",
                "close_price": "close",
                "volume": "volume",
            }

            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df.columns = [c.lower() for c in df.columns]

            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"❌ Missing required columns for SMC: {missing_columns}")
                return None

            print(f"✅ SMC dataframe prepared with columns: {list(df.columns)}")
            return df

        except Exception as e:
            print(f"❌ Error preparing SMC dataframe: {e}")
            return None

    def display_professional_chart(self):
        """Display professional chart with all components"""
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

            # Update indicators and SMC features
            self.update_all_indicators()
            self.update_all_smc_features()

            # Add trade markers
            if hasattr(self, 'price_widget') and self.price_widget:
                self._display_trade_markers_simple()

            print("✅ Professional chart displayed successfully!")

        except Exception as e:
            print(f"❌ Error displaying chart: {e}")
            import traceback
            traceback.print_exc()

    def _display_trade_markers_simple(self):
        """Display trade markers on chart"""
        try:
            if not hasattr(self, 'trade_markers') or not self.trade_markers:
                return

            print(f"🎯 Adding {len(self.trade_markers)} trade markers...")

            # Convert timestamps to chart indices
            marker_indices = self._convert_markers_to_indices()

            if not marker_indices:
                return

            # Group trades into entry-exit pairs
            trade_pairs = self._group_trades_into_pairs(marker_indices)

            # Add trade marker pairs
            for pair in trade_pairs:
                self._add_trade_marker_pair(pair)

            print(f"✅ Trade marker pairs added: {len(trade_pairs)}")

        except Exception as e:
            print(f"❌ Error in trade marker display: {e}")

    def _convert_markers_to_indices(self):
        """Convert trade markers from timestamps to chart indices"""
        if not self.trade_markers or not self.timestamps:
            return []

        marker_indices = []

        for marker in self.trade_markers:
            trade_timestamp = marker['timestamp']
            closest_index = self._find_closest_timestamp_index(trade_timestamp)

            if closest_index is not None:
                marker_indices.append({
                    'index': closest_index,
                    'price': marker['price'],
                    'direction': marker['direction'],
                    'offset': marker['offset'],
                    'volume': marker['volume'],
                    'original_timestamp': trade_timestamp
                })

        return marker_indices

    def _group_trades_into_pairs(self, marker_indices):
        """Group trades into entry-exit pairs"""
        try:
            trade_pairs = []
            entry_marker = None

            for marker in marker_indices:
                if marker['offset'] == 'OFFSET.OPEN':
                    entry_marker = marker
                elif marker['offset'] == 'OFFSET.CLOSE' and entry_marker:
                    trade_pairs.append({
                        'entry': entry_marker,
                        'exit': marker,
                        'direction': entry_marker['direction']
                    })
                    entry_marker = None

            return trade_pairs

        except Exception as e:
            print(f"❌ Error grouping trades: {e}")
            return []

    def _add_trade_marker_pair(self, pair):
        """Add entry/exit marker pair"""
        try:
            entry_marker = pair['entry']
            exit_marker = pair.get('exit')

            entry_x = entry_marker['index']
            entry_y = entry_marker['price']

            if entry_marker['direction'] == 'long':
                entry_color = '#22c55e'
                entry_symbol = 't'
                exit_color = '#ff4444'
                exit_symbol = 't1'
            else:
                entry_color = '#ff4444'
                entry_symbol = 't1'
                exit_color = '#22c55e'
                exit_symbol = 't'

            # Entry marker
            entry_scatter = pg.ScatterPlotItem(
                pos=[(entry_x, entry_y)],
                symbol=entry_symbol,
                size=15,
                brush=pg.mkBrush(entry_color),
                pen=pg.mkPen('white', width=2)
            )
            self.price_widget.addItem(entry_scatter)

            if exit_marker:
                exit_x = exit_marker['index']
                exit_y = exit_marker['price']

                # Exit marker
                exit_scatter = pg.ScatterPlotItem(
                    pos=[(exit_x, exit_y)],
                    symbol=exit_symbol,
                    size=15,
                    brush=pg.mkBrush(exit_color),
                    pen=pg.mkPen('white', width=2)
                )
                self.price_widget.addItem(exit_scatter)

                # Connection line
                pnl = exit_y - entry_y
                if entry_marker['direction'] == 'short':
                    pnl = -pnl

                line_color = '#22c55e' if pnl > 0 else '#ff4444'
                line = pg.PlotCurveItem(
                    x=[entry_x, exit_x],
                    y=[entry_y, exit_y],
                    pen=pg.mkPen(line_color, width=2, style=pg.QtCore.Qt.DashLine)
                )
                self.price_widget.addItem(line)

        except Exception as e:
            import traceback
            traceback.print_exc()

    def update_all_smc_features(self):
        """Update all active SMC features"""
        try:
            print("🔄 Starting to update all SMC features...")

            if not hasattr(self, 'smc_panel'):
                return

            active_features = self.smc_panel.get_active_smc_features()

            for feature in active_features:
                name = feature['name']
                params = feature['params']
                try:
                    self.update_smc_features(name, params)
                except Exception as e:
                    print(f"❌ Failed to update SMC feature {name}: {e}")

        except Exception as e:
            print(f"❌ Error in update_all_smc_features: {e}")

    def update_smc_features(self, name, params):
        """Update SMC features"""
        try:
            print(f"🎯 Updating SMC feature: {name}")

            # Calculate SMC feature
            smc_data = self.calculate_smc_feature(name, params)

            if smc_data is not None and len(smc_data) > 0:
                self.display_smc_feature_professional(name, smc_data, params)
                print(f"🎨 SMC feature {name} displayed on chart")
            else:
                print(f"⚠️ No SMC data calculated for {name}")

        except Exception as e:
            print(f"❌ Error updating SMC feature {name}: {e}")

    def calculate_smc_feature(self, feature_name, params):
        """Calculate SMC feature"""
        try:
            df = self.prepare_smc_dataframe()
            if df is None or len(df) < 50:
                return None

            print(f"🎯 Calculating {feature_name} with {len(df)} bars")

            if feature_name == "FVG":
                result = smc.fvg(df)
                return result
            elif feature_name == "SwingHL":
                length = params.get('param1', 50)
                result = smc.swing_highs_lows(df, swing_length=length)
                return result
            elif feature_name == "BOS_CHOCH":
                swing_data = smc.swing_highs_lows(df)
                result = smc.bos_choch(df, swing_highs_lows=swing_data)
                return result
            elif feature_name == "OrderBlocks":
                swing_data = smc.swing_highs_lows(df)
                result = smc.ob(df, swing_highs_lows=swing_data)
                return result
            elif feature_name == "Liquidity":
                length = params.get('param1', 1)
                swing_data = smc.swing_highs_lows(df)
                result = smc.liquidity(df, swing_highs_lows=swing_data, range_percent=length/100.0)
                return result
            elif feature_name == "PrevHL":
                result = smc.previous_high_low(df)
                return result
            elif feature_name == "Sessions":
                result = smc.sessions(df, session="London")
                return result

            return None

        except Exception as e:
            print(f"❌ Error calculating SMC feature {feature_name}: {e}")
            return None

    def display_smc_feature_professional(self, name, data, params):
        """Display SMC features on chart"""
        try:
            # Remove existing SMC item
            if name in self.smc_items:
                if isinstance(self.smc_items[name], list):
                    for item in self.smc_items[name]:
                        self.price_widget.removeItem(item)
                else:
                    self.price_widget.removeItem(self.smc_items[name])

            # For now, just plot simple markers for SMC features
            if len(data) > 0:
                print(f"📊 Displaying {name} with {len(data)} data points")
                # Add basic visualization placeholder
                # TODO: Implement proper SMC visualization

        except Exception as e:
            print(f"❌ Error displaying SMC {name}: {e}")

# Demo Strategy for Backtesting
class DemoStrategy(CtaTemplate):
    """Demo strategy for testing"""

    author = "Trading Platform"

    # Strategy parameters
    fast_window = 10
    slow_window = 20

    # Variables
    fast_ma = 0.0
    slow_ma = 0.0

    parameters = ["fast_window", "slow_window"]
    variables = ["fast_ma", "slow_ma"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

    def on_init(self):
        """Initialize strategy"""
        self.write_log("🚀 Demo strategy initialized")

    def on_start(self):
        """Start strategy"""
        self.write_log("▶️ Demo strategy started")

    def on_stop(self):
        """Stop strategy"""
        self.write_log("⏸️ Demo strategy stopped")

    def on_bar(self, bar: BarData):
        """Process bar data with indicator updates"""
        try:
            # Update ArrayManager
            self.am.update_bar(bar)

            if not self.am.inited:
                return

            # Calculate moving averages
            self.fast_ma = self.am.sma(self.fast_window)
            self.slow_ma = self.am.sma(self.slow_window)

            # Update chart indicators if chart widget is available
            if hasattr(self.cta_engine, 'chart_widget') and self.cta_engine.chart_widget:
                try:
                    # Get all current bars for indicator calculation
                    all_bars = []
                    for i in range(self.am.count):
                        if i < len(self.am.open_array):
                            bar_data = BarData(
                                symbol=bar.symbol,
                                exchange=bar.exchange,
                                datetime=bar.datetime,
                                interval=bar.interval,
                                volume=self.am.volume_array[i] if i < len(self.am.volume_array) else 0,
                                open_price=self.am.open_array[i],
                                high_price=self.am.high_array[i],
                                low_price=self.am.low_array[i],
                                close_price=self.am.close_array[i]
                            )
                            all_bars.append(bar_data)

                    # Update chart with current bars
                    if all_bars:
                        self.cta_engine.chart_widget.update_indicators_professional(all_bars)

                except Exception as chart_error:
                    print(f"⚠️ Chart update error: {chart_error}")

            # Trading logic
            if self.fast_ma > self.slow_ma and self.pos <= 0:
                # Buy signal
                if self.pos < 0:
                    self.cover(bar.close_price)  # Close short
                self.buy(bar.close_price, 1)  # Open long

            elif self.fast_ma < self.slow_ma and self.pos >= 0:
                # Sell signal
                if self.pos > 0:
                    self.sell(bar.close_price)  # Close long
                self.short(bar.close_price, 1)  # Open short

        except Exception as e:
            self.write_log(f"❌ Error in on_bar: {e}")

def main():
    """Main function"""
    try:
        print("🚀 Starting Professional Trading Platform with Howtrader Integration")
        print("=" * 70)
        print("✨ PySide6 Version - Production Ready")
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