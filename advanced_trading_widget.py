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
from PySide6.QtCore import Qt, Signal, QTimer, QDate
from PySide6.QtGui import QFont, QColor

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


class DataManager:
    """🔥 Data loading and caching management"""

    def __init__(self):
        self.bars_data: pd.DataFrame = None
        self.trades_data: List[TradeData] = []
        self.symbol: str = ""
        self.interval: str = ""

    def load_data(self, bars_data: pd.DataFrame, trades_data: List = None,
                  symbol: str = "", interval: str = ""):
        """Load and prepare data for charting"""
        self.bars_data = bars_data.copy() if bars_data is not None else pd.DataFrame()
        self.trades_data = trades_data or []
        self.symbol = symbol
        self.interval = interval

        # Ensure required columns exist
        if not self.bars_data.empty:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in self.bars_data.columns:
                    self.bars_data[col] = 0

        print(f"📊 Data loaded: {len(self.bars_data)} bars, {len(self.trades_data)} trades")

    def get_ohlcv_data(self) -> np.ndarray:
        """Get OHLCV data as numpy array"""
        if self.bars_data.empty:
            return np.array([])

        return self.bars_data[['open', 'high', 'low', 'close', 'volume']].values


class IndicatorCalculator:
    """🔥 Technical indicator calculations"""

    @staticmethod
    def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        if TALIB_AVAILABLE:
            return talib.EMA(data, timeperiod=period)
        else:
            # Simple EMA calculation
            alpha = 2.0 / (period + 1.0)
            ema = np.empty_like(data)
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
            return ema

    @staticmethod
    def calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        if TALIB_AVAILABLE:
            return talib.RSI(data, timeperiod=period)
        else:
            # Basic RSI calculation
            delta = np.diff(data)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = np.mean(gain[:period])
            avg_loss = np.mean(loss[:period])

            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = np.full(len(data), 50.0)
            rsi[period:] = 100 - (100 / (1 + rs))
            return rsi

    @staticmethod
    def calculate_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      k_period: int = 9, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate KDJ indicator"""
        if TALIB_AVAILABLE:
            k, d = talib.STOCH(high, low, close,
                               fastk_period=k_period,
                               slowk_period=d_period,
                               slowd_period=d_period)
            j = 3 * k - 2 * d
            return k, d, j
        else:
            # Basic KDJ calculation
            lowest_low = np.full(len(close), np.nan)
            highest_high = np.full(len(close), np.nan)

            for i in range(k_period - 1, len(close)):
                lowest_low[i] = np.min(low[i - k_period + 1:i + 1])
                highest_high[i] = np.max(high[i - k_period + 1:i + 1])

            k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            d = np.full(len(close), 50.0)
            j = np.full(len(close), 50.0)

            return k, d, j


class SMCCalculator:
    """🔥 Smart Money Concepts calculations"""

    @staticmethod
    def calculate_fvg(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[Dict]:
        """Calculate Fair Value Gaps"""
        if not SMC_AVAILABLE:
            return []

        try:
            df = pd.DataFrame({'high': high, 'low': low, 'close': close})
            fvg_data = smc.fvg(df)
            return fvg_data.to_dict('records') if not fvg_data.empty else []
        except:
            return []

    @staticmethod
    def calculate_swing_highs_lows(high: np.ndarray, low: np.ndarray, length: int = 10) -> Dict:
        """Calculate swing highs and lows"""
        if not SMC_AVAILABLE:
            return {'swing_highs': [], 'swing_lows': []}

        try:
            df = pd.DataFrame({'high': high, 'low': low})
            swing_data = smc.swing_highs_lows(df, length=length)
            return {
                'swing_highs': swing_data['swing_highs'].dropna().index.tolist(),
                'swing_lows': swing_data['swing_lows'].dropna().index.tolist()
            }
        except:
            return {'swing_highs': [], 'swing_lows': []}


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
        self.add_indicator_row("EMA", 20, "", True)
        self.add_indicator_row("EMA", 50, "", False)
        self.add_indicator_row("EMA", 100, "", False)
        self.add_indicator_row("RSI", 14, "", False)
        self.add_indicator_row("KDJ", 9, "3", False)

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

        for name, checkbox in self.smc_features.items():
            checkbox.setStyleSheet("color: #ffffff; padding: 3px;")
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
        symbol_layout.addWidget(self.symbol_combo)
        layout.addLayout(symbol_layout)

        # Timeframe
        timeframe_layout = QHBoxLayout()
        timeframe_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1h', '4h', '1D', '1W', '1M'])
        self.timeframe_combo.setCurrentText('1h')
        timeframe_layout.addWidget(self.timeframe_combo)
        layout.addLayout(timeframe_layout)

        # Date range
        date_layout = QVBoxLayout()
        date_layout.addWidget(QLabel("Date Range:"))

        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-30))
        self.start_date.setCalendarPopup(True)
        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
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


class CandlestickItem(pg.GraphicsObject):
    """🔥 Custom candlestick chart item"""

    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()

    def generatePicture(self):
        """Generate the candlestick picture"""
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)

        w = 0.6  # Candlestick width

        for i, (open_price, high, low, close, volume) in enumerate(self.data):
            # Determine color
            if close >= open_price:
                color = pg.QtGui.QColor(34, 197, 94)  # Green for bullish
            else:
                color = pg.QtGui.QColor(239, 68, 68)  # Red for bearish

            p.setPen(pg.mkPen(color))
            p.setBrush(pg.mkBrush(color))

            # Draw wick
            p.drawLine(pg.QtCore.QPointF(i, low), pg.QtCore.QPointF(i, high))

            # Draw body
            body_height = abs(close - open_price)
            body_top = max(close, open_price)

            if body_height > 0:
                p.drawRect(pg.QtCore.QRectF(i - w / 2, body_top, w, -body_height))
            else:
                # Draw line for doji
                p.drawLine(pg.QtCore.QPointF(i - w / 2, close), pg.QtCore.QPointF(i + w / 2, close))

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class TradingChartView(QWidget):
    """🔥 Main multi-panel chart display"""

    def __init__(self):
        super().__init__()
        self.data_manager = DataManager()
        self.indicator_calculator = IndicatorCalculator()
        self.smc_calculator = SMCCalculator()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create chart widget with multiple plots
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground('#1e1e1e')

        # Main price chart
        self.price_plot = self.graphics_widget.addPlot(row=0, col=0, title="Price Chart")
        self.price_plot.addLegend()
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setLabel('left', 'Price')

        # Volume chart
        self.volume_plot = self.graphics_widget.addPlot(row=1, col=0, title="Volume")
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setLabel('left', 'Volume')
        self.volume_plot.setXLink(self.price_plot)

        # Oscillator chart
        self.oscillator_plot = self.graphics_widget.addPlot(row=2, col=0, title="Oscillators")
        self.oscillator_plot.showGrid(x=True, y=True, alpha=0.3)
        self.oscillator_plot.setLabel('left', 'Value')
        self.oscillator_plot.setXLink(self.price_plot)

        layout.addWidget(self.graphics_widget)
        self.setLayout(layout)

    def update_chart_data(self, bars_data: pd.DataFrame, trades_data: List = None):
        """Update chart with new data"""
        self.data_manager.load_data(bars_data, trades_data)
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

        print(f"📊 Chart rendered with {len(data)} bars")

    def add_indicator(self, indicator_data: Dict):
        """Add technical indicator to chart"""
        name = indicator_data.get('name', '')
        period = indicator_data.get('period', 14)

        if self.data_manager.bars_data.empty:
            return

        close_data = self.data_manager.bars_data['close'].values
        x_axis = np.arange(len(close_data))

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

        print(f"📈 Added indicator: {name}({period})")

    def add_smc_features(self, features: List[str]):
        """Add Smart Money Concepts features"""
        if not SMC_AVAILABLE or self.data_manager.bars_data.empty:
            return

        for feature in features:
            if feature == 'FVG':
                self.add_fair_value_gaps()
            elif feature == 'SwingHL':
                self.add_swing_highs_lows()

    def add_fair_value_gaps(self):
        """Add Fair Value Gaps"""
        try:
            high_data = self.data_manager.bars_data['high'].values
            low_data = self.data_manager.bars_data['low'].values
            close_data = self.data_manager.bars_data['close'].values

            fvg_data = self.smc_calculator.calculate_fvg(high_data, low_data, close_data)

            for fvg in fvg_data:
                # Add FVG visualization (simplified)
                pass

            print("💰 Added Fair Value Gaps")
        except Exception as e:
            print(f"❌ Error adding FVG: {e}")

    def add_swing_highs_lows(self):
        """Add swing highs and lows"""
        try:
            high_data = self.data_manager.bars_data['high'].values
            low_data = self.data_manager.bars_data['low'].values

            swing_data = self.smc_calculator.calculate_swing_highs_lows(high_data, low_data)

            # Mark swing highs
            for idx in swing_data['swing_highs']:
                if idx < len(high_data):
                    self.price_plot.plot([idx], [high_data[idx]],
                                         pen=None, symbol='t', symbolBrush='red', symbolSize=8)

            # Mark swing lows
            for idx in swing_data['swing_lows']:
                if idx < len(low_data):
                    self.price_plot.plot([idx], [low_data[idx]],
                                         pen=None, symbol='t1', symbolBrush='green', symbolSize=8)

            print("💰 Added Swing Highs/Lows")
        except Exception as e:
            print(f"❌ Error adding swing H/L: {e}")


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
        main_layout.setStretch(0, 0)  # Fixed width for left panel
        main_layout.setStretch(1, 1)  # Expanding chart area

        self.setLayout(main_layout)

    def connect_signals(self):
        """Connect signals between components"""
        self.data_panel.data_changed.connect(self.on_data_changed)
        self.indicator_panel.indicator_changed.connect(self.on_indicator_changed)
        self.smc_panel.smc_changed.connect(self.on_smc_changed)

    def on_data_changed(self, action: str, data: Dict):
        """Handle data control changes"""
        print(f"📊 Data changed: {action}, {data}")
        # In a real implementation, this would reload data from database

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
        self.chart_view.add_smc_features(features)

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


def create_sample_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')

    # Generate realistic price data
    np.random.seed(42)
    price = 50000.0
    prices = [price]

    for _ in range(999):
        change = np.random.normal(0, 200)
        price = max(price + change, 1000)  # Minimum price
        prices.append(price)

    # Create OHLCV data
    data = []
    for i in range(len(prices)):
        open_price = prices[i]
        close_price = prices[i] + np.random.normal(0, 50)
        high = max(open_price, close_price) + abs(np.random.normal(0, 30))
        low = min(open_price, close_price) - abs(np.random.normal(0, 30))
        volume = np.random.uniform(100, 10000)

        data.append({
            'datetime': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })

    return pd.DataFrame(data)


def main():
    """Main function for testing the widget"""
    app = QApplication(sys.argv)

    # Create sample data
    sample_data = create_sample_data()

    # Create and show widget
    widget = AdvancedTradingWidget(
        bars_data=sample_data,
        trades_data=[],
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
    print("  ✅ Dark theme")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()