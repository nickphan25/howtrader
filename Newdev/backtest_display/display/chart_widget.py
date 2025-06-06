"""
Advanced Chart Widget for Howtrader Backtest Display
===================================================

Professional chart widget với PySide6 + PyQtGraph integration.
Part của backtest_display package.
"""

import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import configuration từ package
try:
    from ..config.display_config import DisplayConfig, ChartTheme
except ImportError:
    # Fallback configuration nếu không có config module
    class ChartTheme:
        DARK = "dark"
        LIGHT = "light"

    class DisplayConfig:
        def __init__(self):
            self.theme = ChartTheme.DARK


class ChartConfig:
    """Chart configuration và styling"""

    class Colors:
        # Chart colors
        BACKGROUND = '#2b2b2b'
        GRID_LIGHT = '#404040'
        GRID_MEDIUM = '#505050'
        GRID_MAJOR = '#606060'
        TEXT_PRIMARY = '#ffffff'
        TEXT_SECONDARY = '#cccccc'

        # Candlestick colors
        CANDLE_UP = '#26a69a'
        CANDLE_DOWN = '#ef5350'

        # Indicator colors
        EMA_COLOR = '#ffc107'
        SMA_COLOR = '#2196f3'
        RSI_COLOR = '#9c27b0'
        MACD_COLOR = '#ff9800'
        BOLLINGER_COLOR = '#4caf50'

        # Trading signals
        BUY_SIGNAL = '#00ff00'
        SELL_SIGNAL = '#ff0000'
        PROFIT_COLOR = '#4caf50'
        LOSS_COLOR = '#f44336'

    class Dimensions:
        WINDOW_WIDTH = 1400
        WINDOW_HEIGHT = 900
        LEFT_PANEL_WIDTH = 300
        TITLE_HEIGHT = 40
        PRICE_CHART_MIN_HEIGHT = 400
        OSCILLATOR_HEIGHT = 150
        VOLUME_HEIGHT = 100


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick chart item"""

    def __init__(self, data=None):
        super().__init__()
        self.data = data or []
        self.picture = None
        self.generatePicture()

    def update_data(self, data):
        """Update candlestick data"""
        self.data = data
        self.generatePicture()
        self.update()

    def generatePicture(self):
        """Generate candlestick picture"""
        if not self.data:
            return

        self.picture = QPicture()
        painter = QPainter(self.picture)

        # Setup pens and brushes
        up_pen = pg.mkPen(ChartConfig.Colors.CANDLE_UP, width=1)
        down_pen = pg.mkPen(ChartConfig.Colors.CANDLE_DOWN, width=1)
        up_brush = pg.mkBrush(ChartConfig.Colors.CANDLE_UP)
        down_brush = pg.mkBrush(ChartConfig.Colors.CANDLE_DOWN)

        for i, bar in enumerate(self.data):
            if isinstance(bar, dict):
                open_price = bar.get('open', 0)
                high_price = bar.get('high', 0)
                low_price = bar.get('low', 0)
                close_price = bar.get('close', 0)
            else:
                # Assume tuple/list format (open, high, low, close)
                open_price, high_price, low_price, close_price = bar[:4]

            # Determine color
            is_up = close_price >= open_price
            pen = up_pen if is_up else down_pen
            brush = up_brush if is_up else down_brush

            painter.setPen(pen)
            painter.setBrush(brush)

            # Draw high-low line
            painter.drawLine(QPointF(i, low_price), QPointF(i, high_price))

            # Draw body rectangle
            body_height = abs(close_price - open_price)
            body_top = max(open_price, close_price)

            if body_height > 0:
                rect = QRectF(i - 0.3, body_top, 0.6, -body_height)
                painter.drawRect(rect)
            else:
                # Doji - draw line
                painter.drawLine(QPointF(i - 0.3, open_price), QPointF(i + 0.3, close_price))

        painter.end()

    def paint(self, painter, option, widget):
        if self.picture:
            self.picture.play(painter)

    def boundingRect(self):
        if not self.data:
            return QRectF()

        # Calculate bounds from data
        prices = []
        for bar in self.data:
            if isinstance(bar, dict):
                prices.extend([bar.get('high', 0), bar.get('low', 0)])
            else:
                prices.extend([bar[1], bar[2]])  # high, low

        if not prices:
            return QRectF()

        min_price = min(prices)
        max_price = max(prices)

        return QRectF(0, min_price, len(self.data), max_price - min_price)


class VolumeBarItem(pg.GraphicsObject):
    """Custom volume bar chart item"""

    def __init__(self, data=None):
        super().__init__()
        self.data = data or []
        self.picture = None
        self.generatePicture()

    def update_data(self, data):
        """Update volume data"""
        self.data = data
        self.generatePicture()
        self.update()

    def generatePicture(self):
        """Generate volume bars picture"""
        if not self.data:
            return

        self.picture = QPicture()
        painter = QPainter(self.picture)

        # Setup pen and brush
        pen = pg.mkPen('#666666', width=1)
        brush = pg.mkBrush('#888888')

        painter.setPen(pen)
        painter.setBrush(brush)

        # Find max volume for scaling
        volumes = [bar.get('volume', 0) if isinstance(bar, dict) else bar[-1] for bar in self.data]
        max_volume = max(volumes) if volumes else 1

        for i, bar in enumerate(self.data):
            volume = bar.get('volume', 0) if isinstance(bar, dict) else bar[-1]

            # Normalize volume height
            height = (volume / max_volume) * 100 if max_volume > 0 else 0

            # Draw volume bar
            rect = QRectF(i - 0.4, 0, 0.8, height)
            painter.drawRect(rect)

        painter.end()

    def paint(self, painter, option, widget):
        if self.picture:
            self.picture.play(painter)

    def boundingRect(self):
        if not self.data:
            return QRectF()

        volumes = [bar.get('volume', 0) if isinstance(bar, dict) else bar[-1] for bar in self.data]
        max_volume = max(volumes) if volumes else 1

        return QRectF(0, 0, len(self.data), max_volume)


class IndicatorItem(pg.GraphicsObject):
    """Custom indicator line item"""

    def __init__(self, data=None, color='#ffff00', width=2, name='Indicator'):
        super().__init__()
        self.data = data or []
        self.color = color
        self.width = width
        self.name = name
        self.picture = None
        self.generatePicture()

    def update_data(self, data):
        """Update indicator data"""
        self.data = data
        self.generatePicture()
        self.update()

    def generatePicture(self):
        """Generate indicator line picture"""
        if not self.data or len(self.data) < 2:
            return

        self.picture = QPicture()
        painter = QPainter(self.picture)

        # Setup pen
        pen = pg.mkPen(self.color, width=self.width)
        painter.setPen(pen)

        # Draw line segments
        for i in range(len(self.data) - 1):
            if self.data[i] is not None and self.data[i + 1] is not None:
                painter.drawLine(
                    QPointF(i, self.data[i]),
                    QPointF(i + 1, self.data[i + 1])
                )

        painter.end()

    def paint(self, painter, option, widget):
        if self.picture:
            self.picture.play(painter)

    def boundingRect(self):
        if not self.data:
            return QRectF()

        valid_data = [x for x in self.data if x is not None]
        if not valid_data:
            return QRectF()

        min_val = min(valid_data)
        max_val = max(valid_data)

        return QRectF(0, min_val, len(self.data), max_val - min_val)


class DataLoader(QThread):
    """Mock data loader for demonstration"""

    data_loaded = Signal(dict)
    error_occurred = Signal(str)
    progress_updated = Signal(int, str)

    def __init__(self, symbol="BTCUSDT", timeframe="1h", days=30):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.days = days

    def run(self):
        """Generate mock data"""
        try:
            self.progress_updated.emit(10, "Generating mock data...")

            # Generate sample OHLCV data
            num_bars = self.days * 24  # Hourly data

            # Starting values
            base_price = 50000
            current_price = base_price

            timestamps = []
            bars = []

            start_time = datetime.now() - timedelta(days=self.days)

            for i in range(num_bars):
                timestamp = start_time + timedelta(hours=i)
                timestamps.append(timestamp)

                # Generate OHLC with random walk
                change = np.random.normal(0, base_price * 0.002)  # 0.2% volatility

                open_price = current_price
                high_price = open_price + abs(np.random.normal(0, base_price * 0.001))
                low_price = open_price - abs(np.random.normal(0, base_price * 0.001))
                close_price = open_price + change

                # Ensure high >= max(open, close) and low <= min(open, close)
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)

                volume = np.random.randint(100, 1000)

                bars.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })

                current_price = close_price

                if i % 100 == 0:
                    progress = 10 + (i / num_bars) * 80
                    self.progress_updated.emit(int(progress), f"Generated {i}/{num_bars} bars...")

            self.progress_updated.emit(90, "Processing data...")

            # Create DataFrame
            df = pd.DataFrame(bars)

            data = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'bars': bars,
                'df': df,
                'timestamps': timestamps
            }

            self.progress_updated.emit(100, "Data loaded successfully!")
            self.data_loaded.emit(data)

        except Exception as e:
            self.error_occurred.emit(str(e))


class AdvancedHowtraderChart(QWidget):
    """
    Advanced chart widget thay thế cho chartish.AdvancedHowtraderChart
    Phiên bản tích hợp trong backtest_display package
    """

    def __init__(self, parent=None, config=None):
        super().__init__(parent)

        # Configuration
        self.config = config or DisplayConfig()

        # Chart data
        self.df_ohlc = None
        self.timestamps = []
        self.bars = []

        # Chart items
        self.price_indicator_items = {}
        self.oscillator_indicator_items = {}
        self.plot_items = {}

        # Data loader
        self.data_loader = None

        # Initialize UI
        self.init_ui()

        # Load sample data
        self.load_sample_data()

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout()

        # Title and controls
        title_layout = QHBoxLayout()

        self.symbol_label = QLabel("Symbol: BTCUSDT")
        self.symbol_label.setStyleSheet("font-weight: bold; color: #2E7D32;")

        self.timeframe_label = QLabel("Timeframe: 1h")
        self.timeframe_label.setStyleSheet("color: #666;")

        self.data_range_label = QLabel("Range: Last 30 days")
        self.data_range_label.setStyleSheet("color: #666;")

        self.current_price_label = QLabel("Price: $50,000.00")
        self.current_price_label.setStyleSheet("font-weight: bold; color: #1976D2;")

        # Load data button
        self.load_btn = QPushButton("📊 Load New Data")
        self.load_btn.setMaximumWidth(150)
        self.load_btn.clicked.connect(self.load_sample_data)

        title_layout.addWidget(self.symbol_label)
        title_layout.addWidget(self.timeframe_label)
        title_layout.addWidget(self.data_range_label)
        title_layout.addWidget(self.current_price_label)
        title_layout.addStretch()
        title_layout.addWidget(self.load_btn)

        layout.addLayout(title_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to load data")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        # Chart area
        self.create_chart_area()
        layout.addWidget(self.chart_widget)

        self.setLayout(layout)

    def create_chart_area(self):
        """Create main chart area"""
        # Main chart widget
        self.chart_widget = pg.GraphicsLayoutWidget()
        self.chart_widget.setBackground(ChartConfig.Colors.BACKGROUND)

        # Price chart (main)
        self.price_plot = self.chart_widget.addPlot(row=0, col=0)
        self.price_plot.setLabel('left', 'Price', color=ChartConfig.Colors.TEXT_PRIMARY)
        self.price_plot.setLabel('bottom', 'Time')
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setMinimumHeight(400)

        # Volume chart
        self.volume_plot = self.chart_widget.addPlot(row=1, col=0)
        self.volume_plot.setLabel('left', 'Volume', color=ChartConfig.Colors.TEXT_PRIMARY)
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setMaximumHeight(100)

        # Link x-axes
        self.volume_plot.setXLink(self.price_plot)

        # Setup crosshair
        self.setup_crosshair()

    def setup_crosshair(self):
        """Setup crosshair cursor"""
        # Vertical line
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen(ChartConfig.Colors.TEXT_SECONDARY, width=1))
        self.price_plot.addItem(self.vline, ignoreBounds=True)

        # Horizontal line
        self.hline = pg.InfiniteLine(angle=0, movable=False,
                                     pen=pg.mkPen(ChartConfig.Colors.TEXT_SECONDARY, width=1))
        self.price_plot.addItem(self.hline, ignoreBounds=True)

        # Mouse movement tracking
        self.price_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)

    def on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair"""
        if self.price_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.price_plot.vb.mapSceneToView(pos)
            self.vline.setPos(mouse_point.x())
            self.hline.setPos(mouse_point.y())

    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.load_btn.setEnabled(False)
        self.status_label.setText("Loading data...")

        # Create and start data loader
        self.data_loader = DataLoader()
        self.data_loader.data_loaded.connect(self.on_data_loaded)
        self.data_loader.error_occurred.connect(self.on_data_error)
        self.data_loader.progress_updated.connect(self.on_progress_updated)
        self.data_loader.start()

    def on_progress_updated(self, value, status):
        """Handle progress updates"""
        self.progress_bar.setValue(value)
        self.status_label.setText(status)

    def on_data_loaded(self, data):
        """Handle data loaded successfully"""
        try:
            self.df_ohlc = data['df']
            self.bars = data['bars']
            self.timestamps = data['timestamps']

            # Update chart
            self.display_chart_data()

            # Update labels
            last_bar = self.bars[-1]
            self.current_price_label.setText(f"Price: ${last_bar['close']:,.2f}")
            self.data_range_label.setText(f"Range: {len(self.bars)} bars")

            self.status_label.setText("✅ Data loaded successfully!")

        except Exception as e:
            self.on_data_error(f"Error processing data: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.load_btn.setEnabled(True)

    def on_data_error(self, error_msg):
        """Handle data loading error"""
        self.status_label.setText(f"❌ Error: {error_msg}")
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)

    def display_chart_data(self):
        """Display loaded data on chart"""
        if not self.bars:
            return

        # Clear existing items
        self.price_plot.clear()
        self.volume_plot.clear()

        # Re-add crosshair
        self.setup_crosshair()

        # Add candlestick data
        candlestick_item = CandlestickItem(self.bars)
        self.price_plot.addItem(candlestick_item)

        # Add volume data
        volume_item = VolumeBarItem(self.bars)
        self.volume_plot.addItem(volume_item)

        # Add sample moving averages
        self.add_sample_indicators()

        # Auto-range
        self.price_plot.autoRange()
        self.volume_plot.autoRange()

    def add_sample_indicators(self):
        """Add sample technical indicators"""
        if self.df_ohlc is None or len(self.df_ohlc) < 20:
            return

        # Calculate simple moving averages
        closes = self.df_ohlc['close'].values

        # 20-period SMA
        sma20 = pd.Series(closes).rolling(window=20).mean().values
        sma20_item = IndicatorItem(sma20, color=ChartConfig.Colors.SMA_COLOR, name='SMA 20')
        self.price_plot.addItem(sma20_item)

        # 50-period SMA
        if len(closes) >= 50:
            sma50 = pd.Series(closes).rolling(window=50).mean().values
            sma50_item = IndicatorItem(sma50, color=ChartConfig.Colors.EMA_COLOR, name='SMA 50')
            self.price_plot.addItem(sma50_item)

    # Compatibility methods with original chartish interface
    def add_candlestick_data(self, data):
        """Add candlestick data to chart (compatibility method)"""
        self.bars = data
        self.display_chart_data()

    def add_volume_data(self, data):
        """Add volume data to chart (compatibility method)"""
        # Volume is included in candlestick data
        pass

    def clear_chart(self):
        """Clear all chart data"""
        self.price_plot.clear()
        self.volume_plot.clear()
        self.bars = []
        self.df_ohlc = None
        self.timestamps = []

        # Re-setup crosshair
        self.setup_crosshair()


# Factory functions
def create_chart_widget(config=None):
    """Factory function để tạo chart widget"""
    return AdvancedHowtraderChart(config=config)


def test_chart_widget():
    """Test function"""
    app = QApplication(sys.argv)

    window = AdvancedHowtraderChart()
    window.setWindowTitle("📈 Advanced Howtrader Chart - Package Version")
    window.resize(1200, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    test_chart_widget()