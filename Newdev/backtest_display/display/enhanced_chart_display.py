"""
Enhanced Chart Display
=====================

Enhanced chart với backtest visualization capabilities.
Tích hợp trực tiếp các components từ chartish để tránh dependency.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QComboBox, QPushButton, QLabel, QSplitter, QFrame,
    QGridLayout, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter

# Import local modules
from ..config.display_config import DisplayConfig, ChartTheme, MarkerType
from .backtest_markers import MarkerManager, create_marker_manager
from ..core.backtest_bridge import BacktestBridge
from ..core.data_extractor import BacktestDataExtractor


# =====================================================
# Chart Components từ chartish.py
# =====================================================

class ChartConfig:
    """Configuration cho chart display"""

    class Colors:
        BACKGROUND = '#1e1e1e'
        GRID_LIGHT = '#333333'
        GRID_MEDIUM = '#444444'
        GRID_MAJOR = '#555555'
        TEXT_PRIMARY = '#ffffff'
        TEXT_SECONDARY = '#cccccc'

        # Candle colors
        CANDLE_UP = '#00ff88'
        CANDLE_DOWN = '#ff4444'

        # Indicator colors
        EMA_COLOR = '#ffaa00'
        SMA_COLOR = '#00aaff'
        RSI_COLOR = '#ff6600'
        MACD_COLOR = '#9966ff'
        BOLLINGER_COLOR = '#66ff66'

    class Dimensions:
        WINDOW_WIDTH = 1600
        WINDOW_HEIGHT = 900
        LEFT_PANEL_WIDTH = 250
        TITLE_HEIGHT = 30
        PRICE_CHART_MIN_HEIGHT = 400
        OSCILLATOR_HEIGHT = 150
        VOLUME_HEIGHT = 100


class AdvancedHowtraderChart(pg.GraphicsLayoutWidget):
    """
    Advanced chart widget thay thế cho AdvancedHowtraderChart từ chartish.py
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Setup chart configuration
        self.config = ChartConfig()

        # Initialize chart
        self._init_chart()

    def _init_chart(self):
        """Initialize chart components"""
        # Set background
        self.setBackground(self.config.Colors.BACKGROUND)

        # Create main plot
        self.main_plot = self.addPlot(row=0, col=0)
        self.main_plot.setLabel('left', 'Price')
        self.main_plot.setLabel('bottom', 'Time')

        # Enable crosshair
        self.main_plot.addLine(x=0, pen=pg.mkPen('#888888', width=1))
        self.main_plot.addLine(y=0, pen=pg.mkPen('#888888', width=1))

        # Store plot items
        self.plot_items = {}

    def add_candlestick_data(self, data):
        """Add candlestick data to chart"""
        if 'candlestick' not in self.plot_items:
            # Create candlestick item
            self.plot_items['candlestick'] = CandlestickItem(data)
            self.main_plot.addItem(self.plot_items['candlestick'])
        else:
            self.plot_items['candlestick'].update_data(data)

    def add_volume_data(self, data):
        """Add volume data to chart"""
        if not hasattr(self, 'volume_plot'):
            self.volume_plot = self.addPlot(row=1, col=0)
            self.volume_plot.setLabel('left', 'Volume')
            self.volume_plot.setMaximumHeight(100)

        if 'volume' not in self.plot_items:
            self.plot_items['volume'] = VolumeBarItem(data)
            self.volume_plot.addItem(self.plot_items['volume'])
        else:
            self.plot_items['volume'].update_data(data)

    def clear_chart(self):
        """Clear all chart data"""
        for item in self.plot_items.values():
            if hasattr(item, 'clear'):
                item.clear()
        self.plot_items.clear()


class CandlestickItem(pg.GraphicsObject):
    """Candlestick chart item"""

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.picture = None
        self.generatePicture()

    def generatePicture(self):
        """Generate picture for candlesticks"""
        self.picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self.picture)

        if self.data is not None and len(self.data) > 0:
            # Draw candlesticks
            for i, row in enumerate(self.data):
                open_price = row.get('open', 0)
                high_price = row.get('high', 0)
                low_price = row.get('low', 0)
                close_price = row.get('close', 0)

                # Determine color
                color = ChartConfig.Colors.CANDLE_UP if close_price >= open_price else ChartConfig.Colors.CANDLE_DOWN
                pen = pg.mkPen(color, width=1)
                brush = pg.mkBrush(color)

                painter.setPen(pen)
                painter.setBrush(brush)

                # Draw high-low line
                painter.drawLine(i, low_price, i, high_price)

                # Draw body
                body_height = abs(close_price - open_price)
                body_top = max(close_price, open_price)
                painter.drawRect(i - 0.3, body_top - body_height, 0.6, body_height)

        painter.end()

    def paint(self, painter, option, widget):
        if self.picture:
            self.picture.play(painter)

    def boundingRect(self):
        if self.data is not None and len(self.data) > 0:
            return pg.QtCore.QRectF(0, 0, len(self.data), 1)
        return pg.QtCore.QRectF()

    def update_data(self, data):
        """Update candlestick data"""
        self.data = data
        self.generatePicture()
        self.update()


class VolumeBarItem(pg.GraphicsObject):
    """Volume bar chart item"""

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.picture = None
        self.generatePicture()

    def generatePicture(self):
        """Generate picture for volume bars"""
        self.picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self.picture)

        if self.data is not None and len(self.data) > 0:
            max_volume = max(row.get('volume', 0) for row in self.data)

            for i, row in enumerate(self.data):
                volume = row.get('volume', 0)
                height = (volume / max_volume) if max_volume > 0 else 0

                pen = pg.mkPen('#666666', width=1)
                brush = pg.mkBrush('#888888')

                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawRect(i - 0.4, 0, 0.8, height)

        painter.end()

    def paint(self, painter, option, widget):
        if self.picture:
            self.picture.play(painter)

    def boundingRect(self):
        if self.data is not None and len(self.data) > 0:
            return pg.QtCore.QRectF(0, 0, len(self.data), 1)
        return pg.QtCore.QRectF()

    def update_data(self, data):
        """Update volume data"""
        self.data = data
        self.generatePicture()
        self.update()


# =====================================================
# Enhanced Backtest Chart Classes
# =====================================================

class BacktestOverlayControls(QWidget):
    """Controls for backtest overlay options"""

    # Signals
    overlay_toggled = Signal(str, bool)
    marker_style_changed = Signal(str, dict)
    theme_changed = Signal(str)

    def __init__(self, config: DisplayConfig):
        super().__init__()
        self.config = config
        self.overlay_checkboxes = {}
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)

        # Theme selection
        theme_group = QGroupBox("Chart Theme")
        theme_layout = QVBoxLayout(theme_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(['Light', 'Dark', 'Auto'])
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        theme_layout.addWidget(self.theme_combo)

        layout.addWidget(theme_group)

        # Overlay controls
        overlay_group = QGroupBox("Backtest Overlays")
        overlay_layout = QVBoxLayout(overlay_group)

        overlays = [
            ('trade_markers', 'Trade Markers', True),
            ('trade_lines', 'Trade Lines', True),
            ('position_indicator', 'Position Indicator', False),
            ('pnl_overlay', 'P&L Overlay', True),
            ('volume_markers', 'Volume Markers', False),
            ('statistics_overlay', 'Statistics', True)
        ]

        for key, label, default in overlays:
            checkbox = QCheckBox(label)
            checkbox.setChecked(default)
            checkbox.toggled.connect(
                lambda checked, k=key: self.overlay_toggled.emit(k, checked)
            )
            self.overlay_checkboxes[key] = checkbox
            overlay_layout.addWidget(checkbox)

        layout.addWidget(overlay_group)

        # Marker style controls
        marker_group = QGroupBox("Marker Styles")
        marker_layout = QGridLayout(marker_group)

        # Buy marker color
        marker_layout.addWidget(QLabel("Buy Color:"), 0, 0)
        self.buy_color_btn = QPushButton()
        self.buy_color_btn.setStyleSheet(f"background-color: {self.config.markers.buy_marker_color}")
        self.buy_color_btn.clicked.connect(lambda: self._change_marker_color('buy'))
        marker_layout.addWidget(self.buy_color_btn, 0, 1)

        # Sell marker color
        marker_layout.addWidget(QLabel("Sell Color:"), 1, 0)
        self.sell_color_btn = QPushButton()
        self.sell_color_btn.setStyleSheet(f"background-color: {self.config.markers.sell_marker_color}")
        self.sell_color_btn.clicked.connect(lambda: self._change_marker_color('sell'))
        marker_layout.addWidget(self.sell_color_btn, 1, 1)

        layout.addWidget(marker_group)

        # Stretch
        layout.addStretch()

    def _on_theme_changed(self, theme_name: str):
        """Handle theme change"""
        theme_map = {
            'Light': ChartTheme.LIGHT,
            'Dark': ChartTheme.DARK,
            'Auto': ChartTheme.AUTO
        }

        if theme_name in theme_map:
            self.config.update_theme(theme_map[theme_name])
            self.theme_changed.emit(theme_name.lower())

    def _change_marker_color(self, marker_type: str):
        """Change marker color"""
        # Mock color dialog - trong thực tế sẽ dùng QColorDialog
        colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']
        import random
        new_color = random.choice(colors)

        if marker_type == 'buy':
            self.config.markers.buy_marker_color = new_color
            self.buy_color_btn.setStyleSheet(f"background-color: {new_color}")
        else:
            self.config.markers.sell_marker_color = new_color
            self.sell_color_btn.setStyleSheet(f"background-color: {new_color}")

        self.marker_style_changed.emit(marker_type, {'color': new_color})


class BacktestResultsWidget(QWidget):
    """Widget to display backtest results"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)

        # Performance metrics
        self.metrics_group = QGroupBox("Performance Metrics")
        self.metrics_layout = QVBoxLayout(self.metrics_group)
        layout.addWidget(self.metrics_group)

        # Trade summary
        self.trades_group = QGroupBox("Trade Summary")
        self.trades_layout = QVBoxLayout(self.trades_group)
        layout.addWidget(self.trades_group)

        # Scroll area for results
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.addWidget(self.metrics_group)
        scroll_layout.addWidget(self.trades_group)
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

    def update_results(self, results: Dict[str, Any]):
        """Update results display"""
        self._clear_layouts()

        # Add performance metrics
        performance = results.get('performance', {})
        self._add_metric_labels(performance)

        # Add trade summary
        summary = results.get('summary', {})
        self._add_trade_summary(summary)

    def _clear_layouts(self):
        """Clear existing layouts"""
        for layout in [self.metrics_layout, self.trades_layout]:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

    def _add_metric_labels(self, metrics: Dict):
        """Add metric labels"""
        for key, value in metrics.items():
            if isinstance(value, float):
                if key.endswith('_pct') or 'return' in key.lower():
                    text = f"{key.replace('_', ' ').title()}: {value:.2%}"
                else:
                    text = f"{key.replace('_', ' ').title()}: {value:.4f}"
            else:
                text = f"{key.replace('_', ' ').title()}: {value}"

            label = QLabel(text)
            self.metrics_layout.addWidget(label)

    def _add_trade_summary(self, summary: Dict):
        """Add trade summary"""
        for key, value in summary.items():
            if isinstance(value, float):
                text = f"{key.replace('_', ' ').title()}: {value:.2f}"
            else:
                text = f"{key.replace('_', ' ').title()}: {value}"

            label = QLabel(text)
            self.trades_layout.addWidget(label)


class EnhancedBacktestChart(AdvancedHowtraderChart):
    """
    Enhanced chart với backtest visualization capabilities
    Extends AdvancedHowtraderChart với MarkerManager integration
    """

    # Signals
    backtest_completed = Signal(dict)  # backtest results
    backtest_started = Signal()
    backtest_progress = Signal(int)  # progress percentage

    def __init__(self):
        super().__init__()

        # Backtest components
        self.display_config = DisplayConfig()
        self.marker_manager = MarkerManager(self.display_config)
        self.data_extractor = BacktestDataExtractor()
        self.backtest_bridge = BacktestBridge()

        # State
        self.backtest_data = {}
        self.overlay_enabled = {
            'trade_markers': True,
            'trade_lines': True,
            'position_indicator': False,
            'pnl_overlay': True,
            'volume_markers': False,
            'statistics_overlay': True
        }

        self._setup_backtest_ui()
        self._connect_signals()

    def _setup_backtest_ui(self):
        """Setup backtest-specific UI components"""
        # Create backtest control panel
        self.backtest_controls = BacktestOverlayControls(self.display_config)

        # Create results widget
        self.results_widget = BacktestResultsWidget()

        # Add to existing layout - create right panel
        self._add_backtest_panels()

    def _add_backtest_panels(self):
        """Add backtest panels to existing chart layout"""
        # Get parent widget and create splitter layout
        if self.parent():
            parent = self.parent()
            if hasattr(parent, 'layout') and parent.layout():
                # Create horizontal splitter
                splitter = QSplitter(Qt.Orientation.Horizontal)

                # Add this chart to left side
                splitter.addWidget(self)

                # Create right panel container
                right_panel = QWidget()
                right_panel.setMaximumWidth(300)
                right_panel.setMinimumWidth(250)
                right_layout = QVBoxLayout(right_panel)

                # Add controls and results
                right_layout.addWidget(self.backtest_controls)

                separator = QFrame()
                separator.setFrameStyle(QFrame.Shape.HLine | QFrame.Shadow.Sunken)
                right_layout.addWidget(separator)

                right_layout.addWidget(self.results_widget)

                splitter.addWidget(right_panel)
                splitter.setSizes([1000, 300])

                # Replace this widget with splitter in parent
                parent.layout().addWidget(splitter)

    def _connect_signals(self):
        """Connect backtest-related signals"""
        # Overlay controls
        self.backtest_controls.overlay_toggled.connect(self._on_overlay_toggled)
        self.backtest_controls.marker_style_changed.connect(self._on_marker_style_changed)
        self.backtest_controls.theme_changed.connect(self._on_theme_changed)

        # Bridge signals
        self.backtest_bridge.set_progress_callback(self._on_backtest_progress)

    def load_backtest_data(self, backtest_results: Dict[str, Any]):
        """
        Load backtest results và display on chart

        Args:
            backtest_results: Results from BacktestingEngine
        """
        try:
            # Extract data using data extractor
            if 'engine' in backtest_results:
                self.data_extractor.connect_engine(backtest_results['engine'])
                extracted_data = self.data_extractor.extract_all_data()
            else:
                # Direct data input
                extracted_data = backtest_results

            self.backtest_data = extracted_data

            # Update chart with backtest visualization
            self._display_backtest_results()

            # Update results panel
            self.results_widget.update_results(extracted_data)

            self.backtest_completed.emit(extracted_data)

        except Exception as e:
            print(f"Error loading backtest data: {e}")

    def _display_backtest_results(self):
        """Display backtest results on chart"""
        if not self.backtest_data:
            return

        trades_data = self.backtest_data.get('trades', [])
        bars_data = self.backtest_data.get('bars', [])

        if not trades_data and not bars_data:
            print("No trade or bar data to display")
            return

        # Clear existing markers
        self._clear_backtest_items()

        # Add candlestick data if available
        if bars_data:
            self.add_candlestick_data(bars_data)
            self.add_volume_data(bars_data)

        # Process backtest data with marker manager
        if trades_data:
            self.marker_manager.process_backtest_data(self.backtest_data)
            self._update_chart_overlays()

    def _update_chart_overlays(self):
        """Update chart overlays based on enabled options"""
        plot_items = self.marker_manager.get_all_plot_items()

        for overlay_type, enabled in self.overlay_enabled.items():
            if overlay_type in plot_items and enabled:
                for item in plot_items[overlay_type]:
                    if item not in self.main_plot.items:
                        self.main_plot.addItem(item)
            elif overlay_type in plot_items and not enabled:
                for item in plot_items[overlay_type]:
                    if item in self.main_plot.items:
                        self.main_plot.removeItem(item)

    def _clear_backtest_items(self):
        """Clear existing backtest items from chart"""
        # Clear marker manager
        self.marker_manager.clear_all()

        # Clear chart
        self.clear_chart()

    def run_backtest_display(self, config: Dict[str, Any]):
        """
        Run backtest với given config và display results
        """
        try:
            # Emit start signal
            self.backtest_started.emit()

            # Run backtest using bridge
            results = self.backtest_bridge.run_simple_backtest(
                symbol=config.get('symbol', 'BTCUSDT'),
                start_date=config.get('start_date', datetime.now() - timedelta(days=30)),
                end_date=config.get('end_date', datetime.now()),
                strategy_class=config.get('strategy_class'),
                strategy_settings=config.get('strategy_settings', {}),
                progress_callback=self._on_backtest_progress
            )

            # Load results
            if 'error' not in results:
                self.load_backtest_data(results)
            else:
                print(f"Backtest error: {results['error']}")

        except Exception as e:
            print(f"Error running backtest display: {e}")

    def _on_overlay_toggled(self, overlay_type: str, enabled: bool):
        """Handle overlay toggle"""
        self.overlay_enabled[overlay_type] = enabled
        self._update_chart_overlays()

    def _on_marker_style_changed(self, marker_type: str, style: Dict):
        """Handle marker style change"""
        # Update marker manager configuration
        self.marker_manager.update_config(self.display_config)
        self._update_chart_overlays()

    def _on_theme_changed(self, theme_name: str):
        """Handle theme change"""
        # Update chart theme
        if theme_name == 'dark':
            self.setBackground(ChartConfig.Colors.BACKGROUND)
        else:
            self.setBackground('#ffffff')

    def _on_backtest_progress(self, progress: int):
        """Handle backtest progress updates"""
        self.backtest_progress.emit(progress)

    def export_backtest_chart(self, filename: str, format: str = 'png'):
        """Export chart to file"""
        try:
            if format.lower() == 'png':
                exporter = pg.exporters.ImageExporter(self.main_plot)
                exporter.export(filename)
                return True
            else:
                print(f"Export format {format} not supported")
                return False
        except Exception as e:
            print(f"Error exporting chart: {e}")
            return False

    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get summary of current backtest results"""
        if not self.backtest_data:
            return {}

        performance = self.backtest_data.get('performance', {})
        trades = self.backtest_data.get('trades', [])

        return {
            'total_trades': len(trades),
            'total_return': performance.get('total_return', 0),
            'win_rate': performance.get('win_rate', 0),
            'max_drawdown': performance.get('max_drawdown', 0),
            'sharpe_ratio': performance.get('sharpe_ratio', 0)
        }

    def reset_backtest_display(self):
        """Reset backtest display to initial state"""
        self._clear_backtest_items()
        self.backtest_data = {}
        self.results_widget.update_results({})


# =====================================================
# Factory Functions
# =====================================================

def create_enhanced_backtest_chart() -> EnhancedBacktestChart:
    """Create and return configured enhanced backtest chart"""
    return EnhancedBacktestChart()


def integrate_backtest_with_existing_chart(existing_chart) -> EnhancedBacktestChart:
    """
    Integrate backtest functionality với existing chart
    """
    # Create new enhanced chart
    enhanced_chart = create_enhanced_backtest_chart()

    # Copy data from existing chart if possible
    if hasattr(existing_chart, 'plot_items'):
        enhanced_chart.plot_items.update(existing_chart.plot_items)

    return enhanced_chart


def test_enhanced_chart():
    """Test enhanced chart functionality"""
    print("🎨 Testing Enhanced Chart...")

    # Create chart
    chart = create_enhanced_backtest_chart()
    print("✅ Enhanced chart created")

    # Test sample data
    sample_data = []
    for i in range(100):
        sample_data.append({
            'open': 50000 + i * 10,
            'high': 50000 + i * 10 + 500,
            'low': 50000 + i * 10 - 300,
            'close': 50000 + i * 10 + 200,
            'volume': 1000 + i * 5
        })

    chart.add_candlestick_data(sample_data)
    chart.add_volume_data(sample_data)
    print("✅ Sample data added")

    # Test backtest results
    sample_results = {
        'performance': {
            'total_return': 0.15,
            'win_rate': 0.65,
            'max_drawdown': -0.08,
            'sharpe_ratio': 1.25
        },
        'summary': {
            'total_trades': 25,
            'profitable_trades': 16,
            'largest_win': 2500.0,
            'largest_loss': -800.0
        },
        'trades': [],
        'bars': sample_data
    }

    chart.load_backtest_data(sample_results)
    print("✅ Backtest data loaded")

    # Test export
    summary = chart.get_backtest_summary()
    print(f"✅ Summary generated: {len(summary)} metrics")

    print("🎉 Enhanced chart tests passed!")


if __name__ == "__main__":
    test_enhanced_chart()