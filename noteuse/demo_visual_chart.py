# Tạo file: demo_visual_chart.py
"""
Demo Visual Chart with Backtest Markers
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
import numpy as np
from datetime import datetime, timedelta

from noteuse.config.display_config import DisplayConfig
from .demo_backtest_display import MarkerManager

class BacktestChartDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtest Markers Demo")
        self.setGeometry(100, 100, 1200, 800)

        # Create config and manager
        self.config = DisplayConfig()
        self.marker_manager = MarkerManager(self.config)

        self.setup_ui()
        self.create_sample_data()

    def setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create chart
        self.chart_widget = pg.PlotWidget()
        self.chart_widget.setLabel('left', 'Price')
        self.chart_widget.setLabel('bottom', 'Time')
        self.chart_widget.setTitle('Backtest Results with Trade Markers')
        layout.addWidget(self.chart_widget)

    def create_sample_data(self):
        # Sample price data
        np.random.seed(42)
        n_bars = 100
        base_price = 100
        price_data = base_price + np.cumsum(np.random.randn(n_bars) * 0.5)

        # Sample trade data
        trades_data = [
            {'datetime': datetime.now() + timedelta(hours=10), 'price': price_data[10], 'is_long': True, 'volume': 1.0, 'trade_id': 'T1', 'pnl': 0, 'position': 1, 'cumulative_pnl': 0},
            {'datetime': datetime.now() + timedelta(hours=25), 'price': price_data[25], 'is_long': False, 'volume': 1.0, 'trade_id': 'T1', 'pnl': 2.5, 'position': 0, 'cumulative_pnl': 2.5},
            {'datetime': datetime.now() + timedelta(hours=40), 'price': price_data[40], 'is_long': True, 'volume': 1.5, 'trade_id': 'T2', 'pnl': 0, 'position': 1.5, 'cumulative_pnl': 2.5},
            {'datetime': datetime.now() + timedelta(hours=70), 'price': price_data[70], 'is_long': False, 'volume': 1.5, 'trade_id': 'T2', 'pnl': -1.0, 'position': 0, 'cumulative_pnl': 1.5},
        ]

        bars_data = [{'datetime': datetime.now() + timedelta(hours=i), 'close': price_data[i]} for i in range(n_bars)]

        # Plot price line
        self.chart_widget.plot(range(n_bars), price_data, pen='white', name='Price')

        # Process backtest data
        self.marker_manager.process_backtest_data(trades_data, bars_data)

        # Add markers to chart
        plot_items = self.marker_manager.get_all_plot_items()
        for item in plot_items:
            self.chart_widget.addItem(item)

        # Update statistics
        stats = {
            'Total Return': 1.5,
            'Win Rate': 0.5,
            'Total Trades': 2,
            'Max Drawdown': -1.0
        }
        self.marker_manager.update_statistics(stats)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BacktestChartDemo()
    window.show()
    sys.exit(app.exec())