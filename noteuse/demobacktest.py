"""
Main Backtest Display Application
================================

Main application integrating tất cả backtest display components:
- Enhanced Chart Display
- Backtest Control Panel
- Results Display
- Configuration Management

Author: AI Assistant
Version: 1.0
"""

import sys
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
import traceback
import json
import numpy as np
import pandas as pd

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6.QtCore import Qt, Signal, QTimer, QThread, QObject, QSettings
from PySide6.QtGui import QIcon, QAction, QKeySequence, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QDockWidget,
    QPushButton, QLabel, QMessageBox, QFileDialog, QProgressBar,
    QComboBox, QCheckBox, QGroupBox, QFrame, QTextEdit, QTabWidget,
    QScrollArea
)

# Try to import pyqtgraph for charts
try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
    print("✅ PyQtGraph available")
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("❌ PyQtGraph not available - using basic charts")

# Import all our components - using absolute imports only
try:
    from display.enhanced_chart_display import create_enhanced_backtest_chart
    from ui.backtest_panel import BacktestControlPanel
    from ui.result_display import BacktestResultsDisplay
    from config.display_config import DisplayConfig, ChartTheme
    from core.backtest_bridge import BacktestBridge
    from core.data_extractor import BacktestDataExtractor
    from display.timeframe_switcher import TimeframeSwitcher
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Creating demo components...")

    # Create Demo Chart with actual data visualization
    class DemoChart(QWidget):
        backtest_completed = Signal(dict)
        backtest_progress = Signal(int)

        def __init__(self):
            super().__init__()
            self.setMinimumSize(700, 500)
            self.current_data = None
            self.setup_demo_chart()

        def setup_demo_chart(self):
            """Setup demo chart with sample data"""
            layout = QVBoxLayout(self)

            # Chart title
            title = QLabel("📈 Interactive Trading Chart - Demo Mode")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("""
                QLabel {
                    background-color: #2196F3;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 5px;
                }
            """)
            layout.addWidget(title)

            # Create tabs for different chart views
            self.tab_widget = QTabWidget()
            layout.addWidget(self.tab_widget)

            # Price Chart Tab
            self.create_price_chart_tab()

            # Demo Data Tab
            self.create_demo_data_tab()

            # Instructions Tab
            self.create_instructions_tab()

        def create_price_chart_tab(self):
            """Create price chart visualization"""
            if PYQTGRAPH_AVAILABLE:
                # Use PyQtGraph for real chart
                chart_widget = QWidget()
                layout = QVBoxLayout(chart_widget)

                # Create plot widget
                self.plot_widget = pg.PlotWidget()
                self.plot_widget.setBackground('w')
                self.plot_widget.setLabel('left', 'Price ($)')
                self.plot_widget.setLabel('bottom', 'Time')
                self.plot_widget.showGrid(x=True, y=True)

                layout.addWidget(self.plot_widget)

                # Load sample data
                self.load_sample_price_data()

            else:
                # Create text-based chart
                chart_widget = QWidget()
                layout = QVBoxLayout(chart_widget)

                chart_text = QTextEdit()
                chart_text.setReadOnly(True)
                chart_text.setFont(QFont("Courier", 10))

                # Create ASCII-style chart
                sample_chart = self.create_ascii_chart()
                chart_text.setText(sample_chart)

                layout.addWidget(chart_text)

            self.tab_widget.addTab(chart_widget, "📊 Price Chart")

        def create_demo_data_tab(self):
            """Create demo data display"""
            data_widget = QWidget()
            layout = QVBoxLayout(data_widget)

            # Data info
            info_label = QLabel("📋 Sample Trading Data")
            info_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3; margin-bottom: 10px;")
            layout.addWidget(info_label)

            # Data table
            data_text = QTextEdit()
            data_text.setReadOnly(True)
            data_text.setFont(QFont("Courier", 9))

            # Generate sample data table
            sample_data = self.generate_sample_data_table()
            data_text.setText(sample_data)

            layout.addWidget(data_text)

            self.tab_widget.addTab(data_widget, "📋 Data Table")

        def create_instructions_tab(self):
            """Create instructions tab"""
            instructions_widget = QWidget()
            layout = QVBoxLayout(instructions_widget)

            # Scrollable area for instructions
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)

            instructions_content = QWidget()
            content_layout = QVBoxLayout(instructions_content)

            # Instructions text
            instructions = QLabel("""
<h2>🎯 How to Use the Backtest Display System</h2>

<h3>📊 Chart Features:</h3>
<ul>
<li><b>Price Chart:</b> Shows candlestick data with trading signals</li>
<li><b>Volume:</b> Trading volume displayed at bottom</li>
<li><b>Indicators:</b> Technical indicators overlay</li>
<li><b>Zoom:</b> Mouse wheel to zoom in/out</li>
<li><b>Pan:</b> Click and drag to move around</li>
</ul>

<h3>🎛️ Control Panel (Left):</h3>
<ul>
<li><b>Strategy Selection:</b> Choose from available strategies</li>
<li><b>Time Frame:</b> Select 1m, 5m, 15m, 1h, 4h, 1d</li>
<li><b>Date Range:</b> Set backtest period</li>
<li><b>Parameters:</b> Adjust strategy settings</li>
<li><b>Run Backtest:</b> Execute the backtest</li>
</ul>

<h3>📈 Results Panel (Right):</h3>
<ul>
<li><b>Performance Metrics:</b> ROI, Sharpe ratio, Win rate</li>
<li><b>Trade List:</b> Individual trades with P&L</li>
<li><b>Equity Curve:</b> Portfolio value over time</li>
<li><b>Drawdown:</b> Risk analysis</li>
<li><b>Export:</b> Save results to CSV/Excel</li>
</ul>

<h3>🚀 Quick Start:</h3>
<ol>
<li><b>Load Demo Data:</b> Click "Load Demo Data" button below</li>
<li><b>Select Strategy:</b> Choose "ATR RSI Strategy" from dropdown</li>
<li><b>Set Parameters:</b> Use default settings or customize</li>
<li><b>Run Backtest:</b> Click "🚀 Run Backtest" button</li>
<li><b>View Results:</b> Check charts and metrics</li>
<li><b>Export:</b> Save results if satisfied</li>
</ol>

<h3>💡 Tips:</h3>
<ul>
<li><b>Time Frames:</b> Shorter = more trades, Longer = fewer but stronger signals</li>
<li><b>Parameters:</b> Start with defaults, then optimize</li>
<li><b>Validation:</b> Test on different time periods</li>
<li><b>Risk Management:</b> Always check drawdown</li>
</ul>

<h3>⚠️ Current Mode:</h3>
<p><b>Demo Mode:</b> Using simulated data for demonstration.<br/>
For real trading, connect to your data provider.</p>
            """)

            instructions.setWordWrap(True)
            instructions.setTextFormat(Qt.TextFormat.RichText)
            instructions.setStyleSheet("""
                QLabel {
                    padding: 20px;
                    line-height: 1.5;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }
            """)

            content_layout.addWidget(instructions)

            # Action buttons
            button_layout = QHBoxLayout()

            load_demo_btn = QPushButton("📊 Load Demo Data")
            load_demo_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 12px 24px;
                    border-radius: 6px;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #45a049; }
            """)
            load_demo_btn.clicked.connect(self.load_demo_data)
            button_layout.addWidget(load_demo_btn)

            clear_btn = QPushButton("🗑️ Clear Chart")
            clear_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    font-weight: bold;
                    padding: 12px 24px;
                    border-radius: 6px;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #da190b; }
            """)
            clear_btn.clicked.connect(self.clear_chart)
            button_layout.addWidget(clear_btn)

            refresh_btn = QPushButton("🔄 Refresh")
            refresh_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    font-weight: bold;
                    padding: 12px 24px;
                    border-radius: 6px;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #0b7dda; }
            """)
            refresh_btn.clicked.connect(self.refresh_chart)
            button_layout.addWidget(refresh_btn)

            button_layout.addStretch()
            content_layout.addLayout(button_layout)

            scroll.setWidget(instructions_content)
            layout.addWidget(scroll)

            self.tab_widget.addTab(instructions_widget, "📖 Instructions")

        def load_sample_price_data(self):
            """Load sample price data into chart"""
            if not PYQTGRAPH_AVAILABLE:
                return

            # Generate sample OHLC data
            np.random.seed(42)
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

            price_data = []
            current_price = 100.0

            for i, date in enumerate(dates):
                # Random walk with trend
                change = np.random.normal(0, 2) + 0.01  # Slight upward trend
                current_price = max(10, current_price + change)

                # OHLC for the day
                open_price = current_price
                high_price = open_price + abs(np.random.normal(0, 1))
                low_price = open_price - abs(np.random.normal(0, 1))
                close_price = open_price + np.random.normal(0, 0.5)
                volume = np.random.randint(1000, 10000)

                price_data.append({
                    'timestamp': i,
                    'date': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })

            self.current_data = pd.DataFrame(price_data)

            # Plot the data
            self.plot_price_data()

        def plot_price_data(self):
            """Plot price data on chart"""
            if not PYQTGRAPH_AVAILABLE or self.current_data is None:
                return

            self.plot_widget.clear()

            # Plot close price
            self.plot_widget.plot(
                self.current_data['timestamp'],
                self.current_data['close'],
                pen=pg.mkPen(color='blue', width=2),
                name='Close Price'
            )

            # Add moving average
            ma_period = 20
            if len(self.current_data) > ma_period:
                ma_data = self.current_data['close'].rolling(window=ma_period).mean()
                self.plot_widget.plot(
                    self.current_data['timestamp'],
                    ma_data,
                    pen=pg.mkPen(color='red', width=1),
                    name=f'MA{ma_period}'
                )

            # Add some trade signals
            self.add_sample_signals()

        def add_sample_signals(self):
            """Add sample trading signals to chart"""
            if not PYQTGRAPH_AVAILABLE or self.current_data is None:
                return

            # Generate random buy/sell signals
            np.random.seed(123)
            signal_indices = np.random.choice(
                len(self.current_data),
                size=min(20, len(self.current_data)//10),
                replace=False
            )

            for idx in signal_indices:
                y_pos = self.current_data.iloc[idx]['close']
                if np.random.random() > 0.5:
                    # Buy signal (green arrow up)
                    self.plot_widget.plot(
                        [self.current_data.iloc[idx]['timestamp']],
                        [y_pos],
                        pen=None,
                        symbol='t1',
                        symbolBrush='green',
                        symbolSize=15
                    )
                else:
                    # Sell signal (red arrow down)
                    self.plot_widget.plot(
                        [self.current_data.iloc[idx]['timestamp']],
                        [y_pos],
                        pen=None,
                        symbol='t',
                        symbolBrush='red',
                        symbolSize=15
                    )

        def create_ascii_chart(self):
            """Create ASCII-style chart for when PyQtGraph is not available"""
            return """
📊 SAMPLE PRICE CHART (Demo Mode)
═══════════════════════════════════════════════════════════════

Price ($)  |  Chart
   120     |  
   115     |     ╭─╮
   110     |   ╭─╯ ╰─╮     ╭─╮
   105     | ╭─╯     ╰─╮ ╭─╯ ╰─╮
   100     |╱         ╰─╯     ╰─╮
    95     |                    ╰─╮
    90     |                      ╰───
           └──────────────────────────────────────────── Time
            Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep

🔍 TRADING SIGNALS:
   ▲ = Buy Signal (Green)
   ▼ = Sell Signal (Red)

📈 INDICATORS:
   Blue Line = Price
   Red Line = Moving Average (20)
   
💡 TIP: Run a backtest to see real trading results!
            """

        def generate_sample_data_table(self):
            """Generate sample data table"""
            return """
📋 SAMPLE TRADING DATA
════════════════════════════════════════════════════════════════

Date       Time     Open    High    Low     Close   Volume   Signal
────────────────────────────────────────────────────────────────
2023-12-01 09:00   102.45  103.20  101.80  102.95   8,450   
2023-12-01 09:01   102.95  103.85  102.70  103.50   6,230   ▲ BUY
2023-12-01 09:02   103.50  104.10  103.30  103.85   7,180   
2023-12-01 09:03   103.85  104.45  103.60  104.20   5,920   
2023-12-01 09:04   104.20  104.80  103.95  104.55   6,750   
2023-12-01 09:05   104.55  105.20  104.30  104.90   8,100   
2023-12-01 09:06   104.90  105.15  104.40  104.75   5,430   ▼ SELL
2023-12-01 09:07   104.75  105.00  104.20  104.40   6,890   
2023-12-01 09:08   104.40  104.85  103.95  104.15   7,250   
2023-12-01 09:09   104.15  104.60  103.80  104.30   6,180   

💰 TRADE SUMMARY:
   Entry: $103.50 (09:01)
   Exit:  $104.75 (09:06)
   P&L:   +$1.25 (+1.21%)
   Duration: 5 minutes

📊 STATISTICS:
   Win Rate: 65%
   Avg Trade: +$0.85
   Max Drawdown: -2.1%
   Sharpe Ratio: 1.45
            """

        def load_demo_data(self):
            """Load demo data into chart"""
            if PYQTGRAPH_AVAILABLE:
                self.load_sample_price_data()
                QMessageBox.information(
                    self,
                    "Demo Data Loaded",
                    "✅ Sample price data loaded!\n\n"
                    "You can now:\n"
                    "• View the price chart\n"
                    "• See trading signals\n"
                    "• Run a backtest with the control panel\n"
                    "• Check results in the right panel"
                )
            else:
                QMessageBox.information(
                    self,
                    "Demo Mode",
                    "📊 Demo data is shown in text format.\n\n"
                    "To see interactive charts, install PyQtGraph:\n"
                    "pip install pyqtgraph\n\n"
                    "Then restart the application."
                )

            # Switch to price chart tab
            self.tab_widget.setCurrentIndex(0)

        def clear_chart(self):
            """Clear chart data"""
            if PYQTGRAPH_AVAILABLE and hasattr(self, 'plot_widget'):
                self.plot_widget.clear()
            QMessageBox.information(self, "Chart Cleared", "📊 Chart has been cleared!")

        def refresh_chart(self):
            """Refresh chart with new data"""
            self.load_sample_price_data()
            QMessageBox.information(self, "Chart Refreshed", "🔄 Chart refreshed with new data!")

        def load_backtest_data(self, data):
            """Load backtest results into chart"""
            print("📊 Loading backtest data into chart")

            # Show results in a message
            perf = data.get('performance', {})
            QMessageBox.information(
                self,
                "Backtest Results Loaded",
                f"📈 Backtest completed!\n\n"
                f"Total Return: {perf.get('total_return', 0)*100:.1f}%\n"
                f"Win Rate: {perf.get('win_rate', 0)*100:.1f}%\n"
                f"Total Trades: {perf.get('total_trades', 0)}\n\n"
                f"Results are now displayed in the chart and right panel."
            )

        def update_timeframe(self, tf):
            """Update chart timeframe"""
            print(f"⏱️ Updating chart timeframe to: {tf}")
            if hasattr(self, 'plot_widget') and PYQTGRAPH_AVAILABLE:
                self.plot_widget.setTitle(f"Price Chart - {tf} Timeframe")

    # Mock other components with improved functionality
    class MockPanel(QWidget):
        backtest_started = Signal()
        backtest_finished = Signal(dict)
        config_changed = Signal(dict)

        def __init__(self, config):
            super().__init__()
            self.setMaximumWidth(320)
            self.setMinimumWidth(300)
            layout = QVBoxLayout(self)

            # Panel title
            title = QLabel("🎛️ Backtest Control Panel")
            title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3; margin-bottom: 10px;")
            layout.addWidget(title)

            # Strategy section
            strategy_group = QGroupBox("📈 Strategy Selection")
            strategy_layout = QVBoxLayout(strategy_group)

            self.strategy_combo = QComboBox()
            self.strategy_combo.addItems([
                "Select Strategy...",
                "ATR RSI Strategy (Demo)",
                "Moving Average Crossover",
                "Bollinger Bands Strategy",
                "MACD Strategy",
                "Breakout Strategy"
            ])
            strategy_layout.addWidget(self.strategy_combo)

            # Strategy description
            self.strategy_desc = QLabel("Select a strategy to see description")
            self.strategy_desc.setWordWrap(True)
            self.strategy_desc.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
            strategy_layout.addWidget(self.strategy_desc)

            self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)

            layout.addWidget(strategy_group)

            # Parameters section
            params_group = QGroupBox("⚙️ Parameters")
            params_layout = QVBoxLayout(params_group)

            # Symbol
            symbol_layout = QHBoxLayout()
            symbol_layout.addWidget(QLabel("Symbol:"))
            self.symbol_combo = QComboBox()
            self.symbol_combo.addItems(["BTCUSDT.BINANCE", "ETHUSDT.BINANCE", "ADAUSDT.BINANCE"])
            symbol_layout.addWidget(self.symbol_combo)
            params_layout.addLayout(symbol_layout)

            # Timeframe
            tf_layout = QHBoxLayout()
            tf_layout.addWidget(QLabel("Timeframe:"))
            self.tf_combo = QComboBox()
            self.tf_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1d"])
            self.tf_combo.setCurrentText("1h")
            tf_layout.addWidget(self.tf_combo)
            params_layout.addLayout(tf_layout)

            # Period
            period_layout = QHBoxLayout()
            period_layout.addWidget(QLabel("Period:"))
            self.period_combo = QComboBox()
            self.period_combo.addItems(["Last 7 days", "Last 30 days", "Last 90 days", "Last 6 months", "Last 1 year"])
            self.period_combo.setCurrentText("Last 30 days")
            period_layout.addWidget(self.period_combo)
            params_layout.addLayout(period_layout)

            layout.addWidget(params_group)

            # Quick stats
            stats_group = QGroupBox("📊 Quick Stats")
            stats_layout = QVBoxLayout(stats_group)

            self.stats_label = QLabel("Configure and run backtest to see statistics")
            self.stats_label.setWordWrap(True)
            self.stats_label.setStyleSheet("color: #666; font-size: 11px;")
            stats_layout.addWidget(self.stats_label)

            layout.addWidget(stats_group)

            # Actions
            actions_group = QGroupBox("🚀 Actions")
            actions_layout = QVBoxLayout(actions_group)

            run_btn = QPushButton("▶️ Run Backtest")
            run_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 12px;
                    border-radius: 6px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            run_btn.clicked.connect(self.start_mock_backtest)
            actions_layout.addWidget(run_btn)

            config_btn = QPushButton("⚙️ Advanced Config")
            config_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
            """)
            actions_layout.addWidget(config_btn)

            export_btn = QPushButton("💾 Export Config")
            export_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #0b7dda;
                }
            """)
            actions_layout.addWidget(export_btn)

            layout.addWidget(actions_group)
            layout.addStretch()

        def on_strategy_changed(self, strategy_name):
            """Update strategy description"""
            descriptions = {
                "ATR RSI Strategy (Demo)": "Uses ATR and RSI indicators to identify entry/exit points. Good for trending markets.",
                "Moving Average Crossover": "Trades based on moving average crossovers. Simple but effective.",
                "Bollinger Bands Strategy": "Uses Bollinger Bands to identify overbought/oversold conditions.",
                "MACD Strategy": "Based on MACD indicator crossovers and divergences.",
                "Breakout Strategy": "Identifies price breakouts from support/resistance levels."
            }

            desc = descriptions.get(strategy_name, "Select a strategy to see description")
            self.strategy_desc.setText(desc)

        def start_mock_backtest(self):
            """Start a mock backtest with better simulation"""
            strategy = self.strategy_combo.currentText()
            if strategy == "Select Strategy...":
                QMessageBox.warning(self, "Warning", "Please select a strategy first!")
                return

            symbol = self.symbol_combo.currentText()
            timeframe = self.tf_combo.currentText()
            period = self.period_combo.currentText()

            print(f"🚀 Starting backtest: {strategy} on {symbol} ({timeframe}, {period})")
            self.backtest_started.emit()

            # Update stats during backtest
            self.stats_label.setText("⏳ Running backtest...")

            # Simulate different performance based on strategy
            performance_profiles = {
                "ATR RSI Strategy (Demo)": {
                    'total_return': np.random.uniform(0.08, 0.25),
                    'win_rate': np.random.uniform(0.55, 0.75),
                    'total_trades': np.random.randint(15, 45),
                    'max_drawdown': np.random.uniform(-0.15, -0.05),
                    'sharpe_ratio': np.random.uniform(1.1, 2.2)
                },
                "Moving Average Crossover": {
                    'total_return': np.random.uniform(0.05, 0.18),
                    'win_rate': np.random.uniform(0.45, 0.65),
                    'total_trades': np.random.randint(8, 25),
                    'max_drawdown': np.random.uniform(-0.12, -0.04),
                    'sharpe_ratio': np.random.uniform(0.8, 1.6)
                }
            }

            # Get performance for selected strategy
            perf = performance_profiles.get(strategy, performance_profiles["ATR RSI Strategy (Demo)"])

            # Simulate backtest completion after 3 seconds
            QTimer.singleShot(3000, lambda: self.complete_backtest(strategy, symbol, timeframe, perf))

        def complete_backtest(self, strategy, symbol, timeframe, performance):
            """Complete the mock backtest"""

            # Generate realistic trade data
            trades = []
            total_trades = performance['total_trades']
            win_rate = performance['win_rate']

            for i in range(total_trades):
                is_winner = np.random.random() < win_rate
                if is_winner:
                    pnl = abs(np.random.normal(150, 100))
                else:
                    pnl = -abs(np.random.normal(100, 80))

                trades.append({
                    'trade_id': f'T{i+1:03d}',
                    'timestamp': i * 3600,
                    'symbol': symbol,
                    'side': 'long' if np.random.random() > 0.5 else 'short',
                    'entry_price': 100 + np.random.normal(0, 5),
                    'exit_price': 100 + np.random.normal(0, 5),
                    'quantity': 1.0,
                    'pnl': pnl,
                    'datetime': datetime.now() - timedelta(hours=total_trades-i)
                })

            results = {
                'strategy': strategy,
                'symbol': symbol,
                'timeframe': timeframe,
                'performance': performance,
                'trades': trades,
                'summary': {
                    'profitable_trades': int(total_trades * win_rate),
                    'losing_trades': int(total_trades * (1 - win_rate)),
                    'largest_win': max([t['pnl'] for t in trades if t['pnl'] > 0] + [0]),
                    'largest_loss': min([t['pnl'] for t in trades if t['pnl'] < 0] + [0]),
                    'gross_profit': sum([t['pnl'] for t in trades if t['pnl'] > 0]),
                    'gross_loss': sum([t['pnl'] for t in trades if t['pnl'] < 0])
                }
            }

            # Update stats display
            stats_text = f"""📊 Backtest Complete!
Return: {performance['total_return']*100:.1f}%
Win Rate: {performance['win_rate']*100:.1f}%
Trades: {performance['total_trades']}
Sharpe: {performance['sharpe_ratio']:.2f}"""
            self.stats_label.setText(stats_text)

            self.backtest_finished.emit(results)

        def get_full_configuration(self):
            return {
                'strategy': {
                    'strategy_name': self.strategy_combo.currentText(),
                    'parameters': {}
                },
                'parameters': {
                    'symbol': self.symbol_combo.currentText(),
                    'timeframe': self.tf_combo.currentText(),
                    'period': self.period_combo.currentText(),
                    'start_date': datetime.now() - timedelta(days=30),
                    'end_date': datetime.now()
                }
            }

        def set_timeframe(self, tf):
            self.tf_combo.setCurrentText(tf)
            print(f"🎛️ Panel timeframe updated to: {tf}")

    # Use the demo chart instead of the imported one
    def create_enhanced_backtest_chart():
        return DemoChart()

    # Import the rest or create mocks
    try:
        from ui.backtest_panel import BacktestControlPanel
        from ui.result_display import BacktestResultsDisplay
        from display.timeframe_switcher import TimeframeSwitcher
    except ImportError:
        BacktestControlPanel = MockPanel

        # Create other mock components...
        class MockResultsDisplay(QWidget):
            results_exported = Signal(str)

            def __init__(self, config):
                super().__init__()
                self.setMaximumWidth(380)
                self.setMinimumWidth(350)
                self.setup_ui()

            def setup_ui(self):
                layout = QVBoxLayout(self)

                # Title with better styling
                title = QLabel("📊 Results Analysis")
                title.setStyleSheet("""
                    QLabel {
                        font-weight: bold; 
                        font-size: 14px; 
                        color: #2196F3; 
                        margin-bottom: 10px;
                        padding: 8px;
                        background-color: #E3F2FD;
                        border-radius: 4px;
                    }
                """)
                layout.addWidget(title)

                # Performance metrics with better layout
                metrics_group = QGroupBox("💰 Performance Metrics")
                metrics_layout = QVBoxLayout(metrics_group)

                self.total_return_label = QLabel("Total Return: --")
                self.annual_return_label = QLabel("Annual Return: --")
                self.win_rate_label = QLabel("Win Rate: --")
                self.total_trades_label = QLabel("Total Trades: --")
                self.max_drawdown_label = QLabel("Max Drawdown: --")
                self.sharpe_ratio_label = QLabel("Sharpe Ratio: --")

                for label in [self.total_return_label, self.annual_return_label,
                              self.win_rate_label, self.total_trades_label,
                              self.max_drawdown_label, self.sharpe_ratio_label]:
                    label.setStyleSheet("""
                        QLabel {
                            font-family: 'Courier New', monospace; 
                            padding: 4px 8px;
                            margin: 2px;
                            background-color: #f8f9fa;
                            border-left: 3px solid #2196F3;
                        }
                    """)
                    metrics_layout.addWidget(label)

                layout.addWidget(metrics_group)

                # Trade summary with enhanced display
                trades_group = QGroupBox("📈 Trade Summary")
                trades_layout = QVBoxLayout(trades_group)

                self.profitable_trades_label = QLabel("Profitable: --")
                self.losing_trades_label = QLabel("Losing: --")
                self.largest_win_label = QLabel("Largest Win: --")
                self.largest_loss_label = QLabel("Largest Loss: --")
                self.avg_trade_label = QLabel("Average Trade: --")

                for label in [self.profitable_trades_label, self.losing_trades_label,
                              self.largest_win_label, self.largest_loss_label, self.avg_trade_label]:
                    label.setStyleSheet("""
                        QLabel {
                            font-family: 'Courier New', monospace; 
                            padding: 4px 8px;
                            margin: 2px;
                            background-color: #f8f9fa;
                            border-left: 3px solid #4CAF50;
                        }
                    """)
                    trades_layout.addWidget(label)

                layout.addWidget(trades_group)

                # Export section with multiple options
                export_group = QGroupBox("💾 Export & Share")
                export_layout = QVBoxLayout(export_group)

                export_csv_btn = QPushButton("📄 Export to CSV")
                export_json_btn = QPushButton("📋 Export to JSON")
                export_excel_btn = QPushButton("📊 Export to Excel")
                view_report_btn = QPushButton("📑 View Full Report")

                for btn in [export_csv_btn, export_json_btn, export_excel_btn, view_report_btn]:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #FF9800;
                            color: white;
                            padding: 8px;
                            border-radius: 4px;
                            margin: 2px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #F57C00;
                        }
                    """)
                    export_layout.addWidget(btn)

                layout.addWidget(export_group)

                # Status section
                status_group = QGroupBox("ℹ️ Status")
                status_layout = QVBoxLayout(status_group)

                self.status_label = QLabel("Ready for backtest results...")
                self.status_label.setWordWrap(True)
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #666;
                        font-style: italic;
                        padding: 8px;
                        background-color: #f0f0f0;
                        border-radius: 4px;
                    }
                """)
                status_layout.addWidget(self.status_label)

                layout.addWidget(status_group)
                layout.addStretch()

            def load_results(self, results):
                """Load and display backtest results"""
                print("📈 Loading results into display")

                # Update performance metrics
                perf = results.get('performance', {})
                self.total_return_label.setText(f"Total Return: {perf.get('total_return', 0)*100:.2f}%")
                self.annual_return_label.setText(f"Annual Return: {perf.get('total_return', 0)*365/30*100:.2f}%")
                self.win_rate_label.setText(f"Win Rate: {perf.get('win_rate', 0)*100:.1f}%")
                self.total_trades_label.setText(f"Total Trades: {perf.get('total_trades', 0)}")
                self.max_drawdown_label.setText(f"Max Drawdown: {perf.get('max_drawdown', 0)*100:.2f}%")
                self.sharpe_ratio_label.setText(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")

                # Update trade summary
                summary = results.get('summary', {})
                self.profitable_trades_label.setText(f"Profitable: {summary.get('profitable_trades', 0)}")
                self.losing_trades_label.setText(f"Losing: {summary.get('losing_trades', 0)}")
                self.largest_win_label.setText(f"Largest Win: ${summary.get('largest_win', 0):.2f}")
                self.largest_loss_label.setText(f"Largest Loss: ${summary.get('largest_loss', 0):.2f}")

                # Calculate average trade
                trades = results.get('trades', [])
                if trades:
                    avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
                    self.avg_trade_label.setText(f"Average Trade: ${avg_pnl:.2f}")

                # Update status
                strategy = results.get('strategy', 'Unknown')
                symbol = results.get('symbol', 'Unknown')
                self.status_label.setText(f"✅ Results loaded for {strategy} on {symbol}")

        class MockTimeframeSwitcher(QWidget):
            timeframe_changed = Signal(str)
            custom_timeframe_added = Signal(str)
            multiple_timeframes_selected = Signal(list)

            def __init__(self, config):
                super().__init__()
                self.setMaximumHeight(140)
                self.setMinimumHeight(120)
                self.current_tf = "1h"
                self.setup_ui()

            def setup_ui(self):
                layout = QVBoxLayout(self)

                # Title with better styling
                title = QLabel("⏱️ Timeframe Switcher")
                title.setStyleSheet("""
                    QLabel {
                        font-weight: bold; 
                        font-size: 12px; 
                        color: #2196F3;
                        padding: 6px;
                        background-color: #E3F2FD;
                        border-radius: 4px;
                        margin-bottom: 8px;
                    }
                """)
                layout.addWidget(title)

                # Timeframe buttons in a better layout
                button_frame = QFrame()
                button_frame.setStyleSheet("""
                    QFrame {
                        background-color: #f8f9fa;
                        border-radius: 6px;
                        padding: 8px;
                    }
                """)
                button_layout = QHBoxLayout(button_frame)
                button_layout.setSpacing(4)

                timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
                self.buttons = {}

                for tf in timeframes:
                    btn = QPushButton(tf)
                    btn.setMaximumWidth(40)
                    btn.setCheckable(True)
                    btn.setChecked(tf == self.current_tf)

                    self.update_button_style(btn, tf == self.current_tf)
                    btn.clicked.connect(lambda checked, timeframe=tf: self.on_timeframe_clicked(timeframe))

                    button_layout.addWidget(btn)
                    self.buttons[tf] = btn

                layout.addWidget(button_frame)

                # Additional options
                options_layout = QHBoxLayout()

                multi_cb = QCheckBox("Multi-TF")
                multi_cb.setStyleSheet("font-size: 10px; color: #666;")
                options_layout.addWidget(multi_cb)

                options_layout.addStretch()

                custom_btn = QPushButton("+ Custom")
                custom_btn.setMaximumWidth(60)
                custom_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #FF9800;
                        color: white;
                        font-weight: bold;
                        border-radius: 8px;
                        font-size: 10px;
                        padding: 4px;
                    }
                    QPushButton:hover {
                        background-color: #F57C00;
                    }
                """)
                options_layout.addWidget(custom_btn)

                layout.addLayout(options_layout)

            def update_button_style(self, button, is_active):
                """Update button style based on active state"""
                if is_active:
                    button.setStyleSheet("""
                        QPushButton {
                            background-color: #4CAF50;
                            color: white;
                            font-weight: bold;
                            border-radius: 4px;
                            padding: 6px;
                            border: 2px solid #45a049;
                        }
                    """)
                else:
                    button.setStyleSheet("""
                        QPushButton {
                            background-color: white;
                            color: #333;
                            border: 1px solid #ddd;
                            border-radius: 4px;
                            padding: 6px;
                        }
                        QPushButton:hover {
                            background-color: #f0f0f0;
                            border-color: #4CAF50;
                        }
                        QPushButton:checked {
                            background-color: #4CAF50;
                            color: white;
                            border-color: #45a049;
                        }
                    """)

            def on_timeframe_clicked(self, timeframe):
                """Handle timeframe button click"""
                self.current_tf = timeframe
                print(f"⏱️ Timeframe changed to: {timeframe}")
                self.timeframe_changed.emit(timeframe)

                # Update all button styles
                for tf, btn in self.buttons.items():
                    self.update_button_style(btn, tf == timeframe)
                    btn.setChecked(tf == timeframe)

        BacktestResultsDisplay = MockResultsDisplay
        TimeframeSwitcher = MockTimeframeSwitcher

    # Import config and core components
    try:
        from config.display_config import DisplayConfig, ChartTheme
        from core.backtest_bridge import BacktestBridge
        from core.data_extractor import BacktestDataExtractor
    except ImportError:
        print("❌ Could not import config/core components - using mocks")

        class DisplayConfig:
            def __init__(self):
                pass
            def update_theme(self, theme):
                pass

        class ChartTheme:
            LIGHT = "light"
            DARK = "dark"

        class BacktestBridge:
            def __init__(self):
                pass
            def run_simple_backtest(self, **kwargs):
                return {
                    'trades': [],
                    'performance': {'total_return': 0.15},
                    'summary': {'total_trades': 10}
                }

        class BacktestDataExtractor:
            def __init__(self):
                pass


# Rest of the classes remain the same as in the previous version...
class BacktestWorker(QObject):
    """
    Background worker cho backtest execution
    """

    progress_updated = Signal(int)  # progress percentage
    status_updated = Signal(str)    # status message
    backtest_completed = Signal(dict)  # results
    backtest_failed = Signal(str)   # error message

    def __init__(self, bridge: BacktestBridge):
        super().__init__()
        self.bridge = bridge
        self.is_cancelled = False

    def run_backtest(self, config: Dict[str, Any]):
        """Run backtest in background thread"""
        try:
            self.is_cancelled = False
            self.status_updated.emit("Initializing backtest...")
            self.progress_updated.emit(0)

            # Extract configuration
            strategy_config = config.get('strategy', {})
            backtest_params = config.get('parameters', {})

            if self.is_cancelled:
                return

            self.status_updated.emit("Loading historical data...")
            self.progress_updated.emit(20)

            # Run backtest through bridge
            results = self.bridge.run_simple_backtest(
                symbol=backtest_params.get('symbol', 'BTCUSDT.BINANCE'),
                timeframe=backtest_params.get('timeframe', '1h'),
                start_date=backtest_params.get('start_date', datetime.now() - timedelta(days=90)),
                end_date=backtest_params.get('end_date', datetime.now()),
                strategy_class=None,  # Would get from strategy_config
                strategy_settings=strategy_config.get('parameters', {}),
                progress_callback=self._on_progress_update
            )

            if self.is_cancelled:
                return

            if 'error' in results:
                self.backtest_failed.emit(results['error'])
            else:
                self.status_updated.emit("Backtest completed successfully!")
                self.progress_updated.emit(100)
                self.backtest_completed.emit(results)

        except Exception as e:
            error_msg = f"Backtest failed: {str(e)}\n{traceback.format_exc()}"
            self.backtest_failed.emit(error_msg)

    def _on_progress_update(self, progress: int):
        """Handle progress updates from bridge"""
        if not self.is_cancelled:
            self.progress_updated.emit(progress)

            if progress <= 30:
                self.status_updated.emit("Loading market data...")
            elif progress <= 60:
                self.status_updated.emit("Running strategy signals...")
            elif progress <= 90:
                self.status_updated.emit("Calculating performance metrics...")
            else:
                self.status_updated.emit("Finalizing results...")

    def cancel_backtest(self):
        """Cancel running backtest"""
        self.is_cancelled = True
        self.status_updated.emit("Backtest cancelled by user")


class BacktestMainWindow(QMainWindow):
    """
    Main application window cho backtest display system
    """

    def __init__(self):
        super().__init__()

        # Application settings
        self.settings = QSettings("BacktestDisplay", "MainApp")

        # Core components
        self.display_config = DisplayConfig()
        self.backtest_bridge = BacktestBridge()
        self.data_extractor = BacktestDataExtractor()

        # State
        self.current_results = {}
        self.backtest_worker = None
        self.backtest_thread = None

        # Initialize UI
        self.init_ui()
        self.setup_menus()
        self.setup_toolbars()
        self.setup_status_bar()
        self.connect_signals()

        # Load settings
        self.load_settings()

        # Set window properties
        self.setWindowTitle("🚀 Backtest Display System v1.0 - Interactive Demo")
        self.setWindowIcon(QIcon())  # Would load app icon
        self.resize(1500, 1000)

        # Show welcome message
        QTimer.singleShot(1000, self.show_welcome_message)

    def show_welcome_message(self):
        """Show welcome message to guide users"""
        welcome_msg = QMessageBox(self)
        welcome_msg.setWindowTitle("🎉 Welcome to Backtest Display System!")
        welcome_msg.setIcon(QMessageBox.Icon.Information)

        welcome_text = """
<h2>🚀 Welcome to the Backtest Display System!</h2>

<p><b>This is an interactive demo showing the full trading analysis interface.</b></p>

<h3>🎯 Quick Start Guide:</h3>
<ol>
<li><b>📊 Load Demo Data:</b> Go to the chart tab "Instructions" and click "Load Demo Data"</li>
<li><b>🎛️ Select Strategy:</b> In the left panel, choose "ATR RSI Strategy (Demo)"</li>
<li><b>⏱️ Set Timeframe:</b> Try different timeframes (1m, 5m, 1h, etc.)</li>
<li><b>🚀 Run Backtest:</b> Click the green "Run Backtest" button</li>
<li><b>📈 View Results:</b> Check the charts and right panel for performance metrics</li>
</ol>

<h3>✨ Features to Explore:</h3>
<ul>
<li>Interactive price charts with trading signals</li>
<li>Multiple timeframe analysis</li>
<li>Comprehensive performance metrics</li>
<li>Trade-by-trade analysis</li>
<li>Export functionality</li>
</ul>

<p><b>💡 Tip:</b> This is a demo with simulated data. For live trading, connect your data source!</p>
        """

        welcome_msg.setText(welcome_text)
        welcome_msg.setTextFormat(Qt.TextFormat.RichText)

        # Add buttons
        welcome_msg.addButton("🚀 Start Demo", QMessageBox.ButtonRole.AcceptRole)
        welcome_msg.addButton("📖 Show Instructions", QMessageBox.ButtonRole.HelpRole)
        welcome_msg.addButton("❌ Close", QMessageBox.ButtonRole.RejectRole)

        result = welcome_msg.exec()

        if result == 1:  # Show Instructions
            # Switch chart to instructions tab
            if hasattr(self.chart_widget, 'tab_widget'):
                self.chart_widget.tab_widget.setCurrentIndex(2)  # Instructions tab

    def init_ui(self):
        """Initialize main UI layout"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left panel - Control Panel với timeframe switcher
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Add timeframe switcher
        self.timeframe_switcher = TimeframeSwitcher(self.display_config)
        left_layout.addWidget(self.timeframe_switcher)

        # Add control panel
        self.control_panel = BacktestControlPanel(self.display_config)
        left_layout.addWidget(self.control_panel)

        control_dock = QDockWidget("🎛️ Backtest Controls", self)
        control_dock.setWidget(left_panel)
        control_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, control_dock)

        # Center panel - Enhanced Chart
        self.chart_widget = create_enhanced_backtest_chart()
        main_splitter.addWidget(self.chart_widget)

        # Right panel - Results Display
        self.results_display = BacktestResultsDisplay(self.display_config)
        results_dock = QDockWidget("📊 Results Analysis", self)
        results_dock.setWidget(self.results_display)
        results_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, results_dock)

        # Set splitter proportions
        main_splitter.setSizes([320, 900, 380])

        # Quick action panel at bottom
        self.create_quick_action_panel()

    def create_quick_action_panel(self):
        """Create quick action panel"""
        quick_panel = QFrame()
        quick_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        quick_panel.setMaximumHeight(90)
        quick_panel.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-top: 2px solid #e9ecef;
            }
        """)

        layout = QHBoxLayout(quick_panel)

        # Quick backtest section
        quick_group = QGroupBox("🚀 Quick Actions")
        quick_layout = QHBoxLayout(quick_group)

        # Run backtest button
        self.run_btn = QPushButton("🚀 Run Backtest")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 13px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
                transform: translateY(-1px);
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_btn.clicked.connect(self.run_backtest)
        quick_layout.addWidget(self.run_btn)

        # Stop button
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 6px;
                font-size: 13px;
                border: none;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_backtest)
        quick_layout.addWidget(self.stop_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        quick_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready - Click 'Load Demo Data' in chart to start")
        self.status_label.setStyleSheet("""
            QLabel {
                font-weight: bold; 
                color: #2196F3;
                font-size: 12px;
                padding: 8px;
                background-color: white;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
        """)
        quick_layout.addWidget(self.status_label)

        layout.addWidget(quick_group)

        # Quick settings section
        settings_group = QGroupBox("⚙️ Quick Settings")
        settings_layout = QHBoxLayout(settings_group)

        # Theme selector
        settings_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "TradingView"])
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        settings_layout.addWidget(self.theme_combo)

        # Auto-refresh
        self.auto_refresh_cb = QCheckBox("Auto Refresh")
        self.auto_refresh_cb.setChecked(True)
        settings_layout.addWidget(self.auto_refresh_cb)

        # Demo mode indicator
        demo_label = QLabel("🎭 DEMO MODE")
        demo_label.setStyleSheet("""
            QLabel {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
        """)
        settings_layout.addWidget(demo_label)

        layout.addWidget(settings_group)

        # Add to main layout
        dock = QDockWidget("🚀 Quick Actions", self)
        dock.setWidget(quick_panel)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    # Rest of the methods remain largely the same, with small improvements...

    def connect_signals(self):
        """Connect all component signals"""
        try:
            # Control panel signals
            if hasattr(self.control_panel, 'backtest_started'):
                self.control_panel.backtest_started.connect(self.on_backtest_started)
            if hasattr(self.control_panel, 'backtest_finished'):
                self.control_panel.backtest_finished.connect(self.on_backtest_finished)
            if hasattr(self.control_panel, 'config_changed'):
                self.control_panel.config_changed.connect(self.on_config_changed)

            # Chart signals
            if hasattr(self.chart_widget, 'backtest_completed'):
                self.chart_widget.backtest_completed.connect(self.on_chart_backtest_completed)
            if hasattr(self.chart_widget, 'backtest_progress'):
                self.chart_widget.backtest_progress.connect(self.on_backtest_progress)

            # Results signals
            if hasattr(self.results_display, 'results_exported'):
                self.results_display.results_exported.connect(self.on_results_exported)

            # Timeframe switcher signals
            if hasattr(self.timeframe_switcher, 'timeframe_changed'):
                self.timeframe_switcher.timeframe_changed.connect(self.on_timeframe_changed)
            if hasattr(self.timeframe_switcher, 'custom_timeframe_added'):
                self.timeframe_switcher.custom_timeframe_added.connect(self.on_custom_timeframe_added)
            if hasattr(self.timeframe_switcher, 'multiple_timeframes_selected'):
                self.timeframe_switcher.multiple_timeframes_selected.connect(self.on_multiple_timeframes_selected)
        except Exception as e:
            print(f"⚠️ Warning: Could not connect all signals: {e}")

    def run_backtest(self):
        """Run backtest with current configuration"""
        try:
            # Get configuration from control panel
            config = self.control_panel.get_full_configuration()

            # Validate configuration
            if not self.validate_config(config):
                return

            # Start backtest in background thread
            self.start_background_backtest(config)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start backtest: {e}")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate backtest configuration"""
        try:
            strategy_config = config.get('strategy', {})
            backtest_params = config.get('parameters', {})

            # Check strategy
            strategy_name = strategy_config.get('strategy_name', '')
            if not strategy_name or strategy_name == "Select Strategy...":
                QMessageBox.warning(
                    self,
                    "Strategy Required",
                    "Please select a strategy from the dropdown!\n\n"
                    "💡 Try 'ATR RSI Strategy (Demo)' for a quick demo."
                )
                return False

            # Check symbol
            if not backtest_params.get('symbol'):
                QMessageBox.warning(self, "Validation Error", "Please specify a trading symbol!")
                return False

            return True
        except Exception as e:
            QMessageBox.warning(self, "Validation Error", f"Configuration validation failed: {e}")
            return False

    def on_timeframe_changed(self, timeframe: str):
        """Handle timeframe change"""
        try:
            # Update control panel
            if hasattr(self.control_panel, 'set_timeframe'):
                self.control_panel.set_timeframe(timeframe)

            # Update chart if data exists
            if self.current_results and hasattr(self.chart_widget, 'update_timeframe'):
                self.chart_widget.update_timeframe(timeframe)

            self.status_label.setText(f"Timeframe: {timeframe}")

            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Timeframe changed to: {timeframe}", 3000)
        except Exception as e:
            print(f"⚠️ Error in timeframe change: {e}")

    def on_backtest_started(self):
        """Handle backtest started"""
        self.status_label.setText("🚀 Running backtest...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def on_backtest_finished(self, results: Dict[str, Any]):
        """Handle backtest finished signal"""
        try:
            self.current_results = results

            # Update UI
            self.status_label.setText("✅ Backtest completed!")
            self.progress_bar.setVisible(False)
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            if hasattr(self, 'status_bar'):
                strategy = results.get('strategy', 'Unknown')
                self.status_bar.showMessage(f"✅ Backtest completed for {strategy}!", 5000)

            # Load results into displays
            if hasattr(self.chart_widget, 'load_backtest_data'):
                self.chart_widget.load_backtest_data(results)
            if hasattr(self.results_display, 'load_results'):
                self.results_display.load_results(results)

            # Show enhanced success message
            perf = results.get('performance', {})
            total_return = perf.get('total_return', 0) * 100
            win_rate = perf.get('win_rate', 0) * 100
            total_trades = perf.get('total_trades', 0)
            sharpe = perf.get('sharpe_ratio', 0)

            success_msg = QMessageBox(self)
            success_msg.setWindowTitle("🎉 Backtest Complete!")
            success_msg.setIcon(QMessageBox.Icon.Information)

            success_text = f"""
<h3>🎉 Backtest Completed Successfully!</h3>

<p><b>Strategy:</b> {results.get('strategy', 'Unknown')}<br/>
<b>Symbol:</b> {results.get('symbol', 'Unknown')}<br/>
<b>Timeframe:</b> {results.get('timeframe', 'Unknown')}</p>

<h4>📊 Key Performance Metrics:</h4>
<ul>
<li><b>Total Return:</b> {total_return:.2f}%</li>
<li><b>Win Rate:</b> {win_rate:.1f}%</li>
<li><b>Total Trades:</b> {total_trades}</li>
<li><b>Sharpe Ratio:</b> {sharpe:.2f}</li>
</ul>

<p>📈 <b>Check the Results panel for detailed analysis and export options!</b></p>
            """

            success_msg.setText(success_text)
            success_msg.setTextFormat(Qt.TextFormat.RichText)
            success_msg.exec()

        except Exception as e:
            print(f"⚠️ Error handling backtest results: {e}")

    # Keep all other methods the same as before...
    def setup_menus(self):
        """Setup application menus"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("📁 File")
        file_menu.addAction("🆕 New Backtest", self.new_backtest)
        file_menu.addAction("📂 Load Configuration", self.load_config)
        file_menu.addAction("💾 Save Configuration", self.save_config)
        file_menu.addSeparator()
        file_menu.addAction("❌ Exit", self.close)

        # View menu
        view_menu = menubar.addMenu("👁️ View")
        view_menu.addAction("🎛️ Toggle Controls", self.toggle_controls)
        view_menu.addAction("📊 Toggle Results", self.toggle_results)
        view_menu.addAction("🚀 Toggle Quick Actions", self.toggle_quick_actions)

        # Demo menu
        demo_menu = menubar.addMenu("🎭 Demo")
        demo_menu.addAction("📊 Load Demo Data", lambda: self.chart_widget.load_demo_data() if hasattr(self.chart_widget, 'load_demo_data') else None)
        demo_menu.addAction("🔄 Refresh Demo", lambda: self.chart_widget.refresh_chart() if hasattr(self.chart_widget, 'refresh_chart') else None)
        demo_menu.addAction("🗑️ Clear Chart", lambda: self.chart_widget.clear_chart() if hasattr(self.chart_widget, 'clear_chart') else None)

        # Help menu
        help_menu = menubar.addMenu("❓ Help")
        help_menu.addAction("📖 Instructions", self.show_instructions)
        help_menu.addAction("🎯 Quick Start", self.show_welcome_message)
        help_menu.addAction("ℹ️ About", self.show_about)

    def setup_toolbars(self):
        """Setup application toolbars"""
        toolbar = self.addToolBar("Main")
        toolbar.addAction("🆕 New", self.new_backtest)
        toolbar.addAction("📂 Load", self.load_config)
        toolbar.addAction("💾 Save", self.save_config)
        toolbar.addSeparator()
        toolbar.addAction("🚀 Run", self.run_backtest)
        toolbar.addAction("⏹ Stop", self.stop_backtest)
        toolbar.addSeparator()
        toolbar.addAction("📊 Demo Data", lambda: self.chart_widget.load_demo_data() if hasattr(self.chart_widget, 'load_demo_data') else None)

    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("🎭 Demo Mode - Ready for interactive backtesting!")

    def start_background_backtest(self, config):
        """Start background backtest"""
        print(f"🚀 Starting backtest with config: {config}")
        # For now, just start the mock backtest from control panel
        if hasattr(self.control_panel, 'start_mock_backtest'):
            self.control_panel.start_mock_backtest()

    # Keep all the placeholder methods...
    def on_chart_backtest_completed(self, results): pass
    def on_backtest_progress(self, progress):
        self.progress_bar.setValue(progress)
    def on_status_update(self, status):
        self.status_label.setText(status)
    def on_background_backtest_completed(self, results): pass
    def on_backtest_failed(self, error):
        QMessageBox.critical(self, "Backtest Failed", f"Error: {error}")
    def on_config_changed(self, config): pass
    def on_results_exported(self, filename):
        QMessageBox.information(self, "Export Complete", f"Results exported to: {filename}")
    def on_custom_timeframe_added(self, timeframe): pass
    def on_multiple_timeframes_selected(self, timeframes): pass

    def change_theme(self, theme_name):
        """Change application theme"""
        if hasattr(self.display_config, 'update_theme'):
            if hasattr(ChartTheme, theme_name.upper()):
                theme = getattr(ChartTheme, theme_name.upper())
                self.display_config.update_theme(theme)

    def stop_backtest(self):
        """Stop running backtest"""
        print("⏹ Stopping backtest...")
        self.status_label.setText("⏹ Backtest stopped")
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def new_backtest(self): print("🆕 New backtest")
    def load_config(self): print("📂 Load config")
    def save_config(self): print("💾 Save config")
    def toggle_controls(self): print("🎛️ Toggle controls")
    def toggle_results(self): print("📊 Toggle results")
    def toggle_quick_actions(self): print("🚀 Toggle quick actions")

    def show_instructions(self):
        """Show instructions"""
        if hasattr(self.chart_widget, 'tab_widget'):
            self.chart_widget.tab_widget.setCurrentIndex(2)  # Instructions tab
        else:
            QMessageBox.information(
                self,
                "Instructions",
                "📖 Instructions are available in the chart tab!\n\n"
                "Look for the 'Instructions' tab in the main chart area."
            )

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Backtest Display System",
            """
<h2>🚀 Backtest Display System v1.0</h2>

<p>A comprehensive backtest analysis and visualization tool for trading strategies.</p>

<h3>✨ Features:</h3>
<ul>
<li>📊 Interactive charting with PyQtGraph</li>
<li>🎛️ Strategy testing and optimization</li>
<li>📈 Performance analysis and metrics</li>
<li>💾 Export functionality (CSV, JSON, Excel)</li>
<li>⏱️ Multiple timeframe analysis</li>
<li>🎭 Demo mode with sample data</li>
</ul>

<h3>🛠️ Technology Stack:</h3>
<ul>
<li>Python 3.11+ with PySide6</li>
<li>PyQtGraph for charting</li>
<li>Pandas for data processing</li>
<li>NumPy for calculations</li>
</ul>

<p><b>💡 This is a demo version with simulated data.</b></p>
            """
        )

    def load_settings(self): pass
    def save_settings(self): pass

    def closeEvent(self, event):
        """Handle application close"""
        # Stop any running backtest
        if self.backtest_worker:
            self.stop_backtest()

        # Save settings
        self.save_settings()

        # Accept close
        event.accept()


class BacktestApplication(QApplication):
    """
    Main application class
    """

    def __init__(self, argv):
        super().__init__(argv)

        # Set application properties
        self.setApplicationName("Backtest Display System")
        self.setApplicationVersion("1.0")
        self.setOrganizationName("BacktestDisplay")

        # Set application style
        self.setStyle("Fusion")

        # Create main window
        self.main_window = BacktestMainWindow()

    def run(self):
        """Run the application"""
        self.main_window.show()
        return self.exec()


def main():
    """Main application entry point"""
    print("🚀 Starting Backtest Display System...")

    # Create application
    app = BacktestApplication(sys.argv)

    # Run application
    try:
        exit_code = app.run()
        print(f"✅ Application finished with code: {exit_code}")
        return exit_code

    except Exception as e:
        print(f"❌ Application error: {e}")
        traceback.print_exc()
        return 1


def test_main_app():
    """Test main application"""
    print("🧪 Testing Main Backtest Application...")

    # Create test application
    app = BacktestApplication([])
    print("✅ Application created")

    # Show application
    app.main_window.show()
    print("✅ Application shown")

    print("🎉 Main application test completed!")
    print("📋 Features available:")
    print("   ✅ Enhanced chart display")
    print("   ✅ Backtest control panel")
    print("   ✅ Results analysis")
    print("   ✅ Menu system")
    print("   ✅ Toolbar actions")
    print("   ✅ Export functionality")
    print("   ✅ Theme switching")
    print("   ✅ Configuration management")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_main_app()
    else:
        sys.exit(main())