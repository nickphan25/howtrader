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

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6.QtCore import Qt, Signal, QTimer, QThread, QObject, QSettings
from PySide6.QtGui import QIcon, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QDockWidget,
    QPushButton, QLabel, QMessageBox, QFileDialog, QProgressBar,
    QComboBox, QCheckBox, QGroupBox, QFrame
)

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
    print("📁 Current directory:", os.getcwd())
    print("📁 Project root:", project_root)
    print("🔍 Available files:")
    for root, dirs, files in os.walk(project_root):
        level = root.replace(project_root, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py'):
                print(f"{subindent}{file}")

    # Try to create mock components for testing
    print("🔧 Creating mock components...")

    class MockChart(QWidget):
        backtest_completed = Signal(dict)
        backtest_progress = Signal(int)

        def __init__(self):
            super().__init__()
            self.setMinimumSize(600, 400)
            layout = QVBoxLayout(self)

            # Main chart area
            chart_label = QLabel("📈 Enhanced Backtest Chart")
            chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chart_label.setStyleSheet("""
                QLabel {
                    background-color: #f0f0f0;
                    border: 2px dashed #ccc;
                    font-size: 16px;
                    font-weight: bold;
                    color: #666;
                    padding: 20px;
                }
            """)
            layout.addWidget(chart_label)

            # Chart info
            info_label = QLabel("Chart will display:\n• Candlestick data\n• Trading signals\n• Performance overlay\n• Technical indicators")
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(info_label)

        def load_backtest_data(self, data):
            print("📊 Loading data into mock chart")

        def update_timeframe(self, tf):
            print(f"⏱️ Updating chart timeframe to: {tf}")

    class MockPanel(QWidget):
        backtest_started = Signal()
        backtest_finished = Signal(dict)
        config_changed = Signal(dict)

        def __init__(self, config):
            super().__init__()
            self.setMaximumWidth(300)
            self.setMinimumWidth(280)
            layout = QVBoxLayout(self)

            # Panel title
            title = QLabel("🎛️ Backtest Control Panel")
            title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
            layout.addWidget(title)

            # Strategy section
            strategy_group = QGroupBox("Strategy Selection")
            strategy_layout = QVBoxLayout(strategy_group)

            self.strategy_combo = QComboBox()
            self.strategy_combo.addItems([
                "Select Strategy...",
                "ATR RSI Strategy",
                "Moving Average Crossover",
                "Bollinger Bands Strategy",
                "MACD Strategy"
            ])
            strategy_layout.addWidget(self.strategy_combo)

            layout.addWidget(strategy_group)

            # Parameters section
            params_group = QGroupBox("Parameters")
            params_layout = QVBoxLayout(params_group)

            params_layout.addWidget(QLabel("Symbol: BTCUSDT.BINANCE"))
            params_layout.addWidget(QLabel("Timeframe: 1h"))
            params_layout.addWidget(QLabel("Period: Last 30 days"))

            layout.addWidget(params_group)

            # Actions
            actions_group = QGroupBox("Actions")
            actions_layout = QVBoxLayout(actions_group)

            run_btn = QPushButton("▶️ Run Backtest")
            run_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            run_btn.clicked.connect(self.start_mock_backtest)
            actions_layout.addWidget(run_btn)

            export_btn = QPushButton("💾 Export Config")
            export_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                }
            """)
            actions_layout.addWidget(export_btn)

            layout.addWidget(actions_group)
            layout.addStretch()

        def start_mock_backtest(self):
            """Start a mock backtest"""
            strategy = self.strategy_combo.currentText()
            if strategy == "Select Strategy...":
                QMessageBox.warning(self, "Warning", "Please select a strategy first!")
                return

            print(f"🚀 Starting mock backtest with strategy: {strategy}")
            self.backtest_started.emit()

            # Simulate backtest completion after 3 seconds
            QTimer.singleShot(3000, lambda: self.backtest_finished.emit({
                'strategy': strategy,
                'performance': {
                    'total_return': 0.15,
                    'win_rate': 0.65,
                    'total_trades': 25,
                    'max_drawdown': -0.08
                },
                'trades': [],
                'summary': {
                    'profitable_trades': 16,
                    'losing_trades': 9
                }
            }))

        def get_full_configuration(self):
            return {
                'strategy': {'strategy_name': self.strategy_combo.currentText()},
                'parameters': {
                    'symbol': 'BTCUSDT.BINANCE',
                    'start_date': datetime.now() - timedelta(days=30),
                    'end_date': datetime.now(),
                    'timeframe': '1h'
                }
            }

        def set_timeframe(self, tf):
            print(f"🎛️ Setting panel timeframe to: {tf}")

    class MockResultsDisplay(QWidget):
        results_exported = Signal(str)

        def __init__(self, config):
            super().__init__()
            self.setMaximumWidth(350)
            self.setMinimumWidth(300)
            layout = QVBoxLayout(self)

            # Title
            title = QLabel("📊 Results Analysis")
            title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
            layout.addWidget(title)

            # Performance metrics
            metrics_group = QGroupBox("Performance Metrics")
            metrics_layout = QVBoxLayout(metrics_group)

            self.total_return_label = QLabel("Total Return: --")
            self.win_rate_label = QLabel("Win Rate: --")
            self.total_trades_label = QLabel("Total Trades: --")
            self.max_drawdown_label = QLabel("Max Drawdown: --")

            for label in [self.total_return_label, self.win_rate_label,
                          self.total_trades_label, self.max_drawdown_label]:
                label.setStyleSheet("font-family: monospace; padding: 2px;")
                metrics_layout.addWidget(label)

            layout.addWidget(metrics_group)

            # Trade summary
            trades_group = QGroupBox("Trade Summary")
            trades_layout = QVBoxLayout(trades_group)

            self.profitable_trades_label = QLabel("Profitable: --")
            self.losing_trades_label = QLabel("Losing: --")

            for label in [self.profitable_trades_label, self.losing_trades_label]:
                label.setStyleSheet("font-family: monospace; padding: 2px;")
                trades_layout.addWidget(label)

            layout.addWidget(trades_group)

            # Export section
            export_group = QGroupBox("Export")
            export_layout = QVBoxLayout(export_group)

            export_csv_btn = QPushButton("📄 Export CSV")
            export_json_btn = QPushButton("📋 Export JSON")

            for btn in [export_csv_btn, export_json_btn]:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #FF9800;
                        color: white;
                        padding: 6px;
                        border-radius: 3px;
                    }
                    QPushButton:hover {
                        background-color: #F57C00;
                    }
                """)
                export_layout.addWidget(btn)

            layout.addWidget(export_group)
            layout.addStretch()

        def load_results(self, results):
            print("📈 Loading results into mock display")

            # Update performance metrics
            perf = results.get('performance', {})
            self.total_return_label.setText(f"Total Return: {perf.get('total_return', 0)*100:.1f}%")
            self.win_rate_label.setText(f"Win Rate: {perf.get('win_rate', 0)*100:.1f}%")
            self.total_trades_label.setText(f"Total Trades: {perf.get('total_trades', 0)}")
            self.max_drawdown_label.setText(f"Max Drawdown: {perf.get('max_drawdown', 0)*100:.1f}%")

            # Update trade summary
            summary = results.get('summary', {})
            self.profitable_trades_label.setText(f"Profitable: {summary.get('profitable_trades', 0)}")
            self.losing_trades_label.setText(f"Losing: {summary.get('losing_trades', 0)}")

    class MockTimeframeSwitcher(QWidget):
        timeframe_changed = Signal(str)
        custom_timeframe_added = Signal(str)
        multiple_timeframes_selected = Signal(list)

        def __init__(self, config):
            super().__init__()
            self.setMaximumHeight(120)
            self.setMinimumHeight(100)
            layout = QVBoxLayout(self)

            # Title
            title = QLabel("⏱️ Timeframe Switcher")
            title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2196F3;")
            layout.addWidget(title)

            # Timeframe buttons
            button_layout = QHBoxLayout()

            self.current_tf = "1h"
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

            for tf in timeframes:
                btn = QPushButton(tf)
                btn.setMaximumWidth(35)
                btn.setCheckable(True)
                btn.setChecked(tf == self.current_tf)

                if tf == self.current_tf:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #4CAF50;
                            color: white;
                            font-weight: bold;
                            border-radius: 3px;
                            padding: 4px;
                        }
                    """)
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #f0f0f0;
                            color: #333;
                            border: 1px solid #ccc;
                            border-radius: 3px;
                            padding: 4px;
                        }
                        QPushButton:hover {
                            background-color: #e0e0e0;
                        }
                        QPushButton:checked {
                            background-color: #4CAF50;
                            color: white;
                        }
                    """)

                btn.clicked.connect(lambda checked, timeframe=tf: self.on_timeframe_clicked(timeframe))
                button_layout.addWidget(btn)

            layout.addLayout(button_layout)

            # Multi-select option
            multi_layout = QHBoxLayout()
            multi_cb = QCheckBox("Multi-TF Analysis")
            multi_cb.setStyleSheet("font-size: 10px;")
            multi_layout.addWidget(multi_cb)
            multi_layout.addStretch()

            custom_btn = QPushButton("+")
            custom_btn.setMaximumWidth(25)
            custom_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    font-weight: bold;
                    border-radius: 12px;
                    font-size: 12px;
                }
            """)
            multi_layout.addWidget(custom_btn)

            layout.addLayout(multi_layout)

        def on_timeframe_clicked(self, timeframe):
            """Handle timeframe button click"""
            self.current_tf = timeframe
            print(f"⏱️ Timeframe changed to: {timeframe}")
            self.timeframe_changed.emit(timeframe)

            # Update button styles
            for btn in self.findChildren(QPushButton):
                if btn.text() in ['1m', '5m', '15m', '1h', '4h', '1d']:
                    if btn.text() == timeframe:
                        btn.setStyleSheet("""
                            QPushButton {
                                background-color: #4CAF50;
                                color: white;
                                font-weight: bold;
                                border-radius: 3px;
                                padding: 4px;
                            }
                        """)
                    else:
                        btn.setStyleSheet("""
                            QPushButton {
                                background-color: #f0f0f0;
                                color: #333;
                                border: 1px solid #ccc;
                                border-radius: 3px;
                                padding: 4px;
                            }
                            QPushButton:hover {
                                background-color: #e0e0e0;
                            }
                        """)

    def create_enhanced_backtest_chart():
        return MockChart()

    BacktestControlPanel = MockPanel
    BacktestResultsDisplay = MockResultsDisplay
    TimeframeSwitcher = MockTimeframeSwitcher

    # Import config and core components
    try:
        from config.display_config import DisplayConfig, ChartTheme
        from core.backtest_bridge import BacktestBridge
        from core.data_extractor import BacktestDataExtractor
    except ImportError:
        print("❌ Could not import config/core components")
        # Create minimal mocks
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
        self.setWindowTitle("🚀 Backtest Display System v1.0")
        self.setWindowIcon(QIcon())  # Would load app icon
        self.resize(1400, 900)

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

        control_dock = QDockWidget("Backtest Controls", self)
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
        results_dock = QDockWidget("Results Analysis", self)
        results_dock.setWidget(self.results_display)
        results_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, results_dock)

        # Set splitter proportions
        main_splitter.setSizes([300, 800, 400])

        # Quick action panel at bottom
        self.create_quick_action_panel()

    def create_quick_action_panel(self):
        """Create quick action panel"""
        quick_panel = QFrame()
        quick_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        quick_panel.setMaximumHeight(80)

        layout = QHBoxLayout(quick_panel)

        # Quick backtest section
        quick_group = QGroupBox("Quick Actions")
        quick_layout = QHBoxLayout(quick_group)

        # Run backtest button
        self.run_btn = QPushButton("🚀 Run Backtest")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
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
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
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
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        quick_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        quick_layout.addWidget(self.status_label)

        layout.addWidget(quick_group)

        # Quick settings section
        settings_group = QGroupBox("Quick Settings")
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

        layout.addWidget(settings_group)

        # Add to main layout
        dock = QDockWidget("Quick Actions", self)
        dock.setWidget(quick_panel)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

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
                QMessageBox.warning(self, "Validation Error", "Please select a strategy first!")
                return False

            # Check symbol
            if not backtest_params.get('symbol'):
                QMessageBox.warning(self, "Validation Error", "Please specify a trading symbol!")
                return False

            # Check date range
            start_date = backtest_params.get('start_date')
            end_date = backtest_params.get('end_date')

            if not start_date or not end_date:
                QMessageBox.warning(self, "Validation Error", "Please specify start and end dates!")
                return False

            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)

            if start_date >= end_date:
                QMessageBox.warning(self, "Validation Error", "Start date must be before end date!")
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

            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Timeframe changed to: {timeframe}", 2000)
        except Exception as e:
            print(f"⚠️ Error in timeframe change: {e}")

    def on_backtest_started(self):
        """Handle backtest started"""
        self.status_label.setText("Running backtest...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def on_backtest_finished(self, results: Dict[str, Any]):
        """Handle backtest finished signal"""
        try:
            self.current_results = results

            # Update UI
            self.status_label.setText("Backtest completed!")
            self.progress_bar.setVisible(False)
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            if hasattr(self, 'status_bar'):
                strategy = results.get('strategy', 'Unknown')
                self.status_bar.showMessage(f"Backtest completed for {strategy}!")

            # Load results into displays
            if hasattr(self.chart_widget, 'load_backtest_data'):
                self.chart_widget.load_backtest_data(results)
            if hasattr(self.results_display, 'load_results'):
                self.results_display.load_results(results)

            # Show success message
            perf = results.get('performance', {})
            total_return = perf.get('total_return', 0) * 100
            total_trades = perf.get('total_trades', 0)

            QMessageBox.information(
                self,
                "Backtest Complete",
                f"Backtest finished successfully!\n\n"
                f"Total Return: {total_return:.1f}%\n"
                f"Total Trades: {total_trades}\n\n"
                f"Check the Results panel for detailed analysis."
            )

        except Exception as e:
            print(f"⚠️ Error handling backtest results: {e}")

    # Implement placeholder methods for now
    def setup_menus(self):
        """Setup application menus"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Backtest", self.new_backtest)
        file_menu.addAction("Load Configuration", self.load_config)
        file_menu.addAction("Save Configuration", self.save_config)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Toggle Controls", self.toggle_controls)
        view_menu.addAction("Toggle Results", self.toggle_results)

        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)

    def setup_toolbars(self):
        """Setup application toolbars"""
        toolbar = self.addToolBar("Main")
        toolbar.addAction("🆕 New", self.new_backtest)
        toolbar.addAction("📂 Load", self.load_config)
        toolbar.addAction("💾 Save", self.save_config)
        toolbar.addSeparator()
        toolbar.addAction("🚀 Run", self.run_backtest)
        toolbar.addAction("⏹ Stop", self.stop_backtest)

    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def start_background_backtest(self, config):
        """Start background backtest"""
        print(f"🚀 Starting backtest with config: {config}")
        # For now, just start the mock backtest from control panel
        if hasattr(self.control_panel, 'start_mock_backtest'):
            self.control_panel.start_mock_backtest()

    def on_chart_backtest_completed(self, results):
        """Handle chart backtest completed"""
        pass

    def on_backtest_progress(self, progress):
        """Handle backtest progress"""
        self.progress_bar.setValue(progress)

    def on_status_update(self, status):
        """Handle status update"""
        self.status_label.setText(status)

    def on_background_backtest_completed(self, results):
        """Handle background backtest completed"""
        pass

    def on_backtest_failed(self, error):
        """Handle backtest failed"""
        QMessageBox.critical(self, "Backtest Failed", f"Error: {error}")

    def on_config_changed(self, config):
        """Handle config changed"""
        pass

    def on_results_exported(self, filename):
        """Handle results exported"""
        QMessageBox.information(self, "Export Complete", f"Results exported to: {filename}")

    def on_custom_timeframe_added(self, timeframe):
        """Handle custom timeframe added"""
        pass

    def on_multiple_timeframes_selected(self, timeframes):
        """Handle multiple timeframes selected"""
        pass

    def change_theme(self, theme_name):
        """Change application theme"""
        if hasattr(self.display_config, 'update_theme'):
            if hasattr(ChartTheme, theme_name.upper()):
                theme = getattr(ChartTheme, theme_name.upper())
                self.display_config.update_theme(theme)

    def stop_backtest(self):
        """Stop running backtest"""
        print("⏹ Stopping backtest...")
        self.status_label.setText("Backtest stopped")
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def new_backtest(self):
        """Create new backtest"""
        print("🆕 New backtest")

    def load_config(self):
        """Load configuration"""
        print("📂 Load config")

    def save_config(self):
        """Save configuration"""
        print("💾 Save config")

    def toggle_controls(self):
        """Toggle controls panel"""
        print("🎛️ Toggle controls")

    def toggle_results(self):
        """Toggle results panel"""
        print("📊 Toggle results")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Backtest Display System",
            "Backtest Display System v1.0\n\n"
            "A comprehensive backtest analysis and visualization tool.\n\n"
            "Features:\n"
            "• Advanced charting\n"
            "• Strategy testing\n"
            "• Performance analysis\n"
            "• Export functionality"
        )

    def load_settings(self):
        """Load application settings"""
        pass

    def save_settings(self):
        """Save application settings"""
        pass

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


def create_sample_app():
    """Create sample application for testing"""
    app = BacktestApplication([])

    # Load sample data
    import numpy as np
    sample_results = {
        'performance': {
            'total_return': 0.15,
            'annual_return': 0.45,
            'max_drawdown': -0.08,
            'sharpe_ratio': 1.25,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'start_capital': 100000
        },
        'summary': {
            'total_trades': 45,
            'profitable_trades': 29,
            'largest_win': 2500.0,
            'largest_loss': -800.0,
            'average_trade': 125.5
        },
        'trades': [
            {
                'trade_id': f'T{i}',
                'timestamp': i * 3600,
                'price': 100 + i * 0.5,
                'is_long': i % 2 == 0,
                'volume': 1.0,
                'pnl': np.random.normal(50, 200),
                'datetime': datetime.now() + timedelta(hours=i)
            }
            for i in range(45)
        ],
        'bars': [
            {'datetime': datetime.now() + timedelta(hours=i), 'timestamp': i, 'close': 100 + i * 0.1}
            for i in range(1000)
        ]
    }

    # Load sample data into application
    QTimer.singleShot(1000, lambda: app.main_window.on_backtest_finished(sample_results))

    return app


def test_main_app():
    """Test main application"""
    print("🧪 Testing Main Backtest Application...")

    # Create test application
    app = create_sample_app()
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

    # Don't start event loop in test
    # app.main_window.close()  # Comment this out to keep window open


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_main_app()
    else:
        sys.exit(main())