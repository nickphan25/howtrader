"""
Howtrader Backtest Display Application
====================================

Professional backtesting interface for Howtrader CTA strategies.
Enhanced with strategy categorization and detailed information.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QComboBox, QDateEdit,
                               QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget,
                               QTableWidget, QTableWidgetItem, QProgressBar,
                               QSplitter, QGroupBox, QFormLayout, QCheckBox,
                               QLineEdit, QFrame, QScrollArea, QMessageBox)
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QDate
from PySide6.QtGui import QFont, QIcon, QPalette, QColor

# Import our modules
try:
    from backtest_bridge import RealBacktestBridge

    print("✅ Loaded RealBacktestBridge")
except ImportError as e:
    print(f"⚠️ Failed to load RealBacktestBridge: {e}")
    sys.exit(1)

# Import UI modules with proper error handling
try:
    from ui import ResultDisplayWidget, ChartDisplayWidget

    print("✅ Loaded UI widgets from ui module")
except ImportError as e:
    print(f"⚠️ Failed to load UI widgets: {e}")



class BacktestWorker(QThread):
    """Background worker for running backtests"""

    # Signals
    progress_updated = Signal(int)
    status_updated = Signal(str)
    backtest_completed = Signal(dict)
    backtest_failed = Signal(str)

    def __init__(self, bridge: RealBacktestBridge):
        super().__init__()
        self.bridge = bridge
        self.config = {}
        self.is_cancelled = False

    def setup_backtest(self, config: Dict[str, Any]):
        """Setup backtest configuration"""
        self.config = config
        self.is_cancelled = False

    def cancel_backtest(self):
        """Cancel running backtest"""
        self.is_cancelled = True

    def run(self):
        """Run backtest in background thread"""
        try:
            self.is_cancelled = False
            self.status_updated.emit("Initializing backtest...")
            self.progress_updated.emit(0)

            # Extract configuration
            strategy_name = self.config.get('strategy_name', '')
            symbol = self.config.get('symbol', 'BTCUSDT.BINANCE')
            timeframe = self.config.get('timeframe', '1h')
            start_date = self.config.get('start_date', datetime.now() - timedelta(days=30))
            end_date = self.config.get('end_date', datetime.now())
            capital = self.config.get('capital', 100000)
            commission = self.config.get('commission', 0.001)
            strategy_params = self.config.get('strategy_params', {})

            if self.is_cancelled:
                return

            self.status_updated.emit("Running backtest...")
            self.progress_updated.emit(20)

            # Set progress callback
            self.bridge.set_progress_callback(self._on_progress_update)

            # Run backtest
            results = self.bridge.run_backtest(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                capital=capital,
                commission=commission,
                **strategy_params
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
            error_msg = f"Backtest failed: {str(e)}"
            self.backtest_failed.emit(error_msg)

    def _on_progress_update(self, message: str):
        """Handle progress updates from bridge"""
        self.status_updated.emit(message)


class StrategySelectionWidget(QWidget):
    """Enhanced strategy selection widget with categorization"""

    strategy_changed = Signal(str)

    def __init__(self, bridge: RealBacktestBridge):
        super().__init__()
        self.bridge = bridge
        self.strategy_mapping = {}
        self.init_ui()
        self.load_strategies()

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)

        # Strategy selection group
        strategy_group = QGroupBox("📊 Strategy Selection")
        strategy_layout = QVBoxLayout(strategy_group)

        # Category filter
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Category:"))

        self.category_combo = QComboBox()
        self.category_combo.addItems([
            "All Strategies",
            "📚 Demo/Educational",
            "🏭 Production",
            "🧪 Mock/Testing"
        ])
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        category_layout.addWidget(self.category_combo)

        category_layout.addWidget(QLabel("Complexity:"))
        self.complexity_combo = QComboBox()
        self.complexity_combo.addItems([
            "All Levels",
            "Beginner",
            "Intermediate",
            "Advanced"
        ])
        self.complexity_combo.currentTextChanged.connect(self.on_complexity_changed)
        category_layout.addWidget(self.complexity_combo)

        strategy_layout.addLayout(category_layout)

        # Strategy selection
        strategy_select_layout = QHBoxLayout()
        strategy_select_layout.addWidget(QLabel("Strategy:"))

        self.strategy_combo = QComboBox()
        self.strategy_combo.setMinimumWidth(300)
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_select_layout.addWidget(self.strategy_combo)

        strategy_layout.addLayout(strategy_select_layout)

        # Strategy information display
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ccc;")
        strategy_layout.addWidget(QLabel("Strategy Information:"))
        strategy_layout.addWidget(self.info_text)

        layout.addWidget(strategy_group)

    def load_strategies(self):
        """Load strategies from bridge"""
        try:
            # Get all strategies with display names
            display_names = self.bridge.get_strategy_display_names()

            # Store mapping
            self.strategy_mapping = {}

            # Clear and populate combo
            self.strategy_combo.clear()
            for actual_name, display_name in display_names:
                self.strategy_combo.addItem(display_name)
                self.strategy_mapping[display_name] = actual_name

            print(f"✅ Loaded {len(display_names)} strategies into UI")

            # Update info for first strategy
            if display_names:
                self.update_strategy_info(display_names[0][0])

        except Exception as e:
            print(f"⚠️ Error loading strategies: {e}")
            # Add fallback mock strategies
            self.strategy_combo.addItems([
                "🧪 MockMAStrategy [Beginner]",
                "🧪 MockRSIStrategy [Beginner]",
                "🧪 MockBollingerStrategy [Intermediate]"
            ])

    def on_category_changed(self, category_text: str):
        """Handle category filter change"""
        try:
            # Map display text to internal category
            category_map = {
                "All Strategies": None,
                "📚 Demo/Educational": "demo",
                "🏭 Production": "production",
                "🧪 Mock/Testing": "mock"
            }

            selected_category = category_map.get(category_text)
            self.filter_strategies(category=selected_category)

        except Exception as e:
            print(f"⚠️ Error filtering by category: {e}")

    def on_complexity_changed(self, complexity_text: str):
        """Handle complexity filter change"""
        try:
            complexity = complexity_text.lower() if complexity_text != "All Levels" else None
            self.filter_strategies(complexity=complexity)

        except Exception as e:
            print(f"⚠️ Error filtering by complexity: {e}")

    def filter_strategies(self, category: str = None, complexity: str = None):
        """Filter strategies by category and complexity"""
        try:
            current_category = self.category_combo.currentText()
            current_complexity = self.complexity_combo.currentText()

            # Map category
            category_map = {
                "All Strategies": None,
                "📚 Demo/Educational": "demo",
                "🏭 Production": "production",
                "🧪 Mock/Testing": "mock"
            }
            filter_category = category_map.get(current_category)

            # Map complexity
            filter_complexity = current_complexity.lower() if current_complexity != "All Levels" else None

            # Get filtered strategies
            filtered_strategies = []

            for name, info in self.bridge.strategy_infos.items():
                # Check category filter
                if filter_category and info.category != filter_category:
                    continue

                # Check complexity filter
                if filter_complexity and info.complexity != filter_complexity:
                    continue

                filtered_strategies.append((name, info.get_display_name()))

            # Update combo box
            self.strategy_combo.clear()
            self.strategy_mapping = {}

            for actual_name, display_name in filtered_strategies:
                self.strategy_combo.addItem(display_name)
                self.strategy_mapping[display_name] = actual_name

            # Update info if strategies available
            if filtered_strategies:
                self.update_strategy_info(filtered_strategies[0][0])

        except Exception as e:
            print(f"⚠️ Error filtering strategies: {e}")

    def on_strategy_changed(self, display_name: str):
        """Handle strategy selection change"""
        try:
            if display_name and display_name in self.strategy_mapping:
                actual_name = self.strategy_mapping[display_name]
                self.update_strategy_info(actual_name)
                self.strategy_changed.emit(actual_name)

        except Exception as e:
            print(f"⚠️ Error handling strategy change: {e}")

    def update_strategy_info(self, strategy_name: str):
        """Update strategy information display"""
        try:
            info = self.bridge.get_strategy_info(strategy_name)
            if info:
                info_text = f"""
<b>{info.name}</b><br>
<b>Category:</b> {info.category.title()}<br>
<b>Complexity:</b> {info.complexity.title()}<br>
<b>Description:</b> {info.description}<br>
<b>Features:</b><br>
{'<br>'.join([f"• {feature}" for feature in info.features[:5]])}
                """.strip()

                self.info_text.setHtml(info_text)
            else:
                self.info_text.setPlainText(f"No information available for {strategy_name}")

        except Exception as e:
            print(f"⚠️ Error updating strategy info: {e}")
            self.info_text.setPlainText("Error loading strategy information")

    def get_selected_strategy(self) -> str:
        """Get currently selected strategy name"""
        display_name = self.strategy_combo.currentText()
        return self.strategy_mapping.get(display_name, "")


class MainWindow(QMainWindow):
    """Enhanced main window with categorized strategy support"""

    def __init__(self):
        super().__init__()

        # Initialize bridge
        print("🔧 Initializing Howtrader Bridge...")
        self.bridge = RealBacktestBridge()

        # Initialize worker thread
        self.worker = BacktestWorker(self.bridge)
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.status_updated.connect(self.on_status_updated)
        self.worker.backtest_completed.connect(self.on_backtest_completed)
        self.worker.backtest_failed.connect(self.on_backtest_failed)

        self.init_ui()
        self.setup_style()

        # Show startup message
        self.show_startup_info()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🚀 Howtrader Professional Backtest Platform v2.0")
        self.setMinimumSize(1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Results
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([420, 980])

        main_layout.addWidget(splitter)

    def create_left_panel(self) -> QWidget:
        """Create left control panel"""
        panel = QWidget()
        panel.setMaximumWidth(450)
        panel.setMinimumWidth(400)

        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("🎯 Backtest Configuration")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Strategy selection (enhanced)
        self.strategy_widget = StrategySelectionWidget(self.bridge)
        self.strategy_widget.strategy_changed.connect(self.on_strategy_changed)
        layout.addWidget(self.strategy_widget)

        # Market parameters
        market_group = QGroupBox("📈 Market Parameters")
        market_layout = QFormLayout(market_group)

        self.symbol_input = QLineEdit("BTCUSDT.BINANCE")
        market_layout.addRow("Symbol:", self.symbol_input)

        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        self.timeframe_combo.setCurrentText("1h")
        market_layout.addRow("Timeframe:", self.timeframe_combo)

        # Date range
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-30))
        self.start_date.setCalendarPopup(True)
        market_layout.addRow("Start Date:", self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        market_layout.addRow("End Date:", self.end_date)

        layout.addWidget(market_group)

        # Trading parameters
        trading_group = QGroupBox("💰 Trading Parameters")
        trading_layout = QFormLayout(trading_group)

        self.capital_input = QSpinBox()
        self.capital_input.setRange(1000, 10000000)
        self.capital_input.setValue(100000)
        self.capital_input.setSuffix(" USD")
        trading_layout.addRow("Initial Capital:", self.capital_input)

        self.commission_input = QDoubleSpinBox()
        self.commission_input.setRange(0.0, 0.1)
        self.commission_input.setValue(0.001)
        self.commission_input.setDecimals(4)
        self.commission_input.setSuffix("%")
        trading_layout.addRow("Commission:", self.commission_input)

        self.slippage_input = QDoubleSpinBox()
        self.slippage_input.setRange(0.0, 0.1)
        self.slippage_input.setValue(0.0)
        self.slippage_input.setDecimals(4)
        self.slippage_input.setSuffix("%")
        trading_layout.addRow("Slippage:", self.slippage_input)

        layout.addWidget(trading_group)

        # Strategy parameters (dynamic)
        self.strategy_params_group = QGroupBox("⚙️ Strategy Parameters")
        self.strategy_params_layout = QVBoxLayout(self.strategy_params_group)
        layout.addWidget(self.strategy_params_group)

        # Run button
        self.run_button = QPushButton("🚀 Run Backtest")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.run_button.clicked.connect(self.run_backtest)
        layout.addWidget(self.run_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to run backtest")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        return panel

    def create_right_panel(self) -> QWidget:
        """Create right results panel"""
        panel = QWidget()

        layout = QVBoxLayout(panel)

        # Results tabs
        self.tabs = QTabWidget()

        # Results display
        self.results_widget = ResultDisplayWidget()
        self.tabs.addTab(self.results_widget, "📊 Results")

        # Chart display
        self.chart_widget = ChartDisplayWidget()
        self.tabs.addTab(self.chart_widget, "📈 Charts")

        layout.addWidget(self.tabs)

        return panel

    def setup_style(self):
        """Setup application styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e1e1e1;
                padding: 8px 16px;
                margin-right: 2px;
                border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #4CAF50;
            }
        """)

    def show_startup_info(self):
        """Show startup information"""
        strategies = self.bridge.get_available_strategies()
        categories = self.bridge.get_strategies_by_category()

        print("=" * 80)
        print("🚀 HOWTRADER BACKTEST APPLICATION STARTED")
        print("=" * 80)

        if self.bridge.database:
            print("🔧 Integration Status: ✅ REAL HOWTRADER")
            print("📊 Available Features: Full CTA Strategy Integration")
            print("🎯 Ready for: Professional Backtesting")
        else:
            print("🔧 Integration Status: ⚠️ MOCK MODE")
            print("📊 Available Features: Demo/Testing Only")
            print("🎯 Ready for: Strategy Development")

        print("=" * 80)
        print(f"📚 Demo Strategies: {len(categories.get('demo', []))}")
        print(f"🏭 Production Strategies: {len(categories.get('production', []))}")
        print(f"🧪 Mock Strategies: {len(categories.get('mock', []))}")
        print(f"📊 Total Strategies: {len(strategies)}")
        print("=" * 80)

    def on_strategy_changed(self, strategy_name: str):
        """Handle strategy selection change"""
        # Clear existing parameters
        self._clear_layout(self.strategy_params_layout)

        # Load strategy-specific parameters
        self.bridge.on_strategy_changed(strategy_name, self.strategy_params_layout)

    def _clear_layout(self, layout):
        """Clear all widgets from layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def run_backtest(self):
        """Run backtest in background thread"""
        try:
            # Validate inputs
            strategy_name = self.strategy_widget.get_selected_strategy()
            if not strategy_name:
                QMessageBox.warning(self, "⚠️ Invalid Input", "Please select a strategy")
                return

            # Prepare configuration
            config = {
                'strategy_name': strategy_name,
                'symbol': self.symbol_input.text(),
                'timeframe': self.timeframe_combo.currentText(),
                'start_date': self.start_date.date().toPython(),
                'end_date': self.end_date.date().toPython(),
                'capital': self.capital_input.value(),
                'commission': self.commission_input.value() / 100,
                'slippage': self.slippage_input.value() / 100,
                'strategy_params': {}
            }

            # Update UI state
            self.run_button.setEnabled(False)
            self.run_button.setText("⏳ Running...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Setup and start worker
            self.worker.setup_backtest(config)
            self.worker.start()

        except Exception as e:
            self.on_backtest_failed(f"Failed to start backtest: {e}")

    def on_progress_updated(self, value: int):
        """Handle progress updates"""
        self.progress_bar.setValue(value)

    def on_status_updated(self, status: str):
        """Handle status updates"""
        self.status_label.setText(status)

    def on_backtest_completed(self, results: Dict[str, Any]):
        """Handle backtest completion"""
        # Update UI state
        self.run_button.setEnabled(True)
        self.run_button.setText("🚀 Run Backtest")
        self.progress_bar.setVisible(False)

        # Display results
        self.results_widget.display_results(results)
        self.chart_widget.display_chart(results)

        # Switch to results tab
        self.tabs.setCurrentIndex(0)

        # Show completion message
        total_return = results.get('total_return', 0) * 100
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0) * 100

        QMessageBox.information(
            self,
            "✅ Backtest Completed",
            f"Backtest completed successfully!\n\n"
            f"📊 Total Return: {total_return:.2f}%\n"
            f"📈 Sharpe Ratio: {sharpe_ratio:.2f}\n"
            f"📉 Max Drawdown: {max_drawdown:.2f}%\n\n"
            f"🔧 Mode: {results.get('mode', 'Unknown').title()}"
        )

    def on_backtest_failed(self, error: str):
        """Handle backtest failure"""
        # Update UI state
        self.run_button.setEnabled(True)
        self.run_button.setText("🚀 Run Backtest")
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Backtest failed")

        # Show error message
        QMessageBox.critical(self, "❌ Backtest Failed", error)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Howtrader Backtest Platform")
    app.setApplicationVersion("2.0")

    # Set application icon (if available)
    try:
        app.setWindowIcon(QIcon("assets/icon.png"))
    except:
        pass

    # Create and show main window
    window = MainWindow()
    window.show()

    print("\n🎉 Application started successfully!")
    print("💡 Tip: Select a strategy and configure parameters to run your first backtest")

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()