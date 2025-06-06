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
                               QProgressBar, QCheckBox,
                               QSplitter, QGroupBox, QFormLayout, QLineEdit, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal, QDate, QTimer
from PySide6.QtGui import QFont, QIcon

# Import backtest bridge with error handling
try:
    from backtest_bridge import RealBacktestBridge
    print("✅ Backtest bridge imported successfully")
except ImportError as e:
    print(f"❌ Failed to import backtest bridge: {e}")
    print("🔄 Creating fallback bridge...")

    # Create a minimal fallback bridge
    class RealBacktestBridge:
        def __init__(self):
            self.strategy_infos = {}
            self.loaded_strategies = {}
            self.database = None
            print("⚠️ Using fallback bridge - limited functionality")

        def get_available_strategies(self):
            return ["MockStrategy"]

        def get_strategies_by_category(self):
            return {"mock": ["MockStrategy"]}

        def get_strategy_display_names(self):
            return [("MockStrategy", "🧪 Mock Strategy [Beginner]")]

        def get_strategy_info(self, name):
            class MockInfo:
                def __init__(self):
                    self.name = "MockStrategy"
                    self.category = "mock"
                    self.complexity = "beginner"
                    self.description = "Fallback mock strategy"
                    self.features = ["Basic functionality"]
                def get_display_name(self):
                    return "🧪 Mock Strategy [Beginner]"
            return MockInfo()

        def get_strategy_class(self, name):
            return None

        def run_backtest(self, **kwargs):
            return {"error": "Backtest bridge not available"}

        def set_progress_callback(self, callback):
            pass

        def show_chart(self):
            print("📊 Chart display not available")

        def on_strategy_changed(self, strategy_name, layout):
            pass

try:
    from chart_display import ChartDisplayWidget
except ImportError:
    print("⚠️ Chart display not available")
    class ChartDisplayWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Chart display not available"))
        def display_chart(self, results):
            pass

try:
    from result_display import ResultDisplayWidget
except ImportError:
    print("⚠️ Result display not available")
    class ResultDisplayWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Result display not available"))
        def display_results(self, results):
            pass

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

class AppConfig:
    """Application configuration constants"""

    # Application Info
    APP_NAME = "Howtrader Backtest Platform"
    APP_VERSION = "2.1"
    WINDOW_TITLE = "🚀 Howtrader Professional Backtest Platform v2.1"

    # UI Settings
    MIN_WINDOW_SIZE = (1400, 900)
    LEFT_PANEL_WIDTH = (400, 450)
    SPLITTER_SIZES = [420, 980]

    # Default Values
    DEFAULT_SYMBOL = "BTCUSDT"
    DEFAULT_TIMEFRAME = "1h"
    DEFAULT_CAPITAL = 1000000
    DEFAULT_COMMISSION = 4/10000
    DEFAULT_SLIPPAGE = 0.0
    DEFAULT_DATE_RANGE_DAYS = 60
    DEFAULT_SIZE = 1.0
    DEFAULT_PRICETICK = 0.01

    # Available Options
    TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    EXCHANGES = ["BINANCE", "OKX", "HUOBI"]
    CATEGORIES = ["All Strategies", "📚 Demo/Educational", "🏭 Production", "🧪 Mock/Testing"]
    COMPLEXITY_LEVELS = ["All Levels", "Beginner", "Intermediate", "Advanced"]

    # Category Mapping
    CATEGORY_MAP = {
        "All Strategies": None,
        "📚 Demo/Educational": "demo",
        "🏭 Production": "production",
        "🧪 Mock/Testing": "mock"
    }


class StyleConfig:
    """UI styling configuration"""

    @staticmethod
    def get_main_style():
        """Get main application stylesheet"""
        return """
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
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """

    @staticmethod
    def get_primary_button_style():
        """Get primary button stylesheet"""
        return """
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
        """

    @staticmethod
    def get_secondary_button_style():
        """Get secondary button stylesheet"""
        return """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def show_startup_info(bridge: RealBacktestBridge):
    """Display startup information and system status"""
    try:
        strategies = bridge.get_available_strategies()
        categories = bridge.get_strategies_by_category()

        print("=" * 80)
        print("🚀 HOWTRADER BACKTEST APPLICATION STARTED")
        print("=" * 80)

        if hasattr(bridge, 'database') and bridge.database:
            print("🔧 Integration Status: ✅ REAL HOWTRADER")
            print("📊 Available Features: Full CTA Strategy Integration")
            print("🎯 Ready for: Professional Backtesting")
        else:
            print("🔧 Integration Status: ⚠️ DEMO MODE")
            print("📊 Available Features: Mock Strategies + Demo Capabilities")
            print("🎯 Ready for: Strategy Development & Learning")

        print("=" * 80)
        print(f"📚 Demo Strategies: {len(categories.get('demo', []))}")
        print(f"🏭 Production Strategies: {len(categories.get('production', []))}")
        print(f"🧪 Mock Strategies: {len(categories.get('mock', []))}")
        print(f"📊 Total Strategies: {len(strategies)}")
        print("=" * 80)
    except Exception as e:
        print(f"⚠️ Error showing startup info: {e}")


def clear_layout(layout):
    """Clear all widgets from a layout"""
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


# ============================================================================
# WORKER THREAD
# ============================================================================

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
            self._execute_backtest()
        except Exception as e:
            self.backtest_failed.emit(f"Backtest failed: {str(e)}")

    def _execute_backtest(self):
        """Execute the backtest process using bridge's BacktestingEngine workflow"""
        self.is_cancelled = False
        self.status_updated.emit("Initializing backtest...")
        self.progress_updated.emit(0)

        # Extract configuration
        strategy_name = self.config.get('strategy_name', '')
        symbol = self.config.get('symbol', AppConfig.DEFAULT_SYMBOL)
        timeframe = self.config.get('timeframe', AppConfig.DEFAULT_TIMEFRAME)
        start_date = self.config.get('start_date', datetime.now() - timedelta(days=AppConfig.DEFAULT_DATE_RANGE_DAYS))
        end_date = self.config.get('end_date', datetime.now())
        capital = self.config.get('capital', AppConfig.DEFAULT_CAPITAL)
        commission = self.config.get('commission', AppConfig.DEFAULT_COMMISSION)
        slippage = self.config.get('slippage', AppConfig.DEFAULT_SLIPPAGE)
        size = self.config.get('size', AppConfig.DEFAULT_SIZE)
        pricetick = self.config.get('pricetick', AppConfig.DEFAULT_PRICETICK)
        strategy_params = self.config.get('strategy_params', {})

        if self.is_cancelled:
            return

        self.status_updated.emit("Setting up BacktestingEngine...")
        self.progress_updated.emit(20)

        # Set progress callback
        if hasattr(self.bridge, 'set_progress_callback'):
            self.bridge.set_progress_callback(self._on_progress_update)

        # Run backtest using bridge's BacktestingEngine workflow
        results = self.bridge.run_backtest(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
            commission=commission,
            slippage=slippage,
            size=size,
            pricetick=pricetick,
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

    def _on_progress_update(self, message: str):
        """Handle progress updates from bridge"""
        self.status_updated.emit(message)


# ============================================================================
# UI WIDGETS
# ============================================================================

class StrategySelectionWidget(QWidget):
    """Enhanced strategy selection widget with categorization"""

    strategy_changed = Signal(str)

    def __init__(self, bridge: RealBacktestBridge):
        super().__init__()
        self.bridge = bridge
        self.strategy_mapping = {}
        self._init_ui()
        self._load_strategies()

    def _init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)

        # Strategy selection group
        strategy_group = QGroupBox("📊 Strategy Selection")
        strategy_layout = QVBoxLayout(strategy_group)

        # Category and complexity filters
        filters_layout = QHBoxLayout()

        # Category filter
        filters_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(AppConfig.CATEGORIES)
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        filters_layout.addWidget(self.category_combo)

        # Complexity filter
        filters_layout.addWidget(QLabel("Complexity:"))
        self.complexity_combo = QComboBox()
        self.complexity_combo.addItems(AppConfig.COMPLEXITY_LEVELS)
        self.complexity_combo.currentTextChanged.connect(self._on_complexity_changed)
        filters_layout.addWidget(self.complexity_combo)

        strategy_layout.addLayout(filters_layout)

        # Strategy selection
        strategy_select_layout = QHBoxLayout()
        strategy_select_layout.addWidget(QLabel("Strategy:"))

        self.strategy_combo = QComboBox()
        self.strategy_combo.setMinimumWidth(300)
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        strategy_select_layout.addWidget(self.strategy_combo)

        strategy_layout.addLayout(strategy_select_layout)

        # Strategy information display
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
        self.info_text.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ccc;")
        strategy_layout.addWidget(QLabel("Strategy Information:"))
        strategy_layout.addWidget(self.info_text)

        layout.addWidget(strategy_group)

    def _load_strategies(self):
        """Load strategies from bridge"""
        try:
            display_names = self.bridge.get_strategy_display_names()
            self.strategy_mapping = {}

            self.strategy_combo.clear()
            for actual_name, display_name in display_names:
                self.strategy_combo.addItem(display_name)
                self.strategy_mapping[display_name] = actual_name

            print(f"✅ Loaded {len(display_names)} strategies into UI")

            if display_names:
                self._update_strategy_info(display_names[0][0])

        except Exception as e:
            print(f"⚠️ Error loading strategies: {e}")
            # Add fallback strategy
            self.strategy_combo.addItem("Fallback Strategy")
            self.strategy_mapping["Fallback Strategy"] = "FallbackStrategy"

    def _on_category_changed(self, category_text: str):
        """Handle category filter change"""
        try:
            selected_category = AppConfig.CATEGORY_MAP.get(category_text)
            self._filter_strategies(category=selected_category)
        except Exception as e:
            print(f"⚠️ Error filtering by category: {e}")

    def _on_complexity_changed(self, complexity_text: str):
        """Handle complexity filter change"""
        try:
            complexity = complexity_text.lower() if complexity_text != "All Levels" else None
            self._filter_strategies(complexity=complexity)
        except Exception as e:
            print(f"⚠️ Error filtering by complexity: {e}")

    def _filter_strategies(self, category: str = None, complexity: str = None):
        """Filter strategies by category and complexity"""
        try:
            current_category = self.category_combo.currentText()
            current_complexity = self.complexity_combo.currentText()

            filter_category = AppConfig.CATEGORY_MAP.get(current_category)
            filter_complexity = current_complexity.lower() if current_complexity != "All Levels" else None

            # Get filtered strategies
            filtered_strategies = []
            if hasattr(self.bridge, 'strategy_infos'):
                for name, info in self.bridge.strategy_infos.items():
                    if filter_category and info.category != filter_category:
                        continue
                    if filter_complexity and info.complexity != filter_complexity:
                        continue
                    filtered_strategies.append((name, info.get_display_name()))
            else:
                # Fallback
                filtered_strategies = [("FallbackStrategy", "Fallback Strategy")]

            # Update combo box
            self.strategy_combo.clear()
            self.strategy_mapping = {}

            for actual_name, display_name in filtered_strategies:
                self.strategy_combo.addItem(display_name)
                self.strategy_mapping[display_name] = actual_name

            if filtered_strategies:
                self._update_strategy_info(filtered_strategies[0][0])

        except Exception as e:
            print(f"⚠️ Error filtering strategies: {e}")

    def _on_strategy_changed(self, display_name: str):
        """Handle strategy selection change"""
        try:
            if display_name and display_name in self.strategy_mapping:
                actual_name = self.strategy_mapping[display_name]
                self._update_strategy_info(actual_name)
                self.strategy_changed.emit(actual_name)
        except Exception as e:
            print(f"⚠️ Error handling strategy change: {e}")

    def _update_strategy_info(self, strategy_name: str):
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
                self.info_text.setPlainText(f"Strategy: {strategy_name}")
        except Exception as e:
            print(f"⚠️ Error updating strategy info: {e}")
            self.info_text.setPlainText(f"Strategy: {strategy_name}")

    def get_selected_strategy(self) -> str:
        """Get currently selected strategy name"""
        display_name = self.strategy_combo.currentText()
        return self.strategy_mapping.get(display_name, "")


class MarketParametersWidget(QGroupBox):
    """Market parameters configuration widget"""

    def __init__(self):
        super().__init__("📈 Market Parameters")
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        layout = QFormLayout(self)

        self.symbol_input = QLineEdit(AppConfig.DEFAULT_SYMBOL)
        layout.addRow("Symbol:", self.symbol_input)

        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(AppConfig.TIMEFRAMES)
        self.timeframe_combo.setCurrentText(AppConfig.DEFAULT_TIMEFRAME)
        layout.addRow("Timeframe:", self.timeframe_combo)

        # Date range
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-AppConfig.DEFAULT_DATE_RANGE_DAYS))
        self.start_date.setCalendarPopup(True)
        layout.addRow("Start Date:", self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        layout.addRow("End Date:", self.end_date)

    def get_symbol(self) -> str:
        """Get selected symbol"""
        return self.symbol_input.text()

    def get_timeframe(self) -> str:
        """Get selected timeframe"""
        return self.timeframe_combo.currentText()

    def get_start_date(self) -> datetime:
        """Get start date"""
        return self.start_date.date().toPython()

    def get_end_date(self) -> datetime:
        """Get end date"""
        return self.end_date.date().toPython()


class TradingParametersWidget(QGroupBox):
    """Trading parameters configuration widget with BacktestingEngine parameters"""

    def __init__(self):
        super().__init__("💰 Trading Parameters")
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        layout = QFormLayout(self)

        self.capital_input = QSpinBox()
        self.capital_input.setRange(10000, 100000000)
        self.capital_input.setValue(AppConfig.DEFAULT_CAPITAL)
        self.capital_input.setSuffix(" USD")
        layout.addRow("Initial Capital:", self.capital_input)

        self.commission_input = QDoubleSpinBox()
        self.commission_input.setRange(0.0, 0.01)
        self.commission_input.setValue(AppConfig.DEFAULT_COMMISSION)
        self.commission_input.setDecimals(6)
        self.commission_input.setSuffix(" (rate)")
        layout.addRow("Commission Rate:", self.commission_input)

        self.slippage_input = QDoubleSpinBox()
        self.slippage_input.setRange(0.0, 0.01)
        self.slippage_input.setValue(AppConfig.DEFAULT_SLIPPAGE)
        self.slippage_input.setDecimals(6)
        layout.addRow("Slippage:", self.slippage_input)

        self.size_input = QDoubleSpinBox()
        self.size_input.setRange(0.01, 1000.0)
        self.size_input.setValue(AppConfig.DEFAULT_SIZE)
        self.size_input.setDecimals(2)
        layout.addRow("Size:", self.size_input)

        self.pricetick_input = QDoubleSpinBox()
        self.pricetick_input.setRange(0.00001, 100.0)
        self.pricetick_input.setValue(AppConfig.DEFAULT_PRICETICK)
        self.pricetick_input.setDecimals(5)
        layout.addRow("Price Tick:", self.pricetick_input)

    def get_capital(self) -> int:
        """Get capital value"""
        return self.capital_input.value()

    def get_commission(self) -> float:
        """Get commission value (rate format for BacktestingEngine)"""
        return self.commission_input.value()

    def get_slippage(self) -> float:
        """Get slippage value"""
        return self.slippage_input.value()

    def get_size(self) -> float:
        """Get size value"""
        return self.size_input.value()

    def get_pricetick(self) -> float:
        """Get pricetick value"""
        return self.pricetick_input.value()


class StrategyParametersWidget(QGroupBox):
    """Strategy-specific parameters widget"""

    def __init__(self, bridge: RealBacktestBridge):
        super().__init__("⚙️ Strategy Parameters")
        self.bridge = bridge
        self.param_widgets = {}
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        self.layout = QVBoxLayout(self)

        # Default message
        default_label = QLabel("Select a strategy to see parameters")
        default_label.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        self.layout.addWidget(default_label)

    def on_strategy_changed(self, strategy_name: str):
        """Handle strategy change and update parameters"""
        clear_layout(self.layout)
        self.param_widgets = {}
        try:
            self.bridge.on_strategy_changed(strategy_name, self.layout)
        except Exception as e:
            print(f"⚠️ Error updating strategy parameters: {e}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values"""
        # This would be implemented based on dynamic widgets
        # For now, return empty dict
        return {}


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    """Enhanced main window with BacktestingEngine integration"""

    def __init__(self):
        super().__init__()

        # Initialize core components
        print("🔧 Initializing Howtrader Bridge...")
        self.bridge = RealBacktestBridge()
        self.worker = BacktestWorker(self.bridge)

        # Initialize UI
        self._init_ui()
        self._setup_connections()
        self._setup_style()

        # Show startup info
        show_startup_info(self.bridge)

        # Start status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)  # Update every second

    def _init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(AppConfig.WINDOW_TITLE)
        self.setMinimumSize(*AppConfig.MIN_WINDOW_SIZE)

        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)

        # Create panels
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes(AppConfig.SPLITTER_SIZES)

        main_layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        """Create left control panel"""
        panel = QWidget()
        panel.setMaximumWidth(AppConfig.LEFT_PANEL_WIDTH[1])
        panel.setMinimumWidth(AppConfig.LEFT_PANEL_WIDTH[0])

        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("🎯 Backtest Configuration")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Configuration widgets
        self.strategy_widget = StrategySelectionWidget(self.bridge)
        self.market_widget = MarketParametersWidget()
        self.trading_widget = TradingParametersWidget()
        self.strategy_params_widget = StrategyParametersWidget(self.bridge)

        layout.addWidget(self.strategy_widget)
        layout.addWidget(self.market_widget)
        layout.addWidget(self.trading_widget)
        layout.addWidget(self.strategy_params_widget)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self.run_button = QPushButton("🚀 Run Backtest")
        self.run_button.setStyleSheet(StyleConfig.get_primary_button_style())
        self.run_button.clicked.connect(self._run_backtest)
        buttons_layout.addWidget(self.run_button)

        self.show_chart_button = QPushButton("📊 Show Chart")
        self.show_chart_button.setStyleSheet(StyleConfig.get_secondary_button_style())
        self.show_chart_button.clicked.connect(self._show_chart)
        self.show_chart_button.setEnabled(False)
        buttons_layout.addWidget(self.show_chart_button)

        layout.addLayout(buttons_layout)

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

    def _create_right_panel(self) -> QWidget:
        """Create right results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Results tabs
        self.tabs = QTabWidget()
        self.results_widget = ResultDisplayWidget()
        self.chart_widget = ChartDisplayWidget()

        self.tabs.addTab(self.results_widget, "📊 Results")
        self.tabs.addTab(self.chart_widget, "📈 Charts")

        layout.addWidget(self.tabs)
        return panel

    def _setup_connections(self):
        """Setup signal connections"""
        # Worker connections
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.status_updated.connect(self._on_status_updated)
        self.worker.backtest_completed.connect(self._on_backtest_completed)
        self.worker.backtest_failed.connect(self._on_backtest_failed)

        # Strategy selection connection
        self.strategy_widget.strategy_changed.connect(
            self.strategy_params_widget.on_strategy_changed
        )

    def _setup_style(self):
        """Setup application styling"""
        self.setStyleSheet(StyleConfig.get_main_style())

    def _run_backtest(self):
        """Run backtest with current configuration using BacktestingEngine workflow"""
        try:
            # Build configuration
            config = {
                'strategy_name': self.strategy_widget.get_selected_strategy(),
                'symbol': self.market_widget.get_symbol(),
                'timeframe': self.market_widget.get_timeframe(),
                'start_date': self.market_widget.get_start_date(),
                'end_date': self.market_widget.get_end_date(),
                'capital': self.trading_widget.get_capital(),
                'commission': self.trading_widget.get_commission(),
                'slippage': self.trading_widget.get_slippage(),
                'size': self.trading_widget.get_size(),
                'pricetick': self.trading_widget.get_pricetick(),
                'strategy_params': self.strategy_params_widget.get_parameters()
            }

            # Validate configuration
            if not config['strategy_name']:
                QMessageBox.warning(self, "⚠️ Invalid Input", "Please select a strategy")
                return

            # Update UI state
            self._set_running_state()

            # Start backtest
            self.worker.setup_backtest(config)
            self.worker.start()

        except Exception as e:
            self._on_backtest_failed(f"Failed to start backtest: {e}")

    def _show_chart(self):
        """Show chart using bridge's show_chart method"""
        try:
            self.bridge.show_chart()
        except Exception as e:
            QMessageBox.warning(self, "Chart Error", f"Failed to show chart: {str(e)}")

    def _set_running_state(self):
        """Set UI to running state"""
        self.run_button.setEnabled(False)
        self.run_button.setText("⏳ Running...")
        self.show_chart_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    def _set_ready_state(self):
        """Set UI to ready state"""
        self.run_button.setEnabled(True)
        self.run_button.setText("🚀 Run Backtest")
        self.show_chart_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_progress_updated(self, value: int):
        """Handle progress updates"""
        self.progress_bar.setValue(value)

    def _on_status_updated(self, status: str):
        """Handle status updates"""
        self.status_label.setText(status)

    def _on_backtest_completed(self, results: Dict[str, Any]):
        """Handle backtest completion"""
        self._set_ready_state()

        # Display results
        self.results_widget.display_results(results)
        self.chart_widget.display_chart(results)
        self.tabs.setCurrentIndex(0)

        # Show completion message
        self._show_completion_message(results)

    def _on_backtest_failed(self, error: str):
        """Handle backtest failure"""
        self._set_ready_state()
        self.status_label.setText("❌ Backtest failed")
        QMessageBox.critical(self, "❌ Backtest Failed", error)

    def _show_completion_message(self, results: Dict[str, Any]):
        """Show backtest completion message"""
        total_return = results.get('total_return', 0) * 100
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0) * 100
        mode = results.get('mode', 'Unknown')

        QMessageBox.information(
            self,
            "✅ Backtest Completed",
            f"Backtest completed successfully!\n\n"
            f"📊 Total Return: {total_return:+.2f}%\n"
            f"📈 Sharpe Ratio: {sharpe_ratio:.2f}\n"
            f"📉 Max Drawdown: {max_drawdown:.2f}%\n\n"
            f"🔧 Mode: {mode.title()}"
        )

    def _update_status(self):
        """Update application status periodically"""
        try:
            if hasattr(self, 'bridge') and self.bridge:
                total_strategies = len(self.bridge.get_available_strategies())
                if not self.worker.isRunning():
                    self.status_label.setText(f"Ready • {total_strategies} strategies loaded")
        except Exception as e:
            pass  # Silently ignore status update errors

    def closeEvent(self, event):
        """Handle application close event"""
        if self.worker.isRunning():
            reply = QMessageBox.question(
                self, 'Close Application',
                'A backtest is currently running. Do you want to stop it and exit?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.worker.cancel_backtest()
                self.worker.wait(3000)  # Wait up to 3 seconds
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def setup_application():
    """Setup QApplication with proper configuration"""
    app = QApplication(sys.argv)
    app.setApplicationName(AppConfig.APP_NAME)
    app.setApplicationVersion(AppConfig.APP_VERSION)

    # Set application icon (if available)
    try:
        app.setWindowIcon(QIcon("assets/icon.png"))
    except:
        pass

    return app


def main():
    """Main application entry point"""
    # Create application
    app = setup_application()

    # Create and show main window
    try:
        window = MainWindow()
        window.show()

        print("\n🎉 Application started successfully!")
        print("💡 Tip: Select a strategy and configure parameters to run your first backtest")

        # Run application
        sys.exit(app.exec())

    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        QMessageBox.critical(None, "Application Error", f"Failed to start application:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()