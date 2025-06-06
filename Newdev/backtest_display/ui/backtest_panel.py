"""
Backtest Control Panel
=====================

UI control panel cho backtest operations, strategy configuration,
parameter settings và backtest execution controls.

Author: AI Assistant
Version: 1.0
"""

from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime, timedelta
import json
import os

from PySide6.QtCore import Qt, Signal, QTimer, QThread, QObject
from PySide6.QtGui import QFont, QPixmap, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QTextEdit, QGroupBox, QFrame,
    QDateTimeEdit, QProgressBar, QSlider, QTabWidget,
    QScrollArea, QSplitter, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)

from ..config.display_config import DisplayConfig
from ..core.backtest_bridge import BacktestBridge
from ..core.data_extractor import BacktestDataExtractor


class StrategyConfigWidget(QWidget):
    """
    Widget cho strategy configuration và parameters
    """

    strategy_changed = Signal(str)  # strategy name
    parameters_changed = Signal(dict)  # strategy parameters

    def __init__(self):
        super().__init__()
        self.strategy_parameters = {}
        self.init_ui()

    def init_ui(self):
        """Initialize strategy config UI"""
        layout = QVBoxLayout(self)

        # Strategy selection
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Strategy:"))

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Select Strategy...",
            "BollingerBandsStrategy",
            "MovingAverageCrossStrategy",
            "RSIMeanReversionStrategy",
            "BreakoutStrategy",
            "GridTradingStrategy",
            "Custom Strategy"
        ])
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo)

        # Load strategy button
        load_btn = QPushButton("Load...")
        load_btn.clicked.connect(self._load_strategy_file)
        strategy_layout.addWidget(load_btn)

        layout.addLayout(strategy_layout)

        # Strategy parameters area
        self.params_group = QGroupBox("Strategy Parameters")
        self.params_layout = QVBoxLayout(self.params_group)

        # Scroll area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)

        self.params_widget = QWidget()
        self.params_grid = QGridLayout(self.params_widget)
        scroll_area.setWidget(self.params_widget)

        self.params_layout.addWidget(scroll_area)
        layout.addWidget(self.params_group)

        # Strategy description
        self.description_text = QTextEdit()
        self.description_text.setMaximumHeight(100)
        self.description_text.setPlainText("Select a strategy to see its description and parameters...")
        layout.addWidget(QLabel("Description:"))
        layout.addWidget(self.description_text)

        # Preset management
        preset_layout = QHBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Default")
        preset_layout.addWidget(QLabel("Preset:"))
        preset_layout.addWidget(self.preset_combo)

        save_preset_btn = QPushButton("Save")
        save_preset_btn.clicked.connect(self._save_preset)
        preset_layout.addWidget(save_preset_btn)

        load_preset_btn = QPushButton("Load")
        load_preset_btn.clicked.connect(self._load_preset)
        preset_layout.addWidget(load_preset_btn)

        layout.addLayout(preset_layout)

    def _on_strategy_changed(self, strategy_name: str):
        """Handle strategy selection change"""
        if strategy_name == "Select Strategy...":
            return

        self.strategy_changed.emit(strategy_name)
        self._load_strategy_parameters(strategy_name)
        self._update_strategy_description(strategy_name)

    def _load_strategy_parameters(self, strategy_name: str):
        """Load parameters for selected strategy"""
        # Clear existing parameters
        self._clear_parameters()

        # Define parameter schemas for different strategies
        parameter_schemas = {
            "BollingerBandsStrategy": [
                ("bb_length", "BB Length", "int", 20, 5, 100),
                ("bb_dev", "BB Deviation", "float", 2.0, 0.5, 5.0),
                ("rsi_length", "RSI Length", "int", 14, 5, 50),
                ("rsi_entry", "RSI Entry", "int", 30, 10, 50),
                ("rsi_exit", "RSI Exit", "int", 70, 50, 90),
                ("fixed_size", "Position Size", "float", 1.0, 0.1, 10.0)
            ],
            "MovingAverageCrossStrategy": [
                ("fast_period", "Fast MA Period", "int", 10, 5, 50),
                ("slow_period", "Slow MA Period", "int", 30, 10, 200),
                ("ma_type", "MA Type", "str", "EMA", ["SMA", "EMA", "WMA"]),
                ("fixed_size", "Position Size", "float", 1.0, 0.1, 10.0)
            ],
            "RSIMeanReversionStrategy": [
                ("rsi_length", "RSI Length", "int", 14, 5, 50),
                ("rsi_oversold", "RSI Oversold", "int", 30, 10, 40),
                ("rsi_overbought", "RSI Overbought", "int", 70, 60, 90),
                ("exit_rsi", "Exit RSI", "int", 50, 40, 60),
                ("fixed_size", "Position Size", "float", 1.0, 0.1, 10.0)
            ],
            "BreakoutStrategy": [
                ("lookback_period", "Lookback Period", "int", 20, 5, 100),
                ("breakout_threshold", "Breakout Threshold", "float", 0.02, 0.001, 0.1),
                ("stop_loss", "Stop Loss %", "float", 0.02, 0.005, 0.1),
                ("take_profit", "Take Profit %", "float", 0.04, 0.01, 0.2),
                ("fixed_size", "Position Size", "float", 1.0, 0.1, 10.0)
            ],
            "GridTradingStrategy": [
                ("grid_spacing", "Grid Spacing %", "float", 0.01, 0.001, 0.05),
                ("max_positions", "Max Positions", "int", 5, 1, 20),
                ("take_profit", "Take Profit %", "float", 0.02, 0.005, 0.1),
                ("position_size", "Position Size", "float", 1.0, 0.1, 10.0)
            ]
        }

        if strategy_name in parameter_schemas:
            params = parameter_schemas[strategy_name]
            self._create_parameter_widgets(params)
        else:
            # Custom strategy - show basic parameters
            basic_params = [
                ("fixed_size", "Position Size", "float", 1.0, 0.1, 10.0),
                ("custom_param1", "Custom Param 1", "float", 1.0, 0.0, 100.0),
                ("custom_param2", "Custom Param 2", "int", 10, 1, 100)
            ]
            self._create_parameter_widgets(basic_params)

    def _create_parameter_widgets(self, params: List):
        """Create parameter input widgets"""
        row = 0
        self.parameter_widgets = {}

        for param_data in params:
            if len(param_data) == 6:
                name, label, param_type, default, min_val, max_val = param_data
                choices = None
            else:
                name, label, param_type, default, choices = param_data
                min_val = max_val = None

            # Create label
            label_widget = QLabel(f"{label}:")
            self.params_grid.addWidget(label_widget, row, 0)

            # Create input widget based on type
            if param_type == "int":
                widget = QSpinBox()
                if min_val is not None:
                    widget.setMinimum(min_val)
                if max_val is not None:
                    widget.setMaximum(max_val)
                widget.setValue(default)
                widget.valueChanged.connect(self._on_parameter_changed)

            elif param_type == "float":
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                if min_val is not None:
                    widget.setMinimum(min_val)
                if max_val is not None:
                    widget.setMaximum(max_val)
                widget.setValue(default)
                widget.valueChanged.connect(self._on_parameter_changed)

            elif param_type == "str" and choices:
                widget = QComboBox()
                widget.addItems(choices)
                widget.setCurrentText(default)
                widget.currentTextChanged.connect(self._on_parameter_changed)

            else:  # str without choices
                widget = QLineEdit()
                widget.setText(str(default))
                widget.textChanged.connect(self._on_parameter_changed)

            widget.setProperty("param_name", name)
            self.parameter_widgets[name] = widget
            self.params_grid.addWidget(widget, row, 1)

            # Store initial value
            self.strategy_parameters[name] = default

            row += 1

    def _clear_parameters(self):
        """Clear existing parameter widgets"""
        while self.params_grid.count():
            child = self.params_grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.parameter_widgets = {}
        self.strategy_parameters = {}

    def _update_strategy_description(self, strategy_name: str):
        """Update strategy description"""
        descriptions = {
            "BollingerBandsStrategy": "Trades based on Bollinger Bands and RSI signals. Buys when price touches lower band and RSI is oversold, sells when price touches upper band and RSI is overbought.",
            "MovingAverageCrossStrategy": "Classic moving average crossover strategy. Generates buy signals when fast MA crosses above slow MA, and sell signals when fast MA crosses below slow MA.",
            "RSIMeanReversionStrategy": "Mean reversion strategy using RSI indicator. Buys when RSI is oversold and sells when RSI is overbought, targeting mean reversion moves.",
            "BreakoutStrategy": "Momentum strategy that trades breakouts from price consolidation patterns. Uses lookback period to identify support/resistance levels.",
            "GridTradingStrategy": "Grid trading strategy that places multiple orders at different price levels. Profits from market volatility within a trading range."
        }

        description = descriptions.get(strategy_name, "Custom strategy - configure parameters as needed.")
        self.description_text.setPlainText(description)

    def _on_parameter_changed(self):
        """Handle parameter value changes"""
        sender = self.sender()
        param_name = sender.property("param_name")

        if isinstance(sender, QSpinBox):
            value = sender.value()
        elif isinstance(sender, QDoubleSpinBox):
            value = sender.value()
        elif isinstance(sender, QComboBox):
            value = sender.currentText()
        elif isinstance(sender, QLineEdit):
            value = sender.text()
            # Try to convert to number if possible
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
        else:
            return

        self.strategy_parameters[param_name] = value
        self.parameters_changed.emit(self.strategy_parameters.copy())

    def _load_strategy_file(self):
        """Load strategy from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Strategy File",
            "",
            "Python Files (*.py);;All Files (*)"
        )

        if file_path:
            # For now, just show the file name in combo box
            strategy_name = os.path.basename(file_path).replace('.py', '')
            self.strategy_combo.addItem(strategy_name)
            self.strategy_combo.setCurrentText(strategy_name)

    def _save_preset(self):
        """Save current parameters as preset"""
        if not self.strategy_parameters:
            return

        preset_name = f"Preset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.preset_combo.addItem(preset_name)

        # Save to file (in real implementation)
        print(f"Saved preset: {preset_name} with parameters: {self.strategy_parameters}")

    def _load_preset(self):
        """Load selected preset"""
        preset_name = self.preset_combo.currentText()
        if preset_name == "Default":
            return

        # Load from file (in real implementation)
        print(f"Loading preset: {preset_name}")

    def get_strategy_config(self) -> Dict[str, Any]:
        """Get current strategy configuration"""
        return {
            'strategy_name': self.strategy_combo.currentText(),
            'parameters': self.strategy_parameters.copy()
        }

    def set_strategy_config(self, config: Dict[str, Any]):
        """Set strategy configuration"""
        if 'strategy_name' in config:
            self.strategy_combo.setCurrentText(config['strategy_name'])

        if 'parameters' in config:
            self.strategy_parameters = config['parameters'].copy()
            # Update UI widgets with parameters
            for name, value in self.strategy_parameters.items():
                if name in self.parameter_widgets:
                    widget = self.parameter_widgets[name]
                    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                        widget.setValue(value)
                    elif isinstance(widget, QComboBox):
                        widget.setCurrentText(str(value))
                    elif isinstance(widget, QLineEdit):
                        widget.setText(str(value))


class BacktestParametersWidget(QWidget):
    """
    Widget cho backtest parameters (symbol, timeframe, dates, etc.)
    """

    parameters_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.backtest_parameters = {}
        self.init_ui()

    def init_ui(self):
        """Initialize backtest parameters UI"""
        layout = QVBoxLayout(self)

        # Market data section
        market_group = QGroupBox("Market Data")
        market_layout = QGridLayout(market_group)

        # Symbol
        market_layout.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        self.symbol_combo.addItems([
            "BTCUSDT.BINANCE",
            "ETHUSDT.BINANCE",
            "ADAUSDT.BINANCE",
            "SOLUSDT.BINANCE",
            "DOGEUSDT.BINANCE"
        ])
        self.symbol_combo.currentTextChanged.connect(self._on_parameter_changed)
        market_layout.addWidget(self.symbol_combo, 0, 1)

        # Exchange
        market_layout.addWidget(QLabel("Exchange:"), 1, 0)
        self.exchange_combo = QComboBox()
        self.exchange_combo.addItems(["BINANCE", "OKEX", "HUOBI", "BYBIT"])
        self.exchange_combo.currentTextChanged.connect(self._on_parameter_changed)
        market_layout.addWidget(self.exchange_combo, 1, 1)

        # Timeframe
        market_layout.addWidget(QLabel("Timeframe:"), 2, 0)
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems([
            "1m", "5m", "15m", "30m", "1h", "4h", "1d"
        ])
        self.timeframe_combo.setCurrentText("1h")
        self.timeframe_combo.currentTextChanged.connect(self._on_parameter_changed)
        market_layout.addWidget(self.timeframe_combo, 2, 1)

        layout.addWidget(market_group)

        # Date range section
        date_group = QGroupBox("Date Range")
        date_layout = QGridLayout(date_group)

        # Start date
        date_layout.addWidget(QLabel("Start Date:"), 0, 0)
        self.start_date = QDateTimeEdit()
        self.start_date.setDateTime(datetime.now() - timedelta(days=180))
        self.start_date.setCalendarPopup(True)
        self.start_date.dateTimeChanged.connect(self._on_parameter_changed)
        date_layout.addWidget(self.start_date, 0, 1)

        # End date
        date_layout.addWidget(QLabel("End Date:"), 1, 0)
        self.end_date = QDateTimeEdit()
        self.end_date.setDateTime(datetime.now())
        self.end_date.setCalendarPopup(True)
        self.end_date.dateTimeChanged.connect(self._on_parameter_changed)
        date_layout.addWidget(self.end_date, 1, 1)

        # Quick date buttons
        quick_date_layout = QHBoxLayout()

        date_presets = [
            ("1M", 30),
            ("3M", 90),
            ("6M", 180),
            ("1Y", 365)
        ]

        for label, days in date_presets:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, d=days: self._set_date_range(d))
            quick_date_layout.addWidget(btn)

        date_layout.addLayout(quick_date_layout, 2, 0, 1, 2)

        layout.addWidget(date_group)

        # Trading parameters section
        trading_group = QGroupBox("Trading Parameters")
        trading_layout = QGridLayout(trading_group)

        # Initial capital
        trading_layout.addWidget(QLabel("Initial Capital:"), 0, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setMinimum(1000)
        self.capital_spin.setMaximum(10000000)
        self.capital_spin.setValue(100000)
        self.capital_spin.setSuffix(" USD")
        self.capital_spin.valueChanged.connect(self._on_parameter_changed)
        trading_layout.addWidget(self.capital_spin, 0, 1)

        # Commission rate
        trading_layout.addWidget(QLabel("Commission Rate:"), 1, 0)
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setDecimals(4)
        self.commission_spin.setMinimum(0)
        self.commission_spin.setMaximum(0.01)
        self.commission_spin.setValue(0.0005)
        self.commission_spin.setSuffix("%")
        self.commission_spin.valueChanged.connect(self._on_parameter_changed)
        trading_layout.addWidget(self.commission_spin, 1, 1)

        # Slippage
        trading_layout.addWidget(QLabel("Slippage:"), 2, 0)
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setDecimals(4)
        self.slippage_spin.setMinimum(0)
        self.slippage_spin.setMaximum(0.01)
        self.slippage_spin.setValue(0.0001)
        self.slippage_spin.setSuffix("%")
        self.slippage_spin.valueChanged.connect(self._on_parameter_changed)
        trading_layout.addWidget(self.slippage_spin, 2, 1)

        # Position size
        trading_layout.addWidget(QLabel("Position Size:"), 3, 0)
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setMinimum(0.1)
        self.size_spin.setMaximum(100)
        self.size_spin.setValue(1.0)
        self.size_spin.valueChanged.connect(self._on_parameter_changed)
        trading_layout.addWidget(self.size_spin, 3, 1)

        layout.addWidget(trading_group)

        # Initialize parameters
        self._update_all_parameters()

    def _on_parameter_changed(self):
        """Handle parameter changes"""
        self._update_all_parameters()
        self.parameters_changed.emit(self.backtest_parameters.copy())

    def _update_all_parameters(self):
        """Update all parameters"""
        self.backtest_parameters = {
            'symbol': self.symbol_combo.currentText(),
            'exchange': self.exchange_combo.currentText(),
            'timeframe': self.timeframe_combo.currentText(),
            'start_date': self.start_date.dateTime().toPython(),
            'end_date': self.end_date.dateTime().toPython(),
            'initial_capital': self.capital_spin.value(),
            'commission_rate': self.commission_spin.value() / 100,
            'slippage': self.slippage_spin.value() / 100,
            'position_size': self.size_spin.value()
        }

    def _set_date_range(self, days: int):
        """Set date range based on days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        self.start_date.setDateTime(start_date)
        self.end_date.setDateTime(end_date)

    def get_backtest_parameters(self) -> Dict[str, Any]:
        """Get current backtest parameters"""
        return self.backtest_parameters.copy()

    def set_backtest_parameters(self, params: Dict[str, Any]):
        """Set backtest parameters"""
        if 'symbol' in params:
            self.symbol_combo.setCurrentText(params['symbol'])
        if 'exchange' in params:
            self.exchange_combo.setCurrentText(params['exchange'])
        if 'timeframe' in params:
            self.timeframe_combo.setCurrentText(params['timeframe'])
        if 'start_date' in params:
            self.start_date.setDateTime(params['start_date'])
        if 'end_date' in params:
            self.end_date.setDateTime(params['end_date'])
        if 'initial_capital' in params:
            self.capital_spin.setValue(params['initial_capital'])
        if 'commission_rate' in params:
            self.commission_spin.setValue(params['commission_rate'] * 100)
        if 'slippage' in params:
            self.slippage_spin.setValue(params['slippage'] * 100)
        if 'position_size' in params:
            self.size_spin.setValue(params['position_size'])


class QuickBacktestWidget(QWidget):
    """
    Widget cho quick backtest execution
    """

    backtest_started = Signal()
    backtest_finished = Signal(dict)
    backtest_cancelled = Signal()

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.backtest_bridge = BacktestBridge()
        self.init_ui()

    def init_ui(self):
        """Initialize quick backtest UI"""
        layout = QVBoxLayout(self)

        # Control buttons
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.run_btn.clicked.connect(self._run_backtest)
        button_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.stop_btn.clicked.connect(self._stop_backtest)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to run backtest...")
        progress_layout.addWidget(self.status_label)

        # Estimated time
        self.time_label = QLabel("")
        progress_layout.addWidget(self.time_label)

        layout.addWidget(progress_group)

        # Quick results preview
        results_group = QGroupBox("Quick Results")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(200)

        results_layout.addWidget(self.results_table)
        layout.addWidget(results_group)

    def run_backtest(self, strategy_config: Dict, backtest_params: Dict):
        """
        Run backtest with given configuration

        Args:
            strategy_config: Strategy configuration
            backtest_params: Backtest parameters
        """
        if self.is_running:
            return

        self.is_running = True
        self.backtest_started.emit()

        # Update UI
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing backtest...")

        # Start backtest (mock implementation)
        self._mock_backtest_execution(strategy_config, backtest_params)

    def _run_backtest(self):
        """Handle run button click"""
        # This would be called by parent with proper config
        mock_strategy = {'strategy_name': 'TestStrategy', 'parameters': {}}
        mock_params = {
            'symbol': 'BTCUSDT.BINANCE',
            'timeframe': '1h',
            'start_date': datetime.now() - timedelta(days=90),
            'end_date': datetime.now()
        }
        self.run_backtest(mock_strategy, mock_params)

    def _stop_backtest(self):
        """Stop running backtest"""
        self.is_running = False
        self.backtest_cancelled.emit()
        self._reset_ui()

    def _mock_backtest_execution(self, strategy_config: Dict, backtest_params: Dict):
        """Mock backtest execution with progress updates"""
        # Simulate backtest progress
        self.timer = QTimer()
        self.progress_value = 0

        def update_progress():
            self.progress_value += 5
            self.progress_bar.setValue(self.progress_value)

            if self.progress_value <= 30:
                self.status_label.setText("Loading historical data...")
            elif self.progress_value <= 60:
                self.status_label.setText("Running strategy...")
            elif self.progress_value <= 90:
                self.status_label.setText("Calculating results...")
            else:
                self.status_label.setText("Finalizing...")

            if self.progress_value >= 100:
                self.timer.stop()
                self._finish_backtest()

        self.timer.timeout.connect(update_progress)
        self.timer.start(200)  # Update every 200ms

    def _finish_backtest(self):
        """Finish backtest and show results"""
        self.is_running = False

        # Mock results
        results = {
            'performance': {
                'total_return': 0.15,
                'annual_return': 0.45,
                'max_drawdown': -0.08,
                'sharpe_ratio': 1.25,
                'win_rate': 0.65,
                'profit_factor': 1.8
            },
            'summary': {
                'total_trades': 45,
                'profitable_trades': 29,
                'largest_win': 2500.0,
                'largest_loss': -800.0,
                'average_trade': 125.5
            }
        }

        self._update_results_display(results)
        self._reset_ui()
        self.status_label.setText("Backtest completed successfully!")

        self.backtest_finished.emit(results)

    def _update_results_display(self, results: Dict):
        """Update quick results display"""
        self.results_table.setRowCount(0)

        # Add performance metrics
        performance = results.get('performance', {})
        metrics = [
            ('Total Return', f"{performance.get('total_return', 0):.2%}"),
            ('Max Drawdown', f"{performance.get('max_drawdown', 0):.2%}"),
            ('Win Rate', f"{performance.get('win_rate', 0):.2%}"),
            ('Sharpe Ratio', f"{performance.get('sharpe_ratio', 0):.2f}"),
            ('Total Trades', str(results.get('summary', {}).get('total_trades', 0)))
        ]

        for i, (metric, value) in enumerate(metrics):
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(metric))
            self.results_table.setItem(i, 1, QTableWidgetItem(value))

    def _reset_ui(self):
        """Reset UI to initial state"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)


class BacktestControlPanel(QWidget):
    """
    Main backtest control panel combining all components
    """

    backtest_started = Signal()
    backtest_finished = Signal(dict)
    backtest_cancelled = Signal()
    config_changed = Signal(dict)

    def __init__(self, config: DisplayConfig = None):
        super().__init__()
        self.config = config or DisplayConfig()
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize main control panel UI"""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("Backtest Control Panel")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Strategy configuration tab
        self.strategy_widget = StrategyConfigWidget()
        self.tab_widget.addTab(self.strategy_widget, "Strategy")

        # Backtest parameters tab
        self.parameters_widget = BacktestParametersWidget()
        self.tab_widget.addTab(self.parameters_widget, "Parameters")

        # Quick backtest tab
        self.quick_widget = QuickBacktestWidget()
        self.tab_widget.addTab(self.quick_widget, "Execute")

        layout.addWidget(self.tab_widget)

        # Action buttons
        action_layout = QHBoxLayout()

        save_config_btn = QPushButton("Save Config")
        save_config_btn.clicked.connect(self._save_configuration)
        action_layout.addWidget(save_config_btn)

        load_config_btn = QPushButton("Load Config")
        load_config_btn.clicked.connect(self._load_configuration)
        action_layout.addWidget(load_config_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_configuration)
        action_layout.addWidget(reset_btn)

        layout.addLayout(action_layout)

    def connect_signals(self):
        """Connect internal signals"""
        # Strategy signals
        self.strategy_widget.strategy_changed.connect(self._on_config_changed)
        self.strategy_widget.parameters_changed.connect(self._on_config_changed)

        # Parameters signals
        self.parameters_widget.parameters_changed.connect(self._on_config_changed)

        # Quick backtest signals
        self.quick_widget.backtest_started.connect(self.backtest_started)
        self.quick_widget.backtest_finished.connect(self.backtest_finished)
        self.quick_widget.backtest_cancelled.connect(self.backtest_cancelled)

    def run_backtest(self):
        """Run backtest with current configuration"""
        strategy_config = self.strategy_widget.get_strategy_config()
        backtest_params = self.parameters_widget.get_backtest_parameters()

        if not strategy_config['strategy_name'] or strategy_config['strategy_name'] == "Select Strategy...":
            QMessageBox.warning(self, "Warning", "Please select a strategy first!")
            return

        self.quick_widget.run_backtest(strategy_config, backtest_params)

    def get_full_configuration(self) -> Dict[str, Any]:
        """Get complete backtest configuration"""
        return {
            'strategy': self.strategy_widget.get_strategy_config(),
            'parameters': self.parameters_widget.get_backtest_parameters(),
            'display_config': self.config.to_dict()
        }

    def set_full_configuration(self, config: Dict[str, Any]):
        """Set complete backtest configuration"""
        if 'strategy' in config:
            self.strategy_widget.set_strategy_config(config['strategy'])

        if 'parameters' in config:
            self.parameters_widget.set_backtest_parameters(config['parameters'])

        if 'display_config' in config:
            self.config = DisplayConfig.from_dict(config['display_config'])

    def _on_config_changed(self):
        """Handle configuration changes"""
        config = self.get_full_configuration()
        self.config_changed.emit(config)

    def _save_configuration(self):
        """Save current configuration to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Backtest Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                config = self.get_full_configuration()

                # Convert datetime objects to strings for JSON serialization
                def convert_datetime(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return obj

                import json
                with open(file_path, 'w') as f:
                    json.dump(config, f, default=convert_datetime, indent=2)

                QMessageBox.information(self, "Success", f"Configuration saved to {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")

    def _load_configuration(self):
        """Load configuration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Backtest Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    config = json.load(f)

                # Convert datetime strings back to datetime objects
                if 'parameters' in config:
                    params = config['parameters']
                    for key in ['start_date', 'end_date']:
                        if key in params and isinstance(params[key], str):
                            params[key] = datetime.fromisoformat(params[key])

                self.set_full_configuration(config)
                QMessageBox.information(self, "Success", f"Configuration loaded from {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {e}")

    def _reset_configuration(self):
        """Reset configuration to defaults"""
        reply = QMessageBox.question(
            self,
            "Reset Configuration",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Reset strategy
            self.strategy_widget.strategy_combo.setCurrentIndex(0)

            # Reset parameters to defaults
            self.parameters_widget.symbol_combo.setCurrentText("BTCUSDT.BINANCE")
            self.parameters_widget.timeframe_combo.setCurrentText("1h")
            self.parameters_widget.start_date.setDateTime(datetime.now() - timedelta(days=180))
            self.parameters_widget.end_date.setDateTime(datetime.now())
            self.parameters_widget.capital_spin.setValue(100000)

            # Reset display config
            self.config = DisplayConfig()


# Test function
def test_backtest_panel():
    """Test backtest control panel"""
    print("🎛️ Testing Backtest Control Panel...")

    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Create control panel
    panel = BacktestControlPanel()
    print("✅ Control panel created")

    # Connect test signals
    panel.backtest_started.connect(lambda: print("✅ Backtest started signal"))
    panel.backtest_finished.connect(lambda r: print(f"✅ Backtest finished: {len(r)} results"))
    panel.config_changed.connect(lambda c: print(f"✅ Config changed: {len(c)} parameters"))

    # Show panel
    panel.show()
    print("✅ Panel displayed")

    print("🎉 Backtest panel test completed!")

    # Don't start event loop in test
    panel.close()


if __name__ == "__main__":
    test_backtest_panel()