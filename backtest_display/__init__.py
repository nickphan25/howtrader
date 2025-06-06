"""
Howtrader Backtest Display Package
=================================

This package provides a professional backtesting interface for Howtrader CTA strategies.
"""

# Import core backtesting functionality
from .backtest_bridge import (
    RealBacktestBridge,
    BacktestBridge,
    StrategyInfo
)

# Import UI widgets
from .result_display import ResultDisplayWidget
from .chart_display import ChartDisplayWidget

# Package metadata
__version__ = "2.0"
__author__ = "Howtrader Team"
__description__ = "Professional backtesting interface for Howtrader CTA strategies"

# Define what gets imported with "from backtest_display import *"
__all__ = [
    # Core classes
    "RealBacktestBridge",
    "BacktestBridge",
    "StrategyInfo",

    # UI widgets
    "ResultDisplayWidget",
    "ChartDisplayWidget",
]