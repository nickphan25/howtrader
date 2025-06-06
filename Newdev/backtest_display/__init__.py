"""
Backtest Display Package
=======================

Professional backtest visualization package với PySide6 integration.

Main Components:
- Display: Chart widgets and visualization components
- Config: Configuration management
- Core: Core backtesting integration
- Utils: Utility functions
- UI: User interface components

Author: AI Assistant
Version: 1.0
"""

# Import main components
from .display import (
    AdvancedHowtraderChart,
    ChartConfig,
    create_chart_widget,
    create_complete_display_system
)

from .config import (
    DisplayConfig,
    ChartTheme,
    create_default_config
)

from .core import (
    BacktestBridge,
    BacktestDataExtractor,
    create_backtest_bridge
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    # Display components
    "AdvancedHowtraderChart",
    "ChartConfig",
    "create_chart_widget",
    "create_complete_display_system",

    # Configuration
    "DisplayConfig",
    "ChartTheme",
    "create_default_config",

    # Core components
    "BacktestBridge",
    "BacktestDataExtractor",
    "create_backtest_bridge"
]


def create_app_components():
    """Create all main app components"""
    return {
        'chart': create_chart_widget(),
        'config': create_default_config(),
        'bridge': create_backtest_bridge()
    }


def get_version():
    """Get package version"""
    return __version__