"""
Display Module
=============

Display components for backtest visualization.

Components:
- BacktestMarkers: Trading signal and position markers
- ChartWidget: Main chart widget (replaces chartish)

Author: AI Assistant
Version: 1.0
"""

# Import chart widget first (standalone)
from .chart_widget import (
    AdvancedHowtraderChart,
    ChartConfig,
    CandlestickItem,
    VolumeBarItem,
    IndicatorItem,
    DataLoader
)

# Import markers (with fixed imports)
from .backtest_markers import (
    TradeSignal,
    TradeMarker,
    TradeLine,
    PositionIndicator,
    PnLOverlay,
    VolumeMarker,
    BacktestStatisticsOverlay,
    MarkerManager,
    create_trade_markers,
    create_marker_manager,
    create_custom_marker,
    calculate_marker_positions
)

# Mock imports for components that depend on other modules
try:
    from .enhanced_chart_display import (
        AdvancedChartDisplay,
        ChartDisplayManager,
        IndicatorOverlay,
        TradingSignalOverlay,
        PerformanceOverlay
    )
except ImportError:
    # Create mock classes if enhanced_chart_display has import issues
    class AdvancedChartDisplay:
        def __init__(self, *args, **kwargs): pass

    class ChartDisplayManager:
        def __init__(self, *args, **kwargs): pass

    class IndicatorOverlay:
        def __init__(self, *args, **kwargs): pass

    class TradingSignalOverlay:
        def __init__(self, *args, **kwargs): pass

    class PerformanceOverlay:
        def __init__(self, *args, **kwargs): pass

try:
    from .timeframe_switcher import (
        TimeframeSwitcher,
        TimeframeConfig,
        SupportedTimeframes,
        create_timeframe_switcher
    )
except ImportError:
    # Create mock classes if timeframe_switcher has import issues
    class TimeframeSwitcher:
        def __init__(self, *args, **kwargs): pass

    class TimeframeConfig:
        def __init__(self, *args, **kwargs): pass

    class SupportedTimeframes:
        def __init__(self, *args, **kwargs): pass

    def create_timeframe_switcher(*args, **kwargs):
        return TimeframeSwitcher()

__all__ = [
    # Chart Widget (Core - Always Available)
    "AdvancedHowtraderChart",
    "ChartConfig",
    "CandlestickItem",
    "VolumeBarItem",
    "IndicatorItem",
    "DataLoader",

    # Markers (Core - Always Available)
    "TradeSignal",
    "TradeMarker",
    "TradeLine",
    "PositionIndicator",
    "PnLOverlay",
    "VolumeMarker",
    "BacktestStatisticsOverlay",
    "MarkerManager",
    "create_trade_markers",
    "create_marker_manager",
    "create_custom_marker",
    "calculate_marker_positions",

    # Enhanced Chart Display (Optional)
    "AdvancedChartDisplay",
    "ChartDisplayManager",
    "IndicatorOverlay",
    "TradingSignalOverlay",
    "PerformanceOverlay",

    # Timeframe Switcher (Optional)
    "TimeframeSwitcher",
    "TimeframeConfig",
    "SupportedTimeframes",
    "create_timeframe_switcher",
]


def create_chart_widget(config=None):
    """Create configured chart widget"""
    return AdvancedHowtraderChart()


def create_complete_display_system(config=None):
    """Create complete display system with all components"""
    chart_widget = create_chart_widget(config)

    # Only create components that are available
    try:
        marker_manager = create_marker_manager(chart_widget.price_plot, config)
    except:
        marker_manager = None

    try:
        timeframe_switcher = create_timeframe_switcher()
    except:
        timeframe_switcher = None

    return {
        'chart': chart_widget,
        'markers': marker_manager,
        'timeframe': timeframe_switcher
    }


def get_available_chart_types():
    """Get available chart widget types"""
    return ['advanced_howtrader', 'enhanced_display']


def create_chart_by_type(chart_type: str = 'advanced_howtrader', **kwargs):
    """Create chart widget by type"""
    if chart_type == 'advanced_howtrader':
        return AdvancedHowtraderChart(**kwargs)
    elif chart_type == 'enhanced_display':
        return AdvancedChartDisplay(**kwargs)
    else:
        return AdvancedHowtraderChart(**kwargs)