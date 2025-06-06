"""
Display Configuration
====================

Configuration classes cho backtest display styling và behavior.

Author: AI Assistant
Version: 1.0
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class ChartTheme(Enum):
    """Chart theme options"""
    LIGHT = "light"
    DARK = "dark"
    TRADINGVIEW = "tradingview"
    CUSTOM = "custom"


class ChartStyle(Enum):
    """Chart style options"""
    CANDLESTICK = "candlestick"
    OHLC = "ohlc"
    LINE = "line"
    AREA = "area"
    RENKO = "renko"
    HEIKIN_ASHI = "heikin_ashi"


class MarkerType(Enum):
    """Marker type definitions"""
    BUY_SIGNAL = "buy_signal"
    SELL_SIGNAL = "sell_signal"
    ENTRY_LONG = "entry_long"
    EXIT_LONG = "exit_long"
    ENTRY_SHORT = "entry_short"
    EXIT_SHORT = "exit_short"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    CUSTOM = "custom"


@dataclass
class MarkerStyle:
    """Marker styling configuration"""

    # Buy markers
    buy_marker_symbol: str = 't1'  # Triangle up
    buy_marker_color: str = '#26a69a'  # Green
    buy_marker_size: int = 12
    buy_marker_border_color: str = '#ffffff'
    buy_marker_border_width: int = 1

    # Sell markers
    sell_marker_symbol: str = 't'  # Triangle down
    sell_marker_color: str = '#ef5350'  # Red
    sell_marker_size: int = 12
    sell_marker_border_color: str = '#ffffff'
    sell_marker_border_width: int = 1

    # Trade lines
    profit_line_color: str = '#26a69a'  # Green
    loss_line_color: str = '#ef5350'    # Red
    trade_line_width: int = 2
    trade_line_style: str = "dash"  # solid, dash, dot, dashdot

    # Position indicators
    long_position_color: str = '#26a69a'
    short_position_color: str = '#ef5350'
    position_line_width: int = 2

    # Stop loss markers
    stop_loss_color: str = '#ff9800'
    stop_loss_size: int = 10

    # Take profit markers
    take_profit_color: str = '#2196f3'
    take_profit_size: int = 10


@dataclass
class ColorScheme:
    """Color scheme configuration"""

    # Background colors
    background_color: str = '#ffffff'
    panel_background: str = '#f8fafc'
    border_color: str = '#e2e8f0'

    # Text colors
    text_primary: str = '#1e293b'
    text_secondary: str = '#475569'
    text_muted: str = '#94a3b8'

    # Signal colors
    buy_signal: str = '#26a69a'
    sell_signal: str = '#ef5350'

    # P&L colors
    profit_color: str = '#26a69a'
    loss_color: str = '#ef5350'

    # Chart colors
    grid_color: str = '#f1f5f9'
    axis_color: str = '#94a3b8'

    # Candle colors
    candle_up: str = '#26a69a'
    candle_down: str = '#ef5350'

    # Volume colors
    volume_up: str = '#26a69a80'  # Semi-transparent
    volume_down: str = '#ef535080'


@dataclass
class ChartDimensions:
    """Chart dimension settings"""

    window_width: int = 1200
    window_height: int = 800
    margin_left: int = 60
    margin_right: int = 60
    margin_top: int = 40
    margin_bottom: int = 40

    # Chart heights
    main_chart_height: int = 500
    indicator_chart_height: int = 150
    volume_chart_height: int = 100


@dataclass
class DisplaySettings:
    """General display settings"""

    # Animation settings
    enable_animations: bool = True
    animation_duration: int = 300

    # Tooltip settings
    show_tooltips: bool = True
    tooltip_delay: int = 500

    # Grid settings
    show_grid: bool = True
    grid_opacity: float = 0.3

    # Performance settings
    max_visible_bars: int = 1000
    auto_scale: bool = True

    # Export settings
    default_export_format: str = 'png'
    export_dpi: int = 300


@dataclass
class PerformanceConfig:
    """Performance analysis configuration"""

    # Metrics to display
    show_total_return: bool = True
    show_annual_return: bool = True
    show_sharpe_ratio: bool = True
    show_max_drawdown: bool = True
    show_win_rate: bool = True
    show_profit_factor: bool = True

    # Formatting
    decimal_places: int = 4
    percentage_format: bool = True

    # Analysis periods
    rolling_periods: list = None

    def __post_init__(self):
        if self.rolling_periods is None:
            self.rolling_periods = [30, 60, 90, 180, 365]


@dataclass
class ExportConfig:
    """Export configuration"""

    # File formats
    supported_formats: list = None
    default_format: str = 'xlsx'

    # Image export
    image_dpi: int = 300
    image_format: str = 'png'

    # Excel export
    include_charts: bool = True
    include_raw_data: bool = True

    # PDF export
    pdf_layout: str = 'portrait'  # portrait, landscape
    pdf_margins: float = 1.0  # inches

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['xlsx', 'csv', 'json', 'pdf', 'png', 'jpg']


class DisplayConfig:
    """
    Main display configuration class
    """

    def __init__(self, theme: ChartTheme = ChartTheme.LIGHT):
        self.theme = theme
        self.chart_style = ChartStyle.CANDLESTICK
        self.markers = MarkerStyle()
        self.colors = self._get_color_scheme(theme)
        self.dimensions = ChartDimensions()
        self.settings = DisplaySettings()
        self.performance = PerformanceConfig()
        self.export = ExportConfig()

    def _get_color_scheme(self, theme: ChartTheme) -> ColorScheme:
        """Get color scheme based on theme"""

        if theme == ChartTheme.DARK:
            return ColorScheme(
                background_color='#1e1e1e',
                panel_background='#2d2d2d',
                border_color='#404040',
                text_primary='#ffffff',
                text_secondary='#cccccc',
                text_muted='#888888',
                grid_color='#404040',
                axis_color='#666666',
                candle_up='#00c851',
                candle_down='#ff4444',
                volume_up='#00c85180',
                volume_down='#ff444480'
            )
        elif theme == ChartTheme.TRADINGVIEW:
            return ColorScheme(
                background_color='#ffffff',
                panel_background='#f8fafc',
                border_color='#e2e8f0',
                text_primary='#131722',
                text_secondary='#787b86',
                text_muted='#b2b5be',
                grid_color='#f0f3fa',
                axis_color='#b2b5be',
                candle_up='#26a69a',
                candle_down='#ef5350'
            )
        else:  # LIGHT or CUSTOM
            return ColorScheme()

    def update_theme(self, theme: ChartTheme):
        """Update theme and refresh colors"""
        self.theme = theme
        self.colors = self._get_color_scheme(theme)

    def update_chart_style(self, style: ChartStyle):
        """Update chart style"""
        self.chart_style = style

    def customize_markers(self, **kwargs):
        """Customize marker settings"""
        for key, value in kwargs.items():
            if hasattr(self.markers, key):
                setattr(self.markers, key, value)

    def customize_colors(self, **kwargs):
        """Customize color settings"""
        for key, value in kwargs.items():
            if hasattr(self.colors, key):
                setattr(self.colors, key, value)

    def customize_performance(self, **kwargs):
        """Customize performance settings"""
        for key, value in kwargs.items():
            if hasattr(self.performance, key):
                setattr(self.performance, key, value)

    def customize_export(self, **kwargs):
        """Customize export settings"""
        for key, value in kwargs.items():
            if hasattr(self.export, key):
                setattr(self.export, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'theme': self.theme.value,
            'chart_style': self.chart_style.value,
            'markers': self.markers.__dict__,
            'colors': self.colors.__dict__,
            'dimensions': self.dimensions.__dict__,
            'settings': self.settings.__dict__,
            'performance': self.performance.__dict__,
            'export': self.export.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DisplayConfig':
        """Create config from dictionary"""
        theme = ChartTheme(config_dict.get('theme', 'light'))
        config = cls(theme)

        # Update chart style
        if 'chart_style' in config_dict:
            config.chart_style = ChartStyle(config_dict['chart_style'])

        # Update markers
        if 'markers' in config_dict:
            for key, value in config_dict['markers'].items():
                if hasattr(config.markers, key):
                    setattr(config.markers, key, value)

        # Update colors
        if 'colors' in config_dict:
            for key, value in config_dict['colors'].items():
                if hasattr(config.colors, key):
                    setattr(config.colors, key, value)

        # Update dimensions
        if 'dimensions' in config_dict:
            for key, value in config_dict['dimensions'].items():
                if hasattr(config.dimensions, key):
                    setattr(config.dimensions, key, value)

        # Update settings
        if 'settings' in config_dict:
            for key, value in config_dict['settings'].items():
                if hasattr(config.settings, key):
                    setattr(config.settings, key, value)

        # Update performance
        if 'performance' in config_dict:
            for key, value in config_dict['performance'].items():
                if hasattr(config.performance, key):
                    setattr(config.performance, key, value)

        # Update export
        if 'export' in config_dict:
            for key, value in config_dict['export'].items():
                if hasattr(config.export, key):
                    setattr(config.export, key, value)

        return config

    def get_marker_style_for_type(self, marker_type: MarkerType) -> Dict[str, Any]:
        """Get marker style for specific marker type"""
        base_style = {
            'size': self.markers.buy_marker_size,
            'border_color': self.markers.buy_marker_border_color,
            'border_width': self.markers.buy_marker_border_width
        }

        if marker_type in [MarkerType.BUY_SIGNAL, MarkerType.ENTRY_LONG]:
            base_style.update({
                'symbol': self.markers.buy_marker_symbol,
                'color': self.markers.buy_marker_color,
                'size': self.markers.buy_marker_size
            })
        elif marker_type in [MarkerType.SELL_SIGNAL, MarkerType.EXIT_LONG, MarkerType.EXIT_SHORT]:
            base_style.update({
                'symbol': self.markers.sell_marker_symbol,
                'color': self.markers.sell_marker_color,
                'size': self.markers.sell_marker_size
            })
        elif marker_type == MarkerType.ENTRY_SHORT:
            base_style.update({
                'symbol': self.markers.sell_marker_symbol,
                'color': self.markers.short_position_color,
                'size': self.markers.sell_marker_size
            })
        elif marker_type == MarkerType.STOP_LOSS:
            base_style.update({
                'symbol': 'x',
                'color': self.markers.stop_loss_color,
                'size': self.markers.stop_loss_size
            })
        elif marker_type == MarkerType.TAKE_PROFIT:
            base_style.update({
                'symbol': 'star',
                'color': self.markers.take_profit_color,
                'size': self.markers.take_profit_size
            })

        return base_style


# Predefined configurations
def get_dark_config() -> DisplayConfig:
    """Get dark theme configuration"""
    return DisplayConfig(ChartTheme.DARK)


def get_light_config() -> DisplayConfig:
    """Get light theme configuration"""
    return DisplayConfig(ChartTheme.LIGHT)


def get_tradingview_config() -> DisplayConfig:
    """Get TradingView style configuration"""
    return DisplayConfig(ChartTheme.TRADINGVIEW)


def get_custom_config(**kwargs) -> DisplayConfig:
    """Get custom configuration with overrides"""
    config = DisplayConfig(ChartTheme.CUSTOM)

    # Apply custom settings
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.markers, key):
            setattr(config.markers, key, value)
        elif hasattr(config.colors, key):
            setattr(config.colors, key, value)
        elif hasattr(config.performance, key):
            setattr(config.performance, key, value)
        elif hasattr(config.export, key):
            setattr(config.export, key, value)

    return config


# Preset configurations for different trading styles
def get_scalping_config() -> DisplayConfig:
    """Configuration optimized for scalping strategies"""
    config = DisplayConfig(ChartTheme.DARK)
    config.customize_markers(
        buy_marker_size=8,
        sell_marker_size=8,
        trade_line_width=1
    )
    config.settings.max_visible_bars = 500
    return config


def get_swing_trading_config() -> DisplayConfig:
    """Configuration optimized for swing trading strategies"""
    config = DisplayConfig(ChartTheme.LIGHT)
    config.customize_markers(
        buy_marker_size=14,
        sell_marker_size=14,
        trade_line_width=3
    )
    config.settings.max_visible_bars = 2000
    return config


def get_analysis_config() -> DisplayConfig:
    """Configuration optimized for detailed analysis"""
    config = DisplayConfig(ChartTheme.TRADINGVIEW)
    config.customize_performance(
        decimal_places=6,
        rolling_periods=[7, 14, 30, 60, 90, 180, 365]
    )
    config.settings.show_tooltips = True
    config.settings.show_grid = True
    return config


# Test function
def test_display_config():
    """Test display configuration"""
    print("🎨 Testing Display Configuration...")

    # Test default config
    config = DisplayConfig()
    print(f"✅ Default theme: {config.theme.value}")
    print(f"✅ Buy marker color: {config.markers.buy_marker_color}")

    # Test dark theme
    dark_config = get_dark_config()
    print(f"✅ Dark theme background: {dark_config.colors.background_color}")

    # Test custom config
    custom_config = get_custom_config(
        buy_marker_color='#00ff00',
        background_color='#123456'
    )
    print(f"✅ Custom buy marker: {custom_config.markers.buy_marker_color}")

    # Test marker style retrieval
    marker_style = config.get_marker_style_for_type(MarkerType.BUY_SIGNAL)
    print(f"✅ Buy signal style: {marker_style['color']}")

    # Test serialization
    config_dict = config.to_dict()
    restored_config = DisplayConfig.from_dict(config_dict)
    print(f"✅ Serialization test: {restored_config.theme.value}")

    # Test preset configs
    scalping_config = get_scalping_config()
    print(f"✅ Scalping config theme: {scalping_config.theme.value}")

    swing_config = get_swing_trading_config()
    print(f"✅ Swing trading config marker size: {swing_config.markers.buy_marker_size}")

    analysis_config = get_analysis_config()
    print(f"✅ Analysis config decimal places: {analysis_config.performance.decimal_places}")

    print("🎉 All display config tests passed!")


if __name__ == "__main__":
    test_display_config()