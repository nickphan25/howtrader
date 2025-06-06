"""
Configuration Module
===================

Configuration management for backtest display system.

Components:
- DisplayConfig: Main configuration class
- ChartTheme: Theme definitions
- ChartStyle: Chart style options
- MarkerType: Marker type definitions
- MarkerStyle: Marker styling configuration
- PerformanceConfig: Performance analysis settings
- ExportConfig: Export settings
- Default settings and presets

Author: AI Assistant
Version: 1.0
"""

from .display_config import (
    DisplayConfig,
    ChartTheme,
    ChartStyle,
    MarkerType,
    MarkerStyle,
    ColorScheme,
    ChartDimensions,
    DisplaySettings,
    PerformanceConfig,
    ExportConfig,
    get_dark_config,
    get_light_config,
    get_tradingview_config,
    get_custom_config,
    get_scalping_config,
    get_swing_trading_config,
    get_analysis_config
)

__all__ = [
    "DisplayConfig",
    "ChartTheme",
    "ChartStyle",
    "MarkerType",
    "MarkerStyle",
    "ColorScheme",
    "ChartDimensions",
    "DisplaySettings",
    "PerformanceConfig",
    "ExportConfig",
    "get_dark_config",
    "get_light_config",
    "get_tradingview_config",
    "get_custom_config",
    "get_scalping_config",
    "get_swing_trading_config",
    "get_analysis_config"
]


def create_default_config():
    """Create default configuration"""
    return DisplayConfig()


def create_light_theme_config():
    """Create light theme configuration"""
    config = DisplayConfig()
    config.update_theme(ChartTheme.LIGHT)
    return config


def create_dark_theme_config():
    """Create dark theme configuration"""
    config = DisplayConfig()
    config.update_theme(ChartTheme.DARK)
    return config


def create_tradingview_config():
    """Create TradingView style configuration"""
    config = DisplayConfig()
    config.update_theme(ChartTheme.TRADINGVIEW)
    return config


def get_preset_configs():
    """Get all preset configurations"""
    return {
        'default': create_default_config(),
        'light': create_light_theme_config(),
        'dark': create_dark_theme_config(),
        'tradingview': create_tradingview_config(),
        'scalping': get_scalping_config(),
        'swing_trading': get_swing_trading_config(),
        'analysis': get_analysis_config()
    }


def get_available_themes():
    """Get list of available themes"""
    return [theme.value for theme in ChartTheme]


def get_available_chart_styles():
    """Get list of available chart styles"""
    return [style.value for style in ChartStyle]


def get_available_marker_types():
    """Get list of available marker types"""
    return [marker_type.value for marker_type in MarkerType]


# Quick access functions
def quick_dark():
    """Quick access to dark theme"""
    return get_dark_config()


def quick_light():
    """Quick access to light theme"""
    return get_light_config()


def quick_tradingview():
    """Quick access to TradingView theme"""
    return get_tradingview_config()


# Configuration factory
class ConfigFactory:
    """Factory class for creating configurations"""

    @staticmethod
    def create_config(theme_name: str = 'light', **kwargs) -> DisplayConfig:
        """Create configuration by theme name"""
        theme_map = {
            'light': ChartTheme.LIGHT,
            'dark': ChartTheme.DARK,
            'tradingview': ChartTheme.TRADINGVIEW,
            'custom': ChartTheme.CUSTOM
        }

        theme = theme_map.get(theme_name.lower(), ChartTheme.LIGHT)
        config = DisplayConfig(theme)

        # Apply custom overrides
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    @staticmethod
    def create_trading_style_config(style: str) -> DisplayConfig:
        """Create configuration optimized for trading style"""
        style_configs = {
            'scalping': get_scalping_config,
            'swing': get_swing_trading_config,
            'analysis': get_analysis_config,
            'default': create_default_config
        }

        config_func = style_configs.get(style.lower(), create_default_config)
        return config_func()


# Module test function
def test_config_module():
    """Test configuration module"""
    print("🧪 Testing Configuration Module...")

    # Test all themes
    themes = get_available_themes()
    print(f"✅ Available themes: {themes}")

    # Test all chart styles
    styles = get_available_chart_styles()
    print(f"✅ Available chart styles: {styles}")

    # Test all marker types
    markers = get_available_marker_types()
    print(f"✅ Available marker types: {markers}")

    # Test preset configs
    presets = get_preset_configs()
    print(f"✅ Preset configurations: {list(presets.keys())}")

    # Test factory
    factory_config = ConfigFactory.create_config('dark', buy_marker_color='#ff0000')
    print(f"✅ Factory config theme: {factory_config.theme.value}")

    trading_config = ConfigFactory.create_trading_style_config('scalping')
    print(f"✅ Trading style config: {trading_config.theme.value}")

    print("🎉 Configuration module test completed!")


if __name__ == "__main__":
    test_config_module()