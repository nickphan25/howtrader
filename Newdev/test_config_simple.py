#!/usr/bin/env python3
"""
Test Configuration Module
=========================

Test script to verify all configurations work properly.
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_config_imports():
    """Test all configuration imports"""
    print("🧪 Testing Configuration Imports...")

    try:
        # Test basic imports
        from backtest_display.config.display_config import (
            DisplayConfig, ChartTheme, ChartStyle, MarkerType, MarkerStyle
        )
        print("✅ Basic config imports successful")

        # Test config creation
        config = DisplayConfig()
        print(f"✅ Default config created: {config.theme.value}")

        # Test marker type
        marker = MarkerType.BUY_SIGNAL
        print(f"✅ Marker type: {marker.value}")

        # Test chart style
        style = ChartStyle.CANDLESTICK
        print(f"✅ Chart style: {style.value}")

        # Test theme switching
        config.update_theme(ChartTheme.DARK)
        print(f"✅ Theme updated to: {config.theme.value}")

        return True

    except Exception as e:
        print(f"❌ Configuration import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_display_module():
    """Test display module imports"""
    print("\n🧪 Testing Display Module...")

    try:
        from backtest_display.display.backtest_markers import MarkerType, TradeMarker
        print("✅ Display markers import successful")

        return True

    except Exception as e:
        print(f"❌ Display module error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting Configuration Tests...")

    success = True

    # Test config
    if not test_config_imports():
        success = False

    # Test display
    if not test_display_module():
        success = False

    if success:
        print("\n🎉 All tests passed! Configuration is working correctly.")
        print("\n📋 Now you can run:")
        print("   python run_backtest_app.py")
        print("   python run_backtest_app.py --sample")
        print("   python run_backtest_app.py --test")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())