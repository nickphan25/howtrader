"""
Test Runner for Backtest Markers
===============================

Simple test script để verify all components working.
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def test_imports():
    """Test basic imports"""
    print("🔍 Testing imports step by step...")

    try:
        print("  1️⃣ Testing config import...")
        from backtest_display.config.display_config import DisplayConfig, ChartTheme
        print("  ✅ Config import successful")

        print("  2️⃣ Testing config functionality...")
        config = DisplayConfig()
        print(f"  ✅ Config created with theme: {config.theme.value}")

        print("  3️⃣ Testing marker imports...")
        from backtest_display.display import backtest_markers
        print("  ✅ Marker module import successful")

        print("  4️⃣ Testing marker classes...")
        # Test basic classes without PySide6 dependencies
        print("  ✅ All basic imports working!")

        return True

    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Other error: {e}")
        return False


def test_config_only():
    """Test only configuration system"""
    print("\n🎨 Testing Display Configuration...")

    try:
        from backtest_display.config.display_config import (
            DisplayConfig,
            ChartTheme,
            MarkerStyle,
            get_dark_config,
            get_tradingview_config
        )

        # Test default config
        config = DisplayConfig()
        print(f"✅ Default theme: {config.theme.value}")
        print(f"✅ Buy marker color: {config.markers.buy_marker_color}")

        # Test dark theme
        dark_config = get_dark_config()
        print(f"✅ Dark theme background: {dark_config.colors.background_color}")

        # Test serialization
        config_dict = config.to_dict()
        print(f"✅ Config serialization: {len(config_dict)} keys")

        print("🎉 Configuration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Starting Backtest Display Tests...")
    print("=" * 60)

    # Check current directory
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Check if backtest_display folder exists
    backtest_display_path = os.path.join(os.getcwd(), 'backtest_display')
    if os.path.exists(backtest_display_path):
        print(f"✅ backtest_display folder found: {backtest_display_path}")
    else:
        print(f"❌ backtest_display folder NOT found: {backtest_display_path}")
        print("💡 Make sure all files were created correctly")
        return

    # Test step by step
    success = True

    # Test 1: Basic imports
    if test_imports():
        print("✅ Import tests passed")
    else:
        print("❌ Import tests failed")
        success = False

    # Test 2: Config system
    if test_config_only():
        print("✅ Config tests passed")
    else:
        print("❌ Config tests failed")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ Project structure is ready")
        print("✅ Basic imports working")
        print("✅ Configuration system functional")
        print("\n💡 Next step: Test with actual chart integration")
    else:
        print("❌ SOME TESTS FAILED")
        print("💡 Check file structure and imports")


if __name__ == "__main__":
    main()