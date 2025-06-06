
#!/usr/bin/env python3
"""
Test Configuration Module
=========================

Test script to verify all configurations work properly.
"""

import sys
import os

# Add current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def main():
    """Direct config test"""
    print("🎨 Testing Config Directly...")
    print("=" * 40)

    try:
        # Direct import from file
        sys.path.append(os.path.join(current_dir, 'backtest_display', 'config'))
        import display_config

        print("✅ Direct import successful")

        # Test config creation
        config = display_config.DisplayConfig()
        print(f"✅ Config created: {config.theme.value}")

        # Test theme
        dark_config = display_config.get_dark_config()
        print(f"✅ Dark config: {dark_config.colors.background_color}")

        # Test the test function
        display_config.test_display_config()

        print("\n" + "=" * 40)
        print("🎉 DIRECT CONFIG TEST PASSED!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()