"""
Minimal Test - Just Check if Files Exist
========================================
"""

import os

def check_files():
    """Check if all files exist"""
    base_path = os.path.dirname(os.path.abspath(__file__))

    files_to_check = [
        'backtest_display/__init__.py',
        'backtest_display/config/__init__.py',
        'backtest_display/config/display_config.py',
        'backtest_display/display/__init__.py',
        'backtest_display/display/backtest_markers.py',
        'backtest_display/core/__init__.py',
        'backtest_display/utils/__init__.py'
    ]

    print("📁 Checking file structure...")
    all_exist = True

    for file_path in files_to_check:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING!")
            all_exist = False

    return all_exist

def main():
    """Main test"""
    print("🔍 Minimal File Structure Check")
    print("=" * 40)

    if check_files():
        print("\n🎉 All files exist!")
        print("💡 Now trying import...")

        try:
            # Try importing just the config
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

            # Try step by step
            print("Step 1: Import main package...")
            import backtest_display
            print("✅ Main package imported")

            print("Step 2: Import config...")
            from backtest_display.config import display_config
            print("✅ Config module imported")

            print("Step 3: Create config...")
            config = display_config.DisplayConfig()
            print(f"✅ Config created: {config.theme.value}")

            print("\n🎉 SUCCESS! Package is working!")

        except Exception as e:
            print(f"❌ Import error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n❌ Some files are missing!")
        print("💡 Please create missing files first")

if __name__ == "__main__":
    main()