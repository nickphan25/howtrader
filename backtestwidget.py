"""
🔗 Integration example for Advanced Trading Widget
"""
from examples.strategies.LongShortBalancedStrategy import LongShortBalancedStrategy
from howtrader.trader.object import Interval
from datetime import datetime
import pandas as pd
import sys

from advanced_trading_widget import AdvancedTradingWidget
from howtrader.app.cta_strategy.backtesting import BacktestingEngine

# Initialize Qt Application at module level
from PySide6.QtWidgets import QApplication
# from examples.strategies.atr_rsi_15min_strategy import  AtrRsi15MinStrategy  # 要导入你回测的策略，你自己开发的。
# from examples.strategies.simplesmc import SMCBasic
# from examples.strategies.fixed_trade_time_strategy import FixedTradeTimeStrategy
# from examples.strategies.LongShortBalancedStrategy import LongShortBalancedStrategy
from examples.strategies.SMCSWING import PureSMCStrategy
# Global application instance
app = None

def ensure_qt_application():
    """Ensure QApplication is initialized"""
    global app
    if app is None:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
    return app


# Modify backtesting.py to add show_widget method
class BacktestingEngineExtended(BacktestingEngine):
    """Extended backtesting engine with advanced widget support"""

    def show_advanced_widget(self):
        """Show advanced trading chart widget"""
        if not self.history_data:
            print("❌ No data available. Run backtesting first.")
            return

        # Ensure Qt application is initialized
        ensure_qt_application()

        # Prepare data
        print("🔄 Preparing widget data...")
        bars_df = self._prepare_widget_data()
        trades_data = self._prepare_trades_data()

        # Create and show widget
        print("🎯 Creating Advanced Trading Widget...")
        try:
            widget = AdvancedTradingWidget(
                bars_data=bars_df,
                trades_data=trades_data,
                symbol=self.vt_symbol,
                interval=self.interval.value
            )

            print("✅ Widget created successfully!")
            widget.show()
            print("✅ Widget displayed!")
            return widget
            
        except Exception as e:
            print(f"❌ Error creating widget: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_widget_data(self):
        """🔥 Prepare bar data for widget display"""
        if not self.history_data:
            print("❌ No history data available")
            return pd.DataFrame()

        print(f"🔄 Converting {len(self.history_data)} bars to DataFrame...")
        
        # Convert history_data to DataFrame
        data_list = []
        
        try:
            for i, bar_data in enumerate(self.history_data):
                if i % 10000 == 0:  # Progress indicator for large datasets
                    print(f"   Progress: {i}/{len(self.history_data)} bars processed...")
                    
                data_dict = {
                    'datetime': bar_data.datetime,
                    'open': float(bar_data.open_price),  # Ensure float conversion
                    'high': float(bar_data.high_price),
                    'low': float(bar_data.low_price),
                    'close': float(bar_data.close_price),
                    'volume': float(bar_data.volume),
                }
                data_list.append(data_dict)
            
            df = pd.DataFrame(data_list)
            
            # Set datetime as index if needed
            if not df.empty:
                df = df.set_index('datetime')
                # Reset index to make datetime a column for widget compatibility
                df = df.reset_index()
                
            print(f"📊 Prepared {len(df)} bars for widget display")
            return df
            
        except Exception as e:
            print(f"❌ Error preparing widget data: {e}")
            return pd.DataFrame()

    def _prepare_trades_data(self):
        """🔥 Prepare trades data for widget - CLEAN VERSION"""
        trades_data = []

        try:
            for trade_id, trade in self.trades.items():
                try:
                    if not hasattr(trade, 'datetime'):
                        continue

                    direction_value = trade.direction.value
                    offset_value = trade.offset.value

                    # Map direction and action correctly
                    if direction_value == 'Long' and offset_value == 'OPEN':
                        trade_direction = 'long'
                        action = 'open'
                    elif direction_value == 'Short' and offset_value == 'CLOSE':
                        trade_direction = 'long'  # This is closing a LONG position
                        action = 'close'
                    elif direction_value == 'Short' and offset_value == 'OPEN':
                        trade_direction = 'short'  # Opening a short position
                        action = 'open'
                    elif direction_value == 'Long' and offset_value == 'CLOSE':
                        trade_direction = 'short'  # Closing a short position
                        action = 'close'
                    else:
                        # Fallback for any other combinations
                        trade_direction = 'long' if direction_value == 'Long' else 'short'
                        action = 'open' if offset_value == 'OPEN' else 'close'

                    trade_data = {
                        'datetime': trade.datetime,
                        'price': float(trade.price),
                        'volume': float(trade.volume),
                        'direction': trade_direction,
                        'action': action,
                        'pnl': 0.0,
                        'trade_id': trade_id
                    }
                    trades_data.append(trade_data)

                except Exception as e:
                    continue

            print(f"📈 Prepared {len(trades_data)} trades for widget display")
            return trades_data

        except Exception as e:
            print(f"❌ Error preparing trades data: {e}")
            return []


# Usage in backtestdemo.py
def run_backtest_with_widget():
    """Run backtest and show advanced widget"""

    print("🚀 Initializing Backtesting Engine...")
    engine = BacktestingEngineExtended()
    engine.set_parameters(
        vt_symbol="BTCUSDT.BINANCE",
        interval=Interval.MINUTE,
        start=datetime(2025, 1, 1),
        end=datetime(2025, 4, 30),
        rate=4 / 10000,
        slippage=0,
        size=1,
        pricetick=0.01,
        capital=10000,
    )

    # engine.add_strategy(AtrRsi15MinStrategy, {})
    # engine.add_strategy(FixedTradeTimeStrategy, {})
    engine.add_strategy(PureSMCStrategy, {})
    print("📊 Loading data...")
    engine.load_data()
    
    print("🚀 Running backtesting...")
    engine.run_backtesting()
    
    print("📈 Calculating results...")
    df = engine.calculate_result()
    engine.calculate_statistics()

    # Debug: Check trades structure
    print(f"🔍 Trades type: {type(engine.trades)}")
    print(f"🔍 Number of trades: {len(engine.trades)}")
    if engine.trades:
        first_trade_id = next(iter(engine.trades))
        first_trade = engine.trades[first_trade_id]
        print(f"🔍 First trade type: {type(first_trade)}")

    # Show advanced widget
    print("🎯 Launching Advanced Trading Widget...")
    widget = engine.show_advanced_widget()

    return engine, widget


if __name__ == "__main__":
    try:
        # Ensure Qt application is initialized first
        app = ensure_qt_application()

        print("🎯 Starting Backtesting with Advanced Trading Widget...")
        engine, widget = run_backtest_with_widget()

        if widget is not None:
            print("✅ Backtesting completed successfully!")
            print("🎯 Advanced Trading Widget is now running!")

            # 🔥 THIS IS THE MISSING PART - Start the Qt event loop
            print("🔄 Starting Qt event loop...")
            app.exec()  # This keeps the application running and shows the widget
        else:
            print("❌ Widget creation failed!")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
