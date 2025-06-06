from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Type
import pandas as pd
import traceback

# Try to import howtrader components
try:
    from howtrader.app.cta_strategy.backtesting import BacktestingEngine as HowTraderBacktestingEngine
    from howtrader.trader.object import Interval as HowTraderInterval, Exchange as HowTraderExchange
    from howtrader.trader.database import get_database
    HOWTRADER_AVAILABLE = True
except ImportError as e:
    print(f"HowTrader not available: {e}")
    HOWTRADER_AVAILABLE = False

# Strategy mapping - try to import each strategy individually
STRATEGY_MAPPING = {}

# Try to import strategies from howtrader.app.cta_strategy.strategies
try:
    from howtrader.app.cta_strategy.strategies.future_neutral_grid_strategy import FutureNeutralGridStrategy
    STRATEGY_MAPPING['FutureNeutralGridStrategy'] = FutureNeutralGridStrategy
    print("Loaded FutureNeutralGridStrategy")
except ImportError:
    print("FutureNeutralGridStrategy not available")

try:
    from howtrader.app.cta_strategy.strategies.future_profit_grid_strategy import FutureProfitGridStrategy
    STRATEGY_MAPPING['FutureProfitGridStrategy'] = FutureProfitGridStrategy
    print("Loaded FutureProfitGridStrategy")
except ImportError:
    print("FutureProfitGridStrategy not available")

# Try to import from examples.strategies
try:
    from examples.strategies.atr_rsi_strategy import AtrRsiStrategy
    STRATEGY_MAPPING['AtrRsiStrategy'] = AtrRsiStrategy
    print("Loaded AtrRsiStrategy")
except ImportError:
    print("AtrRsiStrategy not available")

try:
    from examples.strategies.atr_rsi_15min_strategy import AtrRsi15MinStrategy
    STRATEGY_MAPPING['AtrRsi15MinStrategy'] = AtrRsi15MinStrategy
    print("Loaded AtrRsi15MinStrategy")
except ImportError:
    print("AtrRsi15MinStrategy not available")

try:
    from examples.strategies.boll_channel_strategy import BollChannelStrategy
    STRATEGY_MAPPING['BollChannelStrategy'] = BollChannelStrategy
    print("Loaded BollChannelStrategy")
except ImportError:
    print("BollChannelStrategy not available")

try:
    from examples.strategies.turtle_signal_strategy import TurtleSignalStrategy
    STRATEGY_MAPPING['TurtleSignalStrategy'] = TurtleSignalStrategy
    print("Loaded TurtleSignalStrategy")
except ImportError:
    print("TurtleSignalStrategy not available")

# Try alternative import paths
try:
    from strategies.atr_rsi_strategy import AtrRsiStrategy
    if 'AtrRsiStrategy' not in STRATEGY_MAPPING:
        STRATEGY_MAPPING['AtrRsiStrategy'] = AtrRsiStrategy
        print("Loaded AtrRsiStrategy from strategies")
except ImportError:
    pass

try:
    from strategies.atr_rsi_15min_strategy import AtrRsi15MinStrategy
    if 'AtrRsi15MinStrategy' not in STRATEGY_MAPPING:
        STRATEGY_MAPPING['AtrRsi15MinStrategy'] = AtrRsi15MinStrategy
        print("Loaded AtrRsi15MinStrategy from strategies")
except ImportError:
    pass

try:
    from strategies.boll_channel_strategy import BollChannelStrategy
    if 'BollChannelStrategy' not in STRATEGY_MAPPING:
        STRATEGY_MAPPING['BollChannelStrategy'] = BollChannelStrategy
        print("Loaded BollChannelStrategy from strategies")
except ImportError:
    pass

try:
    from strategies.turtle_signal_strategy import TurtleSignalStrategy
    if 'TurtleSignalStrategy' not in STRATEGY_MAPPING:
        STRATEGY_MAPPING['TurtleSignalStrategy'] = TurtleSignalStrategy
        print("Loaded TurtleSignalStrategy from strategies")
except ImportError:
    pass

print(f"Successfully loaded {len(STRATEGY_MAPPING)} strategies: {list(STRATEGY_MAPPING.keys())}")

# Fallback BacktestingEngine if howtrader is not available
if not HOWTRADER_AVAILABLE:
    class BacktestingEngine:
        def __init__(self):
            print("Using mock BacktestingEngine")

        def set_parameters(self, **kwargs):
            print(f"Mock set_parameters: {kwargs}")

        def add_strategy(self, strategy_class, settings):
            print(f"Mock add_strategy: {strategy_class.__name__}, {settings}")

        def load_data(self):
            print("Mock load_data")

        def run_backtesting(self):
            print("Mock run_backtesting")

        def calculate_result(self):
            print("Mock calculate_result")
            return pd.DataFrame()

        def calculate_statistics(self):
            print("Mock calculate_statistics")

        def show_chart(self):
            print("Mock show_chart")

# Exchange and Interval classes for compatibility
class Exchange:
    BINANCE = "BINANCE"
    OKX = "OKX"
    HUOBI = "HUOBI"

class Interval:
    MINUTE = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"

class StrategyInfo:
    def __init__(self, name: str, class_obj: Type, category: str = "Default",
                 description: str = "", complexity: str = "Medium",
                 features: List[str] = None, parameters: Dict = None):
        self.name = name
        self.class_obj = class_obj
        self.category = category
        self.description = description
        self.complexity = complexity
        self.features = features or []
        self.parameters = parameters or {}

    def get_display_name(self) -> str:
        """Get a user-friendly display name for the strategy"""
        # Convert CamelCase to readable format
        import re
        # Insert space before uppercase letters that follow lowercase letters
        readable_name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', self.name)
        return readable_name

class RealBacktestBridge:
    def __init__(self, config: Dict = None):
        """Initialize the backtest bridge with direct strategy mapping"""
        self.config = config or {}
        self.strategy_infos: Dict[str, StrategyInfo] = {}
        self.loaded_strategies: Dict[str, Type] = {}
        self.engine = None
        self.database = None
        self.progress_callback: Optional[Callable] = None
        self.results = None

        # Initialize components
        self._initialize_database()
        self._load_strategies_from_mapping()

    def _initialize_database(self):
        """Initialize database connection if available"""
        if HOWTRADER_AVAILABLE:
            try:
                self.database = get_database()
                print("Database initialized successfully")
            except Exception as e:
                print(f"Failed to initialize database: {e}")
                self.database = None
        else:
            print("Using mock database")
            self.database = None

    def _load_strategies_from_mapping(self):
        """Load strategies from direct mapping instead of scanning"""
        print("Loading strategies from direct mapping...")

        for strategy_name, strategy_class in STRATEGY_MAPPING.items():
            try:
                # Create strategy info
                strategy_info = StrategyInfo(
                    name=strategy_name,
                    class_obj=strategy_class,
                    category=self._get_strategy_category(strategy_name),
                    description=self._get_strategy_description(strategy_class),
                    complexity="Medium",
                    features=self._get_strategy_features(strategy_class),
                    parameters=self._get_strategy_parameters(strategy_class)
                )

                self.strategy_infos[strategy_name] = strategy_info
                self.loaded_strategies[strategy_name] = strategy_class

                print(f"Loaded strategy: {strategy_name}")

            except Exception as e:
                print(f"Failed to load strategy {strategy_name}: {e}")
                continue

        print(f"Total strategies loaded: {len(self.loaded_strategies)}")

    def _get_strategy_category(self, strategy_name: str) -> str:
        """Determine strategy category based on name"""
        name_lower = strategy_name.lower()
        if 'grid' in name_lower:
            return "Grid Trading"
        elif 'atr' in name_lower or 'rsi' in name_lower:
            return "Technical Analysis"
        elif 'boll' in name_lower or 'channel' in name_lower:
            return "Channel Trading"
        elif 'turtle' in name_lower:
            return "Trend Following"
        else:
            return "Other"

    def _get_strategy_description(self, strategy_class: Type) -> str:
        """Get strategy description from docstring or class name"""
        if hasattr(strategy_class, '__doc__') and strategy_class.__doc__:
            return strategy_class.__doc__.strip().split('\n')[0]
        return f"Strategy implementation: {strategy_class.__name__}"

    def _get_strategy_features(self, strategy_class: Type) -> List[str]:
        """Extract strategy features based on class attributes and methods"""
        features = []

        # Check for common strategy features
        if hasattr(strategy_class, 'parameters'):
            features.append("Configurable Parameters")

        # Add more feature detection logic as needed
        features.append("Backtesting Compatible")

        return features

    def _get_strategy_parameters(self, strategy_class: Type) -> Dict:
        """Extract default parameters from strategy class"""
        if hasattr(strategy_class, 'parameters'):
            return dict(strategy_class.parameters)
        return {}

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.loaded_strategies.keys())

    def get_strategies_by_category(self) -> Dict[str, List[str]]:
        """Get strategies organized by category"""
        categories = {}
        for name, info in self.strategy_infos.items():
            category = info.category
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        return categories

    def get_strategy_info(self, strategy_name: str) -> Optional[StrategyInfo]:
        """Get detailed information about a strategy"""
        return self.strategy_infos.get(strategy_name)

    def get_strategy_display_names(self) -> Dict[str, str]:
        """Get mapping of strategy names to display names"""
        return {
            name: info.get_display_name()
            for name, info in self.strategy_infos.items()
        }

    def get_strategy_class(self, strategy_name: str) -> Optional[Type]:
        """Get strategy class by name"""
        return self.loaded_strategies.get(strategy_name)

    def _convert_timeframe(self, timeframe: str) -> Any:
        """Convert timeframe string to appropriate interval object"""
        if HOWTRADER_AVAILABLE:
            timeframe_mapping = {
                "1m": HowTraderInterval.MINUTE,
                "5m": HowTraderInterval.MINUTE_5,
                "15m": HowTraderInterval.MINUTE_15,
                "30m": HowTraderInterval.MINUTE_30,
                "1h": HowTraderInterval.HOUR,
                "4h": HowTraderInterval.HOUR_4,
                "1d": HowTraderInterval.DAILY,
            }
            return timeframe_mapping.get(timeframe, HowTraderInterval.MINUTE)
        else:
            return timeframe

    def _get_exchange_from_symbol(self, symbol: str) -> str:
        """Extract exchange from symbol"""
        if "." in symbol:
            return symbol.split(".")[-1]
        return "BINANCE"  # default

    def run_backtest(self, strategy_name: str, symbol: str, timeframe: str,
                     start_date: datetime, end_date: datetime,
                     initial_capital: float = 1000000,
                     commission_rate: float = 0.0004,
                     slippage: float = 0, size: float = 1,
                     pricetick: float = 0.01, strategy_params: Dict = None) -> Dict:
        """Run backtest with specified parameters"""
        try:
            # Get strategy class
            strategy_class = self.get_strategy_class(strategy_name)
            if not strategy_class:
                raise ValueError(f"Strategy '{strategy_name}' not found")

            if HOWTRADER_AVAILABLE:
                return self._run_real_backtest(
                    strategy_class, symbol, timeframe, start_date, end_date,
                    initial_capital, commission_rate, slippage, size, pricetick,
                    strategy_params or {}
                )
            else:
                return self._run_mock_backtest(
                    strategy_class, symbol, timeframe, start_date, end_date,
                    initial_capital, commission_rate, slippage, size, pricetick,
                    strategy_params or {}
                )

        except Exception as e:
            print(f"Backtest failed: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    def _run_real_backtest(self, strategy_class: Type, symbol: str, timeframe: str,
                           start_date: datetime, end_date: datetime,
                           initial_capital: float, commission_rate: float,
                           slippage: float, size: float, pricetick: float,
                           strategy_params: Dict) -> Dict:
        """Run real backtest using HowTrader"""
        try:
            # Create engine
            self.engine = HowTraderBacktestingEngine()

            # Set parameters
            interval = self._convert_timeframe(timeframe)
            self.engine.set_parameters(
                vt_symbol=symbol,
                interval=interval,
                start=start_date,
                end=end_date,
                rate=commission_rate,
                slippage=slippage,
                size=size,
                pricetick=pricetick,
                capital=initial_capital,
            )

            # Add strategy
            self.engine.add_strategy(strategy_class, strategy_params)

            # Run backtest
            if self.progress_callback:
                self.progress_callback("Loading data...")

            self.engine.load_data()

            if self.progress_callback:
                self.progress_callback("Running backtest...")

            self.engine.run_backtesting()

            if self.progress_callback:
                self.progress_callback("Calculating results...")

            df = self.engine.calculate_result()
            stats = self.engine.calculate_statistics()

            if self.progress_callback:
                self.progress_callback("Backtest completed!")

            # Format results
            results = self._format_backtest_results(df, stats)
            self.results = results

            return results

        except Exception as e:
            error_msg = f"Real backtest failed: {e}"
            print(error_msg)
            traceback.print_exc()
            return {"error": error_msg}

    def _run_mock_backtest(self, strategy_class: Type, symbol: str, timeframe: str,
                           start_date: datetime, end_date: datetime,
                           initial_capital: float, commission_rate: float,
                           slippage: float, size: float, pricetick: float,
                           strategy_params: Dict) -> Dict:
        """Run mock backtest when HowTrader is not available"""
        print("Running mock backtest...")

        try:
            import numpy as np

            # Create mock engine
            self.engine = BacktestingEngine()

            # Set parameters
            self.engine.set_parameters(
                vt_symbol=symbol,
                interval=timeframe,
                start=start_date,
                end=end_date,
                rate=commission_rate,
                slippage=slippage,
                size=size,
                pricetick=pricetick,
                capital=initial_capital,
            )

            # Add strategy
            self.engine.add_strategy(strategy_class, strategy_params)

            # Simulate backtest steps
            if self.progress_callback:
                self.progress_callback("Loading mock data...")

            self.engine.load_data()

            if self.progress_callback:
                self.progress_callback("Running mock backtest...")

            self.engine.run_backtesting()

            if self.progress_callback:
                self.progress_callback("Calculating mock results...")

            # Generate mock results
            days = (end_date - start_date).days
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            # Generate some realistic mock data
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
            cumulative_returns = np.cumprod(1 + returns)
            balance = initial_capital * cumulative_returns

            df = pd.DataFrame({
                'date': dates[:len(balance)],
                'balance': balance,
                'return': returns[:len(balance)],
                'highlevel': balance * (1 + np.random.uniform(0, 0.05, len(balance))),
                'drawdown': np.random.uniform(-0.1, 0, len(balance)),
                'net_pnl': balance - initial_capital,
            })

            # Mock statistics
            total_return = (balance[-1] / initial_capital - 1) * 100
            annual_return = ((balance[-1] / initial_capital) ** (365 / days) - 1) * 100
            max_drawdown = np.min(df['drawdown']) * 100
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

            stats = {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trade_count': np.random.randint(50, 200),
                'winning_rate': np.random.uniform(0.4, 0.7),
            }

            if self.progress_callback:
                self.progress_callback("Mock backtest completed!")

            # Format results
            results = self._format_backtest_results(df, stats)
            self.results = results

            return results

        except Exception as e:
            error_msg = f"Mock backtest failed: {e}"
            print(error_msg)
            traceback.print_exc()
            return {"error": error_msg}

    def _format_backtest_results(self, df: pd.DataFrame, stats: Dict) -> Dict:
        """Format backtest results for consistent output"""
        try:
            # Ensure we have the required columns
            if df is not None and not df.empty:
                # Convert datetime column if it exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                # Calculate additional metrics if not present
                if 'balance' in df.columns and 'net_pnl' not in df.columns:
                    initial_balance = df['balance'].iloc[0] if len(df) > 0 else 1000000
                    df['net_pnl'] = df['balance'] - initial_balance

                # Convert DataFrame to dict for JSON serialization
                df_dict = df.to_dict('records')
            else:
                df_dict = []

            # Format statistics
            formatted_stats = {}
            if stats:
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        formatted_stats[key] = round(float(value), 4)
                    else:
                        formatted_stats[key] = value

            return {
                'success': True,
                'data': df_dict,
                'statistics': formatted_stats,
                'total_records': len(df_dict)
            }

        except Exception as e:
            print(f"Error formatting results: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'statistics': {},
                'total_records': 0
            }

    def show_chart(self):
        """Show backtest chart if engine is available"""
        if self.engine and HOWTRADER_AVAILABLE:
            try:
                self.engine.show_chart()
            except Exception as e:
                print(f"Failed to show chart: {e}")
        else:
            print("Chart display not available (using mock engine or HowTrader not installed)")

    def get_results(self) -> Optional[Dict]:
        """Get the latest backtest results"""
        return self.results

    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set callback function for progress updates"""
        self.progress_callback = callback

    def run_simple_backtest(self, strategy_name: str, **kwargs) -> Dict:
        """Simplified backtest interface"""
        default_params = {
            'symbol': 'BTCUSDT.BINANCE',
            'timeframe': '1m',
            'start_date': datetime(2025, 1, 1),
            'end_date': datetime(2025, 1, 31),
            'initial_capital': 1000000,
            'commission_rate': 0.0004,
            'slippage': 0,
            'size': 1,
            'pricetick': 0.01,
            'strategy_params': {}
        }

        # Update with provided parameters
        default_params.update(kwargs)

        return self.run_backtest(strategy_name, **default_params)

# Simple interface for backward compatibility
class BacktestBridge:
    def __init__(self):
        """Simple backtest bridge wrapper"""
        self.bridge = RealBacktestBridge()

    def run_quick_backtest(self, strategy_name: str, symbol: str = "BTCUSDT.BINANCE",
                           days: int = 30, initial_capital: float = 1000000) -> Dict:
        """Run a quick backtest with minimal parameters"""
        try:
            end_date = datetime.now()
            start_date = datetime(end_date.year, end_date.month, end_date.day - days)

            return self.bridge.run_backtest(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe="1m",
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
        except Exception as e:
            return {"error": str(e)}

# Test functions
def test_strategy_registry():
    """Test the strategy loading"""
    print("Testing strategy registry...")
    print(f"Available strategies: {list(STRATEGY_MAPPING.keys())}")
    return STRATEGY_MAPPING

def test_bridge():
    """Test the backtest bridge"""
    print("Testing backtest bridge...")

    bridge = RealBacktestBridge()

    print(f"Loaded strategies: {bridge.get_available_strategies()}")
    print(f"Strategies by category: {bridge.get_strategies_by_category()}")

    # Test getting strategy info
    strategies = bridge.get_available_strategies()
    if strategies:
        first_strategy = strategies[0]
        info = bridge.get_strategy_info(first_strategy)
        print(f"Strategy info for {first_strategy}: {info.description if info else 'None'}")

        # Test getting strategy class
        strategy_class = bridge.get_strategy_class(first_strategy)
        print(f"Strategy class for {first_strategy}: {strategy_class}")

    return bridge

if __name__ == "__main__":
    # Test the functionality
    test_strategy_registry()
    test_bridge()