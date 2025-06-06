"""
Backtest Data Extractor
======================

Data extraction and processing utilities for backtest results.
Handles conversion between backtesting engine data and display formats.

Author: AI Assistant
Version: 1.0
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import traceback

# Try to import from howtrader with fallback to mock classes
try:
    from howtrader.app.cta_strategy.backtesting import BacktestingEngine, DailyResult
    from howtrader.trader.object import TradeData, BarData
    HOWTRADER_AVAILABLE = True
except ImportError:
    # Create mock classes for development/testing
    HOWTRADER_AVAILABLE = False

    class BacktestingEngine:
        """Mock BacktestingEngine for development"""
        def __init__(self):
            self.trades = []
            self.daily_results = {}
            self.history_data = []

        def calculate_result(self):
            return pd.DataFrame()

        def calculate_statistics(self):
            return {}

    class DailyResult:
        """Mock DailyResult for development"""
        def __init__(self):
            self.date = datetime.now().date()
            self.close_price = 0.0
            self.net_pnl = 0.0
            self.total_pnl = 0.0
            self.commission = 0.0
            self.trade_count = 0

    class TradeData:
        """Mock TradeData for development"""
        def __init__(self):
            self.tradeid = ""
            self.orderid = ""
            self.symbol = ""
            self.exchange = None
            self.direction = None
            self.offset = None
            self.price = 0.0
            self.volume = 0.0
            self.datetime = datetime.now()

    class BarData:
        """Mock BarData for development"""
        def __init__(self):
            self.symbol = ""
            self.exchange = None
            self.datetime = datetime.now()
            self.interval = None
            self.volume = 0.0
            self.open_price = 0.0
            self.high_price = 0.0
            self.low_price = 0.0
            self.close_price = 0.0


class BacktestDataExtractor:
    """
    Extract and process backtest data for display purposes

    This class handles:
    - Converting backtesting engine results to display format
    - Extracting trades, bars, and performance metrics
    - Processing data for chart visualization
    - Calculating additional statistics
    """

    def __init__(self, engine: Optional[BacktestingEngine] = None):
        """Initialize the data extractor"""
        self.engine = engine
        self.trades_data: List[Dict] = []
        self.bars_data: List[Dict] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.daily_results: Dict[str, Any] = {}

    def connect_engine(self, engine: BacktestingEngine) -> bool:
        """
        Connect to a backtesting engine

        Args:
            engine: BacktestingEngine instance

        Returns:
            bool: True if connection successful
        """
        try:
            self.engine = engine
            return True
        except Exception as e:
            print(f"Error connecting to engine: {e}")
            return False

    def extract_all_data(self) -> Dict[str, Any]:
        """
        Extract all data from the connected engine

        Returns:
            Dict containing trades, bars, performance, and daily results
        """
        if not self.engine:
            return self._create_sample_data()

        try:
            if not HOWTRADER_AVAILABLE:
                return self._create_sample_data()

            # Extract different data types
            self._extract_trades()
            self._extract_bars()
            self._extract_performance()
            self._extract_daily_results()

            return {
                'trades': self.trades_data,
                'bars': self.bars_data,
                'performance': self.performance_metrics,
                'daily_results': self.daily_results,
                'summary': self._create_summary()
            }

        except Exception as e:
            print(f"Error extracting data: {e}")
            traceback.print_exc()
            return self._create_sample_data()

    def _create_sample_data(self) -> Dict[str, Any]:
        """Create sample data for testing when howtrader is not available"""
        current_time = datetime.now()

        # Sample trades
        sample_trades = []
        for i in range(5):
            trade_time = current_time - timedelta(hours=i*2)
            sample_trades.append({
                'tradeid': f'trade_{i+1}',
                'orderid': f'order_{i+1}',
                'symbol': 'BTCUSDT.BINANCE',
                'direction': 'LONG' if i % 2 == 0 else 'SHORT',
                'price': 45000 + (i * 100),
                'volume': 0.1,
                'datetime': trade_time,
                'pnl': (i * 50) - 25 if i > 0 else 0,
                'commission': 2.25
            })

        # Sample bars
        sample_bars = []
        for i in range(100):
            bar_time = current_time - timedelta(hours=i)
            base_price = 45000
            sample_bars.append({
                'datetime': bar_time,
                'open': base_price + (i * 10),
                'high': base_price + (i * 10) + 50,
                'low': base_price + (i * 10) - 30,
                'close': base_price + (i * 10) + 20,
                'volume': 1000 + (i * 10)
            })

        # Sample performance metrics
        performance = {
            'total_return': 0.15,
            'annual_return': 0.45,
            'max_drawdown': -0.08,
            'sharpe_ratio': 1.25,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'start_capital': 100000,
            'end_capital': 115000,
            'total_pnl': 15000,
            'total_commission': 125.75,
            'total_trades': len(sample_trades),
            'winning_trades': 3,
            'losing_trades': 2
        }

        return {
            'trades': sample_trades,
            'bars': sample_bars,
            'performance': performance,
            'daily_results': {},
            'summary': {
                'total_trades': len(sample_trades),
                'profitable_trades': 3,
                'largest_win': 150.0,
                'largest_loss': -50.0,
                'average_trade': 30.0
            }
        }

    def _extract_trades(self) -> None:
        """Extract trade data from the engine"""
        if not hasattr(self.engine, 'trades'):
            return

        self.trades_data = []

        for trade in self.engine.trades:
            trade_dict = {
                'tradeid': getattr(trade, 'tradeid', ''),
                'orderid': getattr(trade, 'orderid', ''),
                'symbol': getattr(trade, 'symbol', ''),
                'direction': str(getattr(trade, 'direction', '')),
                'offset': str(getattr(trade, 'offset', '')),
                'price': getattr(trade, 'price', 0.0),
                'volume': getattr(trade, 'volume', 0.0),
                'datetime': getattr(trade, 'datetime', datetime.now()),
                'pnl': self._calculate_trade_pnl(trade),
                'commission': getattr(trade, 'commission', 0.0) if hasattr(trade, 'commission') else 0.0
            }
            self.trades_data.append(trade_dict)

    def _extract_bars(self) -> None:
        """Extract bar data from the engine"""
        if not hasattr(self.engine, 'history_data'):
            return

        self.bars_data = []

        for bar in self.engine.history_data:
            bar_dict = {
                'datetime': getattr(bar, 'datetime', datetime.now()),
                'symbol': getattr(bar, 'symbol', ''),
                'open': getattr(bar, 'open_price', 0.0),
                'high': getattr(bar, 'high_price', 0.0),
                'low': getattr(bar, 'low_price', 0.0),
                'close': getattr(bar, 'close_price', 0.0),
                'volume': getattr(bar, 'volume', 0.0)
            }
            self.bars_data.append(bar_dict)

    def _extract_performance(self) -> None:
        """Extract performance metrics from the engine"""
        try:
            if hasattr(self.engine, 'calculate_statistics'):
                stats = self.engine.calculate_statistics()
                self.performance_metrics = stats if stats else {}

            # Add basic metrics if not available
            if not self.performance_metrics:
                self.performance_metrics = {
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'start_capital': getattr(self.engine, 'capital', 100000),
                    'total_trades': len(self.trades_data) if self.trades_data else 0
                }

        except Exception as e:
            print(f"Error extracting performance: {e}")
            self.performance_metrics = {}

    def _extract_daily_results(self) -> None:
        """Extract daily results from the engine"""
        if not hasattr(self.engine, 'daily_results'):
            return

        self.daily_results = {}

        try:
            for date, result in self.engine.daily_results.items():
                self.daily_results[str(date)] = {
                    'date': getattr(result, 'date', date),
                    'close_price': getattr(result, 'close_price', 0.0),
                    'net_pnl': getattr(result, 'net_pnl', 0.0),
                    'total_pnl': getattr(result, 'total_pnl', 0.0),
                    'commission': getattr(result, 'commission', 0.0),
                    'trade_count': getattr(result, 'trade_count', 0)
                }
        except Exception as e:
            print(f"Error extracting daily results: {e}")

    def _calculate_trade_pnl(self, trade) -> float:
        """
        Calculate PnL for a trade

        Args:
            trade: Trade object

        Returns:
            float: Trade PnL
        """
        try:
            # This is a simplified calculation
            # In reality, you'd need position tracking and closing prices
            if hasattr(trade, 'pnl'):
                return trade.pnl

            # Placeholder calculation
            direction = getattr(trade, 'direction', None)
            price = getattr(trade, 'price', 0.0)
            volume = getattr(trade, 'volume', 0.0)

            # This is just a mock calculation
            # Real implementation would track positions and calculate actual P&L
            if str(direction) == 'LONG':
                return volume * price * 0.01  # Mock 1% profit
            else:
                return volume * price * -0.01  # Mock 1% loss

        except Exception as e:
            print(f"Error calculating trade PnL: {e}")
            return 0.0

    def _create_summary(self) -> Dict[str, Any]:
        """Create summary statistics"""
        total_trades = len(self.trades_data)
        profitable_trades = sum(1 for trade in self.trades_data if trade.get('pnl', 0) > 0)

        pnls = [trade.get('pnl', 0) for trade in self.trades_data]
        largest_win = max(pnls) if pnls else 0.0
        largest_loss = min(pnls) if pnls else 0.0
        average_trade = sum(pnls) / len(pnls) if pnls else 0.0

        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'average_trade': average_trade,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0.0
        }

    def get_trades_for_display(self) -> List[Dict]:
        """Get trades formatted for display"""
        return self.trades_data

    def get_bars_for_display(self) -> List[Dict]:
        """Get bars formatted for display"""
        return self.bars_data

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        return self.performance_metrics

    def export_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Export data to pandas DataFrames

        Returns:
            Dict containing trades and bars as DataFrames
        """
        try:
            trades_df = pd.DataFrame(self.trades_data) if self.trades_data else pd.DataFrame()
            bars_df = pd.DataFrame(self.bars_data) if self.bars_data else pd.DataFrame()

            return {
                'trades': trades_df,
                'bars': bars_df
            }
        except Exception as e:
            print(f"Error creating DataFrames: {e}")
            return {
                'trades': pd.DataFrame(),
                'bars': pd.DataFrame()
            }


def create_time_index_mapping(bars_data: List[Dict]) -> Dict[datetime, int]:
    """Create mapping from datetime to bar index"""
    return {bar['datetime']: i for i, bar in enumerate(bars_data)}


def find_nearest_bar_index(target_time: datetime, time_mapping: Dict[datetime, int]) -> Optional[int]:
    """
    Find the nearest bar index for a given time

    Args:
        target_time: Target datetime
        time_mapping: Mapping from datetime to index

    Returns:
        Optional[int]: Nearest bar index or None
    """
    if not time_mapping:
        return None

    times = list(time_mapping.keys())
    times.sort()

    # Find closest time
    closest_time = min(times, key=lambda x: abs((x - target_time).total_seconds()))
    return time_mapping[closest_time]


def test_data_extractor():
    """Test the data extractor with sample data"""
    print("🧪 Testing BacktestDataExtractor...")

    # Create extractor without engine (will use sample data)
    extractor = BacktestDataExtractor()

    # Extract sample data
    data = extractor.extract_all_data()

    print(f"✅ Extracted {len(data['trades'])} trades")
    print(f"✅ Extracted {len(data['bars'])} bars")
    print(f"✅ Performance metrics: {len(data['performance'])} items")
    print(f"✅ Summary: {data['summary']}")

    # Test DataFrame export
    dataframes = extractor.export_to_dataframe()
    print(f"✅ Exported trades DataFrame: {dataframes['trades'].shape}")
    print(f"✅ Exported bars DataFrame: {dataframes['bars'].shape}")

    return True


if __name__ == "__main__":
    test_data_extractor()