"""
Core Module
===========

Core functionality for backtest display system.

Components:
- BacktestBridge: Integration with backtesting engines
- DataExtractor: Data extraction and processing utilities

Author: AI Assistant
Version: 1.0
"""

from .backtest_bridge import BacktestBridge
from .data_extractor import BacktestDataExtractor

# Define mock classes that don't exist yet
class BacktestEngine:
    """Mock BacktestEngine for compatibility"""
    pass

class BacktestResults:
    """Mock BacktestResults for compatibility"""
    pass

class BacktestParameters:
    """Mock BacktestParameters for compatibility"""
    pass

class DataProcessor:
    """Mock DataProcessor for compatibility"""
    pass

class ResultsAnalyzer:
    """Mock ResultsAnalyzer for compatibility"""
    pass

class TradeAnalyzer:
    """Mock TradeAnalyzer for compatibility"""
    pass

class PerformanceCalculator:
    """Mock PerformanceCalculator for compatibility"""
    pass

__all__ = [
    "BacktestBridge",
    "BacktestEngine",
    "BacktestResults",
    "BacktestParameters",
    "BacktestDataExtractor",
    "DataProcessor",
    "ResultsAnalyzer",
    "TradeAnalyzer",
    "PerformanceCalculator"
]


def create_backtest_bridge():
    """Create configured backtest bridge"""
    return BacktestBridge()


def create_data_extractor():
    """Create configured data extractor"""
    return BacktestDataExtractor()


def setup_core_components():
    """Setup and return all core components"""
    return {
        'bridge': create_backtest_bridge(),
        'extractor': create_data_extractor()
    }