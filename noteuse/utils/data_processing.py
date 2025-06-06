"""
Data Processing Utilities
========================

Utility functions cho data processing và validation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd


class DataProcessor:
    """Data processing utilities"""

    @staticmethod
    def validate_trades_data(trades_data: List[Dict]) -> bool:
        """Validate trades data format"""
        required_fields = ['datetime', 'price', 'volume']

        for trade in trades_data:
            for field in required_fields:
                if field not in trade:
                    return False
        return True

    @staticmethod
    def validate_bars_data(bars_data: List[Dict]) -> bool:
        """Validate bars data format"""
        required_fields = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        for bar in bars_data:
            for field in required_fields:
                if field not in bar:
                    return False
        return True


class TradeDataValidator:
    """Trade data validation utilities"""

    def __init__(self):
        self.errors = []

    def validate(self, data: List[Dict]) -> bool:
        """Validate trade data"""
        self.errors.clear()

        if not data:
            self.errors.append("No data provided")
            return False

        for i, trade in enumerate(data):
            if not isinstance(trade, dict):
                self.errors.append(f"Trade {i}: Not a dictionary")
                continue

            # Validate required fields
            required = ['datetime', 'price', 'volume']
            for field in required:
                if field not in trade:
                    self.errors.append(f"Trade {i}: Missing field '{field}'")

        return len(self.errors) == 0

    def get_errors(self) -> List[str]:
        """Get validation errors"""
        return self.errors.copy()