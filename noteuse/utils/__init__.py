"""
Utils Module
============

Utility functions and helpers for backtest display system.

Components:
- Data processing utilities
- Export helpers
- Mathematical calculations
- UI helpers

Author: AI Assistant
Version: 1.0
"""

# Data utilities
def format_number(value, precision=2):
    """Format number for display"""
    if value is None:
        return "N/A"
    if abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def format_percentage(value, precision=2):
    """Format percentage for display"""
    if value is None:
        return "N/A"
    return f"{value*100:.{precision}f}%"


def format_currency(value, currency="$", precision=2):
    """Format currency for display"""
    if value is None:
        return "N/A"
    return f"{currency}{format_number(value, precision)}"


# Time utilities
def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def parse_timeframe(timeframe_str):
    """Parse timeframe string into components"""
    import re

    if timeframe_str == "tick":
        return {"value": 1, "unit": "tick", "seconds": 0}

    match = re.match(r'(\d+)([smhdwM])', timeframe_str)
    if not match:
        return {"value": 1, "unit": "m", "seconds": 60}

    value, unit = int(match.group(1)), match.group(2)

    unit_seconds = {
        's': 1, 'm': 60, 'h': 3600,
        'd': 86400, 'w': 604800, 'M': 2592000
    }

    return {
        "value": value,
        "unit": unit,
        "seconds": value * unit_seconds.get(unit, 60)
    }


# Color utilities
def get_profit_color(value):
    """Get color based on profit/loss"""
    from PySide6.QtGui import QColor

    if value > 0:
        return QColor("#4CAF50")  # Green
    elif value < 0:
        return QColor("#f44336")  # Red
    else:
        return QColor("#9E9E9E")  # Gray


def interpolate_color(color1, color2, factor):
    """Interpolate between two colors"""
    from PySide6.QtGui import QColor

    if isinstance(color1, str):
        color1 = QColor(color1)
    if isinstance(color2, str):
        color2 = QColor(color2)

    r = int(color1.red() + (color2.red() - color1.red()) * factor)
    g = int(color1.green() + (color2.green() - color1.green()) * factor)
    b = int(color1.blue() + (color2.blue() - color1.blue()) * factor)

    return QColor(r, g, b)


# Math utilities
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    import numpy as np

    if len(returns) == 0:
        return 0

    excess_returns = np.array(returns) - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    import numpy as np

    if len(equity_curve) == 0:
        return 0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)


def calculate_win_rate(trades):
    """Calculate win rate from trades"""
    if len(trades) == 0:
        return 0

    profitable = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    return profitable / len(trades)


# Export utilities
def export_to_csv(data, filename):
    """Export data to CSV"""
    import pandas as pd

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    df.to_csv(filename, index=False)
    return filename


def export_to_excel(data_dict, filename):
    """Export multiple datasets to Excel"""
    import pandas as pd

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, data in data_dict.items():
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return filename


# UI utilities
def set_widget_style(widget, style_dict):
    """Apply style dictionary to widget"""
    style_str = ""
    for property_name, value in style_dict.items():
        style_str += f"{property_name}: {value}; "
    widget.setStyleSheet(style_str)


def create_color_button(color, size=(20, 20)):
    """Create colored button"""
    from PySide6.QtWidgets import QPushButton
    from PySide6.QtGui import QColor

    if isinstance(color, str):
        color = QColor(color)

    button = QPushButton()
    button.setFixedSize(*size)
    button.setStyleSheet(f"""
        QPushButton {{
            background-color: {color.name()};
            border: 1px solid #cccccc;
            border-radius: {min(size)//2}px;
        }}
        QPushButton:hover {{
            border: 2px solid #999999;
        }}
    """)

    return button


__all__ = [
    "format_number", "format_percentage", "format_currency",
    "format_duration", "parse_timeframe",
    "get_profit_color", "interpolate_color",
    "calculate_sharpe_ratio", "calculate_max_drawdown", "calculate_win_rate",
    "export_to_csv", "export_to_excel",
    "set_widget_style", "create_color_button"
]