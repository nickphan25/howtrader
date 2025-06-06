"""
Backtest Trade Markers and Visual Indicators
===========================================

Module tạo và quản lý visual markers cho backtest trade signals,
P&L lines, position indicators trên chart sử dụng PySide6.

Author: AI Assistant
Version: 1.0
"""

from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QPainter
from PySide6.QtWidgets import QGraphicsItem

# Fix import - use absolute import thay vì relative
from backtest_display.config.display_config import DisplayConfig, MarkerStyle


class TradeMarker(pg.ScatterPlotItem):
    """
    Custom marker cho trade entry/exit signals
    """

    def __init__(self, config: DisplayConfig):
        super().__init__()
        self.config = config
        self.signals = []
        self.marker_items = []

    def add_trade_signal(self,
                         timestamp: float,
                         price: float,
                         signal_type: str,
                         volume: float = 0,
                         trade_id: str = "",
                         tooltip_info: Dict = None):
        """
        Add trade signal marker

        Args:
            timestamp: Time index on chart
            price: Price level
            signal_type: 'buy' hoặc 'sell'
            volume: Trade volume
            trade_id: Unique trade ID
            tooltip_info: Additional info cho tooltip
        """

        # Determine marker properties
        if signal_type.lower() == 'buy':
            symbol = self.config.markers.buy_marker_symbol
            color = QColor(self.config.markers.buy_marker_color)
            size = self.config.markers.buy_marker_size
            brush = QBrush(color)
        else:  # sell
            symbol = self.config.markers.sell_marker_symbol
            color = QColor(self.config.markers.sell_marker_color)
            size = self.config.markers.sell_marker_size
            brush = QBrush(color)

        # Create marker data
        marker_data = {
            'pos': [timestamp, price],
            'symbol': symbol,
            'size': size,
            'brush': brush,
            'pen': QPen(QColor(self.config.markers.buy_marker_border_color),
                        self.config.markers.buy_marker_border_width),
            'signal_type': signal_type,
            'volume': volume,
            'trade_id': trade_id,
            'tooltip_info': tooltip_info or {}
        }

        self.signals.append(marker_data)
        self._update_display()

    def add_multiple_signals(self, signals: List[Dict]):
        """
        Add multiple trade signals at once

        Args:
            signals: List of signal dictionaries
        """
        for signal in signals:
            self.add_trade_signal(
                timestamp=signal.get('timestamp', 0),
                price=signal.get('price', 0),
                signal_type=signal.get('signal_type', 'buy'),
                volume=signal.get('volume', 0),
                trade_id=signal.get('trade_id', ''),
                tooltip_info=signal.get('tooltip_info', {})
            )

    def _update_display(self):
        """Update display với current trade data"""
        if not self.signals:
            return

        # Prepare data arrays
        pos = np.array([item['pos'] for item in self.signals])
        symbols = [item['symbol'] for item in self.signals]
        sizes = [item['size'] for item in self.signals]
        brushes = [item['brush'] for item in self.signals]
        pens = [item['pen'] for item in self.signals]

        # Update scatter plot
        self.setData(
            pos=pos,
            symbol=symbols,
            size=sizes,
            brush=brushes,
            pen=pens
        )

    def clear_markers(self):
        """Clear tất cả markers"""
        self.signals.clear()
        self.clear()

    def get_marker_at_position(self, timestamp: float, price: float, tolerance: float = 1.0) -> Optional[Dict]:
        """
        Find marker tại position với tolerance

        Args:
            timestamp: Time coordinate
            price: Price coordinate
            tolerance: Search tolerance

        Returns:
            Marker data hoặc None
        """
        for marker in self.signals:
            pos = marker['pos']
            if (abs(pos[0] - timestamp) < tolerance and
                    abs(pos[1] - price) < tolerance):
                return marker
        return None


class TradeLine(pg.PlotDataItem):
    """
    Line connecting trade entry và exit points
    """

    def __init__(self, config: DisplayConfig):
        super().__init__()
        self.config = config
        self.lines = []

    def add_trade_line(self,
                       entry_timestamp: float,
                       entry_price: float,
                       exit_timestamp: float,
                       exit_price: float,
                       pnl: float,
                       trade_id: str = ""):
        """
        Add line connecting entry và exit

        Args:
            entry_timestamp: Entry time
            entry_price: Entry price
            exit_timestamp: Exit time
            exit_price: Exit price
            pnl: Trade P&L
            trade_id: Trade identifier
        """

        # Determine line color based on P&L
        if pnl > 0:
            color = QColor(self.config.markers.profit_line_color)
        else:
            color = QColor(self.config.markers.loss_line_color)

        # Create line với dash style
        pen = QPen(color, self.config.markers.trade_line_width)
        if self.config.markers.trade_line_style == "dash":
            pen.setStyle(Qt.PenStyle.DashLine)
        elif self.config.markers.trade_line_style == "dot":
            pen.setStyle(Qt.PenStyle.DotLine)
        elif self.config.markers.trade_line_style == "dashdot":
            pen.setStyle(Qt.PenStyle.DashDotLine)
        else:
            pen.setStyle(Qt.PenStyle.SolidLine)

        # Create line
        x_data = [entry_timestamp, exit_timestamp]
        y_data = [entry_price, exit_price]

        line_item = pg.PlotDataItem(
            x_data, y_data,
            pen=pen,
            name=f"Trade_{trade_id}"
        )

        trade_line_data = {
            'line_item': line_item,
            'entry_time': entry_timestamp,
            'entry_price': entry_price,
            'exit_time': exit_timestamp,
            'exit_price': exit_price,
            'pnl': pnl,
            'trade_id': trade_id
        }

        self.lines.append(trade_line_data)

        return line_item

    def get_all_line_items(self) -> List[pg.PlotDataItem]:
        """Get tất cả line items để add vào chart"""
        return [line['line_item'] for line in self.lines]

    def clear_lines(self):
        """Clear tất cả trade lines"""
        self.lines.clear()


class PositionIndicator(pg.PlotDataItem):
    """
    Visual indicator cho current position
    """

    def __init__(self, config: DisplayConfig):
        super().__init__()
        self.config = config
        self.positions = []
        self.current_position = 0

    def update_position(self, timestamp: float, position: float, price: float):
        """
        Update current position

        Args:
            timestamp: Time coordinate
            position: Position size (positive = long, negative = short)
            price: Current price
        """
        self.current_position = position

        position_data = {
            'timestamp': timestamp,
            'position': position,
            'price': price
        }

        self.positions.append(position_data)
        self._update_display()

    def _update_display(self):
        """Update position visualization"""
        if not self.positions:
            return

        # Create position line data
        timestamps = [p['timestamp'] for p in self.positions]
        positions = [p['position'] for p in self.positions]

        # Normalize positions cho display (scale to reasonable range)
        if positions:
            max_pos = max(abs(p) for p in positions)
            if max_pos > 0:
                scale_factor = 0.1  # Adjust as needed
                scaled_positions = [p * scale_factor for p in positions]
            else:
                scaled_positions = positions
        else:
            scaled_positions = []

        # Update plot
        if timestamps and scaled_positions:
            color = (QColor(self.config.markers.long_position_color)
                     if self.current_position >= 0
                     else QColor(self.config.markers.short_position_color))

            pen = QPen(color, 2)

            self.setData(
                timestamps, scaled_positions,
                pen=pen,
                name="Position"
            )


class PnLOverlay(pg.PlotDataItem):
    """
    P&L overlay trên chart
    """

    def __init__(self, config: DisplayConfig):
        super().__init__()
        self.config = config
        self.pnl_points = []

    def add_pnl_point(self, timestamp: float, cumulative_pnl: float):
        """
        Add P&L data point

        Args:
            timestamp: Time coordinate
            cumulative_pnl: Cumulative P&L value
        """
        self.pnl_points.append({
            'timestamp': timestamp,
            'pnl': cumulative_pnl
        })

        self._update_display()

    def _update_display(self):
        """Update P&L visualization"""
        if len(self.pnl_points) < 2:
            return

        timestamps = [p['timestamp'] for p in self.pnl_points]
        pnl_values = [p['pnl'] for p in self.pnl_points]

        # Determine color based on current P&L
        current_pnl = pnl_values[-1] if pnl_values else 0
        color = (QColor(self.config.colors.profit_color)
                 if current_pnl >= 0
                 else QColor(self.config.colors.loss_color))

        pen = QPen(color, 2)

        self.setData(
            timestamps, pnl_values,
            pen=pen,
            name="Cumulative P&L"
        )

    def clear_pnl(self):
        """Clear P&L data"""
        self.pnl_points.clear()
        self.clear()


class VolumeMarker(pg.GraphicsObject):
    """
    Volume markers tại trade points
    """

    def __init__(self, config: DisplayConfig):
        super().__init__()
        self.config = config
        self.markers = []

    def add_volume_marker(self, timestamp: float, volume: float, signal_type: str):
        """
        Add volume marker

        Args:
            timestamp: Time coordinate
            volume: Trade volume
            signal_type: 'buy' hoặc 'sell'
        """
        color = (QColor(self.config.colors.buy_signal)
                 if signal_type.lower() == 'buy'
                 else QColor(self.config.colors.sell_signal))

        self.markers.append({
            'x': timestamp,
            'height': volume,
            'width': 0.8,
            'brush': QBrush(color),
            'signal_type': signal_type
        })

        self.update()

    def paint(self, painter: QPainter, option, widget):
        """Paint volume markers"""
        if not self.markers:
            return

        for marker in self.markers:
            # Set brush and pen
            painter.setBrush(marker['brush'])
            painter.setPen(QPen(marker['brush'].color(), 1))

            # Draw volume bar
            x = marker['x']
            height = marker['height']
            width = marker['width']

            # Simple rectangle for volume
            rect = QPointF(x - width/2, 0)
            painter.drawRect(
                int(rect.x()), int(rect.y()),
                int(width), int(height)
            )

    def boundingRect(self):
        """Return bounding rectangle"""
        from PySide6.QtCore import QRectF

        if not self.markers:
            return QRectF(0, 0, 1, 1)

        min_x = min(m['x'] - m['width']/2 for m in self.markers)
        max_x = max(m['x'] + m['width']/2 for m in self.markers)
        max_height = max(m['height'] for m in self.markers)

        return QRectF(min_x, 0, max_x - min_x, max_height)

    def clear(self):
        """Clear volume markers"""
        self.markers.clear()
        self.update()


class BacktestStatisticsOverlay(pg.GraphicsObject):
    """
    Statistics overlay cho backtest performance
    """

    def __init__(self, config: DisplayConfig):
        super().__init__()
        self.config = config
        self.stats_data = {}
        self.position = QPointF(10, 10)

    def update_statistics(self, stats: Dict[str, Any]):
        """
        Update statistics data

        Args:
            stats: Dictionary với performance statistics
        """
        self.stats_data = stats
        self.update()

    def paint(self, painter: QPainter, option, widget):
        """Paint statistics overlay"""
        if not self.stats_data:
            return

        # Set font và color
        painter.setPen(QPen(QColor(self.config.colors.text_primary), 1))
        painter.setFont(painter.font())

        # Background rectangle
        rect_width = 200
        rect_height = len(self.stats_data) * 20 + 20
        rect = QPointF(self.position.x(), self.position.y())

        # Semi-transparent background
        bg_color = QColor(self.config.colors.panel_background)
        bg_color.setAlpha(200)
        painter.fillRect(
            int(rect.x()), int(rect.y()),
            rect_width, rect_height,
            QBrush(bg_color)
        )

        # Draw border
        painter.setPen(QPen(QColor(self.config.colors.border_color), 1))
        painter.drawRect(
            int(rect.x()), int(rect.y()),
            rect_width, rect_height
        )

        # Draw statistics text
        y_offset = 20
        painter.setPen(QPen(QColor(self.config.colors.text_primary), 1))

        for key, value in self.stats_data.items():
            text = f"{key}: {value}"
            painter.drawText(
                int(rect.x() + 10),
                int(rect.y() + y_offset),
                text
            )
            y_offset += 20

    def boundingRect(self):
        """Return bounding rectangle"""
        from PySide6.QtCore import QRectF
        rect_width = 200
        rect_height = len(self.stats_data) * 20 + 20
        return QRectF(
            self.position.x(), self.position.y(),
            rect_width, rect_height
        )


class MarkerManager:
    """
    Manager class để coordinate tất cả markers
    """

    def __init__(self, config: DisplayConfig):
        self.config = config
        self.trade_markers = TradeMarker(config)
        self.trade_lines = TradeLine(config)
        self.position_indicator = PositionIndicator(config)
        self.pnl_overlay = PnLOverlay(config)
        self.volume_markers = VolumeMarker(config)
        self.stats_overlay = BacktestStatisticsOverlay(config)

        self.all_components = [
            self.trade_markers,
            self.position_indicator,
            self.pnl_overlay,
            self.volume_markers,
            self.stats_overlay
        ]

    def process_backtest_data(self, trades_data: List[Dict], bars_data: List[Dict]):
        """
        Process backtest data và create all markers

        Args:
            trades_data: List of trade dictionaries
            bars_data: List of OHLCV bar data
        """
        self.clear_all()

        # Create time mapping
        time_mapping = {bar['datetime']: i for i, bar in enumerate(bars_data)}

        # Process trades
        entry_trades = {}  # Track entry trades for pairing

        for trade in trades_data:
            trade_time = trade.get('datetime')
            if trade_time not in time_mapping:
                continue

            time_index = time_mapping[trade_time]
            price = trade.get('price', 0)
            volume = trade.get('volume', 0)
            direction = trade.get('direction', 'buy')
            trade_id = trade.get('trade_id', '')
            pnl = trade.get('pnl', 0)

            # Add trade marker
            signal_type = 'buy' if trade.get('is_long', True) else 'sell'
            self.trade_markers.add_trade_signal(
                timestamp=time_index,
                price=price,
                signal_type=signal_type,
                volume=volume,
                trade_id=trade_id,
                tooltip_info={
                    'pnl': pnl,
                    'direction': direction,
                    'datetime': trade_time
                }
            )

            # Add volume marker
            self.volume_markers.add_volume_marker(
                timestamp=time_index,
                volume=volume,
                signal_type=signal_type
            )

            # Track for trade line creation
            if signal_type == 'buy':
                entry_trades[trade_id] = {
                    'entry_time': time_index,
                    'entry_price': price
                }
            else:  # sell - try to pair với entry
                if trade_id in entry_trades:
                    entry_data = entry_trades[trade_id]

                    # Create trade line
                    self.trade_lines.add_trade_line(
                        entry_timestamp=entry_data['entry_time'],
                        entry_price=entry_data['entry_price'],
                        exit_timestamp=time_index,
                        exit_price=price,
                        pnl=pnl,
                        trade_id=trade_id
                    )

                    del entry_trades[trade_id]

            # Update position indicator
            position = trade.get('position', 0)
            self.position_indicator.update_position(time_index, position, price)

            # Update P&L overlay
            cumulative_pnl = trade.get('cumulative_pnl', 0)
            self.pnl_overlay.add_pnl_point(time_index, cumulative_pnl)

    def update_statistics(self, stats: Dict[str, Any]):
        """Update statistics overlay"""
        formatted_stats = {}

        for key, value in stats.items():
            if isinstance(value, float):
                if 'rate' in key.lower() or 'ratio' in key.lower():
                    formatted_stats[key] = f"{value:.2%}"
                elif 'pnl' in key.lower() or 'return' in key.lower():
                    formatted_stats[key] = f"${value:,.2f}"
                else:
                    formatted_stats[key] = f"{value:.2f}"
            else:
                formatted_stats[key] = str(value)

        self.stats_overlay.update_statistics(formatted_stats)

    def get_all_plot_items(self) -> List[pg.PlotDataItem]:
        """Get tất cả plot items để add vào chart"""
        items = [
            self.trade_markers,
            self.position_indicator,
            self.pnl_overlay,
            self.volume_markers,
            self.stats_overlay
        ]

        # Add trade lines
        items.extend(self.trade_lines.get_all_line_items())

        return items

    def clear_all(self):
        """Clear tất cả markers"""
        self.trade_markers.clear_markers()
        self.trade_lines.clear_lines()
        self.position_indicator.clear()
        self.pnl_overlay.clear_pnl()
        self.volume_markers.clear()
        # Stats overlay sẽ clear khi update với empty dict
        self.stats_overlay.update_statistics({})

    def update_config(self, new_config: DisplayConfig):
        """Update configuration cho tất cả components"""
        self.config = new_config

        # Update each component
        for component in self.all_components:
            component.config = new_config

        # Refresh displays
        self.trade_markers._update_display()
        self.position_indicator._update_display()
        self.pnl_overlay._update_display()
        self.volume_markers.update()


# Utility functions
def create_custom_marker(symbol: str, color: str, size: int) -> Dict:
    """
    Create custom marker specification

    Args:
        symbol: Marker symbol
        color: Marker color
        size: Marker size

    Returns:
        Marker specification dictionary
    """
    return {
        'symbol': symbol,
        'brush': QBrush(QColor(color)),
        'size': size,
        'pen': QPen(QColor('#ffffff'), 1)
    }


def calculate_marker_positions(bars_data: List[Dict], trades_data: List[Dict]) -> List[Tuple[int, float]]:
    """
    Calculate marker positions trên chart

    Args:
        bars_data: OHLCV data
        trades_data: Trade data

    Returns:
        List of (time_index, price) tuples
    """
    time_mapping = {bar['datetime']: i for i, bar in enumerate(bars_data)}
    positions = []

    for trade in trades_data:
        trade_time = trade.get('datetime')
        if trade_time in time_mapping:
            time_index = time_mapping[trade_time]
            price = trade.get('price', 0)
            positions.append((time_index, price))

    return positions


# Test function
def test_backtest_markers():
    """Test function for backtest markers"""
    print("📍 Testing Backtest Markers with PySide6...")

    from backtest_display.config.display_config import DisplayConfig

    # Create config
    config = DisplayConfig()
    print("✅ Config created")

    # Test marker manager
    manager = MarkerManager(config)
    print("✅ MarkerManager created")

    # Test trade marker
    manager.trade_markers.add_trade_signal(
        timestamp=10,
        price=100.0,
        signal_type='buy',
        volume=1.0,
        trade_id='test_1'
    )
    print("✅ Trade marker added")

    # Test trade line
    line_item = manager.trade_lines.add_trade_line(
        entry_timestamp=10,
        entry_price=100.0,
        exit_timestamp=20,
        exit_price=105.0,
        pnl=5.0,
        trade_id='test_1'
    )
    print("✅ Trade line added")

    # Test position indicator
    manager.position_indicator.update_position(15, 1.0, 102.0)
    print("✅ Position indicator updated")

    # Test P&L overlay
    manager.pnl_overlay.add_pnl_point(15, 2.0)
    print("✅ P&L overlay updated")

    # Test statistics overlay
    test_stats = {
        'Total Return': 0.15,
        'Win Rate': 0.65,
        'Total Trades': 25,
        'Max Drawdown': -0.08
    }
    manager.update_statistics(test_stats)
    print("✅ Statistics overlay updated")

    # Test clear all
    manager.clear_all()
    print("✅ All markers cleared")

    print("🎉 All PySide6 marker tests passed!")


if __name__ == "__main__":
    test_backtest_markers()