"""
Backtest Markers Module
======================

Trading signal markers và position indicators cho backtest visualization.
"""

import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

# Fixed imports - sử dụng direct imports thay vì relative
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config.display_config import DisplayConfig, MarkerType, MarkerStyle
except ImportError:
    # Fallback to simplified types if config not available
    class MarkerType:
        BUY_SIGNAL = "buy_signal"
        SELL_SIGNAL = "sell_signal"
        POSITION_OPEN = "position_open"
        POSITION_CLOSE = "position_close"
        STOP_LOSS = "stop_loss"
        TAKE_PROFIT = "take_profit"

    class MarkerStyle:
        CIRCLE = "circle"
        TRIANGLE = "triangle"
        ARROW = "arrow"

    class DisplayConfig:
        def __init__(self):
            self.theme = None
            self.markers = type('obj', (object,), {
                'buy_marker_color': '#00FF00',
                'sell_marker_color': '#FF0000',
                'buy_marker_size': 12,
                'sell_marker_size': 12,
                'marker_opacity': 0.8
            })()


@dataclass
class TradeSignal:
    """Data class cho trading signals"""
    timestamp: datetime
    price: float
    signal_type: str  # "BUY", "SELL", "STOP_LOSS", etc.
    volume: Optional[float] = None
    trade_id: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TradeMarker:
    """Quản lý trading markers trên chart"""

    def __init__(self, plot_widget, config: Optional[DisplayConfig] = None):
        self.plot_widget = plot_widget
        self.config = config or DisplayConfig()
        self.signals: List[TradeSignal] = []
        self.marker_items: List[pg.ScatterPlotItem] = []

    def add_trade_signal(self, signal: TradeSignal):
        """Thêm trade signal và cập nhật display"""
        self.signals.append(signal)

        # Convert timestamp to x-coordinate (implement based on your time axis)
        x_pos = self._timestamp_to_x(signal.timestamp)
        y_pos = signal.price

        # Determine marker properties based on signal type
        marker_props = self._get_marker_properties(signal.signal_type)

        # Create scatter plot item
        scatter = pg.ScatterPlotItem(
            pos=[(x_pos, y_pos)],
            size=marker_props['size'],
            brush=pg.mkBrush(marker_props['color']),
            pen=pg.mkPen(marker_props['border_color'], width=2),
            symbol=marker_props['symbol']
        )

        # Add to plot
        self.plot_widget.addItem(scatter)
        self.marker_items.append(scatter)

        # Add tooltip if metadata available
        if signal.metadata:
            tooltip_text = f"""
            Signal: {signal.signal_type}
            Price: ${signal.price:.2f}
            Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
            Volume: {signal.volume or 'N/A'}
            """
            scatter.setToolTip(tooltip_text.strip())

    def _timestamp_to_x(self, timestamp: datetime) -> float:
        """Convert timestamp to x-coordinate"""
        # Implement based on your chart's time axis
        # This is a simplified example
        return timestamp.timestamp()

    def _get_marker_properties(self, signal_type: str) -> Dict[str, Any]:
        """Get marker properties based on signal type"""
        base_props = {
            'size': self.config.markers.buy_marker_size,
            'border_color': '#FFFFFF'
        }

        if signal_type.upper() == "BUY":
            return {
                **base_props,
                'color': self.config.markers.buy_marker_color,
                'symbol': 't1'  # Triangle up
            }
        elif signal_type.upper() == "SELL":
            return {
                **base_props,
                'color': self.config.markers.sell_marker_color,
                'symbol': 't3'  # Triangle down
            }
        elif signal_type.upper() == "STOP_LOSS":
            return {
                **base_props,
                'color': '#FF4444',
                'symbol': 'x'
            }
        elif signal_type.upper() == "TAKE_PROFIT":
            return {
                **base_props,
                'color': '#44FF44',
                'symbol': 'star'
            }
        else:
            return {
                **base_props,
                'color': '#CCCCCC',
                'symbol': 'o'
            }

    def add_multiple_signals(self, signals: List[TradeSignal]):
        """Thêm nhiều signals cùng lúc"""
        for signal in signals:
            self.add_trade_signal(signal)
        self._update_display()

    def _update_display(self):
        """Update display after adding multiple items"""
        # Force redraw
        self.plot_widget.update()

    def clear_markers(self):
        """Clear all markers"""
        for item in self.marker_items:
            self.plot_widget.removeItem(item)
        self.marker_items.clear()
        self.signals.clear()

    def get_marker_at_position(self, x: float, y: float, tolerance: float = 5.0) -> Optional[TradeSignal]:
        """Get marker at given position"""
        for signal in self.signals:
            signal_x = self._timestamp_to_x(signal.timestamp)
            signal_y = signal.price

            if abs(signal_x - x) <= tolerance and abs(signal_y - y) <= tolerance:
                return signal
        return None


class TradeLine:
    """Vẽ các đường nối giữa entry và exit positions"""

    def __init__(self, plot_widget, config: Optional[DisplayConfig] = None):
        self.plot_widget = plot_widget
        self.config = config or DisplayConfig()
        self.lines: List[pg.PlotDataItem] = []

    def add_trade_line(self, entry_signal: TradeSignal, exit_signal: TradeSignal,
                       profit_loss: Optional[float] = None):
        """Thêm đường nối giữa entry và exit"""

        entry_x = self._timestamp_to_x(entry_signal.timestamp)
        exit_x = self._timestamp_to_x(exit_signal.timestamp)

        x_data = [entry_x, exit_x]
        y_data = [entry_signal.price, exit_signal.price]

        # Determine line color based on profit/loss
        if profit_loss is not None:
            color = '#00FF00' if profit_loss > 0 else '#FF0000'
        else:
            color = '#888888'

        # Create line
        line = pg.PlotDataItem(
            x=x_data,
            y=y_data,
            pen=pg.mkPen(color, width=2, style=pg.QtCore.Qt.DashLine)
        )

        self.plot_widget.addItem(line)
        self.lines.append(line)

    def _timestamp_to_x(self, timestamp: datetime) -> float:
        """Convert timestamp to x-coordinate"""
        return timestamp.timestamp()

    def _get_line_style(self, profit_loss: Optional[float]) -> Dict[str, Any]:
        """Get line style based on profit/loss"""
        if profit_loss is None:
            return {'color': '#888888', 'width': 1}
        elif profit_loss > 0:
            return {'color': '#00FF00', 'width': 2}
        else:
            return {'color': '#FF0000', 'width': 2}

    def get_all_line_items(self) -> List[pg.PlotDataItem]:
        """Get all line items"""
        return self.lines.copy()

    def clear_lines(self):
        """Clear all lines"""
        for line in self.lines:
            self.plot_widget.removeItem(line)
        self.lines.clear()


class PositionIndicator:
    """Hiển thị position indicators"""

    def __init__(self, plot_widget, config: Optional[DisplayConfig] = None):
        self.plot_widget = plot_widget
        self.config = config or DisplayConfig()
        self.positions: List[Dict] = []
        self.current_position: Optional[pg.PlotDataItem] = None

    def update_position(self, timestamp: datetime, price: float,
                        position_size: float, position_type: str = "LONG"):
        """Update current position indicator"""

        # Remove previous position indicator
        if self.current_position:
            self.plot_widget.removeItem(self.current_position)

        if position_size != 0:
            # Add new position indicator
            x_pos = timestamp.timestamp()
            color = '#00AA00' if position_type == "LONG" else '#AA0000'

            self.current_position = pg.PlotDataItem(
                x=[x_pos],
                y=[price],
                pen=None,
                symbol='s',  # Square
                symbolBrush=pg.mkBrush(color),
                symbolSize=10
            )

            self.plot_widget.addItem(self.current_position)

        # Store position data
        self.positions.append({
            'timestamp': timestamp,
            'price': price,
            'size': position_size,
            'type': position_type
        })

    def _update_display(self):
        """Update position display"""
        # Position indicators are updated in real-time in update_position
        pass


class PnLOverlay:
    """Overlay hiển thị PnL information"""

    def __init__(self, plot_widget, config: Optional[DisplayConfig] = None):
        self.plot_widget = plot_widget
        self.config = config or DisplayConfig()
        self.pnl_points: List[Dict] = []

    def add_pnl_point(self, timestamp: datetime, price: float, pnl: float):
        """Thêm PnL point"""
        self.pnl_points.append({
            'timestamp': timestamp,
            'price': price,
            'pnl': pnl
        })
        self._update_display()

    def _update_display(self):
        """Update PnL display"""
        # Create text items for PnL values
        for point in self.pnl_points[-10:]:  # Show last 10 points
            text_item = pg.TextItem(
                text=f"${point['pnl']:.2f}",
                color=(0, 255, 0) if point['pnl'] > 0 else (255, 0, 0),
                anchor=(0.5, 1.0)
            )

            x_pos = point['timestamp'].timestamp()
            text_item.setPos(x_pos, point['price'])
            self.plot_widget.addItem(text_item)

    def clear_pnl(self):
        """Clear PnL overlay"""
        self.pnl_points.clear()


class VolumeMarker(pg.GraphicsObject):
    """Custom volume markers"""

    def __init__(self, volume_data: List[Dict], config: Optional[DisplayConfig] = None):
        super().__init__()
        self.config = config or DisplayConfig()
        self.volume_data = volume_data
        self.markers: List[Dict] = []

    def add_volume_marker(self, timestamp: datetime, price: float, volume: float):
        """Add volume marker"""
        self.markers.append({
            'timestamp': timestamp,
            'price': price,
            'volume': volume
        })
        self.update()

    def paint(self, painter: QPainter, option, widget):
        """Paint volume markers"""
        painter.setRenderHint(QPainter.Antialiasing)

        for marker in self.markers:
            x_pos = marker['timestamp'].timestamp()
            y_pos = marker['price']

            # Size based on volume (normalized)
            size = min(max(marker['volume'] / 1000, 3), 15)

            # Color based on volume level
            if marker['volume'] > 10000:
                color = QColor(255, 0, 0, 128)  # Red for high volume
            elif marker['volume'] > 5000:
                color = QColor(255, 255, 0, 128)  # Yellow for medium volume
            else:
                color = QColor(0, 255, 0, 128)  # Green for low volume

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(0, 0, 0), 1))

            # Draw circle
            painter.drawEllipse(
                int(x_pos - size/2),
                int(y_pos - size/2),
                int(size),
                int(size)
            )

    def boundingRect(self) -> QRectF:
        """Return bounding rectangle"""
        if not self.markers:
            return QRectF()

        min_x = min(m['timestamp'].timestamp() for m in self.markers)
        max_x = max(m['timestamp'].timestamp() for m in self.markers)
        min_y = min(m['price'] for m in self.markers)
        max_y = max(m['price'] for m in self.markers)

        return QRectF(min_x - 10, min_y - 10, max_x - min_x + 20, max_y - min_y + 20)

    def clear(self):
        """Clear all markers"""
        self.markers.clear()
        self.update()


class BacktestStatisticsOverlay(pg.GraphicsObject):
    """Overlay hiển thị backtest statistics"""

    def __init__(self, plot_widget, config: Optional[DisplayConfig] = None):
        super().__init__()
        self.plot_widget = plot_widget
        self.config = config or DisplayConfig()
        self.stats_data: Dict[str, Any] = {}
        self.position = (10, 10)  # Top-left position

    def update_statistics(self, stats: Dict[str, Any]):
        """Update statistics data"""
        self.stats_data = stats
        self.update()

    def paint(self, painter: QPainter, option, widget):
        """Paint statistics overlay"""
        if not self.stats_data:
            return

        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        rect = QRectF(self.position[0], self.position[1], 200, 150)
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawRect(rect)

        # Text
        painter.setPen(QPen(QColor(255, 255, 255)))
        y_offset = self.position[1] + 20

        for key, value in self.stats_data.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"

            painter.drawText(self.position[0] + 10, y_offset, text)
            y_offset += 15

    def boundingRect(self) -> QRectF:
        """Return bounding rectangle"""
        return QRectF(self.position[0], self.position[1], 200, 150)


class MarkerManager:
    """Central manager cho tất cả markers và overlays"""

    def __init__(self, plot_widget, config: Optional[DisplayConfig] = None):
        self.plot_widget = plot_widget
        self.config = config or DisplayConfig()

        # Initialize all components
        self.trade_markers = TradeMarker(plot_widget, config)
        self.trade_lines = TradeLine(plot_widget, config)
        self.position_indicator = PositionIndicator(plot_widget, config)
        self.pnl_overlay = PnLOverlay(plot_widget, config)
        self.volume_markers = VolumeMarker([], config)
        self.stats_overlay = BacktestStatisticsOverlay(plot_widget, config)

        # Add volume markers and stats overlay to plot
        self.plot_widget.addItem(self.volume_markers)
        self.plot_widget.addItem(self.stats_overlay)

        # Store all components for batch operations
        self.all_components = [
            self.trade_markers,
            self.trade_lines,
            self.position_indicator,
            self.pnl_overlay,
            self.volume_markers,
            self.stats_overlay
        ]

    def process_backtest_data(self, backtest_data: Dict[str, Any]):
        """Process complete backtest data và create all markers"""

        # Clear existing data
        self.clear_all()

        # Process trades
        if 'trades' in backtest_data:
            trades = backtest_data['trades']

            for trade in trades:
                # Entry signal
                entry_signal = TradeSignal(
                    timestamp=trade.get('entry_time'),
                    price=trade.get('entry_price'),
                    signal_type="BUY" if trade.get('direction') == 'LONG' else "SELL",
                    volume=trade.get('volume'),
                    trade_id=trade.get('trade_id'),
                    metadata={'type': 'entry'}
                )

                # Exit signal
                exit_signal = TradeSignal(
                    timestamp=trade.get('exit_time'),
                    price=trade.get('exit_price'),
                    signal_type="SELL" if trade.get('direction') == 'LONG' else "BUY",
                    volume=trade.get('volume'),
                    trade_id=trade.get('trade_id'),
                    metadata={'type': 'exit'}
                )

                # Add markers
                self.trade_markers.add_trade_signal(entry_signal)
                self.trade_markers.add_trade_signal(exit_signal)

                # Add trade line
                pnl = trade.get('pnl', 0)
                self.trade_lines.add_trade_line(entry_signal, exit_signal, pnl)

                # Add PnL point
                self.pnl_overlay.add_pnl_point(
                    exit_signal.timestamp,
                    exit_signal.price,
                    pnl
                )

        # Process volume data
        if 'volume_data' in backtest_data:
            for vol_data in backtest_data['volume_data']:
                self.volume_markers.add_volume_marker(
                    vol_data.get('timestamp'),
                    vol_data.get('price'),
                    vol_data.get('volume')
                )

        # Process positions
        if 'positions' in backtest_data:
            for position in backtest_data['positions']:
                self.position_indicator.update_position(
                    position.get('timestamp'),
                    position.get('price'),
                    position.get('size'),
                    position.get('type', 'LONG')
                )

    def update_statistics(self, stats: Dict[str, Any]):
        """Update statistics overlay"""
        formatted_stats = {
            'Total Return': f"{stats.get('total_return', 0):.2%}",
            'Win Rate': f"{stats.get('win_rate', 0):.2%}",
            'Profit Factor': f"{stats.get('profit_factor', 0):.2f}",
            'Max Drawdown': f"{stats.get('max_drawdown', 0):.2%}",
            'Total Trades': stats.get('total_trades', 0)
        }

        self.stats_overlay.update_statistics(formatted_stats)

    def get_all_plot_items(self) -> List[pg.GraphicsObject]:
        """Get all plot items for removal"""
        items = []
        items.extend(self.trade_markers.marker_items)
        items.extend(self.trade_lines.lines)

        if self.position_indicator.current_position:
            items.append(self.position_indicator.current_position)

        items.append(self.volume_markers)
        items.append(self.stats_overlay)

        return items

    def clear_all(self):
        """Clear all markers và overlays"""
        self.trade_markers.clear_markers()
        self.trade_lines.clear_lines()
        self.pnl_overlay.clear_pnl()
        self.volume_markers.clear()

        # Clear position indicator
        if self.position_indicator.current_position:
            self.plot_widget.removeItem(self.position_indicator.current_position)
            self.position_indicator.current_position = None

        self.position_indicator.positions.clear()

    def update_config(self, new_config: DisplayConfig):
        """Update configuration cho tất cả components"""
        self.config = new_config

        # Update each component
        for component in self.all_components:
            if hasattr(component, 'config'):
                component.config = new_config

        # Refresh displays
        if hasattr(self.trade_markers, '_update_display'):
            self.trade_markers._update_display()
        if hasattr(self.position_indicator, '_update_display'):
            self.position_indicator._update_display()
        if hasattr(self.pnl_overlay, '_update_display'):
            self.pnl_overlay._update_display()

        self.volume_markers.update()


# Factory functions
def create_trade_markers(plot_widget, config: Optional[DisplayConfig] = None) -> TradeMarker:
    """Factory function to create trade markers"""
    return TradeMarker(plot_widget, config)


def create_marker_manager(plot_widget, config: Optional[DisplayConfig] = None) -> MarkerManager:
    """Factory function to create marker manager"""
    return MarkerManager(plot_widget, config)


def create_custom_marker(plot_widget, marker_type: str,
                         config: Optional[DisplayConfig] = None) -> TradeMarker:
    """Create custom marker với specific type"""
    marker = TradeMarker(plot_widget, config)
    # Customize based on marker_type
    return marker


def calculate_marker_positions(signals: List[TradeSignal],
                               chart_bounds: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    """Calculate optimal marker positions để avoid overlap"""
    positions = []

    for signal in signals:
        x_pos = signal.timestamp.timestamp()
        y_pos = signal.price

        # Simple positioning - in practice, you'd implement overlap avoidance
        positions.append((x_pos, y_pos))

    return positions


# Test function
def test_backtest_markers():
    """Test function for backtest markers"""
    print("🧪 Testing Backtest Markers...")

    # Mock data
    from datetime import datetime, timedelta

    mock_signals = [
        TradeSignal(
            timestamp=datetime.now() - timedelta(hours=2),
            price=50000.0,
            signal_type="BUY",
            volume=1.0,
            trade_id="trade_1"
        ),
        TradeSignal(
            timestamp=datetime.now() - timedelta(hours=1),
            price=50500.0,
            signal_type="SELL",
            volume=1.0,
            trade_id="trade_1"
        )
    ]

    # Create mock plot widget
    import pyqtgraph as pg
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    plot_widget = pg.PlotWidget()

    # Test trade markers
    trade_marker = TradeMarker(plot_widget)
    for signal in mock_signals:
        trade_marker.add_trade_signal(signal)

    print("✅ Trade markers created successfully")

    # Test marker manager
    manager = MarkerManager(plot_widget)

    mock_backtest_data = {
        'trades': [
            {
                'entry_time': datetime.now() - timedelta(hours=2),
                'entry_price': 50000.0,
                'exit_time': datetime.now() - timedelta(hours=1),
                'exit_price': 50500.0,
                'direction': 'LONG',
                'volume': 1.0,
                'trade_id': 'trade_1',
                'pnl': 500.0
            }
        ],
        'volume_data': [
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'price': 50000.0,
                'volume': 15000
            }
        ]
    }

    manager.process_backtest_data(mock_backtest_data)

    # Test statistics
    mock_stats = {
        'total_return': 0.15,
        'win_rate': 0.75,
        'profit_factor': 1.8,
        'max_drawdown': -0.08,
        'total_trades': 25
    }

    manager.update_statistics(mock_stats)

    print("✅ Marker manager tested successfully")
    print("✅ All backtest marker tests passed!")

    app.quit()


if __name__ == "__main__":
    test_backtest_markers()