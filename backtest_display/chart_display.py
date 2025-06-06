"""
Chart Display Widget
===================

Enhanced chart display widget for backtest visualization.
Supports both real Howtrader charts and mock visualization.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt
from typing import Dict, Any
from datetime import datetime


class ChartDisplayWidget(QWidget):
    """Enhanced chart display widget for comprehensive visualization"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.results_data = None

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)

        # Header with controls
        header_layout = QHBoxLayout()

        header_label = QLabel("📈 Chart Analysis")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2;")

        self.interactive_btn = QPushButton("🔗 Interactive")
        self.interactive_btn.setMaximumWidth(100)
        self.interactive_btn.clicked.connect(self.show_interactive_chart)

        self.export_btn = QPushButton("💾 Export")
        self.export_btn.setMaximumWidth(100)
        self.export_btn.clicked.connect(self.export_chart)

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setMaximumWidth(100)
        self.refresh_btn.clicked.connect(self.refresh_chart)

        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.interactive_btn)
        header_layout.addWidget(self.export_btn)
        header_layout.addWidget(self.refresh_btn)

        main_layout.addLayout(header_layout)

        # Chart content area
        self.chart_label = QLabel("📈 Chart visualization will appear here")
        self.chart_label.setAlignment(Qt.AlignTop)
        self.chart_label.setWordWrap(True)
        self.chart_label.setStyleSheet("""
            QLabel {
                padding: 20px;
                border: 2px dashed #2196F3;
                border-radius: 10px;
                background-color: #f8fafe;
                color: #1976D2;
                font-size: 14px;
                min-height: 400px;
                line-height: 1.6;
            }
        """)

        main_layout.addWidget(self.chart_label)

    def display_chart(self, results: Dict[str, Any]):
        """Display comprehensive chart analysis"""
        self.results_data = results

        try:
            mode = results.get('mode', 'unknown')

            if mode == 'mock':
                self._display_mock_chart(results)
            elif mode == 'real':
                self._display_real_chart(results)
            else:
                self._display_legacy_chart(results)

        except Exception as e:
            self._display_chart_error(f"Error displaying chart: {str(e)}", results)

    def _display_real_chart(self, results: Dict[str, Any]):
        """Display real Howtrader chart information"""
        # Extract metrics for chart info
        total_return = results.get('total_return', 0) * 100
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0) * 100
        max_drawdown = results.get('max_drawdown', 0) * 100
        sharpe_ratio = results.get('sharpe_ratio', 0)

        # Strategy info
        strategy_name = results.get('strategy_name', 'Unknown Strategy')
        symbol = results.get('symbol', 'Unknown Symbol')
        timeframe = results.get('timeframe', 'Unknown')
        start_date = results.get('start_date', datetime.now())
        end_date = results.get('end_date', datetime.now())

        # Format dates
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else str(end_date)

        # Create comprehensive chart info
        chart_html = f"""
<div style="font-family: Arial, sans-serif;">

<div style="text-align: center; margin-bottom: 20px; padding: 15px; background: linear-gradient(45deg, #4CAF50, #2196F3); color: white; border-radius: 10px;">
<h2 style="margin: 0; color: white;">📈 Real Howtrader Chart Analysis</h2>
<p style="margin: 5px 0; opacity: 0.9;">Professional Trading Visualization</p>
</div>

<div style="background: #E3F2FD; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #0D47A1; margin-top: 0;">📊 Chart Overview</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px 0;"><b>Strategy:</b></td><td style="text-align: right;">{strategy_name}</td></tr>
<tr><td style="padding: 5px 0;"><b>Symbol:</b></td><td style="text-align: right;">{symbol}</td></tr>
<tr><td style="padding: 5px 0;"><b>Timeframe:</b></td><td style="text-align: right;">{timeframe}</td></tr>
<tr><td style="padding: 5px 0;"><b>Period:</b></td><td style="text-align: right;">{start_str} to {end_str}</td></tr>
<tr><td style="padding: 5px 0;"><b>Analysis Mode:</b></td><td style="text-align: right;"><span style="color: #4CAF50; font-weight: bold;">Real Howtrader</span></td></tr>
</table>
</div>

<div style="background: #E8F5E8; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #2E7D32; margin-top: 0;">📈 Performance Visualization</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
<div>
<h4 style="color: #2E7D32; margin: 10px 0 5px 0;">📊 Equity Curve</h4>
<div style="background: {'#C8E6C9' if total_return > 0 else '#FFCDD2'}; padding: 15px; border-radius: 8px; text-align: center;">
<p style="margin: 5px 0;"><b>Total Return</b></p>
<span style="font-size: 24px; font-weight: bold; color: {'#2E7D32' if total_return > 0 else '#D32F2F'};">
{total_return:+.2f}%
</span>
</div>
</div>
<div>
<h4 style="color: #2E7D32; margin: 10px 0 5px 0;">🎯 Win Rate</h4>
<div style="background: {'#C8E6C9' if win_rate > 50 else '#FFF3E0' if win_rate > 30 else '#FFCDD2'}; padding: 15px; border-radius: 8px; text-align: center;">
<p style="margin: 5px 0;"><b>Success Rate</b></p>
<span style="font-size: 24px; font-weight: bold; color: {'#2E7D32' if win_rate > 50 else '#F57C00' if win_rate > 30 else '#D32F2F'};">
{win_rate:.1f}%
</span>
</div>
</div>
</div>
</div>

<div style="background: #FFF3E0; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #E65100; margin-top: 0;">📋 Available Chart Components</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
<div>
<h4 style="color: #FF6F00; margin: 10px 0 5px 0;">📈 Price Charts</h4>
<ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
<li>✅ OHLC Candlestick Data</li>
<li>✅ Volume Bars</li>
<li>✅ Support/Resistance Levels</li>
<li>✅ Moving Averages</li>
</ul>

<h4 style="color: #FF6F00; margin: 15px 0 5px 0;">📊 Technical Indicators</h4>
<ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
<li>✅ RSI Oscillator</li>
<li>✅ MACD Histogram</li>
<li>✅ Bollinger Bands</li>
<li>✅ Stochastic Oscillator</li>
</ul>
</div>
<div>
<h4 style="color: #FF6F00; margin: 10px 0 5px 0;">🎯 Trading Signals</h4>
<ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
<li>✅ Entry/Exit Points</li>
<li>✅ Stop Loss Levels</li>
<li>✅ Take Profit Targets</li>
<li>✅ Position Sizing</li>
</ul>

<h4 style="color: #FF6F00; margin: 15px 0 5px 0;">📊 Performance Analysis</h4>
<ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
<li>✅ Equity Curve</li>
<li>✅ Drawdown Chart</li>
<li>✅ P&L Distribution</li>
<li>✅ Trade Statistics</li>
</ul>
</div>
</div>
</div>

<div style="background: #F3E5F5; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #7B1FA2; margin-top: 0;">📊 Key Chart Statistics</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 8px; width: 50%;"><b>📊 Total Trades:</b></td>
    <td style="padding: 8px; text-align: center; background: #E1BEE7; border-radius: 4px; font-weight: bold;">{total_trades}</td></tr>
<tr><td style="padding: 8px;"><b>📉 Max Drawdown:</b></td>
    <td style="padding: 8px; text-align: center; background: #FFCDD2; border-radius: 4px; color: #D32F2F; font-weight: bold;">{max_drawdown:.2f}%</td></tr>
<tr><td style="padding: 8px;"><b>⚡ Sharpe Ratio:</b></td>
    <td style="padding: 8px; text-align: center; background: {'#C8E6C9' if sharpe_ratio > 1 else '#FFF3E0' if sharpe_ratio > 0 else '#FFCDD2'}; border-radius: 4px; color: {'#2E7D32' if sharpe_ratio > 1 else '#F57C00' if sharpe_ratio > 0 else '#D32F2F'}; font-weight: bold;">{sharpe_ratio:.3f}</td></tr>
</table>
</div>

<div style="background: #E0F2F1; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #004D40; margin-top: 0;">🔧 Interactive Features</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
<div>
<p style="margin: 5px 0;"><b>🖱️ Mouse Interactions:</b></p>
<ul style="margin: 5px 0; padding-left: 20px; line-height: 1.6;">
<li>Zoom and pan</li>
<li>Crosshair cursor</li>
<li>Data tooltips</li>
<li>Trade annotations</li>
</ul>
</div>
<div>
<p style="margin: 5px 0;"><b>⚙️ Chart Controls:</b></p>
<ul style="margin: 5px 0; padding-left: 20px; line-height: 1.6;">
<li>Toggle indicators</li>
<li>Change timeframes</li>
<li>Export options</li>
<li>Full-screen mode</li>
</ul>
</div>
</div>
</div>

<div style="text-align: center; margin-top: 20px; padding: 15px; background: linear-gradient(45deg, #E3F2FD, #F3E5F5); border-radius: 8px; border: 2px dashed #2196F3;">
<p style="color: #1976D2; font-weight: bold; margin: 5px 0;">📈 Real Howtrader Chart Visualization</p>
<p style="color: #666; margin: 5px 0;">
Click "🔗 Interactive" to open the full Howtrader chart window
</p>
<p style="color: #666; margin: 5px 0; font-style: italic;">
Professional-grade analysis ready • Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</p>
</div>

</div>
        """

        self.chart_label.setText(chart_html)

    def _display_mock_chart(self, results: Dict[str, Any]):
        """Display mock chart information for demo strategies"""
        # Extract metrics for chart info
        total_return = results.get('total_return', 0) * 100
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0) * 100
        max_drawdown = results.get('max_drawdown', 0) * 100
        sharpe_ratio = results.get('sharpe_ratio', 0)

        # Strategy info
        strategy_name = results.get('strategy_name', 'Unknown Strategy')
        symbol = results.get('symbol', 'Unknown Symbol')
        timeframe = results.get('timeframe', 'Unknown')

        # Create mock chart info
        chart_html = f"""
<div style="font-family: Arial, sans-serif;">

<div style="text-align: center; margin-bottom: 20px; padding: 15px; background: linear-gradient(45deg, #FF9800, #2196F3); color: white; border-radius: 10px;">
<h2 style="margin: 0; color: white;">📈 Mock Chart Analysis</h2>
<p style="margin: 5px 0; opacity: 0.9;">Educational Demo Mode</p>
</div>

<div style="background: #FFF3E0; padding: 15px; border-radius: 8px; border: 2px dashed #FF9800; margin-bottom: 15px;">
<h3 style="color: #E65100; margin-top: 0; text-align: center;">🧪 DEMO MODE - Simulated Charts</h3>
<p style="color: #E65100; margin: 5px 0; text-align: center;">
These are educational visualizations for learning purposes.<br>
Real chart data requires Howtrader installation.
</p>
</div>

<div style="background: #E3F2FD; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #0D47A1; margin-top: 0;">📊 Mock Chart Overview</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px 0;"><b>Strategy:</b></td><td style="text-align: right;">{strategy_name}</td></tr>
<tr><td style="padding: 5px 0;"><b>Symbol:</b></td><td style="text-align: right;">{symbol}</td></tr>
<tr><td style="padding: 5px 0;"><b>Timeframe:</b></td><td style="text-align: right;">{timeframe}</td></tr>
<tr><td style="padding: 5px 0;"><b>Analysis Mode:</b></td><td style="text-align: right;"><span style="color: #FF9800; font-weight: bold;">Mock Demo</span></td></tr>
</table>
</div>

<div style="background: #E8F5E8; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #2E7D32; margin-top: 0;">📈 Simulated Performance</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
<div>
<h4 style="color: #2E7D32; margin: 10px 0 5px 0;">📊 Mock Equity</h4>
<div style="background: {'#C8E6C9' if total_return > 0 else '#FFCDD2'}; padding: 15px; border-radius: 8px; text-align: center;">
<p style="margin: 5px 0;"><b>Simulated Return</b></p>
<span style="font-size: 24px; font-weight: bold; color: {'#2E7D32' if total_return > 0 else '#D32F2F'};">
{total_return:+.2f}%
</span>
</div>
</div>
<div>
<h4 style="color: #2E7D32; margin: 10px 0 5px 0;">🎯 Mock Win Rate</h4>
<div style="background: {'#C8E6C9' if win_rate > 50 else '#FFF3E0' if win_rate > 30 else '#FFCDD2'}; padding: 15px; border-radius: 8px; text-align: center;">
<p style="margin: 5px 0;"><b>Success Rate</b></p>
<span style="font-size: 24px; font-weight: bold; color: {'#2E7D32' if win_rate > 50 else '#F57C00' if win_rate > 30 else '#D32F2F'};">
{win_rate:.1f}%
</span>
</div>
</div>
</div>
</div>

<div style="background: #FFF3E0; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #E65100; margin-top: 0;">📋 Educational Chart Components</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
<div>
<h4 style="color: #FF6F00; margin: 10px 0 5px 0;">📈 Chart Types</h4>
<ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
<li>🧪 Mock Candlestick Data</li>
<li>🧪 Simulated Volume</li>
<li>🧪 Demo Support/Resistance</li>
<li>🧪 Example Moving Averages</li>
</ul>

<h4 style="color: #FF6F00; margin: 15px 0 5px 0;">📊 Learning Indicators</h4>
<ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
<li>🧪 Educational RSI</li>
<li>🧪 Demo MACD</li>
<li>🧪 Example Bollinger Bands</li>
<li>🧪 Practice Stochastic</li>
</ul>
</div>
<div>
<h4 style="color: #FF6F00; margin: 10px 0 5px 0;">🎯 Practice Signals</h4>
<ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
<li>🧪 Demo Entry/Exit Points</li>
<li>🧪 Mock Stop Loss</li>
<li>🧪 Example Take Profit</li>
<li>🧪 Learning Position Size</li>
</ul>

<h4 style="color: #FF6F00; margin: 15px 0 5px 0;">📊 Educational Analysis</h4>
<ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
<li>🧪 Mock Equity Curve</li>
<li>🧪 Demo Drawdown</li>
<li>🧪 Simulated P&L</li>
<li>🧪 Practice Statistics</li>
</ul>
</div>
</div>
</div>

<div style="background: #F3E5F5; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #7B1FA2; margin-top: 0;">📊 Mock Statistics</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 8px; width: 50%;"><b>📊 Demo Trades:</b></td>
    <td style="padding: 8px; text-align: center; background: #E1BEE7; border-radius: 4px; font-weight: bold;">{total_trades}</td></tr>
<tr><td style="padding: 8px;"><b>📉 Mock Drawdown:</b></td>
    <td style="padding: 8px; text-align: center; background: #FFCDD2; border-radius: 4px; color: #D32F2F; font-weight: bold;">{max_drawdown:.2f}%</td></tr>
<tr><td style="padding: 8px;"><b>⚡ Demo Sharpe:</b></td>
    <td style="padding: 8px; text-align: center; background: {'#C8E6C9' if sharpe_ratio > 1 else '#FFF3E0' if sharpe_ratio > 0 else '#FFCDD2'}; border-radius: 4px; color: {'#2E7D32' if sharpe_ratio > 1 else '#F57C00' if sharpe_ratio > 0 else '#D32F2F'}; font-weight: bold;">{sharpe_ratio:.3f}</td></tr>
</table>
</div>

<div style="background: #E0F2F1; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #004D40; margin-top: 0;">🎓 Learning Features</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
<div>
<p style="margin: 5px 0;"><b>📚 Educational Tools:</b></p>
<ul style="margin: 5px 0; padding-left: 20px; line-height: 1.6;">
<li>Interactive tutorials</li>
<li>Strategy explanations</li>
<li>Risk management lessons</li>
<li>Market analysis basics</li>
</ul>
</div>
<div>
<p style="margin: 5px 0;"><b>🔧 Practice Features:</b></p>
<ul style="margin: 5px 0; padding-left: 20px; line-height: 1.6;">
<li>Parameter experimentation</li>
<li>Scenario testing</li>
<li>Performance comparison</li>
<li>Strategy optimization</li>
</ul>
</div>
</div>
</div>

<div style="text-align: center; margin-top: 20px; padding: 15px; background: linear-gradient(45deg, #FFF3E0, #E3F2FD); border-radius: 8px; border: 2px dashed #FF9800;">
<p style="color: #E65100; font-weight: bold; margin: 5px 0;">🧪 Educational Demo Mode</p>
<p style="color: #666; margin: 5px 0;">
For real charts, install Howtrader and use production strategies
</p>
<p style="color: #666; margin: 5px 0; font-style: italic;">
Learning environment ready • Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</p>
</div>

</div>
        """

        self.chart_label.setText(chart_html)

    def _display_legacy_chart(self, results: Dict[str, Any]):
        """Display legacy chart format"""
        strategy = results.get('strategy', 'Unknown Strategy')
        total_return = results.get('total_return', 0) * 100

        chart_html = f"""
<div style="font-family: Arial, sans-serif; text-align: center; padding: 20px;">
<h2 style="color: #1976D2;">📈 Legacy Chart View</h2>
<div style="background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0;">
<h3>{strategy}</h3>
<p>Chart visualization for legacy format</p>
<p><b>Return:</b> <span style="color: {'green' if total_return > 0 else 'red'};">{total_return:+.2f}%</span></p>
</div>
<p style="color: #666;">Chart features limited in legacy mode</p>
</div>
        """

        self.chart_label.setText(chart_html)

    def _display_chart_error(self, error_message: str, results: Dict[str, Any]):
        """Display chart error message"""
        error_html = f"""
<div style="text-align: center; color: #D32F2F; padding: 20px;">
<h3>❌ Error Displaying Chart</h3>
<p><b>Error:</b> {error_message}</p>
<p><b>Data type:</b> {type(results)}</p>
<div style="background: #ffebee; padding: 15px; border-radius: 5px; margin: 15px 0;">
<p><b>Available data keys:</b></p>
<p>{', '.join(results.keys()) if isinstance(results, dict) else 'Not a dictionary'}</p>
</div>
<p style="color: #666; margin-top: 15px;">Please check the results data format</p>
</div>
        """
        self.chart_label.setText(error_html)

    def show_interactive_chart(self):
        """Show interactive chart (placeholder for now)"""
        if not self.results_data:
            return

        mode = self.results_data.get('mode', 'unknown')

        if mode == 'real':
            # For real mode, this would trigger Howtrader's show_chart()
            print("🔗 Opening Howtrader interactive chart...")
            # The main window should handle this via bridge.show_chart()
        else:
            print("🧪 Interactive charts not available in demo mode")

    def export_chart(self):
        """Export chart data"""
        if not self.results_data:
            return

        try:
            strategy_name = self.results_data.get('strategy_name', 'Unknown')
            mode = self.results_data.get('mode', 'unknown')

            print(f"💾 Exporting chart data for {strategy_name} ({mode} mode)")
            # In a full implementation, this would save chart images/data

        except Exception as e:
            print(f"Export failed: {e}")

    def refresh_chart(self):
        """Refresh chart display"""
        if self.results_data:
            self.display_chart(self.results_data)
        else:
            self.chart_label.setText("📈 No data to refresh. Run a backtest first.")