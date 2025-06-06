"""
Chart Display Widget
===================

Enhanced chart display widget for backtest visualization.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt


class ChartDisplayWidget(QWidget):
    """Enhanced chart display widget"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.results_data = None

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Header with controls
        header_layout = QHBoxLayout()

        header_label = QLabel(" Chart Analysis")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2;")

        self.refresh_btn = QPushButton(" Refresh")
        self.refresh_btn.setMaximumWidth(100)
        self.refresh_btn.clicked.connect(self.refresh_chart)

        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.refresh_btn)

        main_layout.addLayout(header_layout)

        # Chart content area
        self.chart_label = QLabel(" Chart visualization will appear here")
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

    def display_chart(self, results):
        """Display comprehensive chart analysis"""
        self.results_data = results

        try:
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
            mode = results.get('mode', 'Unknown')

            # Create comprehensive chart info
            chart_html = f"""
<div style="font-family: Arial, sans-serif;">
<h2 style="color: #1976D2; text-align: center; margin-bottom: 20px;">
 Chart Analysis Dashboard
</h2>

<div style="background: #E3F2FD; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #0D47A1; margin-top: 0;"> Chart Overview</h3>
<table style="width: 100%;">
<tr><td><b>Strategy:</b></td><td>{strategy_name}</td></tr>
<tr><td><b>Symbol:</b></td><td>{symbol}</td></tr>
<tr><td><b>Timeframe:</b></td><td>{timeframe}</td></tr>
<tr><td><b>Analysis Mode:</b></td><td>{mode.title()}</td></tr>
</table>
</div>

<div style="background: #E8F5E8; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #2E7D32; margin-top: 0;"> Performance Visualization</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
<div>
<p><b> Total Return:</b></p>
<div style="background: {'#C8E6C9' if total_return > 0 else '#FFCDD2'}; padding: 8px; border-radius: 4px; text-align: center;">
<span style="font-size: 18px; font-weight: bold; color: {'#2E7D32' if total_return > 0 else '#D32F2F'};">
{total_return:+.2f}%
</span>
</div>
</div>
<div>
<p><b> Win Rate:</b></p>
<div style="background: {'#C8E6C9' if win_rate > 50 else '#FFF3E0' if win_rate > 30 else '#FFCDD2'}; padding: 8px; border-radius: 4px; text-align: center;">
<span style="font-size: 18px; font-weight: bold; color: {'#2E7D32' if win_rate > 50 else '#F57C00' if win_rate > 30 else '#D32F2F'};">
{win_rate:.1f}%
</span>
</div>
</div>
</div>
</div>

<div style="background: #FFF3E0; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #E65100; margin-top: 0;"> Chart Components Available</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
<div>
<h4 style="color: #FF6F00; margin: 10px 0 5px 0;"> Price Charts</h4>
<ul style="margin: 0; padding-left: 20px;">
<li>OHLC Candlestick Data</li>
<li>Volume Bars</li>
<li>Support/Resistance Levels</li>
<li>Moving Averages</li>
</ul>

<h4 style="color: #FF6F00; margin: 15px 0 5px 0;"> Technical Indicators</h4>
<ul style="margin: 0; padding-left: 20px;">
<li>RSI Oscillator</li>
<li>MACD Histogram</li>
<li>Bollinger Bands</li>
<li>Stochastic Oscillator</li>
</ul>
</div>
<div>
<h4 style="color: #FF6F00; margin: 10px 0 5px 0;"> Trading Signals</h4>
<ul style="margin: 0; padding-left: 20px;">
<li>Entry/Exit Points</li>
<li>Stop Loss Levels</li>
<li>Take Profit Targets</li>
<li>Position Sizing</li>
</ul>

<h4 style="color: #FF6F00; margin: 15px 0 5px 0;"> Performance Analysis</h4>
<ul style="margin: 0; padding-left: 20px;">
<li>Equity Curve</li>
<li>Drawdown Chart</li>
<li>P&L Distribution</li>
<li>Trade Statistics</li>
</ul>
</div>
</div>
</div>

<div style="background: #F3E5F5; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #7B1FA2; margin-top: 0;"> Key Statistics</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px;"><b>Total Trades:</b></td>
    <td style="padding: 5px; text-align: center; background: #E1BEE7; border-radius: 4px;">{total_trades}</td></tr>
<tr><td style="padding: 5px;"><b>Max Drawdown:</b></td>
    <td style="padding: 5px; text-align: center; background: #FFCDD2; border-radius: 4px; color: #D32F2F; font-weight: bold;">{max_drawdown:.2f}%</td></tr>
<tr><td style="padding: 5px;"><b>Sharpe Ratio:</b></td>
    <td style="padding: 5px; text-align: center; background: {'#C8E6C9' if sharpe_ratio > 1 else '#FFF3E0' if sharpe_ratio > 0 else '#FFCDD2'}; border-radius: 4px; color: {'#2E7D32' if sharpe_ratio > 1 else '#F57C00' if sharpe_ratio > 0 else '#D32F2F'}; font-weight: bold;">{sharpe_ratio:.3f}</td></tr>
</table>
</div>

<div style="text-align: center; margin-top: 20px; padding: 15px; background: linear-gradient(45deg, #E3F2FD, #F3E5F5); border-radius: 8px; border: 2px dashed #2196F3;">
<p style="color: #1976D2; font-weight: bold; margin: 5px 0;"> Chart Visualization Status</p>
<p style="color: #666; margin: 5px 0;">
 Install <b>pyqtgraph</b> and <b>matplotlib</b> for advanced interactive charts
</p>
<p style="color: #666; margin: 5px 0; font-style: italic;">
Current mode: <b>{mode.title()}</b> • Ready for visualization
</p>
</div>
</div>
            """

            self.chart_label.setText(chart_html)

        except Exception as e:
            error_html = f"""
<div style="text-align: center; color: #D32F2F; padding: 20px;">
<h3>❌ Error Displaying Chart</h3>
<p><b>Error:</b> {str(e)}</p>
<p><b>Data type:</b> {type(results)}</p>
<p style="color: #666; margin-top: 15px;">Please check the results data format</p>
</div>
            """
            self.chart_label.setText(error_html)

    def refresh_chart(self):
        """Refresh chart display"""
        if self.results_data:
            self.display_chart(self.results_data)
        else:
            self.chart_label.setText(" No data to refresh. Run a backtest first.")