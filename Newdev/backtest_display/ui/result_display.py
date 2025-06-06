"""
Result Display Widget
====================

Enhanced result display widget for backtest results.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt


class ResultDisplayWidget(QWidget):
    """Enhanced result display widget"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Create scroll area for results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        self.result_label = QLabel(" Ready to display backtest results")
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("""
            QLabel {
                padding: 20px;
                border: 2px dashed #4CAF50;
                border-radius: 10px;
                background-color: #f8f9fa;
                color: #2E7D32;
                font-size: 14px;
                line-height: 1.6;
            }
        """)

        content_layout.addWidget(self.result_label)
        content_layout.addStretch()

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

    def display_results(self, results):
        """Display comprehensive backtest results"""
        try:
            # Extract metrics safely
            total_return = results.get('total_return', 0) * 100
            annual_return = results.get('annual_return', 0) * 100
            sharpe_ratio = results.get('sharpe_ratio', 0)
            max_drawdown = results.get('max_drawdown', 0) * 100
            win_rate = results.get('win_rate', 0) * 100
            total_trades = results.get('total_trades', 0)
            winning_trades = results.get('winning_trades', 0)
            losing_trades = results.get('losing_trades', 0)

            # Financial metrics
            start_capital = results.get('start_capital', 0)
            end_capital = results.get('end_capital', 0)
            total_pnl = results.get('total_pnl', 0)

            # Additional metrics
            profit_factor = results.get('profit_factor', 0)
            avg_trade = results.get('avg_trade', 0)
            largest_win = results.get('largest_win', 0)
            largest_loss = results.get('largest_loss', 0)

            # Strategy info
            strategy_name = results.get('strategy_name', 'Unknown Strategy')
            symbol = results.get('symbol', 'Unknown Symbol')
            timeframe = results.get('timeframe', 'Unknown Timeframe')
            mode = results.get('mode', 'Unknown')

            # Create comprehensive display
            html_content = f"""
<div style="font-family: Arial, sans-serif;">
<h2 style="color: #1976D2; text-align: center; margin-bottom: 20px;">
 Backtest Results Analysis
</h2>

<div style="background: #E3F2FD; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #0D47A1; margin-top: 0;"> Strategy Information</h3>
<table style="width: 100%;">
<tr><td><b>Strategy:</b></td><td>{strategy_name}</td></tr>
<tr><td><b>Symbol:</b></td><td>{symbol}</td></tr>
<tr><td><b>Timeframe:</b></td><td>{timeframe}</td></tr>
<tr><td><b>Mode:</b></td><td>{mode.title()}</td></tr>
</table>
</div>

<div style="background: #E8F5E8; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #2E7D32; margin-top: 0;"> Performance Metrics</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px;"><b>Total Return:</b></td>
    <td style="padding: 3px; color: {'#2E7D32' if total_return > 0 else '#D32F2F'}; font-weight: bold;">
    {total_return:+.2f}%</td></tr>
<tr><td style="padding: 3px;"><b>Annual Return:</b></td>
    <td style="padding: 3px; color: {'#2E7D32' if annual_return > 0 else '#D32F2F'};">
    {annual_return:+.2f}%</td></tr>
<tr><td style="padding: 3px;"><b>Sharpe Ratio:</b></td>
    <td style="padding: 3px; color: {'#2E7D32' if sharpe_ratio > 1 else '#FF9800' if sharpe_ratio > 0 else '#D32F2F'};">
    {sharpe_ratio:.3f}</td></tr>
<tr><td style="padding: 3px;"><b>Max Drawdown:</b></td>
    <td style="padding: 3px; color: #D32F2F; font-weight: bold;">
    {max_drawdown:.2f}%</td></tr>
<tr><td style="padding: 3px;"><b>Profit Factor:</b></td>
    <td style="padding: 3px; color: {'#2E7D32' if profit_factor > 1 else '#D32F2F'};">
    {profit_factor:.2f}</td></tr>
</table>
</div>

<div style="background: #FFF3E0; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #E65100; margin-top: 0;"> Trading Statistics</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px;"><b>Total Trades:</b></td>
    <td style="padding: 3px; font-weight: bold;">{total_trades}</td></tr>
<tr><td style="padding: 3px;"><b>Winning Trades:</b></td>
    <td style="padding: 3px; color: #2E7D32;">{winning_trades}</td></tr>
<tr><td style="padding: 3px;"><b>Losing Trades:</b></td>
    <td style="padding: 3px; color: #D32F2F;">{losing_trades}</td></tr>
<tr><td style="padding: 3px;"><b>Win Rate:</b></td>
    <td style="padding: 3px; color: {'#2E7D32' if win_rate > 50 else '#FF9800' if win_rate > 30 else '#D32F2F'}; font-weight: bold;">
    {win_rate:.1f}%</td></tr>
<tr><td style="padding: 3px;"><b>Average Trade:</b></td>
    <td style="padding: 3px; color: {'#2E7D32' if avg_trade > 0 else '#D32F2F'};">
    ${avg_trade:,.2f}</td></tr>
</table>
</div>

<div style="background: #F3E5F5; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
<h3 style="color: #7B1FA2; margin-top: 0;"> Financial Summary</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px;"><b>Start Capital:</b></td>
    <td style="padding: 3px;">${start_capital:,.2f}</td></tr>
<tr><td style="padding: 3px;"><b>End Capital:</b></td>
    <td style="padding: 3px; color: {'#2E7D32' if end_capital > start_capital else '#D32F2F'}; font-weight: bold;">
    ${end_capital:,.2f}</td></tr>
<tr><td style="padding: 3px;"><b>Total P&L:</b></td>
    <td style="padding: 3px; color: {'#2E7D32' if total_pnl > 0 else '#D32F2F'}; font-weight: bold;">
    ${total_pnl:+,.2f}</td></tr>
<tr><td style="padding: 3px;"><b>Largest Win:</b></td>
    <td style="padding: 3px; color: #2E7D32;">${largest_win:,.2f}</td></tr>
<tr><td style="padding: 3px;"><b>Largest Loss:</b></td>
    <td style="padding: 3px; color: #D32F2F;">${largest_loss:,.2f}</td></tr>
</table>
</div>

<div style="text-align: center; margin-top: 20px; padding: 10px; background: #F5F5F5; border-radius: 8px;">
<p style="color: #666; font-style: italic; margin: 0;">
 Analysis completed • {mode.title()} Mode
</p>
</div>
</div>
            """

            self.result_label.setText(html_content)

        except Exception as e:
            error_html = f"""
<div style="text-align: center; color: #D32F2F;">
<h3>❌ Error Displaying Results</h3>
<p><b>Error:</b> {str(e)}</p>
<p><b>Available data keys:</b> {list(results.keys()) if isinstance(results, dict) else 'Not a dictionary'}</p>
</div>
            """
            self.result_label.setText(error_html)