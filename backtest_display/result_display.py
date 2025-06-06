"""
Result Display Widget
====================

Enhanced result display widget for comprehensive backtest result presentation.
Handles both real Howtrader results and mock demo results.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QTextEdit, QScrollArea
from PySide6.QtCore import Qt
from typing import Dict, Any
from datetime import datetime


class ResultDisplayWidget(QWidget):
    """Enhanced result display widget for comprehensive backtest results"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.results_data = None

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)

        # Header with controls
        header_layout = QHBoxLayout()

        header_label = QLabel("📊 Backtest Results")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2;")

        self.export_btn = QPushButton("📋 Export")
        self.export_btn.setMaximumWidth(100)
        self.export_btn.clicked.connect(self.export_results)

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setMaximumWidth(100)
        self.refresh_btn.clicked.connect(self.refresh_results)

        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.export_btn)
        header_layout.addWidget(self.refresh_btn)

        main_layout.addLayout(header_layout)

        # Scrollable results area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Results content
        self.result_label = QLabel("📊 Run a backtest to see results here")
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("""
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

        scroll_area.setWidget(self.result_label)
        main_layout.addWidget(scroll_area)

    def display_results(self, results: Dict[str, Any]):
        """Display comprehensive backtest results"""
        self.results_data = results

        try:
            # Determine result mode
            mode = results.get('mode', 'unknown')

            if mode == 'mock':
                self._display_mock_results(results)
            elif mode == 'real':
                self._display_real_results(results)
            else:
                self._display_legacy_results(results)

        except Exception as e:
            self._display_error(f"Error displaying results: {str(e)}", results)

    def _display_real_results(self, results: Dict[str, Any]):
        """Display real Howtrader backtest results"""
        # Extract metrics
        strategy_name = results.get('strategy_name', 'Unknown Strategy')
        symbol = results.get('symbol', 'Unknown Symbol')
        timeframe = results.get('timeframe', 'Unknown')
        start_date = results.get('start_date', datetime.now())
        end_date = results.get('end_date', datetime.now())
        capital = results.get('capital', 0)

        total_return = results.get('total_return', 0) * 100
        annual_return = results.get('annual_return', 0) * 100
        max_drawdown = results.get('max_drawdown', 0) * 100
        sharpe_ratio = results.get('sharpe_ratio', 0)
        calmar_ratio = results.get('calmar_ratio', 0)

        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0) * 100
        profit_factor = results.get('profit_factor', 0)
        avg_profit = results.get('avg_profit', 0)
        avg_loss = results.get('avg_loss', 0)
        max_profit = results.get('max_profit', 0)
        max_loss = results.get('max_loss', 0)

        total_commission = results.get('total_commission', 0)
        total_slippage = results.get('total_slippage', 0)

        # Format dates
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else str(end_date)

        # Generate comprehensive HTML report
        html_content = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6;">

<div style="text-align: center; margin-bottom: 20px; padding: 15px; background: linear-gradient(45deg, #4CAF50, #2196F3); color: white; border-radius: 10px;">
<h2 style="margin: 0; color: white;">🚀 Real Howtrader Backtest Results</h2>
<p style="margin: 5px 0; opacity: 0.9;">Professional CTA Strategy Analysis</p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">

<div style="background: #E3F2FD; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3;">
<h3 style="color: #0D47A1; margin-top: 0; margin-bottom: 10px;">📋 Strategy Information</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px 0;"><b>Strategy:</b></td><td style="text-align: right;">{strategy_name}</td></tr>
<tr><td style="padding: 3px 0;"><b>Symbol:</b></td><td style="text-align: right;">{symbol}</td></tr>
<tr><td style="padding: 3px 0;"><b>Timeframe:</b></td><td style="text-align: right;">{timeframe}</td></tr>
<tr><td style="padding: 3px 0;"><b>Period:</b></td><td style="text-align: right;">{start_str} to {end_str}</td></tr>
<tr><td style="padding: 3px 0;"><b>Capital:</b></td><td style="text-align: right;">${capital:,.0f}</td></tr>
</table>
</div>

<div style="background: #E8F5E8; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50;">
<h3 style="color: #2E7D32; margin-top: 0; margin-bottom: 10px;">📈 Key Performance</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px 0;"><b>Total Return:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if total_return > 0 else '#D32F2F'}; font-weight: bold;">
    {total_return:+.2f}%</td></tr>
<tr><td style="padding: 3px 0;"><b>Annual Return:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if annual_return > 0 else '#D32F2F'}; font-weight: bold;">
    {annual_return:+.2f}%</td></tr>
<tr><td style="padding: 3px 0;"><b>Max Drawdown:</b></td>
    <td style="text-align: right; color: #D32F2F; font-weight: bold;">{max_drawdown:.2f}%</td></tr>
<tr><td style="padding: 3px 0;"><b>Sharpe Ratio:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if sharpe_ratio > 1 else '#F57C00' if sharpe_ratio > 0 else '#D32F2F'}; font-weight: bold;">
    {sharpe_ratio:.3f}</td></tr>
<tr><td style="padding: 3px 0;"><b>Calmar Ratio:</b></td>
    <td style="text-align: right; font-weight: bold;">{calmar_ratio:.3f}</td></tr>
</table>
</div>

</div>

<div style="background: #FFF3E0; padding: 15px; border-radius: 8px; border-left: 4px solid #FF9800; margin-bottom: 15px;">
<h3 style="color: #E65100; margin-top: 0; margin-bottom: 15px;">📊 Trading Statistics</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
<div>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px 0;"><b>Total Trades:</b></td><td style="text-align: right; font-weight: bold;">{total_trades}</td></tr>
<tr><td style="padding: 3px 0;"><b>Win Rate:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if win_rate > 50 else '#F57C00' if win_rate > 30 else '#D32F2F'}; font-weight: bold;">
    {win_rate:.1f}%</td></tr>
<tr><td style="padding: 3px 0;"><b>Profit Factor:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if profit_factor > 1 else '#D32F2F'}; font-weight: bold;">
    {profit_factor:.2f}</td></tr>
</table>
</div>
<div>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px 0;"><b>Avg Profit:</b></td><td style="text-align: right; color: #2E7D32; font-weight: bold;">${avg_profit:+.2f}</td></tr>
<tr><td style="padding: 3px 0;"><b>Avg Loss:</b></td><td style="text-align: right; color: #D32F2F; font-weight: bold;">${avg_loss:.2f}</td></tr>
<tr><td style="padding: 3px 0;"><b>Max Profit:</b></td><td style="text-align: right; color: #2E7D32; font-weight: bold;">${max_profit:+.2f}</td></tr>
<tr><td style="padding: 3px 0;"><b>Max Loss:</b></td><td style="text-align: right; color: #D32F2F; font-weight: bold;">${max_loss:.2f}</td></tr>
</table>
</div>
</div>
</div>

<div style="background: #F3E5F5; padding: 15px; border-radius: 8px; border-left: 4px solid #9C27B0; margin-bottom: 15px;">
<h3 style="color: #7B1FA2; margin-top: 0; margin-bottom: 10px;">💰 Cost Analysis</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px 0; width: 60%;"><b>Total Commission:</b></td><td style="text-align: right;">${total_commission:.2f}</td></tr>
<tr><td style="padding: 3px 0;"><b>Total Slippage:</b></td><td style="text-align: right;">${total_slippage:.2f}</td></tr>
<tr><td style="padding: 3px 0;"><b>Total Costs:</b></td><td style="text-align: right; font-weight: bold;">${total_commission + total_slippage:.2f}</td></tr>
</table>
</div>

<div style="background: #E0F2F1; padding: 15px; border-radius: 8px; border-left: 4px solid #00796B;">
<h3 style="color: #004D40; margin-top: 0; margin-bottom: 10px;">🎯 Performance Summary</h3>
<p style="margin: 5px 0;"><b>Overall Assessment:</b> 
{'<span style="color: #2E7D32;">✅ Profitable Strategy</span>' if total_return > 0 else '<span style="color: #D32F2F;">❌ Losing Strategy</span>'}
</p>
<p style="margin: 5px 0;"><b>Risk Assessment:</b> 
{'<span style="color: #2E7D32;">✅ Low Risk</span>' if max_drawdown < 10 else '<span style="color: #F57C00;">⚠️ Medium Risk</span>' if max_drawdown < 20 else '<span style="color: #D32F2F;">❌ High Risk</span>'}
</p>
<p style="margin: 5px 0;"><b>Quality Score:</b> 
{'<span style="color: #2E7D32;">✅ Excellent</span>' if sharpe_ratio > 2 else '<span style="color: #4CAF50;">✅ Good</span>' if sharpe_ratio > 1 else '<span style="color: #F57C00;">⚠️ Fair</span>' if sharpe_ratio > 0 else '<span style="color: #D32F2F;">❌ Poor</span>'}
 (Sharpe: {sharpe_ratio:.2f})
</p>
</div>

<div style="text-align: center; margin-top: 20px; padding: 10px; background: #F5F5F5; border-radius: 6px; color: #666;">
<small>🔧 Real Howtrader Results • Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
</div>

</div>
        """

        self.result_label.setText(html_content)

    def _display_mock_results(self, results: Dict[str, Any]):
        """Display mock/demo backtest results"""
        # Extract metrics
        strategy_name = results.get('strategy_name', 'Unknown Strategy')
        symbol = results.get('symbol', 'Unknown Symbol')
        timeframe = results.get('timeframe', 'Unknown')
        start_date = results.get('start_date', datetime.now())
        end_date = results.get('end_date', datetime.now())
        capital = results.get('capital', 0)

        total_return = results.get('total_return', 0) * 100
        annual_return = results.get('annual_return', 0) * 100
        max_drawdown = results.get('max_drawdown', 0) * 100
        sharpe_ratio = results.get('sharpe_ratio', 0)

        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0) * 100
        profit_factor = results.get('profit_factor', 0)

        # Format dates
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else str(end_date)

        # Generate mock results HTML
        html_content = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6;">

<div style="text-align: center; margin-bottom: 20px; padding: 15px; background: linear-gradient(45deg, #FF9800, #2196F3); color: white; border-radius: 10px;">
<h2 style="margin: 0; color: white;">🧪 Mock Strategy Results</h2>
<p style="margin: 5px 0; opacity: 0.9;">Demo/Educational Mode</p>
</div>

<div style="background: #FFF3E0; padding: 15px; border-radius: 8px; border: 2px dashed #FF9800; margin-bottom: 15px;">
<h3 style="color: #E65100; margin-top: 0; text-align: center;">⚠️ DEMO MODE - Mock Results Only</h3>
<p style="color: #E65100; margin: 5px 0; text-align: center;">
These are simulated results for educational purposes.<br>
Real trading results may differ significantly.
</p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">

<div style="background: #E3F2FD; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3;">
<h3 style="color: #0D47A1; margin-top: 0; margin-bottom: 10px;">📋 Strategy Information</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px 0;"><b>Strategy:</b></td><td style="text-align: right;">{strategy_name}</td></tr>
<tr><td style="padding: 3px 0;"><b>Symbol:</b></td><td style="text-align: right;">{symbol}</td></tr>
<tr><td style="padding: 3px 0;"><b>Timeframe:</b></td><td style="text-align: right;">{timeframe}</td></tr>
<tr><td style="padding: 3px 0;"><b>Period:</b></td><td style="text-align: right;">{start_str} to {end_str}</td></tr>
<tr><td style="padding: 3px 0;"><b>Capital:</b></td><td style="text-align: right;">${capital:,.0f}</td></tr>
</table>
</div>

<div style="background: #E8F5E8; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50;">
<h3 style="color: #2E7D32; margin-top: 0; margin-bottom: 10px;">📈 Mock Performance</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px 0;"><b>Total Return:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if total_return > 0 else '#D32F2F'}; font-weight: bold;">
    {total_return:+.2f}%</td></tr>
<tr><td style="padding: 3px 0;"><b>Annual Return:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if annual_return > 0 else '#D32F2F'}; font-weight: bold;">
    {annual_return:+.2f}%</td></tr>
<tr><td style="padding: 3px 0;"><b>Max Drawdown:</b></td>
    <td style="text-align: right; color: #D32F2F; font-weight: bold;">{max_drawdown:.2f}%</td></tr>
<tr><td style="padding: 3px 0;"><b>Sharpe Ratio:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if sharpe_ratio > 1 else '#F57C00' if sharpe_ratio > 0 else '#D32F2F'}; font-weight: bold;">
    {sharpe_ratio:.3f}</td></tr>
</table>
</div>

</div>

<div style="background: #F3E5F5; padding: 15px; border-radius: 8px; border-left: 4px solid #9C27B0; margin-bottom: 15px;">
<h3 style="color: #7B1FA2; margin-top: 0; margin-bottom: 10px;">📊 Mock Trading Stats</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 3px 0; width: 60%;"><b>Total Trades:</b></td><td style="text-align: right; font-weight: bold;">{total_trades}</td></tr>
<tr><td style="padding: 3px 0;"><b>Win Rate:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if win_rate > 50 else '#F57C00' if win_rate > 30 else '#D32F2F'}; font-weight: bold;">
    {win_rate:.1f}%</td></tr>
<tr><td style="padding: 3px 0;"><b>Profit Factor:</b></td>
    <td style="text-align: right; color: {'#2E7D32' if profit_factor > 1 else '#D32F2F'}; font-weight: bold;">
    {profit_factor:.2f}</td></tr>
</table>
</div>

<div style="background: #E0F2F1; padding: 15px; border-radius: 8px; border-left: 4px solid #00796B;">
<h3 style="color: #004D40; margin-top: 0; margin-bottom: 10px;">🎓 Learning Points</h3>
<ul style="margin: 5px 0; padding-left: 20px;">
<li>This strategy shows <b>{'positive' if total_return > 0 else 'negative'}</b> returns in the test period</li>
<li>Risk level is <b>{'low' if max_drawdown < 10 else 'medium' if max_drawdown < 20 else 'high'}</b> with {max_drawdown:.1f}% max drawdown</li>
<li>Trading frequency: <b>{total_trades}</b> trades over the period</li>
<li>Success rate: <b>{win_rate:.0f}%</b> of trades were profitable</li>
</ul>
</div>

<div style="text-align: center; margin-top: 20px; padding: 15px; background: linear-gradient(45deg, #FFF3E0, #E3F2FD); border-radius: 8px; border: 2px dashed #FF9800;">
<p style="color: #E65100; font-weight: bold; margin: 5px 0;">🧪 Educational Demo Mode</p>
<p style="color: #666; margin: 5px 0;">
To run real backtests, install Howtrader and use production strategies
</p>
<p style="color: #666; margin: 5px 0; font-style: italic;">
Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</p>
</div>

</div>
        """

        self.result_label.setText(html_content)

    def _display_legacy_results(self, results: Dict[str, Any]):
        """Display legacy format results"""
        strategy = results.get('strategy', 'Unknown Strategy')
        total_return = results.get('total_return', 0) * 100
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0) * 100
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0) * 100

        html_content = f"""
<div style="font-family: Arial, sans-serif; text-align: center; padding: 20px;">
<h2 style="color: #1976D2;">📊 Legacy Backtest Results</h2>
<div style="background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0;">
<h3>{strategy}</h3>
<p><b>Total Return:</b> <span style="color: {'green' if total_return > 0 else 'red'};">{total_return:+.2f}%</span></p>
<p><b>Sharpe Ratio:</b> {sharpe_ratio:.2f}</p>
<p><b>Max Drawdown:</b> <span style="color: red;">{max_drawdown:.2f}%</span></p>
<p><b>Total Trades:</b> {total_trades}</p>
<p><b>Win Rate:</b> {win_rate:.1f}%</p>
</div>
</div>
        """

        self.result_label.setText(html_content)

    def _display_error(self, error_message: str, results: Dict[str, Any]):
        """Display error message with available data"""
        html_content = f"""
<div style="text-align: center; color: #D32F2F; padding: 20px;">
<h3>❌ Error Displaying Results</h3>
<p><b>Error:</b> {error_message}</p>
<p><b>Data type:</b> {type(results)}</p>
<div style="background: #ffebee; padding: 15px; border-radius: 5px; margin: 15px 0;">
<p><b>Available data keys:</b></p>
<p>{', '.join(results.keys()) if isinstance(results, dict) else 'Not a dictionary'}</p>
</div>
<p style="color: #666; margin-top: 15px;">Please check the backtest results format</p>
</div>
        """
        self.result_label.setText(html_content)

    def export_results(self):
        """Export results to text format"""
        if not self.results_data:
            return

        try:
            # Generate text summary
            mode = self.results_data.get('mode', 'unknown')
            strategy_name = self.results_data.get('strategy_name', 'Unknown')
            total_return = self.results_data.get('total_return', 0) * 100
            sharpe_ratio = self.results_data.get('sharpe_ratio', 0)
            max_drawdown = self.results_data.get('max_drawdown', 0) * 100

            export_text = f"""
HOWTRADER BACKTEST RESULTS
========================

Strategy: {strategy_name}
Mode: {mode.title()}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE SUMMARY
------------------
Total Return: {total_return:+.2f}%
Sharpe Ratio: {sharpe_ratio:.3f}
Max Drawdown: {max_drawdown:.2f}%

Raw Data: {self.results_data}
            """

            # For now, just print to console
            # In a full implementation, you'd show a save dialog
            print("EXPORTED RESULTS:")
            print(export_text)

        except Exception as e:
            print(f"Export failed: {e}")

    def refresh_results(self):
        """Refresh the results display"""
        if self.results_data:
            self.display_results(self.results_data)
        else:
            self.result_label.setText("📊 No results to refresh. Run a backtest first.")