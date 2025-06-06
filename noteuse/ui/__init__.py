"""
UI Module
=========

User interface components for backtest display application.
"""

# Define the widgets we want to expose
ResultDisplayWidget = None
ChartDisplayWidget = None

# Try to import from existing files first
try:
    from .result_display import BacktestResultsDisplay as ResultDisplayWidget

    print("✅ Loaded ResultDisplayWidget from results_display.py")
except ImportError:
    try:
        from .result_display import ResultDisplayWidget

        print("✅ Loaded ResultDisplayWidget from result_display.py")
    except ImportError:
        print("⚠️ No ResultDisplayWidget found, will create fallback")

try:
    from .chartish import AdvancedHowtraderChart as ChartDisplayWidget

    print("✅ Loaded ChartDisplayWidget from chartish.py")
except ImportError:
    try:
        from .chart_display import ChartDisplayWidget

        print("✅ Loaded ChartDisplayWidget from chart_display.py")
    except ImportError:
        print("⚠️ No ChartDisplayWidget found, will create fallback")

# Create fallback widgets if none were loaded
if ResultDisplayWidget is None:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


    class ResultDisplayWidget(QWidget):
        """Fallback result display widget"""

        def __init__(self):
            super().__init__()
            self.init_ui()

        def init_ui(self):
            layout = QVBoxLayout(self)

            self.result_label = QLabel(" Backtest Results")
            self.result_label.setStyleSheet("""
                QLabel {
                    padding: 20px;
                    border: 2px dashed #4CAF50;
                    border-radius: 10px;
                    background-color: #f0f8f0;
                    color: #2E7D32;
                    font-size: 14px;
                    text-align: center;
                }
            """)
            layout.addWidget(self.result_label)

        def display_results(self, results):
            """Display backtest results"""
            try:
                # Extract key metrics
                total_return = results.get('total_return', 0) * 100
                sharpe_ratio = results.get('sharpe_ratio', 0)
                max_drawdown = results.get('max_drawdown', 0) * 100
                win_rate = results.get('win_rate', 0) * 100
                total_trades = results.get('total_trades', 0)

                # Format display text
                display_text = f"""
<div style="text-align: center;">
<h3 style="color: #2196F3;"> Backtest Results Summary</h3>

<table style="margin: auto; border-collapse: collapse;">
<tr><td style="padding: 5px; text-align: right;"><b> Total Return:</b></td>
    <td style="padding: 5px; color: {'green' if total_return > 0 else 'red'};">{total_return:.2f}%</td></tr>
<tr><td style="padding: 5px; text-align: right;"><b> Sharpe Ratio:</b></td>
    <td style="padding: 5px;">{sharpe_ratio:.2f}</td></tr>
<tr><td style="padding: 5px; text-align: right;"><b> Max Drawdown:</b></td>
    <td style="padding: 5px; color: red;">{max_drawdown:.2f}%</td></tr>
<tr><td style="padding: 5px; text-align: right;"><b> Win Rate:</b></td>
    <td style="padding: 5px;">{win_rate:.1f}%</td></tr>
<tr><td style="padding: 5px; text-align: right;"><b> Total Trades:</b></td>
    <td style="padding: 5px;">{total_trades}</td></tr>
<tr><td style="padding: 5px; text-align: right;"><b> Mode:</b></td>
    <td style="padding: 5px;">{results.get('mode', 'Unknown').title()}</td></tr>
</table>

<p style="margin-top: 15px; color: #666; font-style: italic;">
 Results from {results.get('strategy_name', 'Unknown Strategy')}
</p>
</div>
                """.strip()

                self.result_label.setText(display_text)

            except Exception as e:
                self.result_label.setText(f"❌ Error displaying results: {e}")


    print("✅ Created fallback ResultDisplayWidget")

if ChartDisplayWidget is None:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


    class ChartDisplayWidget(QWidget):
        """Fallback chart display widget"""

        def __init__(self):
            super().__init__()
            self.init_ui()

        def init_ui(self):
            layout = QVBoxLayout(self)

            self.chart_label = QLabel(" Chart Display")
            self.chart_label.setStyleSheet("""
                QLabel {
                    padding: 20px;
                    border: 2px dashed #2196F3;
                    border-radius: 10px;
                    background-color: #f0f4f8;
                    color: #1976D2;
                    font-size: 14px;
                    text-align: center;
                    min-height: 300px;
                }
            """)
            layout.addWidget(self.chart_label)

        def display_chart(self, results):
            """Display chart for backtest results"""
            try:
                # Get some basic info from results
                total_return = results.get('total_return', 0) * 100
                total_trades = results.get('total_trades', 0)
                mode = results.get('mode', 'Unknown')

                # Create a simple chart description
                chart_info = f"""
<div style="text-align: center;">
<h3 style="color: #1976D2;"> Chart Analysis</h3>

<div style="margin: 20px;">
<h4 style="color: #2E7D32;"> Performance Overview</h4>
<p><b>Total Return:</b> <span style="color: {'green' if total_return > 0 else 'red'};">{total_return:.2f}%</span></p>
<p><b>Trade Count:</b> {total_trades}</p>
<p><b>Strategy:</b> {results.get('strategy_name', 'Unknown')}</p>
</div>

<div style="margin: 20px;">
<h4 style="color: #795548;"> Chart Components</h4>
<ul style="text-align: left; max-width: 300px; margin: auto;">
<li>OHLC Candlestick Data</li>
<li>Entry/Exit Points</li>
<li>Moving Averages</li>
<li>Volume Analysis</li>
<li>P&L Visualization</li>
</ul>
</div>

<p style="color: #666; font-style: italic; margin-top: 20px;">
 Mode: {mode.title()}<br>
 Install pyqtgraph for advanced charts
</p>
</div>
                """.strip()

                self.chart_label.setText(chart_info)

            except Exception as e:
                self.chart_label.setText(f"❌ Error displaying chart: {e}")


    print("✅ Created fallback ChartDisplayWidget")

# Export the classes
__all__ = ['ResultDisplayWidget', 'ChartDisplayWidget']


def create_result_widget():
    """Create result display widget"""
    return ResultDisplayWidget()


def create_chart_widget():
    """Create chart display widget"""
    return ChartDisplayWidget()


def get_available_widgets():
    """Get list of available UI widgets"""
    return {
        'result_display': ResultDisplayWidget is not None,
        'chart_display': ChartDisplayWidget is not None,
    }