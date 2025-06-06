"""
Timeframe Switcher Widget
========================

Advanced timeframe switching component với support cho multiple timeframes,
custom intervals, và seamless data conversion.

Author: AI Assistant
Version: 1.0
"""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap, QColor
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QButtonGroup,
    QGroupBox, QFrame, QCheckBox, QSpinBox,
    QSlider, QToolButton, QMenu, QAction
)

from ..config.display_config import DisplayConfig


class TimeframeType(Enum):
    """Timeframe categories"""
    TICK = "tick"
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class StandardTimeframes:
    """Standard trading timeframes"""

    # Tick & Seconds
    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"

    # Minutes
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"

    # Hours
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"

    # Daily+
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"

    @classmethod
    def get_all_timeframes(cls) -> List[str]:
        """Get all standard timeframes"""
        return [
            cls.TICK, cls.SECOND_1, cls.SECOND_5, cls.SECOND_15, cls.SECOND_30,
            cls.MINUTE_1, cls.MINUTE_3, cls.MINUTE_5, cls.MINUTE_15, cls.MINUTE_30,
            cls.HOUR_1, cls.HOUR_2, cls.HOUR_4, cls.HOUR_6, cls.HOUR_8, cls.HOUR_12,
            cls.DAILY, cls.WEEKLY, cls.MONTHLY
        ]

    @classmethod
    def get_by_category(cls, category: TimeframeType) -> List[str]:
        """Get timeframes by category"""
        mapping = {
            TimeframeType.TICK: [cls.TICK],
            TimeframeType.SECONDS: [cls.SECOND_1, cls.SECOND_5, cls.SECOND_15, cls.SECOND_30],
            TimeframeType.MINUTES: [cls.MINUTE_1, cls.MINUTE_3, cls.MINUTE_5, cls.MINUTE_15, cls.MINUTE_30],
            TimeframeType.HOURS: [cls.HOUR_1, cls.HOUR_2, cls.HOUR_4, cls.HOUR_6, cls.HOUR_8, cls.HOUR_12],
            TimeframeType.DAILY: [cls.DAILY, cls.WEEKLY, cls.MONTHLY]
        }
        return mapping.get(category, [])


class TimeframeButton(QPushButton):
    """Custom timeframe button with enhanced styling"""

    def __init__(self, timeframe: str, parent=None):
        super().__init__(timeframe, parent)
        self.timeframe = timeframe
        self.setCheckable(True)
        self.setup_styling()

    def setup_styling(self):
        """Setup button styling"""
        self.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: 11px;
                min-width: 40px;
                max-width: 60px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #999999;
            }
            QPushButton:checked {
                background-color: #2196F3;
                color: white;
                border-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1976D2;
            }
        """)


class CustomTimeframeDialog(QWidget):
    """Dialog for creating custom timeframes"""

    timeframe_created = Signal(str)  # custom timeframe string

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Timeframe")
        self.setWindowFlags(Qt.WindowType.Popup)
        self.init_ui()

    def init_ui(self):
        """Initialize custom timeframe UI"""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("Create Custom Timeframe")
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(title_label)

        # Input section
        input_layout = QGridLayout()

        # Value input
        input_layout.addWidget(QLabel("Value:"), 0, 0)
        self.value_spin = QSpinBox()
        self.value_spin.setRange(1, 999)
        self.value_spin.setValue(1)
        input_layout.addWidget(self.value_spin, 0, 1)

        # Unit selection
        input_layout.addWidget(QLabel("Unit:"), 1, 0)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["seconds", "minutes", "hours", "days", "weeks", "months"])
        self.unit_combo.setCurrentText("minutes")
        input_layout.addWidget(self.unit_combo, 1, 1)

        layout.addLayout(input_layout)

        # Preview
        self.preview_label = QLabel("Preview: 1m")
        self.preview_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        layout.addWidget(self.preview_label)

        # Connect signals
        self.value_spin.valueChanged.connect(self.update_preview)
        self.unit_combo.currentTextChanged.connect(self.update_preview)

        # Buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(cancel_btn)

        create_btn = QPushButton("Create")
        create_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        create_btn.clicked.connect(self.create_timeframe)
        button_layout.addWidget(create_btn)

        layout.addLayout(button_layout)

    def update_preview(self):
        """Update timeframe preview"""
        value = self.value_spin.value()
        unit = self.unit_combo.currentText()

        # Convert to standard format
        unit_map = {
            "seconds": "s",
            "minutes": "m",
            "hours": "h",
            "days": "d",
            "weeks": "w",
            "months": "M"
        }

        timeframe = f"{value}{unit_map[unit]}"
        self.preview_label.setText(f"Preview: {timeframe}")

    def create_timeframe(self):
        """Create and emit custom timeframe"""
        value = self.value_spin.value()
        unit = self.unit_combo.currentText()

        unit_map = {
            "seconds": "s",
            "minutes": "m",
            "hours": "h",
            "days": "d",
            "weeks": "w",
            "months": "M"
        }

        timeframe = f"{value}{unit_map[unit]}"
        self.timeframe_created.emit(timeframe)
        self.close()


class TimeframeSwitcher(QWidget):
    """
    Advanced timeframe switcher widget for chart display

    This component is part of the display package as it directly
    controls chart timeframe visualization and data presentation.
    """

    timeframe_changed = Signal(str)         # new timeframe
    custom_timeframe_added = Signal(str)    # custom timeframe
    multiple_timeframes_selected = Signal(list)  # for multi-timeframe analysis

    def __init__(self, config: DisplayConfig = None):
        super().__init__()
        self.config = config or DisplayConfig()

        # State
        self.current_timeframe = StandardTimeframes.HOUR_1
        self.custom_timeframes = []
        self.multi_select_mode = False
        self.selected_timeframes = []

        # UI components
        self.timeframe_buttons = {}
        self.button_group = QButtonGroup()
        self.custom_dialog = None

        self.init_ui()
        self.setup_default_timeframe()

    def init_ui(self):
        """Initialize timeframe switcher UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with controls
        header_layout = QHBoxLayout()

        title_label = QLabel("Timeframes")
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Multi-select toggle
        self.multi_select_cb = QCheckBox("Multi")
        self.multi_select_cb.setToolTip("Enable multiple timeframe selection")
        self.multi_select_cb.toggled.connect(self.toggle_multi_select)
        header_layout.addWidget(self.multi_select_cb)

        # Custom timeframe button
        custom_btn = QToolButton()
        custom_btn.setText("+")
        custom_btn.setToolTip("Add custom timeframe")
        custom_btn.setStyleSheet("""
            QToolButton {
                background-color: #FFA726;
                color: white;
                border-radius: 12px;
                width: 24px;
                height: 24px;
                font-weight: bold;
            }
            QToolButton:hover {
                background-color: #FF9800;
            }
        """)
        custom_btn.clicked.connect(self.show_custom_dialog)
        header_layout.addWidget(custom_btn)

        layout.addLayout(header_layout)

        # Main timeframe categories
        self.create_timeframe_categories(layout)

        # Custom timeframes section
        self.custom_group = QGroupBox("Custom")
        self.custom_layout = QHBoxLayout(self.custom_group)
        self.custom_group.setVisible(False)  # Hidden initially
        layout.addWidget(self.custom_group)

        # Quick actions
        self.create_quick_actions(layout)

    def create_timeframe_categories(self, parent_layout):
        """Create timeframe category sections"""

        # Standard timeframes by category
        categories = [
            ("Minutes", TimeframeType.MINUTES, True),
            ("Hours", TimeframeType.HOURS, True),
            ("Daily+", TimeframeType.DAILY, True),
            ("Seconds", TimeframeType.SECONDS, False),  # Collapsed by default
        ]

        for cat_name, cat_type, expanded in categories:
            group = QGroupBox(cat_name)
            group_layout = QHBoxLayout(group)
            group_layout.setSpacing(3)

            # Get timeframes for this category
            timeframes = StandardTimeframes.get_by_category(cat_type)

            # Create buttons
            for tf in timeframes:
                btn = TimeframeButton(tf)
                btn.clicked.connect(lambda checked, timeframe=tf: self.on_timeframe_selected(timeframe))

                self.timeframe_buttons[tf] = btn
                self.button_group.addButton(btn)
                group_layout.addWidget(btn)

            group_layout.addStretch()

            # Set initial visibility
            group.setVisible(expanded)
            if not expanded:
                group.setMaximumHeight(0)

            parent_layout.addWidget(group)

    def create_quick_actions(self, parent_layout):
        """Create quick action buttons"""
        quick_group = QGroupBox("Quick")
        quick_layout = QHBoxLayout(quick_group)

        # Popular timeframes
        popular = ["1m", "5m", "15m", "1h", "4h", "1d"]

        for tf in popular:
            if tf in self.timeframe_buttons:
                continue  # Skip if already exists

            btn = TimeframeButton(tf)
            btn.setStyleSheet(btn.styleSheet() + """
                QPushButton {
                    background-color: #E8F5E8;
                    border-color: #4CAF50;
                }
                QPushButton:hover {
                    background-color: #C8E6C9;
                }
                QPushButton:checked {
                    background-color: #4CAF50;
                }
            """)
            btn.clicked.connect(lambda checked, timeframe=tf: self.on_timeframe_selected(timeframe))

            self.timeframe_buttons[tf] = btn
            self.button_group.addButton(btn)
            quick_layout.addWidget(btn)

        quick_layout.addStretch()
        parent_layout.addWidget(quick_group)

    def setup_default_timeframe(self):
        """Setup default selected timeframe"""
        if self.current_timeframe in self.timeframe_buttons:
            self.timeframe_buttons[self.current_timeframe].setChecked(True)

    def on_timeframe_selected(self, timeframe: str):
        """Handle timeframe selection"""
        if self.multi_select_mode:
            # Multi-select mode
            if timeframe in self.selected_timeframes:
                self.selected_timeframes.remove(timeframe)
                self.timeframe_buttons[timeframe].setChecked(False)
            else:
                self.selected_timeframes.append(timeframe)
                self.timeframe_buttons[timeframe].setChecked(True)

            self.multiple_timeframes_selected.emit(self.selected_timeframes.copy())

        else:
            # Single select mode
            self.current_timeframe = timeframe

            # Uncheck all other buttons
            for tf, btn in self.timeframe_buttons.items():
                if tf != timeframe:
                    btn.setChecked(False)

            self.timeframe_buttons[timeframe].setChecked(True)
            self.timeframe_changed.emit(timeframe)

    def toggle_multi_select(self, enabled: bool):
        """Toggle multi-select mode"""
        self.multi_select_mode = enabled

        if enabled:
            # Switch to multi-select
            self.button_group.setExclusive(False)
            self.selected_timeframes = [self.current_timeframe] if self.current_timeframe else []

        else:
            # Switch to single select
            self.button_group.setExclusive(True)
            self.selected_timeframes = []

            # Ensure only current timeframe is selected
            for tf, btn in self.timeframe_buttons.items():
                btn.setChecked(tf == self.current_timeframe)

    def show_custom_dialog(self):
        """Show custom timeframe creation dialog"""
        if not self.custom_dialog:
            self.custom_dialog = CustomTimeframeDialog(self)
            self.custom_dialog.timeframe_created.connect(self.add_custom_timeframe)

        # Position dialog near the button
        pos = self.mapToGlobal(self.pos())
        self.custom_dialog.move(pos.x() + 100, pos.y() + 50)
        self.custom_dialog.show()

    def add_custom_timeframe(self, timeframe: str):
        """Add a custom timeframe"""
        if timeframe in self.custom_timeframes or timeframe in self.timeframe_buttons:
            return  # Already exists

        # Add to custom timeframes
        self.custom_timeframes.append(timeframe)

        # Create button
        btn = TimeframeButton(timeframe)
        btn.setStyleSheet(btn.styleSheet() + """
            QPushButton {
                background-color: #FFF3E0;
                border-color: #FF9800;
            }
            QPushButton:hover {
                background-color: #FFE0B2;
            }
            QPushButton:checked {
                background-color: #FF9800;
            }
        """)
        btn.clicked.connect(lambda checked, tf=timeframe: self.on_timeframe_selected(tf))

        # Add to layout
        self.custom_layout.addWidget(btn)
        self.timeframe_buttons[timeframe] = btn
        self.button_group.addButton(btn)

        # Show custom group
        self.custom_group.setVisible(True)

        # Emit signal
        self.custom_timeframe_added.emit(timeframe)

    def set_timeframe(self, timeframe: str):
        """Programmatically set current timeframe"""
        if timeframe in self.timeframe_buttons:
            self.on_timeframe_selected(timeframe)

    def get_current_timeframe(self) -> str:
        """Get currently selected timeframe"""
        return self.current_timeframe

    def get_selected_timeframes(self) -> List[str]:
        """Get all selected timeframes (for multi-select mode)"""
        if self.multi_select_mode:
            return self.selected_timeframes.copy()
        else:
            return [self.current_timeframe] if self.current_timeframe else []

    def add_preset_timeframes(self, timeframes: List[str]):
        """Add multiple preset timeframes"""
        for tf in timeframes:
            if tf not in self.timeframe_buttons:
                self.add_custom_timeframe(tf)

    def remove_custom_timeframe(self, timeframe: str):
        """Remove a custom timeframe"""
        if timeframe in self.custom_timeframes:
            self.custom_timeframes.remove(timeframe)

            # Remove button
            if timeframe in self.timeframe_buttons:
                btn = self.timeframe_buttons[timeframe]
                self.button_group.removeButton(btn)
                self.custom_layout.removeWidget(btn)
                btn.deleteLater()
                del self.timeframe_buttons[timeframe]

            # Hide custom group if empty
            if not self.custom_timeframes:
                self.custom_group.setVisible(False)

    def get_timeframe_info(self, timeframe: str) -> Dict[str, Any]:
        """Get detailed information about a timeframe"""
        # Parse timeframe string
        if timeframe == "tick":
            return {
                "type": TimeframeType.TICK,
                "seconds": 0,
                "minutes": 0,
                "display_name": "Tick"
            }

        # Extract number and unit
        import re
        match = re.match(r'(\d+)([smhdwM])', timeframe)
        if not match:
            return {"type": TimeframeType.MINUTES, "seconds": 60, "minutes": 1, "display_name": timeframe}

        value, unit = int(match.group(1)), match.group(2)

        # Convert to seconds
        unit_seconds = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
            'M': 2592000  # Approximate month
        }

        total_seconds = value * unit_seconds.get(unit, 60)

        # Determine type
        if unit == 's':
            tf_type = TimeframeType.SECONDS
        elif unit == 'm':
            tf_type = TimeframeType.MINUTES
        elif unit == 'h':
            tf_type = TimeframeType.HOURS
        elif unit == 'd':
            tf_type = TimeframeType.DAILY
        elif unit == 'w':
            tf_type = TimeframeType.WEEKLY
        else:
            tf_type = TimeframeType.MONTHLY

        return {
            "type": tf_type,
            "value": value,
            "unit": unit,
            "seconds": total_seconds,
            "minutes": total_seconds / 60,
            "display_name": timeframe
        }


def test_timeframe_switcher():
    """Test timeframe switcher functionality"""
    print("🕐 Testing Timeframe Switcher...")

    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Create timeframe switcher
    switcher = TimeframeSwitcher()
    print("✅ Timeframe switcher created")

    # Test signals
    def on_timeframe_changed(tf):
        print(f"📊 Timeframe changed to: {tf}")

    def on_custom_added(tf):
        print(f"➕ Custom timeframe added: {tf}")

    def on_multiple_selected(tfs):
        print(f"📋 Multiple timeframes selected: {tfs}")

    switcher.timeframe_changed.connect(on_timeframe_changed)
    switcher.custom_timeframe_added.connect(on_custom_added)
    switcher.multiple_timeframes_selected.connect(on_multiple_selected)

    # Test methods
    switcher.set_timeframe("5m")
    print(f"✅ Current timeframe: {switcher.get_current_timeframe()}")

    # Test timeframe info
    info = switcher.get_timeframe_info("1h")
    print(f"✅ 1h timeframe info: {info}")

    # Add custom timeframes
    switcher.add_preset_timeframes(["2m", "7m", "3h"])
    print("✅ Custom timeframes added")

    # Show switcher
    switcher.show()
    print("✅ Timeframe switcher shown")

    print("🎉 Timeframe switcher test completed!")

    # Don't start event loop in test
    switcher.close()


if __name__ == "__main__":
    test_timeframe_switcher()