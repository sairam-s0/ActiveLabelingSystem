# app/dialogs.py
"""
Dialog windows for the application
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QListWidget, QLineEdit, QAbstractItemView,
                              QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt


# Professional Color Palette (matching window.py)
COLORS = {
    "bg": "#f7f6f2",          # cream background
    "panel": "#ebe9e1",       # light panel
    "olive": "#556b2f",       # primary olive
    "olive_dark": "#3d4a2c",  # dark olive
    "accent": "#8b7a5c",      # accent brown
    "text": "#1f2415",        # dark text
    "success": "#4f7f5f",     # success green
    "warning": "#c2a45d",     # warning gold
    "danger": "#a05a54",      # danger red
    "muted": "#9a9688",       # muted gray
    "border": "#d4d0c8",      # subtle border
}


class ClassSelectorDialog(QDialog):
    """Dialog for selecting multiple detection classes."""

    def __init__(self, parent, all_classes, selected_classes):
        super().__init__(parent)
        self.setWindowTitle("Select Classes for Detection")
        self.setMinimumSize(500, 600)
        self.setStyleSheet(f"background-color: {COLORS['bg']};")
        self.all_classes = all_classes
        self.selected_classes = selected_classes.copy()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("Select Detection Classes")
        header.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {COLORS['olive_dark']};
                padding: 10px;
                background-color: {COLORS['panel']};
                border-radius: 4px;
                border-left: 4px solid {COLORS['olive']};
            }}
        """)
        layout.addWidget(header)
        
        # Instruction text
        instruction = QLabel("Select one or more classes for object detection:")
        instruction.setStyleSheet(f"""
            QLabel {{
                font-size: 12px;
                color: {COLORS['text']};
                padding: 5px;
            }}
        """)
        layout.addWidget(instruction)

        # Class list
        self.listbox = QListWidget()
        self.listbox.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.listbox.setStyleSheet(f"""
            QListWidget {{
                background-color: white;
                border: 2px solid {COLORS['border']};
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
                color: {COLORS['text']};
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {COLORS['border']};
            }}
            QListWidget::item:hover {{
                background-color: {COLORS['panel']};
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['olive']};
                color: white;
            }}
        """)
        
        for cls in self.all_classes:
            self.listbox.addItem(cls)
        for i in range(self.listbox.count()):
            if self.listbox.item(i).text() in self.selected_classes:
                self.listbox.item(i).setSelected(True)
        layout.addWidget(self.listbox, stretch=1)

        # Add custom class section
        add_label = QLabel("Add Custom Class:")
        add_label.setStyleSheet(f"""
            QLabel {{
                font-size: 11px;
                font-weight: bold;
                color: {COLORS['text']};
                padding: 5px 0;
            }}
        """)
        layout.addWidget(add_label)
        
        add_frame = QHBoxLayout()
        add_frame.setSpacing(8)
        
        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText("Enter new class name...")
        self.new_class_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: white;
                border: 2px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
                color: {COLORS['text']};
            }}
            QLineEdit:focus {{
                border: 2px solid {COLORS['olive']};
            }}
        """)
        add_frame.addWidget(self.new_class_input)
        
        add_btn = QPushButton("+ Add")
        add_btn.setFixedWidth(80)
        add_btn.setFixedHeight(35)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['olive_dark']};
            }}
        """)
        add_btn.clicked.connect(self.add_custom_class)
        add_frame.addWidget(add_btn)
        
        layout.addLayout(add_frame)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedSize(100, 40)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['muted']};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['danger']};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.setFixedSize(100, 40)
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['olive_dark']};
            }}
        """)
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)

    def add_custom_class(self):
        """Add a custom class to the list."""
        val = self.new_class_input.text().strip()
        if val and val not in self.all_classes:
            self.all_classes.append(val)
            self.all_classes.sort()
            self.listbox.clear()
            for cls in self.all_classes:
                self.listbox.addItem(cls)
            for i in range(self.listbox.count()):
                if self.listbox.item(i).text() in self.selected_classes:
                    self.listbox.item(i).setSelected(True)
            self.new_class_input.clear()

    def get_selected(self):
        """Return list of selected class names."""
        return [item.text() for item in self.listbox.selectedItems()]


class LabelFormatDialog(QDialog):
    """Dialog for selecting label output format."""

    def __init__(self, parent=None, current_format=None):
        super().__init__(parent)
        self.setWindowTitle("Label Output Format")
        self.setMinimumSize(360, 220)
        self.setStyleSheet(f"background-color: {COLORS['bg']};")
        self.current_format = current_format
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        header = QLabel("Select Label Format")
        header.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {COLORS['olive_dark']};
                padding: 8px;
                background-color: {COLORS['panel']};
                border-radius: 4px;
                border-left: 4px solid {COLORS['olive']};
            }}
        """)
        layout.addWidget(header)

        info = QLabel("Choose how labels should be saved in the images folder:")
        info.setStyleSheet(f"""
            QLabel {{
                font-size: 11px;
                color: {COLORS['text']};
                padding: 4px;
            }}
        """)
        layout.addWidget(info)

        self.group = QButtonGroup(self)
        self.coco_radio = QRadioButton("COCO JSON")
        self.json_radio = QRadioButton("Plain JSON")

        for rb in (self.coco_radio, self.json_radio):
            rb.setStyleSheet(f"""
                QRadioButton {{
                    font-size: 12px;
                    color: {COLORS['text']};
                    padding: 6px;
                }}
            """)

        self.group.addButton(self.coco_radio, 1)
        self.group.addButton(self.json_radio, 2)

        if self.current_format == "coco":
            self.coco_radio.setChecked(True)
        else:
            self.json_radio.setChecked(True)

        layout.addWidget(self.coco_radio)
        layout.addWidget(self.json_radio)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedSize(90, 36)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['muted']};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['danger']};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setFixedSize(90, 36)
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
        """)
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_format(self):
        if self.coco_radio.isChecked():
            return "coco"
        return "json"
