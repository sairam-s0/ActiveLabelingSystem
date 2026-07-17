# src/features/toolbar_widget.py
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt

from features.toolbar_styles import TOOLBAR_STYLE, BUTTON_ICONS


class FloatingToolbar(QWidget):
    def __init__(self, parent, callbacks):
        super().__init__(parent)
        self.callbacks = callbacks or {}
        self.setWindowFlags(Qt.WindowType.ToolTip)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        add_btn = QPushButton(f"{BUTTON_ICONS['add']} Add")
        add_btn.clicked.connect(self.callbacks.get("add_another", lambda: None))
        layout.addWidget(add_btn)

        done_btn = QPushButton(f"{BUTTON_ICONS['done']} Done")
        done_btn.clicked.connect(self.callbacks.get("done", lambda: None))
        layout.addWidget(done_btn)

        del_btn = QPushButton(BUTTON_ICONS["delete"])
        del_btn.setObjectName("delete_btn")
        del_btn.clicked.connect(self.callbacks.get("delete", lambda: None))
        layout.addWidget(del_btn)

        self.setStyleSheet(TOOLBAR_STYLE)
