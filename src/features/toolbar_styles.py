# src/features/toolbar_styles.py

TOOLBAR_STYLE = """
QWidget {
    background-color: rgba(20, 20, 20, 210);
    border-radius: 8px;
}
QPushButton {
    background-color: #2f6f4e;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3d8a62;
}
QPushButton#delete_btn {
    background-color: #a05a54;
}
QPushButton#delete_btn:hover {
    background-color: #b46861;
}
"""

BUTTON_ICONS = {
    "add": "+",
    "done": "",
    "delete": "⌫"
}

AUTO_HIDE_DELAY = 3000
TOOLBAR_OFFSET_X = 10
TOOLBAR_OFFSET_Y = 10
