# src/features/shortcut_config.py
from PyQt6.QtCore import Qt

SHORTCUTS = {
    "save_and_next": Qt.Key.Key_Space,
    "save_and_next_alt": Qt.Key.Key_Return,
    "exit_manual": Qt.Key.Key_Escape,
    "undo_box": Qt.Key.Key_Z,
    "delete_box": Qt.Key.Key_Delete,
    "class_1": Qt.Key.Key_1,
    "class_2": Qt.Key.Key_2,
    "class_3": Qt.Key.Key_3,
    "class_4": Qt.Key.Key_4,
    "class_5": Qt.Key.Key_5,
    "class_6": Qt.Key.Key_6,
    "class_7": Qt.Key.Key_7,
    "class_8": Qt.Key.Key_8,
    "class_9": Qt.Key.Key_9,
}

SHORTCUT_HELP = {
    "save_and_next": "Space: save and move to next image",
    "save_and_next_alt": "Enter: save and move to next image",
    "exit_manual": "Esc: exit manual mode",
    "undo_box": "Ctrl+Z: undo last box",
    "delete_box": "Delete: delete last box",
    "class_1": "1: select class 1",
    "class_2": "2: select class 2",
    "class_3": "3: select class 3",
    "class_4": "4: select class 4",
    "class_5": "5: select class 5",
    "class_6": "6: select class 6",
    "class_7": "7: select class 7",
    "class_8": "8: select class 8",
    "class_9": "9: select class 9",
}
