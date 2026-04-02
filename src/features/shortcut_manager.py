# src/features/shortcut_manager.py
from PyQt6.QtCore import Qt
from features.shortcut_config import SHORTCUTS


class ShortcutManager:
    def __init__(self, canvas, manual_manager):
        self.canvas = canvas
        self.manual = manual_manager
        self._orig_key_press = None

    def enable(self):
        if self.canvas is None:
            return
        self._orig_key_press = self.canvas.keyPressEvent
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()
        self.canvas.keyPressEvent = self._on_key_press

    def disable(self):
        if self.canvas is None:
            return
        if self._orig_key_press:
            self.canvas.keyPressEvent = self._orig_key_press
        self._orig_key_press = None

    def _on_key_press(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == SHORTCUTS["save_and_next"] or key == SHORTCUTS["save_and_next_alt"]:
            self.manual.finish_manual_labeling()
            return

        if key == SHORTCUTS["exit_manual"]:
            self.manual._cleanup()
            return

        if key == SHORTCUTS["delete_box"]:
            self.manual._delete_last_box()
            return

        if key == SHORTCUTS["undo_box"] and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.manual._undo_last_box()
            return

        for i in range(1, 10):
            if key == SHORTCUTS.get(f"class_{i}"):
                self.manual._set_manual_class_index(i - 1)
                return

        if self._orig_key_press:
            self._orig_key_press(event)
