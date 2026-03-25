# toolbar_manager.py
from PyQt6.QtCore import QTimer

from features.toolbar_widget import FloatingToolbar
from features.toolbar_styles import AUTO_HIDE_DELAY, TOOLBAR_OFFSET_X, TOOLBAR_OFFSET_Y


class ToolbarManager:
    def __init__(self, canvas, manual_manager):
        self.canvas = canvas
        self.manual = manual_manager
        self.toolbar = FloatingToolbar(
            parent=self.canvas,
            callbacks={
                "add_another": lambda: None,
                "done": self.manual.finish_manual_labeling,
                "delete": self.manual._delete_last_box
            }
        )
        self._hide_timer = QTimer()
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

    def show_near(self, x, y):
        if self.canvas is None:
            return

        cw = self.canvas.width()
        ch = self.canvas.height()
        tw = self.toolbar.sizeHint().width()
        th = self.toolbar.sizeHint().height()

        px = int(x + TOOLBAR_OFFSET_X)
        py = int(y + TOOLBAR_OFFSET_Y)

        if px + tw > cw:
            px = max(0, int(x - tw - TOOLBAR_OFFSET_X))
        if py + th > ch:
            py = max(0, int(y - th - TOOLBAR_OFFSET_Y))

        self.toolbar.move(px, py)
        self.toolbar.show()
        self.toolbar.raise_()
        self._hide_timer.start(AUTO_HIDE_DELAY)

    def hide(self):
        self._hide_timer.stop()
        self.toolbar.hide()
