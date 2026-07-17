# src/features/manual.py

import hashlib
from datetime import datetime
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QRadioButton, QButtonGroup,
                              QScrollArea, QWidget, QFrame,
                              QMessageBox)
from PyQt6.QtCore import Qt, QRect, QTimer, QObject, QPointF

from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF

from features.shortcut_manager import ShortcutManager
from features.toolbar_manager import ToolbarManager


from app.theme import COLORS
from core.sam_adapter import segment_boxes, segment_point


def default_color_for_name(name: str) -> str:
    h = int(hashlib.md5(name.encode('utf-8')).hexdigest()[:6], 16)
    return f"#{h:06x}"


def safe_class_name(name: str) -> str:
    return ' '.join(name.strip().split())


# drawingoverlay class
class DrawingOverlay(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.rect = None
        self.polygon_points = []
        self.color = QColor("red")
        self.completed_boxes = []  # store qrect
        self.completed_regions = []
        self.highlight_enabled = True

    def set_completed_shapes(self, boxes, regions):
        self.completed_boxes = boxes
        self.completed_regions = regions
        self.update()

    def set_highlight_enabled(self, enabled):
        self.highlight_enabled = bool(enabled)
        self.update()

    def set_completed_boxes(self, boxes):
        self.set_completed_shapes(boxes, self.completed_regions)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # draw completed
        for box_rect, cls_name, color in self.completed_boxes:
            shape_color = QColor(color)
            if self.highlight_enabled:
                fill = QColor(shape_color)
                fill.setAlpha(45)
                painter.fillRect(box_rect, fill)

            pen = QPen(shape_color, 3)
            painter.setPen(pen)
            painter.drawRect(box_rect)
            
            # draw label
            font = painter.font()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            
            label_text = cls_name
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(label_text)
            text_height = metrics.height()
            
            label_rect = QRect(
                box_rect.x(),
                box_rect.y() - text_height - 6,
                text_width + 8,
                text_height + 4
            )
            
            # draw label
            painter.fillRect(label_rect, QColor(color))
            painter.setPen(QColor("white"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label_text)

        for points, cls_name, color in self.completed_regions:
            if len(points) < 2:
                continue
            qpoints = [QPointF(float(x), float(y)) for x, y in points]
            polygon = QPolygonF(qpoints)
            fill = QColor(color)
            fill.setAlpha(95 if self.highlight_enabled else 45)
            painter.setBrush(QBrush(fill) if self.highlight_enabled else Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(color), 3))
            if len(qpoints) >= 3:
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)

            first = qpoints[0]
            label_rect = QRect(int(first.x()), int(first.y()) - 22, 120, 18)
            painter.fillRect(label_rect, QColor(color))
            painter.setPen(QColor("white"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, cls_name)
        
        # draw current
        if self.rect:
            pen = QPen(QColor(COLORS['warning']), 3, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.rect)

        if self.polygon_points:
            qpoints = [QPointF(float(x), float(y)) for x, y in self.polygon_points]
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(COLORS['warning']), 3, Qt.PenStyle.DashLine))
            painter.drawPolyline(QPolygonF(qpoints))
            painter.setBrush(QBrush(QColor(COLORS['warning'])))
            for point in qpoints:
                painter.drawEllipse(point, 4, 4)


class ManualToolbox(QDialog):
    
    def __init__(
        self,
        parent,
        classes,
        current_class,
        on_done,
        on_exit,
        on_mode_changed,
        on_finish_region,
        on_highlight_changed,
        on_undo_point,
    ):
        super().__init__(parent)
        self.setWindowTitle("Manual Labeling Toolbox")
        self.setMinimumSize(260, 480)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(f"background-color: {COLORS['bg']};")
        self.on_done = on_done
        self.on_exit = on_exit
        self.on_mode_changed = on_mode_changed
        self.on_finish_region = on_finish_region
        self.on_highlight_changed = on_highlight_changed
        self.on_undo_point = on_undo_point
        self.current_class = current_class
        self.mode_group = None
        self.setup_ui(classes)
        
    def setup_ui(self, classes):
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # header
        header = QLabel(" Manual Labeling")
        header.setStyleSheet(f"""
            QLabel {{
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                background-color: {COLORS['olive']};
                color: white;
                border-radius: 4px;
            }}
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # instructions
        info = QLabel(
            "1. Select a class below\n"
            "2. Box: click & drag\n"
            "3. Freehand Region: click points, double-click to close\n"
            "   (Backspace / Ctrl+Z to undo points)\n"
            "4. SAM: click object to add detections\n"
            "5. Click Done to save all"
        )
        info.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text']};
                font-size: 10px;
                padding: 10px;
                background-color: {COLORS['panel']};
                border-radius: 4px;
                border-left: 3px solid {COLORS['olive']};
            }}
        """)
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self.class_count_label = QLabel(f"Classes: {len(classes)}")
        self.class_count_label.setStyleSheet(f"""
            QLabel {{
                font-size: 10px;
                color: {COLORS['muted']};
                font-weight: bold;
            }}
        """)
        layout.addWidget(self.class_count_label)

        mode_frame = QFrame()
        mode_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 6px;
            }}
        """)
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(8, 6, 8, 6)
        mode_layout.setSpacing(10)

        self.mode_group = QButtonGroup(self)
        box_mode = QRadioButton("Box")
        region_mode = QRadioButton("Freehand Region")
        sam_mode = QRadioButton("SAM Click")
        box_mode.setChecked(True)
        for idx, radio in enumerate((box_mode, region_mode, sam_mode)):
            radio.setStyleSheet(f"color: {COLORS['text']}; font-size: 11px; font-weight: bold;")
            self.mode_group.addButton(radio, idx)
            mode_layout.addWidget(radio)
        self.mode_group.buttonClicked.connect(lambda btn: self.on_mode_changed(self._mode_from_button(btn)))
        layout.addWidget(mode_frame)

        self.highlight_btn = QPushButton("Highlight On")
        self.highlight_btn.setCheckable(True)
        self.highlight_btn.setChecked(True)
        self.highlight_btn.setFixedHeight(32)
        self.highlight_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['panel']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:checked {{
                background-color: {COLORS['accent']};
                color: white;
                border: 1px solid {COLORS['accent']};
            }}
        """)
        self.highlight_btn.toggled.connect(self._on_highlight_toggled)
        layout.addWidget(self.highlight_btn)
        
        # scrollable class
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 2px solid {COLORS['border']};
                border-radius: 4px;
                background-color: {COLORS['bg']};
            }}
        """)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(6)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        
        self.button_group = QButtonGroup(self)
        
        for idx, cls in enumerate(classes):
            color = default_color_for_name(cls)
            frame = QFrame()
            frame.setStyleSheet(f"background-color: {COLORS['bg']}; border: 1px solid {COLORS['border']}; border-radius: 4px; padding: 2px;")
            frame_layout = QHBoxLayout(frame)
            frame_layout.setContentsMargins(8, 6, 8, 6)
            frame_layout.setSpacing(10)
            
            # color indicator
            color_label = QLabel("  ")
            color_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    border: 2px solid {COLORS['text']};
                    border-radius: 3px;
                }}
            """)
            color_label.setFixedSize(18, 18)
            frame_layout.addWidget(color_label)
            
            # radio button
            radio = QRadioButton(cls)
            radio.setStyleSheet(f"""
                QRadioButton {{
                    font-size: 11px;
                    color: {COLORS['text']};
                    font-weight: bold;
                }}
                QRadioButton::indicator {{
                    width: 16px;
                    height: 16px;
                }}
                QRadioButton::indicator::unchecked {{
                    border: 2px solid {COLORS['muted']};
                    border-radius: 8px;
                    background-color: {COLORS['panel']};
                }}
                QRadioButton::indicator::checked {{
                    border: 2px solid {COLORS['olive']};
                    border-radius: 8px;
                    background-color: {COLORS['olive']};
                }}
            """)
            
            if cls == self.current_class:
                radio.setChecked(True)
            
            self.button_group.addButton(radio, idx)
            frame_layout.addWidget(radio)
            frame_layout.addStretch()
            
            scroll_layout.addWidget(frame)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # box count
        self.box_count_label = QLabel("Shapes drawn: 0")
        self.box_count_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text']};
                font-size: 11px;
                font-weight: bold;
                padding: 8px;
                background-color: {COLORS['panel']};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.box_count_label)
        
        # action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.finish_region_btn = QPushButton("Finish Region")
        self.finish_region_btn.setFixedHeight(40)
        self.finish_region_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
        """)
        self.finish_region_btn.clicked.connect(self.on_finish_region)
        self.finish_region_btn.setVisible(False)
        btn_layout.addWidget(self.finish_region_btn)

        self.undo_point_btn = QPushButton("Undo Point")
        self.undo_point_btn.setFixedHeight(40)
        self.undo_point_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
        """)
        self.undo_point_btn.clicked.connect(self.on_undo_point)
        self.undo_point_btn.setVisible(False)
        btn_layout.addWidget(self.undo_point_btn)
        
        done_btn = QPushButton(" Done")
        done_btn.setFixedHeight(40)
        done_btn.setStyleSheet(f"""
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
        """)
        done_btn.clicked.connect(self.on_done)
        btn_layout.addWidget(done_btn)
        
        exit_btn = QPushButton(" Exit Manual")
        exit_btn.setFixedHeight(40)
        exit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive_dark']};
            }}
        """)
        exit_btn.clicked.connect(self.on_exit)
        btn_layout.addWidget(exit_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def update_mode_ui(self, mode):
        is_polygon = (mode == "polygon")
        self.finish_region_btn.setVisible(is_polygon)
        self.undo_point_btn.setVisible(is_polygon)
    
    def get_selected_class(self):
        checked = self.button_group.checkedButton()
        return checked.text() if checked else None
    
    def update_box_count(self, count):
        self.box_count_label.setText(f"Shapes drawn: {count}")

    def _mode_from_button(self, button):
        text = button.text()
        if text.startswith("Freehand"):
            return "polygon"
        if text.startswith("SAM"):
            return "sam"
        return "box"

    def _on_highlight_toggled(self, checked):
        self.highlight_btn.setText("Highlight On" if checked else "Highlight Off")
        self.on_highlight_changed(checked)


class ManualManager(QObject):
    
    def __init__(self, host, window, state):
        super().__init__(window)
        self.host = host
        self.window = window
        self.state = state
        
        # manual mode
        self._active = False
        self._overlay = None
        self._toolbox = None
        self._start_point = None
        self._polygon_points = []
        self._manual_mode = "box"
        self._current_manual_class = None
        self._manual_boxes = []  # list of
        self._box_history = []  # list of
        self._active_image_path = None  # track which
        self._event_filter_disabled = False  # flag to
        self.shortcuts = None
        self.toolbar = None
        
    def start_manual_labeling(self):
        if self._active:
            print("[Manual] Already in manual mode")
            return
        
        # require class
        if not self.state.selected_classes:
            QMessageBox.warning(
                self.window,
                "No Classes Selected",
                "Please select at least one class before starting manual labeling."
            )
            return
        
        # require image
        if not self.state.current_image:
            QMessageBox.warning(
                self.window,
                "No Image",
                "Please load an image first."
            )
            return
        
        self._active = True
        self._event_filter_disabled = False  # re enable
        self._manual_boxes = []
        self._box_history = []
        self._start_point = None
        self._polygon_points = []
        self._manual_mode = "box"
        self._active_image_path = str(self.state.current_image_path)
        self._current_manual_class = self.state.selected_classes[0]

        try:
            # create overlay
            self._overlay = DrawingOverlay(self.window.canvas_label)
            self._overlay.setGeometry(self.window.canvas_label.rect())
            self._overlay.show()

            # install event
            self.window.canvas_label.installEventFilter(self)

            # create floating
            self._toolbox = ManualToolbox(
                parent=self.window,
                classes=self.state.selected_classes,
                current_class=self._current_manual_class,
                on_done=self.finish_manual_labeling,
                on_exit=self.exit_manual_mode,
                on_mode_changed=self._set_manual_mode,
                on_finish_region=self._finish_polygon_region,
                on_highlight_changed=self._set_highlight_enabled,
                on_undo_point=self._undo_last_point,
            )
            self._toolbox.show()
            self.shortcuts = ShortcutManager(self.window.canvas_label, self)
            self.shortcuts.enable()
            self.toolbar = ToolbarManager(self.window.canvas_label, self)
        except Exception as e:
            self._cleanup()
            QMessageBox.critical(
                self.window,
                "Manual Mode Error",
                f"Failed to start manual labeling:\n{e}"
            )
            return
        
        print(f"[Manual] Starting manual mode with classes: {self.state.selected_classes}")
    
    def exit_manual_mode(self):
        if not self._active:
            return
        
        # confirm exit
        if self._manual_boxes:
            reply = QMessageBox.question(
                self.window,
                "Exit Manual Mode",
                f"You have {len(self._manual_boxes)} unsaved box(es).\n\n"
                "Exit without saving?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        self._cleanup()
    
    def on_image_changed(self):
        if not self._active:
            return
        
        # update active
        if self.state.current_image_path:
            self._active_image_path = str(self.state.current_image_path)
        
        # reset box
        self._manual_boxes = []
        self._box_history = []
        self._start_point = None
        self._polygon_points = []
        
        # update overlay
        if self._overlay:
            try:
                self._overlay.setGeometry(self.window.canvas_label.rect())
                self._overlay.polygon_points = []
                self._overlay.set_completed_shapes([], [])
                self._overlay.update()
            except RuntimeError:
                # overlay was
                self._overlay = None
        
        # update toolbox
        if self._toolbox:
            try:
                self._toolbox.update_box_count(0)
            except RuntimeError:
                self._toolbox = None

    def _set_manual_class_index(self, index: int):
        if not self.state.selected_classes:
            return
        if index < 0 or index >= len(self.state.selected_classes):
            return
        self._current_manual_class = self.state.selected_classes[index]
        if self._toolbox:
            try:
                buttons = self._toolbox.button_group.buttons()
                if index < len(buttons):
                    buttons[index].setChecked(True)
            except Exception:
                pass

    def _set_manual_mode(self, mode: str):
        if mode not in {"box", "polygon", "sam"}:
            return
        if self._manual_mode == "polygon" and self._polygon_points:
            self._finish_polygon_region()
        self._manual_mode = mode
        self._start_point = None
        if self._overlay:
            self._overlay.rect = None
            self._overlay.polygon_points = []
            self._overlay.update()

    def _set_highlight_enabled(self, enabled):
        if self._overlay:
            try:
                self._overlay.set_highlight_enabled(enabled)
            except RuntimeError:
                self._overlay = None
    
    def _bind_mouse(self):
        self.window.canvas_label.installEventFilter(self)
    
    def _unbind_mouse(self):
        self._event_filter_disabled = True  # disable event
        try:
            self.window.canvas_label.removeEventFilter(self)
        except RuntimeError:
            pass  # canvas was
    
    def eventFilter(self, obj, event):
        # early return
        if (self._event_filter_disabled or 
            not self._active or 
            obj != self.window.canvas_label or 
            not self.window.canvas_label.isVisible()):
            return False
        
        from PyQt6.QtCore import QEvent
        
        if event.type() == QEvent.Type.MouseButtonPress:
            self._on_mouse_press(event)
            return True
        elif event.type() == QEvent.Type.MouseButtonDblClick:
            self._on_mouse_double_click(event)
            return True
        elif event.type() == QEvent.Type.MouseMove:
            self._on_mouse_move(event)
            return True
        elif event.type() == QEvent.Type.MouseButtonRelease:
            self._on_mouse_release(event)
            return True
        
        return False

    def _safe_scale_factor(self):
        scale = getattr(self.host, "scale_factor", None)
        if not isinstance(scale, (int, float)) or scale <= 0:
            return 1.0
        return float(scale)
    
    def _get_image_coords(self, event_pos):
        try:
            pixmap = self.window.canvas_label.pixmap()
        except RuntimeError:
            return None, None
            
        if not pixmap:
            return None, None
        
        canvas_w = self.window.canvas_label.width()
        canvas_h = self.window.canvas_label.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        
        offset_x = (canvas_w - pixmap_w) / 2.0
        offset_y = (canvas_h - pixmap_h) / 2.0
        
        scale = self._safe_scale_factor()
        
        # convert to
        img_x = (event_pos.x() - offset_x) / scale
        img_y = (event_pos.y() - offset_y) / scale
        
        return img_x, img_y

    def _get_display_coords(self, img_x, img_y):
        try:
            pixmap = self.window.canvas_label.pixmap()
        except RuntimeError:
            return None

        if not pixmap:
            return None

        canvas_w = self.window.canvas_label.width()
        canvas_h = self.window.canvas_label.height()
        offset_x = (canvas_w - pixmap.width()) / 2.0
        offset_y = (canvas_h - pixmap.height()) / 2.0
        scale = self._safe_scale_factor()
        return img_x * scale + offset_x, img_y * scale + offset_y

    def _clamp_image_point(self, img_x, img_y):
        try:
            img_w, img_h = self.state.current_image.size
            img_x = max(0.0, min(float(img_w), float(img_x)))
            img_y = max(0.0, min(float(img_h), float(img_y)))
        except Exception:
            pass
        return img_x, img_y
    
    def _on_mouse_press(self, event):
        # get current
        if self._toolbox:
            try:
                selected = self._toolbox.get_selected_class()
                if selected:
                    self._current_manual_class = selected
            except RuntimeError:
                self._toolbox = None
        
        img_x, img_y = self._get_image_coords(event.position())
        if img_x is None:
            return

        if self._manual_mode == "sam":
            if event.button() == Qt.MouseButton.LeftButton:
                img_x, img_y = self._clamp_image_point(img_x, img_y)
                self._add_sam_matches_at_point(img_x, img_y)
            return

        if self._manual_mode == "polygon":
            if event.button() == Qt.MouseButton.RightButton:
                self._finish_polygon_region()
                return

            img_x, img_y = self._clamp_image_point(img_x, img_y)
            if len(self._polygon_points) >= 3:
                first_x, first_y = self._polygon_points[0]
                if abs(img_x - first_x) <= 8 and abs(img_y - first_y) <= 8:
                    self._finish_polygon_region()
                    return

            self._polygon_points.append((img_x, img_y))
            self._update_polygon_preview()
            return
        
        self._start_point = (img_x, img_y)
    
    def _on_mouse_move(self, event):
        if self._manual_mode in {"polygon", "sam"}:
            return

        if not self._start_point or not self._overlay:
            return
        
        # safety check
        try:
            if not self._overlay.isVisible():
                return
        except RuntimeError:
            self._overlay = None
            return
        
        img_x, img_y = self._get_image_coords(event.position())
        if img_x is None:
            return
        
        x1, y1 = self._start_point
        
        # convert to
        try:
            pixmap = self.window.canvas_label.pixmap()
        except RuntimeError:
            return
            
        if not pixmap:
            return
        
        canvas_w = self.window.canvas_label.width()
        canvas_h = self.window.canvas_label.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        
        offset_x = (canvas_w - pixmap_w) / 2.0
        offset_y = (canvas_h - pixmap_h) / 2.0
        
        scale = self._safe_scale_factor()
        
        display_x1 = x1 * scale + offset_x
        display_y1 = y1 * scale + offset_y
        display_x2 = img_x * scale + offset_x
        display_y2 = img_y * scale + offset_y
        
        rect = QRect(
            int(min(display_x1, display_x2)),
            int(min(display_y1, display_y2)),
            int(abs(display_x2 - display_x1)),
            int(abs(display_y2 - display_y1))
        )
        
        try:
            self._overlay.rect = rect
            self._overlay.update()
        except RuntimeError:
            self._overlay = None
    
    def _on_mouse_release(self, event):
        if self._manual_mode in {"polygon", "sam"}:
            return

        if not self._start_point:
            return
        
        img_x, img_y = self._get_image_coords(event.position())
        if img_x is None:
            return
        
        x1, y1 = self._start_point
        x2, y2 = img_x, img_y
        
        # normalize coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # clamp to
        try:
            img_w, img_h = self.state.current_image.size
            x1 = max(0.0, min(float(img_w), x1))
            x2 = max(0.0, min(float(img_w), x2))
            y1 = max(0.0, min(float(img_h), y1))
            y2 = max(0.0, min(float(img_h), y2))
        except Exception:
            pass
        
        # check if
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            cls_name = safe_class_name(self._current_manual_class or "manual")
            box = [x1, y1, x2, y2, cls_name]
            self._manual_boxes.append(box)
            self._box_history.append(list(box))
            print(f"[Manual] Box added: {cls_name} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            # update toolbox
            if self._toolbox:
                try:
                    self._toolbox.update_box_count(len(self._manual_boxes))
                except RuntimeError:
                    # toolbox was
                    self._toolbox = None
            
            # show toolbar
            if self.toolbar:
                try:
                    pixmap = self.window.canvas_label.pixmap()
                except RuntimeError:
                    pixmap = None
                if pixmap:
                    canvas_w = self.window.canvas_label.width()
                    canvas_h = self.window.canvas_label.height()
                    pixmap_w = pixmap.width()
                    pixmap_h = pixmap.height()
                    offset_x = (canvas_w - pixmap_w) / 2.0
                    offset_y = (canvas_h - pixmap_h) / 2.0
                    scale = self._safe_scale_factor()
                    disp_x = x2 * scale + offset_x
                    disp_y = y2 * scale + offset_y
                    self.toolbar.show_near(disp_x, disp_y)
        
        # clear current
        self._start_point = None
        if self._overlay:
            try:
                self._overlay.rect = None
            except RuntimeError:
                self._overlay = None
                return
        
        # redraw with
        self._redraw_with_boxes()

    def _on_mouse_double_click(self, event):
        if self._manual_mode != "polygon":
            return
        img_x, img_y = self._get_image_coords(event.position())
        if img_x is not None and event.button() == Qt.MouseButton.LeftButton:
            img_x, img_y = self._clamp_image_point(img_x, img_y)
            if not self._polygon_points or self._polygon_points[-1] != (img_x, img_y):
                self._polygon_points.append((img_x, img_y))
        self._finish_polygon_region()

    def _update_polygon_preview(self):
        if not self._overlay:
            return

        display_points = []
        for img_x, img_y in self._polygon_points:
            display = self._get_display_coords(img_x, img_y)
            if display:
                display_points.append(display)

        try:
            self._overlay.polygon_points = display_points
            self._overlay.update()
        except RuntimeError:
            self._overlay = None

    def _finish_polygon_region(self):
        if len(self._polygon_points) < 3:
            self._polygon_points = []
            if self._overlay:
                self._overlay.polygon_points = []
                self._overlay.update()
            return

        xs = [p[0] for p in self._polygon_points]
        ys = [p[1] for p in self._polygon_points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if abs(x2 - x1) <= 5 or abs(y2 - y1) <= 5:
            self._polygon_points = []
            return

        cls_name = safe_class_name(self._current_manual_class or "manual")
        item = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "segmentation": [[[float(x), float(y)] for x, y in self._polygon_points]],
            "class": cls_name,
            "manual": True,
            "shape": "polygon",
        }
        self._manual_boxes.append(item)
        self._box_history.append(item.copy())
        print(f"[Manual] Region added: {cls_name} with {len(self._polygon_points)} points")

        self._polygon_points = []
        if self._overlay:
            self._overlay.polygon_points = []

        if self._toolbox:
            try:
                self._toolbox.update_box_count(len(self._manual_boxes))
            except RuntimeError:
                self._toolbox = None

        self._redraw_with_boxes()

    def _add_sam_matches_at_point(self, img_x, img_y):
        img_path = str(self.state.current_image_path)
        cls_name = safe_class_name(self._current_manual_class or "manual")
        detections = list(getattr(self.state, "current_detections", []) or [])
        clicked = self._find_detection_at_point(detections, img_x, img_y)

        if clicked:
            clicked_class = clicked.get("class")
            if clicked_class:
                cls_name = safe_class_name(clicked_class)
            candidates = [
                det for det in detections
                if det.get("class") == clicked.get("class") and self._valid_bbox(det.get("bbox"))
            ]
            if not candidates:
                candidates = [clicked]

            segmented = segment_boxes(img_path, [dict(det) for det in candidates])
            added = 0
            for det in segmented:
                item = self._manual_item_from_detection(det, cls_name, "sam_similar")
                if item and not self._has_similar_manual_shape(item["bbox"], cls_name):
                    self._manual_boxes.append(item)
                    self._box_history.append(item.copy())
                    added += 1
            print(f"[Manual] SAM added {added} similar {cls_name} object(s)")
        else:
            result = segment_point(img_path, img_x, img_y)
            if not result:
                QMessageBox.information(
                    self.window,
                    "SAM",
                    "SAM could not segment that click. Try clicking closer to the center of the object."
                )
                return
            item = self._manual_item_from_detection(result, cls_name, "sam_point")
            if item:
                self._manual_boxes.append(item)
                self._box_history.append(item.copy())
                print(f"[Manual] SAM point region added: {cls_name}")

        if self._toolbox:
            try:
                self._toolbox.update_box_count(len(self._manual_boxes))
            except RuntimeError:
                self._toolbox = None
        self._redraw_with_boxes()

    def _find_detection_at_point(self, detections, img_x, img_y):
        matches = []
        for det in detections:
            bbox = det.get("bbox")
            if not self._valid_bbox(bbox):
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                matches.append((area, det))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    def _manual_item_from_detection(self, det, cls_name, source):
        bbox = det.get("bbox")
        if not self._valid_bbox(bbox):
            return None
        item = {
            "bbox": [float(v) for v in bbox],
            "class": cls_name,
            "manual": True,
            "shape": "polygon" if det.get("segmentation") else "box",
            "mask_source": source,
        }
        if det.get("segmentation"):
            item["segmentation"] = det["segmentation"]
        return item

    def _valid_bbox(self, bbox):
        if not bbox or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return x2 - x1 > 5 and y2 - y1 > 5

    def _has_similar_manual_shape(self, bbox, cls_name):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        for existing in self._manual_boxes:
            if isinstance(existing, dict):
                ex_cls = existing.get("class")
                ex_box = existing.get("bbox")
            else:
                ex_cls = existing[4]
                ex_box = existing[:4]
            if ex_cls != cls_name or not self._valid_bbox(ex_box):
                continue
            ex1, ey1, ex2, ey2 = [float(v) for v in ex_box]
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area = (x2 - x1) * (y2 - y1)
            ex_area = (ex2 - ex1) * (ey2 - ey1)
            union = area + ex_area - inter
            if union > 0 and inter / union > 0.85:
                return True
        return False
    
    def _redraw_with_boxes(self):
        if not self.state.current_image or not self._overlay:
            return
        
        # safety check
        try:
            if not self._overlay.isVisible():
                return
        except RuntimeError:
            # overlay was
            self._overlay = None
            return
        
        # safety check
        current_image_path = str(self.state.current_image_path)
        if self._active_image_path != current_image_path:
            return
        
        # get canvas
        try:
            pixmap = self.window.canvas_label.pixmap()
        except RuntimeError:
            return
            
        if not pixmap:
            return
        
        canvas_w = self.window.canvas_label.width()
        canvas_h = self.window.canvas_label.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        
        offset_x = (canvas_w - pixmap_w) / 2.0
        offset_y = (canvas_h - pixmap_h) / 2.0
        
        scale = self._safe_scale_factor()
        
        # convert manual
        overlay_boxes = []
        overlay_regions = []
        for box in self._manual_boxes:
            if isinstance(box, dict):
                cls_name = box.get("class", "manual")
                color = default_color_for_name(cls_name)
                has_region = False

                # Render segmentation polygon if present
                seg = box.get("segmentation")
                if seg and seg[0]:
                    region_points = []
                    for img_x, img_y in seg[0]:
                        display = self._get_display_coords(img_x, img_y)
                        if display:
                            region_points.append(display)
                    if region_points:
                        overlay_regions.append((region_points, cls_name, color))
                        has_region = True

                # Always render bbox as a box overlay (even with segmentation for visibility)
                bbox = box.get("bbox")
                if bbox and len(bbox) == 4:
                    bx1, by1, bx2, by2 = [float(v) for v in bbox]
                    display_x1 = bx1 * scale + offset_x
                    display_y1 = by1 * scale + offset_y
                    display_x2 = bx2 * scale + offset_x
                    display_y2 = by2 * scale + offset_y
                    rect = QRect(
                        int(display_x1),
                        int(display_y1),
                        int(display_x2 - display_x1),
                        int(display_y2 - display_y1)
                    )
                    overlay_boxes.append((rect, cls_name, color))
                continue

            x1, y1, x2, y2, cls_name = box
            
            # convert from
            display_x1 = x1 * scale + offset_x
            display_y1 = y1 * scale + offset_y
            display_x2 = x2 * scale + offset_x
            display_y2 = y2 * scale + offset_y
            
            rect = QRect(
                int(display_x1),
                int(display_y1),
                int(display_x2 - display_x1),
                int(display_y2 - display_y1)
            )
            
            color = default_color_for_name(cls_name)
            overlay_boxes.append((rect, cls_name, color))
        
        # update overlay
        try:
            self._overlay.set_completed_shapes(overlay_boxes, overlay_regions)
            self._overlay.update()
        except RuntimeError:
            # overlay was
            self._overlay = None
    
    def finish_manual_labeling(self):
        if self.toolbar:
            self.toolbar.hide()
        self._box_history = []

        if not self._manual_boxes:
            print("[Manual] No boxes to save, advancing to next image")
            self.host.current_index += 1
            self.host.process_next()
            self.on_image_changed()
            return
        
        # save to
        img_path = str(self.state.current_image_path)
        entry = self.host.labels.get(img_path, {})
        dets = entry.get('detections', [])
        
        box_count = 0
        for box in self._manual_boxes:
            if isinstance(box, dict):
                cls_name = safe_class_name(box.get("class", "manual"))
                det = {
                    'bbox': [float(v) for v in box.get("bbox", [0, 0, 0, 0])],
                    'confidence': 100.0,
                    'class': cls_name,
                    'manual': True,
                    'shape': 'polygon',
                }
                if box.get("segmentation"):
                    det['segmentation'] = box["segmentation"]
                    det['mask_source'] = 'manual'
                dets.append(det)
                box_count += 1
                continue

            x1, y1, x2, y2, cls_name = box
            cls_name = safe_class_name(cls_name)

            dets.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': 100.0,
                'class': cls_name,
                'manual': True,
                'shape': 'box'
            })
            box_count += 1
        
        entry['detections'] = dets
        
        # store metadata
        try:
            w, h = self.state.current_image.size
            entry['image_width'] = w
            entry['image_height'] = h
        except Exception:
            pass
        
        entry['timestamp'] = datetime.now().isoformat()
        self.host.labels[img_path] = entry
        
        # persist through
        img_w = entry.get('image_width', 0)
        img_h = entry.get('image_height', 0)
        image_entropy = float(getattr(self.state, "last_image_entropy", 0.0))
        if hasattr(self.host, 'data_manager'):
            self.host.data_manager.save_labels(
                image_path=img_path,
                detections=dets,
                entropy=image_entropy,
                img_width=img_w,
                img_height=img_h
            )
        
        self.host.on_label_saved(img_path, dets)
        # update training
        for box in self._manual_boxes:
            cls_name = safe_class_name(box.get("class", "manual") if isinstance(box, dict) else box[4])
            if cls_name not in self.host.custom_classes:
                self.host.custom_classes.append(cls_name)
            self.host.class_samples[cls_name] = self.host.class_samples.get(cls_name, 0) + 1
        
        # save and
        try:
            self.host.save_autosave()
            self.host.update_stats()
        except Exception as e:
            print(f"[Manual] Save error: {e}")
        
        print(f"[Manual] Saved {box_count} boxes for {img_path}")
        
        # clear and
        self._manual_boxes = []
        self._start_point = None
        self._polygon_points = []
        self._active_image_path = None
        
        self.host.current_index += 1
        self.host.process_next()
        self.on_image_changed()
    
    def _cleanup(self):
        print("[Manual] Cleanup - exiting manual mode completely")
        if self.shortcuts:
            self.shortcuts.disable()
            self.shortcuts = None
        if self.toolbar:
            self.toolbar.hide()
            self.toolbar = None
        self._box_history = []
        self._active = False
        
        # hard reset
        self._manual_boxes = []
        self._start_point = None
        self._polygon_points = []
        self._active_image_path = None
        
        # unbind early
        self._unbind_mouse()
        
        # defer widget
        def safe_cleanup_widgets():
            if self._overlay:
                try:
                    self._overlay.hide()  # hide first
                    self._overlay.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
                    self._overlay.close()  # let qt
                except RuntimeError:
                    pass
                finally:
                    self._overlay = None
            
            if self._toolbox:
                try:
                    self._toolbox.close()
                    self._toolbox.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
                except RuntimeError:
                    pass
                finally:
                    self._toolbox = None
        
        # queue on
        QTimer.singleShot(0, safe_cleanup_widgets)

    def _delete_last_box(self):
        if not self._manual_boxes:
            return
        self._manual_boxes.pop()
        if self._toolbox:
            try:
                self._toolbox.update_box_count(len(self._manual_boxes))
            except RuntimeError:
                self._toolbox = None
        self._redraw_with_boxes()
        if self.toolbar:
            self.toolbar.hide()

    def _undo_last_point(self):
        if self._polygon_points:
            self._polygon_points.pop()
            self._update_polygon_preview()
            print("[Manual] Undid last polygon point")
            return True
        return False

    def _undo_last_box(self):
        if self._polygon_points:
            self._undo_last_point()
            return
        if not self._box_history:
            return
        last = self._box_history.pop()
        if self._manual_boxes and self._manual_boxes[-1] == last:
            self._manual_boxes.pop()
        elif last in self._manual_boxes:
            try:
                self._manual_boxes.remove(last)
            except ValueError:
                pass
        if self._toolbox:
            try:
                self._toolbox.update_box_count(len(self._manual_boxes))
            except RuntimeError:
                self._toolbox = None
        self._redraw_with_boxes()
        if self.toolbar:
            self.toolbar.hide()
    
    def get_persist_data(self):
        return {
            'custom_classes': list(self.host.custom_classes),
            'class_samples': dict(self.host.class_samples)
        }
    
    def load_persist_data(self, data):
        if not data:
            return
        
        cc = data.get('custom_classes', [])
        cs = data.get('class_samples', {})
        
        for c in cc:
            clean_c = safe_class_name(c)
            if clean_c not in self.host.custom_classes:
                self.host.custom_classes.append(clean_c)
        
        for k, v in cs.items():
            clean_k = safe_class_name(k)
            self.host.class_samples[clean_k] = v
