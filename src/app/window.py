# app/window.py - Main Window with Professional Stable UI
"""
Main application window with enterprise-grade stable layout
"""
from PIL import Image

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QFileDialog, QMessageBox, QFrame, QComboBox,
    QDialog, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QKeySequence, QShortcut, QFont
from PyQt6.QtWidgets import QSizePolicy

from pathlib import Path
from datetime import datetime

from numpy import size

from app import state, utils
from app.dialogs import ClassSelectorDialog, LabelFormatDialog


# Professional Color Palette
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


class TopControlBar(QFrame):
    """Fixed-height top control bar"""
    
    folder_clicked = pyqtSignal()
    class_clicked = pyqtSignal()
    start_clicked = pyqtSignal()
    threshold_changed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.setFixedHeight(60)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border-bottom: 2px solid {COLORS['border']};
            }}
        """)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(15)
        
        # Folder button
        self.folder_btn = QPushButton("📁 Select Folder")
        self.folder_btn.setFixedSize(140, 40)
        self.folder_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['olive']};
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive_dark']};
            }}
        """)
        self.folder_btn.clicked.connect(self.folder_clicked.emit)
        layout.addWidget(self.folder_btn)
        
        # Separator
        layout.addWidget(self._create_separator())
        
        # Class selector
        class_label = QLabel("Classes:")
        class_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px; font-weight: bold;")
        layout.addWidget(class_label)
        
        self.class_button = QPushButton("Select Classes")
        self.class_button.setFixedSize(120, 35)
        self.class_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
        """)
        self.class_button.clicked.connect(self.class_clicked.emit)
        layout.addWidget(self.class_button)
        
        self.class_label = QLabel("None selected")
        self.class_label.setStyleSheet(f"color: {COLORS['muted']}; font-size: 11px;")
        self.class_label.setMinimumWidth(150)
        layout.addWidget(self.class_label)
        
        # Separator
        layout.addWidget(self._create_separator())
        
        # Threshold
        thresh_label = QLabel("Confidence:")
        thresh_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px; font-weight: bold;")
        layout.addWidget(thresh_label)
        
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setMinimum(0)
        self.thresh_slider.setMaximum(100)
        self.thresh_slider.setValue(70)
        self.thresh_slider.setFixedWidth(150)
        self.thresh_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {COLORS['border']};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['olive']};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
        """)
        self.thresh_slider.valueChanged.connect(self.threshold_changed.emit)
        layout.addWidget(self.thresh_slider)
        
        self.thresh_value_label = QLabel("70%")
        self.thresh_value_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 11px; min-width: 35px;")
        self.thresh_slider.valueChanged.connect(lambda v: self.thresh_value_label.setText(f"{v}%"))
        layout.addWidget(self.thresh_value_label)
        
        layout.addStretch()
        
        # Progress
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 12px; font-weight: bold;")
        layout.addWidget(self.progress_label)
        
        # GPU indicator
        self.gpu_label = QLabel("🟢 GPU" if state.has_gpu else "⚪ CPU")
        self.gpu_label.setStyleSheet(f"""
            color: {COLORS['success'] if state.has_gpu else COLORS['muted']};
            font-size: 11px;
            padding: 5px 10px;
            background-color: {COLORS['bg']};
            border-radius: 3px;
        """)
        layout.addWidget(self.gpu_label)
        
        # Start button
        self.start_btn = QPushButton("▶ START")
        self.start_btn.setFixedSize(100, 40)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
        """)
        self.start_btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.start_btn)
    
    def _create_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"background-color: {COLORS['border']};")
        sep.setFixedWidth(1)
        return sep


class LeftSidePanel(QFrame):
    """Fixed-width left panel with stats and controls"""
    
    stats_clicked = pyqtSignal()
    force_retrain_clicked = pyqtSignal()
    create_version_clicked = pyqtSignal()
    list_versions_clicked = pyqtSignal()
    promote_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border-right: 2px solid {COLORS['border']};
            }}
        """)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)
        
        # Active Learning Section
        layout.addWidget(self._create_section_header("🎯 Active Learning"))
        
        al_panel = QFrame()
        al_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        al_layout = QVBoxLayout(al_panel)
        al_layout.setSpacing(8)
        
        # Strategy selector
        strategy_label = QLabel("Selection Strategy:")
        strategy_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 11px; font-weight: bold;")
        al_layout.addWidget(strategy_label)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Uncertainty", "Margin", "Diversity", "Balanced"])
        self.strategy_combo.setStyleSheet(f"""
                                          QComboBox {{
                                          background-color: white;
                                          color: {COLORS['text']};
                                          border: 1px solid {COLORS['border']};
                                          border-radius: 3px;
                                          padding: 6px;
                                          font-size: 11px;}}
                                          QComboBox::drop-down {{
                                            border: none;
                                            width: 20px;}}
                                            QComboBox::down-arrow {{
                                                image: none;
                                                border-left: 5px solid transparent;
                                                border-right: 5px solid transparent;
                                                border-top: 7px solid {COLORS['text']};
                                                margin-right: 6px;
                                                }}
                                                QComboBox QAbstractItemView {{
                                                    background-color: white;
                                                    color: {COLORS['text']};
                                                    selection-background-color: {COLORS['olive']};
                                                    selection-color: white;
                                                    border: 1px solid {COLORS['border']};
                                                    outline: none;
                                                    }}
                                                    QComboBox QAbstractItemView::item {{
                                                        padding: 6px;
                                                        color: {COLORS['text']};}}
                                                        QComboBox QAbstractItemView::item:selected {{
                                                            background-color: {COLORS['olive']};
                                                            color: white;}}""")

        al_layout.addWidget(self.strategy_combo)
        
        # Entropy display
        self.entropy_label = QLabel("Entropy: --")
        self.entropy_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 12px; font-weight: bold;")
        al_layout.addWidget(self.entropy_label)
        
        # Queue status
        self.queue_label = QLabel("Queue: 0/30")
        self.queue_label.setStyleSheet(f"color: {COLORS['muted']}; font-size: 11px;")
        al_layout.addWidget(self.queue_label)
        
        layout.addWidget(al_panel)
        
        # Training Section
        layout.addWidget(self._create_section_header("🔄 Training"))
        
        train_panel = QFrame()
        train_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        train_layout = QVBoxLayout(train_panel)
        train_layout.setSpacing(8)
        
        self.training_status_label = QLabel("Status: Not started")
        self.training_status_label.setStyleSheet(f"color: {COLORS['muted']}; font-size: 11px;")
        self.training_status_label.setWordWrap(True)
        train_layout.addWidget(self.training_status_label)
        
        # Force retrain button
        force_btn = QPushButton("Force Retrain")
        force_btn.setFixedHeight(32)
        force_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
        """)
        force_btn.clicked.connect(self.force_retrain_clicked.emit)
        train_layout.addWidget(force_btn)
        
        layout.addWidget(train_panel)
        
        # Dataset Version Section
        layout.addWidget(self._create_section_header("📦 Versions"))
        
        version_panel = QFrame()
        version_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        version_layout = QVBoxLayout(version_panel)
        version_layout.setSpacing(6)
        
        # Version buttons
        create_btn = QPushButton("Create Version")
        create_btn.setFixedHeight(28)
        create_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
        """)
        create_btn.clicked.connect(self.create_version_clicked.emit)
        version_layout.addWidget(create_btn)
        
        list_btn = QPushButton("List Versions")
        list_btn.setFixedHeight(28)
        list_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive']};
            }}
        """)
        list_btn.clicked.connect(self.list_versions_clicked.emit)
        version_layout.addWidget(list_btn)
        
        promote_btn = QPushButton("⭐ Promote Shadow")
        promote_btn.setFixedHeight(28)
        promote_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['olive']};
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['olive_dark']};
            }}
        """)
        promote_btn.clicked.connect(self.promote_clicked.emit)
        version_layout.addWidget(promote_btn)
        
        layout.addWidget(version_panel)
        
        layout.addStretch()
        
        # Stats button at bottom
        stats_btn = QPushButton("📊 View Statistics")
        stats_btn.setFixedHeight(35)
        stats_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['olive']};
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
        stats_btn.clicked.connect(self.stats_clicked.emit)
        layout.addWidget(stats_btn)
    
    def _create_section_header(self, text):
        label = QLabel(text)
        label.setStyleSheet(f"""
            color: {COLORS['olive_dark']};
            font-size: 12px;
            font-weight: bold;
            padding: 5px 0;
        """)
        return label


class BottomActionBar(QFrame):
    """Fixed-height bottom action bar"""
    
    accept_clicked = pyqtSignal()
    reject_clicked = pyqtSignal()
    skip_clicked = pyqtSignal()
    manual_clicked = pyqtSignal()
    log_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setFixedHeight(70)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border-top: 2px solid {COLORS['border']};
            }}
        """)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(10)
        
        buttons = [
            ("✓ Accept", "A", COLORS['success'], self.accept_clicked),
            ("✗ Reject", "R", COLORS['danger'], self.reject_clicked),
            ("→ Skip", "N", COLORS['muted'], self.skip_clicked),
            ("✏ Manual", "M", COLORS['warning'], self.manual_clicked),
            ("📋 Log", "", COLORS['accent'], self.log_clicked),
        ]
        
        for text, shortcut, color, signal in buttons:
            btn = QPushButton(text)
            if shortcut:
                btn.setText(f"{text} ({shortcut})")
            
            btn.setFixedHeight(45)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 0 20px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['olive_dark']};
                }}
            """)
            btn.clicked.connect(signal.emit)
            layout.addWidget(btn)


class MainWindow(QMainWindow):
    result_ready = pyqtSignal(list)
    monitor_signal = pyqtSignal(dict)  # Signal for orchestrator monitor events
    
    def __init__(self, app_context):
        super().__init__()
        self.app = app_context
        self.setWindowTitle("Smart Labeling v2.4 - Active Learning System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set window style
        self.setStyleSheet(f"background-color: {COLORS['bg']};")
        
        self.setup_ui()
        self.setup_shortcuts()
        self.setup_signals()
        self.setup_timers()
    
    def setup_ui(self):
        """Setup stable professional UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main vertical layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top control bar
        self.top_bar = TopControlBar()
        main_layout.addWidget(self.top_bar)
        
        # Middle section with splitter (left panel + canvas)
        middle_splitter = QSplitter(Qt.Orientation.Horizontal)
        middle_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['border']};
                width: 2px;
            }}
        """)
        
        # Left side panel
        self.left_panel = LeftSidePanel()
        middle_splitter.addWidget(self.left_panel)
        
        # Canvas area
        canvas_container = QFrame()
        canvas_container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg']};
            }}
        """)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(10, 10, 10, 10)
        
        self.canvas_label = QLabel()
        self.canvas_label.setStyleSheet(f"""
            QLabel {{
                background-color: #2c3e50;
                color: white;
                border: 2px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        self.canvas_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.canvas_label.setMinimumSize(400, 300)
        
        canvas_layout.addWidget(self.canvas_label)
        middle_splitter.addWidget(canvas_container)
        
        # Set splitter sizes (left panel: 280px, rest: expanding)
        middle_splitter.setSizes([280, 1000])
        middle_splitter.setCollapsible(0, False)
        middle_splitter.setCollapsible(1, False)
        
        main_layout.addWidget(middle_splitter, stretch=1)
        
        # Bottom action bar
        self.bottom_bar = BottomActionBar()
        main_layout.addWidget(self.bottom_bar)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        shortcuts = [
            ('a', self.accept),
            ('r', self.reject),
            ('n', self.skip),
            ('m', lambda: self.app.manual.start_manual_labeling())
        ]
        
        for key, callback in shortcuts:
            QShortcut(QKeySequence(key), self).activated.connect(callback)
    
    def setup_signals(self):
        """Setup signal connections."""
        self.result_ready.connect(self.handle_result)
        self.monitor_signal.connect(self.app._process_monitor_event)
        
        # Connect top bar signals
        self.top_bar.folder_clicked.connect(self.select_folder)
        self.top_bar.class_clicked.connect(self.open_class_selector)
        self.top_bar.start_clicked.connect(self.start_labeling)
        self.top_bar.threshold_changed.connect(lambda v: setattr(state, 'threshold', v))
        
        # Connect left panel signals
        self.left_panel.stats_clicked.connect(self.show_al_stats)
        self.left_panel.force_retrain_clicked.connect(self.force_retrain)
        self.left_panel.create_version_clicked.connect(self.create_dataset_version)
        self.left_panel.list_versions_clicked.connect(self.list_dataset_versions)
        self.left_panel.promote_clicked.connect(self.promote_shadow_model)
        
        # Connect bottom bar signals
        self.bottom_bar.accept_clicked.connect(self.accept)
        self.bottom_bar.reject_clicked.connect(self.reject)
        self.bottom_bar.skip_clicked.connect(self.skip)
        self.bottom_bar.manual_clicked.connect(lambda: self.app.manual.start_manual_labeling())
        self.bottom_bar.log_clicked.connect(self.show_log)
    
    def setup_timers(self):
        """Setup update timers."""
        # Training status timer
        self.training_timer = QTimer()
        self.training_timer.timeout.connect(self._check_training_status)
        self.training_timer.start(2000)
        
        # Queue status timer
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self._update_queue_status)
        self.queue_timer.start(5000)
    
    # ============ Event Handlers ============
    
    def select_folder(self):
        """Select image folder with Active Learning prioritization."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        
        folder_p = Path(folder)

        format_dialog = LabelFormatDialog(self, current_format=state.label_format)
        if format_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        label_format = format_dialog.get_format()
        self.app.configure_label_output(folder_p, label_format)
        
        # Use Active Learning to prioritize images
        state.image_files = self.app.select_folder_with_active_learning(folder_p)
        state.current_index = 0
        
        QMessageBox.information(
            self,
            "Success",
            f"Loaded {len(state.image_files)} images\n"
            f"(Prioritized by uncertainty)"
        )
        
        self.app._ensure_autosave_setup(folder_p)
        self.app.load_autosave()
        self.update_stats()
    
    def open_class_selector(self):
        """Open class selection dialog."""
        if not state.coco_classes:
            QMessageBox.warning(self, "Not Ready", "Wait for model to load")
            return
        
        all_classes = sorted(state.coco_classes + state.custom_classes)
        dialog = ClassSelectorDialog(self, all_classes, state.selected_classes)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            state.selected_classes = dialog.get_selected()
            state.custom_classes = [
                c for c in dialog.all_classes if c not in state.coco_classes
            ]
            
            if state.selected_classes:
                display = utils.format_class_display(state.selected_classes)
                self.top_bar.class_label.setText(display)
                self.top_bar.class_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 11px;")
            else:
                self.top_bar.class_label.setText("None selected")
                self.top_bar.class_label.setStyleSheet(f"color: {COLORS['muted']}; font-size: 11px;")
    
    def start_labeling(self):
        """Start labeling process."""
        if not state.worker_ready:
            QMessageBox.critical(
                self,
                "Error",
                "Worker not ready – wait for model to finish loading."
            )
            return
        
        if not state.image_files:
            QMessageBox.critical(self, "Error", "Select folder first")
            return
        
        if not state.selected_classes:
            QMessageBox.critical(self, "Error", "Please select at least one class.")
            return
        
        state.threshold = self.top_bar.thresh_slider.value()
        state.current_index = max(0, state.current_index)
        self.top_bar.progress_label.setText("Starting...")
        QTimer.singleShot(150, self.app.process_next)
    
    def handle_result(self, detections):
        """Handle detection results."""
        state.current_detections = detections or []
        
        # Store image size for COCO export
        if state.current_image:
            state._current_img_size = state.current_image.size
        else:
            state._current_img_size = (0, 0)
        
        # Update entropy display
        if detections:
            entropy = state.last_image_entropy
            self.left_panel.entropy_label.setText(f"Entropy: {entropy:.3f}")
            color = COLORS['danger'] if entropy > 0.7 else COLORS['warning'] if entropy > 0.4 else COLORS['success']
            self.left_panel.entropy_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")
        
        self.display_image(detections)
        
        if detections:
            max_conf = max((d['confidence'] for d in detections), default=0.0)
            if max_conf >= state.threshold:
                if self.app.should_auto_accept():
                    self.save_label(detections, auto=True)
                    state.auto_accepted_log.append(str(state.image_files[state.current_index]))
                    state.current_index += 1
                    
                    try:
                        self.app.save_autosave()
                        self.update_stats()
                    except Exception:
                        pass
                    
                    QTimer.singleShot(50, self.app.process_next)
                    return
        
        self.top_bar.progress_label.setText("Review required")
    
    def display_image(self, detections):
        if not state.current_image:
            self.canvas_label.setText("No image loaded")
            return
        
        # Validate image
        try:
            _ = state.current_image.size
        except Exception:
            self.canvas_label.setText("Image closed or corrupted")
            return
        
        canvas_w = self.canvas_label.width()
        canvas_h = self.canvas_label.height()
        
        if canvas_w < 100 or canvas_h < 100:
            QTimer.singleShot(50, lambda: self.display_image(detections))
            return
        
        try:
            from PIL import ImageDraw
            
            img_draw = state.current_image.copy()
            draw = ImageDraw.Draw(img_draw)
            
            for d in (detections or []):
                b = d.get('bbox', [0, 0, 0, 0])
                cls_name = d.get('class', 'unknown')
                conf = d.get('confidence', 0)
                entropy = d.get('entropy', 0)
                
                color = utils.default_color_for_name(cls_name)
                iw, ih = img_draw.size
                b = [max(0, min(iw, x)) for x in b]
                
                draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline=color, width=3)
                label = f"{cls_name}: {conf:.1f}% (e:{entropy:.2f})"
                draw.text((b[0], max(0, b[1] - 14)), label, fill=color)
            
            # Scale and display
            iw, ih = img_draw.size
            scale = min(canvas_w / iw, canvas_h / ih, 1.0)
            if scale <= 0:
                scale = 1.0
            state.scale_factor = scale
            
            new_w = max(1, int(iw * scale))
            new_h = max(1, int(ih * scale))
            img_resized = img_draw.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_rgb = img_resized.convert('RGB')
            raw = img_rgb.tobytes()
            bytes_per_line = new_w * 3
            qimage = QImage(raw, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
            
            if qimage.isNull():
                self.canvas_label.setText("Failed to create image")
                return
            
            pixmap = QPixmap.fromImage(qimage.copy())
            self.canvas_label.setPixmap(pixmap)
            
            img_resized.close()
            img_draw.close()
        
        except Exception as e:
            print(f"Display error: {e}")
            self.canvas_label.setText(f"Error: {str(e)}")
    
    def save_label(self, detections, auto=False):
        """Save label with metadata."""
        img_path = str(state.image_files[state.current_index])
        img_w, img_h = getattr(state, '_current_img_size', (0, 0))
        entropy = getattr(state, 'last_image_entropy', 0.0)
        
        # Save using data_manager
        self.app.data_manager.save_labels(
            image_path=img_path,
            detections=detections,
            entropy=entropy,
            img_width=img_w,
            img_height=img_h
        )
        
        # Check if retraining should be triggered
        self.app.on_label_saved(img_path, detections)
        
        self.update_stats()
    
    # Action handlers
    def accept(self):
        """Accept current detections."""
        if state.current_detections:
            self.save_label(state.current_detections, auto=False)
        state.current_index += 1
        self.app.process_next()
    
    def reject(self):
        """Reject current detections."""
        try:
            self.app.save_autosave()
        except Exception:
            pass
        state.current_index += 1
        self.app.process_next()
    
    def skip(self):
        """Skip current image."""
        try:
            self.app.save_autosave()
        except Exception:
            pass
        state.current_index += 1
        self.app.process_next()
    
    def show_log(self):
        """Show auto-accepted images log."""
        if not state.auto_accepted_log:
            QMessageBox.information(self, "Log", "No auto-accepted images yet")
            return
        
        msg = "\n".join([Path(p).name for p in state.auto_accepted_log[-20:]])
        QMessageBox.information(self, "Auto-accepted (last 20)", msg)
    
    # Active Learning UI handlers
    
    def show_al_stats(self):
        """Show Active Learning statistics."""
        stats_dialog = QDialog(self)
        stats_dialog.setWindowTitle("Active Learning Statistics")
        stats_dialog.setMinimumSize(600, 500)
        stats_dialog.setStyleSheet(f"background-color: {COLORS['bg']};")
        
        layout = QVBoxLayout()
        
        # Get stats from all components
        data_stats = self.app.data_manager.get_stats()
        selector_stats = self.app.sample_selector.get_stats()
        policy_status = self.app.retrain_policy.get_status()
        replay_stats = self.app.replay_buffer.get_stats()
        
        # Build stats text
        stats_text = f"""
=== Dataset Statistics ===
Total Labeled: {data_stats['total_labeled']}
Training Queue: {data_stats['training_queue_size']}
Trained Count: {data_stats['trained_count']}
Average Entropy: {data_stats['avg_entropy']:.4f}
Classes: {data_stats['num_classes']}

Class Distribution:
{self._format_dict(data_stats.get('class_counts', {}))}

=== Sample Selection ===
Total Selections: {selector_stats.get('total_selections', 0)}
Selection Rounds: {selector_stats.get('selection_rounds', 0)}
By Strategy: {self._format_dict(selector_stats.get('by_strategy', {}))}

=== Retraining Policy ===
Should Retrain: {policy_status.get('should_retrain', False)}
Last Trained: {policy_status.get('last_trained', 'Never')}

=== Replay Buffer ===
Size: {replay_stats['size']}/{replay_stats['capacity']}
Utilization: {replay_stats['utilization']:.1%}
Average Entropy: {replay_stats.get('avg_entropy', 0):.4f}
"""
        
        text_edit = QTextEdit()
        text_edit.setPlainText(stats_text)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: white;
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 10px;
                font-family: monospace;
            }}
        """)
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(35)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['olive']};
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
        close_btn.clicked.connect(stats_dialog.accept)
        layout.addWidget(close_btn)
        
        stats_dialog.setLayout(layout)
        stats_dialog.exec()
    
    def _format_dict(self, d):
        """Format dictionary for display."""
        if not d:
            return "  (empty)"
        return "\n".join(f"  {k}: {v}" for k, v in sorted(d.items()))
    
    def force_retrain(self):
        """Force retraining regardless of policy."""
        can_train, info = self.app.retrain_policy.force_retrain()
        
        if not can_train:
            QMessageBox.warning(
                self,
                "Cannot Train",
                info.get('error', 'Unknown error')
            )
            return
        
        reply = QMessageBox.question(
            self,
            "Force Retrain",
            "Force shadow model retraining?\n\n"
            "This bypasses policy checks.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.app.trigger_training_with_validation()
    
    def create_dataset_version(self):
        """Create new dataset version."""
        try:
            version = self.app.dataset_versioner.create_version(
                data_manager=self.app.data_manager,
                description=f"Manual snapshot - {len(state.labels)} images",
                parent_version=getattr(state, 'last_dataset_version', None)
            )
            
            state.last_dataset_version = version['version']
            
            QMessageBox.information(
                self,
                "Version Created",
                f"Dataset version created:\n\n"
                f"Version: {version['version']}\n"
                f"Images: {version['statistics']['total_images']}\n"
                f"Instances: {version['statistics']['total_instances']}\n"
                f"Hash: {version['hash']}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create version:\n{e}")
    
    def list_dataset_versions(self):
        """List all dataset versions."""
        versions = self.app.dataset_versioner.list_versions()
        
        if not versions:
            QMessageBox.information(self, "No Versions", "No dataset versions exist yet")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Dataset Versions")
        dialog.setMinimumSize(650, 450)
        dialog.setStyleSheet(f"background-color: {COLORS['bg']};")
        
        layout = QVBoxLayout()
        
        text = "=== Dataset Versions ===\n\n"
        for v in versions:
            text += f"Version: {v['version']}\n"
            text += f"  Created: {v['created']}\n"
            text += f"  Images: {v['images']}\n"
            text += f"  Instances: {v['instances']}\n"
            text += f"  Classes: {v['classes']}\n"
            text += f"  Hash: {v['hash']}\n"
            text += f"  Description: {v['description']}\n\n"
        
        text_edit = QTextEdit()
        text_edit.setPlainText(text)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: white;
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 10px;
                font-family: monospace;
            }}
        """)
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(35)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['olive']};
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
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def promote_shadow_model(self):
        """Promote shadow model to active."""
        if not hasattr(self.app, 'orchestrator'):
            QMessageBox.warning(self, "Error", "Training system not initialized")
            return
        
        reply = QMessageBox.question(
            self,
            "Promote Shadow Model",
            "Promote shadow model to active?\n\n"
            "This will make it the default for inference.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        result = self.app.orchestrator.promote_shadow_model(validate=True)
        
        if result['success']:
            QMessageBox.information(
                self,
                "Success",
                f"Shadow model promoted!\n\n"
                f"Version: {result['version']}\n"
                f"Path: {result['path']}\n\n"
                f"Restart the app to use the new model."
            )
        elif result.get('requires_confirmation'):
            reply = QMessageBox.question(
                self,
                "Validation Warning",
                f"Model validation detected issues:\n\n{result['error']}\n\n"
                f"Promote anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                result = self.app.orchestrator.promote_shadow_model(validate=False)
                if result['success']:
                    QMessageBox.information(self, "Success", "Model promoted!")
        else:
            QMessageBox.critical(self, "Error", f"Promotion failed:\n\n{result['error']}")
    
    # ============ Update Timers ============
    
    def _check_training_status(self):
        """Update training status display."""
        if not hasattr(self.app, 'orchestrator'):
            return
        
        self.app.orchestrator.check_training_completion()
        
        status = self.app.orchestrator.get_training_status()
        
        if status.get('training'):
            epoch = status.get('epoch', 0)
            total = status.get('total_epochs', 1)
            loss = status.get('loss', 0)
            percent = status.get('percent', 0)
            
            self.left_panel.training_status_label.setText(
                f"Training: {epoch}/{total}\n({percent:.0f}%) Loss: {loss:.4f}"
            )
            self.left_panel.training_status_label.setStyleSheet(
                f"color: {COLORS['warning']}; font-size: 11px; font-weight: bold;"
            )
        else:
            size = len(self.app.replay_buffer)
            min_samples = self.app.retrain_policy.min_samples
            
            self.left_panel.training_status_label.setText(
                f"Queue: {size}/{min_samples} samples\nReady to train"
            )
            
            color = COLORS['success'] if size >= min_samples else COLORS['muted']
            self.left_panel.training_status_label.setStyleSheet(
                f"color: {color}; font-size: 11px;"
            )
    
    def _update_queue_status(self):
        if not hasattr(self.app, 'replay_buffer'):
            return
        
        size = len(self.app.replay_buffer)
        capacity = self.app.replay_buffer.max_size
        self.left_panel.queue_label.setText(f"Queue: {size}/{capacity}")
        
        if size >= self.app.retrain_policy.min_samples:
            self.left_panel.queue_label.setStyleSheet(
                f"color: {COLORS['success']}; font-size: 11px; font-weight: bold;"
            )
        else:
            self.left_panel.queue_label.setStyleSheet(
                f"color: {COLORS['muted']}; font-size: 11px;"
            )
    
    def update_stats(self):
        """Update statistics display."""
        total = len(state.image_files) if state.image_files else 0
        labeled = len(state.labels) if state.labels else 0
        self.top_bar.progress_label.setText(f"Labeled: {labeled}/{total}")
        self.top_bar.progress_label.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 12px; font-weight: bold;"
        )
