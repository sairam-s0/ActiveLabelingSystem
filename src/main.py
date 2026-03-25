# main.py -- Smart Labeling v2.3
"""
Smart Labeling v2.3 - Multi-class detection with modular architecture
Entry point and application coordinator
No of times edited: 49
"""

from multiprocessing import Process
from core.dataset_versioner import DatasetVersioner
from core.feedback_validator import FeedbackValidator
from core.retrain_policy import RetrainingPolicy
from core.sample_selector import SampleSelector
from core.entropy import EntropyCalculator


import sys
import os
import json
import atexit
import random
import threading
import tempfile
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Path setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

# Import modular components
from app import state, utils
from app.window import MainWindow

# Core modules
from core.training_orchestrator import TrainingOrchestrator, start_orchestrator_monitor
from core.replay_buffer import ReplayBuffer
from core.data_manager import DataManager
from core.model_manager import ModelManager

# Features
try:
    from features.manual import ManualManager
except Exception:
    class ManualManager:
        def __init__(self, host, window, state):
            self.host = host
            self.window = window
            self.state = state
        def get_persist_data(self):
            return {}
        def load_persist_data(self, data):
            return
        def start_manual_labeling(self):
            QMessageBox.information(None, "Manual", "Manual labeling module not found")
        def on_image_changed(self):
            pass

_WORKER_MODEL = None

def init_worker(weights, device):
    global _WORKER_MODEL
    try:
        from ultralytics import YOLO
        print(f"[Worker] Loading {weights} on {device}...")
        _WORKER_MODEL = YOLO(weights)
        if device == 'cuda':
            try:
                _WORKER_MODEL.to('cuda')
            except Exception:
                pass
        print("[Worker] Ready")
    except Exception as e:
        print(f"[Worker] Init failed: {e}")


def detect_worker(args):
    """Worker detection function with multi-class support."""
    from core.entropy import EntropyCalculator

    global _WORKER_MODEL
    if not _WORKER_MODEL:
        return {'error': 'Model not loaded in worker', 'detections': []}

    img_path, class_names, thresh, max_dim = args
    temp_fd, temp_path = None, None
    try:
        from PIL import Image
        import numpy as _np

        img = Image.open(img_path)
        orig_size = img.size
        scale = 1.0
        detect_path = img_path

        if max(orig_size) > max_dim:
            scale = max_dim / max(orig_size)
            new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
            img_r = img.resize(new_size, Image.Resampling.LANCZOS)
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            img_r.save(temp_path)
            img_r.close()
            detect_path = temp_path

        img.close()

        results = _WORKER_MODEL(detect_path, verbose=False)

        matching = []
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes:
            boxes = results[0].boxes
            model_names = getattr(_WORKER_MODEL, 'names', {})
            for i in range(len(boxes)):
                try:
                    box = boxes[i]
                    try:
                        cls_t = box.cls
                        if hasattr(cls_t, 'cpu'):
                            cls_id = int(cls_t.cpu().item())
                        else:
                            cls_id = int(cls_t)
                    except Exception:
                        cls_id = int(box.cls)

                    try:
                        conf_t = box.conf
                        if hasattr(conf_t, 'cpu'):
                            conf = float(conf_t.cpu().item()) * 100.0
                        else:
                            conf = float(conf_t) * 100.0
                    except Exception:
                        conf = float(box.conf) * 100.0

                    try:
                        xy = box.xyxy
                        if hasattr(xy, 'cpu'):
                            arr = xy.cpu().numpy()
                        else:
                            arr = _np.array(xy)
                        arr_flat = _np.asarray(arr).reshape(-1)[:4].astype(float)
                        bbox = arr_flat.tolist()
                    except Exception:
                        bbox = [0.0, 0.0, 0.0, 0.0]

                    name = model_names.get(cls_id, None) if isinstance(model_names, dict) else None
                    if name in class_names and conf >= thresh:
                        if scale != 1.0 and scale > 0:
                            bbox = [c / scale for c in bbox]
                        # ---- Active Learning: entropy calculation ----
                        try:
                            num_classes = len(model_names)
                            entropy = EntropyCalculator.from_yolo_output(box, num_classes)
                        except Exception:
                            # fallback if entropy calc fails
                            entropy = 1.0 - (conf / 100.0)
                        matching.append({
                                'bbox': bbox,
                                'confidence': conf,
                                'class': name,
                                'entropy': entropy
                            })
                except Exception:
                    continue

        return {'error': None, 'detections': matching}

    except Exception as e:
        return {'error': f'Worker exception: {e}', 'detections': []}
    finally:
        try:
            if temp_fd:
                os.close(temp_fd)
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass


# ============ APPLICATION CONTEXT ============

class SmartLabelingApp:    
    # Property accessors for backward compatibility
    @property
    def current_image(self):
        return state.current_image
    
    @current_image.setter
    def current_image(self, value):
        state.current_image = value
    
    @property
    def current_image_path(self):
        return state.current_image_path
    
    @current_image_path.setter
    def current_image_path(self, value):
        state.current_image_path = value
    
    @property
    def current_detections(self):
        return state.current_detections
    
    @current_detections.setter
    def current_detections(self, value):
        state.current_detections = value
    
    @property
    def labels(self):
        return state.labels
    
    @labels.setter
    def labels(self, value):
        state.labels = value
    
    @property
    def image_files(self):
        return state.image_files
    
    @property
    def current_index(self):
        return state.current_index
    
    @current_index.setter
    def current_index(self, value):
        state.current_index = value
    
    @property
    def selected_classes(self):
        return state.selected_classes
    
    @property
    def scale_factor(self):
        return state.scale_factor
    
    @scale_factor.setter
    def scale_factor(self, value):
        state.scale_factor = value
    
    @property
    def class_samples(self):
        return state.class_samples
    
    @class_samples.setter
    def class_samples(self, value):
        state.class_samples = value
    
    @property
    def custom_classes(self):
        return state.custom_classes
    
    @custom_classes.setter
    def custom_classes(self, value):
        state.custom_classes = value
    
    @property
    def min_training_samples(self):
        return state.min_training_samples
    
    @min_training_samples.setter
    def min_training_samples(self, value):
        state.min_training_samples = value
    
    def __init__(self):
        # Initialize GPU detection
        state.has_gpu = utils.detect_gpu()
        state.device = 'cuda' if state.has_gpu else 'cpu'
        
        # Load model class names
        self._load_model_classes()
        
        # Initialize managers
        self.data_manager = DataManager("labels.json")
        self.replay_buffer = ReplayBuffer(max_size=200)
        self.model_manager = ModelManager("models")
        
        # Initialize training orchestrator
        self.orchestrator = TrainingOrchestrator(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
            replay_buffer=self.replay_buffer,
            min_samples=30,
            num_gpus=0
        )

        
        # Setup callbacks
        self.orchestrator.on_status_change = self._on_training_status_change
        self.orchestrator.on_training_complete = self._on_training_complete
        self.orchestrator.on_training_failed = self._on_training_failed
        
        # Create window FIRST (no manual manager yet)
        self.window = MainWindow(self)
        
        # NOW create manual manager with all dependencies available
        self.manual = ManualManager(
            host=self,
            window=self.window,
            state=state
        )
        
        # Hook for graceful shutdown - override window close event
        self.window.closeEvent = self._on_window_close
        
        self.entropy_calculator = EntropyCalculator()
        self.sample_selector = SampleSelector(
            self.data_manager,
            self.entropy_calculator)
        self.retrain_policy = RetrainingPolicy(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
            min_samples=30,
            max_wait_hours=24,
            perf_delta_threshold=0.05,
            entropy_shift_threshold=0.15
    )
        self.dataset_versioner = DatasetVersioner("datasets")
        self._monitor_started = False
        
        
        # Initialize worker
        self.init_worker()
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def _on_window_close(self, event):
        """Graceful shutdown before app closes."""
        # Clean up manual mode first if active
        if hasattr(self, 'manual') and self.manual._active:
            print("[Main] Cleaning up manual mode before shutdown...")
            self.manual.exit_manual_mode()
        
        # Accept the close event and proceed
        event.accept()
    
    def _load_model_classes(self):
        """Load class names from model metadata."""
        try:
            from ultralytics import YOLO
            print("Loading model class names from metadata...")
            temp_model = YOLO(state.weights)
            state.coco_classes = list(temp_model.names.values())
            print(f"✅ Loaded {len(state.coco_classes)} classes.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load class names from {state.weights}: {e}")
            # Fallback to common COCO classes
            state.coco_classes = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow"
            ]
    def select_folder_with_active_learning(self, folder_path):
        from pathlib import Path

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        folder_path = Path(folder_path)

        all_images = sorted(
            str(p) for p in folder_path.iterdir()
        if p.suffix.lower() in exts
    )

        if not all_images:
            return []
        labeled = set(state.labels.keys())

        unlabeled = [p for p in all_images if p not in labeled]

        if not unlabeled:
            return all_images

        try:
            prioritized = self.sample_selector.select_batch(
                unlabeled_pool=unlabeled,
                batch_size=len(unlabeled),
                strategy="uncertainty"
            )
        except Exception as e:
            print(f"[AL] selector fallback: {e}")
            prioritized = unlabeled

        labeled_images = [p for p in all_images if p in labeled]
        return prioritized + labeled_images

    def on_label_saved(self, image_path, detections):
        sample = {
            "image_path": image_path,
            "detections": detections,
            "entropy": float(getattr(state, "last_image_entropy", 0.0))
        }
        self.replay_buffer.add(sample)
        self.persist_labels()

        

        should_train, info = self.retrain_policy.should_retrain()
        if should_train:
            print(f"[AL] Retraining suggested: {info.get('recommendation')}")
            self.trigger_training_with_validation()

    def configure_label_output(self, folder_path: Path, label_format: str):
        state.label_format = label_format
        state.labels_output_dir = str(folder_path)
        if label_format == "coco":
            output_file = folder_path / "labels_coco.json"
        else:
            output_file = folder_path / "labels.json"
        state.labels_output_path = str(output_file)

        internal_file = folder_path / ".labels_internal.json"
        if hasattr(self, "data_manager"):
            self.data_manager.set_path(str(internal_file))

    def persist_labels(self):
        if not state.labels_output_path:
            return
        try:
            out_path = Path(state.labels_output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if state.label_format == "coco":
                root_dir = Path(state.labels_output_dir) if state.labels_output_dir else None
                payload = self.data_manager.export_coco(image_root=root_dir)
            else:
                payload = self.data_manager.export_simple_json()

            fd, tmp = tempfile.mkstemp(
                prefix="labels_",
                suffix=".tmp",
                dir=str(out_path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

                if os.name == "nt" and out_path.exists():
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass

                os.replace(tmp, out_path)
            except Exception:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
                raise
        except Exception as e:
            print(f"[persist_labels] failed: {e}")



    def trigger_training_with_validation(self):
        # Lazy-init Ray/monitor only when training is requested.
        self._start_background_monitor()
        state.validation_before_model = self.model_manager.resolve_active_path()
        self.orchestrator.trigger_training()
    def init_worker(self):
        """Initialize worker pool in background thread"""
        def init():
            try:
                print("[Worker] Initializing in background...")
                state.executor = ProcessPoolExecutor(
                    max_workers=1,
                    initializer=init_worker,
                    initargs=(state.weights, state.device)
                )
                print("[Worker] Ready for inference")
                state.worker_ready = True
            except Exception as e:
                print(f"[Worker] Failed: {e}")
                state.worker_ready = False
        
        t = threading.Thread(target=init, daemon=True)
        t.start()
    
    def _start_background_monitor(self):
        """Start Ray and orchestrator monitoring after UI is ready."""
        if self._monitor_started:
            return

        def init_ray_and_monitor():
            # Initialize Ray
            print("[Main] Starting Ray initialization...")
            if not self.orchestrator.initialize_ray():
                print("[Main] Ray initialization failed - background training disabled")
                return
            print("[Main] Ray ready, starting monitor")
            
            # Start background monitor
            start_orchestrator_monitor(
                self.orchestrator,
                lambda event: self.window.monitor_signal.emit(event)
            )
            self._monitor_started = True
        
        # Run in background thread to avoid blocking UI
        t = threading.Thread(target=init_ray_and_monitor, daemon=True)
        t.start()
    
    def select_folder(self):
        """Select folder with active learning prioritization."""
        folder = QFileDialog.getExistingDirectory(
            self.window,
            "Select Image Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not folder:
            return
        
        folder_path = Path(folder)
        
        # Ensure autosave setup
        self._ensure_autosave_setup(folder_path)
        
        # Load previous session if exists
        self.load_autosave()
        
        # Get prioritized image list
        state.image_files = self.select_folder_with_active_learning(folder_path)
        
        if not state.image_files:
            QMessageBox.warning(
                self.window,
                "No Images",
                "No supported images found in selected folder.\n"
                "Supported formats: .jpg, .jpeg, .png, .bmp, .webp"
            )
            return
        
        # Update folder label
        folder_name = folder_path.name
        if len(folder_name) > 25:
            folder_name = folder_name[:22] + "..."
        self.window.top_bar.folder_btn.setText(f"📁 {folder_name}")
        
        # Reset navigation
        if state.current_index >= len(state.image_files):
            state.current_index = 0
        
        # Update UI
        self.window.update_stats()
        
        print(f"[Folder] Loaded {len(state.image_files)} images (AL prioritized)")
    
    def select_classes(self):
        """Open class selection dialog."""
        from app.dialogs import ClassSelectorDialog
        
        dialog = ClassSelectorDialog(
            self.window,
            state.coco_classes,
            state.selected_classes,
            state.custom_classes
        )
        
        if dialog.exec():
            selected = dialog.get_selected()
            if selected:
                state.selected_classes = selected
                display = utils.format_class_display(selected)
                self.window.top_bar.class_label.setText(display)
                self.window.top_bar.class_label.setStyleSheet("color: white; font-size: 9px;")
                print(f"[Classes] Selected: {selected}")
    
    def start_labeling(self):
        """Start the labeling workflow."""
        if not state.image_files:
            QMessageBox.warning(
                self.window,
                "No Folder",
                "Please select a folder first."
            )
            return
        
        if not state.selected_classes:
            QMessageBox.warning(
                self.window,
                "No Classes",
                "Please select at least one class."
            )
            return
        
        # Start from current index
        self.process_next()
    
    def process_next(self):
        """Process next image in workflow."""
        if state.current_index >= len(state.image_files):
            QMessageBox.information(
                self.window,
                "Complete",
                f"All {len(state.image_files)} images processed!"
            )
            return
        
        img_path = state.image_files[state.current_index]
        
        # Check if already labeled
        if img_path in state.labels:
            print(f"[Skip] Already labeled: {img_path}")
            state.current_index += 1
            QTimer.singleShot(0, self.process_next)
            return
        
        # Load and display image
        try:
            state.current_image = utils.load_image(img_path)
            state.current_image_path = img_path
            
            pixmap = utils.pil_to_pixmap(state.current_image)
            scaled_pixmap, state.scale_factor = utils.scale_to_fit(
                pixmap,
                self.window.canvas_label.size()
            )
            self.window.canvas_label.setPixmap(scaled_pixmap)
            
            # Update filename display
            filename = Path(img_path).name
            if len(filename) > 40:
                filename = filename[:37] + "..."
            if hasattr(self.window.bottom_bar, "filename_label"):
                self.window.bottom_bar.filename_label.setText(filename)
            
            # Update stats
            self.window.update_stats()
            
            # Notify manual mode of image change
            if hasattr(self, 'manual'):
                self.manual.on_image_changed()
            
            # Run detection
            self.run_detect(img_path)
            
        except Exception as e:
            print(f"[Error] Loading {img_path}: {e}")
            state.current_index += 1
            QTimer.singleShot(0, self.process_next)
    
    def run_detect(self, img_path):
        """Run detection on image in worker process."""
        try:
            args = (
                img_path,
                state.selected_classes,
                state.threshold,
                state.max_dim
            )
            future = state.executor.submit(detect_worker, args)
            result = future.result(timeout=30)
            
            if result.get('error'):
                state.detection_errors.append((img_path, result.get('error')))
            
            detections = result.get('detections', [])
            self.window.result_ready.emit(detections)
            if detections:
                state.last_image_entropy = max(
                    (d.get("entropy", 0.0) for d in detections),
                    default=0.0
                )
            else:
                state.last_image_entropy = 0.0
        
        except Exception as e:
            print(f"[run_detect] {e}")
            self.window.result_ready.emit([])
    
    def should_auto_accept(self):

        return random.random() >= state.qa_rate
    
    def _ensure_autosave_setup(self, folder_path):
        """Setup autosave file path."""
        try:
            state.autosave_file = str(folder_path / "labels_autosave.json")
        except Exception:
            state.autosave_file = None
    
    def save_autosave(self):
        """Save current progress to autosave file."""
        if not state.autosave_file:
            return
        
        try:
            import json
            from datetime import datetime
            
            persist = {
                'labels': state.labels,
                'current_index': state.current_index,
                'auto_accepted_log': state.auto_accepted_log,
                'detection_errors': state.detection_errors,
                'selected_classes': state.selected_classes,
                'saved_at': datetime.now().isoformat()
            }
            
            if hasattr(self, 'manual') and hasattr(self.manual, 'get_persist_data'):
                try:
                    persist.update(self.manual.get_persist_data())
                except Exception:
                    pass
            
            autosave_path = Path(state.autosave_file)
            autosave_path.parent.mkdir(parents=True, exist_ok=True)
            
            with state.autosave_lock:
                fd, tmp = tempfile.mkstemp(
                    prefix="autosave_",
                    suffix=".tmp",
                    dir=str(autosave_path.parent)
                )
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        json.dump(persist, f, indent=2, ensure_ascii=False)
                    
                    if os.name == 'nt' and autosave_path.exists():
                        try:
                            os.remove(autosave_path)
                        except Exception:
                            pass
                    
                    os.replace(tmp, autosave_path)
                except Exception:
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass
                    raise
        except Exception as e:
            print(f"[save_autosave] failed: {e}")
    
    def load_autosave(self):
        """Load progress from autosave file."""
        if not state.autosave_file:
            return
        
        path = Path(state.autosave_file)
        if not path.exists():
            self._restore_from_internal_labels()
            return
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state.labels = data.get('labels', {}) or {}
            state.current_index = int(data.get('current_index', 0) or 0)
            state.auto_accepted_log = data.get('auto_accepted_log', [])
            state.detection_errors = data.get('detection_errors', [])
            
            saved_classes = data.get('selected_classes', [])
            if saved_classes:
                state.selected_classes = saved_classes
                display = utils.format_class_display(saved_classes)
                self.window.top_bar.class_label.setText(display)
                self.window.top_bar.class_label.setStyleSheet("color: white; font-size: 9px;")
            
            if hasattr(self, 'manual') and hasattr(self.manual, 'load_persist_data'):
                try:
                    self.manual.load_persist_data(data)
                except Exception:
                    pass
            
            if state.labels:
                msg = (
                    f"Autosave loaded — {len(state.labels)} labeled images. "
                    f"Resuming at index {state.current_index + 1}."
                )
                print("[load_autosave] " + msg)
                self.window.top_bar.progress_label.setText("Autosave loaded")
                QMessageBox.information(self.window, "Autosave", msg)
                self.window.update_stats()
            
            if not state.labels and hasattr(self, "data_manager") and self.data_manager.data.get("images"):
                self._restore_from_internal_labels()
        
        except Exception as e:
            print(f"[load_autosave] failed: {e}")

    def _restore_from_internal_labels(self):
        """Restore progress from internal labels store when autosave is missing."""
        if not hasattr(self, "data_manager"):
            return
        if not self.data_manager.data.get("images"):
            return

        try:
            state.labels = self.data_manager.export_simple_json()
            labeled = set(state.labels.keys())

            if state.image_files:
                idx = 0
                while idx < len(state.image_files) and str(state.image_files[idx]) in labeled:
                    idx += 1
                state.current_index = idx

            msg = (
                f"Recovered from internal labels - {len(state.labels)} labeled images. "
                f"Resuming at index {state.current_index + 1}."
            )
            print("[restore] " + msg)
            if hasattr(self, "window"):
                self.window.top_bar.progress_label.setText("Recovered labels")
                QMessageBox.information(self.window, "Recovered", msg)
                self.window.update_stats()
        except Exception as e:
            print(f"[restore] failed: {e}")
    def _process_monitor_event(self, event):

        try:
            if event['type'] == 'status':
                self._cached_training_status = event['data']
            elif event['type'] == 'completion':
                self._on_training_complete(event['data'])
        except Exception as e:
            print(f"[UI] Monitor event error: {e}")
    
    def update_stats(self):
        """Update statistics display."""
        if hasattr(self, 'window'):
            self.window.update_stats()
    
    # Training callbacks
    def _on_training_status_change(self, status):
        """Handle training status changes."""
        if status['status'] == 'training_started':
            msg = f"Training started: {status['sample_count']} samples"
            self.window.top_bar.progress_label.setText(msg)
            print(f"[Main] {msg}")
    
    def _on_training_complete(self, result):
        """Handle training completion."""
        msg = f"Training complete! Model saved to: {result['save_path']}"
        self.window.top_bar.progress_label.setText("Training complete")
        
        QMessageBox.information(
            self.window,
            "Training Complete",
            f"Shadow model training completed!\n\n"
            f"Samples: {result['sample_count']}\n"
            f"Model: {result['save_path']}\n\n"
            f"You can now promote the shadow model."
        )
    
    def _on_training_failed(self, result):
        """Handle training failure."""
        error = result.get('error', 'Unknown error')
        QMessageBox.critical(
            self.window,
            "Training Failed",
            f"Shadow model training failed:\n\n{error}\n\n"
            f"Check console for details."
        )
    
    def cleanup(self):
        """Cleanup resources on exit."""
        print("[Cleanup] shutting down executor...")
        try:
            if state.executor:
                state.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        
        utils.safe_close_image(state.current_image)
        
        if hasattr(self, 'orchestrator'):
            self.orchestrator.shutdown()
        
        print("[Cleanup] done")
    
    def show(self):
        """Show the main window."""
        self.window.show()



def main():
    qt_app = QApplication(sys.argv)
    
    # Suppress Qt stylesheet warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="PyQt6")
    
    # Fix QMessageBox text visibility
    qt_app.setStyleSheet("""
        QMessageBox {
            background-color: white;
        }
        QMessageBox QLabel {
            color: #1f2415;
            font-size: 13px;
        }
        QMessageBox QPushButton {
            background-color: #556b2f;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 20px;
            min-width: 80px;
            font-weight: bold;
        }
        QMessageBox QPushButton:hover {
            background-color: #3d4a2c;
        }
    """)
    
    app = SmartLabelingApp()
    app.show()
    sys.exit(qt_app.exec())

if __name__ == "__main__":
    main()
