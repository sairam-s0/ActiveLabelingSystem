"""GUI entry point for Active Labeling System v2."""

from __future__ import annotations

import atexit
import json
import os
import random
import sys
import tempfile
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.log_adjuster import setup_clean_logging
setup_clean_logging()

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from app import state, utils
from app.window import MainWindow
from core.data_manager import DataManager
from core.dataset_versioner import DatasetVersioner
from core.entropy import EntropyCalculator
from core.inference import InferenceEngine
from core.model_manager import ModelManager
from core.replay_buffer import ReplayBuffer
from core.retrain_policy import RetrainingPolicy
from core.sample_selector import SampleSelector
from core.training_orchestrator import TrainingOrchestrator, start_orchestrator_monitor
from features.manual import ManualManager


_WORKER_ENGINE: InferenceEngine | None = None


def init_worker(model_path: str) -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = InferenceEngine(model_path)


def detect_worker(args: tuple[str, list[str], int, int]) -> dict:
    image_path, selected_classes, threshold_percent, _max_dim = args
    try:
        if _WORKER_ENGINE is None:
            raise RuntimeError("Inference worker was not initialized.")
        detections, image_entropy = _WORKER_ENGINE.predict(
            image_path=image_path,
            selected_classes=selected_classes,
            threshold=max(0.0, min(1.0, threshold_percent / 100.0)),
        )
        return {"detections": detections, "entropy": image_entropy, "error": None}
    except Exception as exc:
        return {"detections": [], "entropy": 0.0, "error": str(exc)}


class SmartLabelingApp:
    @property
    def current_index(self):
        return state.current_index

    @current_index.setter
    def current_index(self, value):
        state.current_index = value

    @property
    def labels(self):
        return state.labels

    @labels.setter
    def labels(self, value):
        state.labels = value

    @property
    def custom_classes(self):
        return state.custom_classes

    @custom_classes.setter
    def custom_classes(self, value):
        state.custom_classes = value

    @property
    def class_samples(self):
        return state.class_samples

    @class_samples.setter
    def class_samples(self, value):
        state.class_samples = value

    @property
    def scale_factor(self):
        return state.scale_factor

    @scale_factor.setter
    def scale_factor(self, value):
        state.scale_factor = value

    def __init__(self):
        state.has_gpu = utils.detect_gpu()
        state.device = "cuda" if state.has_gpu else "cpu"

        self.data_manager = DataManager("labels.json")
        self.replay_buffer = ReplayBuffer(max_size=200)
        self.model_manager = ModelManager("models", base_model_name=state.weights)
        state.weights = self.model_manager.resolve_active_path()

        self._load_model_classes()

        self.orchestrator = TrainingOrchestrator(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
            replay_buffer=self.replay_buffer,
            min_samples=state.min_training_samples,
            num_gpus=1 if state.has_gpu else 0,
        )
        self.orchestrator.on_status_change = self._on_training_status_change
        self.orchestrator.on_training_complete = self._on_training_complete
        self.orchestrator.on_training_failed = self._on_training_failed

        self.window = MainWindow(self)
        self.manual = ManualManager(host=self, window=self.window, state=state)
        self.window.closeEvent = self._on_window_close

        self.entropy_calculator = EntropyCalculator()
        self.sample_selector = SampleSelector(self.data_manager, self.entropy_calculator)
        self.retrain_policy = RetrainingPolicy(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
            min_samples=state.min_training_samples,
            max_wait_hours=24,
            perf_delta_threshold=0.05,
            entropy_shift_threshold=0.15,
        )
        self.dataset_versioner = DatasetVersioner("datasets")
        self._monitor_started = False

        self.init_worker()
        atexit.register(self.cleanup)

    def _on_window_close(self, event):
        if hasattr(self, "manual") and self.manual._active:
            self.manual.exit_manual_mode()
        event.accept()

    def _load_model_classes(self):
        try:
            from ultralytics import YOLO

            model = YOLO(state.weights)
            state.coco_classes = list(model.names.values())
            print(f"[Model] Loaded {len(state.coco_classes)} class names.")
        except Exception as exc:
            print(f"[Model] Could not load class names from {state.weights}: {exc}")
            state.coco_classes = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            ]

    def init_worker(self):
        def initialize():
            try:
                state.executor = ProcessPoolExecutor(
                    max_workers=1,
                    initializer=init_worker,
                    initargs=(state.weights,),
                )
                state.worker_ready = True
                print(f"[Worker] Ready with model: {state.weights}")
            except Exception as exc:
                state.worker_ready = False
                print(f"[Worker] Failed: {exc}")

        threading.Thread(target=initialize, daemon=True).start()

    def reload_worker(self, new_model_path):
        """Shut down old inference worker and restart with a new model."""
        print(f"[Worker] Reloading with new model: {new_model_path}")
        state.worker_ready = False
        old_executor = state.executor
        state.executor = None

        def do_reload():
            # shut down old executor
            if old_executor:
                try:
                    old_executor.shutdown(wait=True, cancel_futures=True)
                except Exception as exc:
                    print(f"[Worker] Old executor shutdown error: {exc}")

            # update weights path
            state.weights = str(new_model_path)

            # create new executor with new model
            try:
                state.executor = ProcessPoolExecutor(
                    max_workers=1,
                    initializer=init_worker,
                    initargs=(state.weights,),
                )
                state.worker_ready = True
                # reload class names from new model
                self._load_model_classes()
                print(f"[Worker] Reloaded successfully with: {state.weights}")
            except Exception as exc:
                state.worker_ready = False
                print(f"[Worker] Reload failed: {exc}")

        threading.Thread(target=do_reload, daemon=True).start()

    def select_folder_with_active_learning(self, folder_path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        folder_path = Path(folder_path)
        all_images = sorted(str(p) for p in folder_path.iterdir() if p.suffix.lower() in exts)
        if not all_images:
            return []

        labeled = set(state.labels.keys())
        unlabeled = [p for p in all_images if p not in labeled]
        if not unlabeled:
            return all_images

        try:
            strategy = "uncertainty"
            if hasattr(self.window, "left_panel"):
                strategy = self.window.left_panel.strategy_combo.currentText().lower()
            prioritized = self.sample_selector.select_batch(
                unlabeled_pool=unlabeled,
                batch_size=len(unlabeled),
                strategy=strategy,
            )
        except Exception as exc:
            print(f"[AL] Selector fallback: {exc}")
            prioritized = unlabeled

        return prioritized + [p for p in all_images if p in labeled]

    def on_label_saved(self, image_path, detections):
        sample = {
            "image_path": image_path,
            "detections": detections,
            "entropy": float(getattr(state, "last_image_entropy", 0.0)),
        }
        self.replay_buffer.add(sample)
        self.persist_labels()

        # Direct training trigger: check queue size vs min_samples
        stats = self.data_manager.get_stats()
        queue_size = stats['training_queue_size']
        print(f"[AL] Training queue: {queue_size}/{state.min_training_samples}")
        if queue_size >= state.min_training_samples:
            if not self.orchestrator.is_training():
                print(f"[AL] Queue reached {queue_size} samples - triggering training!")
                self.trigger_training_with_validation()
            else:
                print("[AL] Training already in progress, will retrain after completion.")

    def configure_label_output(self, folder_path: Path, label_format: str):
        state.label_format = label_format
        state.labels_output_dir = str(folder_path)
        state.labels_output_path = str(folder_path / ("labels_coco.json" if label_format == "coco" else "labels.json"))
        self.data_manager.set_path(str(folder_path / ".labels_internal.json"))

    def persist_labels(self):
        if not state.labels_output_path:
            return
        try:
            out_path = Path(state.labels_output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                self.data_manager.export_coco(image_root=Path(state.labels_output_dir))
                if state.label_format == "coco"
                else self.data_manager.export_simple_json()
            )
            fd, tmp = tempfile.mkstemp(prefix="labels_", suffix=".tmp", dir=str(out_path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, ensure_ascii=False)
                os.replace(tmp, out_path)
            except Exception:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
                raise
        except Exception as exc:
            print(f"[persist_labels] failed: {exc}")

    def process_next(self):
        if state.current_index >= len(state.image_files):
            QMessageBox.information(self.window, "Complete", f"All {len(state.image_files)} images processed.")
            return

        img_path = state.image_files[state.current_index]
        if img_path in state.labels:
            state.current_index += 1
            QTimer.singleShot(0, self.process_next)
            return

        try:
            utils.safe_close_image(state.current_image)
            state.current_image = utils.load_image(img_path)
            state.current_image_path = img_path
            state._current_img_size = state.current_image.size
            self.window.display_image([])
            filename = Path(img_path).name
            if hasattr(self.window.bottom_bar, "filename_label"):
                self.window.bottom_bar.filename_label.setText(filename[:40])
            self.window.update_stats()
            self.manual.on_image_changed()
            self.run_detect(img_path)
        except Exception as exc:
            print(f"[Error] Loading {img_path}: {exc}")
            state.current_index += 1
            QTimer.singleShot(0, self.process_next)

    def run_detect(self, img_path):
        def worker_thread():
            try:
                if not state.executor:
                    raise RuntimeError("Inference executor is not available.")
                args = (img_path, state.selected_classes, state.threshold, state.max_dim)
                result = state.executor.submit(detect_worker, args).result(timeout=60)
                if result.get("error"):
                    state.detection_errors.append((img_path, result["error"]))
                    print(f"[Detect] {result['error']}")
                detections = result.get("detections", [])
                state.last_image_entropy = result.get("entropy") or max(
                    (d.get("entropy", 0.0) for d in detections),
                    default=0.0,
                )
                self.window.result_ready.emit(detections)
            except Exception as exc:
                print(f"[run_detect] {exc}")
                self.window.result_ready.emit([])

        threading.Thread(target=worker_thread, daemon=True).start()

    def should_auto_accept(self):
        return random.random() >= state.qa_rate

    def _ensure_autosave_setup(self, folder_path):
        state.autosave_file = str(Path(folder_path) / "labels_autosave.json")

    def save_autosave(self):
        if not state.autosave_file:
            return
        payload = {
            "labels": state.labels,
            "current_index": state.current_index,
            "auto_accepted_log": state.auto_accepted_log,
            "detection_errors": state.detection_errors,
            "selected_classes": state.selected_classes,
        }
        path = Path(state.autosave_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with state.autosave_lock:
            fd, tmp = tempfile.mkstemp(prefix="autosave_", suffix=".tmp", dir=str(path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, ensure_ascii=False)
                os.replace(tmp, path)
            except Exception:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
                raise

    def load_autosave(self):
        if not state.autosave_file:
            return
        path = Path(state.autosave_file)
        if not path.exists():
            self._restore_from_internal_labels()
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state.labels = data.get("labels", {}) or {}
            state.current_index = int(data.get("current_index", 0) or 0)
            state.auto_accepted_log = data.get("auto_accepted_log", [])
            state.detection_errors = data.get("detection_errors", [])
            if data.get("selected_classes"):
                state.selected_classes = data["selected_classes"]
                self.window.top_bar.class_label.setText(utils.format_class_display(state.selected_classes))
            if state.labels:
                QMessageBox.information(
                    self.window,
                    "Autosave",
                    f"Autosave loaded - {len(state.labels)} labeled images. Resuming at index {state.current_index + 1}.",
                )
            self.window.update_stats()
        except Exception as exc:
            print(f"[load_autosave] failed: {exc}")

    def _restore_from_internal_labels(self):
        if not self.data_manager.data.get("images"):
            return
        state.labels = self.data_manager.export_simple_json()
        labeled = set(state.labels.keys())
        while state.current_index < len(state.image_files) and str(state.image_files[state.current_index]) in labeled:
            state.current_index += 1
        self.window.update_stats()

    def _process_monitor_event(self, event):
        try:
            if event["type"] == "completion":
                self._on_training_complete(event["data"])
        except Exception as exc:
            print(f"[UI] Monitor event error: {exc}")

    def update_stats(self):
        if hasattr(self, "window"):
            self.window.update_stats()

    def _start_background_monitor(self):
        if self._monitor_started:
            return

        def initialize():
            if not self.orchestrator.initialize_ray():
                print("[Main] Ray initialization failed - background training disabled.")
                return
            start_orchestrator_monitor(self.orchestrator, lambda event: self.window.monitor_signal.emit(event))
            self._monitor_started = True

        threading.Thread(target=initialize, daemon=True).start()

    def trigger_training_with_validation(self):
        self._start_background_monitor()
        self.orchestrator.trigger_training()

    def _on_training_status_change(self, status):
        if status.get("status") == "training_started":
            self.window.top_bar.progress_label.setText(f"Training started: {status.get('sample_count', 0)} samples")

    def _on_training_complete(self, result):
        if not result:
            return
        save_path = result.get('save_path', '')
        sample_count = result.get('sample_count', 0)
        self.window.top_bar.progress_label.setText("Training complete - reloading model...")
        print(f"[AL] Training complete! Samples: {sample_count}, Model: {save_path}")

        # Auto-promote: reload inference worker with the shadow model
        if save_path and Path(save_path).exists():
            self.reload_worker(save_path)
            self.window.top_bar.progress_label.setText(
                f"Model retrained ({sample_count} samples) - predictions active!"
            )
            print(f"[AL] Auto-promoted shadow model: {save_path}")
        else:
            self.window.top_bar.progress_label.setText("Training complete")
            print(f"[AL] Shadow model not found at {save_path}, skipping auto-promote")

    def _on_training_failed(self, result):
        QMessageBox.critical(self.window, "Training Failed", result.get("error", "Unknown error"))

    def cleanup(self):
        try:
            if state.executor:
                state.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        utils.safe_close_image(state.current_image)
        if hasattr(self, "orchestrator"):
            self.orchestrator.shutdown()

    def show(self):
        self.window.show()


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="PyQt6")
    qt_app = QApplication(sys.argv)
    
    from app.theme import COLORS
    qss = f"""
        * {{
            font-family: "Segoe UI", "SF Pro Display", -apple-system, sans-serif;
        }}
        QMainWindow {{
            background-color: {COLORS['bg']};
            color: {COLORS['text']};
        }}
        QLabel {{
            color: {COLORS['text']};
            background: transparent;
        }}
        QPushButton {{
            background-color: {COLORS['olive']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 14px;
            font-weight: bold;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: {COLORS['olive_dark']};
        }}
        QPushButton:pressed {{
            background-color: {COLORS['border']};
        }}
        QPushButton:disabled {{
            background-color: {COLORS['border']};
            color: {COLORS['muted']};
        }}
        QComboBox {{
            background-color: {COLORS['panel']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 5px;
            min-width: 120px;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 0px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {COLORS['panel']};
            color: {COLORS['text']};
            selection-background-color: {COLORS['olive']};
            selection-color: white;
            border: 1px solid {COLORS['border']};
        }}
        QSlider::groove:horizontal {{
            background: {COLORS['border']};
            height: 6px;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {COLORS['olive']};
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }}
        QScrollArea {{
            border: 1px solid {COLORS['border']};
            background-color: {COLORS['panel']};
        }}
        QScrollArea > QWidget > QWidget {{
            background-color: {COLORS['panel']};
        }}
        QRadioButton {{
            color: {COLORS['text']};
            background: transparent;
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
        QCheckBox {{
            color: {COLORS['text']};
            background: transparent;
        }}
        QTextEdit, QPlainTextEdit {{
            background-color: {COLORS['bg']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
        }}
        QMessageBox {{
            background-color: {COLORS['panel']};
        }}
        QMessageBox QLabel {{
            color: {COLORS['text']};
            font-size: 13px;
        }}
        QMessageBox QPushButton {{
            min-width: 80px;
            padding: 8px 20px;
        }}
        QDialog {{
            background-color: {COLORS['panel']};
            color: {COLORS['text']};
        }}
        QSplitter::handle {{
            background-color: {COLORS['border']};
            width: 2px;
        }}
        QToolTip {{
            background-color: {COLORS['panel']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            padding: 4px;
        }}
    """
    qt_app.setStyleSheet(qss)
    
    app = SmartLabelingApp()
    app.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
