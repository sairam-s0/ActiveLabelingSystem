# src/core/inference.py
from ultralytics import YOLO
from .entropy import EntropyCalculator
from .sam_adapter import segment_boxes
import torch
import numpy as np

class InferenceEngine:
    def __init__(self, model_path: str, enable_sam: bool = True):
        self.model_path = model_path
        self.enable_sam = enable_sam
        self.model = YOLO(model_path)
        self.num_classes = len(self.model.names)

    def reload(self):
        self.model = YOLO(self.model_path)

    def predict(self, image_path: str, selected_classes: list = None, threshold: float = 0.5):
        results = self.model(image_path, verbose=False)
        detections = []

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return detections, 0.0

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = self.model.names[cls_id]

            # critical fix
            if hasattr(box, 'probs') and box.probs is not None:
                # classification model
                probs = box.probs.cpu().numpy()
            elif hasattr(box, 'data') and box.data.shape[-1] > 6:
                # custom detection
                logits = box.data[5:] # adjust index
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                # fallback standard
                # we construct
                # ideally you
                probs = np.zeros(self.num_classes)
                probs[cls_id] = conf
                remainder = (1.0 - conf) / (self.num_classes - 1)
                probs = np.where(probs == 0, remainder, probs)

            entropy = EntropyCalculator.normalized_entropy(probs.tolist())

            # filters
            if selected_classes and class_name not in selected_classes:
                continue
            if conf < threshold:
                continue

            detections.append({
                "bbox": [float(x) for x in box.xyxy[0].tolist()],
                "confidence": conf * 100,
                "class": class_name,
                "entropy": entropy
            })

        if self.enable_sam:
            detections = segment_boxes(image_path, detections)

        return detections, EntropyCalculator.image_entropy(detections)

    def predict_batch(self, image_paths: list) -> list:
        results = []
        for path in image_paths:
            det, ent = self.predict(path)
            results.append((det, ent))
        return results
