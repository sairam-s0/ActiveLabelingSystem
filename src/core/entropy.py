# src/core/entropy.py
import math
import numpy as np
import torch

class EntropyCalculator:
    @staticmethod
    def normalized_entropy(probs: list[float]) -> float:
        if not probs:
            return 0.0

        probs = np.clip(np.array(probs, dtype=float), 1e-9, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = math.log(len(probs))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    @staticmethod
    def from_yolo_output(box_output, num_classes: int = 80) -> float:
        # option 1
        if hasattr(box_output, 'probs') and box_output.probs is not None:
            probs = box_output.probs.cpu().numpy()
            return EntropyCalculator.normalized_entropy(probs.tolist())
        
        # option 2
        if hasattr(box_output, 'data') and isinstance(box_output.data, torch.Tensor):
            data = box_output.data
            if data.ndim == 2 and data.shape[1] > 6:
                # assume format
                if data.shape[1] == 6 + num_classes:
                    logits = data[:, 6:]  # extract logits
                    # use the
                    if logits.shape[0] > 0:
                        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
                        return EntropyCalculator.normalized_entropy(probs.tolist())

        try:
            # yolov8 results
            if hasattr(box_output, 'boxes') and box_output.boxes is not None:
                boxes = box_output.boxes  # ultralytics yolov8
                if hasattr(boxes, 'conf') and len(boxes.conf) > 0:
                    # use the
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    max_conf_idx = int(np.argmax(confs))
                    confidence = float(confs[max_conf_idx])
                    pred_class = int(classes[max_conf_idx])

                    # build synthetic
                    if num_classes <= 1:
                        return 0.0

                    probs = np.full(num_classes, (1.0 - confidence) / (num_classes - 1))
                    probs[pred_class] = confidence

                    # clip to
                    probs = np.clip(probs, 1e-9, 1.0)
                    probs /= probs.sum()

                    return EntropyCalculator.normalized_entropy(probs.tolist())
        except (AttributeError, IndexError, ValueError, TypeError) as e:
            # if any
            pass

        # absolute fallback
        return 0.0

    @staticmethod
    def image_entropy(detections: list[dict]) -> float:
        if not detections:
            return 0.0
        entropies = [d.get("entropy", 0.0) for d in detections]
        return max(entropies, default=0.0)

    @staticmethod
    def aggregate_entropy(detections: list[dict], method='max') -> float:
        if not detections:
            return 0.0
        
        entropies = [d.get("entropy", 0.0) for d in detections]
        confidences = [d.get('confidence', 0.0) for d in detections]
        
        if method == 'max':
            return max(entropies)
        elif method == 'mean':
            return sum(entropies) / len(entropies)
        elif method == 'weighted':
            total_conf = sum(confidences)
            if total_conf == 0:
                return 0.0
            weighted_sum = sum(e * c for e, c in zip(entropies, confidences))
            return weighted_sum / total_conf
            
        return max(entropies)