# src/core/data_manager.py
import json
from pathlib import Path
from threading import Lock
from datetime import datetime

class DataManager:
    def __init__(self, json_path: str, entropy_threshold: float = 0.6):
        self.path = Path(json_path)
        self.entropy_threshold = entropy_threshold  # ✅ Configurable threshold
        self.lock = Lock()
        self.data = self._empty_data()
        self._load()
        # Build class mapping on init in case file exists
        if self.data["images"]:
            self.build_class_mapping()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    loaded.setdefault("class_mapping", {})
                    self.data = loaded
            except Exception as e:
                print(f"[DataManager] Error loading data: {e}")

    def _empty_data(self):
        return {
            "images": {},
            "training_queue": [],
            "trained_images": [],
            "class_mapping": {}
        }

    def reset(self):
        self.data = self._empty_data()

    def set_path(self, json_path: str):
        self.path = Path(json_path)
        self.reset()
        self._load()
        if self.data["images"]:
            self.build_class_mapping()

    def save(self):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)

    def save_labels(self, image_path: str, detections: list, 
                   entropy: float = 0.0, img_width: int = 0, img_height: int = 0):
        """Save labels with full metadata including dimensions."""
        
        # Validate required fields
        if img_width <= 0 or img_height <= 0:
            print(f"[DataManager] WARNING: Invalid dimensions for {image_path}: {img_width}x{img_height}")
            return False
        
        self.data["images"][image_path] = {
            "detections": detections,
            "entropy": entropy,
            "width": img_width,
            "height": img_height,
            "timestamp": datetime.now().isoformat(),
            "status": "labeled"
        }

        # 🔍 DEBUG LOGGING + USE CONFIGURABLE THRESHOLD
        will_queue = (entropy >= self.entropy_threshold) and (image_path not in self.data["training_queue"])
        print(f"[DataManager] Entropy check: {image_path} → entropy={entropy:.4f}, "
              f"threshold={self.entropy_threshold:.2f}, queued={will_queue}")

        # Add to training queue if high entropy
        if entropy >= self.entropy_threshold and image_path not in self.data["training_queue"]:
            self.data["training_queue"].append(image_path)
            self.data["images"][image_path]["status"] = "queued"

        # 🔒 CRITICAL: Rebuild class mapping after every label save
        self.build_class_mapping()
        self.save()
        return True

    def build_class_mapping(self):
        """Scan all detections and build deterministic {'class_name': id} mapping."""
        class_names = set()
        for img_data in self.data["images"].values():
            for det in img_data.get("detections", []):
                class_name = det.get("class")
                if class_name:
                    class_names.add(class_name)
        
        # Sort for deterministic ordering across sessions
        sorted_classes = sorted(class_names)
        self.data["class_mapping"] = {
            name: idx for idx, name in enumerate(sorted_classes)
        }
        
        if sorted_classes:
            print(f"[DataManager] Class mapping updated: {self.data['class_mapping']}")

    def get_class_id(self, class_name: str) -> int:
        """Return class ID; -1 if unknown."""
        return self.data["class_mapping"].get(class_name, -1)

    def get_class_counts(self) -> dict:
        """Count total labeled instances per class across all images."""
        counts = {}
        for img_data in self.data["images"].values():
            for det in img_data.get("detections", []):
                cls_name = det.get("class")
                if cls_name:
                    counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts

    def get_class_balance(self, target_classes: list = None, min_samples: int = 50) -> dict:
        """
        Identify which classes need more samples.
        Returns: {'person': 5, 'seatbelt': 42} → "need 5 more 'person', 42 more 'seatbelt'"
        """
        if target_classes is None:
            target_classes = list(self.data["class_mapping"].keys())
        
        current_counts = self.get_class_counts()
        needed = {}
        for cls in target_classes:
            current = current_counts.get(cls, 0)
            deficit = max(0, min_samples - current)
            if deficit > 0:
                needed[cls] = deficit
        return needed

    def prepare_training_samples(self, image_paths: list) -> list[dict]:
        """
        Convert image paths into full sample dicts for shadow trainer.
        
        Returns list of dicts with:
        - image_path: str
        - detections: list with class_id instead of class name
        - width: int (validated)
        - height: int (validated)
        - entropy: float
        - timestamp: str (for replay buffer aging)
        """
        samples = []
        skipped = 0
        
        for path in image_paths:
            img_data = self.data["images"].get(path)
            if not img_data:
                skipped += 1
                continue
            
            # Validate dimensions
            width = img_data.get("width", 0)
            height = img_data.get("height", 0)
            if width <= 0 or height <= 0:
                print(f"[DataManager] Skipping {path}: invalid dimensions {width}x{height}")
                skipped += 1
                continue
            
            # Convert class names to IDs
            detections_with_ids = []
            for det in img_data.get("detections", []):
                cls_name = det.get("class")
                cls_id = self.get_class_id(cls_name)
                
                if cls_id == -1:
                    print(f"[DataManager] Unknown class '{cls_name}', skipping detection")
                    continue
                
                # Create new detection dict with class_id
                det_copy = {
                    "bbox": det.get("bbox"),
                    "confidence": det.get("confidence", 0),
                    "class_id": cls_id,
                    "class_name": cls_name,  # Keep for reference
                    "entropy": det.get("entropy", 0.0)
                }
                detections_with_ids.append(det_copy)
            
            if not detections_with_ids:
                print(f"[DataManager] Skipping {path}: no valid detections")
                skipped += 1
                continue
            
            sample = {
                "image_path": path,
                "detections": detections_with_ids,
                "width": width,
                "height": height,
                "entropy": img_data.get("entropy", 0.0),
                "timestamp": img_data.get("timestamp", datetime.now().isoformat())
            }
            samples.append(sample)
        
        if skipped > 0:
            print(f"[DataManager] Prepared {len(samples)} samples, skipped {skipped}")
        
        return samples

    def get_training_batch(self, count=30, new_only=True, return_full_samples=False) -> list:
        """
        Get training batch.
        
        Args:
            count: Number of samples to return
            new_only: If True, only return samples from training_queue
            return_full_samples: If True, returns full sample dicts via prepare_training_samples()
                                If False, returns just paths (legacy behavior)
        
        Returns:
            List of paths or list of full sample dicts
        """
        if new_only:
            paths = self.data["training_queue"][:count]
        else:
            paths = self.get_all_labeled_images()[:count]

        if return_full_samples:
            return self.prepare_training_samples(paths)
        else:
            return paths

    def get_labels(self, image_path: str) -> dict:
        """Get label data for specific image."""
        return self.data["images"].get(image_path)

    def get_all_labeled_images(self) -> list:
        """Get all labeled images sorted by entropy (highest first)."""
        return sorted(
            self.data["images"].keys(),
            key=lambda k: self.data["images"][k].get("entropy", 0),
            reverse=True
        )

    def get_replay_samples(self, count=10, min_entropy=0.5) -> list:
        """Get high-quality samples from trained set for replay buffer."""
        candidates = [
            path for path in self.data["trained_images"]
            if self.data["images"].get(path, {}).get("entropy", 0) > min_entropy
        ]
        candidates.sort(
            key=lambda k: self.data["images"][k]["entropy"], 
            reverse=True
        )
        return candidates[:count]

    def mark_trained(self, image_paths: list):
        """Mark images as trained and remove from training queue."""
        for path in image_paths:
            if path in self.data["training_queue"]:
                self.data["training_queue"].remove(path)
            if path not in self.data["trained_images"]:
                self.data["trained_images"].append(path)
            if path in self.data["images"]:
                self.data["images"][path]["status"] = "trained"
        self.save()
        print(f"[DataManager] Marked {len(image_paths)} images as trained")

    def get_stats(self) -> dict:
        """Get current statistics."""
        total_entropy = sum(
            img.get("entropy", 0) for img in self.data["images"].values()
        )
        avg_entropy = total_entropy / len(self.data["images"]) if self.data["images"] else 0.0
        
        class_counts = self.get_class_counts()
        
        return {
            "total_labeled": len(self.data["images"]),
            "training_queue_size": len(self.data["training_queue"]),
            "trained_count": len(self.data.get("trained_images", [])),
            "avg_entropy": round(avg_entropy, 4),
            "class_counts": class_counts,
            "num_classes": len(self.data["class_mapping"])
        }

    def export_simple_json(self) -> dict:
        """Export labels to a simple JSON mapping."""
        exported = {}
        for img_path, img_data in self.data["images"].items():
            exported[img_path] = {
                "detections": img_data.get("detections", []),
                "timestamp": img_data.get("timestamp"),
                "image_width": img_data.get("width", 0),
                "image_height": img_data.get("height", 0),
                "entropy": img_data.get("entropy", 0.0),
                "status": img_data.get("status", "labeled")
            }
        return exported

    def export_coco(self, image_root=None) -> dict:
        """Export labels to COCO JSON format."""
        coco = {"images": [], "annotations": [], "categories": []}

        class_names = sorted(self.data["class_mapping"].keys())
        class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}
        for name in class_names:
            coco["categories"].append({
                "id": class_to_id[name],
                "name": name,
                "supercategory": "object"
            })

        ann_id = 1
        image_items = sorted(self.data["images"].items(), key=lambda x: x[0])
        for img_id, (img_path, img_data) in enumerate(image_items, start=1):
            file_name = Path(img_path).name
            if image_root:
                try:
                    file_name = str(Path(img_path).relative_to(image_root))
                except Exception:
                    file_name = Path(img_path).name

            coco["images"].append({
                "id": img_id,
                "file_name": file_name,
                "width": img_data.get("width", 0),
                "height": img_data.get("height", 0)
            })

            for det in img_data.get("detections", []):
                bbox = det.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                w = max(0.0, float(x2) - float(x1))
                h = max(0.0, float(y2) - float(y1))
                cls_name = det.get("class", "unknown")
                cat_id = class_to_id.get(cls_name)
                if not cat_id:
                    continue

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

        return coco
