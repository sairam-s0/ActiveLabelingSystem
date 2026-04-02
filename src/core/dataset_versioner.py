# src/core/dataset_versioner.py
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import zipfile


class DatasetVersioner:
    
    def __init__(self, versions_dir: str = "datasets"):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
        
        self.manifest_file = self.versions_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {
            'versions': {},
            'latest': None,
            'lineage': {}
        }
    
    def _save_manifest(self):
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def create_version(
        self,
        data_manager,
        version_name: Optional[str] = None,
        description: str = "",
        parent_version: Optional[str] = None
    ) -> Dict:
        # generate version
        if not version_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"v_{timestamp}"
        
        # create version
        version_dir = self.versions_dir / version_name
        if version_dir.exists():
            raise ValueError(f"Version {version_name} already exists")
        
        version_dir.mkdir(parents=True)
        
        # get current
        all_images = data_manager.get_all_labeled_images()
        
        # copy images
        images_dir = version_dir / "images"
        labels_dir = version_dir / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        copied_count = 0
        class_distribution = {}
        total_instances = 0
        
        for img_path in all_images:
            src_path = Path(img_path)
            if not src_path.exists():
                continue
            
            # copy image
            dst_img = images_dir / src_path.name
            shutil.copy2(src_path, dst_img)
            
            # get labels
            img_data = data_manager.get_labels(img_path)
            if not img_data or not img_data.get('detections'):
                continue
            
            # create yolo
            label_file = labels_dir / f"{src_path.stem}.txt"
            
            w = img_data.get('width', 0)
            h = img_data.get('height', 0)
            
            with open(label_file, 'w') as f:
                for det in img_data['detections']:
                    cls_name = det.get('class', 'unknown')
                    class_id = data_manager.get_class_id(cls_name)
                    
                    if class_id == -1:
                        continue
                    
                    bbox = det['bbox']
                    
                    # convert to
                    x_c = ((bbox[0] + bbox[2]) / 2) / w
                    y_c = ((bbox[1] + bbox[3]) / 2) / h
                    width = (bbox[2] - bbox[0]) / w
                    height = (bbox[3] - bbox[1]) / h
                    
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {width:.6f} {height:.6f}\n")
                    
                    # track class
                    class_distribution[cls_name] = class_distribution.get(cls_name, 0) + 1
                    total_instances += 1
            
            copied_count += 1
        
        # create data
        class_mapping = data_manager.data.get('class_mapping', {})
        class_names = {v: k for k, v in class_mapping.items()}
        
        yaml_data = {
            'path': str(version_dir.absolute()),
            'train': 'images',
            'val': 'images',  # can split
            'names': class_names
        }
        
        with open(version_dir / "data.yaml", 'w') as f:
            import yaml
            yaml.dump(yaml_data, f)
        
        # calculate dataset
        dataset_hash = self._calculate_dataset_hash(images_dir, labels_dir)
        
        # create version
        stats = data_manager.get_stats()
        
        metadata = {
            'version': version_name,
            'created': datetime.now().isoformat(),
            'description': description,
            'parent_version': parent_version,
            'hash': dataset_hash,
            'statistics': {
                'total_images': copied_count,
                'total_instances': total_instances,
                'num_classes': len(class_distribution),
                'class_distribution': class_distribution,
                'avg_entropy': stats.get('avg_entropy', 0)
            },
            'class_mapping': class_mapping,
            'files': {
                'images': str(images_dir),
                'labels': str(labels_dir),
                'yaml': str(version_dir / "data.yaml")
            }
        }
        
        # save metadata
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # update manifest
        self.manifest['versions'][version_name] = metadata
        self.manifest['latest'] = version_name
        
        if parent_version:
            if version_name not in self.manifest['lineage']:
                self.manifest['lineage'][version_name] = []
            self.manifest['lineage'][version_name].append(parent_version)
        
        self._save_manifest()
        
        print(f"[DatasetVersioner] Created version '{version_name}' with {copied_count} images")
        
        return metadata
    
    def _calculate_dataset_hash(self, images_dir: Path, labels_dir: Path) -> str:
        hasher = hashlib.sha256()
        
        # hash all
        for img_file in sorted(images_dir.glob("*")):
            if img_file.is_file():
                with open(img_file, 'rb') as f:
                    hasher.update(f.read())
        
        # hash all
        for label_file in sorted(labels_dir.glob("*.txt")):
            with open(label_file, 'rb') as f:
                hasher.update(f.read())
        
        return hasher.hexdigest()[:16]  # short hash
    
    def export_version(self, version_name: str, export_path: Optional[str] = None) -> str:
        if version_name not in self.manifest['versions']:
            raise ValueError(f"Version {version_name} not found")
        
        version_dir = self.versions_dir / version_name
        
        if not export_path:
            export_path = self.versions_dir / f"{version_name}.zip"
        
        # create zip
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in version_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(version_dir.parent)
                    zipf.write(file_path, arcname)
        
        print(f"[DatasetVersioner] Exported {version_name} to {export_path}")
        return str(export_path)
    
    def compare_versions(self, version_a: str, version_b: str) -> Dict:
        if version_a not in self.manifest['versions']:
            raise ValueError(f"Version {version_a} not found")
        if version_b not in self.manifest['versions']:
            raise ValueError(f"Version {version_b} not found")
        
        meta_a = self.manifest['versions'][version_a]
        meta_b = self.manifest['versions'][version_b]
        
        stats_a = meta_a['statistics']
        stats_b = meta_b['statistics']
        
        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'image_count_delta': stats_b['total_images'] - stats_a['total_images'],
            'instance_count_delta': stats_b['total_instances'] - stats_a['total_instances'],
            'new_classes': [],
            'removed_classes': [],
            'class_distribution_changes': {}
        }
        
        # class changes
        classes_a = set(stats_a['class_distribution'].keys())
        classes_b = set(stats_b['class_distribution'].keys())
        
        comparison['new_classes'] = list(classes_b - classes_a)
        comparison['removed_classes'] = list(classes_a - classes_b)
        
        # per class
        for cls in classes_a | classes_b:
            count_a = stats_a['class_distribution'].get(cls, 0)
            count_b = stats_b['class_distribution'].get(cls, 0)
            delta = count_b - count_a
            
            if delta != 0:
                comparison['class_distribution_changes'][cls] = delta
        
        return comparison
    
    def get_lineage(self, version_name: str) -> List[str]:
        lineage = []
        current = version_name
        
        while current in self.manifest['lineage']:
            parents = self.manifest['lineage'][current]
            if not parents:
                break
            parent = parents[0]  # take first
            lineage.append(parent)
            current = parent
        
        return lineage
    
    def list_versions(self) -> List[Dict]:
        versions = []
        for name, meta in self.manifest['versions'].items():
            versions.append({
                'version': name,
                'created': meta['created'],
                'images': meta['statistics']['total_images'],
                'instances': meta['statistics']['total_instances'],
                'classes': meta['statistics']['num_classes'],
                'description': meta.get('description', ''),
                'hash': meta['hash']
            })
        
        # sort by
        versions.sort(key=lambda v: v['created'], reverse=True)
        return versions
    
    def get_version_metadata(self, version_name: str) -> Dict:
        if version_name not in self.manifest['versions']:
            raise ValueError(f"Version {version_name} not found")
        
        return self.manifest['versions'][version_name]
    
    def verify_integrity(self, version_name: str) -> bool:
        if version_name not in self.manifest['versions']:
            raise ValueError(f"Version {version_name} not found")
        
        version_dir = self.versions_dir / version_name
        images_dir = version_dir / "images"
        labels_dir = version_dir / "labels"
        
        current_hash = self._calculate_dataset_hash(images_dir, labels_dir)
        stored_hash = self.manifest['versions'][version_name]['hash']
        
        match = current_hash == stored_hash
        
        if not match:
            print(f"[DatasetVersioner] WARNING: Hash mismatch for {version_name}")
            print(f"  Expected: {stored_hash}")
            print(f"  Got:      {current_hash}")
        
        return match