# src/core/feedback_validator.py

import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json


class FeedbackValidator:    
    def __init__(self, data_manager, model_manager):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.validation_history = []
    
    def validate_training_impact(
        self, 
        trained_paths: List[str],
        before_model: str,
        after_model: str
    ) -> Dict:
        # Load models
        try:
            from ultralytics import YOLO
            model_before = YOLO(before_model)
            model_after = YOLO(after_model)
        except Exception as e:
            return {'error': f'Could not load models: {e}'}
        
        # Metrics to track
        results = {
            'timestamp': np.datetime64('now').astype(str),
            'sample_count': len(trained_paths),
            'improvements': {},
            'regressions': {},
            'overall': {}
        }
        
        # Test on trained samples (should improve)
        trained_metrics = self._compare_on_samples(
            trained_paths, 
            model_before, 
            model_after
        )
        results['trained_set'] = trained_metrics
        
        # Test on held-out samples (generalization check)
        all_labeled = self.data_manager.get_all_labeled_images()
        holdout = [p for p in all_labeled if p not in trained_paths][:50]
        
        if holdout:
            holdout_metrics = self._compare_on_samples(
                holdout,
                model_before,
                model_after
            )
            results['holdout_set'] = holdout_metrics
        
        # Analyze by class
        class_analysis = self._analyze_per_class_improvement(
            trained_paths,
            model_before,
            model_after
        )
        results['per_class'] = class_analysis
        
        # Entropy calibration check
        entropy_analysis = self._validate_entropy_calibration(
            trained_paths,
            model_after
        )
        results['entropy_calibration'] = entropy_analysis
        
        # Overall verdict
        results['overall'] = self._compute_overall_verdict(results)
        
        # Store for history
        self.validation_history.append(results)
        
        return results
    
    def _compare_on_samples(
        self,
        image_paths: List[str],
        model_before,
        model_after
    ) -> Dict:
        before_stats = {'confidences': [], 'entropies': [], 'ious': []}
        after_stats = {'confidences': [], 'entropies': [], 'ious': []}
        
        for img_path in image_paths[:30]:  # Limit for speed
            if not Path(img_path).exists():
                continue
            
            # Get ground truth labels
            gt_data = self.data_manager.get_labels(img_path)
            if not gt_data or not gt_data.get('detections'):
                continue
            
            gt_boxes = [det['bbox'] for det in gt_data['detections']]
            
            # Before model predictions
            try:
                results_before = model_before(img_path, verbose=False)
                if results_before and results_before[0].boxes:
                    for box in results_before[0].boxes:
                        conf = float(box.conf)
                        before_stats['confidences'].append(conf)
                        
                        # Calculate IoU with GT
                        pred_box = box.xyxy[0].tolist()
                        max_iou = max(
                            [self._calculate_iou(pred_box, gt) for gt in gt_boxes],
                            default=0
                        )
                        before_stats['ious'].append(max_iou)
            except Exception:
                pass
            
            # After model predictions
            try:
                results_after = model_after(img_path, verbose=False)
                if results_after and results_after[0].boxes:
                    for box in results_after[0].boxes:
                        conf = float(box.conf)
                        after_stats['confidences'].append(conf)
                        
                        pred_box = box.xyxy[0].tolist()
                        max_iou = max(
                            [self._calculate_iou(pred_box, gt) for gt in gt_boxes],
                            default=0
                        )
                        after_stats['ious'].append(max_iou)
            except Exception:
                pass
        
        # Compute deltas
        metrics = {}
        
        if before_stats['confidences'] and after_stats['confidences']:
            metrics['avg_confidence_before'] = float(np.mean(before_stats['confidences']))
            metrics['avg_confidence_after'] = float(np.mean(after_stats['confidences']))
            metrics['confidence_delta'] = metrics['avg_confidence_after'] - metrics['avg_confidence_before']
        
        if before_stats['ious'] and after_stats['ious']:
            metrics['avg_iou_before'] = float(np.mean(before_stats['ious']))
            metrics['avg_iou_after'] = float(np.mean(after_stats['ious']))
            metrics['iou_delta'] = metrics['avg_iou_after'] - metrics['avg_iou_before']
        
        metrics['sample_count'] = len(image_paths)
        
        return metrics
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection area
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_per_class_improvement(
        self,
        trained_paths: List[str],
        model_before,
        model_after
    ) -> Dict:
        class_metrics = {}
        
        # Group samples by class
        class_samples = {}
        for img_path in trained_paths:
            gt_data = self.data_manager.get_labels(img_path)
            if not gt_data or not gt_data.get('detections'):
                continue
            
            for det in gt_data['detections']:
                cls = det.get('class', 'unknown')
                if cls not in class_samples:
                    class_samples[cls] = []
                class_samples[cls].append(img_path)
        
        # Test each class
        for cls, samples in class_samples.items():
            metrics = self._compare_on_samples(
                samples[:20],  # Limit per class
                model_before,
                model_after
            )
            
            class_metrics[cls] = {
                'sample_count': len(samples),
                'iou_improvement': metrics.get('iou_delta', 0),
                'conf_improvement': metrics.get('confidence_delta', 0)
            }
        
        return class_metrics
    
    def _validate_entropy_calibration(
        self,
        trained_paths: List[str],
        model: object
    ) -> Dict:
        entropy_changes = []
        
        for img_path in trained_paths[:30]:
            if not Path(img_path).exists():
                continue
            
            # Get old entropy (from when we selected this sample)
            gt_data = self.data_manager.get_labels(img_path)
            old_entropy = gt_data.get('entropy', 0) if gt_data else 0
            
            # Run new model and calculate new entropy
            try:
                from core.entropy import EntropyCalculator
                results = model(img_path, verbose=False)
                
                if results and results[0].boxes:
                    # Calculate entropy on new predictions
                    new_entropies = []
                    for box in results[0].boxes:
                        # This requires model to output probabilities
                        # Simplified: use confidence as proxy
                        conf = float(box.conf)
                        # Entropy ≈ uncertainty ≈ (1 - confidence)
                        pseudo_entropy = 1.0 - conf
                        new_entropies.append(pseudo_entropy)
                    
                    new_entropy = max(new_entropies) if new_entropies else old_entropy
                    
                    entropy_changes.append({
                        'old': old_entropy,
                        'new': new_entropy,
                        'delta': old_entropy - new_entropy
                    })
            except Exception:
                pass
        
        if not entropy_changes:
            return {'status': 'no_data'}
        
        avg_delta = np.mean([e['delta'] for e in entropy_changes])
        
        return {
            'avg_entropy_reduction': float(avg_delta),
            'samples_analyzed': len(entropy_changes),
            'positive_reductions': sum(1 for e in entropy_changes if e['delta'] > 0),
            'calibration_quality': 'good' if avg_delta > 0.1 else 'poor'
        }
    
    def _compute_overall_verdict(self, results: Dict) -> Dict:
        verdict = {
            'success': False,
            'recommendation': '',
            'confidence': 0.0
        }
        
        # Check trained set improvement
        trained = results.get('trained_set', {})
        iou_improved = trained.get('iou_delta', 0) > 0.05
        
        # Check entropy calibration
        entropy = results.get('entropy_calibration', {})
        entropy_reduced = entropy.get('avg_entropy_reduction', 0) > 0.1
        
        # Check per-class improvements
        per_class = results.get('per_class', {})
        improved_classes = sum(
            1 for cls_data in per_class.values()
            if cls_data.get('iou_improvement', 0) > 0.03
        )
        
        # Decision logic
        success_indicators = [iou_improved, entropy_reduced, improved_classes > 0]
        success_count = sum(success_indicators)
        
        verdict['success'] = success_count >= 2
        verdict['confidence'] = success_count / 3.0
        
        if verdict['success']:
            verdict['recommendation'] = "Training effective - continue labeling high-entropy samples"
        else:
            verdict['recommendation'] = "Training impact unclear - review sample selection strategy"
        
        return verdict
    
    def get_validation_summary(self) -> Dict:
        """Get summary of all validation runs."""
        if not self.validation_history:
            return {'status': 'no_validations'}
        
        return {
            'total_validations': len(self.validation_history),
            'success_rate': sum(
                1 for v in self.validation_history 
                if v.get('overall', {}).get('success', False)
            ) / len(self.validation_history),
            'latest': self.validation_history[-1]
        }