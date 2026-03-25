# src/core/retrain_policy.py

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
class RetrainingPolicy:
    
    def __init__(
        self,
        data_manager,
        model_manager,
        min_samples: int = 30,
        max_wait_hours: int = 24,
        perf_delta_threshold: float = 0.05,
        entropy_shift_threshold: float = 0.15
    ):
        self.data_manager = data_manager
        self.model_manager = model_manager
        
        # Thresholds
        self.min_samples = min_samples
        self.max_wait_hours = max_wait_hours
        self.perf_delta_threshold = perf_delta_threshold
        self.entropy_shift_threshold = entropy_shift_threshold
        
        # State tracking
        self.last_train_time = None
        self.last_train_stats = None
        self.baseline_entropy_dist = None
    
    def should_retrain(self) -> Tuple[bool, Dict]:
        reasons = {
            'triggered': False,
            'policies': {},
            'recommendation': None
        }
        
        # Policy 1: Sample count threshold (HARD REQUIREMENT)
        sample_trigger, sample_info = self._check_sample_threshold()
        reasons['policies']['sample_count'] = sample_info
        
        if not sample_trigger:
            reasons['recommendation'] = f"Need {sample_info['needed']} more samples"
            return False, reasons
        
        # Policy 2: Time-based trigger
        time_trigger, time_info = self._check_time_threshold()
        reasons['policies']['time_elapsed'] = time_info
        
        # Policy 3: Entropy distribution shift
        entropy_trigger, entropy_info = self._check_entropy_shift()
        reasons['policies']['entropy_shift'] = entropy_info
        
        # Policy 4: Class imbalance
        balance_trigger, balance_info = self._check_class_balance()
        reasons['policies']['class_balance'] = balance_info
        
        # Policy 5: Performance degradation (if we have validation data)
        perf_trigger, perf_info = self._check_performance_delta()
        reasons['policies']['performance'] = perf_info
        
        # Decision logic: Sample threshold + ANY other trigger
        triggers = [
            ('time', time_trigger),
            ('entropy', entropy_trigger),
            ('balance', balance_trigger),
            ('performance', perf_trigger)
        ]
        
        active_triggers = [name for name, flag in triggers if flag]
        
        if active_triggers:
            reasons['triggered'] = True
            reasons['recommendation'] = f"Retrain triggered by: {', '.join(active_triggers)}"
            return True, reasons
        
        # No triggers but samples ready
        reasons['recommendation'] = "Samples ready but no urgency detected"
        return False, reasons
    
    def _check_sample_threshold(self) -> Tuple[bool, Dict]:
        """Check if we have minimum samples for training."""
        stats = self.data_manager.get_stats()
        queue_size = stats['training_queue_size']
        
        info = {
            'current': queue_size,
            'threshold': self.min_samples,
            'ready': queue_size >= self.min_samples,
            'needed': max(0, self.min_samples - queue_size)
        }
        
        return info['ready'], info
    
    def _check_time_threshold(self) -> Tuple[bool, Dict]:
        if not self.last_train_time:
            return False, {'status': 'never_trained'}
        
        now = datetime.now()
        elapsed = now - self.last_train_time
        hours_elapsed = elapsed.total_seconds() / 3600
        
        info = {
            'hours_elapsed': round(hours_elapsed, 1),
            'threshold_hours': self.max_wait_hours,
            'last_trained': self.last_train_time.isoformat()
        }
        
        trigger = hours_elapsed >= self.max_wait_hours
        return trigger, info
    
    def _check_entropy_shift(self) -> Tuple[bool, Dict]:
        stats = self.data_manager.get_stats()
        current_avg_entropy = stats.get('avg_entropy', 0)
        
        if not self.baseline_entropy_dist:
            # First time - establish baseline
            self.baseline_entropy_dist = current_avg_entropy
            return False, {
                'status': 'baseline_established',
                'baseline': current_avg_entropy
            }
        
        # Calculate shift
        shift = abs(current_avg_entropy - self.baseline_entropy_dist)
        relative_shift = shift / (self.baseline_entropy_dist + 1e-6)
        
        info = {
            'current_entropy': round(current_avg_entropy, 4),
            'baseline_entropy': round(self.baseline_entropy_dist, 4),
            'absolute_shift': round(shift, 4),
            'relative_shift': round(relative_shift, 4),
            'threshold': self.entropy_shift_threshold
        }
        
        trigger = relative_shift >= self.entropy_shift_threshold
        return trigger, info
    
    def _check_class_balance(self) -> Tuple[bool, Dict]:
        class_counts = self.data_manager.get_class_counts()
        
        if not class_counts:
            return False, {'status': 'no_classes'}
        
        total = sum(class_counts.values())
        if total == 0:
            return False, {'status': 'no_samples'}
        
        # Calculate statistics
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        # Class proportions
        proportions = {
            cls: count / total 
            for cls, count in class_counts.items()
        }
        
        # Check balance
        min_proportion = min(proportions.values())
        imbalance_ratio = max_count / (min_count + 1e-6)
        
        info = {
            'class_counts': class_counts,
            'min_proportion': round(min_proportion, 3),
            'imbalance_ratio': round(imbalance_ratio, 2),
            'needs_balancing': min_proportion < 0.1 or imbalance_ratio > 10
        }
        
        # Trigger if severely imbalanced
        trigger = info['needs_balancing']
        return trigger, info
    
    def _check_performance_delta(self) -> Tuple[bool, Dict]:
        if not self.last_train_stats:
            return False, {'status': 'no_baseline'}
        
        # Get recent samples (last 20)
        recent_paths = self.data_manager.get_all_labeled_images()[:20]
        
        if len(recent_paths) < 10:
            return False, {'status': 'insufficient_recent_data'}
        
        # Calculate average confidence on recent samples
        recent_confs = []
        for path in recent_paths:
            img_data = self.data_manager.get_labels(path)
            if img_data and img_data.get('detections'):
                for det in img_data['detections']:
                    conf = det.get('confidence', 0) / 100.0
                    recent_confs.append(conf)
        
        if not recent_confs:
            return False, {'status': 'no_confidence_data'}
        
        current_avg_conf = np.mean(recent_confs)
        baseline_avg_conf = self.last_train_stats.get('avg_confidence', current_avg_conf)
        
        # Performance delta
        delta = baseline_avg_conf - current_avg_conf
        
        info = {
            'current_confidence': round(current_avg_conf, 4),
            'baseline_confidence': round(baseline_avg_conf, 4),
            'delta': round(delta, 4),
            'threshold': self.perf_delta_threshold
        }
        
        # Trigger if performance dropped significantly
        trigger = delta >= self.perf_delta_threshold
        return trigger, info
    
    def on_training_complete(self, training_result: Dict):
        self.last_train_time = datetime.now()
        
        # Store training statistics
        stats = self.data_manager.get_stats()
        self.last_train_stats = {
            'sample_count': training_result.get('sample_count', 0),
            'avg_entropy': stats.get('avg_entropy', 0),
            'class_counts': stats.get('class_counts', {}),
            'avg_confidence': self._calculate_avg_confidence(),
            'timestamp': self.last_train_time.isoformat()
        }
        
        # Update baseline entropy
        self.baseline_entropy_dist = stats.get('avg_entropy', 0)
        
        print(f"[RetrainPolicy] Training completed, baseline updated")
    
    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence across all labeled samples."""
        all_confs = []
        for img_path in self.data_manager.get_all_labeled_images():
            img_data = self.data_manager.get_labels(img_path)
            if img_data and img_data.get('detections'):
                for det in img_data['detections']:
                    conf = det.get('confidence', 0) / 100.0
                    all_confs.append(conf)
        
        return np.mean(all_confs) if all_confs else 0.5
    
    def force_retrain(self) -> Tuple[bool, Dict]:
        sample_trigger, sample_info = self._check_sample_threshold()
        
        if not sample_trigger:
            return False, {
                'error': 'Insufficient samples for training',
                'info': sample_info
            }
        
        return True, {
            'status': 'forced_retrain',
            'bypassed_policies': True
        }
    
    def get_status(self) -> Dict:
        should_train, reasons = self.should_retrain()
        
        return {
            'should_retrain': should_train,
            'reasons': reasons,
            'last_trained': self.last_train_time.isoformat() if self.last_train_time else None,
            'baseline_stats': self.last_train_stats
        }