# src/core/sample_selector.py


import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from collections import defaultdict


class SampleSelector:
    
    def __init__(self, data_manager, entropy_calculator):
        self.data_manager = data_manager
        self.entropy_calc = entropy_calculator
        self.selection_history = []
        
    def select_batch(
        self, 
        unlabeled_pool: List[str],
        batch_size: int = 10,
        strategy: str = 'uncertainty'
    ) -> List[str]:
        if strategy == 'uncertainty':
            return self._uncertainty_sampling(unlabeled_pool, batch_size)
        elif strategy == 'margin':
            return self._margin_sampling(unlabeled_pool, batch_size)
        elif strategy == 'diversity':
            return self._diversity_sampling(unlabeled_pool, batch_size)
        elif strategy == 'balanced':
            return self._balanced_sampling(unlabeled_pool, batch_size)
        else:
            return self._uncertainty_sampling(unlabeled_pool, batch_size)
    
    def _uncertainty_sampling(self, pool: List[str], k: int) -> List[str]:
        # Get entropy scores for all unlabeled images
        scores = []
        for img_path in pool:
            img_data = self.data_manager.get_labels(img_path)
            if img_data:
                entropy = img_data.get('entropy', 0.0)
            else:
                # No detections yet - assign medium priority
                entropy = 0.5
            scores.append((img_path, entropy))
        
        # Sort by entropy (descending) and take top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [path for path, _ in scores[:k]]
        
        self._log_selection(selected, 'uncertainty')
        return selected
    
    def _margin_sampling(self, pool: List[str], k: int) -> List[str]:
        scores = []
        for img_path in pool:
            img_data = self.data_manager.get_labels(img_path)
            if not img_data or not img_data.get('detections'):
                scores.append((img_path, 0.5))  # Default medium priority
                continue
            
            # Calculate margin from detections
            margins = []
            for det in img_data['detections']:
                # If we have class probabilities, calculate margin
                # Otherwise use confidence as proxy
                conf = det.get('confidence', 0) / 100.0
                # Margin = 1 - conf (smaller = more uncertain)
                margin = 1.0 - conf
                margins.append(margin)
            
            # Use max margin (most uncertain detection)
            max_margin = max(margins) if margins else 0.5
            scores.append((img_path, max_margin))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [path for path, _ in scores[:k]]
        
        self._log_selection(selected, 'margin')
        return selected
    
    def _diversity_sampling(self, pool: List[str], k: int) -> List[str]:
        # Start with uncertainty scores
        uncertainty_scores = {}
        for img_path in pool:
            img_data = self.data_manager.get_labels(img_path)
            entropy = img_data.get('entropy', 0.5) if img_data else 0.5
            uncertainty_scores[img_path] = entropy
        
        selected = []
        remaining = set(pool)
        
        # Greedy selection loop
        for _ in range(min(k, len(pool))):
            if not remaining:
                break
            
            best_score = -1
            best_path = None
            
            for img_path in remaining:
                # Score = uncertainty * diversity_bonus
                uncertainty = uncertainty_scores[img_path]
                
                # Diversity bonus: penalize similarity to already selected
                diversity = self._calculate_diversity(img_path, selected)
                
                combined_score = uncertainty * (1.0 + diversity)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_path = img_path
            
            if best_path:
                selected.append(best_path)
                remaining.remove(best_path)
        
        self._log_selection(selected, 'diversity')
        return selected
    
    def _balanced_sampling(self, pool: List[str], k: int) -> List[str]:
        # Group by predicted classes
        class_samples = defaultdict(list)
        
        for img_path in pool:
            img_data = self.data_manager.get_labels(img_path)
            if not img_data or not img_data.get('detections'):
                class_samples['unknown'].append((img_path, 0.5))
                continue
            
            # Get dominant class and entropy
            detections = img_data['detections']
            max_conf_det = max(detections, key=lambda d: d.get('confidence', 0))
            cls = max_conf_det.get('class', 'unknown')
            entropy = img_data.get('entropy', 0.5)
            
            class_samples[cls].append((img_path, entropy))
        
        # Sort each class by entropy
        for cls in class_samples:
            class_samples[cls].sort(key=lambda x: x[1], reverse=True)
        
        # Round-robin selection across classes
        selected = []
        class_list = list(class_samples.keys())
        idx = 0
        
        while len(selected) < k:
            # Try current class
            cls = class_list[idx % len(class_list)]
            if class_samples[cls]:
                path, _ = class_samples[cls].pop(0)
                selected.append(path)
            
            idx += 1
            
            # Check if we've exhausted all classes
            if all(len(samples) == 0 for samples in class_samples.values()):
                break
        
        self._log_selection(selected, 'balanced')
        return selected
    
    def _calculate_diversity(self, candidate: str, selected: List[str]) -> float:
        if not selected:
            return 1.0
        
        # Get class distribution of candidate
        candidate_data = self.data_manager.get_labels(candidate)
        if not candidate_data or not candidate_data.get('detections'):
            return 0.5
        
        candidate_classes = set()
        for det in candidate_data['detections']:
            candidate_classes.add(det.get('class', 'unknown'))
        
        # Calculate average Jaccard distance to selected samples
        distances = []
        for sel_path in selected:
            sel_data = self.data_manager.get_labels(sel_path)
            if not sel_data or not sel_data.get('detections'):
                continue
            
            sel_classes = set()
            for det in sel_data['detections']:
                sel_classes.add(det.get('class', 'unknown'))
            
            # Jaccard distance = 1 - Jaccard similarity
            intersection = len(candidate_classes & sel_classes)
            union = len(candidate_classes | sel_classes)
            
            if union > 0:
                jaccard_sim = intersection / union
                jaccard_dist = 1.0 - jaccard_sim
                distances.append(jaccard_dist)
        
        return np.mean(distances) if distances else 0.5
    
    def _log_selection(self, selected: List[str], strategy: str):
        """Log selection for analysis."""
        self.selection_history.append({
            'timestamp': np.datetime64('now'),
            'strategy': strategy,
            'count': len(selected),
            'paths': selected
        })
    
    def get_priority_queue(self, unlabeled_pool: List[str]) -> List[Tuple[str, float]]:
        scores = []
        for img_path in unlabeled_pool:
            img_data = self.data_manager.get_labels(img_path)
            entropy = img_data.get('entropy', 0.5) if img_data else 0.5
            scores.append((img_path, entropy))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_stats(self) -> Dict:
        """Get selection statistics."""
        if not self.selection_history:
            return {'total_selections': 0}
        
        total = sum(h['count'] for h in self.selection_history)
        by_strategy = defaultdict(int)
        for h in self.selection_history:
            by_strategy[h['strategy']] += h['count']
        
        return {
            'total_selections': total,
            'selection_rounds': len(self.selection_history),
            'by_strategy': dict(by_strategy)
        }