# src/core/training_orchestrator.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import os
os.environ['RAY_SCHEDULER_EVENTS'] = '0'
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'
os.environ.setdefault('RAY_USAGE_STATS_ENABLED', '0')

import ray
from typing import Optional, Dict, List, Callable
import threading
import time

def start_orchestrator_monitor(orchestrator, callback):
    def loop():
        while True:
            try:
                # check completion
                result = orchestrator.check_training_completion()
                if result:
                    callback({'type': 'completion', 'data': result})
                
                # check status
                status = orchestrator.get_training_status()
                callback({'type': 'status', 'data': status})
                
            except Exception as e:
                print(f"[Monitor] Error: {e}")
            
            time.sleep(1.0)
    
    t = threading.Thread(target=loop, daemon=True)
    t.start()

class TrainingOrchestrator:
    
    def __init__(
        self,
        data_manager,
        model_manager,
        replay_buffer,
        min_samples: int = 30,
        num_gpus: int = 1
    ):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.replay_buffer = replay_buffer
        self.min_samples = min_samples
        self.num_gpus = num_gpus
        
        # training state
        self.shadow_trainer = None
        self.training_future = None
        self.ray_initialized = False
        
        # callbacks for
        self.on_status_change: Optional[Callable] = None
        self.on_training_complete: Optional[Callable] = None
        self.on_training_failed: Optional[Callable] = None
        
        # initialize ray
        # self init
    
    def _init_ray(self) -> bool:
        try:
            if ray.is_initialized():
                print("[Orchestrator] Ray already initialized")
                self.ray_initialized = True
                return True
            
            ray.init(
                num_gpus=self.num_gpus,
                num_cpus=2,
                ignore_reinit_error=True,
                logging_level="ERROR",
                include_dashboard=False,
                _metrics_export_port=0,
            )
            print("[Orchestrator] Ray initialized successfully")
            self.ray_initialized = True
            
            # try to
            self._try_create_trainer()
            return True
            
        except ImportError:
            print("[Orchestrator] Ray not installed - background training disabled")
            self.ray_initialized = False
            return False
        except Exception as e:
            print(f"[Orchestrator] Ray init failed: {e}")
            self.ray_initialized = False
            return False
    def initialize_ray(self):
        if self.ray_initialized:
            return True
        return self._init_ray()
    
    def _try_create_trainer(self):
        if not self.ray_initialized:
            return False
        
        class_mapping = self.data_manager.data.get('class_mapping', {})
        if not class_mapping:
            print("[Orchestrator] No class mapping yet - waiting for first label")
            return False
        
        try:
            from core.shadow_trainer import ShadowTrainer
            
            base_model = self.model_manager.resolve_active_path()
            cluster = ray.cluster_resources() or {}
            available_gpus = float(cluster.get("GPU", 0.0) or 0.0)
            actor_gpus = 0.4 if (self.num_gpus and available_gpus > 0) else 0.0
            self.shadow_trainer = ShadowTrainer.options(num_gpus=actor_gpus).remote(
                base_model_path=base_model,
                class_mapping=class_mapping,
                min_samples=self.min_samples
            )
            print(
                f"[Orchestrator] Shadow trainer created with {len(class_mapping)} classes "
                f"(num_gpus={actor_gpus})"
            )
            return True
            
        except Exception as e:
            print(f"[Orchestrator] Failed to create shadow trainer: {e}")
            return False
    
    def check_training_trigger(self) -> bool:
        if not self.ray_initialized:
            return False
        
        # ensure trainer
        if not self.shadow_trainer:
            self._try_create_trainer()
            if not self.shadow_trainer:
                return False
        
        # check if
        if self.is_training():
            print("[Orchestrator] Training already in progress")
            return False
        
        # check queue
        stats = self.data_manager.get_stats()
        queue_size = stats['training_queue_size']
        
        if queue_size >= self.min_samples:
            print(f"[Orchestrator] Queue full ({queue_size}/{self.min_samples}) - triggering training")
            return self.trigger_training()
        
        return False
    
    def is_training(self) -> bool:
        if not self.training_future:
            return False
        
        try:
            ready, _ = ray.wait([self.training_future], timeout=0)
            return len(ready) == 0  # not ready
        except:
            return False
    
    def trigger_training(self) -> bool:
        if not self.shadow_trainer:
            print("[Orchestrator] Shadow trainer not available")
            return False
        
        if self.is_training():
            print("[Orchestrator] Training already in progress")
            return False
        
        try:
            # get training
            samples = self.data_manager.get_training_batch(
                count=self.min_samples,
                new_only=True,
                return_full_samples=True
            )
            
            if len(samples) < self.min_samples:
                print(f"[Orchestrator] Not enough samples: {len(samples)}/{self.min_samples}")
                return False
            
            # get replay
            replay_paths = self.data_manager.get_replay_samples(count=10)
            replay_samples = self.data_manager.prepare_training_samples(replay_paths)
            
            # fill trainer
            print(f"[Orchestrator] Starting training: {len(samples)} new + {len(replay_samples)} replay")
            add_result = ray.get(self.shadow_trainer.add_labels.remote(samples), timeout=10)
            if not add_result.get('ready_to_train'):
                print(
                    "[Orchestrator] Trainer not ready after add_labels: "
                    f"{add_result.get('buffer_size', 0)}/{self.min_samples}"
                )
                return False
            self.training_future = self.shadow_trainer.train.remote(replay_samples)
            
            # notify ui
            if self.on_status_change:
                self.on_status_change({
                    'status': 'training_started',
                    'sample_count': len(samples),
                    'replay_count': len(replay_samples)
                })
            
            return True
            
        except Exception as e:
            print(f"[Orchestrator] Error triggering training: {e}")
            if self.on_training_failed:
                self.on_training_failed({'error': str(e)})
            return False
    
    def get_training_status(self) -> Dict:
        if not self.shadow_trainer:
            return {
                'available': False,
                'training': False,
                'reason': 'trainer_not_initialized'
            }
        
        try:
            status_future = self.shadow_trainer.get_training_progress.remote()
            status = ray.get(status_future, timeout=1)
            status['available'] = True
            return status
            
        except ray.exceptions.GetTimeoutError:
            return {
                'available': True,
                'training': False,
                'reason': 'status_timeout'
            }
        except Exception as e:
            return {
                'available': False,
                'training': False,
                'error': str(e)
            }
    
    def check_training_completion(self) -> Optional[Dict]:
        if not self.training_future:
            return None
        
        try:
            ready, _ = ray.wait([self.training_future], timeout=0)
            
            if not ready:
                return None  # still training
            
            # training completed
            result = ray.get(self.training_future)
            self.training_future = None
            
            if result['success']:
                self._handle_training_success(result)
            else:
                self._handle_training_failure(result)
            
            return result
            
        except Exception as e:
            print(f"[Orchestrator] Error checking completion: {e}")
            return None
    
    def _handle_training_success(self, result: Dict):
        print(f"[Orchestrator] Training completed successfully!")
        print(f"[Orchestrator] Trained on {result['sample_count']} samples")
        print(f"[Orchestrator] Model saved to: {result['save_path']}")
        
        # mark samples
        trained_paths = result.get('trained_paths', [])
        self.data_manager.mark_trained(trained_paths)
        
        # add to
        replay_samples = self.data_manager.prepare_training_samples(trained_paths)
        self.replay_buffer.add(replay_samples)
        
        # notify ui
        if self.on_training_complete:
            self.on_training_complete(result)
    
    def _handle_training_failure(self, result: Dict):
        error = result.get('error', 'Unknown error')
        print(f"[Orchestrator] Training failed: {error}")
        
        # notify ui
        if self.on_training_failed:
            self.on_training_failed(result)
    
    def promote_shadow_model(self, validate: bool = True) -> Dict:
        shadow_path = "models/shadow_candidate.pt"
        
        if not Path(shadow_path).exists():
            return {
                'success': False,
                'error': 'No shadow model found. Train a model first.'
            }
        
        # optional validation
        if validate:
            comparison = self.model_manager.compare_models(
                base_path=self.model_manager.resolve_active_path(),
                shadow_path=shadow_path
            )
            
            if not comparison.get('recommend_promote', False):
                return {
                    'success': False,
                    'error': comparison.get('error', 'Model validation failed'),
                    'requires_confirmation': True,
                    'comparison': comparison
                }
        
        # perform promotion
        result = self.model_manager.promote_shadow(shadow_path, validate=validate)
        
        if result['success']:
            print(f"[Orchestrator] Shadow model promoted: {result['version']}")
        
        return result
    
    def get_queue_status(self) -> Dict:
        stats = self.data_manager.get_stats()
        return {
            'queue_size': stats['training_queue_size'],
            'min_samples': self.min_samples,
            'ready_to_train': stats['training_queue_size'] >= self.min_samples,
            'progress_percent': min(100, (stats['training_queue_size'] / self.min_samples) * 100)
        }
    
    def shutdown(self):
        print("[Orchestrator] Shutting down...")
        
        # explicitly shutdown
        if ray.is_initialized():
            try:
                ray.shutdown()
                print("[Orchestrator] Ray shutdown complete")
            except Exception as e:
                print(f"[Orchestrator] Ray shutdown error: {e}")
        
        self.shadow_trainer = None
        self.training_future = None
        print("[Orchestrator] Shutdown complete")
