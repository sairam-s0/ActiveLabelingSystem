# src/core/config.py
import os
from pathlib import Path
import yaml

# Base directory of the project
ROOT_DIR = Path(__file__).resolve().parents[2]

# Default values
DEFAULTS = {
    "qa_rate": 0.01,
    "threshold": 70,
    "max_dim": 1600,
    "weights": "yolov8m.pt",
    "min_training_samples": 30,
    "max_wait_hours": 24,
    "perf_delta_threshold": 0.05,
    "entropy_shift_threshold": 0.15,
    "entropy_threshold": 0.6
}

class Config:
    def __init__(self):
        self.config_data = dict(DEFAULTS)
        self.load_from_file()
        self.load_from_env()

    def load_from_file(self):
        config_paths = [
            ROOT_DIR / "config.yaml",
            ROOT_DIR / "config.json"
        ]
        for path in config_paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        if path.suffix == ".yaml" or path.suffix == ".yml":
                            loaded = yaml.safe_load(f) or {}
                        else:
                            import json
                            loaded = json.load(f) or {}
                        # Update config with loaded values
                        for k, v in loaded.items():
                            if k in self.config_data:
                                self.config_data[k] = type(DEFAULTS[k])(v)
                    print(f"[Config] Loaded configurations from {path.name}")
                    break
                except Exception as e:
                    print(f"[Config] Error loading {path.name}: {e}")

    def load_from_env(self):
        for k, default_val in DEFAULTS.items():
            env_key = f"ALS_{k.upper()}"
            env_val = os.getenv(env_key)
            if env_val is not None:
                try:
                    self.config_data[k] = type(default_val)(env_val)
                    print(f"[Config] Override {k} from environment variable {env_key}")
                except Exception as e:
                    print(f"[Config] Error parsing env var {env_key}: {e}")

    def __getattr__(self, name):
        if name in self.config_data:
            return self.config_data[name]
        raise AttributeError(f"Config object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "config_data":
            super().__setattr__(name, value)
        else:
            self.config_data[name] = value

# Global config instance
config = Config()
