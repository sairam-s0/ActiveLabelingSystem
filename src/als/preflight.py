"""Standard-library preflight checks used by the ALS bootstrap batch file."""

from __future__ import annotations

import importlib.util
import sys

MIN_PYTHON = (3, 10)
MODULE_SPECS = [
    ("numpy", "numpy", True),
    ("torch", "torch", True),
    ("torchvision", "torchvision", False),
    ("ultralytics", "ultralytics", True),
    ("cv2", "opencv-python (cv2)", True),
    ("PIL", "Pillow (PIL)", True),
    ("PyQt6", "PyQt6", True),
    ("yaml", "PyYAML (yaml)", True),
    ("pandas", "pandas", False),
    ("tqdm", "tqdm", False),
    ("ray", "ray", False),
]


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def run_preflight() -> int:
    fail_count = 0
    warn_count = 0

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print("[INFO] ALS preflight checks starting...")
    print(f"[PASS] Python detected: {version}")

    if sys.version_info >= MIN_PYTHON:
        print("[PASS] Python version requirement satisfied.")
    else:
        print("[FAIL] Python 3.10+ is required.")
        fail_count += 1

    for module_name, label, required in MODULE_SPECS:
        if module_available(module_name):
            print(f"[PASS] {label}")
            continue

        if required:
            print(f"[FAIL] Missing required module: {label}")
            fail_count += 1
        else:
            print(f"[WARNING] Missing optional module: {label}")
            warn_count += 1

    try:
        import torch  # type: ignore

        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_version:
            print(f"[PASS] CUDA-enabled torch detected (CUDA version: {cuda_version}).")
        else:
            print("[WARNING] CUDA-enabled torch is unavailable. ALS may run in CPU mode.")
            warn_count += 1
    except Exception:
        print("[WARNING] torch could not be inspected for CUDA support.")
        warn_count += 1

    if fail_count:
        print(f"[ERROR] Preflight failed with {fail_count} major issue(s).")
        return 1

    if warn_count:
        print(f"[PASS] Preflight passed with {warn_count} warning(s).")
        return 0

    print("[PASS] Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_preflight())
