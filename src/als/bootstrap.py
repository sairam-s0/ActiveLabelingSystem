"""Bootstrap the local ALS runtime before launching the GUI."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from als.preflight import MODULE_SPECS, module_available

MIN_PYTHON = (3, 10)
CPU_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cpu"
NVIDIA_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"


@dataclass
class GpuInfo:
    vendor: str = "unknown"
    devices: list[str] = field(default_factory=list)
    accelerator: str = "cpu"
    notes: list[str] = field(default_factory=list)


@dataclass
class BootstrapResult:
    ready: bool
    gpu: GpuInfo
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="als",
        description="Bootstrap the ALS runtime or launch the ALS GUI.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Run setup first, then start the GUI.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run checks without installing missing packages.",
    )
    args = parser.parse_args(argv)

    result = bootstrap_environment(auto_install=not args.check_only)
    if not result.ready:
        return 1

    if args.start:
        return start_application()

    print("[ALS] Environment is ready.")
    print("[ALS] Launch the app with: als --start")
    return 0


def bootstrap_environment(auto_install: bool) -> BootstrapResult:
    print("[ALS] Starting environment bootstrap...")

    if not python_version_supported():
        handle_python_version_problem(auto_install)
        return BootstrapResult(
            ready=False,
            gpu=GpuInfo(),
            notes=["Python 3.10 or newer is required."],
        )

    gpu = detect_gpu()
    print_gpu_summary(gpu)

    run_preflight_batch()
    missing_required, missing_optional = find_missing_modules()

    notes: list[str] = []
    if gpu.vendor == "amd":
        notes.append(
            "AMD GPU detected. ALS will use CPU mode because the current runtime only "
            "auto-configures NVIDIA CUDA acceleration."
        )
    elif gpu.vendor == "intel":
        notes.append(
            "Intel GPU detected. ALS will use CPU mode because CUDA acceleration is "
            "not available for this setup path."
        )

    if auto_install:
        try:
            if torch_install_needed(gpu):
                install_torch_runtime(gpu)
                missing_required, missing_optional = find_missing_modules()

            if missing_required or missing_optional:
                install_runtime_requirements()
                missing_required, missing_optional = find_missing_modules()

            run_preflight_batch()
        except subprocess.CalledProcessError as exc:
            notes.append(f"Dependency installation failed with exit code {exc.returncode}.")
            print(f"[ALS] Dependency installation failed with exit code {exc.returncode}.")
            return BootstrapResult(
                ready=False,
                gpu=gpu,
                missing_required=missing_required,
                missing_optional=missing_optional,
                notes=notes,
            )

    ready = not missing_required and python_version_supported()
    if ready:
        print("[ALS] Bootstrap finished successfully.")
    else:
        print("[ALS] Bootstrap is incomplete. Resolve the remaining issues and retry.")

    for note in notes:
        print(f"[ALS] {note}")

    return BootstrapResult(
        ready=ready,
        gpu=gpu,
        missing_required=missing_required,
        missing_optional=missing_optional,
        notes=notes,
    )


def python_version_supported() -> bool:
    return sys.version_info >= MIN_PYTHON


def handle_python_version_problem(auto_install: bool) -> None:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"[ALS] Python {version} detected, but ALS requires Python 3.10+.")
    if not auto_install:
        return
    if platform.system() == "Windows" and shutil.which("winget"):
        print("[ALS] Attempting to install Python 3.10 with winget...")
        command = [
            "winget",
            "install",
            "--exact",
            "--id",
            "Python.Python.3.10",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ]
        subprocess.run(command, check=False)
        print("[ALS] Re-run `als` after Python 3.10 installation completes.")
    else:
        print("[ALS] Install Python 3.10 or newer, then rerun `als`.")


def run_preflight_batch() -> None:
    if platform.system() != "Windows":
        print("[ALS] Skipping run_tests.bat because this machine is not Windows.")
        return

    batch_path = packaged_file("run_tests.bat")
    if not batch_path.exists():
        print("[ALS] Packaged run_tests.bat was not found; skipping batch preflight.")
        return

    print(f"[ALS] Running preflight batch: {batch_path.name}")
    env = dict(os.environ)
    env["ALS_PYTHON"] = sys.executable
    subprocess.run(
        ["cmd", "/c", str(batch_path)],
        check=False,
        cwd=str(batch_path.parent),
        env=env,
    )


def find_missing_modules() -> tuple[list[str], list[str]]:
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for module_name, package_name, required in MODULE_SPECS:
        if not module_available(module_name):
            if required:
                missing_required.append(package_name)
            else:
                missing_optional.append(package_name)
    return missing_required, missing_optional


def ensure_pip() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    print("[ALS] pip was not available. Bootstrapping pip with ensurepip...")
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)


def torch_install_needed(gpu: GpuInfo) -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return True

    if gpu.accelerator != "cuda":
        return False

    return not bool(getattr(torch.version, "cuda", None))


def install_torch_runtime(gpu: GpuInfo) -> None:
    ensure_pip()

    index_url = CPU_TORCH_INDEX_URL
    label = "CPU-only"
    if gpu.accelerator == "cuda":
        index_url = NVIDIA_TORCH_INDEX_URL
        label = "NVIDIA CUDA"

    print(f"[ALS] Installing PyTorch runtime for {label}...")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "--index-url",
        index_url,
    ]
    subprocess.run(command, check=True)


def install_runtime_requirements() -> None:
    ensure_pip()
    requirements_path = packaged_file("requirements.txt")
    if not requirements_path.exists():
        raise FileNotFoundError(f"Packaged requirements file not found: {requirements_path}")

    print(f"[ALS] Installing runtime requirements from {requirements_path.name}...")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "-r",
        str(requirements_path),
    ]
    subprocess.run(command, check=True)


def detect_gpu() -> GpuInfo:
    names: list[str] = []
    if platform.system() == "Windows":
        names.extend(read_command_output([
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ]))
        if not names:
            names.extend(read_command_output([
                "wmic",
                "path",
                "win32_VideoController",
                "get",
                "name",
            ]))
    elif platform.system() == "Linux":
        names.extend(read_command_output(["lspci"]))
    elif platform.system() == "Darwin":
        names.extend(read_command_output([
            "system_profiler",
            "SPDisplaysDataType",
        ]))

    nvidia_names = read_command_output([
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
    ])
    names.extend(nvidia_names)

    unique_names = dedupe_nonempty(names)
    joined = " ".join(unique_names).lower()

    if "nvidia" in joined:
        return GpuInfo(vendor="nvidia", devices=unique_names, accelerator="cuda")
    if "amd" in joined or "radeon" in joined:
        return GpuInfo(vendor="amd", devices=unique_names, accelerator="cpu")
    if "intel" in joined or "arc" in joined:
        return GpuInfo(vendor="intel", devices=unique_names, accelerator="cpu")
    if unique_names:
        return GpuInfo(vendor="unknown", devices=unique_names, accelerator="cpu")
    return GpuInfo(vendor="none", devices=[], accelerator="cpu")


def read_command_output(command: list[str]) -> list[str]:
    if not shutil.which(command[0]):
        return []

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    lines = [line.strip() for line in result.stdout.splitlines()]
    return [line for line in lines if line and line.lower() != "name"]


def dedupe_nonempty(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def print_gpu_summary(gpu: GpuInfo) -> None:
    if not gpu.devices:
        print("[ALS] No GPU detected. ALS will install CPU dependencies.")
        return

    print(f"[ALS] Detected GPU vendor: {gpu.vendor}")
    for device in gpu.devices:
        print(f"[ALS] GPU: {device}")
    if gpu.accelerator == "cuda":
        print("[ALS] NVIDIA GPU detected. ALS will install CUDA-enabled PyTorch.")
    else:
        print("[ALS] ALS will use CPU mode for this GPU configuration.")


def packaged_file(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def start_application() -> int:
    print("[ALS] Starting the GUI...")
    try:
        from main import main as app_main
    except Exception as exc:
        print(f"[ALS] Failed to import the GUI entry point: {exc}")
        return 1

    app_main()
    return 0
