@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM run_tests.bat - LabelOps preflight checker (PASS/FAIL/WARNING only)
REM Runs from this script directory so relative paths are stable.
cd /d "%~dp0"

set "FAIL_COUNT=0"
set "WARN_COUNT=0"

echo [INFO] LabelOps preflight checks starting...

where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Python not found in PATH. Install Python 3.10+ and retry.
    set /a FAIL_COUNT+=1
    goto :summary
)

for /f "tokens=1,2 delims= " %%A in ('python --version 2^>^&1') do set "PY_VER=%%B"
echo [PASS] Python detected: !PY_VER!

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo [FAIL] Python 3.10+ is required.
    set /a FAIL_COUNT+=1
) else (
    echo [PASS] Python version requirement satisfied.
)

call :check_module numpy "numpy" required
call :check_module torch "torch" required
call :check_module ultralytics "ultralytics" required
call :check_module cv2 "opencv-python (cv2)" required
call :check_module PIL "Pillow (PIL)" required
call :check_module PyQt6 "PyQt6" required
call :check_module yaml "PyYAML (yaml)" required

call :check_module pandas "pandas" optional
call :check_module tqdm "tqdm" optional
call :check_module ray "ray" optional

python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo [WARNING] GPU/CUDA is unavailable. CPU mode will be used.
    set /a WARN_COUNT+=1
) else (
    for /f %%V in ('python -c "import torch; print(torch.version.cuda or \"unknown\")"') do set "CUDA_VER=%%V"
    echo [PASS] CUDA is available ^(torch CUDA version: !CUDA_VER!^).
)

:summary
if !FAIL_COUNT! GTR 0 (
    echo [ERROR] Preflight failed with !FAIL_COUNT! major issue^(s^).
    echo [ERROR] Resolve failures, then run: pip install -r requirements.txt
    exit /b 1
)

if !WARN_COUNT! GTR 0 (
    echo [PASS] Preflight passed with !WARN_COUNT! warning^(s^).
    exit /b 0
)

echo [PASS] Preflight passed.
exit /b 0

:check_module
set "MOD_NAME=%~1"
set "LABEL=%~2"
set "MODE=%~3"

python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('!MOD_NAME!') else 1)"
if errorlevel 1 (
    if /I "!MODE!"=="required" (
        echo [FAIL] Missing required module: !LABEL!
        set /a FAIL_COUNT+=1
    ) else (
        echo [WARNING] Missing optional module: !LABEL!
        set /a WARN_COUNT+=1
    )
) else (
    echo [PASS] !LABEL!
)
exit /b 0
