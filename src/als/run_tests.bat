@echo off
setlocal EnableExtensions

set "PY_CMD=%ALS_PYTHON%"
if not defined PY_CMD set "PY_CMD=python"

cd /d "%~dp0"
"%PY_CMD%" "%~dp0preflight.py"
exit /b %errorlevel%
