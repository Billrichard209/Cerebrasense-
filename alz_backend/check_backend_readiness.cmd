@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\check_backend_readiness.py %*
