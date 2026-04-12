@echo off
setlocal
cd /d "%~dp0alz_backend"
".venv\Scripts\python.exe" scripts\check_backend_readiness.py %*
