@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\check_oasis2_training_readiness.py %*
