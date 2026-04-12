@echo off
setlocal
cd /d "%~dp0alz_backend"
".venv\Scripts\python.exe" scripts\train_oasis.py %*
