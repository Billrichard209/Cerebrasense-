@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\predict_scan.py %*
