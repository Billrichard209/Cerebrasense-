@echo off
setlocal
cd /d "%~dp0alz_backend"
".venv\Scripts\python.exe" scripts\explain_scan.py %*
