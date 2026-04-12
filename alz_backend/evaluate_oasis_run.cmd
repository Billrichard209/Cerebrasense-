@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\evaluate_oasis_run.py %*
