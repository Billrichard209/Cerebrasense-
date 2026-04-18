@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\import_promoted_oasis_run.py %*
