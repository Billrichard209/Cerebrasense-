@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\inspect_datasets.py %*
