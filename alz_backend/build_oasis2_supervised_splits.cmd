@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\build_oasis2_supervised_splits.py %*
