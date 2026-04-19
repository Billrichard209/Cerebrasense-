@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\train_oasis2.py %*
