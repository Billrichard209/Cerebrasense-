@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\build_structural_metrics_report.py %*
