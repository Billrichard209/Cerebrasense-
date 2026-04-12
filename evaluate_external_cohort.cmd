@echo off
setlocal
cd /d "%~dp0alz_backend"
".venv\Scripts\python.exe" scripts\evaluate_external_cohort.py %*
