@echo off
setlocal
cd /d "%~dp0alz_backend"
".venv\Scripts\python.exe" scripts\build_external_manifest.py %*
