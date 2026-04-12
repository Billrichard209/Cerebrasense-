@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%.venv\Scripts\python.exe" "%ROOT%scripts\run_longitudinal_volumetrics.py" %*
