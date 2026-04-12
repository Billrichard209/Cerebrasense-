@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%alz_backend\.venv\Scripts\python.exe" "%ROOT%scripts\run_oasis_volumetrics.py" %*
