@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%scripts\check_oasis2_metadata_adapter.py" %*
