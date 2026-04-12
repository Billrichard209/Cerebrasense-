@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%alz_backend\.venv\Scripts\python.exe" "%ROOT%scripts\check_splits.py" %*
