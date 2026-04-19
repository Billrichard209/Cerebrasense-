@echo off
setlocal
set "ROOT=%~dp0alz_backend"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\check_oasis2_adapter.py" %*
