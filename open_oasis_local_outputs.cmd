@echo off
setlocal
set "ROOT=%~dp0alz_backend"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\open_oasis_local_outputs.py" %*
