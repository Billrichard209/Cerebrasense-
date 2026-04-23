@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%scripts\batch_predict_oasis_scans.py" %*
