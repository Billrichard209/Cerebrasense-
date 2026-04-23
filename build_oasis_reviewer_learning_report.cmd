@echo off
setlocal
set "ROOT=%~dp0alz_backend"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\build_oasis_reviewer_learning_report.py" %*
