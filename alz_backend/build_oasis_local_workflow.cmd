@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%scripts\build_oasis_local_workflow.py" %*
