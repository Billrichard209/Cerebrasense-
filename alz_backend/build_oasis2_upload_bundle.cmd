@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%scripts\build_oasis2_upload_bundle.py" %*
