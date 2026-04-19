@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%alz_backend\scripts\train_oasis2_colab.py" %*
