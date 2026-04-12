@echo off
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" scripts\build_dataset_inventory.py
