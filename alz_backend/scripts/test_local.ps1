Set-Location (Join-Path $PSScriptRoot "..")
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $python -m pytest tests -q
