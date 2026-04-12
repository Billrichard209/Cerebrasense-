param(
    [int]$Port = 8000
)

Set-Location (Join-Path $PSScriptRoot "..")
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $python -m uvicorn src.api.main:app --host 127.0.0.1 --port $Port --reload
