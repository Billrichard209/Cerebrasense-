"""Convenience wrapper to run the OASIS split checker from the parent workspace."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = WORKSPACE_ROOT / "alz_backend"
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "check_splits.py"
PYTHON_PATH = BACKEND_ROOT / ".venv" / "Scripts" / "python.exe"

if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [str(PYTHON_PATH), str(SCRIPT_PATH), *sys.argv[1:]],
            cwd=str(BACKEND_ROOT),
        )
    )
