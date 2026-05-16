"""Path and filesystem helpers for reproducible backend jobs."""

from __future__ import annotations

from pathlib import Path


import json
from datetime import datetime
from typing import Any

def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def resolve_project_root() -> Path:
    """Return the repository root that contains the backend source tree."""
    return Path(__file__).resolve().parents[2]

def log_structured_event(event_type: str, payload: dict[str, Any], log_dir: Path | None = None) -> None:
    """Append a structured JSON event to the daily audit log."""
    from src.configs.runtime import get_app_settings
    settings = get_app_settings()
    audit_root = ensure_directory(log_dir or settings.outputs_root / "logs" / "audit")
    log_path = audit_root / f"{datetime.now().strftime('%Y-%m-%d')}_audit.jsonl"
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "payload": payload
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
