"""Path and filesystem helpers for reproducible backend jobs."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path object."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_project_root() -> Path:
    """Return the repository root that contains the backend source tree."""

    return Path(__file__).resolve().parents[2]
