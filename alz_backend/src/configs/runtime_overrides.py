"""Helpers for temporarily overriding runtime storage settings in scripts/tests."""

from __future__ import annotations

from contextlib import contextmanager
from os import environ
from pathlib import Path
from typing import Iterator

from .runtime import get_app_settings


def apply_runtime_database_override(database_path: str | Path | None) -> Path | None:
    """Override the configured SQLite path for the current process."""

    if database_path is None:
        return None
    resolved_path = Path(database_path).expanduser().resolve()
    environ["ALZ_DATABASE_PATH"] = str(resolved_path)
    get_app_settings.cache_clear()
    return resolved_path


@contextmanager
def temporary_runtime_database_override(database_path: str | Path | None) -> Iterator[Path | None]:
    """Temporarily point the backend runtime at a different SQLite database."""

    previous_value = environ.get("ALZ_DATABASE_PATH")
    resolved_path = None if database_path is None else Path(database_path).expanduser().resolve()
    if resolved_path is not None:
        environ["ALZ_DATABASE_PATH"] = str(resolved_path)
    get_app_settings.cache_clear()
    try:
        yield resolved_path
    finally:
        if previous_value is None:
            environ.pop("ALZ_DATABASE_PATH", None)
        else:
            environ["ALZ_DATABASE_PATH"] = previous_value
        get_app_settings.cache_clear()
