"""Tests for runtime override helpers used by review workflow demos."""

from __future__ import annotations

from pathlib import Path

from src.configs.runtime import get_app_settings
from src.configs.runtime_overrides import temporary_runtime_database_override


def test_temporary_runtime_database_override_swaps_cached_database_path(tmp_path: Path) -> None:
    """The runtime DB override should update and then restore cached settings."""

    get_app_settings.cache_clear()
    original_database_path = get_app_settings().database_path
    override_database_path = (tmp_path / "review_demo.sqlite3").resolve()

    with temporary_runtime_database_override(override_database_path) as resolved_path:
        assert resolved_path == override_database_path
        assert get_app_settings().database_path == override_database_path

    assert get_app_settings().database_path == original_database_path
