"""Tests for runtime path overrides."""

from __future__ import annotations

from pathlib import Path

from src.configs.runtime import AppSettings


def test_runtime_env_can_override_data_and_outputs_roots(monkeypatch, tmp_path: Path) -> None:
    """Explicit env overrides should take precedence over project defaults."""

    data_root = tmp_path / "drive_runtime" / "data"
    outputs_root = tmp_path / "drive_runtime" / "outputs"
    monkeypatch.setenv("ALZ_DATA_ROOT", str(data_root))
    monkeypatch.setenv("ALZ_OUTPUTS_ROOT", str(outputs_root))

    settings = AppSettings.from_env()

    assert settings.data_root == data_root.resolve()
    assert settings.outputs_root == outputs_root.resolve()
