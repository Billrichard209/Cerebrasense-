"""Tests for the Colab training wrapper helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_colab_script_module():
    """Load the Colab wrapper script as a module from its file path."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_oasis_colab.py"
    spec = importlib.util.spec_from_file_location("train_oasis_colab", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_running_in_colab_returns_bool() -> None:
    """Colab detection should return a simple boolean."""

    module = _load_colab_script_module()
    assert isinstance(module._running_in_colab(), bool)


def test_colab_config_exists() -> None:
    """The Colab GPU config should be present in the repository."""

    config_path = Path(__file__).resolve().parents[1] / "configs" / "oasis_train_colab_gpu.yaml"
    assert config_path.exists()
