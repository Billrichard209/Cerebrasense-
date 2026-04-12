"""Tests for Colab workflow assets."""

from __future__ import annotations

import json
from pathlib import Path


def test_colab_notebook_exists_and_has_expected_cells() -> None:
    """The Colab notebook should exist and include the key training steps."""

    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "oasis_colab_training.ipynb"
    assert notebook_path.exists()

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell_sources = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "drive.mount('/content/drive')" in cell_sources
    assert "train_oasis_colab.py" in cell_sources
    assert "calibrate_oasis_threshold.py" in cell_sources
    assert "promote_oasis_checkpoint.py" in cell_sources
