"""Tests for promoted model registry entries."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings
from src.models.registry import load_current_oasis_model_entry, promote_oasis_checkpoint


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=project_root / "data",
        outputs_root=project_root / "outputs",
        oasis_source_root=project_root.parent / "OASIS",
        kaggle_source_root=project_root.parent / "archive (1)",
    )


def test_promote_oasis_checkpoint_writes_decision_support_registry_entry(tmp_path: Path) -> None:
    """Promoted OASIS checkpoints should preserve metrics and safety wording."""

    settings = _settings(tmp_path)
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    val_metrics_path = tmp_path / "val_metrics.json"
    test_metrics_path = tmp_path / "test_metrics.json"
    val_metrics_path.write_text(json.dumps({"accuracy": 0.7}), encoding="utf-8")
    test_metrics_path.write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")
    calibration_path = tmp_path / "threshold_calibration.json"
    calibration_path.write_text(json.dumps({"threshold": 0.3, "selection_metric": "f1"}), encoding="utf-8")

    entry, output_path = promote_oasis_checkpoint(
        run_name="unit_run",
        checkpoint_path=checkpoint_path,
        val_metrics_path=val_metrics_path,
        test_metrics_path=test_metrics_path,
        threshold_calibration_path=calibration_path,
        image_size=(64, 64, 64),
        settings=settings,
    )

    assert output_path.exists()
    assert entry.dataset == "oasis1"
    assert entry.decision_support_only is True
    assert "not a diagnosis" in entry.clinical_disclaimer.lower()
    assert entry.validation_metrics["accuracy"] == 0.7
    assert entry.test_metrics["accuracy"] == 0.8
    assert entry.default_threshold == 0.5
    assert entry.recommended_threshold == 0.3
    assert entry.threshold_calibration["selection_metric"] == "f1"
    assert entry.temperature_scaling["temperature"] == 1.0
    assert "high_confidence_min" in entry.confidence_policy
    assert entry.operational_status == "active"
    assert entry.serving_restrictions["force_manual_review"] is False

    loaded = load_current_oasis_model_entry(settings=settings)
    assert loaded.run_name == "unit_run"
    assert loaded.recommended_threshold == 0.3
    assert loaded.operational_status == "active"
