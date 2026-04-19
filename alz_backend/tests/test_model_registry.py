"""Tests for promoted model registry entries."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings
from src.models.registry import import_promoted_oasis_run, load_current_oasis_model_entry, promote_oasis_checkpoint


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


def test_import_promoted_oasis_run_copies_run_and_rewrites_registry_paths(tmp_path: Path) -> None:
    """Imported Colab artifacts should be copied locally and point the registry at local paths."""

    settings = _settings(tmp_path)
    source_run_root = tmp_path / "drive_export" / "training_runs" / "oasis" / "colab_run"
    source_checkpoint = source_run_root / "checkpoints" / "best_model.pt"
    source_checkpoint.parent.mkdir(parents=True)
    source_checkpoint.write_bytes(b"checkpoint")
    source_registry_path = tmp_path / "drive_export" / "model_registry" / "oasis_current_baseline.json"
    source_registry_path.parent.mkdir(parents=True)
    source_registry_path.write_text(
        json.dumps(
            {
                "registry_version": "1.0",
                "model_id": "oasis_current_baseline",
                "dataset": "oasis1",
                "run_name": "colab_run",
                "checkpoint_path": "/content/Cerebrasense-/alz_backend/outputs/runs/oasis/colab_run/checkpoints/best_model.pt",
                "model_config_path": "/content/Cerebrasense-/alz_backend/configs/oasis_model.yaml",
                "preprocessing_config_path": "/content/Cerebrasense-/alz_backend/configs/oasis_transforms.yaml",
                "image_size": [64, 64, 64],
                "promoted_at_utc": "2026-01-01T00:00:00+00:00",
                "decision_support_only": True,
                "clinical_disclaimer": "Decision-support only, not a diagnosis.",
                "recommended_threshold": 0.42,
                "default_threshold": 0.5,
                "threshold_calibration": {"threshold": 0.42, "selection_metric": "f1"},
            }
        ),
        encoding="utf-8",
    )

    local_model_config = settings.project_root / "configs" / "oasis_model.yaml"
    local_model_config.parent.mkdir(parents=True)
    local_model_config.write_text("dataset: oasis1\n", encoding="utf-8")
    local_preprocessing = settings.project_root / "configs" / "oasis_transforms.yaml"
    local_preprocessing.write_text("spatial:\n  spatial_size: [64, 64, 64]\n", encoding="utf-8")

    result = import_promoted_oasis_run(
        source_run_root=source_run_root,
        source_registry_path=source_registry_path,
        settings=settings,
    )

    assert result.local_run_root.exists()
    assert result.local_checkpoint_path.exists()
    assert result.local_registry_path is not None and result.local_registry_path.exists()

    loaded = load_current_oasis_model_entry(path=result.local_registry_path, settings=settings)
    assert loaded.run_name == "colab_run"
    assert loaded.checkpoint_path == str(result.local_checkpoint_path)
    assert loaded.model_config_path == str(local_model_config)
    assert loaded.preprocessing_config_path == str(local_preprocessing)
    assert loaded.recommended_threshold == 0.42


def test_import_promoted_oasis_run_supports_canonical_runtime_root(tmp_path: Path) -> None:
    """Canonical backend_runtime imports should resolve run and registry paths automatically."""

    settings = _settings(tmp_path)
    runtime_root = tmp_path / "drive_sync" / "backend_runtime"
    source_run_root = runtime_root / "outputs" / "runs" / "oasis" / "oasis_colab_full_v3_auroc_monitor"
    source_checkpoint = source_run_root / "checkpoints" / "best_model.pt"
    source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_checkpoint.write_bytes(b"checkpoint")

    source_registry_path = runtime_root / "outputs" / "model_registry" / "oasis_current_baseline.json"
    source_registry_path.parent.mkdir(parents=True, exist_ok=True)
    source_registry_path.write_text(
        json.dumps(
            {
                "registry_version": "1.0",
                "model_id": "oasis_current_baseline",
                "dataset": "oasis1",
                "run_name": "oasis_colab_full_v3_auroc_monitor",
                "checkpoint_path": "/content/drive/MyDrive/Cerebrasensecloud/backend_runtime/outputs/runs/oasis/oasis_colab_full_v3_auroc_monitor/checkpoints/best_model.pt",
                "model_config_path": "/content/Cerebrasense-/alz_backend/configs/oasis_model.yaml",
                "preprocessing_config_path": "/content/Cerebrasense-/alz_backend/configs/oasis_transforms.yaml",
                "image_size": [64, 64, 64],
                "promoted_at_utc": "2026-01-01T00:00:00+00:00",
                "decision_support_only": True,
                "clinical_disclaimer": "Decision-support only, not a diagnosis.",
                "recommended_threshold": 0.45,
                "default_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    local_model_config = settings.project_root / "configs" / "oasis_model.yaml"
    local_model_config.parent.mkdir(parents=True, exist_ok=True)
    local_model_config.write_text("dataset: oasis1\n", encoding="utf-8")
    local_preprocessing = settings.project_root / "configs" / "oasis_transforms.yaml"
    local_preprocessing.write_text("spatial:\n  spatial_size: [64, 64, 64]\n", encoding="utf-8")

    result = import_promoted_oasis_run(
        source_runtime_root=runtime_root,
        settings=settings,
    )

    assert result.run_name == "oasis_colab_full_v3_auroc_monitor"
    assert result.local_run_root.exists()
    assert result.local_checkpoint_path.exists()
    assert result.local_registry_path is not None and result.local_registry_path.exists()


def test_import_promoted_oasis_run_runtime_root_missing_fails_clearly(tmp_path: Path) -> None:
    """Missing backend_runtime paths should raise a direct runtime-root error."""

    settings = _settings(tmp_path)
    missing_runtime_root = tmp_path / "missing_drive_sync" / "backend_runtime"

    try:
        import_promoted_oasis_run(
            source_runtime_root=missing_runtime_root,
            settings=settings,
        )
    except FileNotFoundError as exc:
        assert "Source runtime root not found" in str(exc)
    else:
        raise AssertionError("Expected missing runtime root to raise FileNotFoundError")


def test_load_current_oasis_model_entry_resolves_workspace_relative_paths(tmp_path: Path) -> None:
    """Legacy registry entries using workspace-relative `alz_backend/...` paths should load cleanly."""

    settings = _settings(tmp_path)
    checkpoint_path = settings.outputs_root / "runs" / "oasis" / "unit_run" / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")

    model_config_path = settings.project_root / "configs" / "oasis_model.yaml"
    model_config_path.parent.mkdir(parents=True, exist_ok=True)
    model_config_path.write_text("architecture: densenet121_3d\n", encoding="utf-8")

    preprocessing_config_path = settings.project_root / "configs" / "oasis_transforms.yaml"
    preprocessing_config_path.write_text("spatial:\n  spatial_size: [64, 64, 64]\n", encoding="utf-8")

    registry_path = settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "registry_version": "1.0",
                "model_id": "oasis_current_baseline",
                "dataset": "oasis1",
                "run_name": "unit_run",
                "checkpoint_path": "alz_backend/outputs/runs/oasis/unit_run/checkpoints/best_model.pt",
                "model_config_path": "alz_backend/configs/oasis_model.yaml",
                "preprocessing_config_path": "alz_backend/configs/oasis_transforms.yaml",
                "image_size": [64, 64, 64],
                "promoted_at_utc": "2026-01-01T00:00:00+00:00",
                "decision_support_only": True,
                "clinical_disclaimer": "Decision-support only, not a diagnosis.",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_current_oasis_model_entry(path=registry_path, settings=settings)

    assert loaded.checkpoint_path == str(checkpoint_path.resolve())
    assert loaded.model_config_path == str(model_config_path.resolve())
    assert loaded.preprocessing_config_path == str(preprocessing_config_path.resolve())
