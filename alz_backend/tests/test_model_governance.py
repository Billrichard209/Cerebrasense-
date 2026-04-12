"""Tests for benchmark registration and checkpoint promotion gates."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings
from src.models.governance import (
    calibrate_active_oasis_model,
    evaluate_oasis_promotion_candidate,
    register_benchmark,
    run_oasis_promotion_workflow,
)
from src.models.registry import save_oasis_model_entry, ModelRegistryEntry


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
        storage_root=project_root / "storage",
        database_path=project_root / "storage" / "backend.sqlite3",
    )


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "image,label,subject_id",
                "scan1.nii.gz,0,OAS1_0001",
                "scan2.nii.gz,1,OAS1_0002",
                "scan3.nii.gz,1,OAS1_0003",
            ]
        ),
        encoding="utf-8",
    )


def test_register_benchmark_writes_hash_and_subject_safe_summary(tmp_path: Path) -> None:
    """Benchmark registration should write a hashed manifest summary."""

    settings = _settings(tmp_path)
    manifest_path = tmp_path / "test_manifest.csv"
    _write_manifest(manifest_path)

    entry, output_path = register_benchmark(
        manifest_path=manifest_path,
        benchmark_name="unit_test_benchmark",
        settings=settings,
    )

    assert output_path.exists()
    assert entry.sample_count == 3
    assert entry.subject_safe is True
    assert entry.subject_count == 3
    assert entry.label_distribution["1"] == 2


def test_evaluate_oasis_promotion_candidate_requires_explicit_thresholds(tmp_path: Path) -> None:
    """Promotion decisions should fail when metrics miss the configured floor."""

    manifest_path = tmp_path / "test_manifest.csv"
    _write_manifest(manifest_path)
    benchmark_entry, _ = register_benchmark(
        manifest_path=manifest_path,
        benchmark_name="unit_test_benchmark",
        settings=_settings(tmp_path),
    )

    decision = evaluate_oasis_promotion_candidate(
        run_name="unit_run",
        benchmark_entry=benchmark_entry,
        validation_metrics={"auroc": 0.80},
        test_metrics={
            "sample_count": 36,
            "auroc": 0.70,
            "sensitivity": 0.90,
            "review_required_count": 4,
            "mean_calibrated_confidence": 0.70,
        },
    )

    assert decision.approved is False
    assert "test_auroc" in decision.failed_checks


def test_run_oasis_promotion_workflow_updates_active_registry_when_candidate_passes(tmp_path: Path) -> None:
    """A passing candidate should write a benchmarked active registry entry."""

    settings = _settings(tmp_path)
    manifest_path = tmp_path / "test_manifest.csv"
    _write_manifest(manifest_path)
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    val_metrics_path = tmp_path / "val_metrics.json"
    test_metrics_path = tmp_path / "test_metrics.json"
    val_metrics_path.write_text(json.dumps({"auroc": 0.80}), encoding="utf-8")
    test_metrics_path.write_text(
        json.dumps(
            {
                "sample_count": 36,
                "auroc": 0.84,
                "sensitivity": 0.88,
                "review_required_count": 8,
                "mean_calibrated_confidence": 0.70,
            }
        ),
        encoding="utf-8",
    )

    result = run_oasis_promotion_workflow(
        run_name="unit_run",
        checkpoint_path=checkpoint_path,
        validation_metrics_path=val_metrics_path,
        test_metrics_path=test_metrics_path,
        manifest_path=manifest_path,
        benchmark_name="unit_test_benchmark",
        settings=settings,
    )

    assert result.decision.approved is True
    assert result.active_registry_path is not None
    payload = json.loads(result.active_registry_path.read_text(encoding="utf-8"))
    assert payload["benchmark"]["benchmark_name"] == "unit_test_benchmark"
    assert payload["promotion_decision"]["approved"] is True


def test_calibrate_active_oasis_model_updates_registry_threshold(tmp_path: Path) -> None:
    """Active-model calibration should persist the selected threshold into the registry."""

    settings = _settings(tmp_path)
    settings.outputs_root.mkdir(parents=True, exist_ok=True)
    val_dir = tmp_path / "val_eval"
    test_dir = tmp_path / "test_eval"
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    val_predictions = val_dir / "predictions.csv"
    test_predictions = test_dir / "predictions.csv"
    val_predictions.write_text(
        "\n".join(
            [
                "true_label,probability_class_1",
                "0,0.20",
                "0,0.49",
                "1,0.51",
                "1,0.90",
            ]
        ),
        encoding="utf-8",
    )
    test_predictions.write_text(
        "\n".join(
            [
                "true_label,probability_class_1",
                "0,0.30",
                "1,0.80",
            ]
        ),
        encoding="utf-8",
    )
    registry_entry = ModelRegistryEntry(
        registry_version="1.0",
        model_id="oasis_current_baseline",
        dataset="oasis1",
        run_name="unit_run",
        checkpoint_path=str((tmp_path / "checkpoints" / "best_model.pt").relative_to(settings.workspace_root)),
        model_config_path=None,
        preprocessing_config_path=None,
        image_size=[64, 64, 64],
        promoted_at_utc="2026-01-01T00:00:00+00:00",
        decision_support_only=True,
        clinical_disclaimer="Decision-support only.",
        evidence={
            "validation_metrics_path": str((val_dir / "metrics.json").relative_to(settings.workspace_root)),
            "test_metrics_path": str((test_dir / "metrics.json").relative_to(settings.workspace_root)),
        },
    )
    (val_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (test_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoints").mkdir(exist_ok=True)
    (tmp_path / "checkpoints" / "best_model.pt").write_bytes(b"checkpoint")
    save_oasis_model_entry(registry_entry, settings=settings)

    result = calibrate_active_oasis_model(settings=settings, selection_metric="f1", threshold_step=0.05)

    assert result.registry_entry.recommended_threshold >= 0.0
    assert result.registry_entry.threshold_calibration["selection_metric"] == "f1"
    assert "threshold_calibration_path" in result.registry_entry.evidence
