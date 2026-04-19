"""Tests for OASIS productization/alignment checks."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings

from scripts.check_oasis_productization import build_oasis_productization_report, save_oasis_productization_report


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    config_root = project_root / "configs"
    outputs_root = project_root / "outputs"
    data_root = project_root / "data"
    config_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    (config_root / "backend_serving.yaml").write_text(
        "active_oasis_model_registry: outputs/model_registry/oasis_current_baseline.json\n",
        encoding="utf-8",
    )
    (config_root / "oasis_model.yaml").write_text("architecture: densenet121_3d\n", encoding="utf-8")
    (config_root / "oasis_transforms.yaml").write_text("spatial:\n  spatial_size: [64, 64, 64]\n", encoding="utf-8")
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path,
        oasis_source_root=tmp_path / "OASIS",
        serving_config_path=config_root / "backend_serving.yaml",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_active_registry(settings: AppSettings, *, run_name: str = "oasis_colab_full_v3_auroc_monitor") -> None:
    checkpoint_path = settings.outputs_root / "runs" / "oasis" / run_name / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    _write_json(
        settings.outputs_root / "model_registry" / "oasis_current_baseline.json",
        {
            "registry_version": "1.0",
            "model_id": "oasis_current_baseline",
            "dataset": "oasis1",
            "run_name": run_name,
            "checkpoint_path": str(checkpoint_path),
            "model_config_path": str(settings.project_root / "configs" / "oasis_model.yaml"),
            "preprocessing_config_path": str(settings.project_root / "configs" / "oasis_transforms.yaml"),
            "image_size": [64, 64, 64],
            "promoted_at_utc": "2026-01-01T00:00:00+00:00",
            "decision_support_only": True,
            "clinical_disclaimer": "Decision-support only, not a diagnosis. Use clinical judgment.",
            "recommended_threshold": 0.45,
            "default_threshold": 0.5,
        },
    )


def test_productization_report_passes_when_local_and_evidence_align(tmp_path: Path) -> None:
    """Aligned local registry and evidence artifacts should pass the productization check."""

    settings = _settings(tmp_path)
    _seed_active_registry(settings)
    _write_json(
        settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.json",
        {
            "oasis_primary": {
                "run_name": "oasis_colab_full_v3_auroc_monitor",
                "recommended_threshold": 0.45,
            }
        },
    )
    _write_json(
        settings.outputs_root / "model_selection" / "oasis_repeated_splits_gpu_full_seed42" / "study_summary.json",
        {
            "study_name": "oasis_repeated_splits_gpu_full_seed42",
            "runs": [{"run_name": "oasis_colab_full_v3_auroc_monitor"}],
            "split_seeds": [42, 43, 44],
            "aggregate_metrics": [{"split": "test", "metric_name": "sample_count", "mean": 36.0}],
        },
    )

    report = build_oasis_productization_report(settings, expected_run_name="oasis_colab_full_v3_auroc_monitor")

    assert report.overall_status == "pass"
    assert any(check.name == "scope_evidence_alignment" and check.status == "pass" for check in report.checks)


def test_productization_report_fails_when_scope_evidence_drifts(tmp_path: Path) -> None:
    """Evidence drift should fail the productization check."""

    settings = _settings(tmp_path)
    _seed_active_registry(settings, run_name="oasis_colab_full_v3_auroc_monitor")
    _write_json(
        settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.json",
        {
            "oasis_primary": {
                "run_name": "oasis_baseline_rtx2050_gpu_seed42_split42",
                "recommended_threshold": 0.45,
            }
        },
    )

    report = build_oasis_productization_report(settings, expected_run_name="oasis_colab_full_v3_auroc_monitor")

    assert report.overall_status == "fail"
    assert any(check.name == "scope_evidence_alignment" and check.status == "fail" for check in report.checks)


def test_save_productization_report_writes_json_and_markdown(tmp_path: Path) -> None:
    """Productization reports should save JSON and Markdown artifacts."""

    settings = _settings(tmp_path)
    _seed_active_registry(settings)
    report = build_oasis_productization_report(settings, expected_run_name="oasis_colab_full_v3_auroc_monitor")

    json_path, md_path = save_oasis_productization_report(report, settings)

    assert json_path.exists()
    assert md_path.exists()
    assert "OASIS Productization Status" in md_path.read_text(encoding="utf-8")
