"""Tests for the active-vs-candidate OASIS baseline comparison report."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings

from scripts import build_oasis_baseline_comparison as comparison_module


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


def _write_registry(
    path: Path,
    *,
    settings: AppSettings,
    run_name: str,
    recommended_threshold: float,
    test_auroc: float,
    threshold_test_f1: float,
    review_required_count: int,
) -> None:
    checkpoint_path = settings.outputs_root / "runs" / "oasis" / run_name / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    _write_json(
        path,
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
            "recommended_threshold": recommended_threshold,
            "default_threshold": 0.5,
            "validation_metrics": {
                "sample_count": 35,
                "accuracy": 0.74,
                "auroc": 0.80 if "candidate" in run_name else 0.77,
                "review_required_count": 10 if "candidate" not in run_name else 22,
            },
            "test_metrics": {
                "sample_count": 36,
                "accuracy": 0.83 if "candidate" in run_name else 0.86,
                "auroc": test_auroc,
                "review_required_count": review_required_count,
            },
            "threshold_calibration": {
                "selection_metric": "f1",
                "threshold": recommended_threshold,
                "test_metrics": {
                    "sample_count": 36,
                    "f1": threshold_test_f1,
                },
            },
        },
    )


def test_build_baseline_comparison_prefers_current_active_when_candidate_trails(tmp_path: Path, monkeypatch) -> None:
    """A weaker candidate should keep the active local baseline in place."""

    settings = _settings(tmp_path)
    active_registry_path = settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    candidate_registry_path = settings.outputs_root / "model_registry" / "oasis_candidate_v3.json"
    _write_registry(
        active_registry_path,
        settings=settings,
        run_name="oasis_baseline_rtx2050_gpu_seed42_split42",
        recommended_threshold=0.45,
        test_auroc=0.8793650793650795,
        threshold_test_f1=0.8484848484848485,
        review_required_count=10,
    )
    _write_registry(
        candidate_registry_path,
        settings=settings,
        run_name="oasis_colab_full_v3_auroc_monitor",
        recommended_threshold=0.41,
        test_auroc=0.8634920634920635,
        threshold_test_f1=0.8108108108108109,
        review_required_count=23,
    )

    monkeypatch.setattr(
        comparison_module,
        "build_oasis_demo_bundle",
        lambda **kwargs: {"bundle_root": f"demo::{Path(kwargs['registry_path']).stem}"},
    )

    report = comparison_module.build_oasis_baseline_comparison_report(
        settings=settings,
        active_registry_path=active_registry_path,
        candidate_registry_path=candidate_registry_path,
        build_demo_bundles=True,
    )

    assert report.recommendation["action"] == "keep_active"
    assert report.delta["test_auroc"] is not None and report.delta["test_auroc"] < 0
    assert report.demo_bundles["active"]["bundle_root"] == "demo::oasis_current_baseline"
    assert report.demo_bundles["candidate"]["bundle_root"] == "demo::oasis_candidate_v3"

    json_path, md_path = comparison_module.save_oasis_baseline_comparison_report(report, settings)

    assert json_path.exists()
    assert md_path.exists()
    assert "OASIS Baseline Comparison" in md_path.read_text(encoding="utf-8")
