"""Tests for scope-aligned evidence reporting."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings
from src.evaluation.evidence_reporting import (
    build_scope_aligned_evidence_report,
    resolve_scope_evidence_paths,
    save_scope_aligned_evidence_report,
)


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for evidence-report tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    config_root = project_root / "configs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "backend_serving.yaml").write_text(
        "active_oasis_model_registry: outputs/model_registry/oasis_current_baseline.json\n",
        encoding="utf-8",
    )
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path / "kaggle",
        oasis_source_root=tmp_path / "OASIS",
        serving_config_path=config_root / "backend_serving.yaml",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_resolve_scope_evidence_paths_finds_latest_kaggle_run(tmp_path: Path) -> None:
    """The resolver should locate the newest Kaggle final metrics artifact."""

    settings = _build_settings(tmp_path)
    _write_json(settings.outputs_root / "model_registry" / "oasis_current_baseline.json", {"model_id": "x"})
    older = settings.outputs_root / "runs" / "kaggle" / "older_run" / "metrics" / "final_metrics.json"
    newer = settings.outputs_root / "runs" / "kaggle" / "newer_run" / "metrics" / "final_metrics.json"
    _write_json(older, {"run_name": "older_run"})
    _write_json(newer, {"run_name": "newer_run"})

    paths = resolve_scope_evidence_paths(settings)

    assert paths.kaggle_final_metrics_path == newer
    assert paths.oasis_registry_path == settings.outputs_root / "model_registry" / "oasis_current_baseline.json"


def test_resolve_scope_evidence_paths_prefers_repeated_split_study_with_real_sample_support(tmp_path: Path) -> None:
    """Repeated-split selection should prefer stronger held-out sample support over tiny smoke runs."""

    settings = _build_settings(tmp_path)
    _write_json(settings.outputs_root / "model_registry" / "oasis_current_baseline.json", {"model_id": "x"})
    smoke_path = settings.outputs_root / "model_selection" / "validation_depth_smoke" / "study_summary.json"
    full_path = settings.outputs_root / "model_selection" / "oasis_repeated_splits_gpu_full_seed42" / "study_summary.json"
    _write_json(
        smoke_path,
        {
            "split_seeds": [101, 102],
            "runs": [{}, {}, {}, {}, {}, {}],
            "aggregate_metrics": [
                {"split": "test", "metric_name": "sample_count", "mean": 2.0},
                {"split": "test", "metric_name": "auroc", "mean": 0.9},
            ],
        },
    )
    _write_json(
        full_path,
        {
            "split_seeds": [42, 43, 44],
            "runs": [{}, {}, {}],
            "aggregate_metrics": [
                {"split": "test", "metric_name": "sample_count", "mean": 36.0},
                {"split": "test", "metric_name": "auroc", "mean": 0.88},
            ],
        },
    )

    paths = resolve_scope_evidence_paths(settings)

    assert paths.oasis_repeated_split_study_path == full_path


def test_build_scope_aligned_evidence_report_preserves_dataset_roles(tmp_path: Path) -> None:
    """The report should keep OASIS primary and Kaggle secondary."""

    settings = _build_settings(tmp_path)
    oasis_registry_path = settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    repeated_split_path = settings.outputs_root / "model_selection" / "oasis_repeat" / "study_summary.json"
    kaggle_metrics_path = settings.outputs_root / "runs" / "kaggle" / "unit_run" / "metrics" / "final_metrics.json"
    _write_json(
        oasis_registry_path,
        {
            "dataset": "oasis1",
            "run_name": "oasis_active",
            "model_id": "oasis_current_baseline",
            "recommended_threshold": 0.45,
            "promotion_decision": {"approved": True},
            "operational_status": "active",
            "validation_metrics": {"sample_count": 35, "accuracy": 0.74, "auroc": 0.77, "sensitivity": 0.73, "specificity": 0.75, "f1": 0.71, "review_required_count": 11},
            "test_metrics": {"sample_count": 36, "accuracy": 0.86, "auroc": 0.88, "sensitivity": 0.93, "specificity": 0.81, "f1": 0.85, "review_required_count": 10},
            "review_monitoring": {"total_reviews": 2, "override_rate": 0.5, "high_risk": False, "risk_signals": []},
            "threshold_calibration": {"selection_metric": "balanced_accuracy", "threshold": 0.45},
            "notes": ["OASIS-only."],
        },
    )
    _write_json(
        repeated_split_path,
        {
            "study_name": "oasis_repeat",
            "runs": [{"experiment_name": "best", "run_name": "oasis_active"}],
            "best_experiment_name": "best",
            "selection_metric": "auroc",
            "best_selection_score": 0.79,
            "seeds": [42],
            "split_seeds": [42, 43, 44],
            "aggregate_metrics": [
                {"split": "test", "metric_name": "auroc", "mean": 0.88, "std": 0.01, "ci95_low": 0.86, "ci95_high": 0.89},
                {"split": "test", "metric_name": "accuracy", "mean": 0.85, "std": 0.01, "ci95_low": 0.84, "ci95_high": 0.87},
                {"split": "test", "metric_name": "sensitivity", "mean": 0.89, "std": 0.06, "ci95_low": 0.82, "ci95_high": 0.96},
                {"split": "test", "metric_name": "specificity", "mean": 0.82, "std": 0.02, "ci95_low": 0.80, "ci95_high": 0.85},
                {"split": "test", "metric_name": "f1", "mean": 0.83, "std": 0.02, "ci95_low": 0.81, "ci95_high": 0.86},
                {"split": "test", "metric_name": "review_required_count", "mean": 12.0, "std": 5.0, "ci95_low": 6.0, "ci95_high": 18.0},
            ],
            "notes": ["Repeated split support."],
        },
    )
    _write_json(
        kaggle_metrics_path,
        {
            "run_name": "kaggle_unit",
            "dataset": "kaggle_alz",
            "dataset_type": "2d_slices",
            "class_names": ["A", "B", "C", "D"],
            "validation": {"sample_count": 960, "accuracy": 0.7, "balanced_accuracy": 0.55, "macro_f1": 0.5, "macro_ovr_auroc": 0.8, "loss": 1.0},
            "test": {"sample_count": 960, "accuracy": 0.71, "balanced_accuracy": 0.56, "macro_f1": 0.51, "macro_ovr_auroc": 0.81, "loss": 0.98},
            "warnings": ["Kaggle-only evidence."],
        },
    )
    _write_json(
        settings.data_root / "interim" / "kaggle_alz_split_summary.json",
        {
            "train_rows": 100,
            "val_rows": 20,
            "test_rows": 20,
            "subset_distribution_by_split": {"train": {"AugmentedAlzheimerDataset": 80}},
            "warnings": ["Augmented rows stay in train."],
        },
    )
    _write_json(
        settings.data_root / "interim" / "kaggle_alz_manifest_summary.json",
        {
            "dataset_type": "2d_slices",
            "manifest_row_count": 140,
            "organization": "class_folders",
            "warnings": ["Slice-based dataset."],
        },
    )
    _write_json(
        settings.outputs_root / "reports" / "readiness" / "training_device_profile.json",
        {
            "recommended_device": "cuda",
            "cuda_available": True,
            "cuda_device_name": "RTX",
            "total_memory_gb": 8.0,
            "available_memory_gb": 2.0,
            "warnings": ["Low RAM."],
            "recommendations": ["Use the GPU carefully."],
        },
    )

    report = build_scope_aligned_evidence_report(
        settings,
        oasis_registry_path=oasis_registry_path,
        oasis_repeated_split_study_path=repeated_split_path,
        kaggle_final_metrics_path=kaggle_metrics_path,
    )

    assert report["scope_alignment"]["primary_dataset_role"] == "oasis_primary_3d"
    assert report["scope_alignment"]["secondary_dataset_role"] == "kaggle_secondary_2d_comparison"
    assert report["oasis_primary"]["role"] == "primary_evidence_track"
    assert report["kaggle_secondary"]["role"] == "secondary_comparison_branch"
    assert report["kaggle_secondary"]["dataset_type"] == "2d_slices"
    assert any("Keep OASIS as the primary evidence track" in item for item in report["recommendations"])


def test_save_scope_aligned_evidence_report_writes_json_and_markdown(tmp_path: Path) -> None:
    """The scope-aligned evidence report should save both JSON and Markdown."""

    settings = _build_settings(tmp_path)
    report = {
        "goal_statement": "scope",
        "oasis_primary": {"run_name": "oasis", "approval_status": "approved", "operational_status": "active", "test_metrics": {"accuracy": 0.8, "auroc": 0.9, "sensitivity": 0.85, "specificity": 0.75, "f1": 0.8}},
        "oasis_repeated_splits": {"available": False, "study_name": None, "run_count": None, "split_seed_count": None, "test_aggregate": {"auroc": {"mean": None}, "accuracy": {"mean": None}}},
        "kaggle_secondary": {"available": False, "dataset_type": None, "run_name": None, "test_metrics": {"accuracy": None, "balanced_accuracy": None, "macro_f1": None, "macro_ovr_auroc": None}},
        "comparison_notes": ["note"],
        "recommendations": ["rec"],
    }

    json_path, md_path = save_scope_aligned_evidence_report(report, settings, file_stem="unit_scope_report")

    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["goal_statement"] == "scope"
    assert "Scope-Aligned Evidence Report" in md_path.read_text(encoding="utf-8")
