"""Tests for the OASIS-first longitudinal focus report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.evaluation.oasis_longitudinal_focus import (
    build_oasis_longitudinal_focus_report,
    resolve_oasis_longitudinal_focus_paths,
    save_oasis_longitudinal_focus_report,
)


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for longitudinal-focus tests."""

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


def test_resolve_oasis_longitudinal_focus_paths_finds_split_artifacts(tmp_path: Path) -> None:
    """The path resolver should follow registry-linked manifest files back to split artifacts."""

    settings = _build_settings(tmp_path)
    split_root = settings.outputs_root / "reports" / "oasis_loaders_seed42_split42_train70_val15_test15"
    manifest_path = split_root / "oasis_test_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("subject_id,image,label\n", encoding="utf-8")
    (split_root / "oasis_split_summary.json").write_text("{}", encoding="utf-8")
    (split_root / "oasis_longitudinal_subject_summary.csv").write_text(
        "subject_id,split,label,label_name,subject_session_count,is_longitudinal_subject\n",
        encoding="utf-8",
    )
    _write_json(
        settings.outputs_root / "model_registry" / "oasis_current_baseline.json",
        {
            "benchmark": {
                "manifest_path": "alz_backend/outputs/reports/oasis_loaders_seed42_split42_train70_val15_test15/oasis_test_manifest.csv",
            }
        },
    )

    paths = resolve_oasis_longitudinal_focus_paths(settings)

    assert paths.oasis_split_summary_path == split_root / "oasis_split_summary.json"
    assert paths.oasis_longitudinal_subject_summary_path == split_root / "oasis_longitudinal_subject_summary.csv"


def test_build_oasis_longitudinal_focus_report_marks_longitudinal_evidence_limited(tmp_path: Path) -> None:
    """The report should honestly reflect when current OASIS-1 coverage lacks repeated subjects."""

    settings = _build_settings(tmp_path)
    split_root = settings.outputs_root / "reports" / "oasis_loaders_seed42_split42_train70_val15_test15"
    manifest_path = split_root / "oasis_test_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("subject_id,image,label\n", encoding="utf-8")
    _write_json(
        split_root / "oasis_split_summary.json",
        {
            "subject_counts": {"train": 10, "val": 3, "test": 3},
            "row_counts": {"train": 10, "val": 3, "test": 3},
            "longitudinal": {
                "subjects_with_multiple_sessions": 0,
                "subjects_with_single_session": 16,
                "max_sessions_per_subject": 1,
                "visit_order_sources": {"session_id_pattern": 16},
                "session_id_sources": {"meta.session_id": 16},
            },
        },
    )
    pd.DataFrame(
        [
            {
                "subject_id": "OAS1_0001",
                "split": "train",
                "label": 0,
                "label_name": "nondemented",
                "subject_session_count": 1,
                "is_longitudinal_subject": False,
                "first_session_id": "OAS1_0001_MR1",
                "last_session_id": "OAS1_0001_MR1",
                "first_scan_timestamp": "",
                "last_scan_timestamp": "",
                "visit_order_source": "session_id_pattern",
            },
            {
                "subject_id": "OAS1_0002",
                "split": "test",
                "label": 1,
                "label_name": "demented",
                "subject_session_count": 1,
                "is_longitudinal_subject": False,
                "first_session_id": "OAS1_0002_MR1",
                "last_session_id": "OAS1_0002_MR1",
                "first_scan_timestamp": "",
                "last_scan_timestamp": "",
                "visit_order_source": "session_id_pattern",
            },
        ]
    ).to_csv(split_root / "oasis_longitudinal_subject_summary.csv", index=False)
    _write_json(
        settings.outputs_root / "model_selection" / "oasis_repeated_splits_gpu_full_seed42" / "study_summary.json",
        {
            "study_name": "oasis_repeated_splits_gpu_full_seed42",
            "selection_metric": "auroc",
            "runs": [{"run_name": "oasis_active"}],
            "split_seeds": [42, 43, 44],
            "aggregate_metrics": [
                {"split": "test", "metric_name": "accuracy", "mean": 0.85, "std": 0.01, "ci95_low": 0.84, "ci95_high": 0.86},
                {"split": "test", "metric_name": "auroc", "mean": 0.88, "std": 0.01, "ci95_low": 0.87, "ci95_high": 0.89},
                {"split": "test", "metric_name": "sensitivity", "mean": 0.89, "std": 0.02, "ci95_low": 0.87, "ci95_high": 0.92},
                {"split": "test", "metric_name": "specificity", "mean": 0.82, "std": 0.03, "ci95_low": 0.79, "ci95_high": 0.85},
                {"split": "test", "metric_name": "f1", "mean": 0.84, "std": 0.02, "ci95_low": 0.82, "ci95_high": 0.86},
                {"split": "test", "metric_name": "review_required_count", "mean": 10.0, "std": 1.0, "ci95_low": 9.0, "ci95_high": 11.0},
            ],
        },
    )
    _write_json(
        settings.outputs_root / "model_registry" / "oasis_current_baseline.json",
        {
            "dataset": "oasis1",
            "run_name": "oasis_active",
            "benchmark": {
                "subject_safe": True,
                "subject_count": 16,
                "sample_count": 16,
                "manifest_path": "alz_backend/outputs/reports/oasis_loaders_seed42_split42_train70_val15_test15/oasis_test_manifest.csv",
            },
            "validation_metrics": {
                "accuracy": 0.74,
                "auroc": 0.77,
                "sensitivity": 0.73,
                "specificity": 0.75,
                "f1": 0.71,
            },
            "test_metrics": {
                "accuracy": 0.86,
                "auroc": 0.88,
                "sensitivity": 0.93,
                "specificity": 0.81,
                "f1": 0.85,
                "review_required_count": 10,
                "sample_count": 36,
            },
            "notes": ["OASIS-only"],
        },
    )

    report = build_oasis_longitudinal_focus_report(settings)

    assert report["focus_assessment"]["classification_evidence_status"] == "strong_baseline_present"
    assert report["focus_assessment"]["longitudinal_evidence_status"] == "limited"
    assert report["focus_assessment"]["next_dataset_priority"] == "oasis2"
    assert report["current_oasis"]["subject_summary"]["multi_session_subject_count"] == 0
    assert report["oasis2_readiness"]["overall_status"] == "warn"
    assert any("Do not overclaim longitudinal trend evidence" in item for item in report["recommendations"])


def test_save_oasis_longitudinal_focus_report_writes_json_and_markdown(tmp_path: Path) -> None:
    """The report saver should write both JSON and Markdown artifacts."""

    settings = _build_settings(tmp_path)
    report = {
        "goal_statement": "oasis-longitudinal",
        "current_oasis": {
            "run_name": "oasis_active",
            "benchmark_subject_safe": True,
            "test_metrics": {"accuracy": 0.86, "auroc": 0.88, "sensitivity": 0.93},
            "subject_summary": {
                "subject_count": 16,
                "multi_session_subject_count": 0,
                "multi_session_fraction": 0.0,
                "max_sessions_per_subject": 1,
                "timestamp_coverage_fraction": 0.0,
            },
        },
        "repeated_splits": {
            "available": True,
            "study_name": "oasis_repeat",
            "test_aggregate": {"auroc": {"mean": 0.88}, "accuracy": {"mean": 0.85}},
        },
        "oasis2_readiness": {
            "overall_status": "warn",
            "source_root": "missing",
            "supported_volume_file_count": 0,
            "unique_subject_id_count": 0,
            "longitudinal_subject_count": 0,
        },
        "recommendations": ["keep oasis first"],
    }

    json_path, md_path = save_oasis_longitudinal_focus_report(
        report,
        settings,
        file_stem="unit_oasis_longitudinal_focus",
    )

    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["goal_statement"] == "oasis-longitudinal"
    assert "OASIS Longitudinal Focus Report" in md_path.read_text(encoding="utf-8")
