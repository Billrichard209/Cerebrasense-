"""Tests for validation-depth and stability reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.api.services import build_validation_depth_payload, build_validation_studies_payload
from src.configs.runtime import AppSettings
from src.models.registry import ModelRegistryEntry, save_oasis_model_entry
from src.models.validation_depth import build_validation_depth_dashboard, load_validation_depth_studies


def _build_settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    storage_root = project_root / "storage"
    data_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)
    storage_root.mkdir(parents=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=project_root.parent,
        oasis_source_root=project_root.parent / "OASIS",
        storage_root=storage_root,
        database_path=storage_root / "backend.sqlite3",
    )


def _save_active_entry(settings: AppSettings) -> None:
    entry = ModelRegistryEntry(
        registry_version="1.0",
        model_id="oasis_current_baseline",
        dataset="oasis1",
        run_name="oasis_baseline_rtx2050_gpu_seed42_split42",
        checkpoint_path="checkpoint.pt",
        model_config_path=None,
        preprocessing_config_path=None,
        image_size=[64, 64, 64],
        promoted_at_utc="2026-01-01T00:00:00+00:00",
        decision_support_only=True,
        clinical_disclaimer="Decision-support only.",
        promotion_decision={"approved": True},
        validation_metrics={"auroc": 0.78},
        test_metrics={"auroc": 0.82},
    )
    save_oasis_model_entry(entry, settings=settings)


def _write_study_summary(study_root: Path, payload: dict[str, object]) -> None:
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "study_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_validation_depth_studies_distinguishes_fixed_split_and_repeated_split(
    tmp_path: Path,
) -> None:
    """Saved studies should classify repeated splits conservatively and keep smoke runs honest."""

    settings = _build_settings(tmp_path)
    _save_active_entry(settings)

    _write_study_summary(
        settings.outputs_root / "model_selection" / "fixed_split_family",
        {
            "study_name": "fixed_split_family",
            "seeds": [42, 43, 44],
            "split_seeds": [42],
            "selection_split": "val",
            "selection_metric": "auroc",
            "best_experiment_name": "fixed_seed42",
            "best_selection_score": 0.81,
            "runs": [
                {"experiment_name": "fixed_seed42", "run_name": "oasis_baseline_rtx2050_gpu_seed42_split42"},
                {"experiment_name": "fixed_seed43", "run_name": "oasis_baseline_rtx2050_gpu_seed43_split42"},
                {"experiment_name": "fixed_seed44", "run_name": "oasis_baseline_rtx2050_gpu_seed44_split42"},
            ],
            "aggregate_metrics": [
                {"split": "val", "metric_name": "auroc", "mean": 0.82, "std": 0.03, "values": [0.79, 0.82, 0.85]},
                {"split": "test", "metric_name": "auroc", "mean": 0.77, "std": 0.04, "values": [0.74, 0.77, 0.80]},
                {
                    "split": "test",
                    "metric_name": "sensitivity",
                    "mean": 0.79,
                    "std": 0.05,
                    "values": [0.74, 0.79, 0.84],
                },
                {
                    "split": "test",
                    "metric_name": "specificity",
                    "mean": 0.72,
                    "std": 0.04,
                    "values": [0.68, 0.72, 0.76],
                },
                {"split": "test", "metric_name": "sample_count", "mean": 36.0, "std": 0.0, "values": [36, 36, 36]},
                {
                    "split": "test",
                    "metric_name": "review_required_count",
                    "mean": 12.0,
                    "std": 2.0,
                    "values": [10, 12, 14],
                },
            ],
            "notes": ["Fixed split seed sweep for the active OASIS family."],
        },
    )
    _write_study_summary(
        settings.outputs_root / "model_selection" / "repeated_split_smoke",
        {
            "study_name": "repeated_split_smoke",
            "seeds": [42, 43, 44],
            "split_seeds": [101, 202],
            "selection_split": "val",
            "selection_metric": "auroc",
            "best_experiment_name": "smoke_seed42_split101",
            "best_selection_score": 0.0,
            "runs": [
                {"experiment_name": "smoke_seed42_split101", "run_name": "oasis_baseline_rtx2050_gpu_seed42_split101"},
                {"experiment_name": "smoke_seed43_split101", "run_name": "oasis_baseline_rtx2050_gpu_seed43_split101"},
                {"experiment_name": "smoke_seed44_split101", "run_name": "oasis_baseline_rtx2050_gpu_seed44_split101"},
                {"experiment_name": "smoke_seed42_split202", "run_name": "oasis_baseline_rtx2050_gpu_seed42_split202"},
                {"experiment_name": "smoke_seed43_split202", "run_name": "oasis_baseline_rtx2050_gpu_seed43_split202"},
                {"experiment_name": "smoke_seed44_split202", "run_name": "oasis_baseline_rtx2050_gpu_seed44_split202"},
            ],
            "aggregate_metrics": [
                {"split": "val", "metric_name": "auroc", "mean": 0.0, "std": 0.0, "values": [0, 0, 0, 0, 0, 0]},
                {
                    "split": "test",
                    "metric_name": "auroc",
                    "mean": 0.67,
                    "std": 0.47,
                    "values": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                },
                {
                    "split": "test",
                    "metric_name": "sensitivity",
                    "mean": 0.67,
                    "std": 0.47,
                    "values": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                },
                {
                    "split": "test",
                    "metric_name": "specificity",
                    "mean": 0.33,
                    "std": 0.47,
                    "values": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                },
                {"split": "test", "metric_name": "sample_count", "mean": 2.0, "std": 0.0, "values": [2, 2, 2, 2, 2, 2]},
                {
                    "split": "test",
                    "metric_name": "review_required_count",
                    "mean": 2.0,
                    "std": 0.0,
                    "values": [2, 2, 2, 2, 2, 2],
                },
            ],
            "notes": ["Smoke repeated-split run with tiny held-out samples."],
        },
    )

    studies = load_validation_depth_studies(limit=10, settings=settings)

    assert len(studies) == 2
    assert studies[0].study_name == "fixed_split_family"
    assert studies[0].evaluation_type == "multi_seed_fixed_split"
    assert studies[0].active_run_included is True
    assert studies[0].validation_depth_level == "moderate"
    assert studies[0].stability_status == "moderate"
    assert studies[1].study_name == "repeated_split_smoke"
    assert studies[1].evaluation_type == "repeated_subject_safe_splits"
    assert studies[1].repeated_split is True
    assert studies[1].active_family_included is True
    assert studies[1].validation_depth_level == "insufficient"
    assert studies[1].stability_status == "insufficient"
    assert any("Held-out sample count is too small" in warning for warning in studies[1].warnings)


def test_validation_depth_dashboard_and_service_payloads_summarize_family_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dashboard payloads should summarize validation depth without overstating smoke evidence."""

    settings = _build_settings(tmp_path)
    _save_active_entry(settings)
    _write_study_summary(
        settings.outputs_root / "model_selection" / "fixed_split_family",
        {
            "study_name": "fixed_split_family",
            "seeds": [42, 43, 44],
            "split_seeds": [42],
            "selection_split": "val",
            "selection_metric": "auroc",
            "best_experiment_name": "fixed_seed42",
            "best_selection_score": 0.81,
            "runs": [
                {"experiment_name": "fixed_seed42", "run_name": "oasis_baseline_rtx2050_gpu_seed42_split42"},
                {"experiment_name": "fixed_seed43", "run_name": "oasis_baseline_rtx2050_gpu_seed43_split42"},
                {"experiment_name": "fixed_seed44", "run_name": "oasis_baseline_rtx2050_gpu_seed44_split42"},
            ],
            "aggregate_metrics": [
                {"split": "val", "metric_name": "auroc", "mean": 0.82, "std": 0.03, "values": [0.79, 0.82, 0.85]},
                {"split": "test", "metric_name": "auroc", "mean": 0.77, "std": 0.04, "values": [0.74, 0.77, 0.80]},
                {
                    "split": "test",
                    "metric_name": "sensitivity",
                    "mean": 0.79,
                    "std": 0.05,
                    "values": [0.74, 0.79, 0.84],
                },
                {"split": "test", "metric_name": "sample_count", "mean": 36.0, "std": 0.0, "values": [36, 36, 36]},
            ],
        },
    )
    _write_study_summary(
        settings.outputs_root / "model_selection" / "repeated_split_smoke",
        {
            "study_name": "repeated_split_smoke",
            "seeds": [42, 43, 44],
            "split_seeds": [101, 202],
            "selection_split": "val",
            "selection_metric": "auroc",
            "best_experiment_name": "smoke_seed42_split101",
            "best_selection_score": 0.0,
            "runs": [
                {"experiment_name": "smoke_seed42_split101", "run_name": "oasis_baseline_rtx2050_gpu_seed42_split101"},
                {"experiment_name": "smoke_seed43_split101", "run_name": "oasis_baseline_rtx2050_gpu_seed43_split101"},
                {"experiment_name": "smoke_seed44_split101", "run_name": "oasis_baseline_rtx2050_gpu_seed44_split101"},
                {"experiment_name": "smoke_seed42_split202", "run_name": "oasis_baseline_rtx2050_gpu_seed42_split202"},
                {"experiment_name": "smoke_seed43_split202", "run_name": "oasis_baseline_rtx2050_gpu_seed43_split202"},
                {"experiment_name": "smoke_seed44_split202", "run_name": "oasis_baseline_rtx2050_gpu_seed44_split202"},
            ],
            "aggregate_metrics": [
                {"split": "val", "metric_name": "auroc", "mean": 0.0, "std": 0.0, "values": [0, 0, 0, 0, 0, 0]},
                {"split": "test", "metric_name": "auroc", "mean": 0.67, "std": 0.47, "values": [1, 1, 0, 1, 1, 0]},
                {"split": "test", "metric_name": "sensitivity", "mean": 0.67, "std": 0.47, "values": [1, 0, 1, 1, 0, 1]},
                {"split": "test", "metric_name": "sample_count", "mean": 2.0, "std": 0.0, "values": [2, 2, 2, 2, 2, 2]},
            ],
        },
    )
    monkeypatch.setattr("src.api.services.get_app_settings", lambda: settings)

    dashboard = build_validation_depth_dashboard(limit=10, settings=settings)
    studies_payload = build_validation_studies_payload(limit=10)
    depth_payload = build_validation_depth_payload(limit=10)

    assert dashboard.overall_validation_depth == "moderate"
    assert dashboard.direct_active_run_studies == 1
    assert dashboard.related_family_studies == 2
    assert dashboard.repeated_split_family_studies == 1
    assert "repeat subject-safe studies" in dashboard.recommended_action.lower()
    assert studies_payload["total"] == 2
    assert studies_payload["items"][0]["study_name"] == "fixed_split_family"
    assert depth_payload["summary"]["overall_validation_depth"] == "moderate"
    assert depth_payload["summary"]["strongest_study_name"] == "fixed_split_family"
    assert depth_payload["studies"][1]["validation_depth_level"] == "insufficient"
