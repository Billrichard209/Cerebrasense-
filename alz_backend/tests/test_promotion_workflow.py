"""Tests for read-only promotion workflow helpers and dashboard payloads."""

from __future__ import annotations

import json
from pathlib import Path

from src.api.services import build_promotion_candidates_payload, build_promotion_dashboard_payload
from src.configs.runtime import AppSettings
from src.inference.serving import BackendServingConfig
from src.models.promotion_workflow import load_promotion_candidates
from src.models.registry import ModelRegistryEntry, save_oasis_model_entry


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


def _write_candidate_summary(
    experiment_dir: Path,
    *,
    experiment_name: str,
    run_name: str,
    checkpoint_path: str,
    val_auroc: float,
    test_auroc: float,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "run_name": run_name,
                "tags": ["baseline"],
                "primary_split": "val",
                "training": {
                    "best_checkpoint_path": checkpoint_path,
                },
                "evaluations": {
                    "val": {
                        "metrics": {
                            "sample_count": 35,
                            "auroc": val_auroc,
                            "accuracy": 0.76,
                            "sensitivity": 0.8,
                            "specificity": 0.72,
                            "f1": 0.75,
                            "review_required_count": 9,
                            "mean_calibrated_confidence": 0.71,
                        }
                    },
                    "test": {
                        "metrics": {
                            "sample_count": 36,
                            "auroc": test_auroc,
                            "accuracy": 0.86,
                            "sensitivity": 0.9,
                            "specificity": 0.82,
                            "f1": 0.84,
                            "review_required_count": 8,
                            "mean_calibrated_confidence": 0.73,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "config.yaml").write_text("experiment_name: candidate\n", encoding="utf-8")
    (experiment_dir / "final_metrics.json").write_text(json.dumps({"accuracy": 0.86}), encoding="utf-8")


def test_load_promotion_candidates_builds_preflight_against_active_policy(
    tmp_path: Path,
) -> None:
    """Tracked experiments should become promotion candidates with advisory preflight checks."""

    settings = _build_settings(tmp_path)
    active_entry = ModelRegistryEntry(
        registry_version="1.0",
        model_id="oasis_current_baseline",
        dataset="oasis1",
        run_name="active_run",
        checkpoint_path="checkpoint.pt",
        model_config_path=None,
        preprocessing_config_path=None,
        image_size=[64, 64, 64],
        promoted_at_utc="2026-01-01T00:00:00+00:00",
        decision_support_only=True,
        clinical_disclaimer="Decision-support only.",
        benchmark={
            "benchmark_id": "oasis1_test_split42_frozen",
            "benchmark_name": "oasis1_test_split42_frozen",
            "dataset": "oasis1",
            "split_name": "test",
            "manifest_path": "manifest.csv",
            "manifest_hash_sha256": "abc",
            "sample_count": 36,
            "subject_count": 36,
            "subject_id_column": "subject_id",
            "subject_safe": True,
            "label_distribution": {"0": 21, "1": 15},
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "notes": [],
        },
        validation_metrics={"auroc": 0.78},
        test_metrics={"auroc": 0.82},
    )
    save_oasis_model_entry(active_entry, settings=settings)
    _write_candidate_summary(
        settings.outputs_root / "experiments" / "candidate_pass",
        experiment_name="candidate_pass",
        run_name="candidate_run",
        checkpoint_path="candidate_checkpoint.pt",
        val_auroc=0.82,
        test_auroc=0.84,
    )

    candidates = load_promotion_candidates(limit=5, settings=settings)

    assert len(candidates) == 1
    assert candidates[0].experiment_name == "candidate_pass"
    assert candidates[0].promotion_preflight.evaluable is True
    assert candidates[0].promotion_preflight.approved is True
    assert candidates[0].comparison_to_active["test"]["auroc"] == 0.02


def test_promotion_dashboard_payload_combines_candidates_studies_and_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Promotion dashboard should combine the active model with tracked candidates and studies."""

    settings = _build_settings(tmp_path)
    active_entry = ModelRegistryEntry(
        registry_version="1.0",
        model_id="oasis_current_baseline",
        dataset="oasis1",
        run_name="active_run",
        checkpoint_path="checkpoint.pt",
        model_config_path=None,
        preprocessing_config_path=None,
        image_size=[64, 64, 64],
        promoted_at_utc="2026-01-01T00:00:00+00:00",
        decision_support_only=True,
        clinical_disclaimer="Decision-support only.",
        promotion_decision={"approved": True},
        benchmark={
            "benchmark_id": "oasis1_test_split42_frozen",
            "benchmark_name": "oasis1_test_split42_frozen",
            "dataset": "oasis1",
            "split_name": "test",
            "manifest_path": "manifest.csv",
            "manifest_hash_sha256": "abc",
            "sample_count": 36,
            "subject_count": 36,
            "subject_id_column": "subject_id",
            "subject_safe": True,
            "label_distribution": {"0": 21, "1": 15},
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "notes": [],
        },
        validation_metrics={"auroc": 0.78},
        test_metrics={"auroc": 0.82},
    )
    save_oasis_model_entry(active_entry, settings=settings)
    _write_candidate_summary(
        settings.outputs_root / "experiments" / "candidate_pass",
        experiment_name="candidate_pass",
        run_name="candidate_run",
        checkpoint_path="candidate_checkpoint.pt",
        val_auroc=0.82,
        test_auroc=0.84,
    )

    study_root = settings.outputs_root / "model_selection" / "seed_sweep"
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "study_summary.json").write_text(
        json.dumps(
            {
                "study_name": "seed_sweep",
                "selection_split": "val",
                "selection_metric": "auroc",
                "best_experiment_name": "candidate_pass",
                "best_selection_score": 0.82,
                "best_checkpoint_path": "candidate_checkpoint.pt",
                "runs": [
                    {
                        "experiment_name": "candidate_pass",
                        "run_name": "candidate_run",
                    }
                ],
                "aggregate_metrics": [
                    {
                        "split": "val",
                        "metric_name": "auroc",
                        "mean": 0.8,
                        "ci95_low": 0.75,
                        "ci95_high": 0.85,
                    },
                    {
                        "split": "test",
                        "metric_name": "auroc",
                        "mean": 0.78,
                        "ci95_low": 0.7,
                        "ci95_high": 0.86,
                    },
                ],
                "notes": ["validation depth summary"],
            }
        ),
        encoding="utf-8",
    )

    history_root = settings.outputs_root / "model_registry" / "promotion_history"
    history_root.mkdir(parents=True, exist_ok=True)
    (history_root / "promotion_unit.json").write_text(
        json.dumps(
            {
                "run_name": "active_run",
                "benchmark": {"benchmark_name": "oasis1_test_split42_frozen"},
                "decision": {
                    "decision_id": "decision-1",
                    "checked_at_utc": "2026-01-01T00:00:00+00:00",
                    "approved": True,
                    "policy_name": "oasis_research_gate_v1",
                    "failed_checks": [],
                    "notes": [],
                },
                "active_registry_path": "registry.json",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.api.services.get_app_settings", lambda: settings)
    monkeypatch.setattr(
        "src.api.services.load_backend_serving_config",
        lambda *_, **__: BackendServingConfig(
            active_oasis_model_registry=settings.outputs_root / "model_registry" / "oasis_current_baseline.json",
        ),
    )

    candidates_payload = build_promotion_candidates_payload(limit=5)
    dashboard = build_promotion_dashboard_payload(candidate_limit=5, study_limit=5, history_limit=5)

    assert candidates_payload["total"] == 1
    assert candidates_payload["items"][0]["promotion_preflight"]["approved"] is True
    assert dashboard["summary"]["promotion_ready_candidates"] == 1
    assert dashboard["summary"]["top_candidate_experiment"] == "candidate_pass"
    assert dashboard["studies"][0]["study_name"] == "seed_sweep"
    assert dashboard["recent_promotion_decisions"][0]["decision_id"] == "decision-1"
