"""Tests for experiment-tracking exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.configs.runtime import AppSettings
from src.evaluation.evaluate_oasis import OASISEvaluationResult, OASISPredictionRecord
from src.evaluation.oasis_run import LoadedCheckpoint, OASISRunEvaluationConfig, OASISRunEvaluationPaths, OASISRunEvaluationResult
from src.training.experiment_tracking import ExperimentTrackingConfig, export_tracked_oasis_experiment, load_experiment_rows
from src.training.oasis_research import ResearchTrainingResult


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    data_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=project_root.parent,
        oasis_source_root=project_root.parent / "OASIS",
    )


def test_export_tracked_oasis_experiment_writes_comparison_ready_artifacts(tmp_path: Path) -> None:
    """Tracked experiments should save config, metrics, and publication figures."""

    pytest.importorskip("matplotlib")
    settings = _settings(tmp_path)
    run_root = settings.outputs_root / "runs" / "oasis" / "unit_run"
    checkpoint_root = run_root / "checkpoints"
    metrics_root = run_root / "metrics"
    reports_root = run_root / "reports"
    config_root = run_root / "configs"
    for folder in (checkpoint_root, metrics_root, reports_root, config_root):
        folder.mkdir(parents=True, exist_ok=True)
    (metrics_root / "epoch_metrics.csv").write_text("epoch,accuracy\n1,0.5\n", encoding="utf-8")
    (metrics_root / "epoch_metrics.json").write_text('[{"epoch":1,"accuracy":0.5}]', encoding="utf-8")
    (config_root / "resolved_config.json").write_text(json.dumps({"training": {"epochs": 1}}), encoding="utf-8")
    best_checkpoint = checkpoint_root / "best_model.pt"
    best_checkpoint.write_text("best", encoding="utf-8")

    training = ResearchTrainingResult(
        run_name="unit_run",
        run_root=run_root,
        best_checkpoint_path=best_checkpoint,
        last_checkpoint_path=best_checkpoint,
        epoch_metrics_csv_path=metrics_root / "epoch_metrics.csv",
        epoch_metrics_json_path=metrics_root / "epoch_metrics.json",
        confusion_matrix_path=metrics_root / "confusion_matrix.json",
        summary_report_path=reports_root / "summary_report.md",
        resolved_config_path=config_root / "resolved_config.json",
        best_epoch=1,
        best_monitor_value=0.4,
        stopped_early=False,
        final_metrics={"accuracy": 0.5},
    )
    predictions = [
        OASISPredictionRecord(
            sample_id="OAS1_0001",
            true_label=0,
            true_label_name="nondemented",
            predicted_label=0,
            predicted_label_name="nondemented",
            probabilities=[0.8, 0.2],
            confidence=0.8,
            entropy=0.4,
            normalized_entropy=0.4,
            probability_margin=0.6,
            uncertainty_score=0.2,
            calibrated_probabilities=[0.8, 0.2],
            calibrated_probability_score=0.2,
            confidence_level="medium",
            review_flag=False,
            meta={"image_path": "scan_001.hdr"},
        ),
        OASISPredictionRecord(
            sample_id="OAS1_0002",
            true_label=1,
            true_label_name="demented",
            predicted_label=1,
            predicted_label_name="demented",
            probabilities=[0.1, 0.9],
            confidence=0.9,
            entropy=0.2,
            normalized_entropy=0.2,
            probability_margin=0.8,
            uncertainty_score=0.1,
            calibrated_probabilities=[0.1, 0.9],
            calibrated_probability_score=0.9,
            confidence_level="high",
            review_flag=False,
            meta={"image_path": "scan_002.hdr"},
        ),
    ]
    evaluation = OASISRunEvaluationResult(
        config=OASISRunEvaluationConfig(run_name="unit_run", split="val"),
        checkpoint=LoadedCheckpoint(path=best_checkpoint, model_state_dict={}, metadata={}),
        result=OASISEvaluationResult(
            dataset="oasis1",
            dataset_type="3d_volumes",
            class_names=("nondemented", "demented"),
            metrics={"accuracy": 1.0, "auroc": 1.0, "sensitivity": 1.0, "specificity": 1.0},
            predictions=predictions,
        ),
        paths=OASISRunEvaluationPaths(
            evaluation_root=run_root / "evaluation" / "val",
            report_json_path=run_root / "evaluation" / "val" / "evaluation_report.json",
            predictions_csv_path=run_root / "evaluation" / "val" / "predictions.csv",
            metrics_json_path=run_root / "evaluation" / "val" / "metrics.json",
            summary_report_path=run_root / "evaluation" / "val" / "summary_report.md",
        ),
    )

    tracked = export_tracked_oasis_experiment(
        ExperimentTrackingConfig(experiment_name="tracked_unit", run_name="unit_run", tags=("baseline",)),
        training=training,
        evaluations=[evaluation],
        settings=settings,
    )

    assert tracked.paths.config_yaml_path.exists()
    assert tracked.paths.final_metrics_path.exists()
    assert tracked.paths.confusion_matrix_png_path.exists()
    assert tracked.paths.roc_curve_png_path.exists()
    rows = load_experiment_rows(settings.outputs_root / "experiments")
    assert rows.iloc[0]["experiment_name"] == "tracked_unit"
