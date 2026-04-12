"""Tests for one-command OASIS train-then-evaluate orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings
from src.evaluation.evaluate_oasis import OASISEvaluationResult
from src.evaluation.oasis_run import LoadedCheckpoint, OASISRunEvaluationPaths, OASISRunEvaluationResult
from src.training.oasis_experiment import OASISExperimentConfig, run_oasis_experiment
from src.training.oasis_research import ResearchDataConfig, ResearchOASISTrainingConfig, ResearchTrainingResult


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for experiment tests."""

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


def test_run_oasis_experiment_writes_combined_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The orchestrator should train, evaluate requested splits, and summarize the run."""

    settings = _build_settings(tmp_path)
    run_root = settings.outputs_root / "runs" / "oasis" / "unit_experiment"
    checkpoint_root = run_root / "checkpoints"
    metrics_root = run_root / "metrics"
    reports_root = run_root / "reports"
    config_root = run_root / "configs"
    for folder in (checkpoint_root, metrics_root, reports_root, config_root):
        folder.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_root / "best_model.pt"
    last_checkpoint = checkpoint_root / "last_model.pt"
    best_checkpoint.write_text("best", encoding="utf-8")
    last_checkpoint.write_text("last", encoding="utf-8")

    def fake_train(training_cfg, *, settings=None):
        return ResearchTrainingResult(
            run_name=training_cfg.run_name,
            run_root=run_root,
            best_checkpoint_path=best_checkpoint,
            last_checkpoint_path=last_checkpoint,
            epoch_metrics_csv_path=metrics_root / "epoch_metrics.csv",
            epoch_metrics_json_path=metrics_root / "epoch_metrics.json",
            confusion_matrix_path=metrics_root / "confusion_matrix.json",
            summary_report_path=reports_root / "summary_report.md",
            resolved_config_path=config_root / "resolved_config.json",
            best_epoch=1,
            best_monitor_value=0.5,
            stopped_early=False,
            final_metrics={"accuracy": 0.5, "val_loss": 0.7},
        )

    def fake_evaluate(eval_cfg, *, settings=None):
        evaluation_root = run_root / "evaluation" / eval_cfg.split
        evaluation_root.mkdir(parents=True, exist_ok=True)
        return OASISRunEvaluationResult(
            config=eval_cfg,
            checkpoint=LoadedCheckpoint(path=best_checkpoint, model_state_dict={}, metadata={"epoch": 1}),
            result=OASISEvaluationResult(
                dataset="oasis1",
                dataset_type="3d_volumes",
                class_names=("nondemented", "demented"),
                metrics={
                    "sample_count": 2,
                    "accuracy": 0.5,
                    "auroc": 0.5,
                    "f1": 0.5,
                    "mean_confidence": 0.6,
                },
                predictions=[],
            ),
            paths=OASISRunEvaluationPaths(
                evaluation_root=evaluation_root,
                report_json_path=evaluation_root / "evaluation_report.json",
                predictions_csv_path=evaluation_root / "predictions.csv",
                metrics_json_path=evaluation_root / "metrics.json",
                summary_report_path=evaluation_root / "summary_report.md",
            ),
        )

    monkeypatch.setattr("src.training.oasis_experiment.run_research_oasis_training", fake_train)
    monkeypatch.setattr("src.training.oasis_experiment.evaluate_oasis_run_checkpoint", fake_evaluate)

    cfg = OASISExperimentConfig(
        training=ResearchOASISTrainingConfig(
            run_name="unit_experiment",
            data=ResearchDataConfig(image_size=(32, 32, 32), max_train_batches=1, max_val_batches=1),
        ),
        evaluate_splits=("val", "test"),
        max_eval_batches=1,
    )
    result = run_oasis_experiment(cfg, settings=settings)

    assert result.summary_json_path.exists()
    assert result.summary_report_path.exists()
    assert result.tracked_experiment is not None
    assert result.tracked_experiment.paths.final_metrics_path.exists()
    assert [evaluation.config.split for evaluation in result.evaluations] == ["val", "test"]
    assert all(evaluation.config.max_batches == 1 for evaluation in result.evaluations)

    payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert payload["run_name"] == "unit_experiment"
    assert payload["experiment_tracking_root"] is not None
    assert payload["training"]["best_epoch"] == 1
    assert [evaluation["split"] for evaluation in payload["evaluations"]] == ["val", "test"]
