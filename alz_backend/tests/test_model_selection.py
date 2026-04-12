"""Tests for multi-seed OASIS model selection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.configs.runtime import AppSettings
from src.training.model_selection import OASISModelSelectionConfig, run_oasis_model_selection_study
from src.training.oasis_experiment import OASISExperimentConfig
from src.training.oasis_research import ResearchDataConfig, ResearchOASISTrainingConfig


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for model-selection tests."""

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


def test_run_oasis_model_selection_study_aggregates_runs_and_picks_best_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The study runner should flatten seed runs and select the best validation experiment."""

    settings = _build_settings(tmp_path)

    def fake_run_oasis_experiment(cfg, *, settings=None):
        seed = cfg.training.data.seed
        split_seed = cfg.training.data.split_seed
        run_dir = f"seed_{seed}" if split_seed is None else f"seed_{seed}_split_{split_seed}"
        checkpoint_path = settings.outputs_root / "runs" / run_dir / "best_model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        experiment_root = settings.outputs_root / "experiments" / run_dir
        experiment_root.mkdir(parents=True, exist_ok=True)
        split_offset = 0.0 if split_seed is None else {101: 0.00, 202: -0.02}[split_seed]
        val_auroc = {11: 0.71, 12: 0.79, 13: 0.74}[seed] + split_offset
        test_auroc = {11: 0.68, 12: 0.76, 13: 0.72}[seed] + split_offset
        return SimpleNamespace(
            training=SimpleNamespace(run_name=cfg.training.run_name, best_checkpoint_path=checkpoint_path),
            tracked_experiment=SimpleNamespace(paths=SimpleNamespace(experiment_root=experiment_root)),
            evaluations=[
                SimpleNamespace(config=SimpleNamespace(split="val"), result=SimpleNamespace(metrics={"accuracy": 0.5, "auroc": val_auroc})),
                SimpleNamespace(config=SimpleNamespace(split="test"), result=SimpleNamespace(metrics={"accuracy": 0.5, "auroc": test_auroc})),
            ],
        )

    monkeypatch.setattr("src.training.model_selection.run_oasis_experiment", fake_run_oasis_experiment)

    result = run_oasis_model_selection_study(
        OASISModelSelectionConfig(
            study_name="unit_selection",
            base_experiment=OASISExperimentConfig(training=ResearchOASISTrainingConfig(run_name="baseline")),
            seeds=(11, 12, 13),
            selection_split="val",
            selection_metric="auroc",
        ),
        settings=settings,
    )

    assert result.best_row.seed == 12
    assert result.runs_csv_path.exists()
    assert result.summary_json_path.exists()
    assert result.aggregate_json_path.exists()
    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary_payload["best_seed"] == 12
    assert summary_payload["best_experiment_name"] == "unit_selection_seed12"
    assert any(item["metric_name"] == "auroc" and item["split"] == "val" for item in summary_payload["aggregate_metrics"])


def test_run_oasis_model_selection_study_supports_repeated_split_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The study runner should support repeated subject-safe split seeds."""

    settings = _build_settings(tmp_path)

    def fake_run_oasis_experiment(cfg, *, settings=None):
        seed = cfg.training.data.seed
        split_seed = cfg.training.data.split_seed
        assert cfg.training.data.gradient_accumulation_steps == 2
        checkpoint_path = settings.outputs_root / "runs" / f"seed_{seed}_split_{split_seed}" / "best_model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        experiment_root = settings.outputs_root / "experiments" / f"seed_{seed}_split_{split_seed}"
        experiment_root.mkdir(parents=True, exist_ok=True)
        base_score = {42: 0.75, 43: 0.73}[seed]
        split_adjustment = {101: 0.02, 202: -0.01}[split_seed]
        return SimpleNamespace(
            training=SimpleNamespace(run_name=cfg.training.run_name, best_checkpoint_path=checkpoint_path),
            tracked_experiment=SimpleNamespace(paths=SimpleNamespace(experiment_root=experiment_root)),
            evaluations=[
                SimpleNamespace(config=SimpleNamespace(split="val"), result=SimpleNamespace(metrics={"accuracy": 0.5, "auroc": base_score + split_adjustment})),
                SimpleNamespace(config=SimpleNamespace(split="test"), result=SimpleNamespace(metrics={"accuracy": 0.5, "auroc": base_score + split_adjustment - 0.03})),
            ],
        )

    monkeypatch.setattr("src.training.model_selection.run_oasis_experiment", fake_run_oasis_experiment)

    result = run_oasis_model_selection_study(
        OASISModelSelectionConfig(
            study_name="unit_repeated_splits",
            base_experiment=OASISExperimentConfig(
                training=ResearchOASISTrainingConfig(
                    run_name="baseline",
                    data=ResearchDataConfig(gradient_accumulation_steps=2),
                )
            ),
            seeds=(42, 43),
            split_seeds=(101, 202),
            selection_split="val",
            selection_metric="auroc",
        ),
        settings=settings,
    )

    assert len(result.rows) == 4
    assert result.best_row.seed == 42
    assert result.best_row.split_seed == 101
    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary_payload["split_seeds"] == [101, 202]
    assert summary_payload["best_split_seed"] == 101
    assert any(row["split_seed"] == 202 for row in summary_payload["runs"])


def test_run_oasis_model_selection_study_can_resume_existing_runs(
    tmp_path: Path,
) -> None:
    """The study runner should reuse completed run artifacts when resume_existing is enabled."""

    settings = _build_settings(tmp_path)
    run_name = "baseline_seed42_split101"
    experiment_name = "resume_study_seed42_split101"
    run_root = settings.outputs_root / "runs" / "oasis" / run_name
    checkpoint_path = run_root / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    for split, metrics in {
        "val": {"accuracy": 0.7, "auroc": 0.81},
        "test": {"accuracy": 0.68, "auroc": 0.79},
    }.items():
        eval_root = run_root / "evaluation" / f"post_train_{split}_best_model"
        eval_root.mkdir(parents=True, exist_ok=True)
        (eval_root / "evaluation_report.json").write_text(
            json.dumps({"metrics": metrics}),
            encoding="utf-8",
        )
    tracked_root = settings.outputs_root / "experiments" / experiment_name
    tracked_root.mkdir(parents=True, exist_ok=True)

    result = run_oasis_model_selection_study(
        OASISModelSelectionConfig(
            study_name="resume_study",
            base_experiment=OASISExperimentConfig(training=ResearchOASISTrainingConfig(run_name="baseline")),
            seeds=(42,),
            split_seeds=(101,),
            selection_split="val",
            selection_metric="auroc",
            resume_existing=True,
        ),
        settings=settings,
    )

    assert len(result.rows) == 1
    assert result.best_row.run_name == run_name
    assert result.best_row.best_checkpoint_path == str(checkpoint_path)
    assert result.best_row.tracked_experiment_root == str(tracked_root)
    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary_payload["resume_existing"] is True
