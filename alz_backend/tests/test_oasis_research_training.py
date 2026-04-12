"""Tests for the research-grade OASIS MONAI training runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.models.factory import OASISModelConfig
from src.training.oasis_research import (
    EarlyStoppingConfig,
    ResearchDataConfig,
    ResearchOASISTrainingConfig,
    _resolve_monitor_value,
    build_run_paths,
    load_research_oasis_training_config,
    run_research_oasis_training,
)


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for training-output tests."""

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


def test_load_research_oasis_training_config_parses_yaml(tmp_path: Path) -> None:
    """The YAML loader should produce a strongly typed training config."""

    config_path = tmp_path / "oasis_train.yaml"
    config_path.write_text(
        """
run_name: unit_yaml
epochs: 3
dry_run: true
data:
  batch_size: 2
  gradient_accumulation_steps: 3
  image_size: [32, 48, 64]
  split_seed: 123
optimizer:
  name: sgd
  learning_rate: 0.01
scheduler:
  name: step_lr
early_stopping:
  monitor: val_auroc
  mode: max
checkpoint:
  resume_from: checkpoint.pt
""",
        encoding="utf-8",
    )

    cfg = load_research_oasis_training_config(config_path)

    assert cfg.run_name == "unit_yaml"
    assert cfg.epochs == 3
    assert cfg.dry_run is True
    assert cfg.data.batch_size == 2
    assert cfg.data.gradient_accumulation_steps == 3
    assert cfg.data.image_size == (32, 48, 64)
    assert cfg.data.split_seed == 123
    assert cfg.optimizer.name == "sgd"
    assert cfg.scheduler.name == "step_lr"
    assert cfg.early_stopping.monitor == "val_auroc"
    assert cfg.early_stopping.mode == "max"
    assert cfg.checkpoint.resume_from == Path("checkpoint.pt")


def test_resolve_monitor_value_supports_user_friendly_aliases() -> None:
    """Early stopping should accept common validation metric aliases."""

    metrics = {"loss": 0.4, "accuracy": 0.75, "auroc": 0.8}

    assert _resolve_monitor_value(metrics, "val_loss") == pytest.approx(0.4)
    assert _resolve_monitor_value(metrics, "val_accuracy") == pytest.approx(0.75)
    assert _resolve_monitor_value(metrics, "val_auroc") == pytest.approx(0.8)


def test_build_run_paths_creates_reproducible_folder_structure(tmp_path: Path) -> None:
    """Run paths should live under outputs/runs/oasis and include required artifacts."""

    settings = _build_settings(tmp_path)
    paths = build_run_paths(settings, "unit_run")

    assert paths.run_root == settings.outputs_root / "runs" / "oasis" / "unit_run"
    assert paths.checkpoint_root.exists()
    assert paths.metrics_root.exists()
    assert paths.reports_root.exists()
    assert paths.config_root.exists()
    assert paths.best_checkpoint_path.name == "best_model.pt"
    assert paths.last_checkpoint_path.name == "last_model.pt"


class _TinyClassifier:
    """Small torch module for exercising the training loop without MONAI volumes."""

    def __init__(self, torch: object) -> None:
        self.module = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 2))

    def __getattr__(self, name: str) -> object:
        return getattr(self.module, name)

    def __call__(self, image: object) -> object:
        return self.module(image)


def test_run_research_oasis_training_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny patched run should still write checkpoints, metrics, config, and reports."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    train_loader = [
        {"image": torch.ones((2, 1, 2, 2, 2), dtype=torch.float32), "label": torch.tensor([0, 1])}
    ]
    val_loader = [
        {"image": torch.zeros((2, 1, 2, 2, 2), dtype=torch.float32), "label": torch.tensor([0, 1])}
    ]

    monkeypatch.setattr(
        "src.training.oasis_research._build_loaders",
        lambda _cfg: SimpleNamespace(train_loader=train_loader, val_loader=val_loader),
    )
    monkeypatch.setattr("src.training.oasis_research.build_model", lambda _cfg: _TinyClassifier(torch))
    monkeypatch.setattr(
        "src.training.oasis_research.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 2, 2, 2)),
    )

    cfg = ResearchOASISTrainingConfig(
        run_name="unit_research_run",
        epochs=1,
        device="cpu",
        dry_run=False,
        data=ResearchDataConfig(image_size=(2, 2, 2), split_seed=77, max_train_batches=1, max_val_batches=1),
        early_stopping=EarlyStoppingConfig(enabled=False),
    )

    result = run_research_oasis_training(cfg, settings=settings)

    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.exists()
    assert result.last_checkpoint_path is not None
    assert result.last_checkpoint_path.exists()
    assert result.epoch_metrics_csv_path.exists()
    assert result.epoch_metrics_json_path.exists()
    assert result.confusion_matrix_path.exists()
    assert result.summary_report_path.exists()
    assert result.resolved_config_path.exists()
    assert result.final_metrics["val_batches"] == 1
    assert result.final_metrics["train_batches"] == 1

    metrics_frame = pd.read_csv(result.epoch_metrics_csv_path)
    assert list(metrics_frame["epoch"]) == [1]
    assert {"train_loss", "val_loss", "accuracy", "auroc", "precision", "recall", "f1"} <= set(
        metrics_frame.columns
    )

    confusion_payload = json.loads(result.confusion_matrix_path.read_text(encoding="utf-8"))
    assert confusion_payload["layout"] == "[[true_negative, false_positive], [false_negative, true_positive]]"
    resolved_payload = json.loads(result.resolved_config_path.read_text(encoding="utf-8"))
    assert resolved_payload["training"]["data"]["split_seed"] == 77


def test_run_research_oasis_training_supports_gradient_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gradient accumulation should allow batch_size=1 runs to produce full training artifacts."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    train_loader = [
        {"image": torch.ones((1, 1, 2, 2, 2), dtype=torch.float32), "label": torch.tensor([0])},
        {"image": torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32), "label": torch.tensor([1])},
    ]
    val_loader = [
        {"image": torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32), "label": torch.tensor([0])},
        {"image": torch.ones((1, 1, 2, 2, 2), dtype=torch.float32), "label": torch.tensor([1])},
    ]

    monkeypatch.setattr(
        "src.training.oasis_research._build_loaders",
        lambda _cfg: SimpleNamespace(train_loader=train_loader, val_loader=val_loader),
    )
    monkeypatch.setattr("src.training.oasis_research.build_model", lambda _cfg: _TinyClassifier(torch))
    monkeypatch.setattr(
        "src.training.oasis_research.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 2, 2, 2)),
    )

    cfg = ResearchOASISTrainingConfig(
        run_name="unit_research_accumulation",
        epochs=1,
        device="cpu",
        dry_run=False,
        data=ResearchDataConfig(
            batch_size=1,
            gradient_accumulation_steps=2,
            image_size=(2, 2, 2),
            max_train_batches=2,
            max_val_batches=2,
        ),
        early_stopping=EarlyStoppingConfig(enabled=False),
    )

    result = run_research_oasis_training(cfg, settings=settings)

    assert result.epoch_metrics_csv_path.exists()
    metrics_frame = pd.read_csv(result.epoch_metrics_csv_path)
    assert int(metrics_frame.iloc[0]["train_batches"]) == 2
