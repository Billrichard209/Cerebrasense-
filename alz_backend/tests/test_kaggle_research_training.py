"""Tests for the separate Kaggle research training runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.training.kaggle_research import (
    ResearchKaggleDataConfig,
    ResearchKaggleTrainingConfig,
    _compute_macro_ovr_auroc,
    build_kaggle_run_paths,
    load_research_kaggle_training_config,
    run_research_kaggle_training,
)


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for Kaggle training-output tests."""

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


def test_load_research_kaggle_training_config_parses_yaml(tmp_path: Path) -> None:
    """The YAML loader should produce a strongly typed Kaggle training config."""

    config_path = tmp_path / "kaggle_train.yaml"
    config_path.write_text(
        """
run_name: unit_kaggle_yaml
epochs: 2
dry_run: true
dropout_prob: 0.2
data:
  batch_size: 4
  gradient_accumulation_steps: 2
  image_size_2d: [128, 160]
  image_size_3d: [32, 32, 32]
optimizer:
  name: sgd
early_stopping:
  monitor: val_macro_ovr_auroc
  mode: max
checkpoint:
  resume_from: checkpoint.pt
""",
        encoding="utf-8",
    )

    cfg = load_research_kaggle_training_config(config_path)

    assert cfg.run_name == "unit_kaggle_yaml"
    assert cfg.epochs == 2
    assert cfg.dry_run is True
    assert cfg.dropout_prob == pytest.approx(0.2)
    assert cfg.data.batch_size == 4
    assert cfg.data.gradient_accumulation_steps == 2
    assert cfg.data.image_size_2d == (128, 160)
    assert cfg.data.image_size_3d == (32, 32, 32)
    assert cfg.optimizer.name == "sgd"
    assert cfg.early_stopping.monitor == "val_macro_ovr_auroc"
    assert cfg.checkpoint.resume_from == Path("checkpoint.pt")


def test_build_kaggle_run_paths_creates_reproducible_folder_structure(tmp_path: Path) -> None:
    """Run paths should live under outputs/runs/kaggle and include required artifacts."""

    settings = _build_settings(tmp_path)
    paths = build_kaggle_run_paths(settings, "unit_kaggle_run")

    assert paths.run_root == settings.outputs_root / "runs" / "kaggle" / "unit_kaggle_run"
    assert paths.checkpoint_root.exists()
    assert paths.metrics_root.exists()
    assert paths.reports_root.exists()
    assert paths.config_root.exists()
    assert paths.evaluation_root.exists()
    assert paths.best_checkpoint_path.name == "best_model.pt"
    assert paths.test_predictions_path.name == "test_predictions.csv"


class _TinyKaggleClassifier:
    """Small torch module for exercising the Kaggle training loop."""

    def __init__(self, torch: object) -> None:
        self.module = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4))

    def __getattr__(self, name: str) -> object:
        return getattr(self.module, name)

    def __call__(self, image: object) -> object:
        return self.module(image)


def test_run_research_kaggle_training_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny patched Kaggle run should still write checkpoints and evaluation artifacts."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    class_names = ("NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented")
    train_loader = [
        {
            "image": torch.ones((4, 1, 2, 2), dtype=torch.float32),
            "label": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "label_name": list(class_names),
            "subject_id": ["s0", "s1", "s2", "s3"],
            "image_path": [f"train_{index}.jpg" for index in range(4)],
            "meta": [{"subset": "OriginalDataset"} for _ in range(4)],
        }
    ]
    val_loader = [
        {
            "image": torch.zeros((4, 1, 2, 2), dtype=torch.float32),
            "label": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "label_name": list(class_names),
            "subject_id": ["v0", "v1", "v2", "v3"],
            "image_path": [f"val_{index}.jpg" for index in range(4)],
            "meta": [{"subset": "OriginalDataset"} for _ in range(4)],
        }
    ]
    test_loader = [
        {
            "image": torch.full((4, 1, 2, 2), 0.5, dtype=torch.float32),
            "label": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "label_name": list(class_names),
            "subject_id": ["t0", "t1", "t2", "t3"],
            "image_path": [f"test_{index}.jpg" for index in range(4)],
            "meta": [{"subset": "OriginalDataset"} for _ in range(4)],
        }
    ]

    monkeypatch.setattr(
        "src.training.kaggle_research._build_loaders",
        lambda _cfg, _settings: SimpleNamespace(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset_type="2d_slices",
            class_names=class_names,
            runtime_label_map=None,
        ),
    )
    monkeypatch.setattr("src.training.kaggle_research.build_kaggle_monai_network", lambda _cfg: _TinyKaggleClassifier(torch))

    cfg = ResearchKaggleTrainingConfig(
        run_name="unit_kaggle_research",
        epochs=1,
        device="cpu",
        dry_run=False,
        data=ResearchKaggleDataConfig(
            batch_size=4,
            image_size_2d=(2, 2),
            max_train_batches=1,
            max_val_batches=1,
            max_test_batches=1,
        ),
    )

    result = run_research_kaggle_training(cfg, settings=settings)

    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.exists()
    assert result.last_checkpoint_path is not None
    assert result.last_checkpoint_path.exists()
    assert result.epoch_metrics_csv_path.exists()
    assert result.epoch_metrics_json_path.exists()
    assert result.final_metrics_path.exists()
    assert result.val_confusion_matrix_path.exists()
    assert result.test_confusion_matrix_path.exists()
    assert result.val_predictions_path.exists()
    assert result.test_predictions_path.exists()
    assert result.summary_report_path.exists()
    assert result.resolved_config_path.exists()
    assert result.final_metrics["validation"]["sample_count"] == 4
    assert result.final_metrics["test"]["sample_count"] == 4

    metrics_frame = pd.read_csv(result.epoch_metrics_csv_path)
    assert list(metrics_frame["epoch"]) == [1]
    assert {"train_loss", "val_loss", "accuracy", "macro_f1", "macro_ovr_auroc"} <= set(metrics_frame.columns)

    predictions_frame = pd.read_csv(result.val_predictions_path)
    assert len(predictions_frame) == 4
    assert "confidence" in predictions_frame.columns

    resolved_payload = json.loads(result.resolved_config_path.read_text(encoding="utf-8"))
    assert resolved_payload["dataset_type"] == "2d_slices"
    assert resolved_payload["class_names"] == list(class_names)


def test_compute_macro_ovr_auroc_normalizes_probability_rows() -> None:
    """Macro AUROC should handle half-precision style probability rows cleanly."""

    y_true = [0, 1, 2, 3]
    y_score = [
        [0.9970703125, 0.0001, 0.00202178955078125, 0.000812530517578125],
        [0.0001, 0.9970703125, 0.00202178955078125, 0.000812530517578125],
        [0.0001, 0.00202178955078125, 0.9970703125, 0.000812530517578125],
        [0.0001, 0.00202178955078125, 0.000812530517578125, 0.9970703125],
    ]

    auroc = _compute_macro_ovr_auroc(y_true, y_score, labels=[0, 1, 2, 3])

    assert auroc > 0.99
