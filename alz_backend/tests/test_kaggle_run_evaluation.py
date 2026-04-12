"""Tests for run-level Kaggle checkpoint evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.evaluation.kaggle_run import (
    KaggleRunEvaluationConfig,
    evaluate_kaggle_run_checkpoint,
    load_kaggle_checkpoint,
    resolve_kaggle_checkpoint_path,
)
from src.training.kaggle_research import ResearchKaggleDataConfig, ResearchKaggleTrainingConfig


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for Kaggle run-evaluation tests."""

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


class _TinyKaggleClassifier:
    """Small torch module for checkpoint evaluation tests."""

    def __init__(self, torch: object) -> None:
        self.module = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4))

    def __getattr__(self, name: str) -> object:
        return getattr(self.module, name)

    def __call__(self, image: object) -> object:
        return self.module(image)


def _write_checkpoint(settings: AppSettings, run_name: str, model: object, torch: object) -> Path:
    """Write a research-style Kaggle checkpoint."""

    checkpoint_root = settings.outputs_root / "runs" / "kaggle" / run_name / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / "best_model.pt"
    torch.save(
        {
            "epoch": 1,
            "best_epoch": 1,
            "best_monitor_value": 0.5,
            "dataset_type": "2d_slices",
            "class_names": ["A", "B", "C", "D"],
            "config": {"run_name": run_name},
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_load_kaggle_checkpoint_supports_research_payload(tmp_path: Path) -> None:
    """Checkpoint loader should extract model_state_dict from rich training payloads."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    model = _TinyKaggleClassifier(torch)
    checkpoint_path = _write_checkpoint(settings, "unit_run", model, torch)

    checkpoint = load_kaggle_checkpoint(checkpoint_path, device="cpu")

    assert checkpoint.path == checkpoint_path
    assert "1.weight" in checkpoint.model_state_dict
    assert checkpoint.metadata["epoch"] == 1


def test_resolve_kaggle_checkpoint_path_uses_run_folder(tmp_path: Path) -> None:
    """Checkpoint path resolution should default to the run-local checkpoint."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    model = _TinyKaggleClassifier(torch)
    checkpoint_path = _write_checkpoint(settings, "unit_run", model, torch)

    resolved = resolve_kaggle_checkpoint_path(KaggleRunEvaluationConfig(run_name="unit_run"), settings=settings)

    assert resolved == checkpoint_path


def test_evaluate_kaggle_run_checkpoint_writes_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run-level Kaggle evaluation should save metrics, predictions, and reports."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    model = _TinyKaggleClassifier(torch)
    _write_checkpoint(settings, "unit_run", model, torch)
    class_names = ("A", "B", "C", "D")
    loader = [
        {
            "image": torch.ones((4, 1, 2, 2), dtype=torch.float32),
            "label": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "label_name": list(class_names),
            "subject_id": ["s0", "s1", "s2", "s3"],
            "image_path": [f"sample_{index}.jpg" for index in range(4)],
            "meta": [{"subset": "OriginalDataset"} for _ in range(4)],
        }
    ]

    monkeypatch.setattr(
        "src.evaluation.kaggle_run.load_research_kaggle_training_config",
        lambda _path=None: ResearchKaggleTrainingConfig(
            run_name="ignored",
            epochs=1,
            device="cpu",
            data=ResearchKaggleDataConfig(batch_size=4, image_size_2d=(2, 2), max_val_batches=1, max_test_batches=1),
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.kaggle_run._build_loaders",
        lambda _cfg, _settings: SimpleNamespace(
            val_loader=loader,
            test_loader=loader,
            dataset_type="2d_slices",
            class_names=class_names,
            runtime_label_map=None,
        ),
    )
    monkeypatch.setattr("src.evaluation.kaggle_run.build_kaggle_monai_network", lambda _cfg: _TinyKaggleClassifier(torch))

    evaluation = evaluate_kaggle_run_checkpoint(
        KaggleRunEvaluationConfig(run_name="unit_run", output_name="unit_eval"),
        settings=settings,
    )

    assert evaluation.paths.report_json_path.exists()
    assert evaluation.paths.final_metrics_path.exists()
    assert evaluation.paths.val_predictions_path.exists()
    assert evaluation.paths.test_predictions_path.exists()
    assert evaluation.paths.summary_report_path.exists()
    assert evaluation.final_metrics["validation"]["sample_count"] == 4
    assert evaluation.final_metrics["test"]["sample_count"] == 4

    payload = json.loads(evaluation.paths.report_json_path.read_text(encoding="utf-8"))
    assert payload["run"]["run_name"] == "unit_run"
    assert payload["final_metrics"]["evaluation_type"] == "checkpoint_only"

    predictions = pd.read_csv(evaluation.paths.val_predictions_path)
    assert len(predictions) == 4
    assert {"image_path", "true_label", "predicted_label"} <= set(predictions.columns)
