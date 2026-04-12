"""Tests for run-level OASIS checkpoint evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.evaluation.calibration import ConfidenceBandConfig
from src.evaluation.oasis_run import (
    OASISRunEvaluationConfig,
    evaluate_oasis_run_checkpoint,
    load_oasis_checkpoint,
    resolve_oasis_checkpoint_path,
)
from src.inference.serving import BackendServingConfig, OASISDecisionPolicy
from src.models.factory import OASISModelConfig


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for run evaluation tests."""

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


def _build_tiny_classifier(torch: object) -> object:
    """Small torch module for checkpoint evaluation tests."""

    class _TinyClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, image: object) -> object:
            return self.linear(image.reshape(image.shape[0], -1))

    return _TinyClassifier()


def _write_checkpoint(settings: AppSettings, run_name: str, model: object, torch: object) -> Path:
    """Write a research-style checkpoint for one test run."""

    checkpoint_root = settings.outputs_root / "runs" / "oasis" / run_name / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / "best_model.pt"
    torch.save(
        {
            "epoch": 2,
            "best_epoch": 1,
            "best_monitor_value": 0.5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "config": {"run_name": run_name, "resume_from": checkpoint_path},
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_load_oasis_checkpoint_supports_research_payload(tmp_path: Path) -> None:
    """Checkpoint loader should extract model_state_dict from rich training payloads."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    model = _build_tiny_classifier(torch)
    checkpoint_path = _write_checkpoint(settings, "unit_run", model, torch)

    checkpoint = load_oasis_checkpoint(checkpoint_path, device="cpu")

    assert checkpoint.path == checkpoint_path
    assert "linear.weight" in checkpoint.model_state_dict
    assert checkpoint.metadata["epoch"] == 2
    assert "optimizer_state_dict" in checkpoint.metadata


def test_resolve_oasis_checkpoint_path_uses_run_folder(tmp_path: Path) -> None:
    """Checkpoint path resolution should default to the run-local best checkpoint."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    model = _build_tiny_classifier(torch)
    checkpoint_path = _write_checkpoint(settings, "unit_run", model, torch)

    resolved = resolve_oasis_checkpoint_path(
        OASISRunEvaluationConfig(run_name="unit_run"),
        settings=settings,
    )

    assert resolved == checkpoint_path


def test_evaluate_oasis_run_checkpoint_writes_run_local_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run-level evaluation should save metrics, predictions, and summary reports."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    saved_model = _build_tiny_classifier(torch)
    _write_checkpoint(settings, "unit_run", saved_model, torch)

    loader = [
        {
            "image": torch.tensor(
                [
                    [[[[1.0, 0.0], [0.0, 1.0]]]],
                    [[[[0.0, 1.0], [1.0, 0.0]]]],
                ],
                dtype=torch.float32,
            ),
            "label": torch.tensor([0, 1]),
            "subject_id": ["OAS1_0001", "OAS1_0002"],
            "session_id": ["OAS1_0001_MR1", "OAS1_0002_MR1"],
            "image_path": ["scan_001.hdr", "scan_002.hdr"],
        }
    ]

    monkeypatch.setattr("src.evaluation.oasis_run._build_loader", lambda _cfg: loader)
    monkeypatch.setattr("src.evaluation.oasis_run.build_model", lambda _cfg: _build_tiny_classifier(torch))
    monkeypatch.setattr(
        "src.evaluation.oasis_run.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 1, 2, 2)),
    )

    evaluation = evaluate_oasis_run_checkpoint(
        OASISRunEvaluationConfig(run_name="unit_run", split="val", max_batches=1),
        settings=settings,
    )

    assert evaluation.paths.report_json_path.exists()
    assert evaluation.paths.metrics_json_path.exists()
    assert evaluation.paths.predictions_csv_path.exists()
    assert evaluation.paths.summary_report_path.exists()
    assert evaluation.result.metrics["sample_count"] == 2

    payload = json.loads(evaluation.paths.report_json_path.read_text(encoding="utf-8"))
    assert payload["run"]["run_name"] == "unit_run"
    assert payload["run"]["checkpoint_metadata"]["epoch"] == 2
    assert "optimizer_state_dict_keys" in payload["run"]["checkpoint_metadata"]

    predictions = pd.read_csv(evaluation.paths.predictions_csv_path)
    assert len(predictions) == 2
    assert {"meta_subject_id", "probability_class_0", "probability_class_1"} <= set(predictions.columns)


def test_evaluate_oasis_run_checkpoint_can_use_active_serving_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run-level evaluation should honor the shared serving decision threshold when requested."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    saved_model = _build_tiny_classifier(torch)
    _write_checkpoint(settings, "unit_run", saved_model, torch)

    class _ThresholdModel(torch.nn.Module):
        def forward(self, image: object) -> object:
            batch_size = image.shape[0]
            logits = torch.tensor([[0.0, 0.4054651]] * batch_size, dtype=torch.float32)
            return logits.to(image.device)

    loader = [
        {
            "image": torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32),
            "label": torch.tensor([1]),
            "subject_id": ["OAS1_0001"],
            "session_id": ["OAS1_0001_MR1"],
            "image_path": ["scan_001.hdr"],
        }
    ]

    monkeypatch.setattr("src.evaluation.oasis_run._build_loader", lambda _cfg: loader)
    monkeypatch.setattr(
        "src.evaluation.oasis_run.load_oasis_model_for_evaluation",
        lambda _cfg, settings=None: (
            _ThresholdModel(),
            OASISModelConfig(expected_input_shape=(1, 1, 1, 2, 2)),
            SimpleNamespace(path=settings.outputs_root / "runs" / "oasis" / "unit_run" / "checkpoints" / "best_model.pt", metadata={}),
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.oasis_run._resolve_run_decision_policy",
        lambda _cfg, settings=None: OASISDecisionPolicy(
            threshold=0.7,
            confidence_config=ConfidenceBandConfig(),
            serving_config=BackendServingConfig(active_oasis_model_registry=settings.outputs_root / "registry.json"),
            registry_entry=None,
        ),
    )

    evaluation = evaluate_oasis_run_checkpoint(
        OASISRunEvaluationConfig(run_name="unit_run", split="val", max_batches=1, use_active_serving_policy=True),
        settings=settings,
    )

    assert evaluation.result.metrics["threshold"] == pytest.approx(0.7)
    assert evaluation.result.predictions[0].predicted_label == 0
