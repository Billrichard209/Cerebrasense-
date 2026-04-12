"""Tests for standalone OASIS checkpoint evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.evaluation.oasis_standalone import StandaloneOASISEvaluationConfig, evaluate_oasis_standalone
from src.models.factory import OASISModelConfig


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for standalone evaluation tests."""

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
    """Build a tiny torch classifier with predictable shape handling."""

    class _TinyClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, image: object) -> object:
            return self.linear(image.reshape(image.shape[0], -1))

    return _TinyClassifier()


def test_evaluate_oasis_standalone_writes_publication_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone evaluator should save metrics, predictions, ROC, and confusion matrix artifacts."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    settings = _build_settings(tmp_path)
    model = _build_tiny_classifier(torch)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, checkpoint_path)
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

    monkeypatch.setattr("src.evaluation.oasis_standalone._build_oasis_eval_loader", lambda _cfg: loader)
    monkeypatch.setattr("src.evaluation.oasis_standalone.build_model", lambda _cfg: _build_tiny_classifier(torch))
    monkeypatch.setattr(
        "src.evaluation.oasis_standalone.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 1, 2, 2)),
    )

    result = evaluate_oasis_standalone(
        StandaloneOASISEvaluationConfig(
            checkpoint_path=checkpoint_path,
            output_name="unit_standalone_eval",
            threshold=0.4,
            max_batches=1,
        ),
        settings=settings,
    )

    assert result.paths.metrics_json_path.exists()
    assert result.paths.metrics_csv_path.exists()
    assert result.paths.predictions_csv_path.exists()
    assert result.paths.confusion_matrix_json_path.exists()
    assert result.paths.confusion_matrix_csv_path.exists()
    assert result.paths.roc_curve_csv_path.exists()
    assert result.paths.roc_curve_png_path.exists()
    assert result.paths.confusion_matrix_png_path.exists()
    assert result.paths.summary_report_path.exists()
    assert result.metrics["sample_count"] == 2
    assert result.metrics["threshold"] == 0.4
    assert "confidence_level_counts" in result.metrics

    predictions = pd.read_csv(result.paths.predictions_csv_path)
    assert {"subject_id", "source_path", "true_label", "predicted_label", "probability", "dataset_name", "confidence_level", "review_flag"} <= set(
        predictions.columns
    )
    assert predictions["dataset_name"].unique().tolist() == ["oasis1"]

    confusion_payload = json.loads(result.paths.confusion_matrix_json_path.read_text(encoding="utf-8"))
    assert confusion_payload["layout"] == "[[true_negative, false_positive], [false_negative, true_positive]]"
