"""Tests for checkpoint-backed OASIS inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.configs.runtime import AppSettings
from src.inference.predict_oasis import (
    OASISCheckpointPrediction,
    OASISInferenceConfig,
    predict_oasis_checkpoint,
    predict_oasis_image,
)
from src.models.factory import OASISModelConfig


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for inference tests."""

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
    """Build a tiny model that returns two-class logits."""

    class _TinyClassifier(torch.nn.Module):
        def forward(self, image: object) -> object:
            batch_size = image.shape[0]
            return torch.tensor([[0.1, 1.4]] * batch_size, dtype=torch.float32, device=image.device)

    return _TinyClassifier()


class _FakeTransform:
    def __init__(self, torch: object) -> None:
        self.torch = torch

    def __call__(self, sample: dict[str, object]) -> dict[str, object]:
        return {**sample, "image": self.torch.zeros((1, 2, 2, 2), dtype=self.torch.float32)}


def test_predict_oasis_image_applies_threshold_and_uncertainty(tmp_path: Path) -> None:
    """Single-image inference should produce probability and uncertainty metadata."""

    torch = pytest.importorskip("torch")
    image_path = tmp_path / "scan.hdr"
    image_path.write_text("placeholder", encoding="utf-8")

    result = predict_oasis_image(
        image_path,
        model=_build_tiny_classifier(torch),
        transforms=_FakeTransform(torch),
        threshold=0.7,
        meta={"subject_id": "OAS1_0001"},
    )

    assert result.source_dataset == "oasis1"
    assert result.predicted_index == 1
    assert result.label == "demented"
    assert result.probabilities[1] > 0.7
    assert result.meta["subject_id"] == "OAS1_0001"
    assert result.meta["threshold"] == 0.7
    assert "normalized_entropy" in result.meta


def test_predict_oasis_checkpoint_saves_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint-backed inference should save an auditable JSON report."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    image_path = tmp_path / "scan.hdr"
    image_path.write_text("placeholder", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": _build_tiny_classifier(torch).state_dict(), "epoch": 1}, checkpoint_path)

    monkeypatch.setattr("src.inference.predict_oasis.build_model", lambda _cfg: _build_tiny_classifier(torch))
    monkeypatch.setattr(
        "src.inference.predict_oasis.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 2, 2, 2)),
    )
    monkeypatch.setattr("src.inference.predict_oasis.build_oasis_infer_transforms", lambda _cfg: _FakeTransform(torch))

    result = predict_oasis_checkpoint(
        OASISInferenceConfig(
            checkpoint_path=checkpoint_path,
            image_path=image_path,
            output_name="unit_prediction",
            threshold=0.5,
            image_size=(2, 2, 2),
            subject_id="OAS1_0001",
        ),
        settings=settings,
    )

    assert isinstance(result, OASISCheckpointPrediction)
    assert result.prediction_json_path is not None
    assert result.prediction_json_path.exists()
    assert result.summary_report_path is not None
    assert result.summary_report_path.exists()

    payload = json.loads(result.prediction_json_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "oasis1"
    assert payload["predicted_label_name"] == "demented"
    assert payload["meta"]["subject_id"] == "OAS1_0001"
    assert payload["outputs"]["prediction_json"] == str(result.prediction_json_path)
