"""Tests for OASIS baseline classification evaluation and uncertainty scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.configs.runtime import AppSettings
from src.evaluation.evaluate_oasis import (
    evaluate_oasis_model_on_loader,
    evaluate_oasis_probabilities,
    save_oasis_evaluation_report,
)
from src.evaluation.metrics import compute_binary_classification_metrics, compute_uncertainty_from_probabilities
from src.evaluation.metrics import compute_binary_roc_curve, threshold_binary_scores


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for report-writing tests."""

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


def test_binary_classification_metrics_are_explicit() -> None:
    """Binary metrics should expose confusion counts and derived rates."""

    metrics = compute_binary_classification_metrics([0, 0, 1, 1], [0, 1, 1, 0])

    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall_sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["confusion_counts"] == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }


def test_threshold_binary_scores_is_configurable() -> None:
    """Thresholding should be explicit and configurable for decision-support reports."""

    assert threshold_binary_scores([0.2, 0.5, 0.8], threshold=0.5) == [0, 1, 1]
    assert threshold_binary_scores([0.2, 0.5, 0.8], threshold=0.7) == [0, 0, 1]


def test_compute_binary_roc_curve_returns_points_and_auc() -> None:
    """ROC utility should produce curve points independent from plotting."""

    curve = compute_binary_roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])

    assert curve["is_defined"] is True
    assert curve["auroc"] == pytest.approx(0.75)
    assert curve["fpr"][0] == pytest.approx(0.0)
    assert curve["tpr"][0] == pytest.approx(0.0)


def test_compute_binary_roc_curve_handles_single_class_splits() -> None:
    """Single-class smoke runs should save reports without pretending ROC is meaningful."""

    curve = compute_binary_roc_curve([0, 0], [0.1, 0.2])

    assert curve["is_defined"] is False
    assert curve["auroc"] == 0.0
    assert "one class" in curve["warning"]


def test_uncertainty_from_probabilities_scores_confidence_margin_and_entropy() -> None:
    """Uncertainty helper should score confident and uncertain predictions differently."""

    rows = compute_uncertainty_from_probabilities([[0.95, 0.05], [0.5, 0.5]])

    assert rows[0]["confidence"] == pytest.approx(0.95)
    assert rows[0]["probability_margin"] == pytest.approx(0.90)
    assert rows[0]["uncertainty_score"] == pytest.approx(0.05)
    assert rows[1]["confidence"] == pytest.approx(0.5)
    assert rows[1]["normalized_entropy"] == pytest.approx(1.0)
    assert rows[1]["probability_margin"] == pytest.approx(0.0)


def test_evaluate_oasis_probabilities_builds_records_and_metrics(tmp_path: Path) -> None:
    """OASIS probability evaluation should produce reportable records and metrics."""

    result = evaluate_oasis_probabilities(
        [0, 1, 1],
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]],
        sample_meta=[
            {"subject_id": "OAS1_0001", "session_id": "OAS1_0001_MR1"},
            {"subject_id": "OAS1_0002", "session_id": "OAS1_0002_MR1"},
            {"subject_id": "OAS1_0003", "session_id": "OAS1_0003_MR1"},
        ],
    )

    assert result.metrics["sample_count"] == 3
    assert result.metrics["accuracy"] == pytest.approx(2 / 3)
    assert result.metrics["auroc"] == pytest.approx(1.0)
    assert result.metrics["confusion_counts"]["false_negative"] == 1
    assert result.predictions[0].sample_id == "OAS1_0001"
    assert result.predictions[1].predicted_label_name == "demented"

    settings = _build_settings(tmp_path)
    output_path = save_oasis_evaluation_report(result, settings=settings, run_name="unit_eval")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "oasis1"
    assert payload["metrics"]["sample_count"] == 3
    assert len(payload["predictions"]) == 3


def test_evaluate_oasis_probabilities_uses_explicit_decision_threshold() -> None:
    """Probability evaluation should respect an explicit binary decision threshold."""

    result = evaluate_oasis_probabilities(
        [1],
        [[0.4, 0.6]],
        decision_threshold=0.7,
    )

    assert result.metrics["threshold"] == pytest.approx(0.7)
    assert result.predictions[0].predicted_label == 0
    assert result.predictions[0].predicted_label_name == "nondemented"


class _FakeModel:
    def __init__(self, logits: object) -> None:
        self.logits = logits

    def to(self, _device: str) -> "_FakeModel":
        return self

    def eval(self) -> None:
        return None

    def __call__(self, _images: object) -> object:
        return self.logits


def test_evaluate_oasis_model_on_loader_handles_torch_batches() -> None:
    """Model-on-loader evaluation should consume torch-style image/label batches."""

    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    loader = [
        {
            "image": torch.zeros((2, 1, 8, 8, 8), dtype=torch.float32),
            "image_path": ["scan_001.hdr", "scan_002.hdr"],
            "label": torch.tensor([0, 1], dtype=torch.long),
            "subject_id": ["OAS1_0001", "OAS1_0002"],
            "session_id": ["OAS1_0001_MR1", "OAS1_0002_MR1"],
        }
    ]

    result = evaluate_oasis_model_on_loader(model=_FakeModel(logits), loader=loader, device="cpu")

    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["sample_count"] == 2
    assert result.predictions[0].sample_id == "OAS1_0001"
    assert result.predictions[0].meta["image_path"] == "scan_001.hdr"
    assert "image" not in result.predictions[0].meta
    assert result.predictions[1].predicted_label_name == "demented"
