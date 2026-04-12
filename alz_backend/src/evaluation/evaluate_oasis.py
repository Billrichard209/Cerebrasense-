"""OASIS-1 classification evaluation and probability scoring utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.configs.runtime import AppSettings, get_app_settings
from src.models.oasis_model import build_oasis_class_names
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_torch_symbols

from .calibration import ConfidenceBandConfig, summarize_calibrated_confidence
from .metrics import (
    accuracy_score,
    compute_binary_classification_metrics,
    compute_uncertainty_from_probabilities,
    normalize_probabilities,
    threshold_binary_scores,
)

_load_torch_symbols = load_torch_symbols


@dataclass(slots=True)
class OASISPredictionRecord:
    """One model prediction with probability and uncertainty details."""

    sample_id: str
    true_label: int | None
    true_label_name: str | None
    predicted_label: int
    predicted_label_name: str
    probabilities: list[float]
    confidence: float
    entropy: float
    normalized_entropy: float
    probability_margin: float
    uncertainty_score: float
    calibrated_probabilities: list[float] = field(default_factory=list)
    calibrated_probability_score: float | None = None
    confidence_level: str = "low"
    review_flag: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the prediction record for reports."""

        return {
            "sample_id": self.sample_id,
            "true_label": self.true_label,
            "true_label_name": self.true_label_name,
            "predicted_label": self.predicted_label,
            "predicted_label_name": self.predicted_label_name,
            "probabilities": list(self.probabilities),
            "confidence": self.confidence,
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "probability_margin": self.probability_margin,
            "uncertainty_score": self.uncertainty_score,
            "calibrated_probabilities": list(self.calibrated_probabilities),
            "calibrated_probability_score": self.calibrated_probability_score,
            "confidence_level": self.confidence_level,
            "review_flag": self.review_flag,
            "meta": dict(self.meta),
        }


@dataclass(slots=True)
class OASISEvaluationResult:
    """Evaluation result for OASIS binary classification."""

    dataset: str
    dataset_type: str
    class_names: tuple[str, ...]
    metrics: dict[str, Any]
    predictions: list[OASISPredictionRecord]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the evaluation result for JSON reports."""

        return {
            "dataset": self.dataset,
            "dataset_type": self.dataset_type,
            "class_names": list(self.class_names),
            "metrics": self.metrics,
            "predictions": [prediction.to_payload() for prediction in self.predictions],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def evaluate_oasis_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Return legacy starter evaluation metrics for OASIS predictions."""

    return {"accuracy": accuracy_score(y_true, y_pred)}


def _label_name(label: int | None, class_names: tuple[str, ...]) -> str | None:
    """Resolve a label index to a class name."""

    if label is None:
        return None
    if 0 <= int(label) < len(class_names):
        return class_names[int(label)]
    return f"class_{label}"


def _default_sample_id(index: int, meta: dict[str, Any] | None) -> str:
    """Resolve a stable sample identifier for prediction records."""

    if meta:
        for key in ("sample_id", "subject_id", "session_id", "image_path"):
            value = meta.get(key)
            if value:
                return str(value)
    return f"sample_{index:05d}"


def build_prediction_records(
    probabilities: Any,
    *,
    y_true: list[int] | None = None,
    class_names: tuple[str, ...] | None = None,
    sample_meta: list[dict[str, Any]] | None = None,
    calibration_config: ConfidenceBandConfig | None = None,
    decision_threshold: float = 0.5,
) -> list[OASISPredictionRecord]:
    """Build per-sample OASIS prediction records from probabilities."""

    resolved_class_names = class_names or build_oasis_class_names()
    probability_array = normalize_probabilities(probabilities)
    if probability_array.shape[1] >= 2:
        predictions = np.asarray(
            threshold_binary_scores(
                [float(row[1]) for row in probability_array.tolist()],
                threshold=decision_threshold,
            )
        )
    else:
        predictions = np.argmax(probability_array, axis=1)
    uncertainty_rows = compute_uncertainty_from_probabilities(probability_array)
    calibrated_rows = summarize_calibrated_confidence(probability_array, config=calibration_config)
    if y_true is not None and len(y_true) != len(predictions):
        raise ValueError(f"Expected y_true length {len(y_true)} to match predictions length {len(predictions)}.")
    if sample_meta is not None and len(sample_meta) != len(predictions):
        raise ValueError(f"Expected sample_meta length {len(sample_meta)} to match predictions length {len(predictions)}.")

    records: list[OASISPredictionRecord] = []
    for index, predicted_label in enumerate(predictions.tolist()):
        true_label = None if y_true is None else int(y_true[index])
        meta = {} if sample_meta is None else dict(sample_meta[index])
        uncertainty = uncertainty_rows[index]
        calibrated = calibrated_rows[index]
        records.append(
            OASISPredictionRecord(
                sample_id=_default_sample_id(index, meta),
                true_label=true_label,
                true_label_name=_label_name(true_label, resolved_class_names),
                predicted_label=int(predicted_label),
                predicted_label_name=_label_name(int(predicted_label), resolved_class_names) or f"class_{predicted_label}",
                probabilities=[float(value) for value in probability_array[index].tolist()],
                confidence=uncertainty["confidence"],
                entropy=uncertainty["entropy"],
                normalized_entropy=uncertainty["normalized_entropy"],
                probability_margin=uncertainty["probability_margin"],
                uncertainty_score=uncertainty["uncertainty_score"],
                calibrated_probabilities=list(calibrated.calibrated_probabilities),
                calibrated_probability_score=calibrated.calibrated_probability_score,
                confidence_level=calibrated.confidence_level,
                review_flag=calibrated.review_flag,
                meta=meta,
            )
        )
    return records


def evaluate_oasis_probabilities(
    y_true: list[int],
    probabilities: Any,
    *,
    class_names: tuple[str, ...] | None = None,
    sample_meta: list[dict[str, Any]] | None = None,
    calibration_config: ConfidenceBandConfig | None = None,
    decision_threshold: float = 0.5,
) -> OASISEvaluationResult:
    """Evaluate OASIS binary predictions from probability scores."""

    resolved_class_names = class_names or build_oasis_class_names()
    prediction_records = build_prediction_records(
        probabilities,
        y_true=y_true,
        class_names=resolved_class_names,
        sample_meta=sample_meta,
        calibration_config=calibration_config,
        decision_threshold=decision_threshold,
    )
    y_pred = [record.predicted_label for record in prediction_records]
    positive_scores = [
        record.probabilities[1] if len(record.probabilities) > 1 else 0.0
        for record in prediction_records
    ]
    metrics = compute_binary_classification_metrics(
        y_true,
        y_pred,
        y_score=positive_scores,
        positive_label=1,
        negative_label=0,
    )
    metrics["threshold"] = float(decision_threshold)
    if prediction_records:
        metrics["mean_confidence"] = float(np.mean([record.confidence for record in prediction_records]))
        metrics["mean_calibrated_confidence"] = float(
            np.mean([max(record.calibrated_probabilities or record.probabilities) for record in prediction_records])
        )
        metrics["mean_normalized_entropy"] = float(
            np.mean([record.normalized_entropy for record in prediction_records])
        )
        metrics["mean_uncertainty_score"] = float(np.mean([record.uncertainty_score for record in prediction_records]))
        metrics["confidence_level_counts"] = {
            "high": int(sum(1 for record in prediction_records if record.confidence_level == "high")),
            "medium": int(sum(1 for record in prediction_records if record.confidence_level == "medium")),
            "low": int(sum(1 for record in prediction_records if record.confidence_level == "low")),
        }
        metrics["review_required_count"] = int(sum(1 for record in prediction_records if record.review_flag))
    else:
        metrics["mean_confidence"] = 0.0
        metrics["mean_calibrated_confidence"] = 0.0
        metrics["mean_normalized_entropy"] = 0.0
        metrics["mean_uncertainty_score"] = 0.0
        metrics["confidence_level_counts"] = {"high": 0, "medium": 0, "low": 0}
        metrics["review_required_count"] = 0

    return OASISEvaluationResult(
        dataset="oasis1",
        dataset_type="3d_volumes",
        class_names=resolved_class_names,
        metrics=metrics,
        predictions=prediction_records,
        notes=[
            "Probabilities are decision-support scores and are not standalone diagnoses.",
            "Uncertainty uses entropy, probability margin, and 1 - max probability as baseline indicators.",
            "Calibrated confidence uses optional temperature scaling with explicit review bands.",
        ],
    )


def _coerce_labels(raw_labels: object, torch: object, device: str) -> object:
    """Convert dataloader labels to a torch long tensor."""

    if hasattr(raw_labels, "to"):
        return raw_labels.to(device).long()
    return torch.as_tensor(raw_labels, device=device).long()


def evaluate_oasis_model_on_loader(
    *,
    model: object,
    loader: object,
    device: str = "cpu",
    class_names: tuple[str, ...] | None = None,
    max_batches: int | None = None,
    calibration_config: ConfidenceBandConfig | None = None,
    decision_threshold: float = 0.5,
) -> OASISEvaluationResult:
    """Evaluate a MONAI/PyTorch OASIS classifier on a dataloader."""

    torch = _load_torch_symbols()["torch"]
    model = model.to(device)
    model.eval()

    all_probabilities: list[list[float]] = []
    all_labels: list[int] = []
    sample_meta: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device)
            labels = _coerce_labels(batch["label"], torch, device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            all_probabilities.extend(probabilities.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

            batch_size = int(labels.numel())
            for item_index in range(batch_size):
                meta: dict[str, Any] = {"batch_index": batch_index, "batch_item_index": item_index}
                for key in ("subject_id", "session_id", "image_path"):
                    if key in batch:
                        value = batch[key]
                        if isinstance(value, (list, tuple)) and item_index < len(value):
                            meta[key] = value[item_index]
                        elif item_index == 0:
                            meta[key] = str(value)
                sample_meta.append(meta)

    if not all_labels:
        raise ValueError("No OASIS samples were evaluated. Check the dataloader or max_batches setting.")

    return evaluate_oasis_probabilities(
        [int(label) for label in all_labels],
        all_probabilities,
        class_names=class_names,
        sample_meta=sample_meta,
        calibration_config=calibration_config,
        decision_threshold=decision_threshold,
    )


def save_oasis_evaluation_report(
    result: OASISEvaluationResult,
    *,
    settings: AppSettings | None = None,
    run_name: str = "oasis_baseline_evaluation",
) -> Path:
    """Save an OASIS evaluation payload under outputs/metrics."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "metrics")
    output_path = output_root / f"{run_name}.json"
    output_path.write_text(json.dumps(result.to_payload(), indent=2), encoding="utf-8")
    return output_path
