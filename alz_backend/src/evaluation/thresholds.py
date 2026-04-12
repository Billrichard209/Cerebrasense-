"""Threshold calibration utilities for binary OASIS model outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import compute_binary_classification_metrics, threshold_binary_scores
from src.utils.io_utils import ensure_directory


@dataclass(slots=True)
class ThresholdCalibrationResult:
    """Result of selecting a decision threshold from validation predictions."""

    threshold: float
    selection_metric: str
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None
    output_dir: Path
    calibration_report_path: Path
    threshold_grid_path: Path
    test_metrics_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe calibration summary."""

        payload = asdict(self)
        for key in ("output_dir", "calibration_report_path", "threshold_grid_path", "test_metrics_path"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


def _read_prediction_frame(path: str | Path) -> pd.DataFrame:
    """Read a prediction CSV with true labels and positive-class probabilities."""

    frame = pd.read_csv(path)
    required = {"true_label", "probability_class_1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Prediction CSV is empty: {path}")
    return frame


def _candidate_thresholds(step: float) -> list[float]:
    """Build a stable threshold grid from 0.0 to 1.0 inclusive."""

    if step <= 0 or step > 1:
        raise ValueError(f"Threshold step must be in (0, 1], got {step}")
    values: list[float] = []
    current = 0.0
    while current < 1.0:
        values.append(round(current, 10))
        current += step
    values.append(1.0)
    return values


def evaluate_threshold_frame(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Evaluate one threshold against a prediction frame."""

    y_true = [int(value) for value in frame["true_label"].tolist()]
    y_score = [float(value) for value in frame["probability_class_1"].tolist()]
    y_pred = threshold_binary_scores(y_score, threshold=threshold)
    return compute_binary_classification_metrics(y_true, y_pred, y_score=y_score)


def _score_for_selection(metrics: dict[str, Any], selection_metric: str) -> float:
    """Return a scalar score for threshold selection."""

    if selection_metric == "balanced_accuracy":
        return float((metrics["sensitivity"] + metrics["specificity"]) / 2.0)
    if selection_metric in metrics:
        return float(metrics[selection_metric])
    raise ValueError(
        f"Unsupported selection metric: {selection_metric}. "
        "Use f1, accuracy, sensitivity, specificity, precision, auroc, or balanced_accuracy."
    )


def calibrate_binary_threshold(
    *,
    validation_predictions_path: str | Path,
    test_predictions_path: str | Path | None = None,
    output_dir: str | Path,
    selection_metric: str = "f1",
    threshold_step: float = 0.01,
) -> ThresholdCalibrationResult:
    """Select a threshold on validation predictions and optionally apply it to test predictions."""

    validation_frame = _read_prediction_frame(validation_predictions_path)
    thresholds = _candidate_thresholds(threshold_step)

    rows: list[dict[str, Any]] = []
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    best_score = float("-inf")

    for threshold in thresholds:
        metrics = evaluate_threshold_frame(validation_frame, threshold)
        score = _score_for_selection(metrics, selection_metric)
        rows.append(
            {
                "threshold": threshold,
                "selection_score": score,
                "balanced_accuracy": (metrics["sensitivity"] + metrics["specificity"]) / 2.0,
                **{key: value for key, value in metrics.items() if key != "confusion_counts"},
                **{f"confusion_{key}": value for key, value in metrics["confusion_counts"].items()},
            }
        )
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    if best_metrics is None:
        raise ValueError("Could not select a threshold from validation predictions.")

    test_metrics: dict[str, Any] | None = None
    if test_predictions_path is not None:
        test_metrics = evaluate_threshold_frame(_read_prediction_frame(test_predictions_path), best_threshold)

    resolved_output_dir = ensure_directory(output_dir)
    threshold_grid_path = resolved_output_dir / "threshold_grid.csv"
    calibration_report_path = resolved_output_dir / "threshold_calibration.json"
    test_metrics_path = resolved_output_dir / "test_metrics_at_calibrated_threshold.json" if test_metrics else None

    pd.DataFrame(rows).to_csv(threshold_grid_path, index=False)
    payload = {
        "threshold": best_threshold,
        "selection_metric": selection_metric,
        "validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "notes": [
            "Threshold was selected on validation predictions only.",
            "Test metrics are reported after threshold selection and should not be used to tune the threshold.",
            "This is a research decision-support calibration artifact, not clinical validation.",
        ],
    }
    calibration_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if test_metrics_path is not None and test_metrics is not None:
        test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    return ThresholdCalibrationResult(
        threshold=best_threshold,
        selection_metric=selection_metric,
        validation_metrics=best_metrics,
        test_metrics=test_metrics,
        output_dir=resolved_output_dir,
        calibration_report_path=calibration_report_path,
        threshold_grid_path=threshold_grid_path,
        test_metrics_path=test_metrics_path,
    )
