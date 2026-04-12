"""Tests for OASIS threshold calibration utilities."""

from __future__ import annotations

import pandas as pd

from src.evaluation.thresholds import calibrate_binary_threshold, evaluate_threshold_frame


def test_evaluate_threshold_frame_computes_thresholded_metrics() -> None:
    """Threshold evaluation should convert probabilities into labels."""

    frame = pd.DataFrame({"true_label": [0, 1, 1], "probability_class_1": [0.2, 0.4, 0.9]})
    metrics = evaluate_threshold_frame(frame, 0.5)

    assert metrics["confusion_counts"]["true_positive"] == 1
    assert metrics["confusion_counts"]["false_negative"] == 1


def test_calibrate_binary_threshold_writes_reports(tmp_path) -> None:
    """Calibration should select a validation threshold and apply it to test predictions."""

    val_path = tmp_path / "val_predictions.csv"
    test_path = tmp_path / "test_predictions.csv"
    pd.DataFrame(
        {
            "true_label": [0, 0, 1, 1],
            "probability_class_1": [0.1, 0.45, 0.5, 0.9],
        }
    ).to_csv(val_path, index=False)
    pd.DataFrame(
        {
            "true_label": [0, 1],
            "probability_class_1": [0.4, 0.7],
        }
    ).to_csv(test_path, index=False)

    result = calibrate_binary_threshold(
        validation_predictions_path=val_path,
        test_predictions_path=test_path,
        output_dir=tmp_path / "calibration",
        selection_metric="f1",
        threshold_step=0.05,
    )

    assert result.calibration_report_path.exists()
    assert result.threshold_grid_path.exists()
    assert result.test_metrics_path is not None
    assert result.test_metrics_path.exists()
    assert 0.0 <= result.threshold <= 1.0
