"""Tests for prediction error analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.evaluation.error_analysis import ErrorAnalysisConfig, analyze_prediction_errors


def _settings(tmp_path: Path) -> AppSettings:
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


def test_analyze_prediction_errors_saves_summary_and_slice_artifacts(tmp_path: Path) -> None:
    """Error analysis should count FP/FN cases and save representative slice images."""

    nib = pytest.importorskip("nibabel")
    pytest.importorskip("matplotlib")
    settings = _settings(tmp_path)
    image_path = tmp_path / "scan_001.nii.gz"
    nii = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), affine=np.eye(4))
    nib.save(nii, str(image_path))

    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"sample_id": "s1", "subject_id": "OAS1_0001", "source_path": str(image_path), "true_label": 0, "predicted_label": 1, "probability": 0.81, "confidence": 0.81},
            {"sample_id": "s2", "subject_id": "OAS1_0002", "source_path": str(image_path), "true_label": 1, "predicted_label": 0, "probability": 0.24, "confidence": 0.76},
        ]
    ).to_csv(predictions_path, index=False)

    result = analyze_prediction_errors(
        ErrorAnalysisConfig(predictions_csv_path=predictions_path, output_name="unit_errors"),
        settings=settings,
    )

    assert result.paths.summary_json_path.exists()
    assert result.paths.misclassifications_csv_path.exists()
    summary = json.loads(result.paths.summary_json_path.read_text(encoding="utf-8"))
    assert summary["total_false_positives"] == 1
    assert summary["total_false_negatives"] == 1
    assert summary["average_confidence_of_errors"] > 0.0
    assert (result.paths.output_root / "false_positive" / "s1" / "axial_mid.png").exists()
    assert (result.paths.output_root / "false_negative" / "s2" / "sagittal_mid.png").exists()
