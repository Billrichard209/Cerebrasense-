"""Tests for separate external-cohort evaluation preparation and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.data.external_cohort import (
    ExternalCohortManifestError,
    ExternalCohortManifestSummary,
    summarize_external_cohort_manifest,
)
from src.evaluation.external_cohort import (
    ExternalCohortEvaluationConfig,
    evaluate_external_cohort,
)
from src.models.factory import OASISModelConfig


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for external evaluation tests."""

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


def test_summarize_external_cohort_manifest_requires_separate_named_3d_cohort(
    tmp_path: Path,
) -> None:
    """External manifest summaries should stay separate from internal dataset names."""

    image_path = tmp_path / "sample_a.nii.gz"
    image_path.write_bytes(b"nii")
    manifest_path = tmp_path / "external_manifest.csv"
    pd.DataFrame(
        [
            {
                "image": str(image_path),
                "dataset": "local_pilot_cohort",
                "dataset_type": "3d_volumes",
                "label": 1,
                "label_name": "possible_ad_like",
                "subject_id": "SUBJ-001",
            }
        ]
    ).to_csv(manifest_path, index=False)

    summary = summarize_external_cohort_manifest(
        manifest_path,
        require_labels=True,
        expected_dataset_type="3d_volumes",
    )

    assert summary.dataset_name == "local_pilot_cohort"
    assert summary.dataset_type == "3d_volumes"
    assert summary.sample_count == 1
    assert summary.subject_count == 1
    assert summary.label_distribution == {"1": 1}
    assert summary.source_label_name_distribution == {"possible_ad_like": 1}

    reserved_manifest_path = tmp_path / "reserved_manifest.csv"
    pd.DataFrame(
        [
            {
                "image": str(image_path),
                "dataset": "oasis1",
                "dataset_type": "3d_volumes",
                "label": 0,
            }
        ]
    ).to_csv(reserved_manifest_path, index=False)

    with pytest.raises(ExternalCohortManifestError):
        summarize_external_cohort_manifest(
            reserved_manifest_path,
            require_labels=True,
            expected_dataset_type="3d_volumes",
        )


def test_evaluate_external_cohort_writes_separate_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External evaluation should save a separate report bundle under outputs/evaluations/external."""

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
            "subject_id": ["EXT_0001", "EXT_0002"],
            "session_id": ["EXT_0001_MR1", "EXT_0002_MR1"],
            "image_path": ["external_scan_001.nii.gz", "external_scan_002.nii.gz"],
            "scan_timestamp": ["2024-01-01", "2024-01-08"],
            "label_name": ["control_like", "ad_like"],
            "dataset": ["local_pilot_cohort", "local_pilot_cohort"],
            "dataset_type": ["3d_volumes", "3d_volumes"],
        }
    ]
    manifest_summary = ExternalCohortManifestSummary(
        manifest_path=str(tmp_path / "external_manifest.csv"),
        manifest_hash_sha256="abc123",
        dataset_name="local_pilot_cohort",
        dataset_type="3d_volumes",
        sample_count=2,
        labeled_sample_count=2,
        subject_count=2,
        subject_id_available=True,
        label_distribution={"0": 1, "1": 1},
        source_label_name_distribution={"control_like": 1, "ad_like": 1},
    )

    monkeypatch.setattr(
        "src.evaluation.external_cohort._build_external_eval_loader",
        lambda _cfg: (loader, manifest_summary),
    )
    monkeypatch.setattr(
        "src.evaluation.external_cohort.build_model",
        lambda _cfg: _build_tiny_classifier(torch),
    )
    monkeypatch.setattr(
        "src.evaluation.external_cohort.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 1, 2, 2)),
    )

    result = evaluate_external_cohort(
        ExternalCohortEvaluationConfig(
            manifest_path=tmp_path / "external_manifest.csv",
            checkpoint_path=checkpoint_path,
            output_name="unit_external_eval",
            threshold=0.45,
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
    assert result.paths.manifest_summary_path.exists()
    assert "evaluations" in str(result.paths.output_root)
    assert "external" in str(result.paths.output_root)
    assert result.metrics["sample_count"] == 2
    assert result.metrics["threshold"] == 0.45
    assert result.metrics["dataset_name"] == "local_pilot_cohort"

    predictions = pd.read_csv(result.paths.predictions_csv_path)
    assert {
        "dataset_name",
        "subject_id",
        "source_path",
        "true_label",
        "predicted_label",
        "probability",
        "confidence_level",
        "review_flag",
    } <= set(predictions.columns)
    assert predictions["dataset_name"].unique().tolist() == ["local_pilot_cohort"]
    assert predictions["source_label_name"].tolist() == ["control_like", "ad_like"]

    manifest_payload = json.loads(result.paths.manifest_summary_path.read_text(encoding="utf-8"))
    assert manifest_payload["dataset_name"] == "local_pilot_cohort"
    report_payload = json.loads(result.paths.report_json_path.read_text(encoding="utf-8"))
    assert report_payload["dataset_name"] == "local_pilot_cohort"
