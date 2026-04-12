"""Tests for subject-level longitudinal structural proxy summaries."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.configs.runtime import AppSettings
from src.longitudinal.structural import build_oasis_structural_longitudinal_summary


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for longitudinal structural tests."""

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


def _write_volume(path: Path, *, x_stop: int) -> None:
    """Write a synthetic MRI volume with a controllable foreground size."""

    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[1:x_stop, 2:7, 3:8] = 10.0
    image = nib.Nifti1Image(data, np.eye(4))
    nib.save(image, str(path))


def _write_longitudinal_manifest(settings: AppSettings, *, single_timepoint: bool = False) -> Path:
    """Create an OASIS-like manifest with one subject and one or two visits."""

    interim_root = settings.data_root / "interim"
    image_root = interim_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    baseline_path = image_root / "OAS1_9000_MR1.nii.gz"
    followup_path = image_root / "OAS1_9000_MR2.nii.gz"
    _write_volume(baseline_path, x_stop=7)
    _write_volume(followup_path, x_stop=8)

    rows = [
        {
            "image": str(followup_path),
            "label": 1,
            "label_name": "demented",
            "subject_id": "OAS1_9000",
            "scan_timestamp": "2002-01-01",
            "dataset": "oasis1",
            "meta": json.dumps({"session_id": "OAS1_9000_MR2"}),
        },
        {
            "image": str(baseline_path),
            "label": 1,
            "label_name": "demented",
            "subject_id": "OAS1_9000",
            "scan_timestamp": "2001-01-01",
            "dataset": "oasis1",
            "meta": json.dumps({"session_id": "OAS1_9000_MR1"}),
        },
    ]
    if single_timepoint:
        rows = rows[:1]
    manifest_path = interim_root / "oasis1_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_oasis_structural_longitudinal_summary_orders_visits_and_computes_change(tmp_path: Path) -> None:
    """Subject visits should be ordered by time and produce change metrics."""

    settings = _build_settings(tmp_path)
    manifest_path = _write_longitudinal_manifest(settings)
    summary = build_oasis_structural_longitudinal_summary(
        "OAS1_9000",
        settings=settings,
        manifest_path=manifest_path,
    )

    assert summary.timepoint_count == 2
    assert [timepoint.session_id for timepoint in summary.timepoints] == ["OAS1_9000_MR1", "OAS1_9000_MR2"]
    assert len(summary.changes) == 1
    assert summary.changes[0].delta_from_baseline["foreground_proxy_brain"] == 25.0
    assert summary.changes[0].percent_change_from_baseline["foreground_proxy_brain"] == pytest.approx(16.6666667)


def test_oasis_structural_longitudinal_summary_warns_for_single_timepoint(tmp_path: Path) -> None:
    """A single available visit should still return a report with an explicit warning."""

    settings = _build_settings(tmp_path)
    manifest_path = _write_longitudinal_manifest(settings, single_timepoint=True)
    summary = build_oasis_structural_longitudinal_summary(
        "OAS1_9000",
        settings=settings,
        manifest_path=manifest_path,
    )

    assert summary.timepoint_count == 1
    assert summary.changes == []
    assert any("only one timepoint" in warning.lower() for warning in summary.warnings)


def test_oasis_longitudinal_volumetric_api_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API should expose the subject-level structural longitudinal endpoint."""

    monkeypatch.setattr(
        "src.api.routers.longitudinal.build_oasis_longitudinal_structural_payload",
        lambda **_: {
            "subject_id": "OAS1_9000",
            "dataset": "oasis1",
            "dataset_type": "3d_volumes",
            "timepoint_count": 1,
            "timepoints": [
                {
                    "subject_id": "OAS1_9000",
                    "session_id": "OAS1_9000_MR1",
                    "visit_order": 1,
                    "scan_timestamp": None,
                    "image": "scan.nii.gz",
                    "metrics": {"foreground_proxy_brain": 100.0},
                    "warnings": [],
                }
            ],
            "changes": [],
            "warnings": ["Only one timepoint is available."],
            "notes": ["Decision-support only."],
        },
    )
    client = TestClient(create_app())
    response = client.get("/longitudinal/oasis/OAS1_9000/volumetrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["subject_id"] == "OAS1_9000"
    assert payload["timepoint_count"] == 1
    assert payload["timepoints"][0]["metrics"]["foreground_proxy_brain"] == 100.0
