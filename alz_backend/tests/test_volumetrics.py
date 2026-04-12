"""Tests for structural foreground-proxy volumetric analysis."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.volumetrics.measurements import analyze_mri_volume, summarize_volumetrics
from src.volumetrics.service import analyze_oasis_volume


def _write_synthetic_volume(tmp_path: Path) -> Path:
    """Create a simple 3D test volume with known spacing and foreground size."""

    data = np.zeros((10, 12, 14), dtype=np.float32)
    data[1:8, 2:8, 3:9] = 25.0
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    image = nib.Nifti1Image(data, affine)
    output_path = tmp_path / "synthetic_oasis.nii.gz"
    nib.save(image, str(output_path))
    return output_path


def _write_singleton_4d_volume(tmp_path: Path) -> Path:
    """Create a 4D volume with a singleton trailing axis."""

    data = np.zeros((10, 12, 14, 1), dtype=np.float32)
    data[1:8, 2:8, 3:9, 0] = 25.0
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    image = nib.Nifti1Image(data, affine)
    output_path = tmp_path / "synthetic_singleton_4d.nii.gz"
    nib.save(image, str(output_path))
    return output_path


def test_analyze_mri_volume_returns_expected_proxy_measurements(tmp_path: Path) -> None:
    """Synthetic volume geometry should produce predictable foreground-proxy volumes."""

    image_path = _write_synthetic_volume(tmp_path)
    result = analyze_mri_volume(
        image_path,
        dataset="oasis1",
        dataset_type="3d_volumes",
        subject_id="OAS1_9000",
        session_id="OAS1_9000_MR1",
    )

    measurements = summarize_volumetrics(result.measurements)
    expected_foreground_voxels = 7 * 6 * 6
    expected_voxel_volume_mm3 = 8.0
    expected_foreground_volume_mm3 = expected_foreground_voxels * expected_voxel_volume_mm3

    assert result.geometry.shape == (10, 12, 14)
    assert result.geometry.voxel_spacing_mm == (2.0, 2.0, 2.0)
    assert result.foreground_voxels == expected_foreground_voxels
    assert result.foreground_threshold >= 0.0
    assert measurements["foreground_proxy_brain"] == expected_foreground_volume_mm3
    assert measurements["foreground_proxy_bounding_box"] == expected_foreground_volume_mm3
    assert measurements["left_hemisphere_proxy"] == 4 * 6 * 6 * expected_voxel_volume_mm3
    assert measurements["right_hemisphere_proxy"] == 3 * 6 * 6 * expected_voxel_volume_mm3
    assert measurements["hemisphere_asymmetry_index"] > 0.0
    assert any("decision support" in note.lower() for note in result.notes)


def test_analyze_oasis_volume_accepts_direct_path_requests(tmp_path: Path) -> None:
    """Service-level OASIS analysis should work directly from a path."""

    image_path = _write_synthetic_volume(tmp_path)
    result = analyze_oasis_volume(
        image_path=image_path,
        subject_id="OAS1_9001",
        session_id="OAS1_9001_MR1",
        scan_timestamp="2001-01-01",
    )

    assert result.dataset == "oasis1"
    assert result.dataset_type == "3d_volumes"
    assert result.subject_id == "OAS1_9001"
    assert result.session_id == "OAS1_9001_MR1"
    assert result.scan_timestamp == "2001-01-01"


def test_analyze_mri_volume_accepts_singleton_4d_input(tmp_path: Path) -> None:
    """A singleton channel/time dimension should be squeezed safely for OASIS-like inputs."""

    image_path = _write_singleton_4d_volume(tmp_path)
    result = analyze_mri_volume(image_path, dataset="oasis1", dataset_type="3d_volumes")
    assert result.geometry.shape == (10, 12, 14)
    assert result.foreground_voxels == 7 * 6 * 6


def test_oasis_volumetric_api_endpoint_returns_structured_payload(tmp_path: Path) -> None:
    """The FastAPI endpoint should expose the volumetric baseline report."""

    image_path = _write_synthetic_volume(tmp_path)
    client = TestClient(create_app())

    response = client.get(
        "/volumetrics/oasis",
        params={
            "image_path": str(image_path),
            "subject_id": "OAS1_9002",
            "session_id": "OAS1_9002_MR1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "oasis1"
    assert payload["dataset_type"] == "3d_volumes"
    assert payload["subject_id"] == "OAS1_9002"
    assert payload["session_id"] == "OAS1_9002_MR1"
    assert payload["geometry"]["shape"] == [10, 12, 14]
    assert payload["measurements"][0]["region_name"] == "foreground_proxy_brain"
