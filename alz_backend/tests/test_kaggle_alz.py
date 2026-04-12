"""Tests for the flexible Kaggle Alzheimer adapter and manifest builder."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from src.configs.runtime import AppSettings
from src.data.kaggle_alz import build_kaggle_manifest, detect_kaggle_dataset_organization


def _build_settings(tmp_path: Path, kaggle_source_root: Path) -> AppSettings:
    """Create isolated app settings for manifest tests."""

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
        kaggle_source_root=kaggle_source_root,
        oasis_source_root=project_root.parent / "OASIS",
    )


def _write_test_jpg(path: Path, value: int) -> None:
    """Write a tiny grayscale JPG for class-folder tests."""

    array = np.full((16, 16), value, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def test_detects_class_folder_dataset_and_warns_for_slices(tmp_path: Path) -> None:
    """Class-folder Kaggle datasets should be treated as 2D slices by default."""

    source_root = tmp_path / "kaggle_source"
    original_root = source_root / "OriginalDataset"
    (original_root / "NonDemented").mkdir(parents=True)
    (original_root / "VeryMildDemented").mkdir(parents=True)

    _write_test_jpg(original_root / "NonDemented" / "img_001.jpg", 10)
    _write_test_jpg(original_root / "VeryMildDemented" / "img_002.jpg", 240)

    organization = detect_kaggle_dataset_organization(source_root)
    assert organization.kind == "class_folders"
    assert organization.dataset_type == "2d_slices"

    settings = _build_settings(tmp_path, source_root)
    result = build_kaggle_manifest(settings=settings, output_format="both")

    assert result.manifest_csv_path is not None and result.manifest_csv_path.exists()
    assert result.manifest_jsonl_path is not None and result.manifest_jsonl_path.exists()
    assert result.manifest_row_count == 2
    assert any("slice-based" in warning.lower() for warning in result.warnings)

    manifest_frame = pd.read_csv(result.manifest_csv_path)
    assert manifest_frame["dataset_type"].eq("2d_slices").all()
    assert manifest_frame["label"].isna().all()
    assert set(manifest_frame["label_name"].tolist()) == {"NonDemented", "VeryMildDemented"}

    meta_payload = json.loads(manifest_frame.loc[0, "meta"])
    assert meta_payload["original_class_name"] in {"NonDemented", "VeryMildDemented"}
    assert meta_payload["label_mapping_applied"] is False


def test_applies_explicit_label_remap_without_overwriting_original_label_name(tmp_path: Path) -> None:
    """Explicit remapping should set numeric labels while preserving original class names in metadata."""

    source_root = tmp_path / "kaggle_source"
    augmented_root = source_root / "AugmentedAlzheimerDataset"
    (augmented_root / "NonDemented").mkdir(parents=True)
    (augmented_root / "MildDemented").mkdir(parents=True)

    _write_test_jpg(augmented_root / "NonDemented" / "img_001.jpg", 5)
    _write_test_jpg(augmented_root / "MildDemented" / "img_002.jpg", 220)

    settings = _build_settings(tmp_path, source_root)
    remap = {
        "NonDemented": {"label": 0, "label_name": "control"},
        "MildDemented": {"label": 1, "label_name": "ad_like"},
    }
    result = build_kaggle_manifest(settings=settings, output_format="csv", label_remap=remap)

    manifest_frame = pd.read_csv(result.manifest_csv_path)
    labels = dict(zip(manifest_frame["label_name"], manifest_frame["label"]))
    assert labels["control"] == 0
    assert labels["ad_like"] == 1

    meta_payloads = [json.loads(payload) for payload in manifest_frame["meta"].tolist()]
    assert {payload["original_class_name"] for payload in meta_payloads} == {"NonDemented", "MildDemented"}
    assert all(payload["label_mapping_applied"] is True for payload in meta_payloads)


def test_builds_metadata_driven_3d_manifest(tmp_path: Path) -> None:
    """Metadata-table Kaggle datasets with NIfTI files should be typed as 3D volumes."""

    source_root = tmp_path / "kaggle_source"
    volume_root = source_root / "volumes"
    volume_root.mkdir(parents=True)

    first_volume = volume_root / "case_001.nii.gz"
    second_volume = volume_root / "case_002.nii.gz"
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), affine), first_volume)
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), affine), second_volume)

    metadata_path = source_root / "metadata.csv"
    pd.DataFrame(
        [
            {
                "image_path": "volumes/case_001.nii.gz",
                "label": "Control",
                "subject_id": "sub-001",
                "scan_timestamp": "2024-02-01",
            },
            {
                "image_path": "volumes/case_002.nii.gz",
                "label": "AD_like",
                "subject_id": "sub-002",
                "scan_timestamp": "2024-02-02",
            },
        ]
    ).to_csv(metadata_path, index=False)

    organization = detect_kaggle_dataset_organization(source_root)
    assert organization.kind == "metadata_table"
    assert organization.dataset_type == "3d_volumes"

    settings = _build_settings(tmp_path, source_root)
    result = build_kaggle_manifest(settings=settings, output_format="csv")
    manifest_frame = pd.read_csv(result.manifest_csv_path)

    assert result.manifest_row_count == 2
    assert manifest_frame["dataset_type"].eq("3d_volumes").all()
    assert manifest_frame["subject_id"].tolist() == ["sub-001", "sub-002"]
    assert manifest_frame["scan_timestamp"].tolist() == ["2024-02-01", "2024-02-02"]
    assert manifest_frame["label"].isna().all()
    assert manifest_frame["label_name"].tolist() == ["Control", "AD_like"]
