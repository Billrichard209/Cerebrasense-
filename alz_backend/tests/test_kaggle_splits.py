"""Tests for leakage-aware Kaggle split generation."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from src.configs.runtime import AppSettings
from src.data.kaggle_alz import build_kaggle_manifest
from src.data.kaggle_splits import build_kaggle_splits


def _build_settings(tmp_path: Path, kaggle_source_root: Path) -> AppSettings:
    """Create isolated app settings for Kaggle split tests."""

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
    """Write a tiny grayscale JPG for Kaggle split tests."""

    array = np.full((16, 16), value, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def test_augmented_rows_are_forced_into_train_and_subset_stays_visible(tmp_path: Path) -> None:
    """Augmented Kaggle subsets should stay in train while original provenance remains explicit."""

    source_root = tmp_path / "kaggle_source"
    original_root = source_root / "OriginalDataset"
    augmented_root = source_root / "AugmentedAlzheimerDataset"

    for class_name, value in (("NonDemented", 15), ("MildDemented", 220)):
        (original_root / class_name).mkdir(parents=True, exist_ok=True)
        (augmented_root / class_name).mkdir(parents=True, exist_ok=True)
        for index in range(8):
            _write_test_jpg(original_root / class_name / f"orig_{index:02d}.jpg", value + index)
        for index in range(2):
            _write_test_jpg(augmented_root / class_name / f"aug_{index:02d}.jpg", value + 30 + index)

    settings = _build_settings(tmp_path, source_root)
    build_kaggle_manifest(settings=settings, output_format="csv")
    result = build_kaggle_splits(
        settings=settings,
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        random_state=7,
    )

    assignments = pd.read_csv(result.split_assignments_path)
    augmented_rows = assignments.loc[assignments["subset_source"] == "AugmentedAlzheimerDataset"]
    assert not augmented_rows.empty
    assert augmented_rows["split"].eq("train").all()
    assert augmented_rows["assignment_reason"].eq("augmented_train_only").all()

    validation_like = assignments.loc[assignments["split"].isin(["val", "test"])]
    assert validation_like["subset_source"].eq("OriginalDataset").all()

    train_manifest = pd.read_csv(result.train_manifest_path)
    assert "subset_source" in train_manifest.columns
    assert "group_id" in train_manifest.columns


def test_subject_ids_keep_multiscan_metadata_rows_together(tmp_path: Path) -> None:
    """Metadata-driven Kaggle rows with subject IDs should never leak across splits."""

    source_root = tmp_path / "kaggle_source"
    volume_root = source_root / "volumes"
    volume_root.mkdir(parents=True)
    affine = np.eye(4)

    rows = []
    for label_name, subject_indices, fill_value in (
        ("Control", range(1, 5), 0.0),
        ("AD_like", range(5, 9), 1.0),
    ):
        for subject_index in subject_indices:
            subject_id = f"sub-{subject_index:03d}"
            for visit in range(2):
                image_name = f"{subject_id}_visit{visit + 1}.nii.gz"
                image_path = volume_root / image_name
                nib.save(
                    nib.Nifti1Image(np.full((6, 6, 6), fill_value + visit, dtype=np.float32), affine),
                    image_path,
                )
                rows.append(
                    {
                        "image_path": f"volumes/{image_name}",
                        "label": label_name,
                        "subject_id": subject_id,
                        "scan_timestamp": f"2024-03-{subject_index + visit:02d}",
                    }
                )

    pd.DataFrame(rows).to_csv(source_root / "metadata.csv", index=False)

    settings = _build_settings(tmp_path, source_root)
    build_kaggle_manifest(settings=settings, output_format="csv")
    result = build_kaggle_splits(
        settings=settings,
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        random_state=11,
    )

    assignments = pd.read_csv(result.split_assignments_path)
    subject_split_counts = assignments.groupby("subject_id")["split"].nunique()
    assert subject_split_counts.max() == 1
    assert assignments["group_kind"].eq("subject").all()
