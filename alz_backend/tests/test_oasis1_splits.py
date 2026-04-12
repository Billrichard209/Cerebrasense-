"""Tests for subject-safe OASIS-1 split generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.data.oasis1_splits import (
    OASIS1SplitError,
    apply_subject_splits,
    build_oasis1_splits,
    build_subject_table,
)


def _make_manifest_frame() -> pd.DataFrame:
    """Create a balanced synthetic manifest for split tests."""

    rows = []
    for index in range(20):
        label = 0 if index < 10 else 1
        rows.append(
            {
                "image": f"/tmp/image_{index}.hdr",
                "label": label,
                "label_name": "nondemented" if label == 0 else "demented",
                "subject_id": f"OAS1_{index:04d}",
                "scan_timestamp": None,
                "dataset": "oasis1",
                "meta": "{}",
            }
        )
    return pd.DataFrame(rows)


def test_build_subject_table_rejects_inconsistent_subject_labels() -> None:
    """Subjects with conflicting labels should fail before splitting."""

    frame = pd.DataFrame(
        [
            {
                "image": "/tmp/a.hdr",
                "label": 0,
                "label_name": "nondemented",
                "subject_id": "OAS1_0001",
                "scan_timestamp": None,
                "dataset": "oasis1",
                "meta": "{}",
            },
            {
                "image": "/tmp/b.hdr",
                "label": 1,
                "label_name": "demented",
                "subject_id": "OAS1_0001",
                "scan_timestamp": None,
                "dataset": "oasis1",
                "meta": "{}",
            },
        ]
    )
    with pytest.raises(OASIS1SplitError):
        build_subject_table(frame)


def test_build_oasis1_splits_creates_subject_safe_partitions(tmp_path: Path) -> None:
    """The split builder should create non-overlapping subject partitions."""

    manifest_frame = _make_manifest_frame()
    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    interim_root = data_root / "interim"
    outputs_root = project_root / "outputs"
    interim_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)

    manifest_path = interim_root / "oasis1_manifest.csv"
    manifest_frame.to_csv(manifest_path, index=False)

    settings = AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=project_root.parent,
        oasis_source_root=project_root.parent,
    )

    result = build_oasis1_splits(settings=settings, random_state=7)
    assert result.split_assignments_path.exists()
    assert result.train_manifest_path.exists()
    assert result.val_manifest_path.exists()
    assert result.test_manifest_path.exists()

    assignments = pd.read_csv(result.split_assignments_path)
    assert assignments["subject_id"].nunique() == len(assignments)

    train_subjects = set(assignments.loc[assignments["split"] == "train", "subject_id"])
    val_subjects = set(assignments.loc[assignments["split"] == "val", "subject_id"])
    test_subjects = set(assignments.loc[assignments["split"] == "test", "subject_id"])

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_apply_subject_splits_requires_complete_assignments() -> None:
    """Every manifest row should receive a split assignment."""

    manifest_frame = _make_manifest_frame().iloc[:2].copy()
    assignments = pd.DataFrame([{"subject_id": "OAS1_0000", "label": 0, "label_name": "nondemented", "split": "train"}])
    with pytest.raises(OASIS1SplitError):
        apply_subject_splits(manifest_frame, assignments)
