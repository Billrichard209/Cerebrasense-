"""Tests for external-cohort manifest building helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.external_cohort import summarize_external_cohort_manifest
from src.data.external_manifest_builder import (
    ExternalManifestBuildError,
    ExternalManifestBuilderConfig,
    build_external_cohort_manifest,
)


def test_build_external_cohort_manifest_from_metadata_and_validate(tmp_path: Path) -> None:
    """The builder should create a usable external manifest from images plus metadata."""

    images_root = tmp_path / "external_images"
    images_root.mkdir()
    scan_a = images_root / "subject_0001_scan.nii.gz"
    scan_b = images_root / "subject_0002_scan.nii.gz"
    scan_a.write_bytes(b"nii-a")
    scan_b.write_bytes(b"nii-b")

    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "filename": "subject_0001_scan.nii.gz",
                "label": 0,
                "label_name": "control_like",
                "subject_id": "SUBJ-0001",
                "session_id": "SUBJ-0001_MR1",
                "scan_timestamp": "2024-01-01",
                "site": "site_a",
                "scanner": "3T",
            },
            {
                "filename": "subject_0002_scan.nii.gz",
                "label": 1,
                "label_name": "ad_like",
                "subject_id": "SUBJ-0002",
                "session_id": "SUBJ-0002_MR1",
                "scan_timestamp": "2024-01-08",
                "site": "site_b",
                "scanner": "3T",
            },
        ]
    ).to_csv(metadata_path, index=False)

    output_path = tmp_path / "external_manifest.csv"
    result = build_external_cohort_manifest(
        ExternalManifestBuilderConfig(
            images_root=images_root,
            dataset_name="adni_pilot",
            output_path=output_path,
            metadata_csv_path=metadata_path,
            label_column="label",
            label_name_column="label_name",
            subject_id_column="subject_id",
            session_id_column="session_id",
            scan_timestamp_column="scan_timestamp",
            meta_columns=("site", "scanner"),
            require_labels=True,
        )
    )

    assert result.manifest_path.exists()
    assert result.report_path.exists()
    assert result.row_count == 2
    assert result.matched_image_count == 2
    assert result.unmatched_image_count == 0

    frame = pd.read_csv(result.manifest_path)
    assert frame["dataset"].unique().tolist() == ["adni_pilot"]
    assert frame["dataset_type"].unique().tolist() == ["3d_volumes"]
    assert set(frame["label"].tolist()) == {0, 1}
    meta_payload = json.loads(frame.loc[0, "meta"])
    assert meta_payload["site"] == "site_a"
    assert meta_payload["scanner"] == "3T"

    summary = summarize_external_cohort_manifest(
        result.manifest_path,
        require_labels=True,
        expected_dataset_type="3d_volumes",
    )
    assert summary.dataset_name == "adni_pilot"
    assert summary.sample_count == 2
    assert summary.subject_count == 2


def test_build_external_cohort_manifest_rejects_ambiguous_filename_match(tmp_path: Path) -> None:
    """The builder should fail loudly when a metadata row could match multiple image files."""

    images_root = tmp_path / "external_images"
    (images_root / "site_a").mkdir(parents=True)
    (images_root / "site_b").mkdir(parents=True)
    (images_root / "site_a" / "scan.nii.gz").write_bytes(b"a")
    (images_root / "site_b" / "scan.nii.gz").write_bytes(b"b")

    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame([{"filename": "scan.nii.gz", "label": 0}]).to_csv(metadata_path, index=False)

    with pytest.raises(ExternalManifestBuildError):
        build_external_cohort_manifest(
            ExternalManifestBuilderConfig(
                images_root=images_root,
                dataset_name="adni_pilot",
                output_path=tmp_path / "manifest.csv",
                metadata_csv_path=metadata_path,
                label_column="label",
                require_labels=True,
            )
        )


def test_build_external_cohort_manifest_without_metadata_warns_about_missing_labels(tmp_path: Path) -> None:
    """Building directly from image discovery should stay honest about missing evaluation labels."""

    images_root = tmp_path / "external_images"
    images_root.mkdir()
    (images_root / "scan_a.nii.gz").write_bytes(b"a")

    result = build_external_cohort_manifest(
        ExternalManifestBuilderConfig(
            images_root=images_root,
            dataset_name="aibl_pilot",
            output_path=tmp_path / "manifest.csv",
        )
    )

    assert result.row_count == 1
    assert any("not labeled evaluation yet" in warning for warning in result.warnings)
    frame = pd.read_csv(result.manifest_path)
    assert pd.isna(frame.loc[0, "label"])
