"""Tests for OASIS-2 raw source discovery and inventory building."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.data.oasis2 import (
    build_oasis2_raw_inventory,
    build_oasis2_session_manifest,
    resolve_oasis2_source_layout,
)


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for OASIS-2 inventory tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path / "workspace",
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path / "kaggle",
        oasis_source_root=tmp_path / "OASIS",
    )


def test_resolve_oasis2_source_layout_detects_split_part_roots(tmp_path: Path) -> None:
    """The layout resolver should detect OAS2 raw part folders automatically."""

    settings = _build_settings(tmp_path)
    part1 = settings.collection_root / "OAS2_RAW_PART1"
    part2 = settings.collection_root / "OAS2_RAW_PART2"
    part1.mkdir(parents=True, exist_ok=True)
    part2.mkdir(parents=True, exist_ok=True)

    layout = resolve_oasis2_source_layout(settings)

    assert layout.source_root == settings.collection_root.resolve()
    assert layout.source_resolution == "auto_detected_part_roots"
    assert layout.inspection_roots == (part1.resolve(), part2.resolve())


def test_build_oasis2_raw_inventory_creates_session_rows_and_summary(tmp_path: Path) -> None:
    """The raw inventory should index split-part OASIS-2 session volumes honestly."""

    settings = _build_settings(tmp_path)
    part1 = settings.collection_root / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW"
    part2 = settings.collection_root / "OAS2_RAW_PART2" / "OAS2_0001_MR2" / "RAW"
    part1.mkdir(parents=True, exist_ok=True)
    part2.mkdir(parents=True, exist_ok=True)

    (part1 / "mpr-1.nifti.hdr").write_bytes(b"hdr")
    (part1 / "mpr-1.nifti.img").write_bytes(b"img")
    (part2 / "mpr-2.nifti.hdr").write_bytes(b"hdr")
    (part2 / "mpr-2.nifti.img").write_bytes(b"img")

    result = build_oasis2_raw_inventory(settings=settings)

    assert result.inventory_path.exists()
    assert result.summary_path.exists()
    assert result.session_row_count == 2
    assert result.unique_subject_count == 1
    assert result.unique_session_count == 2

    frame = pd.read_csv(result.inventory_path)
    assert sorted(frame["session_id"].tolist()) == ["OAS2_0001_MR1", "OAS2_0001_MR2"]
    assert set(frame["volume_format"].tolist()) == {"analyze_pair"}

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["longitudinal_subject_count"] == 1
    assert summary["part_counts"]["OAS2_RAW_PART1"] == 1
    assert summary["part_counts"]["OAS2_RAW_PART2"] == 1


def test_build_oasis2_session_manifest_selects_one_acquisition_per_session(tmp_path: Path) -> None:
    """The session manifest should choose one representative acquisition per session."""

    settings = _build_settings(tmp_path)
    inventory_path = settings.data_root / "interim" / "oasis2_raw_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_frame = pd.DataFrame(
        [
            {
                "image": str(tmp_path / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-2.nifti.hdr"),
                "paired_image": str(tmp_path / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-2.nifti.img"),
                "label": None,
                "label_name": None,
                "subject_id": "OAS2_0001",
                "session_id": "OAS2_0001_MR1",
                "visit_number": 1,
                "scan_timestamp": None,
                "dataset": "oasis2_raw",
                "dataset_type": "3d_volumes",
                "source_part": "OAS2_RAW_PART1",
                "acquisition_id": "mpr-2",
                "volume_format": "analyze_pair",
                "meta": json.dumps({"source_part": "OAS2_RAW_PART1"}),
            },
            {
                "image": str(tmp_path / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"),
                "paired_image": str(tmp_path / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.img"),
                "label": None,
                "label_name": None,
                "subject_id": "OAS2_0001",
                "session_id": "OAS2_0001_MR1",
                "visit_number": 1,
                "scan_timestamp": None,
                "dataset": "oasis2_raw",
                "dataset_type": "3d_volumes",
                "source_part": "OAS2_RAW_PART1",
                "acquisition_id": "mpr-1",
                "volume_format": "analyze_pair",
                "meta": json.dumps({"source_part": "OAS2_RAW_PART1"}),
            },
            {
                "image": str(tmp_path / "OAS2_RAW_PART2" / "OAS2_0001_MR2" / "RAW" / "mpr-3.nifti.hdr"),
                "paired_image": str(tmp_path / "OAS2_RAW_PART2" / "OAS2_0001_MR2" / "RAW" / "mpr-3.nifti.img"),
                "label": None,
                "label_name": None,
                "subject_id": "OAS2_0001",
                "session_id": "OAS2_0001_MR2",
                "visit_number": 2,
                "scan_timestamp": None,
                "dataset": "oasis2_raw",
                "dataset_type": "3d_volumes",
                "source_part": "OAS2_RAW_PART2",
                "acquisition_id": "mpr-3",
                "volume_format": "analyze_pair",
                "meta": json.dumps({"source_part": "OAS2_RAW_PART2"}),
            },
        ]
    )
    inventory_frame.to_csv(inventory_path, index=False)

    result = build_oasis2_session_manifest(
        settings=settings,
        inventory_path=inventory_path,
        source_root=tmp_path,
    )

    manifest = pd.read_csv(result.manifest_path)
    longitudinal = pd.read_csv(result.longitudinal_records_path)
    subject_summary = pd.read_csv(result.subject_summary_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert len(manifest) == 2
    assert manifest.loc[0, "image"].endswith("mpr-1.nifti.hdr")
    assert list(longitudinal["visit_order"]) == [1, 2]
    assert subject_summary.loc[0, "session_count"] == 2
    assert summary["longitudinal_subject_count"] == 1
