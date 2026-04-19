"""Tests for the first dedicated OASIS-2 manifest adapter stub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.configs.runtime import AppSettings
from src.data.oasis2_dataset import (
    build_oasis2_adapter_summary,
    build_oasis2_monai_records,
    load_oasis2_session_manifest,
    save_oasis2_adapter_summary,
)


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path / "kaggle",
        oasis_source_root=tmp_path / "OASIS",
    )


def _write_manifest(settings: AppSettings, image_path: Path) -> Path:
    manifest_path = settings.data_root / "interim" / "oasis2_session_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                "image,label,label_name,subject_id,session_id,visit_number,scan_timestamp,dataset,dataset_type,meta",
                f"{image_path},,,OAS2_0001,OAS2_0001_MR1,1,,oasis2_raw,3d_volumes,\"{{\"\"source_part\"\":\"\"OAS2_RAW_PART1\"\"}}\"",
                f"{image_path},,,OAS2_0001,OAS2_0001_MR2,2,,oasis2_raw,3d_volumes,\"{{\"\"source_part\"\":\"\"OAS2_RAW_PART2\"\"}}\"",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_load_oasis2_session_manifest_and_build_records(tmp_path: Path) -> None:
    """The OASIS-2 adapter stub should load the unlabeled session manifest and preserve visit info."""

    settings = _settings(tmp_path)
    image_path = tmp_path / "dataset" / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"hdr")
    _write_manifest(settings, image_path)

    frame = load_oasis2_session_manifest(settings)
    records = build_oasis2_monai_records(settings)

    assert len(frame) == 2
    assert len(records) == 2
    assert records[0]["label"] is None
    assert records[0]["visit_number"] == 1
    assert records[0]["meta"]["oasis2_adapter_mode"] == "unlabeled_structural_longitudinal_stub"


def test_oasis2_adapter_stub_blocks_require_labels_on_unlabeled_manifest(tmp_path: Path) -> None:
    """The adapter stub should fail clearly if someone tries to require labels too early."""

    settings = _settings(tmp_path)
    image_path = tmp_path / "dataset" / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"hdr")
    _write_manifest(settings, image_path)

    with pytest.raises(ValueError, match="OASIS-2 adapter stub only supports unlabeled onboarding"):
        build_oasis2_monai_records(settings, require_labels=True)


def test_oasis2_adapter_summary_can_be_saved(tmp_path: Path) -> None:
    """The adapter summary should reflect the unlabeled contract and write reports."""

    settings = _settings(tmp_path)
    source_root = tmp_path / "dataset"
    image_path = source_root / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"hdr")
    (source_root / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.img").write_bytes(b"img")
    image_path_2 = source_root / "OAS2_RAW_PART2" / "OAS2_0001_MR2" / "RAW" / "mpr-2.nifti.hdr"
    image_path_2.parent.mkdir(parents=True, exist_ok=True)
    image_path_2.write_bytes(b"hdr")
    (source_root / "OAS2_RAW_PART2" / "OAS2_0001_MR2" / "RAW" / "mpr-2.nifti.img").write_bytes(b"img")
    _write_manifest(settings, image_path)

    summary = build_oasis2_adapter_summary(settings, source_root=source_root)
    json_path, md_path = save_oasis2_adapter_summary(summary, settings, file_stem="unit_oasis2_adapter")

    assert summary.ready_for_supervised_training is False
    assert summary.unlabeled_row_count == 2
    assert json.loads(json_path.read_text(encoding="utf-8"))["adapter_mode"] == "unlabeled_structural_longitudinal_stub"
    assert "Blocked Uses" in md_path.read_text(encoding="utf-8")
