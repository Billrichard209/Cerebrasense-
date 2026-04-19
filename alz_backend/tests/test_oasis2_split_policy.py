"""Tests for the planning-only OASIS-2 subject-safe split preview."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.data.oasis2_metadata import build_oasis2_metadata_template
from src.data.oasis2_split_policy import build_oasis2_subject_safe_split_plan


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


def _write_manifest(settings: AppSettings, dataset_root: Path) -> None:
    manifest_path = settings.data_root / "interim" / "oasis2_session_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    image_a = dataset_root / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"
    image_b = dataset_root / "OAS2_RAW_PART1" / "OAS2_0002_MR1" / "RAW" / "mpr-1.nifti.hdr"
    image_a.parent.mkdir(parents=True, exist_ok=True)
    image_b.parent.mkdir(parents=True, exist_ok=True)
    image_a.write_bytes(b"hdr")
    image_b.write_bytes(b"hdr")
    manifest_path.write_text(
        "\n".join(
            [
                "image,label,label_name,subject_id,session_id,visit_number,scan_timestamp,dataset,dataset_type,meta",
                f"{image_a},,,OAS2_0001,OAS2_0001_MR1,1,,oasis2_raw,3d_volumes,\"{{}}\"",
                f"{image_b},,,OAS2_0002,OAS2_0002_MR1,1,,oasis2_raw,3d_volumes,\"{{}}\"",
            ]
        ),
        encoding="utf-8",
    )


def test_build_oasis2_subject_safe_split_plan_from_metadata(tmp_path: Path) -> None:
    """The split preview should assign stable subject-safe buckets from the metadata template."""

    settings = _settings(tmp_path)
    _write_manifest(settings, tmp_path / "dataset")
    template_result = build_oasis2_metadata_template(settings=settings)
    frame = pd.read_csv(template_result.template_path)
    frame["diagnosis_label"] = [0, None]
    frame["diagnosis_label_name"] = ["nondemented_like", None]
    frame.to_csv(template_result.template_path, index=False)

    summary = build_oasis2_subject_safe_split_plan(settings=settings, bucket_count=4)

    plan_frame = pd.read_csv(summary.plan_csv_path)
    assert summary.subject_count == 2
    assert summary.bucket_count == 4
    assert "subject_safe_bucket" in plan_frame.columns
    assert "future_role_hint" in plan_frame.columns
    assert json.loads(Path(settings.data_root / "interim" / "oasis2_subject_safe_split_plan_summary.json").read_text(encoding="utf-8"))["bucket_count"] == 4
