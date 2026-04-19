"""Tests for the OASIS-2 upload bundle builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.data.oasis2_upload_bundle import build_oasis2_upload_bundle


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for OASIS-2 upload bundle tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    config_root = project_root / "configs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path / "workspace",
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path / "kaggle",
        oasis_source_root=tmp_path / "OASIS",
        serving_config_path=config_root / "backend_serving.yaml",
    )


def test_build_oasis2_upload_bundle_materializes_selected_session_files(tmp_path: Path) -> None:
    """The OASIS-2 bundle builder should create a portable unlabeled session subset."""

    settings = _build_settings(tmp_path)
    source_root = settings.collection_root
    hdr_path = source_root / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"
    img_path = hdr_path.with_suffix(".img")
    hdr_path.parent.mkdir(parents=True, exist_ok=True)
    hdr_path.write_text("hdr", encoding="utf-8")
    img_path.write_text("img", encoding="utf-8")

    interim_root = settings.data_root / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)
    manifest_frame = pd.DataFrame(
        [
            {
                "image": str(hdr_path),
                "label": None,
                "label_name": None,
                "subject_id": "OAS2_0001",
                "session_id": "OAS2_0001_MR1",
                "visit_number": 1,
                "scan_timestamp": None,
                "dataset": "oasis2_raw",
                "dataset_type": "3d_volumes",
                "meta": json.dumps(
                    {
                        "paired_image": str(img_path),
                        "selected_acquisition_id": "mpr-1",
                        "acquisition_count": 1,
                    }
                ),
            }
        ]
    )
    manifest_frame.to_csv(interim_root / "oasis2_session_manifest.csv", index=False)
    manifest_frame.rename(columns={"image": "source_path", "visit_number": "visit_order"}).assign(
        record_type="oasis2_session"
    ).to_csv(interim_root / "oasis2_longitudinal_records.csv", index=False)
    pd.DataFrame(
        [{"subject_id": "OAS2_0001", "session_count": 1, "first_visit": 1, "last_visit": 1, "session_ids": "OAS2_0001_MR1"}]
    ).to_csv(interim_root / "oasis2_subject_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "image": str(hdr_path),
                "paired_image": str(img_path),
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
                "meta": "{}",
            }
        ]
    ).to_csv(interim_root / "oasis2_raw_inventory.csv", index=False)
    (interim_root / "oasis2_raw_inventory_dropped_rows.csv").write_text("reason\n", encoding="utf-8")
    (interim_root / "oasis2_raw_inventory_summary.json").write_text("{}", encoding="utf-8")
    (interim_root / "oasis2_session_manifest_summary.json").write_text("{}", encoding="utf-8")
    (interim_root / "oasis2_metadata_template.csv").write_text("subject_id,session_id\nOAS2_0001,OAS2_0001_MR1\n", encoding="utf-8")
    (interim_root / "oasis2_metadata_template_summary.json").write_text("{}", encoding="utf-8")
    (interim_root / "oasis2_labeled_prep_manifest.csv").write_text("image,label\n", encoding="utf-8")
    (interim_root / "oasis2_subject_safe_split_plan.csv").write_text("subject_id,subject_safe_bucket\nOAS2_0001,0\n", encoding="utf-8")
    (interim_root / "oasis2_subject_safe_split_plan_summary.json").write_text("{}", encoding="utf-8")

    readiness_root = settings.outputs_root / "reports" / "readiness"
    readiness_root.mkdir(parents=True, exist_ok=True)
    (readiness_root / "oasis2_readiness.json").write_text("{}", encoding="utf-8")
    (readiness_root / "oasis2_readiness.md").write_text("# ready", encoding="utf-8")
    onboarding_root = settings.outputs_root / "reports" / "onboarding"
    onboarding_root.mkdir(parents=True, exist_ok=True)
    (onboarding_root / "oasis2_adapter_status.json").write_text("{}", encoding="utf-8")
    (onboarding_root / "oasis2_adapter_status.md").write_text("# adapter", encoding="utf-8")
    (onboarding_root / "oasis2_metadata_adapter_status.json").write_text("{}", encoding="utf-8")
    (onboarding_root / "oasis2_metadata_adapter_status.md").write_text("# metadata", encoding="utf-8")
    (onboarding_root / "oasis2_subject_safe_split_plan.md").write_text("# split", encoding="utf-8")

    result = build_oasis2_upload_bundle(
        settings=settings,
        source_root=source_root,
        materialize_mode="copy",
        output_root=tmp_path / "bundle",
    )

    copied_hdr = result.bundle_root / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"
    copied_img = copied_hdr.with_suffix(".img")
    assert copied_hdr.exists()
    assert copied_img.exists()
    assert result.included_session_count == 1
    assert result.missing_reference_count == 0

    relative_manifest = pd.read_csv(result.relative_manifest_path)
    assert relative_manifest.loc[0, "image"].startswith("OAS2_RAW_PART1/")
    assert (result.bundle_root / "backend_reference" / "oasis2_longitudinal_records_relative.csv").exists()
    assert (result.bundle_root / "backend_reference" / "oasis2_readiness.json").exists()
    assert (result.bundle_root / "backend_reference" / "oasis2_metadata_template.csv").exists()
    assert (result.bundle_root / "backend_reference" / "oasis2_metadata_adapter_status.json").exists()
    assert (result.bundle_root / "backend_reference" / "oasis2_subject_safe_split_plan.csv").exists()
    assert (result.bundle_root / "README.md").exists()
