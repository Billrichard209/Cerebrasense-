"""Tests for the OASIS upload bundle builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.data.oasis_upload_bundle import build_oasis_upload_bundle


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for upload bundle tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    config_root = project_root / "configs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path / "kaggle",
        oasis_source_root=tmp_path / "OASIS",
        serving_config_path=config_root / "backend_serving.yaml",
    )


def test_build_oasis_upload_bundle_materializes_expected_subset(tmp_path: Path) -> None:
    """The bundle builder should create a portable labeled OASIS subset."""

    settings = _build_settings(tmp_path)
    source_root = settings.oasis_source_root
    session_root = source_root / "oasis_cross-sectional_disc1" / "disc1" / "OAS1_0001_MR1"
    image_root = session_root / "PROCESSED" / "MPRAGE" / "T88_111"
    image_root.mkdir(parents=True, exist_ok=True)

    metadata_path = source_root / "oasis_cross-sectional.csv"
    pd.DataFrame([{"ID": "OAS1_0001_MR1", "CDR": 0.0}]).to_csv(metadata_path, index=False)

    hdr_path = image_root / "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr"
    img_path = image_root / "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.img"
    hdr_path.write_text("hdr", encoding="utf-8")
    img_path.write_text("img", encoding="utf-8")
    xml_path = session_root / "OAS1_0001_MR1.xml"
    xml_path.write_text("<root />", encoding="utf-8")

    interim_root = settings.data_root / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "image": str(hdr_path),
                "label": 0,
                "label_name": "nondemented",
                "subject_id": "OAS1_0001",
                "scan_timestamp": "",
                "dataset": "oasis1",
                "meta": json.dumps(
                    {
                        "session_id": "OAS1_0001_MR1",
                        "xml_path": str(xml_path),
                    }
                ),
            }
        ]
    ).to_csv(interim_root / "oasis1_manifest.csv", index=False)
    (interim_root / "oasis1_manifest_summary.json").write_text("{}", encoding="utf-8")
    (interim_root / "oasis1_manifest_dropped_rows.csv").write_text("row_index,reason\n", encoding="utf-8")

    result = build_oasis_upload_bundle(
        settings=settings,
        materialize_mode="copy",
        output_root=tmp_path / "bundle",
    )

    copied_hdr = result.oasis_subset_root / "oasis_cross-sectional_disc1" / "disc1" / "OAS1_0001_MR1" / "PROCESSED" / "MPRAGE" / "T88_111" / "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr"
    copied_img = copied_hdr.with_suffix(".img")
    copied_xml = result.oasis_subset_root / "oasis_cross-sectional_disc1" / "disc1" / "OAS1_0001_MR1" / "OAS1_0001_MR1.xml"
    copied_metadata = result.oasis_subset_root / "oasis_cross-sectional.csv"

    assert copied_hdr.exists()
    assert copied_img.exists()
    assert copied_xml.exists()
    assert copied_metadata.exists()
    assert result.included_session_count == 1
    assert result.missing_reference_count == 0

    relative_manifest = pd.read_csv(result.relative_manifest_path)
    assert relative_manifest.loc[0, "image"].startswith("oasis_cross-sectional_disc1/")
    assert (result.bundle_root / "README.md").exists()
