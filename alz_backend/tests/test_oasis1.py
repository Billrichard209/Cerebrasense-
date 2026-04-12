"""Tests for the dedicated OASIS-1 adapter and manifest builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.oasis1 import (
    OASIS1ManifestError,
    build_oasis1_manifest,
    choose_preferred_oasis1_image,
    extract_subject_id,
    map_oasis1_binary_label,
    parse_oasis1_xml_candidates,
    resolve_oasis1_metadata_columns,
)


def test_resolve_oasis1_metadata_columns_uses_id_and_cdr() -> None:
    """The adapter should resolve the real OASIS cross-sectional columns."""

    frame = pd.DataFrame({"ID": ["OAS1_0001_MR1"], "CDR": [0.0], "Delay": [None]})
    columns = resolve_oasis1_metadata_columns(frame)
    assert columns["session_id"] == "ID"
    assert columns["label"] == "CDR"
    assert columns["scan_timestamp"] is None


def test_resolve_oasis1_metadata_columns_fails_on_ambiguous_labels() -> None:
    """The adapter should fail loudly when multiple label columns are plausible."""

    frame = pd.DataFrame(
        {"ID": ["OAS1_0001_MR1"], "CDR": [0.0], "Group": ["Nondemented"]}
    )
    with pytest.raises(OASIS1ManifestError):
        resolve_oasis1_metadata_columns(frame)


def test_map_oasis1_binary_label_numeric_cdr() -> None:
    """Binary label mapping should follow the requested CDR rule."""

    assert map_oasis1_binary_label(0.0) == (0, "nondemented")
    assert map_oasis1_binary_label(0.5) == (1, "demented")
    assert map_oasis1_binary_label(2.0) == (1, "demented")


def test_extract_subject_id_from_session_id() -> None:
    """Subject IDs should remain reliable for future split logic."""

    assert extract_subject_id("OAS1_0123_MR1") == "OAS1_0123"


def test_parse_oasis1_xml_candidates_and_build_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter should parse XML resources, prefer MASKED images, and log drops."""

    source_root = tmp_path / "OASIS"
    session_root = source_root / "disc2" / "OAS1_0001_MR1"
    raw_root = session_root / "RAW"
    processed_root = session_root / "PROCESSED" / "MPRAGE" / "T88_111"
    raw_root.mkdir(parents=True)
    processed_root.mkdir(parents=True)

    (raw_root / "OAS1_0001_MR1_mpr-1_anon.hdr").write_text("hdr", encoding="utf-8")
    (raw_root / "OAS1_0001_MR1_mpr-1_anon.img").write_text("img", encoding="utf-8")
    (processed_root / "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr").write_text("hdr", encoding="utf-8")
    (processed_root / "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.img").write_text("img", encoding="utf-8")

    xml_path = session_root / "OAS1_0001_MR1.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<xnat:MRSession xmlns:xnat="http://nrg.wustl.edu/xnat">
  <xnat:file URI="cross-sectional/disc2/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.img" content="MPRAGE_RAW" format="ANALYZE 7.5">
    <xnat:dimensions x="256" y="256" z="128" />
    <xnat:voxelRes x="1.0" y="1.0" z="1.25" />
  </xnat:file>
  <xnat:file URI="cross-sectional/disc2/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.img" content="MASKED" format="ANALYZE 7.5">
    <xnat:dimensions x="176" y="208" z="176" />
    <xnat:voxelRes x="1.0" y="1.0" z="1.0" />
  </xnat:file>
</xnat:MRSession>
""",
        encoding="utf-8",
    )

    metadata_path = source_root / "oasis_cross-sectional.csv"
    pd.DataFrame(
        [
            {"ID": "OAS1_0001_MR1", "CDR": 0.0},
            {"ID": "OAS1_0002_MR1", "CDR": 0.5},
        ]
    ).to_csv(metadata_path, index=False)

    candidates = parse_oasis1_xml_candidates(source_root)
    preferred = choose_preferred_oasis1_image(candidates["OAS1_0001_MR1"])
    assert preferred is not None
    assert preferred.content == "MASKED"
    assert preferred.image_path.name.endswith(".hdr")

    from src.configs.runtime import AppSettings

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    data_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)

    settings = AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=project_root.parent,
        oasis_source_root=source_root,
    )

    result = build_oasis1_manifest(settings=settings, output_format="both")
    assert result.manifest_csv_path is not None and result.manifest_csv_path.exists()
    assert result.manifest_jsonl_path is not None and result.manifest_jsonl_path.exists()
    assert result.dropped_rows_path.exists()
    assert result.manifest_row_count == 1
    assert result.dropped_row_count == 1

    manifest_frame = pd.read_csv(result.manifest_csv_path)
    assert manifest_frame.loc[0, "label"] == 0
    assert manifest_frame.loc[0, "label_name"] == "nondemented"
    assert manifest_frame.loc[0, "subject_id"] == "OAS1_0001"
    assert "MASKED" in manifest_frame.loc[0, "meta"]
