"""Tests for OASIS-2 metadata template and merge helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.data.oasis2_metadata import (
    build_oasis2_metadata_template,
    import_oasis2_official_demographics_into_metadata_template,
    load_oasis2_metadata_template,
    merge_oasis2_metadata_template,
    save_oasis2_metadata_adapter_summary,
    save_oasis2_official_demographics_import_summary,
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


def _write_manifest(settings: AppSettings, image_root: Path) -> Path:
    manifest_path = settings.data_root / "interim" / "oasis2_session_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    session_a = image_root / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW" / "mpr-1.nifti.hdr"
    session_b = image_root / "OAS2_RAW_PART2" / "OAS2_0001_MR2" / "RAW" / "mpr-2.nifti.hdr"
    session_a.parent.mkdir(parents=True, exist_ok=True)
    session_b.parent.mkdir(parents=True, exist_ok=True)
    session_a.write_bytes(b"hdr")
    session_b.write_bytes(b"hdr")
    manifest_path.write_text(
        "\n".join(
            [
                "image,label,label_name,subject_id,session_id,visit_number,scan_timestamp,dataset,dataset_type,meta",
                f"{session_a},,,OAS2_0001,OAS2_0001_MR1,1,2024-01-01,oasis2_raw,3d_volumes,\"{{\"\"source_part\"\":\"\"OAS2_RAW_PART1\"\"}}\"",
                f"{session_b},,,OAS2_0001,OAS2_0001_MR2,2,2024-06-01,oasis2_raw,3d_volumes,\"{{\"\"source_part\"\":\"\"OAS2_RAW_PART2\"\"}}\"",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_official_demographics(path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "Subject ID": "OAS2_0001",
                "MRI ID": "OAS2_0001_MR1",
                "Group": "Converted",
                "Visit": 1,
                "MR Delay": 0,
                "M/F": "F",
                "Hand": "R",
                "Age": 72,
                "EDUC": 16,
                "SES": 2.0,
                "MMSE": 30.0,
                "CDR": 0.0,
                "eTIV": 1500.0,
                "nWBV": 0.72,
                "ASF": 1.1,
            },
            {
                "Subject ID": "OAS2_0001",
                "MRI ID": "OAS2_0001_MR2",
                "Group": "Nondemented",
                "Visit": 2,
                "MR Delay": 120,
                "M/F": "F",
                "Hand": "R",
                "Age": 73,
                "EDUC": 16,
                "SES": 2.0,
                "MMSE": 27.0,
                "CDR": 0.5,
                "eTIV": 1490.0,
                "nWBV": 0.70,
                "ASF": 1.1,
            },
        ]
    )
    frame.to_excel(path, index=False)
    return path


def test_build_oasis2_metadata_template_from_manifest(tmp_path: Path) -> None:
    """The metadata template should mirror the unlabeled session manifest keys."""

    settings = _settings(tmp_path)
    _write_manifest(settings, tmp_path / "dataset")

    result = build_oasis2_metadata_template(settings=settings)
    frame = load_oasis2_metadata_template(settings=settings)

    assert result.row_count == 2
    assert frame["subject_id"].tolist() == ["OAS2_0001", "OAS2_0001"]
    assert frame["session_id"].tolist() == ["OAS2_0001_MR1", "OAS2_0001_MR2"]
    assert "diagnosis_label" in frame.columns
    assert "split_group_hint" in frame.columns


def test_merge_oasis2_metadata_template_stays_unready_without_labels(tmp_path: Path) -> None:
    """The merge path should work before real labels exist while staying honest about readiness."""

    settings = _settings(tmp_path)
    _write_manifest(settings, tmp_path / "dataset")
    template_result = build_oasis2_metadata_template(settings=settings)

    frame = pd.read_csv(template_result.template_path)
    frame["metadata_complete"] = [True, False]
    frame["age_at_visit"] = [72, 73]
    frame["clinical_status"] = ["followup_needed", "followup_needed"]
    frame.to_csv(template_result.template_path, index=False)

    summary = merge_oasis2_metadata_template(settings=settings)
    json_path, md_path = save_oasis2_metadata_adapter_summary(summary, settings, file_stem="unit_oasis2_metadata")

    assert summary.ready_for_labeled_manifest is False
    assert summary.matched_metadata_row_count == 2
    assert summary.rows_with_candidate_labels == 0
    merged_frame = pd.read_csv(summary.merged_manifest_path)
    merged_meta = json.loads(merged_frame.loc[0, "meta"])
    assert merged_meta["oasis2_metadata"]["age_at_visit"] == 72.0
    assert "Recommendations" in md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["matched_metadata_row_count"] == 2


def test_merge_oasis2_metadata_template_can_mark_labeled_prep_ready(tmp_path: Path) -> None:
    """Filled diagnosis labels should let the adapter report labeled-manifest readiness."""

    settings = _settings(tmp_path)
    _write_manifest(settings, tmp_path / "dataset")
    template_result = build_oasis2_metadata_template(settings=settings)

    frame = pd.read_csv(template_result.template_path)
    frame["metadata_complete"] = [True, True]
    frame["diagnosis_label"] = [0, 1]
    frame["diagnosis_label_name"] = ["nondemented_like", "demented_like"]
    frame["metadata_source"] = ["manual_review", "manual_review"]
    frame.to_csv(template_result.template_path, index=False)

    summary = merge_oasis2_metadata_template(settings=settings)

    assert summary.ready_for_labeled_manifest is True
    assert summary.rows_with_candidate_labels == 2
    merged_frame = pd.read_csv(summary.merged_manifest_path)
    assert merged_frame["label"].tolist() == [0, 1]


def test_merge_oasis2_metadata_template_auto_builds_template_when_missing(tmp_path: Path) -> None:
    """The metadata adapter should bootstrap the template if the CSV does not exist yet."""

    settings = _settings(tmp_path)
    _write_manifest(settings, tmp_path / "dataset")

    summary = merge_oasis2_metadata_template(settings=settings)

    assert Path(summary.metadata_path).exists()
    assert summary.matched_metadata_row_count == 2
    assert summary.ready_for_labeled_manifest is False


def test_import_oasis2_official_demographics_fills_binary_labels_from_cdr(tmp_path: Path) -> None:
    """The official OASIS-2 demographics sheet should auto-fill the metadata template safely."""

    settings = _settings(tmp_path)
    _write_manifest(settings, tmp_path / "dataset")
    template_result = build_oasis2_metadata_template(settings=settings)
    demographics_path = _write_official_demographics(tmp_path / "oasis2_demographics.xlsx")

    summary = import_oasis2_official_demographics_into_metadata_template(
        demographics_path,
        settings=settings,
        metadata_path=template_result.template_path,
    )
    json_path, md_path = save_oasis2_official_demographics_import_summary(
        summary,
        settings=settings,
        file_stem="unit_oasis2_official_demographics_import",
    )

    frame = load_oasis2_metadata_template(settings=settings, metadata_path=template_result.template_path)
    assert frame["diagnosis_label"].tolist() == [0, 1]
    assert frame["diagnosis_label_name"].tolist() == ["nondemented", "demented"]
    assert frame["clinical_status"].tolist() == ["Converted", "Nondemented"]
    assert frame["metadata_complete"].tolist() == [True, True]
    assert summary.labeled_row_count == 2
    assert summary.converted_group_row_count == 1
    assert summary.group_cdr_disagreement_row_count == 1
    assert "binary_label_policy=derived_from_cdr_global" in frame.loc[0, "notes"]
    assert json.loads(json_path.read_text(encoding="utf-8"))["matched_row_count"] == 2
    assert "Recommendations" in md_path.read_text(encoding="utf-8")
