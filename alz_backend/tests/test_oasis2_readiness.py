"""Tests for future OASIS-2 readiness reporting."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings
from src.data.oasis2_readiness import (
    OASIS2_SOURCE_ENV_VAR,
    build_oasis2_readiness_report,
    resolve_oasis2_source_root,
    save_oasis2_readiness_report,
)


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for OASIS-2 readiness tests."""

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


def test_oasis2_readiness_warns_when_source_is_missing(tmp_path: Path) -> None:
    """Missing OASIS-2 data should be a warning, not a false success."""

    settings = _build_settings(tmp_path)
    report = build_oasis2_readiness_report(settings, source_root=tmp_path / "missing_oasis2")
    payload = report.to_payload()

    assert report.overall_status == "warn"
    assert payload["dataset_summary"]["source_exists"] is False
    assert payload["summary"]["warn"] >= 1
    assert any(check["name"] == "source_root" for check in payload["checks"])
    assert any("ALZ_OASIS2_SOURCE_DIR" in recommendation for recommendation in payload["recommendations"])


def test_oasis2_readiness_detects_subjects_sessions_and_volume_formats(tmp_path: Path) -> None:
    """A plausible OASIS-2-like folder should surface longitudinal-ready signals."""

    settings = _build_settings(tmp_path)
    source_root = tmp_path / "OASIS2"
    (source_root / "OAS2_0001_MR1").mkdir(parents=True, exist_ok=True)
    (source_root / "OAS2_0001_MR2").mkdir(parents=True, exist_ok=True)
    (source_root / "metadata").mkdir(parents=True, exist_ok=True)
    (source_root / "OAS2_0001_MR1" / "mprage.nii.gz").write_bytes(b"")
    (source_root / "OAS2_0001_MR2" / "mprage.nii.gz").write_bytes(b"")
    (source_root / "metadata" / "oasis2_clinical.csv").write_text("subject_id,label\n", encoding="utf-8")

    report = build_oasis2_readiness_report(settings, source_root=source_root)
    payload = report.to_payload()

    assert report.overall_status == "pass"
    assert payload["dataset_summary"]["supported_volume_file_count"] == 2
    assert payload["dataset_summary"]["metadata_file_count"] == 1
    assert payload["dataset_summary"]["unique_subject_id_count"] == 1
    assert payload["dataset_summary"]["unique_session_id_count"] == 2
    assert payload["dataset_summary"]["longitudinal_subject_count"] == 1
    assert payload["dataset_summary"]["format_counts"][".nii.gz"] == 2
    assert any(check["name"] == "pipeline_fit" and check["status"] == "pass" for check in payload["checks"])


def test_oasis2_readiness_report_can_be_saved(tmp_path: Path) -> None:
    """The readiness helper should write JSON and Markdown artifacts."""

    settings = _build_settings(tmp_path)
    source_root = tmp_path / "OASIS2"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "OAS2_0002_MR1.hdr").write_bytes(b"")
    (source_root / "OAS2_0002_MR1.img").write_bytes(b"")

    report = build_oasis2_readiness_report(settings, source_root=source_root)
    json_path, md_path = save_oasis2_readiness_report(report, settings, file_stem="unit_oasis2_readiness")

    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] in {"pass", "warn", "fail"}
    assert "OASIS-2 Readiness Report" in md_path.read_text(encoding="utf-8")


def test_oasis2_readiness_detects_split_raw_parts_automatically(tmp_path: Path) -> None:
    """Split raw OAS2 part folders should be recognized as a usable readiness source."""

    settings = _build_settings(tmp_path)
    raw_dir_1 = tmp_path / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW"
    raw_dir_2 = tmp_path / "OAS2_RAW_PART2" / "OAS2_0001_MR2" / "RAW"
    raw_dir_1.mkdir(parents=True, exist_ok=True)
    raw_dir_2.mkdir(parents=True, exist_ok=True)
    (raw_dir_1 / "mpr-1.nifti.hdr").write_bytes(b"")
    (raw_dir_1 / "mpr-1.nifti.img").write_bytes(b"")
    (raw_dir_2 / "mpr-2.nifti.hdr").write_bytes(b"")
    (raw_dir_2 / "mpr-2.nifti.img").write_bytes(b"")

    report = build_oasis2_readiness_report(settings)
    payload = report.to_payload()

    assert report.overall_status == "warn"
    assert payload["dataset_summary"]["source_exists"] is True
    assert payload["dataset_summary"]["supported_volume_file_count"] == 4
    assert payload["dataset_summary"]["unique_subject_id_count"] == 1
    assert payload["dataset_summary"]["longitudinal_subject_count"] == 1
    assert payload["source_resolution"] == "auto_detected_part_roots"


def test_resolve_oasis2_source_root_can_use_environment_override(tmp_path: Path, monkeypatch) -> None:
    """Environment overrides should win when present."""

    settings = _build_settings(tmp_path)
    override_root = tmp_path / "custom_oasis2"
    monkeypatch.setenv(OASIS2_SOURCE_ENV_VAR, str(override_root))

    resolved_root, candidates, resolution = resolve_oasis2_source_root(settings)

    assert resolved_root == override_root.resolve()
    assert candidates == []
    assert resolution == "env"
