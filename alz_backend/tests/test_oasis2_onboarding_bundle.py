"""Tests for the OASIS-2 onboarding bundle builder."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings

from scripts import build_oasis2_onboarding_bundle as bundle_module


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    config_root = project_root / "configs"
    docs_root = project_root / "docs"
    outputs_root = project_root / "outputs"
    data_root = project_root / "data"
    config_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    (config_root / "backend_serving.yaml").write_text(
        "active_oasis_model_registry: outputs/model_registry/oasis_current_baseline.json\n",
        encoding="utf-8",
    )
    for doc_name in ["PROJECT_BACKBONE.md", "project_scope.md", "oasis2_readiness.md"]:
        (docs_root / doc_name).write_text(doc_name, encoding="utf-8")
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


def test_build_oasis2_onboarding_bundle_collects_local_artifacts(tmp_path: Path) -> None:
    """The OASIS-2 onboarding bundle should regenerate and gather local readiness artifacts."""

    settings = _settings(tmp_path)
    raw_dir_1 = tmp_path / "dataset" / "OAS2_RAW_PART1" / "OAS2_0001_MR1" / "RAW"
    raw_dir_2 = tmp_path / "dataset" / "OAS2_RAW_PART2" / "OAS2_0001_MR2" / "RAW"
    raw_dir_1.mkdir(parents=True, exist_ok=True)
    raw_dir_2.mkdir(parents=True, exist_ok=True)
    (raw_dir_1 / "mpr-1.nifti.hdr").write_bytes(b"hdr")
    (raw_dir_1 / "mpr-1.nifti.img").write_bytes(b"img")
    (raw_dir_2 / "mpr-2.nifti.hdr").write_bytes(b"hdr")
    (raw_dir_2 / "mpr-2.nifti.img").write_bytes(b"img")

    focus_json = settings.outputs_root / "reports" / "evidence" / "oasis_longitudinal_focus_report.json"
    focus_md = settings.outputs_root / "reports" / "evidence" / "oasis_longitudinal_focus_report.md"
    focus_json.parent.mkdir(parents=True, exist_ok=True)
    focus_json.write_text(json.dumps({"goal_statement": "focus"}), encoding="utf-8")
    focus_md.write_text("focus", encoding="utf-8")

    result = bundle_module.build_oasis2_onboarding_bundle(
        settings=settings,
        source_root=tmp_path / "dataset",
        output_name="unit_oasis2_bundle",
    )

    bundle_root = Path(result.bundle_root)
    assert bundle_root.exists()
    assert (bundle_root / "files" / "generated" / "oasis2_readiness.json").exists()
    assert (bundle_root / "files" / "generated" / "oasis2_session_manifest.csv").exists()
    assert (bundle_root / "files" / "generated" / "oasis_longitudinal_focus_report.md").exists()
    assert (bundle_root / "files" / "docs" / "oasis2_readiness.md").exists()
    assert result.upload_to_drive_now is False
    assert result.unique_subject_count == 1
    assert result.unique_session_count == 2
