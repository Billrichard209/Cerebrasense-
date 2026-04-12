"""Tests for the GitHub/Drive scope data plan."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings
from src.data.scope_data_plan import build_scope_data_plan, save_scope_data_plan


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for scope data plan tests."""

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
        kaggle_source_root=tmp_path,
        oasis_source_root=tmp_path / "OASIS",
        serving_config_path=config_root / "backend_serving.yaml",
    )


def test_build_scope_data_plan_detects_oasis_and_kaggle_paths(tmp_path: Path) -> None:
    """The scope data plan should discover the current narrowed-scope dataset units."""

    settings = _build_settings(tmp_path)
    (tmp_path / ".gitignore").write_text("", encoding="utf-8")
    (tmp_path / "OASIS").mkdir()
    (tmp_path / "OriginalDataset").mkdir()
    (tmp_path / "AugmentedAlzheimerDataset").mkdir()

    report = build_scope_data_plan(settings)

    assert report["workspace"]["gitignore_present"] is True
    assert report["datasets_for_drive"]["oasis1"]["present"] is True
    assert report["datasets_for_drive"]["kaggle_alz"]["present"] is True
    assert report["datasets_for_drive"]["oasis2"]["present"] is False
    assert report["code_for_github"]["preferred_code_source"] == "github"
    assert any("Keep OASIS-1 as the primary 3D branch" in item for item in report["recommendations"])


def test_save_scope_data_plan_writes_json_and_markdown(tmp_path: Path) -> None:
    """The scope data plan saver should emit JSON and Markdown artifacts."""

    settings = _build_settings(tmp_path)
    report = {
        "goal_statement": "scope-data-plan",
        "workspace": {"gitignore_present": True},
        "code_for_github": {"preferred_code_source": "github", "keep_local_working_copy": True},
        "datasets_for_drive": {
            "oasis1": {"present": True, "recommended_drive_root": "x", "existing_local_paths": ["OASIS"]},
        },
        "recommendations": ["keep datasets separate"],
    }

    json_path, md_path = save_scope_data_plan(report, settings, file_stem="unit_scope_data_plan")

    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["goal_statement"] == "scope-data-plan"
    assert "Scope Data Plan" in md_path.read_text(encoding="utf-8")
