"""Tests for backend readiness diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings

from scripts.check_backend_readiness import build_backend_readiness_report, save_backend_readiness_report


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated settings with the required minimal project layout."""

    project_root = tmp_path / "alz_backend"
    required_files = [
        "src/api/main.py",
        "src/inference/pipeline.py",
        "src/explainability/gradcam.py",
        "src/longitudinal/tracker.py",
        "src/volumetrics/structural.py",
        "src/security/disclaimers.py",
        "configs/oasis_transforms.yaml",
        "configs/kaggle_transforms.yaml",
        "configs/oasis_model.yaml",
        "configs/oasis_train.yaml",
        "configs/longitudinal_report_schema.example.json",
        "configs/structural_metrics_schema.example.json",
    ]
    for relative_path in required_files:
        path = project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
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
        kaggle_source_root=tmp_path,
        oasis_source_root=tmp_path / "OASIS",
    )


def test_backend_readiness_report_builds_with_required_checks(tmp_path: Path) -> None:
    """Readiness should pass required structure and warn for optional artifacts."""

    settings = _build_settings(tmp_path)
    report = build_backend_readiness_report(settings)
    payload = report.to_payload()

    assert payload["summary"]["fail"] == 0
    assert payload["summary"]["warn"] >= 1
    assert any(check.name == "decision_support_disclaimer" and check.status == "pass" for check in report.checks)
    assert any(check.name == "trained_checkpoint" and check.status == "warn" for check in report.checks)


def test_save_backend_readiness_report_writes_json_and_markdown(tmp_path: Path) -> None:
    """Readiness reports should be saved as JSON and Markdown."""

    settings = _build_settings(tmp_path)
    report = build_backend_readiness_report(settings)
    json_path, md_path = save_backend_readiness_report(report, settings)

    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] in {"pass", "warn", "fail"}
    assert "Backend Readiness Report" in md_path.read_text(encoding="utf-8")

