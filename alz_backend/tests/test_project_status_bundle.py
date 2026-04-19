"""Tests for the current project status bundle builder."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings

from scripts import build_project_status_bundle as bundle_module


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
    for doc_name in ["PROJECT_BACKBONE.md", "project_scope.md", "oasis_productization_workflow.md", "github_drive_workflow.md"]:
        (docs_root / doc_name).write_text(doc_name, encoding="utf-8")
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_project_status_bundle_collects_core_artifacts(tmp_path: Path) -> None:
    """The project status bundle should gather the latest reports and demo bundles into one folder."""

    settings = _settings(tmp_path)
    _write_json(
        settings.outputs_root / "reports" / "comparison" / "oasis_baseline_comparison.json",
        {
            "demo_bundles": {
                "active": {"bundle_root": str(settings.outputs_root / "reports" / "demo" / "active_demo")},
                "candidate": {"bundle_root": str(settings.outputs_root / "reports" / "demo" / "candidate_demo")},
            }
        },
    )
    (settings.outputs_root / "reports" / "comparison" / "oasis_baseline_comparison.md").write_text("comparison", encoding="utf-8")
    _write_json(
        settings.outputs_root / "reports" / "presentation" / "oasis_presentation_summary.json",
        {
            "recommendation": "keep_active",
            "headline": "headline",
            "project_state": {
                "active_run_name": "active_run",
                "candidate_run_name": "candidate_run",
            },
        },
    )
    (settings.outputs_root / "reports" / "presentation" / "oasis_presentation_summary.md").write_text("presentation", encoding="utf-8")
    _write_json(settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.json", {"goal": "scope"})
    (settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.md").write_text("evidence", encoding="utf-8")
    _write_json(settings.outputs_root / "reports" / "readiness" / "backend_readiness.json", {"overall_status": "pass"})
    (settings.outputs_root / "reports" / "readiness" / "backend_readiness.md").write_text("readiness", encoding="utf-8")
    _write_json(settings.outputs_root / "reports" / "productization" / "oasis_productization_status.json", {"overall_status": "fail"})
    (settings.outputs_root / "reports" / "productization" / "oasis_productization_status.md").write_text("productization", encoding="utf-8")
    _write_json(settings.outputs_root / "model_registry" / "oasis_current_baseline.json", {"run_name": "active_run"})
    _write_json(settings.outputs_root / "model_registry" / "oasis_candidate_v3.json", {"run_name": "candidate_run"})
    active_demo_root = settings.outputs_root / "reports" / "demo" / "active_demo"
    active_demo_root.mkdir(parents=True, exist_ok=True)
    (active_demo_root / "demo_summary.json").write_text("{}", encoding="utf-8")
    candidate_demo_root = settings.outputs_root / "reports" / "demo" / "candidate_demo"
    candidate_demo_root.mkdir(parents=True, exist_ok=True)
    (candidate_demo_root / "demo_summary.json").write_text("{}", encoding="utf-8")

    result = bundle_module.build_project_status_bundle(settings=settings, output_name="unit_status_bundle")

    bundle_root = Path(result.bundle_root)
    assert bundle_root.exists()
    assert (bundle_root / "files" / "oasis_presentation_summary.md").exists()
    assert (bundle_root / "demo" / "active_bundle" / "demo_summary.json").exists()
    assert (bundle_root / "demo" / "candidate_bundle" / "demo_summary.json").exists()
    assert result.recommendation == "keep_active"
