"""Tests for opening the key local OASIS workflow artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings

from scripts import open_oasis_local_outputs as open_module


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    outputs_root = project_root / "outputs"
    data_root = project_root / "data"
    outputs_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path,
        oasis_source_root=tmp_path / "OASIS",
    )


def test_open_oasis_local_outputs_supports_older_workflow_summaries(tmp_path: Path, monkeypatch) -> None:
    """Older workflow summaries without summary_md_path should still open cleanly."""

    settings = _settings(tmp_path)
    workflow_root = settings.outputs_root / "reports" / "workflows" / "oasis_local_workflow"
    workflow_root.mkdir(parents=True, exist_ok=True)
    batch_root = settings.outputs_root / "reports" / "batch_inference" / "oasis_local_workflow_batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    demo_root = settings.outputs_root / "reports" / "demo" / "oasis_local_workflow_demo"
    demo_root.mkdir(parents=True, exist_ok=True)
    presentation_root = settings.outputs_root / "reports" / "presentation"
    presentation_root.mkdir(parents=True, exist_ok=True)

    workflow_md = workflow_root / "workflow_summary.md"
    workflow_md.write_text("workflow", encoding="utf-8")
    batch_csv = batch_root / "batch_predictions.csv"
    batch_csv.write_text("scan_path,status\nscan.hdr,ok\n", encoding="utf-8")
    presentation_md = presentation_root / "oasis_local_path_summary.md"
    presentation_md.write_text("presentation", encoding="utf-8")

    (workflow_root / "workflow_summary.json").write_text(
        json.dumps(
            {
                "demo_bundle_root": str(demo_root),
                "batch_report_root": str(batch_root),
                "batch_predictions_csv": str(batch_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    opened_paths: list[str] = []
    monkeypatch.setattr(open_module.os, "startfile", lambda path: opened_paths.append(str(path)))

    opened = open_module.open_oasis_local_outputs(settings=settings)

    assert str(workflow_md.resolve()) in [str(path) for path in opened]
    assert str(presentation_md.resolve()) in [str(path) for path in opened]
    assert len(opened_paths) == 5
