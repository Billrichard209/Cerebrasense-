"""Tests for the local OASIS presentation/status summary builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings

from scripts import build_oasis_local_presentation_summary as summary_module


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


def test_build_oasis_local_presentation_summary_writes_outputs(tmp_path: Path) -> None:
    """The local presentation summary should read workflow artifacts and save clean outputs."""

    settings = _settings(tmp_path)
    workflow_root = settings.outputs_root / "reports" / "workflows" / "oasis_local_workflow"
    workflow_root.mkdir(parents=True, exist_ok=True)
    batch_root = settings.outputs_root / "reports" / "batch_inference" / "oasis_local_workflow_batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    demo_root = settings.outputs_root / "reports" / "demo" / "oasis_local_workflow_demo"
    demo_root.mkdir(parents=True, exist_ok=True)

    batch_predictions_path = batch_root / "batch_predictions.csv"
    pd.DataFrame(
        [
            {
                "scan_path": "scan_a.hdr",
                "subject_id": "OAS1_0001",
                "session_id": "OAS1_0001_MR1",
                "status": "ok",
                "predicted_label": 0,
                "label_name": "nondemented",
                "probability_score": 0.11,
                "confidence_score": 0.89,
                "confidence_level": "high",
                "review_flag": False,
                "prediction_json": "prediction_a.json",
                "error": None,
            },
            {
                "scan_path": "scan_b.hdr",
                "subject_id": "OAS1_0002",
                "session_id": "OAS1_0002_MR1",
                "status": "ok",
                "predicted_label": 1,
                "label_name": "demented",
                "probability_score": 0.55,
                "confidence_score": 0.55,
                "confidence_level": "low",
                "review_flag": True,
                "prediction_json": "prediction_b.json",
                "error": None,
            },
        ]
    ).to_csv(batch_predictions_path, index=False)

    workflow_summary = {
        "run_name": "oasis_active",
        "workflow_root": str(workflow_root),
        "demo_bundle_root": str(demo_root),
        "batch_report_root": str(batch_root),
        "batch_predictions_csv": str(batch_predictions_path),
        "summary_md_path": str(workflow_root / "workflow_summary.md"),
    }
    (workflow_root / "workflow_summary.json").write_text(json.dumps(workflow_summary, indent=2), encoding="utf-8")

    summary = summary_module.build_oasis_local_presentation_summary(settings=settings)

    assert summary["scan_count"] == 2
    assert summary["review_required"] == 1
    assert summary["label_counts"]["demented"] == 1
    assert summary["label_counts"]["nondemented"] == 1
    assert Path(summary["summary_json_path"]).exists()
    assert Path(summary["summary_md_path"]).exists()
