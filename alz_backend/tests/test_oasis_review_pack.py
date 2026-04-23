"""Tests for the local OASIS review pack builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings

from scripts import build_oasis_review_pack as review_pack_module


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


def test_build_oasis_review_pack_writes_only_flagged_cases(tmp_path: Path) -> None:
    """The review pack should keep only flagged cases and copy their prediction payloads."""

    settings = _settings(tmp_path)
    workflow_root = settings.outputs_root / "reports" / "workflows" / "oasis_local_workflow"
    workflow_root.mkdir(parents=True, exist_ok=True)
    batch_root = settings.outputs_root / "reports" / "batch_inference" / "oasis_local_workflow_batch"
    batch_root.mkdir(parents=True, exist_ok=True)

    prediction_a = settings.outputs_root / "predictions" / "case_a" / "prediction.json"
    prediction_b = settings.outputs_root / "predictions" / "case_b" / "prediction.json"
    prediction_a.parent.mkdir(parents=True, exist_ok=True)
    prediction_b.parent.mkdir(parents=True, exist_ok=True)
    prediction_a.write_text("{}", encoding="utf-8")
    prediction_b.write_text("{}", encoding="utf-8")

    batch_csv = batch_root / "batch_predictions.csv"
    pd.DataFrame(
        [
            {
                "scan_path": "scan_a.hdr",
                "subject_id": "OAS1_0001",
                "session_id": "OAS1_0001_MR1",
                "status": "ok",
                "label_name": "nondemented",
                "probability_score": 0.12,
                "confidence_level": "high",
                "review_flag": False,
                "prediction_json": str(prediction_a),
            },
            {
                "scan_path": "scan_b.hdr",
                "subject_id": "OAS1_0002",
                "session_id": "OAS1_0002_MR1",
                "status": "ok",
                "label_name": "demented",
                "probability_score": 0.52,
                "confidence_level": "low",
                "review_flag": True,
                "prediction_json": str(prediction_b),
            },
        ]
    ).to_csv(batch_csv, index=False)

    (workflow_root / "workflow_summary.json").write_text(
        json.dumps({"batch_predictions_csv": str(batch_csv)}, indent=2),
        encoding="utf-8",
    )

    summary = review_pack_module.build_oasis_review_pack(settings=settings)

    assert summary["review_case_count"] == 1
    assert Path(summary["review_cases_csv"]).exists()
    assert Path(summary["summary_json_path"]).exists()
    assert Path(summary["summary_md_path"]).exists()
    copied = Path(summary["pack_root"]) / "01_prediction.json"
    assert copied.exists()
