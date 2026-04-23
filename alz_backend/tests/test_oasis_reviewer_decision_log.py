"""Tests for the OASIS reviewer decision log builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings

from scripts import build_oasis_reviewer_decision_log as decision_log_module


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


def test_build_oasis_reviewer_decision_log_writes_template_and_preserves_manual_fields(tmp_path: Path) -> None:
    """The decision log builder should create a template and keep reviewer edits on rebuild."""

    settings = _settings(tmp_path)
    review_root = settings.outputs_root / "reports" / "review" / "oasis_review_pack"
    review_root.mkdir(parents=True, exist_ok=True)

    review_cases = pd.DataFrame(
        [
            {
                "rank": 1,
                "subject_id": "OAS1_0001",
                "session_id": "OAS1_0001_MR1",
                "label_name": "demented",
                "probability_score": 0.51,
                "confidence_level": "low",
                "review_flag": True,
                "scan_path": "scan_a.hdr",
                "prediction_json": "01_prediction.json",
            }
        ]
    )
    review_cases.to_csv(review_root / "review_cases.csv", index=False)
    (review_root / "review_pack_summary.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "rank": 1,
                        "subject_id": "OAS1_0001",
                        "session_id": "OAS1_0001_MR1",
                        "label_name": "demented",
                        "probability_score": 0.51,
                        "confidence_level": "low",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    first_summary = decision_log_module.build_oasis_reviewer_decision_log(settings=settings)
    decision_log_path = Path(first_summary["decision_log_csv"])
    assert decision_log_path.exists()
    frame = pd.read_csv(decision_log_path, dtype=str).fillna("")
    assert frame.loc[0, "reviewer_status"] == "pending"
    assert Path(first_summary["summary_json_path"]).exists()
    assert Path(first_summary["summary_md_path"]).exists()

    frame.loc[0, "reviewer_status"] = "completed"
    frame.loc[0, "reviewer_decision"] = "nondemented_after_manual_review"
    frame.loc[0, "reviewed_by"] = "reviewer_a"
    frame.to_csv(decision_log_path, index=False)

    second_summary = decision_log_module.build_oasis_reviewer_decision_log(settings=settings)
    rebuilt = pd.read_csv(Path(second_summary["decision_log_csv"]), dtype=str).fillna("")
    assert rebuilt.loc[0, "reviewer_status"] == "completed"
    assert rebuilt.loc[0, "reviewer_decision"] == "nondemented_after_manual_review"
    assert rebuilt.loc[0, "reviewed_by"] == "reviewer_a"
