"""Tests for the OASIS specialist handoff pack builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings

from scripts import build_oasis_specialist_handoff_pack as handoff_module


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


def test_build_oasis_specialist_handoff_pack_copies_escalated_cases_and_references(tmp_path: Path) -> None:
    """The handoff pack should gather escalated cases and key supporting artifacts."""

    settings = _settings(tmp_path)
    review_root = settings.outputs_root / "reports" / "review"
    decision_root = review_root / "oasis_review_decision_log"
    pack_root = review_root / "oasis_review_pack"
    learning_root = settings.outputs_root / "reports" / "reviewer_learning" / "oasis_reviewer_learning_report"
    workflow_root = settings.outputs_root / "reports" / "workflows" / "oasis_local_workflow"
    presentation_root = settings.outputs_root / "reports" / "presentation"
    for path in [decision_root, pack_root, learning_root, workflow_root, presentation_root]:
        path.mkdir(parents=True, exist_ok=True)

    prediction = settings.outputs_root / "predictions" / "case_1" / "prediction.json"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_text("{}", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "rank": 1,
                "subject_id": "OAS1_0001",
                "session_id": "OAS1_0001_MR1",
                "label_name": "demented",
                "probability_score": "0.51",
                "reviewer_priority": "high",
                "follow_up_action": "needs_specialist_review",
                "reviewer_notes": "triaged only",
                "resolution_state": "escalated",
                "prediction_json": str(prediction),
            }
        ]
    ).to_csv(decision_root / "reviewer_decision_log.csv", index=False)

    for path in [
        pack_root / "review_pack_summary.md",
        decision_root / "reviewer_decision_log_summary.md",
        learning_root / "oasis_reviewer_learning_report.md",
        workflow_root / "workflow_summary.md",
        presentation_root / "oasis_local_path_summary.md",
    ]:
        path.write_text("# stub", encoding="utf-8")

    summary = handoff_module.build_oasis_specialist_handoff_pack(settings=settings)

    assert summary["escalated_case_count"] == 1
    assert Path(summary["escalated_cases_csv"]).exists()
    assert Path(summary["summary_json_path"]).exists()
    assert Path(summary["summary_md_path"]).exists()
    assert summary["copied_prediction_file_count"] == 1
