"""Tests for the OASIS reviewer-learning report builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings

from scripts import build_oasis_reviewer_learning_report as learning_module


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


def test_build_oasis_reviewer_learning_report_handles_pending_only_state(tmp_path: Path) -> None:
    """Pending-only decision logs should yield a readiness-style learning report."""

    settings = _settings(tmp_path)
    review_root = settings.outputs_root / "reports" / "review" / "oasis_review_decision_log"
    review_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "rank": 1,
                "subject_id": "OAS1_0001",
                "session_id": "OAS1_0001_MR1",
                "label_name": "demented",
                "probability_score": "0.51",
                "reviewer_status": "pending",
                "reviewer_agrees_with_model": "",
                "resolution_state": "open",
                "reviewer_priority": "normal",
            }
        ]
    ).to_csv(review_root / "reviewer_decision_log.csv", index=False)

    summary = learning_module.build_oasis_reviewer_learning_report(settings=settings)

    assert summary["review_case_count"] == 1
    assert summary["completed_case_count"] == 0
    assert summary["pending_case_count"] == 1
    assert summary["recommended_action"] == "fill_reviewer_decision_log"
    assert Path(summary["summary_json_path"]).exists()
    assert Path(summary["summary_md_path"]).exists()


def test_build_oasis_reviewer_learning_report_counts_disagreements(tmp_path: Path) -> None:
    """Completed disagreement cases should drive the next-step recommendation."""

    settings = _settings(tmp_path)
    review_root = settings.outputs_root / "reports" / "review" / "oasis_review_decision_log"
    review_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "rank": 1,
                "subject_id": "OAS1_0001",
                "session_id": "OAS1_0001_MR1",
                "label_name": "demented",
                "probability_score": "0.51",
                "reviewer_status": "completed",
                "reviewer_agrees_with_model": "no",
                "resolution_state": "open",
                "reviewer_priority": "high",
                "reviewer_decision": "nondemented_after_manual_review",
                "follow_up_action": "inspect_threshold_policy",
            }
        ]
    ).to_csv(review_root / "reviewer_decision_log.csv", index=False)

    summary = learning_module.build_oasis_reviewer_learning_report(settings=settings)

    assert summary["completed_case_count"] == 1
    assert summary["disagreement_count"] == 1
    assert summary["recommended_action"] == "analyze_disagreement_cases"


def test_build_oasis_reviewer_learning_report_handles_triaged_only_state(tmp_path: Path) -> None:
    """Triaged-only logs should recommend specialist review instead of learning actions."""

    settings = _settings(tmp_path)
    review_root = settings.outputs_root / "reports" / "review" / "oasis_review_decision_log"
    review_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "rank": 1,
                "subject_id": "OAS1_0001",
                "session_id": "OAS1_0001_MR1",
                "label_name": "demented",
                "probability_score": "0.51",
                "reviewer_status": "triaged",
                "reviewer_agrees_with_model": "",
                "resolution_state": "escalated",
                "reviewer_priority": "high",
                "reviewer_decision": "uncertain_non_clinical_triage",
                "follow_up_action": "needs_specialist_review",
            }
        ]
    ).to_csv(review_root / "reviewer_decision_log.csv", index=False)

    summary = learning_module.build_oasis_reviewer_learning_report(settings=settings)

    assert summary["completed_case_count"] == 0
    assert summary["triaged_case_count"] == 1
    assert summary["escalated_case_count"] == 1
    assert summary["recommended_action"] == "seek_specialist_review_capacity"
