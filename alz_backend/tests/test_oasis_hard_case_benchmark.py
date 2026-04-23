"""Tests for the OASIS hard-case benchmark builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings

from scripts import build_oasis_hard_case_benchmark as benchmark_module


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


def test_build_oasis_hard_case_benchmark_writes_summary(tmp_path: Path) -> None:
    """The hard-case benchmark should summarize the escalated case set."""

    settings = _settings(tmp_path)
    handoff_root = settings.outputs_root / "reports" / "review" / "oasis_specialist_handoff_pack"
    handoff_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "rank": 1,
                "subject_id": "OAS1_0001",
                "session_id": "OAS1_0001_MR1",
                "label_name": "demented",
                "probability_score": "0.51",
                "confidence_level": "low",
                "reviewer_priority": "high",
            },
            {
                "rank": 2,
                "subject_id": "OAS1_0002",
                "session_id": "OAS1_0002_MR1",
                "label_name": "nondemented",
                "probability_score": "0.41",
                "confidence_level": "low",
                "reviewer_priority": "normal",
            },
        ]
    ).to_csv(handoff_root / "escalated_cases.csv", index=False)

    summary = benchmark_module.build_oasis_hard_case_benchmark(settings=settings)

    assert summary["hard_case_count"] == 2
    assert summary["high_priority_count"] == 1
    assert Path(summary["benchmark_cases_csv"]).exists()
    assert Path(summary["summary_json_path"]).exists()
    assert Path(summary["summary_md_path"]).exists()
