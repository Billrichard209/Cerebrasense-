"""Build a specialist-facing handoff pack for escalated OASIS review cases."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import AppSettings, get_app_settings  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402


def _safe_name(value: str) -> str:
    """Return a path-safe label."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _copy_if_exists(source: Path, destination: Path) -> bool:
    """Copy one file when it exists."""

    if not source.exists():
        return False
    shutil.copy2(source, destination)
    return True


def build_oasis_specialist_handoff_pack(
    *,
    settings: AppSettings | None = None,
    decision_log_name: str = "oasis_review_decision_log",
    review_pack_name: str = "oasis_review_pack",
    learning_report_name: str = "oasis_reviewer_learning_report",
    output_name: str = "oasis_specialist_handoff_pack",
) -> dict[str, Any]:
    """Build a specialist-facing handoff pack for escalated OASIS review cases."""

    resolved_settings = settings or get_app_settings()
    safe_decision_log_name = _safe_name(decision_log_name)
    safe_review_pack_name = _safe_name(review_pack_name)
    safe_learning_report_name = _safe_name(learning_report_name)
    safe_output_name = _safe_name(output_name)

    review_root = resolved_settings.outputs_root / "reports" / "review"
    decision_log_root = review_root / safe_decision_log_name
    review_pack_root = review_root / safe_review_pack_name
    learning_root = resolved_settings.outputs_root / "reports" / "reviewer_learning" / safe_learning_report_name

    decision_log_csv = decision_log_root / "reviewer_decision_log.csv"
    if not decision_log_csv.exists():
        raise FileNotFoundError(f"Reviewer decision log not found: {decision_log_csv}")

    decision_log = pd.read_csv(decision_log_csv, dtype=str).fillna("")
    escalated = decision_log.loc[decision_log["resolution_state"].astype(str).str.strip().str.lower() == "escalated"].copy()

    output_root = ensure_directory(review_root / safe_output_name)
    escalated_csv_path = output_root / "escalated_cases.csv"
    escalated.to_csv(escalated_csv_path, index=False)

    copied_prediction_files: list[str] = []
    for index, row in enumerate(escalated.to_dict(orient="records"), start=1):
        prediction_path = Path(str(row.get("prediction_json", ""))).expanduser()
        if prediction_path.exists():
            destination = output_root / f"{index:02d}_{prediction_path.name}"
            if _copy_if_exists(prediction_path, destination):
                copied_prediction_files.append(destination.name)

    copied_reference_files: list[str] = []
    for source in [
        review_pack_root / "review_pack_summary.md",
        decision_log_root / "reviewer_decision_log_summary.md",
        learning_root / "oasis_reviewer_learning_report.md",
        resolved_settings.outputs_root / "reports" / "presentation" / "oasis_local_path_summary.md",
        resolved_settings.outputs_root / "reports" / "workflows" / "oasis_local_workflow" / "workflow_summary.md",
    ]:
        if _copy_if_exists(source, output_root / source.name):
            copied_reference_files.append(source.name)

    cases: list[dict[str, Any]] = []
    for row in escalated.to_dict(orient="records"):
        cases.append(
            {
                "rank": row.get("rank"),
                "subject_id": row.get("subject_id"),
                "session_id": row.get("session_id"),
                "label_name": row.get("label_name"),
                "probability_score": row.get("probability_score"),
                "reviewer_priority": row.get("reviewer_priority"),
                "follow_up_action": row.get("follow_up_action"),
                "reviewer_notes": row.get("reviewer_notes"),
                "resolution_state": row.get("resolution_state"),
            }
        )

    summary = {
        "decision_log_name": safe_decision_log_name,
        "review_pack_name": safe_review_pack_name,
        "learning_report_name": safe_learning_report_name,
        "output_name": safe_output_name,
        "output_root": str(output_root),
        "escalated_case_count": int(len(escalated)),
        "escalated_cases_csv": str(escalated_csv_path),
        "copied_prediction_file_count": len(copied_prediction_files),
        "copied_prediction_files": copied_prediction_files,
        "copied_reference_files": copied_reference_files,
        "cases": cases,
        "notes": [
            "This handoff pack is for specialist follow-up on escalated low-confidence OASIS cases.",
            "These rows represent non-clinical triage/escalation, not adjudicated labels.",
            "Do not treat these cases as confirmed training labels unless a qualified reviewer signs off later.",
        ],
    }
    summary_json_path = output_root / "specialist_handoff_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# OASIS Specialist Handoff Pack",
        "",
        f"- escalated_case_count: {summary['escalated_case_count']}",
        f"- escalated_cases_csv: {escalated_csv_path}",
        f"- copied_prediction_file_count: {len(copied_prediction_files)}",
        "",
        "## Escalated Cases",
        "",
    ]
    if cases:
        for case in cases:
            probability = case.get("probability_score")
            probability_text = f"{float(probability):.3f}" if probability not in {"", None} else "n/a"
            md_lines.append(
                f"- {case.get('rank')}. {case.get('subject_id')} / {case.get('session_id')} "
                f"({case.get('label_name')}, p={probability_text}, priority={case.get('reviewer_priority')})"
            )
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Notes", ""])
    md_lines.extend(f"- {note}" for note in summary["notes"])
    summary_md_path = output_root / "specialist_handoff_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_md_path"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build a specialist-facing handoff pack for escalated OASIS review cases.")
    parser.add_argument("--decision-log-name", type=str, default="oasis_review_decision_log")
    parser.add_argument("--review-pack-name", type=str, default="oasis_review_pack")
    parser.add_argument("--learning-report-name", type=str, default="oasis_reviewer_learning_report")
    parser.add_argument("--output-name", type=str, default="oasis_specialist_handoff_pack")
    return parser


def main() -> None:
    """Build the specialist handoff pack and print a compact summary."""

    args = build_parser().parse_args()
    summary = build_oasis_specialist_handoff_pack(
        decision_log_name=args.decision_log_name,
        review_pack_name=args.review_pack_name,
        learning_report_name=args.learning_report_name,
        output_name=args.output_name,
    )
    print(f"output_root={summary['output_root']}")
    print(f"escalated_case_count={summary['escalated_case_count']}")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
