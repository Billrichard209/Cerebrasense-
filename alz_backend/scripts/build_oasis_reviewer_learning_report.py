"""Build an OASIS reviewer-learning report from the local reviewer decision log."""

from __future__ import annotations

import argparse
import json
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
    """Return a path-safe name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _normalized(series: pd.Series) -> pd.Series:
    """Normalize one text column for comparisons."""

    return series.fillna("").astype(str).str.strip().str.lower()


def _recommended_action(*, completed_count: int, disagreement_count: int) -> str:
    """Return the next best reviewer-learning action."""

    if completed_count == 0:
        return "fill_reviewer_decision_log"
    if disagreement_count > 0:
        return "analyze_disagreement_cases"
    if completed_count < 5:
        return "collect_more_reviewed_cases"
    return "consider_threshold_review_experiment"


def build_oasis_reviewer_learning_report(
    *,
    settings: AppSettings | None = None,
    decision_log_name: str = "oasis_review_decision_log",
    output_name: str = "oasis_reviewer_learning_report",
) -> dict[str, Any]:
    """Build an OASIS reviewer-learning report from the local decision log."""

    resolved_settings = settings or get_app_settings()
    safe_decision_log_name = _safe_name(decision_log_name)
    safe_output_name = _safe_name(output_name)

    review_root = resolved_settings.outputs_root / "reports" / "review" / safe_decision_log_name
    decision_log_csv = review_root / "reviewer_decision_log.csv"
    if not decision_log_csv.exists():
        raise FileNotFoundError(f"Reviewer decision log not found: {decision_log_csv}")

    frame = pd.read_csv(decision_log_csv, dtype=str).fillna("")
    reviewer_status = _normalized(frame.get("reviewer_status", pd.Series(dtype="string")))
    reviewer_agreement = _normalized(frame.get("reviewer_agrees_with_model", pd.Series(dtype="string")))
    resolution_state = _normalized(frame.get("resolution_state", pd.Series(dtype="string")))

    completed = frame.loc[reviewer_status == "completed"].copy()
    pending = frame.loc[reviewer_status != "completed"].copy()
    disagreements = completed.loc[reviewer_agreement == "no"].copy()
    agreements = completed.loc[reviewer_agreement == "yes"].copy()
    escalated = frame.loc[resolution_state == "escalated"].copy()

    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "reviewer_learning" / safe_output_name)
    completed_cases_csv = output_root / "completed_review_cases.csv"
    pending_cases_csv = output_root / "pending_review_cases.csv"
    disagreements_csv = output_root / "model_disagreement_cases.csv"
    completed.to_csv(completed_cases_csv, index=False)
    pending.to_csv(pending_cases_csv, index=False)
    disagreements.to_csv(disagreements_csv, index=False)

    review_case_count = int(len(frame))
    completed_count = int(len(completed))
    pending_count = int(len(pending))
    agreement_count = int(len(agreements))
    disagreement_count = int(len(disagreements))
    escalated_count = int(len(escalated))
    agreement_rate = (agreement_count / completed_count) if completed_count else None
    disagreement_rate = (disagreement_count / completed_count) if completed_count else None
    recommended_action = _recommended_action(completed_count=completed_count, disagreement_count=disagreement_count)

    pending_preview = pending[
        [column for column in ["rank", "subject_id", "session_id", "label_name", "probability_score", "reviewer_priority"] if column in pending.columns]
    ].head(5)
    disagreement_preview = disagreements[
        [column for column in ["rank", "subject_id", "session_id", "label_name", "reviewer_decision", "follow_up_action"] if column in disagreements.columns]
    ].head(5)

    summary = {
        "decision_log_name": safe_decision_log_name,
        "output_name": safe_output_name,
        "decision_log_csv": str(decision_log_csv),
        "output_root": str(output_root),
        "review_case_count": review_case_count,
        "completed_case_count": completed_count,
        "pending_case_count": pending_count,
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "escalated_case_count": escalated_count,
        "agreement_rate": agreement_rate,
        "disagreement_rate": disagreement_rate,
        "recommended_action": recommended_action,
        "completed_cases_csv": str(completed_cases_csv),
        "pending_cases_csv": str(pending_cases_csv),
        "model_disagreement_cases_csv": str(disagreements_csv),
        "pending_preview": pending_preview.to_dict(orient="records"),
        "disagreement_preview": disagreement_preview.to_dict(orient="records"),
        "notes": [
            "This report is based on the local reviewer decision log, not on automatic model updates.",
            "Use it to track review throughput, disagreement patterns, and whether the current threshold/review policy needs a later experiment.",
            "No serving threshold should change directly from this report without a separate validation step.",
        ],
    }
    summary_json_path = output_root / "oasis_reviewer_learning_report.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# OASIS Reviewer Learning Report",
        "",
        f"- decision_log_name: {safe_decision_log_name}",
        f"- review_case_count: {review_case_count}",
        f"- completed_case_count: {completed_count}",
        f"- pending_case_count: {pending_count}",
        f"- disagreement_count: {disagreement_count}",
        f"- escalated_case_count: {escalated_count}",
        f"- recommended_action: {recommended_action}",
        "",
        "## Interpretation",
        "",
    ]
    if completed_count == 0:
        md_lines.extend(
            [
                "- No reviewer-completed cases exist yet, so this report is acting as a readiness/throughput checkpoint.",
                "- The next best move is to review the highest-priority pending cases in the reviewer decision log.",
            ]
        )
    else:
        md_lines.extend(
            [
                f"- Reviewer/model agreement rate: {agreement_rate:.3f}" if agreement_rate is not None else "- Reviewer/model agreement rate: n/a",
                f"- Reviewer/model disagreement rate: {disagreement_rate:.3f}" if disagreement_rate is not None else "- Reviewer/model disagreement rate: n/a",
            ]
        )
    md_lines.extend(["", "## Pending Priority Cases", ""])
    if summary["pending_preview"]:
        for row in summary["pending_preview"]:
            probability = row.get("probability_score")
            probability_text = f"{float(probability):.3f}" if probability not in {"", None} else "n/a"
            md_lines.append(
                f"- {row.get('rank')}. {row.get('subject_id')} / {row.get('session_id')} "
                f"({row.get('label_name')}, p={probability_text}, priority={row.get('reviewer_priority', 'normal')})"
            )
    else:
        md_lines.append("- none")
    if summary["disagreement_preview"]:
        md_lines.extend(["", "## Disagreement Preview", ""])
        for row in summary["disagreement_preview"]:
            md_lines.append(
                f"- {row.get('rank')}. {row.get('subject_id')} / {row.get('session_id')}: "
                f"reviewer_decision={row.get('reviewer_decision')} follow_up_action={row.get('follow_up_action')}"
            )
    md_lines.extend(["", "## Notes", ""])
    md_lines.extend(f"- {note}" for note in summary["notes"])
    summary_md_path = output_root / "oasis_reviewer_learning_report.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_md_path"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build an OASIS reviewer-learning report from the local decision log.")
    parser.add_argument("--decision-log-name", type=str, default="oasis_review_decision_log")
    parser.add_argument("--output-name", type=str, default="oasis_reviewer_learning_report")
    return parser


def main() -> None:
    """Build the learning report and print a compact summary."""

    args = build_parser().parse_args()
    summary = build_oasis_reviewer_learning_report(
        decision_log_name=args.decision_log_name,
        output_name=args.output_name,
    )
    print(f"output_root={summary['output_root']}")
    print(f"recommended_action={summary['recommended_action']}")
    print(f"pending_case_count={summary['pending_case_count']}")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
