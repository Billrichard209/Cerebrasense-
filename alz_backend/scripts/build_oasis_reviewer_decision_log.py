"""Build a persistent reviewer decision log from one OASIS review pack."""

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

_MANUAL_COLUMNS = [
    "reviewer_status",
    "reviewer_decision",
    "reviewer_agrees_with_model",
    "reviewer_priority",
    "follow_up_action",
    "reviewer_notes",
    "reviewed_by",
    "reviewed_at",
    "resolution_state",
]

_DEFAULT_MANUAL_VALUES: dict[str, str] = {
    "reviewer_status": "pending",
    "reviewer_decision": "",
    "reviewer_agrees_with_model": "",
    "reviewer_priority": "normal",
    "follow_up_action": "",
    "reviewer_notes": "",
    "reviewed_by": "",
    "reviewed_at": "",
    "resolution_state": "open",
}


def _safe_name(value: str) -> str:
    """Return a path-safe name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _merge_existing_manual_fields(
    *,
    template: pd.DataFrame,
    existing_log_path: Path,
) -> pd.DataFrame:
    """Preserve reviewer-entered fields when the log is rebuilt."""

    if not existing_log_path.exists():
        return template

    existing = pd.read_csv(existing_log_path, dtype=str).fillna("")
    merge_columns = ["session_id", "scan_path"]
    existing_manual = existing[merge_columns + [column for column in _MANUAL_COLUMNS if column in existing.columns]].copy()
    merged = template.merge(existing_manual, on=merge_columns, how="left", suffixes=("", "_existing"))

    for column in _MANUAL_COLUMNS:
        existing_column = f"{column}_existing"
        if existing_column in merged.columns:
            merged[column] = merged[existing_column].where(merged[existing_column].astype(str).str.strip() != "", merged[column])
            merged = merged.drop(columns=[existing_column])

    return merged


def build_oasis_reviewer_decision_log(
    *,
    settings: AppSettings | None = None,
    review_pack_name: str = "oasis_review_pack",
    output_name: str = "oasis_review_decision_log",
) -> dict[str, Any]:
    """Build a reviewer decision log template from the latest OASIS review pack."""

    resolved_settings = settings or get_app_settings()
    safe_review_pack_name = _safe_name(review_pack_name)
    safe_output_name = _safe_name(output_name)

    review_root = resolved_settings.outputs_root / "reports" / "review" / safe_review_pack_name
    review_cases_csv = review_root / "review_cases.csv"
    review_summary_json = review_root / "review_pack_summary.json"
    if not review_cases_csv.exists():
        raise FileNotFoundError(f"Review cases CSV not found: {review_cases_csv}")
    if not review_summary_json.exists():
        raise FileNotFoundError(f"Review pack summary not found: {review_summary_json}")

    review_summary = json.loads(review_summary_json.read_text(encoding="utf-8"))
    cases = pd.read_csv(review_cases_csv).fillna("")
    if "rank" not in cases.columns:
        cases.insert(0, "rank", list(range(1, len(cases) + 1)))

    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "review" / safe_output_name)
    decision_log_csv = output_root / "reviewer_decision_log.csv"

    template = cases.copy()
    template["source_review_pack"] = safe_review_pack_name
    prediction_series = (
        template["prediction_json"].astype(str)
        if "prediction_json" in template.columns
        else pd.Series([""] * len(template), index=template.index, dtype="string")
    )
    template["source_prediction_json_copy"] = prediction_series.map(lambda value: Path(value).name if value else "")
    for column, default_value in _DEFAULT_MANUAL_VALUES.items():
        template[column] = default_value
        template[column] = template[column].astype("string")

    template = _merge_existing_manual_fields(template=template, existing_log_path=decision_log_csv)
    ordered_columns = [
        "rank",
        "subject_id",
        "session_id",
        "label_name",
        "probability_score",
        "confidence_level",
        "review_flag",
        "scan_path",
        "prediction_json",
        "source_prediction_json_copy",
        "source_review_pack",
        *_MANUAL_COLUMNS,
    ]
    ordered_columns = [column for column in ordered_columns if column in template.columns]
    decision_log = template[ordered_columns].sort_values(by=["rank"], kind="stable").reset_index(drop=True)
    for column in _MANUAL_COLUMNS:
        if column in decision_log.columns:
            decision_log[column] = decision_log[column].astype("string")
    decision_log.to_csv(decision_log_csv, index=False)

    pending_count = int((decision_log["reviewer_status"].astype(str).str.strip() == "pending").sum())
    completed_count = int((decision_log["reviewer_status"].astype(str).str.strip() == "completed").sum())
    escalated_count = int((decision_log["resolution_state"].astype(str).str.strip() == "escalated").sum())
    model_agreement_count = int((decision_log["reviewer_agrees_with_model"].astype(str).str.lower() == "yes").sum())
    model_disagreement_count = int((decision_log["reviewer_agrees_with_model"].astype(str).str.lower() == "no").sum())

    summary = {
        "review_pack_name": safe_review_pack_name,
        "output_name": safe_output_name,
        "review_pack_root": str(review_root),
        "decision_log_root": str(output_root),
        "review_cases_csv": str(review_cases_csv),
        "decision_log_csv": str(decision_log_csv),
        "review_case_count": int(len(decision_log)),
        "pending_case_count": pending_count,
        "completed_case_count": completed_count,
        "escalated_case_count": escalated_count,
        "model_agreement_count": model_agreement_count,
        "model_disagreement_count": model_disagreement_count,
        "notes": [
            "This decision log is a reviewer-facing template for the flagged low-confidence OASIS cases.",
            "Rebuilding the log preserves any existing reviewer-entered fields for matching session_id/scan_path rows.",
            "Use reviewer_status, reviewer_decision, and follow_up_action to turn uncertain predictions into structured feedback.",
        ],
        "cases": review_summary.get("cases", []),
    }
    summary_json_path = output_root / "reviewer_decision_log_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# OASIS Reviewer Decision Log",
        "",
        f"- review_pack_name: {safe_review_pack_name}",
        f"- review_case_count: {summary['review_case_count']}",
        f"- pending_case_count: {pending_count}",
        f"- completed_case_count: {completed_count}",
        f"- escalated_case_count: {escalated_count}",
        f"- decision_log_csv: {decision_log_csv}",
        "",
        "## Suggested Review Order",
        "",
    ]
    if summary["cases"]:
        for case in summary["cases"]:
            md_lines.append(
                f"- {case['rank']}. {case['subject_id']} / {case['session_id']} "
                f"({case['label_name']}, p={float(case['probability_score']):.3f}, confidence={case['confidence_level']})"
            )
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Notes", ""])
    md_lines.extend(f"- {note}" for note in summary["notes"])
    summary_md_path = output_root / "reviewer_decision_log_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_md_path"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build a persistent reviewer decision log from one OASIS review pack.")
    parser.add_argument("--review-pack-name", type=str, default="oasis_review_pack")
    parser.add_argument("--output-name", type=str, default="oasis_review_decision_log")
    return parser


def main() -> None:
    """Build the decision log and print a compact summary."""

    args = build_parser().parse_args()
    summary = build_oasis_reviewer_decision_log(
        review_pack_name=args.review_pack_name,
        output_name=args.output_name,
    )
    print(f"decision_log_root={summary['decision_log_root']}")
    print(f"decision_log_csv={summary['decision_log_csv']}")
    print(f"pending_case_count={summary['pending_case_count']}")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
