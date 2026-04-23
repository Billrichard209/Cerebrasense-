"""Build a focused review pack from one local OASIS batch inference run."""

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
    """Return a path-safe name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _copy_if_exists(source: Path | None, destination: Path) -> None:
    """Copy one file when present."""

    if source is not None and source.exists():
        shutil.copy2(source, destination)


def build_oasis_review_pack(
    *,
    settings: AppSettings | None = None,
    workflow_name: str = "oasis_local_workflow",
    output_name: str = "oasis_review_pack",
    limit: int | None = None,
) -> dict[str, Any]:
    """Build a focused review pack from the current local OASIS workflow batch CSV."""

    resolved_settings = settings or get_app_settings()
    safe_workflow_name = _safe_name(workflow_name)
    safe_output_name = _safe_name(output_name)

    workflow_summary_path = (
        resolved_settings.outputs_root
        / "reports"
        / "workflows"
        / safe_workflow_name
        / "workflow_summary.json"
    )
    if not workflow_summary_path.exists():
        raise FileNotFoundError(f"Workflow summary not found: {workflow_summary_path}")

    workflow_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))
    batch_predictions_path = Path(str(workflow_summary["batch_predictions_csv"])).expanduser().resolve()
    if not batch_predictions_path.exists():
        raise FileNotFoundError(f"Batch predictions CSV not found: {batch_predictions_path}")

    frame = pd.read_csv(batch_predictions_path)
    flagged = frame.loc[(frame["status"] == "ok") & (frame["review_flag"] == True)].copy()  # noqa: E712
    if "probability_score" in flagged.columns:
        flagged["distance_from_decision_boundary"] = (flagged["probability_score"].astype(float) - 0.5).abs()
        flagged = flagged.sort_values(
            by=["distance_from_decision_boundary", "probability_score"],
            ascending=[True, False],
            kind="stable",
        )
    if limit is not None:
        flagged = flagged.head(limit)

    pack_root = ensure_directory(resolved_settings.outputs_root / "reports" / "review" / safe_output_name)
    review_csv_path = pack_root / "review_cases.csv"
    flagged.to_csv(review_csv_path, index=False)

    copied_prediction_files: list[str] = []
    case_rows: list[dict[str, Any]] = []
    for index, row in enumerate(flagged.to_dict(orient="records"), start=1):
        prediction_json = row.get("prediction_json")
        prediction_source = None if prediction_json in {None, ""} else Path(str(prediction_json)).expanduser().resolve()
        prediction_copy_name = None
        if prediction_source is not None and prediction_source.exists():
            prediction_copy_name = f"{index:02d}_{prediction_source.name}"
            _copy_if_exists(prediction_source, pack_root / prediction_copy_name)
            copied_prediction_files.append(prediction_copy_name)
        case_rows.append(
            {
                "rank": index,
                "subject_id": row.get("subject_id"),
                "session_id": row.get("session_id"),
                "label_name": row.get("label_name"),
                "probability_score": row.get("probability_score"),
                "confidence_level": row.get("confidence_level"),
                "scan_path": row.get("scan_path"),
                "prediction_json_copy": prediction_copy_name,
            }
        )

    summary = {
        "workflow_name": safe_workflow_name,
        "output_name": safe_output_name,
        "pack_root": str(pack_root),
        "workflow_summary_path": str(workflow_summary_path),
        "batch_predictions_csv": str(batch_predictions_path),
        "review_case_count": int(len(flagged)),
        "copied_prediction_file_count": len(copied_prediction_files),
        "review_cases_csv": str(review_csv_path),
        "cases": case_rows,
        "notes": [
            "This pack contains only scans that were flagged for manual review in the latest local OASIS workflow.",
            "Cases are ordered from closest to the decision boundary outward, so the most ambiguous cases appear first.",
        ],
    }
    summary_json_path = pack_root / "review_pack_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# OASIS Review Pack",
        "",
        f"- workflow_name: {safe_workflow_name}",
        f"- review_case_count: {summary['review_case_count']}",
        f"- review_cases_csv: {review_csv_path}",
        "",
        "## Cases",
        "",
    ]
    if case_rows:
        for row in case_rows:
            probability = float(row["probability_score"]) if row["probability_score"] is not None else float("nan")
            md_lines.append(
                f"- {row['rank']}. {row['subject_id']} / {row['session_id']}: {row['label_name']} "
                f"(p={probability:.3f}, confidence={row['confidence_level']})"
            )
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Notes", ""])
    md_lines.extend(f"- {note}" for note in summary["notes"])
    summary_md_path = pack_root / "review_pack_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_md_path"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build a focused review pack from the local OASIS batch workflow.")
    parser.add_argument("--workflow-name", type=str, default="oasis_local_workflow")
    parser.add_argument("--output-name", type=str, default="oasis_review_pack")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    """Build the review pack and print a compact summary."""

    args = build_parser().parse_args()
    summary = build_oasis_review_pack(
        workflow_name=args.workflow_name,
        output_name=args.output_name,
        limit=args.limit,
    )
    print(f"pack_root={summary['pack_root']}")
    print(f"review_case_count={summary['review_case_count']}")
    print(f"review_cases_csv={summary['review_cases_csv']}")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
