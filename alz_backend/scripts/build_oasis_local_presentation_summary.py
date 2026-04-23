"""Build a compact presentation/status summary from the local OASIS workflow artifacts."""

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
    """Return a path-safe workflow label."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def build_oasis_local_presentation_summary(
    *,
    settings: AppSettings | None = None,
    workflow_name: str = "oasis_local_workflow",
    output_name: str = "oasis_local_path_summary",
) -> dict[str, Any]:
    """Build a clean OASIS-only status summary from one local workflow run."""

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

    workflow_summary = _load_json(workflow_summary_path)
    batch_predictions_csv = workflow_summary.get("batch_predictions_csv")
    if not batch_predictions_csv:
        raise FileNotFoundError("The workflow summary does not include a batch_predictions_csv path.")
    batch_predictions_path = Path(str(batch_predictions_csv)).expanduser().resolve()
    if not batch_predictions_path.exists():
        raise FileNotFoundError(f"Batch predictions CSV not found: {batch_predictions_path}")

    batch_frame = pd.read_csv(batch_predictions_path)
    ok_rows = batch_frame.loc[batch_frame["status"] == "ok"].copy()
    review_rows = ok_rows.loc[ok_rows["review_flag"] == True].copy()  # noqa: E712

    label_counts = {
        str(key): int(value)
        for key, value in ok_rows["label_name"].value_counts(dropna=True).items()
    }
    confidence_counts = {
        str(key): int(value)
        for key, value in ok_rows["confidence_level"].value_counts(dropna=True).items()
    }
    review_examples = review_rows[
        ["subject_id", "session_id", "label_name", "probability_score", "confidence_level"]
    ].head(8)

    summary_root = ensure_directory(resolved_settings.outputs_root / "reports" / "presentation")
    summary_path = summary_root / f"{safe_output_name}.json"
    summary_md_path = summary_root / f"{safe_output_name}.md"

    summary = {
        "workflow_name": safe_workflow_name,
        "output_name": safe_output_name,
        "run_name": workflow_summary.get("run_name"),
        "workflow_root": workflow_summary.get("workflow_root"),
        "demo_bundle_root": workflow_summary.get("demo_bundle_root"),
        "batch_report_root": workflow_summary.get("batch_report_root"),
        "batch_predictions_csv": str(batch_predictions_path),
        "scan_count": int(len(batch_frame)),
        "succeeded": int((batch_frame["status"] == "ok").sum()),
        "failed": int((batch_frame["status"] != "ok").sum()),
        "review_required": int(len(review_rows)),
        "label_counts": label_counts,
        "confidence_counts": confidence_counts,
        "review_examples": review_examples.to_dict(orient="records"),
        "status": "stable_local_oasis_path",
        "recommendation": "Keep this as the stable OASIS-only local workflow while OASIS-2 cloud training is paused.",
        "notes": [
            "This summary is built from the local OASIS workflow artifacts only.",
            "It is intended to replace ad hoc terminal inspection with one compact status readout.",
            "Review-flagged scans are not failures; they are the subset that should be manually checked first.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# OASIS Local Path Summary",
        "",
        f"- run_name: {summary['run_name']}",
        f"- workflow_root: {summary['workflow_root']}",
        f"- demo_bundle_root: {summary['demo_bundle_root']}",
        f"- batch_report_root: {summary['batch_report_root']}",
        f"- scan_count: {summary['scan_count']}",
        f"- succeeded: {summary['succeeded']}",
        f"- failed: {summary['failed']}",
        f"- review_required: {summary['review_required']}",
        f"- recommendation: {summary['recommendation']}",
        "",
        "## Label Counts",
        "",
    ]
    if label_counts:
        md_lines.extend(f"- {label}: {count}" for label, count in sorted(label_counts.items()))
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Confidence Counts", ""])
    if confidence_counts:
        md_lines.extend(f"- {level}: {count}" for level, count in sorted(confidence_counts.items()))
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Review Examples", ""])
    if review_examples.empty:
        md_lines.append("- none")
    else:
        for row in review_examples.to_dict(orient="records"):
            md_lines.append(
                f"- {row['subject_id']} / {row['session_id']}: {row['label_name']} "
                f"(p={float(row['probability_score']):.3f}, confidence={row['confidence_level']})"
            )
    md_lines.extend(["", "## Notes", ""])
    md_lines.extend(f"- {note}" for note in summary["notes"])
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary["summary_json_path"] = str(summary_path)
    summary["summary_md_path"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build a presentation/status summary from the local OASIS workflow.")
    parser.add_argument("--workflow-name", type=str, default="oasis_local_workflow")
    parser.add_argument("--output-name", type=str, default="oasis_local_path_summary")
    return parser


def main() -> None:
    """Build the summary and print a compact result."""

    args = build_parser().parse_args()
    summary = build_oasis_local_presentation_summary(
        workflow_name=args.workflow_name,
        output_name=args.output_name,
    )
    print(f"summary_json_path={summary['summary_json_path']}")
    print(f"summary_md_path={summary['summary_md_path']}")
    print(f"scan_count={summary['scan_count']}")
    print(f"review_required={summary['review_required']}")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
