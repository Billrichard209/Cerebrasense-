"""Open the key local OASIS workflow artifacts in the default Windows apps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import AppSettings, get_app_settings  # noqa: E402


def _safe_name(value: str) -> str:
    """Return a path-safe workflow label."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _resolve_workflow_summary_md_path(workflow_summary: dict[str, Any], workflow_summary_path: Path) -> Path:
    """Resolve the workflow markdown path even for older summary JSON files."""

    raw_value = workflow_summary.get("summary_md_path")
    if isinstance(raw_value, str) and raw_value.strip():
        return Path(raw_value).expanduser().resolve()
    return workflow_summary_path.with_name("workflow_summary.md").resolve()


def open_oasis_local_outputs(
    *,
    settings: AppSettings | None = None,
    workflow_name: str = "oasis_local_workflow",
    summary_name: str = "oasis_local_path_summary",
) -> list[Path]:
    """Open the key local OASIS workflow outputs."""

    resolved_settings = settings or get_app_settings()
    workflow_summary_path = (
        resolved_settings.outputs_root
        / "reports"
        / "workflows"
        / _safe_name(workflow_name)
        / "workflow_summary.json"
    )
    summary_path = (
        resolved_settings.outputs_root
        / "reports"
        / "presentation"
        / f"{_safe_name(summary_name)}.md"
    )
    if not workflow_summary_path.exists():
        raise FileNotFoundError(f"Workflow summary not found: {workflow_summary_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Presentation summary not found: {summary_path}")

    workflow_summary = _load_json(workflow_summary_path)
    workflow_md_path = _resolve_workflow_summary_md_path(workflow_summary, workflow_summary_path)
    if not workflow_md_path.exists():
        raise FileNotFoundError(f"Workflow markdown summary not found: {workflow_md_path}")
    review_summary_path = (
        resolved_settings.outputs_root
        / "reports"
        / "review"
        / "oasis_review_pack"
        / "review_pack_summary.md"
    )
    reviewer_log_summary_path = (
        resolved_settings.outputs_root
        / "reports"
        / "review"
        / "oasis_review_decision_log"
        / "reviewer_decision_log_summary.md"
    )
    reviewer_learning_summary_path = (
        resolved_settings.outputs_root
        / "reports"
        / "reviewer_learning"
        / "oasis_reviewer_learning_report"
        / "oasis_reviewer_learning_report.md"
    )
    specialist_handoff_summary_path = (
        resolved_settings.outputs_root
        / "reports"
        / "review"
        / "oasis_specialist_handoff_pack"
        / "specialist_handoff_summary.md"
    )
    hard_case_benchmark_summary_path = (
        resolved_settings.outputs_root
        / "reports"
        / "benchmark"
        / "oasis_hard_case_benchmark"
        / "hard_case_benchmark_summary.md"
    )
    paths = [
        summary_path,
        workflow_md_path,
        Path(str(workflow_summary["batch_predictions_csv"])).resolve(),
        Path(str(workflow_summary["demo_bundle_root"])).resolve(),
        Path(str(workflow_summary["batch_report_root"])).resolve(),
    ]
    if review_summary_path.exists():
        paths.append(review_summary_path)
    if reviewer_log_summary_path.exists():
        paths.append(reviewer_log_summary_path)
    if reviewer_learning_summary_path.exists():
        paths.append(reviewer_learning_summary_path)
    if specialist_handoff_summary_path.exists():
        paths.append(specialist_handoff_summary_path)
    if hard_case_benchmark_summary_path.exists():
        paths.append(hard_case_benchmark_summary_path)
    for path in paths:
        os.startfile(str(path))  # type: ignore[attr-defined]
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Open the key local OASIS workflow outputs.")
    parser.add_argument("--workflow-name", type=str, default="oasis_local_workflow")
    parser.add_argument("--summary-name", type=str, default="oasis_local_path_summary")
    return parser


def main() -> None:
    """Open the outputs and print the paths."""

    args = build_parser().parse_args()
    opened = open_oasis_local_outputs(
        workflow_name=args.workflow_name,
        summary_name=args.summary_name,
    )
    print("opened=" + json.dumps([str(path) for path in opened], indent=2))


if __name__ == "__main__":
    main()
