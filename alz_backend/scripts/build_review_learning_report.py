"""Build and save an advisory reviewer-outcome learning report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services import build_review_learning_payload  # noqa: E402
from src.configs.runtime import get_app_settings  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402

_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str) -> str:
    normalized = _SAFE_NAME_PATTERN.sub("_", value.strip())
    return normalized.strip("._") or "review_learning_report"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build an advisory learning report from reviewed cases. "
            "Suggestions are reviewer-guided cues for later validation, not automatic threshold changes."
        )
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--active-model-only", action="store_true")
    parser.add_argument("--selection-metric", type=str, default="balanced_accuracy")
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--report-name", type=str, default="review_learning_report")
    return parser


def main() -> None:
    """Build and save the reviewer learning report."""

    args = build_parser().parse_args()
    payload = build_review_learning_payload(
        limit=args.limit,
        model_name=args.model_name,
        active_model_only=args.active_model_only,
        selection_metric=args.selection_metric,
        threshold_step=args.threshold_step,
    )
    settings = get_app_settings()
    output_root = ensure_directory(
        settings.outputs_root / "reports" / "reviewer_learning" / _safe_name(args.report_name)
    )
    report_path = output_root / "review_learning_report.json"
    threshold_grid_path = output_root / "threshold_grid.csv"
    summary_path = output_root / "summary.md"

    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(payload.get("threshold_grid", [])).to_csv(threshold_grid_path, index=False)
    summary_lines = [
        f"# Review Learning Report: {args.report_name}",
        "",
        "This report is advisory only. It should inform validation experiments and retraining review, not directly change the live model.",
        "",
        f"- scope: {payload.get('scope')}",
        f"- current_threshold: {payload.get('current_threshold')}",
        f"- override_rate: {payload.get('override_rate')}",
        f"- reviewer_labeled_samples: {payload.get('reviewer_labeled_samples')}",
        f"- recommended_action: {payload.get('recommended_action')}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"output_root={output_root}")
    print(f"report_json={report_path}")
    print(f"threshold_grid_csv={threshold_grid_path}")
    print(f"summary_md={summary_path}")
    print(f"recommended_action={payload.get('recommended_action')}")


if __name__ == "__main__":
    main()
