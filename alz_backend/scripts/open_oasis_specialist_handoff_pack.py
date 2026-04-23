"""Open the key OASIS specialist handoff artifacts in default Windows apps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import AppSettings, get_app_settings  # noqa: E402


def _safe_name(value: str) -> str:
    """Return a path-safe label."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def open_oasis_specialist_handoff_pack(
    *,
    settings: AppSettings | None = None,
    output_name: str = "oasis_specialist_handoff_pack",
) -> list[Path]:
    """Open the key specialist handoff artifacts."""

    resolved_settings = settings or get_app_settings()
    pack_root = resolved_settings.outputs_root / "reports" / "review" / _safe_name(output_name)
    summary_md = pack_root / "specialist_handoff_summary.md"
    escalated_csv = pack_root / "escalated_cases.csv"
    learning_md = pack_root / "oasis_reviewer_learning_report.md"
    decision_md = pack_root / "reviewer_decision_log_summary.md"
    review_pack_md = pack_root / "review_pack_summary.md"

    paths = [summary_md, escalated_csv, learning_md, decision_md, review_pack_md, pack_root]
    missing = [path for path in paths[:-1] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Specialist handoff artifacts missing: {', '.join(str(path) for path in missing)}")

    for path in paths:
        os.startfile(str(path))  # type: ignore[attr-defined]
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Open the key OASIS specialist handoff artifacts.")
    parser.add_argument("--output-name", type=str, default="oasis_specialist_handoff_pack")
    return parser


def main() -> None:
    """Open the handoff pack and print the paths."""

    args = build_parser().parse_args()
    opened = open_oasis_specialist_handoff_pack(output_name=args.output_name)
    print("opened=" + json.dumps([str(path) for path in opened], indent=2))


if __name__ == "__main__":
    main()
