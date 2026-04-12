"""Inspect a candidate OASIS-2 source root and save a readiness report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2_readiness import (  # noqa: E402
    build_oasis2_readiness_report,
    save_oasis2_readiness_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Check future OASIS-2 onboarding readiness without claiming that the "
            "dataset is already integrated."
        )
    )
    parser.add_argument("--source-root", type=Path, default=None, help="Optional OASIS-2 source directory override.")
    parser.add_argument(
        "--file-stem",
        type=str,
        default="oasis2_readiness",
        help="Output file stem under outputs/reports/readiness/.",
    )
    parser.add_argument("--max-examples", type=int, default=5, help="Maximum example paths/IDs to include in the report.")
    parser.add_argument("--no-save", action="store_true", help="Print a summary without writing JSON/Markdown artifacts.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on warnings as well as failures.")
    return parser


def main() -> None:
    """Run the OASIS-2 readiness inspection."""

    args = build_parser().parse_args()
    report = build_oasis2_readiness_report(
        source_root=args.source_root,
        max_examples=args.max_examples,
    )

    if not args.no_save:
        json_path, md_path = save_oasis2_readiness_report(report, file_stem=args.file_stem)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")

    summary = report.dataset_summary
    print(f"overall_status={report.overall_status}")
    print(f"source_root={report.source_root}")
    print(f"source_resolution={report.source_resolution}")
    print(f"supported_volume_file_count={summary.get('supported_volume_file_count', 0)}")
    print(f"metadata_file_count={summary.get('metadata_file_count', 0)}")
    print(f"unique_subject_id_count={summary.get('unique_subject_id_count', 0)}")
    print(f"longitudinal_subject_count={summary.get('longitudinal_subject_count', 0)}")
    print(f"recommendations={report.recommendations}")

    if report.overall_status == "fail" or (args.strict and report.overall_status != "pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
