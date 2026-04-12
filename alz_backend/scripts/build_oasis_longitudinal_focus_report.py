"""Build an OASIS-first longitudinal focus report for the narrowed project goal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.oasis_longitudinal_focus import (  # noqa: E402
    build_oasis_longitudinal_focus_report,
    save_oasis_longitudinal_focus_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build an OASIS-first planning report that keeps OASIS-1 as the primary "
            "3D evidence base and treats OASIS-2 as the next dataset priority for "
            "real longitudinal strength."
        )
    )
    parser.add_argument(
        "--oasis-registry-path",
        type=Path,
        default=None,
        help="Optional active OASIS registry JSON override.",
    )
    parser.add_argument(
        "--file-stem",
        type=str,
        default="oasis_longitudinal_focus_report",
        help="Output file stem under outputs/reports/evidence/.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print a summary without writing JSON/Markdown artifacts.",
    )
    return parser


def main() -> None:
    """Build and optionally save the OASIS longitudinal focus report."""

    args = build_parser().parse_args()
    report = build_oasis_longitudinal_focus_report(
        oasis_registry_path=args.oasis_registry_path,
    )
    if not args.no_save:
        json_path, md_path = save_oasis_longitudinal_focus_report(report, file_stem=args.file_stem)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")

    print(f"goal_statement={report['goal_statement']}")
    print(f"classification_evidence_status={report['focus_assessment'].get('classification_evidence_status')}")
    print(f"longitudinal_evidence_status={report['focus_assessment'].get('longitudinal_evidence_status')}")
    print(f"next_dataset_priority={report['focus_assessment'].get('next_dataset_priority')}")
    print(
        "multi_session_subject_count="
        f"{report.get('current_oasis', {}).get('subject_summary', {}).get('multi_session_subject_count')}"
    )
    print(f"oasis2_readiness_status={report.get('oasis2_readiness', {}).get('overall_status')}")
    print(f"recommendations={report.get('recommendations', [])}")


if __name__ == "__main__":
    main()
