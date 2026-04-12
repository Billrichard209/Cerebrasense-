"""Build a timeline-ready longitudinal report from CSV or structural JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.longitudinal.service import (  # noqa: E402
    build_and_save_longitudinal_report,
    load_feature_configs,
    records_from_csv,
    records_from_structural_summary_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build a subject-level longitudinal tracking report.")
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--input-csv", type=Path, default=None, help="CSV with subject_id plus vol__/prob__ feature columns.")
    parser.add_argument("--structural-summary-json", type=Path, default=None, help="Existing OASIS structural longitudinal JSON.")
    parser.add_argument("--feature-config-json", type=Path, default=None, help="Optional trend feature config JSON.")
    parser.add_argument("--output-stem", type=str, default=None)
    return parser


def main() -> None:
    """Build and save the longitudinal report."""

    args = build_parser().parse_args()
    if args.input_csv is None and args.structural_summary_json is None:
        raise ValueError("Provide --input-csv or --structural-summary-json.")

    records = []
    if args.input_csv is not None:
        records.extend(records_from_csv(args.input_csv, subject_id=args.subject_id))
    if args.structural_summary_json is not None:
        records.extend(records_from_structural_summary_json(args.structural_summary_json))

    report, output_path = build_and_save_longitudinal_report(
        records,
        subject_id=args.subject_id,
        feature_configs=load_feature_configs(args.feature_config_json),
        file_stem=args.output_stem,
    )
    print(f"report={output_path}")
    print(f"subject_id={report.subject_id}")
    print(f"timepoint_count={report.timepoint_count}")
    print(f"interval_change_count={len(report.interval_changes)}")
    print(f"trend_count={len(report.trend_summaries)}")
    print(f"alert_count={len(report.alerts)}")
    print(f"warnings={report.warnings}")


if __name__ == "__main__":
    main()

