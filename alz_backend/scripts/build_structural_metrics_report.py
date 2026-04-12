"""Build structural metrics reports from external neuroimaging outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.volumetrics.structural import (  # noqa: E402
    build_freesurfer_structural_report,
    compare_report_to_reference_ranges,
    load_structural_reference_ranges,
    save_structural_metrics_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build a structural metrics report from externally generated FreeSurfer stats files."
    )
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--aseg-stats-path", type=Path, default=None)
    parser.add_argument("--lh-aparc-stats-path", type=Path, default=None)
    parser.add_argument("--rh-aparc-stats-path", type=Path, default=None)
    parser.add_argument("--reference-ranges", type=Path, default=None)
    parser.add_argument("--output-stem", type=str, default=None)
    return parser


def main() -> None:
    """Build and save the structural metrics report."""

    args = build_parser().parse_args()
    report = build_freesurfer_structural_report(
        subject_id=args.subject_id,
        session_id=args.session_id,
        aseg_stats_path=args.aseg_stats_path,
        lh_aparc_stats_path=args.lh_aparc_stats_path,
        rh_aparc_stats_path=args.rh_aparc_stats_path,
    )
    if args.reference_ranges is not None:
        report.reference_comparisons = compare_report_to_reference_ranges(
            report,
            load_structural_reference_ranges(args.reference_ranges),
        )
    output_path = save_structural_metrics_report(report, file_stem=args.output_stem)
    print(f"report={output_path}")
    print(f"subject_id={report.subject_id}")
    print(f"session_id={report.session_id}")
    print(f"hippocampal_volume_count={len(report.hippocampal_volumes)}")
    print(f"cortical_thickness_count={len(report.cortical_thickness)}")
    print(f"brain_region_volume_count={len(report.brain_region_volumes)}")
    print(f"global_measure_count={len(report.global_measures)}")
    print(f"reference_comparison_count={len(report.reference_comparisons)}")
    print(f"warning_count={len(report.warnings)}")


if __name__ == "__main__":
    main()
