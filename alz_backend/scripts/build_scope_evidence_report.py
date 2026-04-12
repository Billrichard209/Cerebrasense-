"""Build a scope-aligned evidence report for the current backend goal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evidence_reporting import (  # noqa: E402
    build_scope_aligned_evidence_report,
    save_scope_aligned_evidence_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build one scope-aligned evidence report that keeps OASIS as the primary "
            "3D evidence track and Kaggle as a separate 2D comparison branch."
        )
    )
    parser.add_argument("--kaggle-run-name", type=str, default=None, help="Optional explicit Kaggle run name.")
    parser.add_argument("--oasis-registry-path", type=Path, default=None, help="Optional active OASIS registry JSON override.")
    parser.add_argument(
        "--oasis-repeated-split-study-path",
        type=Path,
        default=None,
        help="Optional repeated-split study_summary.json override.",
    )
    parser.add_argument(
        "--kaggle-final-metrics-path",
        type=Path,
        default=None,
        help="Optional Kaggle final_metrics.json override.",
    )
    parser.add_argument("--file-stem", type=str, default="scope_aligned_evidence_report")
    parser.add_argument("--no-save", action="store_true", help="Print a summary without writing report files.")
    return parser


def main() -> None:
    """Build and optionally save the scope-aligned evidence report."""

    args = build_parser().parse_args()
    report = build_scope_aligned_evidence_report(
        kaggle_run_name=args.kaggle_run_name,
        oasis_registry_path=args.oasis_registry_path,
        oasis_repeated_split_study_path=args.oasis_repeated_split_study_path,
        kaggle_final_metrics_path=args.kaggle_final_metrics_path,
    )
    if not args.no_save:
        json_path, md_path = save_scope_aligned_evidence_report(report, file_stem=args.file_stem)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")

    print(f"goal_statement={report['goal_statement']}")
    print(f"oasis_run_name={report['oasis_primary'].get('run_name')}")
    print(f"kaggle_run_name={report['kaggle_secondary'].get('run_name')}")
    print(f"oasis_test_auroc={report['oasis_primary'].get('test_metrics', {}).get('auroc')}")
    print(f"kaggle_test_macro_ovr_auroc={report['kaggle_secondary'].get('test_metrics', {}).get('macro_ovr_auroc')}")
    print(f"recommendations={report.get('recommendations', [])}")


if __name__ == "__main__":
    main()
