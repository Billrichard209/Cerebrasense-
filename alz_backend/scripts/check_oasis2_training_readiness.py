"""Inspect whether OASIS-2 is honestly ready for supervised training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2_supervised import (  # noqa: E402
    build_oasis2_training_readiness_report,
    save_oasis2_training_readiness_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Check whether the current OASIS-2 labeled-prep manifest and split plan are safe to use "
            "for supervised training."
        )
    )
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional OASIS-2 labeled-prep manifest CSV override.")
    parser.add_argument("--split-plan-path", type=Path, default=None, help="Optional OASIS-2 split-plan CSV override.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--file-stem", type=str, default="oasis2_training_readiness")
    parser.add_argument("--no-save", action="store_true", help="Print a summary without writing JSON/Markdown artifacts.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on warnings as well as failures.")
    return parser


def main() -> None:
    """Build the readiness report and optionally save it."""

    args = build_parser().parse_args()
    report = build_oasis2_training_readiness_report(
        manifest_path=args.manifest_path,
        split_plan_path=args.split_plan_path,
        seed=args.seed,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    if not args.no_save:
        json_path, md_path = save_oasis2_training_readiness_report(report, file_stem=args.file_stem)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")

    print(f"overall_status={report.overall_status}")
    print(f"labeled_manifest_path={report.labeled_manifest_path}")
    print(f"split_plan_path={report.split_plan_path}")
    print(f"labeled_row_count={report.dataset_summary.get('labeled_row_count', 0)}")
    print(f"unlabeled_row_count={report.dataset_summary.get('unlabeled_row_count', 0)}")
    print(f"split_group_count={report.dataset_summary.get('split_group_count', 0)}")
    print(f"recommendations={report.recommendations}")

    if report.overall_status == "fail" or (args.strict and report.overall_status != "pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
