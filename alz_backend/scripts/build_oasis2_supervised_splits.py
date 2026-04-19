"""Materialize subject-safe supervised OASIS-2 train/val/test manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2_supervised import (  # noqa: E402
    OASIS2SupervisedError,
    OASIS2SupervisedSplitConfig,
    build_oasis2_supervised_split_artifacts,
    build_oasis2_training_readiness_report,
    save_oasis2_training_readiness_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build subject-safe supervised OASIS-2 train/val/test manifests from the labeled-prep manifest "
            "and split plan."
        )
    )
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional OASIS-2 labeled-prep manifest CSV override.")
    parser.add_argument("--split-plan-path", type=Path, default=None, help="Optional OASIS-2 split-plan CSV override.")
    parser.add_argument("--reports-root", type=Path, default=None, help="Optional output reports root override.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser


def main() -> None:
    """Build supervised split artifacts and print a compact summary."""

    args = build_parser().parse_args()
    try:
        artifacts = build_oasis2_supervised_split_artifacts(
            OASIS2SupervisedSplitConfig(
                manifest_path=args.manifest_path,
                split_plan_path=args.split_plan_path,
                reports_root=args.reports_root,
                seed=args.seed,
                split_seed=args.split_seed,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
            )
        )
    except OASIS2SupervisedError as error:
        readiness_report = build_oasis2_training_readiness_report(
            manifest_path=args.manifest_path,
            split_plan_path=args.split_plan_path,
            seed=args.seed,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
        json_path, md_path = save_oasis2_training_readiness_report(readiness_report)
        print(
            "OASIS-2 supervised split materialization is blocked. "
            f"{error} See {md_path} (JSON: {json_path}).",
            file=sys.stderr,
        )
        raise SystemExit(1) from error

    print(f"report_root={artifacts.report_root}")
    print(f"train_manifest_path={artifacts.train_manifest_path}")
    print(f"val_manifest_path={artifacts.val_manifest_path}")
    print(f"test_manifest_path={artifacts.test_manifest_path}")
    print("summary=" + json.dumps(artifacts.summary_payload, indent=2))


if __name__ == "__main__":
    main()
