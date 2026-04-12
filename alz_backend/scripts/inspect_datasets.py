"""CLI entry point for separate OASIS and Kaggle dataset inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inspectors import inspect_datasets


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Inspect OASIS and Kaggle datasets separately.")
    parser.add_argument(
        "--dataset",
        choices=("all", "oasis", "kaggle"),
        default="all",
        help="Dataset branch to inspect.",
    )
    parser.add_argument(
        "--max-intensity-samples",
        type=int,
        default=8,
        help="How many scans/images to load fully for intensity statistics.",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=4,
        help="How many sample preview PNGs to save per dataset.",
    )
    return parser


def main() -> None:
    """Run the dataset inspector and print the saved artifact paths."""

    parser = build_parser()
    args = parser.parse_args()
    reports = inspect_datasets(
        dataset_name=args.dataset,
        max_intensity_samples=args.max_intensity_samples,
        max_visualizations=args.max_visualizations,
    )

    for dataset_name, report in reports.items():
        print(f"[{dataset_name}] report saved to {report.artifacts['report_json']}")
        for artifact_key, artifact_value in report.artifacts.items():
            if artifact_key == "report_json":
                continue
            print(f"[{dataset_name}] {artifact_key}: {artifact_value}")


if __name__ == "__main__":
    main()
