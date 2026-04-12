"""Analyze OASIS prediction errors and save review artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import ErrorAnalysisConfig, analyze_prediction_errors  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Analyze false positives and false negatives from a predictions CSV.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default="oasis_error_analysis")
    parser.add_argument("--max-examples-per-bucket", type=int, default=None)
    parser.add_argument("--no-slices", action="store_true")
    return parser


def main() -> None:
    """Run error analysis and print artifact paths."""

    args = build_parser().parse_args()
    result = analyze_prediction_errors(
        ErrorAnalysisConfig(
            predictions_csv_path=args.predictions_csv,
            output_name=args.output_name,
            max_examples_per_bucket=args.max_examples_per_bucket,
            save_slices=not args.no_slices,
        )
    )
    print(f"output_root={result.paths.output_root}")
    print(f"summary_json={result.paths.summary_json_path}")
    print(f"misclassifications_csv={result.paths.misclassifications_csv_path}")
    print("summary=" + json.dumps(result.summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
