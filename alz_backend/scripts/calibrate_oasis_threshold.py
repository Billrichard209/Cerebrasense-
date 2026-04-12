"""Calibrate an OASIS binary classification threshold from validation predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.thresholds import calibrate_binary_threshold  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Calibrate OASIS threshold from validation predictions.")
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-metric", default="f1")
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    """Run threshold calibration and print a compact summary."""

    args = parse_args()
    result = calibrate_binary_threshold(
        validation_predictions_path=args.validation_predictions,
        test_predictions_path=args.test_predictions,
        output_dir=args.output_dir,
        selection_metric=args.selection_metric,
        threshold_step=args.threshold_step,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
