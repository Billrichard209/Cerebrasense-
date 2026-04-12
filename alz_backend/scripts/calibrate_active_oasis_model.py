"""Calibrate the active approved OASIS model and update the registry threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.governance import calibrate_active_oasis_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Calibrate the active approved OASIS model and persist the recommended threshold."
    )
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--validation-predictions-path", type=Path, default=None)
    parser.add_argument("--test-predictions-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--selection-metric", type=str, default="balanced_accuracy")
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser


def main() -> None:
    """Run active-model threshold calibration and print a compact summary."""

    args = build_parser().parse_args()
    result = calibrate_active_oasis_model(
        registry_path=args.registry_path,
        validation_predictions_path=args.validation_predictions_path,
        test_predictions_path=args.test_predictions_path,
        output_dir=args.output_dir,
        selection_metric=args.selection_metric,
        threshold_step=args.threshold_step,
    )
    payload = {
        "registry_path": str(result.registry_path),
        "recommended_threshold": result.registry_entry.recommended_threshold,
        "selection_metric": result.calibration_result.selection_metric,
        "calibration_report_path": str(result.calibration_result.calibration_report_path),
        "threshold_grid_path": str(result.calibration_result.threshold_grid_path),
        "test_metrics_path": None
        if result.calibration_result.test_metrics_path is None
        else str(result.calibration_result.test_metrics_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
