"""Promote a trained OASIS checkpoint into the backend model registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import promote_oasis_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Promote an OASIS checkpoint as the current baseline.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--val-metrics-path", type=Path, default=None)
    parser.add_argument("--test-metrics-path", type=Path, default=None)
    parser.add_argument("--model-config-path", type=Path, default=None)
    parser.add_argument("--preprocessing-config-path", type=Path, default=None)
    parser.add_argument("--image-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--recommended-threshold", type=float, default=None)
    parser.add_argument("--threshold-calibration-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Promote the requested checkpoint and print a compact summary."""

    args = parse_args()
    entry, output_path = promote_oasis_checkpoint(
        run_name=args.run_name,
        checkpoint_path=args.checkpoint_path,
        val_metrics_path=args.val_metrics_path,
        test_metrics_path=args.test_metrics_path,
        model_config_path=args.model_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        image_size=tuple(args.image_size),
        default_threshold=args.default_threshold,
        recommended_threshold=args.recommended_threshold,
        threshold_calibration_path=args.threshold_calibration_path,
        output_path=args.output_path,
    )
    print(json.dumps(entry.to_dict(), indent=2))
    print(f"registry_entry={output_path}")


if __name__ == "__main__":
    main()
