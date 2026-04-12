"""Register a benchmark, evaluate promotion gates, and promote an OASIS checkpoint if approved."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.governance import run_oasis_promotion_workflow  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Promote an OASIS checkpoint only if it passes research gates.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--validation-metrics-path", type=Path, required=True)
    parser.add_argument("--test-metrics-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--benchmark-name", type=str, required=True)
    parser.add_argument("--split-name", type=str, default="test")
    parser.add_argument("--model-config-path", type=Path, default=None)
    parser.add_argument("--preprocessing-config-path", type=Path, default=None)
    parser.add_argument("--threshold-calibration-path", type=Path, default=None)
    parser.add_argument("--recommended-threshold", type=float, default=None)
    parser.add_argument("--policy-config-path", type=Path, default=None)
    parser.add_argument("--image-size", nargs=3, type=int, default=(64, 64, 64))
    return parser


def main() -> None:
    """Run the promotion workflow and print the result."""

    args = build_parser().parse_args()
    result = run_oasis_promotion_workflow(
        run_name=args.run_name,
        checkpoint_path=args.checkpoint_path,
        validation_metrics_path=args.validation_metrics_path,
        test_metrics_path=args.test_metrics_path,
        manifest_path=args.manifest_path,
        benchmark_name=args.benchmark_name,
        split_name=args.split_name,
        model_config_path=args.model_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        threshold_calibration_path=args.threshold_calibration_path,
        recommended_threshold=args.recommended_threshold,
        policy_config_path=args.policy_config_path,
        image_size=tuple(args.image_size),
    )
    print(f"benchmark_path={result.benchmark_path}")
    print(f"decision_path={result.decision_path}")
    print(f"approved={result.decision.approved}")
    print(f"active_registry_path={result.active_registry_path}")


if __name__ == "__main__":
    main()
