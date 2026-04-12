"""Standalone OASIS-1 checkpoint evaluation CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.oasis_standalone import (  # noqa: E402
    StandaloneOASISEvaluationConfig,
    evaluate_oasis_standalone,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Evaluate an OASIS-1 checkpoint on val/test split.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-name", type=str, default="oasis_standalone_evaluation")
    parser.add_argument("--model-config-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--use-active-serving-policy", action="store_true")
    parser.add_argument("--serving-config-path", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    return parser


def main() -> None:
    """Run standalone OASIS evaluation and print artifact paths."""

    args = build_parser().parse_args()
    result = evaluate_oasis_standalone(
        StandaloneOASISEvaluationConfig(
            checkpoint_path=args.checkpoint_path,
            split=args.split,
            output_name=args.output_name,
            model_config_path=args.model_config_path,
            threshold=args.threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_rate=args.cache_rate,
            image_size=tuple(args.image_size),
            seed=args.seed,
            device=args.device,
            max_batches=args.max_batches,
            use_active_serving_policy=args.use_active_serving_policy,
            serving_config_path=args.serving_config_path,
            registry_path=args.registry_path,
        )
    )
    print(f"output_root={result.paths.output_root}")
    print(f"metrics_json={result.paths.metrics_json_path}")
    print(f"predictions_csv={result.paths.predictions_csv_path}")
    print(f"roc_curve_png={result.paths.roc_curve_png_path}")
    print(f"confusion_matrix_png={result.paths.confusion_matrix_png_path}")
    print(f"summary_report={result.paths.summary_report_path}")
    print(f"sample_count={result.metrics['sample_count']}")
    print(f"accuracy={result.metrics['accuracy']}")
    print(f"auroc={result.metrics['auroc']}")
    print(f"threshold={result.metrics['threshold']}")


if __name__ == "__main__":
    main()
