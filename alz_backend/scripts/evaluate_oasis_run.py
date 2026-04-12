"""Evaluate a research OASIS training run checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.oasis_run import OASISRunEvaluationConfig, evaluate_oasis_run_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Evaluate an OASIS research run checkpoint on val/test data.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--model-config-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--use-active-serving-policy", action="store_true")
    parser.add_argument("--serving-config-path", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    return parser


def main() -> None:
    """Run evaluation and print the saved artifact paths."""

    args = build_parser().parse_args()
    cfg = OASISRunEvaluationConfig(
        run_name=args.run_name,
        split=args.split,
        checkpoint_name=args.checkpoint_name,
        checkpoint_path=args.checkpoint_path,
        model_config_path=args.model_config_path,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        image_size=tuple(args.image_size),
        seed=args.seed,
        device=args.device,
        max_batches=args.max_batches,
        output_name=args.output_name,
        use_active_serving_policy=args.use_active_serving_policy,
        serving_config_path=args.serving_config_path,
        registry_path=args.registry_path,
    )
    evaluation = evaluate_oasis_run_checkpoint(cfg)

    print(f"run_name={cfg.run_name}")
    print(f"split={cfg.split}")
    print(f"checkpoint={evaluation.checkpoint.path}")
    print(f"evaluation_root={evaluation.paths.evaluation_root}")
    print(f"evaluation_report={evaluation.paths.report_json_path}")
    print(f"metrics={evaluation.paths.metrics_json_path}")
    print(f"predictions={evaluation.paths.predictions_csv_path}")
    print(f"summary={evaluation.paths.summary_report_path}")
    print(f"sample_count={evaluation.result.metrics['sample_count']}")
    print(f"accuracy={evaluation.result.metrics['accuracy']}")
    print(f"auroc={evaluation.result.metrics['auroc']}")
    print(f"threshold={evaluation.result.metrics['threshold']}")


if __name__ == "__main__":
    main()
