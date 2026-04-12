"""Evaluate a saved Kaggle checkpoint on validation and test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.kaggle_run import (  # noqa: E402
    KaggleRunEvaluationConfig,
    evaluate_kaggle_run_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Evaluate a saved Kaggle run checkpoint on val/test splits.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Training config used for the checkpoint.")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--output-name", type=str, default=None)
    return parser


def main() -> None:
    """Run the checkpoint evaluation and print artifact paths."""

    args = build_parser().parse_args()
    result = evaluate_kaggle_run_checkpoint(
        KaggleRunEvaluationConfig(
            run_name=args.run_name,
            config_path=args.config,
            checkpoint_name=args.checkpoint_name,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches,
            output_name=args.output_name,
        )
    )
    print(f"run_name={result.config.run_name}")
    print(f"checkpoint_path={result.checkpoint.path}")
    print(f"evaluation_root={result.paths.evaluation_root}")
    print(f"report_json={result.paths.report_json_path}")
    print(f"final_metrics={result.paths.final_metrics_path}")
    print(f"val_predictions={result.paths.val_predictions_path}")
    print(f"test_predictions={result.paths.test_predictions_path}")
    print(f"summary_report={result.paths.summary_report_path}")
    print(f"final_metrics_payload={result.final_metrics}")


if __name__ == "__main__":
    main()
