"""Run a real one-epoch OASIS MONAI training job from the normalized manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_oasis import OASISTrainingConfig, run_oasis_monai_training_epoch


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Run a one-epoch OASIS MONAI training job.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-samples", type=int, default=8)
    parser.add_argument("--max-val-samples", type=int, default=4)
    parser.add_argument("--image-size", nargs=3, type=int, default=(96, 96, 96))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser


def main() -> None:
    """Run the OASIS MONAI training smoke job and print output paths."""

    parser = build_parser()
    args = parser.parse_args()
    config = OASISTrainingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        image_size=tuple(args.image_size),
        save_checkpoint=not args.no_checkpoint,
    )
    result = run_oasis_monai_training_epoch(config=config, run_name=args.run_name)
    if result.checkpoint_path:
        print(f"checkpoint={result.checkpoint_path}")
    print(f"metrics={result.metrics_path}")
    print(f"report={result.report_path}")
    print(f"train_batches={result.train_batches}")
    print(f"val_batches={result.val_batches}")
    print(f"train_loss={result.train_loss}")
    print(f"val_loss={result.val_loss}")
    print(f"val_accuracy={result.val_accuracy}")
    print(f"device={result.device}")


if __name__ == "__main__":
    main()
