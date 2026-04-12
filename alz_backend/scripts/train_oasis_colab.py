"""Colab-friendly OASIS training wrapper."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _running_in_colab() -> bool:
    """Return whether the current interpreter looks like Google Colab."""

    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _mount_drive_if_requested(should_mount: bool) -> None:
    """Mount Google Drive when running inside Colab and requested."""

    if not should_mount or not _running_in_colab():
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def build_parser() -> argparse.ArgumentParser:
    """Create the Colab CLI parser."""

    parser = argparse.ArgumentParser(description="Run OASIS training in Google Colab with Drive-friendly settings.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="oasis_baseline_colab_gpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--oasis-source-dir", type=Path, required=True)
    parser.add_argument("--kaggle-source-dir", type=Path, default=None)
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", nargs=3, type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-rate", type=float, default=None)
    parser.add_argument("--evaluate-splits", nargs="+", choices=["val", "test"], default=("val", "test"))
    return parser


def main() -> None:
    """Run the Colab wrapper."""

    args = build_parser().parse_args()
    _mount_drive_if_requested(args.mount_drive)

    project_root = args.project_root.resolve() if args.project_root is not None else Path(__file__).resolve().parents[1]
    os.environ["ALZ_OASIS_SOURCE_DIR"] = str(args.oasis_source_dir.resolve())
    if args.kaggle_source_dir is not None:
        os.environ["ALZ_KAGGLE_SOURCE_DIR"] = str(args.kaggle_source_dir.resolve())

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.training.oasis_experiment import OASISExperimentConfig, run_oasis_experiment
    from src.training.oasis_research import (
        ResearchDataConfig,
        load_research_oasis_training_config,
    )

    config_path = args.config or (project_root / "configs" / "oasis_train_colab_gpu.yaml")
    training_cfg = load_research_oasis_training_config(config_path)

    if args.epochs is not None:
        training_cfg.epochs = int(args.epochs)
    training_cfg.run_name = args.run_name
    training_cfg.device = "cuda"
    training_cfg.mixed_precision = True
    training_cfg.data = ResearchDataConfig(
        batch_size=int(args.batch_size) if args.batch_size is not None else training_cfg.data.batch_size,
        num_workers=int(args.num_workers) if args.num_workers is not None else training_cfg.data.num_workers,
        cache_rate=float(args.cache_rate) if args.cache_rate is not None else training_cfg.data.cache_rate,
        image_size=tuple(args.image_size) if args.image_size is not None else training_cfg.data.image_size,
        seed=training_cfg.data.seed,
        train_fraction=training_cfg.data.train_fraction,
        val_fraction=training_cfg.data.val_fraction,
        test_fraction=training_cfg.data.test_fraction,
        weighted_sampling=training_cfg.data.weighted_sampling,
        max_train_batches=training_cfg.data.max_train_batches,
        max_val_batches=training_cfg.data.max_val_batches,
    )

    result = run_oasis_experiment(
        OASISExperimentConfig(
            training=training_cfg,
            evaluate_splits=tuple(args.evaluate_splits),
            checkpoint_name="best_model.pt",
        )
    )

    print(f"run_name={result.training.run_name}")
    print(f"run_root={result.training.run_root}")
    print(f"best_checkpoint={result.training.best_checkpoint_path}")
    for split, evaluation in result.evaluations.items():
        print(f"{split}_accuracy={evaluation.result.metrics['accuracy']}")
        print(f"{split}_auroc={evaluation.result.metrics['auroc']}")


if __name__ == "__main__":
    main()
