"""Run a complete OASIS train-then-evaluate experiment."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.oasis_experiment import OASISExperimentConfig, run_oasis_experiment  # noqa: E402
from src.training.oasis_research import (  # noqa: E402
    ResearchDataConfig,
    load_research_oasis_training_config,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Train and evaluate an OASIS-1 MONAI baseline in one command.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--image-size", nargs=3, type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--evaluate-splits", nargs="+", choices=["val", "test"], default=["val"])
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--evaluation-output-prefix", type=str, default="post_train")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    return parser


def _coalesce(current: object, override: object) -> object:
    """Return override when supplied, otherwise current."""

    return current if override is None else override


def main() -> None:
    """Run training, evaluation, and combined report generation."""

    args = build_parser().parse_args()
    training_cfg = load_research_oasis_training_config(args.config)
    data_cfg = ResearchDataConfig(
        batch_size=int(_coalesce(training_cfg.data.batch_size, args.batch_size)),
        gradient_accumulation_steps=int(
            _coalesce(training_cfg.data.gradient_accumulation_steps, args.gradient_accumulation_steps)
        ),
        num_workers=training_cfg.data.num_workers,
        cache_rate=training_cfg.data.cache_rate,
        image_size=tuple(args.image_size) if args.image_size is not None else training_cfg.data.image_size,
        seed=int(_coalesce(training_cfg.data.seed, args.seed)),
        split_seed=_coalesce(training_cfg.data.split_seed, args.split_seed),
        train_fraction=training_cfg.data.train_fraction,
        val_fraction=training_cfg.data.val_fraction,
        test_fraction=training_cfg.data.test_fraction,
        weighted_sampling=training_cfg.data.weighted_sampling,
        max_train_batches=args.max_train_batches
        if args.max_train_batches is not None
        else training_cfg.data.max_train_batches,
        max_val_batches=args.max_val_batches if args.max_val_batches is not None else training_cfg.data.max_val_batches,
    )
    training_cfg = replace(
        training_cfg,
        run_name=str(_coalesce(training_cfg.run_name, args.run_name)),
        epochs=int(_coalesce(training_cfg.epochs, args.epochs)),
        device=str(_coalesce(training_cfg.device, args.device)),
        dry_run=bool(_coalesce(training_cfg.dry_run, args.dry_run)),
        data=data_cfg,
    )
    experiment_cfg = OASISExperimentConfig(
        training=training_cfg,
        evaluate_splits=tuple(args.evaluate_splits),
        checkpoint_name=args.checkpoint_name,
        max_eval_batches=args.max_eval_batches,
        evaluation_output_prefix=args.evaluation_output_prefix,
        experiment_name=args.experiment_name,
        tags=tuple(args.tags or ()),
    )
    result = run_oasis_experiment(experiment_cfg)

    print(f"run_name={result.training.run_name}")
    print(f"run_root={result.training.run_root}")
    print(f"best_checkpoint={result.training.best_checkpoint_path}")
    print(f"last_checkpoint={result.training.last_checkpoint_path}")
    print(f"experiment_summary_json={result.summary_json_path}")
    print(f"experiment_summary_report={result.summary_report_path}")
    if result.tracked_experiment is not None:
        print(f"experiment_tracking_root={result.tracked_experiment.paths.experiment_root}")
        print(f"experiment_final_metrics={result.tracked_experiment.paths.final_metrics_path}")
    for evaluation in result.evaluations:
        print(f"evaluation_{evaluation.config.split}_root={evaluation.paths.evaluation_root}")
        print(f"evaluation_{evaluation.config.split}_accuracy={evaluation.result.metrics['accuracy']}")
        print(f"evaluation_{evaluation.config.split}_auroc={evaluation.result.metrics['auroc']}")


if __name__ == "__main__":
    main()
