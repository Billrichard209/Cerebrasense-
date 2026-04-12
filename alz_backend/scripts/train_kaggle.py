"""Run the separate Kaggle MONAI training pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.kaggle_research import (  # noqa: E402
    ResearchKaggleDataConfig,
    ResearchKaggleTrainingConfig,
    load_research_kaggle_training_config,
    run_research_kaggle_training,
)
from src.training.oasis_research import CheckpointConfig, EarlyStoppingConfig, LossConfig, OptimizerConfig, SchedulerConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Train the separate Kaggle Alzheimer baseline with MONAI.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a Kaggle training YAML config.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dropout-prob", type=float, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-rate", type=float, default=None)
    parser.add_argument("--image-size-2d", nargs=2, type=int, default=None)
    parser.add_argument("--image-size-3d", nargs=3, type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)

    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)

    parser.add_argument("--scheduler", choices=["none", "step_lr", "reduce_on_plateau", "cosine"], default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--scheduler-factor", type=float, default=None)

    parser.add_argument("--loss", choices=["cross_entropy", "ce", "nll_loss", "nll"], default=None)

    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--monitor", type=str, default=None)
    parser.add_argument("--monitor-mode", choices=["min", "max"], default=None)

    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _coalesce(current: object, override: object) -> object:
    """Return the override when provided, otherwise keep the current value."""

    return current if override is None else override


def apply_cli_overrides(cfg: ResearchKaggleTrainingConfig, args: argparse.Namespace) -> ResearchKaggleTrainingConfig:
    """Apply explicit command-line overrides to a loaded Kaggle config."""

    data = ResearchKaggleDataConfig(
        batch_size=int(_coalesce(cfg.data.batch_size, args.batch_size)),
        gradient_accumulation_steps=int(
            _coalesce(cfg.data.gradient_accumulation_steps, args.gradient_accumulation_steps)
        ),
        num_workers=int(_coalesce(cfg.data.num_workers, args.num_workers)),
        cache_rate=float(_coalesce(cfg.data.cache_rate, args.cache_rate)),
        image_size_2d=tuple(args.image_size_2d) if args.image_size_2d is not None else cfg.data.image_size_2d,
        image_size_3d=tuple(args.image_size_3d) if args.image_size_3d is not None else cfg.data.image_size_3d,
        seed=int(_coalesce(cfg.data.seed, args.seed)),
        label_map=cfg.data.label_map,
        max_train_batches=args.max_train_batches if args.max_train_batches is not None else cfg.data.max_train_batches,
        max_val_batches=args.max_val_batches if args.max_val_batches is not None else cfg.data.max_val_batches,
        max_test_batches=args.max_test_batches if args.max_test_batches is not None else cfg.data.max_test_batches,
    )
    optimizer = OptimizerConfig(
        name=str(_coalesce(cfg.optimizer.name, args.optimizer)),
        learning_rate=float(_coalesce(cfg.optimizer.learning_rate, args.learning_rate)),
        weight_decay=float(_coalesce(cfg.optimizer.weight_decay, args.weight_decay)),
        momentum=float(_coalesce(cfg.optimizer.momentum, args.momentum)),
    )
    scheduler = SchedulerConfig(
        name=str(_coalesce(cfg.scheduler.name, args.scheduler)),
        step_size=int(_coalesce(cfg.scheduler.step_size, args.step_size)),
        gamma=float(_coalesce(cfg.scheduler.gamma, args.gamma)),
        patience=int(_coalesce(cfg.scheduler.patience, args.scheduler_patience)),
        factor=float(_coalesce(cfg.scheduler.factor, args.scheduler_factor)),
    )
    loss = LossConfig(name=str(_coalesce(cfg.loss.name, args.loss)))
    early_stopping = EarlyStoppingConfig(
        enabled=bool(_coalesce(cfg.early_stopping.enabled, args.early_stopping)),
        patience=int(_coalesce(cfg.early_stopping.patience, args.early_stopping_patience)),
        min_delta=float(_coalesce(cfg.early_stopping.min_delta, args.early_stopping_min_delta)),
        monitor=str(_coalesce(cfg.early_stopping.monitor, args.monitor)),
        mode=str(_coalesce(cfg.early_stopping.mode, args.monitor_mode)),
    )
    checkpoint = CheckpointConfig(
        resume_from=args.resume_from if args.resume_from is not None else cfg.checkpoint.resume_from,
        save_best=bool(_coalesce(cfg.checkpoint.save_best, args.save_best)),
        save_last=bool(_coalesce(cfg.checkpoint.save_last, args.save_last)),
    )
    return replace(
        cfg,
        run_name=str(_coalesce(cfg.run_name, args.run_name)),
        epochs=int(_coalesce(cfg.epochs, args.epochs)),
        device=str(_coalesce(cfg.device, args.device)),
        mixed_precision=bool(_coalesce(cfg.mixed_precision, args.mixed_precision)),
        deterministic=bool(_coalesce(cfg.deterministic, args.deterministic)),
        dry_run=bool(_coalesce(cfg.dry_run, args.dry_run)),
        dropout_prob=float(_coalesce(cfg.dropout_prob, args.dropout_prob)),
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
    )


def main() -> None:
    """Run the training pipeline and print key artifact paths."""

    args = build_parser().parse_args()
    cfg = load_research_kaggle_training_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    result = run_research_kaggle_training(cfg)

    print(f"run_name={result.run_name}")
    print(f"run_root={result.run_root}")
    print(f"best_checkpoint={result.best_checkpoint_path}")
    print(f"last_checkpoint={result.last_checkpoint_path}")
    print(f"epoch_metrics_csv={result.epoch_metrics_csv_path}")
    print(f"epoch_metrics_json={result.epoch_metrics_json_path}")
    print(f"final_metrics={result.final_metrics_path}")
    print(f"val_confusion_matrix={result.val_confusion_matrix_path}")
    print(f"test_confusion_matrix={result.test_confusion_matrix_path}")
    print(f"val_predictions={result.val_predictions_path}")
    print(f"test_predictions={result.test_predictions_path}")
    print(f"summary_report={result.summary_report_path}")
    print(f"resolved_config={result.resolved_config_path}")
    print(f"best_epoch={result.best_epoch}")
    print(f"best_monitor_value={result.best_monitor_value}")
    print(f"stopped_early={result.stopped_early}")
    print(f"final_metrics_payload={result.final_metrics}")


if __name__ == "__main__":
    main()
