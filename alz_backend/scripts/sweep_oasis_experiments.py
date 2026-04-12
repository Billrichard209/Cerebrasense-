"""Run a multi-seed OASIS experiment study and summarize model selection."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.model_selection import OASISModelSelectionConfig, run_oasis_model_selection_study  # noqa: E402
from src.training.oasis_experiment import OASISExperimentConfig  # noqa: E402
from src.training.oasis_research import ResearchDataConfig, load_research_oasis_training_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Run a multi-seed OASIS model-selection study.")
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--pair-seeds-with-splits", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--image-size", nargs=3, type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--evaluate-splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--selection-split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--selection-metric", type=str, default="auroc")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--evaluation-output-prefix", type=str, default="post_train")
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def _coalesce(current: object, override: object) -> object:
    """Return override when present, else current."""

    return current if override is None else override


def main() -> None:
    """Run the study and print saved artifact paths."""

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
        seed=training_cfg.data.seed,
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
    study = run_oasis_model_selection_study(
        OASISModelSelectionConfig(
            study_name=args.study_name,
            base_experiment=OASISExperimentConfig(
                training=training_cfg,
                evaluate_splits=tuple(args.evaluate_splits),
                checkpoint_name=args.checkpoint_name,
                max_eval_batches=args.max_eval_batches,
                evaluation_output_prefix=args.evaluation_output_prefix,
                tags=tuple(args.tags or ()),
            ),
            seeds=tuple(args.seeds),
            split_seeds=tuple(args.split_seeds or ()),
            selection_split=args.selection_split,
            selection_metric=args.selection_metric,
            pair_seed_and_split_seed=bool(args.pair_seeds_with_splits),
            resume_existing=bool(args.resume_existing),
        )
    )

    print(f"study_root={study.study_root}")
    print(f"seed_runs_csv={study.runs_csv_path}")
    print(f"study_summary_json={study.summary_json_path}")
    print(f"best_experiment_json={study.best_run_json_path}")
    print(f"aggregate_metrics_json={study.aggregate_json_path}")
    print(f"best_seed={study.best_row.seed}")
    print(f"best_experiment_name={study.best_row.experiment_name}")
    print(f"best_selection_score={study.best_row.selection_score}")


if __name__ == "__main__":
    main()
