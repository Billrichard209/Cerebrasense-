"""Multi-seed OASIS model-selection helpers for research-grade comparison."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

from .oasis_experiment import OASISExperimentConfig, OASISExperimentResult, run_oasis_experiment
from .oasis_research import ResearchDataConfig, ResearchOASISTrainingConfig


class ModelSelectionError(ValueError):
    """Raised when a model-selection study cannot be executed safely."""


@dataclass(slots=True, frozen=True)
class OASISModelSelectionConfig:
    """Configuration for a multi-seed OASIS experiment study."""

    study_name: str
    base_experiment: OASISExperimentConfig = field(default_factory=OASISExperimentConfig)
    seeds: tuple[int, ...] = (42, 43, 44)
    split_seeds: tuple[int, ...] = ()
    selection_split: str = "val"
    selection_metric: str = "auroc"
    maximize_metric: bool | None = None
    pair_seed_and_split_seed: bool = False
    resume_existing: bool = False


@dataclass(slots=True, frozen=True)
class StudyMetricAggregate:
    """Aggregate statistics for one metric across seed runs."""

    split: str
    metric_name: str
    values: list[float]
    mean: float
    std: float
    minimum: float
    maximum: float
    ci95_low: float
    ci95_high: float

    def to_payload(self) -> dict[str, Any]:
        """Serialize aggregate statistics."""

        return asdict(self)


@dataclass(slots=True)
class SeedStudyRow:
    """Compact row for one seed run in a study."""

    seed: int
    split_seed: int | None
    run_name: str
    experiment_name: str
    best_checkpoint_path: str | None
    tracked_experiment_root: str | None
    selection_score: float
    split_metrics: dict[str, dict[str, Any]]

    def to_row(self) -> dict[str, Any]:
        """Flatten a seed run into a tabular row."""

        payload: dict[str, Any] = {
            "seed": self.seed,
            "split_seed": self.split_seed,
            "run_name": self.run_name,
            "experiment_name": self.experiment_name,
            "best_checkpoint_path": self.best_checkpoint_path,
            "tracked_experiment_root": self.tracked_experiment_root,
            "selection_score": self.selection_score,
        }
        for split_name, metrics in self.split_metrics.items():
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    payload[f"{split_name}_{metric_name}"] = float(metric_value)
        return payload


@dataclass(slots=True)
class OASISModelSelectionStudyResult:
    """Artifacts and summary metadata for a completed study."""

    config: OASISModelSelectionConfig
    study_root: Path
    runs_csv_path: Path
    summary_json_path: Path
    best_run_json_path: Path
    aggregate_json_path: Path
    rows: list[SeedStudyRow]
    best_row: SeedStudyRow
    metric_aggregates: list[StudyMetricAggregate]


def _metric_direction(metric_name: str, *, explicit: bool | None) -> bool:
    """Return whether higher metric values should be preferred."""

    if explicit is not None:
        return bool(explicit)
    normalized = metric_name.strip().lower()
    return normalized not in {"loss", "val_loss", "train_loss"}


def _replace_training_seed(
    training_cfg: ResearchOASISTrainingConfig,
    seed: int,
    split_seed: int | None,
    run_name: str,
) -> ResearchOASISTrainingConfig:
    """Clone the nested research config with new training and split seeds."""

    return replace(
        training_cfg,
        run_name=run_name,
        data=ResearchDataConfig(
            batch_size=training_cfg.data.batch_size,
            gradient_accumulation_steps=training_cfg.data.gradient_accumulation_steps,
            num_workers=training_cfg.data.num_workers,
            cache_rate=training_cfg.data.cache_rate,
            image_size=training_cfg.data.image_size,
            seed=seed,
            split_seed=split_seed,
            train_fraction=training_cfg.data.train_fraction,
            val_fraction=training_cfg.data.val_fraction,
            test_fraction=training_cfg.data.test_fraction,
            weighted_sampling=training_cfg.data.weighted_sampling,
            max_train_batches=training_cfg.data.max_train_batches,
            max_val_batches=training_cfg.data.max_val_batches,
        ),
    )


def build_seeded_experiment_config(
    base_config: OASISExperimentConfig,
    *,
    study_name: str,
    seed: int,
    split_seed: int | None = None,
) -> OASISExperimentConfig:
    """Create one per-seed experiment config from a common template."""

    base_run_name = base_config.training.run_name
    split_suffix = "" if split_seed is None else f"_split{split_seed}"
    run_name = f"{base_run_name}_seed{seed}{split_suffix}"
    experiment_name = f"{study_name}_seed{seed}{split_suffix}"
    existing_tags = tuple(base_config.tags)
    added_tags = ["model_selection", f"seed_{seed}"]
    if split_seed is not None:
        added_tags.append(f"split_{split_seed}")
    merged_tags = tuple(tag for tag in (*existing_tags, *added_tags) if tag)
    return replace(
        base_config,
        training=_replace_training_seed(base_config.training, seed, split_seed, run_name),
        experiment_name=experiment_name,
        tags=merged_tags,
    )


def _resolve_seed_split_pairs(config: OASISModelSelectionConfig) -> list[tuple[int, int | None]]:
    """Resolve the run matrix for training seeds and subject-safe split seeds."""

    if not config.seeds:
        raise ModelSelectionError("At least one training seed is required for model selection.")
    if not config.split_seeds:
        return [(seed, None) for seed in config.seeds]
    if config.pair_seed_and_split_seed:
        if len(config.seeds) != len(config.split_seeds):
            raise ModelSelectionError(
                "When pair_seed_and_split_seed=True, seeds and split_seeds must have the same length."
            )
        return list(zip(config.seeds, config.split_seeds))
    return [(seed, split_seed) for split_seed in config.split_seeds for seed in config.seeds]


def _evaluation_metrics(result: OASISExperimentResult) -> dict[str, dict[str, Any]]:
    """Collect evaluation metrics by split."""

    return {
        evaluation.config.split: dict(evaluation.result.metrics)
        for evaluation in result.evaluations
    }


def _selection_score(
    *,
    metrics_by_split: dict[str, dict[str, Any]],
    split: str,
    metric_name: str,
) -> float:
    """Resolve the selection score from one experiment."""

    if split not in metrics_by_split:
        raise ModelSelectionError(
            f"Selection split {split!r} is unavailable. Present splits: {sorted(metrics_by_split)}"
        )
    split_metrics = metrics_by_split[split]
    if metric_name not in split_metrics:
        raise ModelSelectionError(
            f"Selection metric {metric_name!r} is unavailable for split {split!r}. "
            f"Available metrics: {sorted(split_metrics)}"
        )
    return float(split_metrics[metric_name])


def _confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Return an approximate 95% confidence interval for the mean."""

    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    std = pstdev(values)
    margin = 1.96 * (std / math.sqrt(len(values)))
    center = mean(values)
    return float(center - margin), float(center + margin)


def _aggregate_metric(
    *,
    split: str,
    metric_name: str,
    rows: list[SeedStudyRow],
) -> StudyMetricAggregate:
    """Aggregate a numeric metric across all seed rows."""

    values = [
        float(row.split_metrics[split][metric_name])
        for row in rows
        if split in row.split_metrics and metric_name in row.split_metrics[split]
    ]
    if not values:
        raise ModelSelectionError(f"No values found for {split}.{metric_name} during aggregation.")
    ci95_low, ci95_high = _confidence_interval_95(values)
    return StudyMetricAggregate(
        split=split,
        metric_name=metric_name,
        values=[float(value) for value in values],
        mean=float(mean(values)),
        std=float(pstdev(values)) if len(values) > 1 else 0.0,
        minimum=float(min(values)),
        maximum=float(max(values)),
        ci95_low=ci95_low,
        ci95_high=ci95_high,
    )


def _aggregate_metrics(rows: list[SeedStudyRow]) -> list[StudyMetricAggregate]:
    """Aggregate all shared numeric split metrics across runs."""

    split_metric_names: dict[str, set[str]] = {}
    for row in rows:
        for split_name, metrics in row.split_metrics.items():
            numeric_names = {
                metric_name
                for metric_name, metric_value in metrics.items()
                if isinstance(metric_value, (int, float))
            }
            split_metric_names.setdefault(split_name, set()).update(numeric_names)
    return [
        _aggregate_metric(split=split_name, metric_name=metric_name, rows=rows)
        for split_name, metric_names in sorted(split_metric_names.items())
        for metric_name in sorted(metric_names)
    ]


def _study_root(settings: AppSettings, study_name: str) -> Path:
    """Build the model-selection study root."""

    safe_name = study_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return ensure_directory(settings.outputs_root / "model_selection" / safe_name)


def _existing_run_root(settings: AppSettings, run_name: str) -> Path:
    """Return the canonical OASIS run root for an experiment run name."""

    return settings.outputs_root / "runs" / "oasis" / run_name


def _existing_eval_report_path(
    *,
    run_root: Path,
    split: str,
    evaluation_output_prefix: str,
    checkpoint_name: str,
) -> Path:
    """Return the expected evaluation report path for one split."""

    checkpoint_stem = Path(checkpoint_name).stem
    return run_root / "evaluation" / f"{evaluation_output_prefix}_{split}_{checkpoint_stem}" / "evaluation_report.json"


def _try_load_existing_seed_row(
    *,
    settings: AppSettings,
    seeded_config: OASISExperimentConfig,
    selection_split: str,
    selection_metric: str,
    seed: int,
    split_seed: int | None,
) -> SeedStudyRow | None:
    """Load a completed study row from existing run/evaluation artifacts when available."""

    run_root = _existing_run_root(settings, seeded_config.training.run_name)
    checkpoint_path = run_root / "checkpoints" / seeded_config.checkpoint_name
    if not checkpoint_path.exists():
        return None

    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split in seeded_config.evaluate_splits:
        report_json_path = _existing_eval_report_path(
            run_root=run_root,
            split=split,
            evaluation_output_prefix=seeded_config.evaluation_output_prefix,
            checkpoint_name=seeded_config.checkpoint_name,
        )
        if not report_json_path.exists():
            return None
        payload = json.loads(report_json_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            return None
        metrics_by_split[split] = metrics

    tracked_experiment_root = settings.outputs_root / "experiments" / (
        seeded_config.experiment_name or seeded_config.training.run_name
    )
    return SeedStudyRow(
        seed=seed,
        split_seed=split_seed,
        run_name=seeded_config.training.run_name,
        experiment_name=seeded_config.experiment_name or seeded_config.training.run_name,
        best_checkpoint_path=str(checkpoint_path),
        tracked_experiment_root=str(tracked_experiment_root) if tracked_experiment_root.exists() else None,
        selection_score=_selection_score(
            metrics_by_split=metrics_by_split,
            split=selection_split,
            metric_name=selection_metric,
        ),
        split_metrics=metrics_by_split,
    )


def _write_rows_csv(path: Path, rows: list[SeedStudyRow]) -> None:
    """Write flattened seed results as CSV."""

    pd.DataFrame([row.to_row() for row in rows]).to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with a stable format."""

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_oasis_model_selection_study(
    config: OASISModelSelectionConfig,
    *,
    settings: AppSettings | None = None,
) -> OASISModelSelectionStudyResult:
    """Run repeated per-seed experiments and summarize the best checkpoint."""

    if not config.seeds:
        raise ModelSelectionError("At least one seed is required for model selection.")
    resolved_settings = settings or get_app_settings()
    rows: list[SeedStudyRow] = []
    maximize_metric = _metric_direction(config.selection_metric, explicit=config.maximize_metric)
    seed_split_pairs = _resolve_seed_split_pairs(config)

    for seed, split_seed in seed_split_pairs:
        seeded_config = build_seeded_experiment_config(
            config.base_experiment,
            study_name=config.study_name,
            seed=seed,
            split_seed=split_seed,
        )
        if config.resume_existing:
            existing_row = _try_load_existing_seed_row(
                settings=resolved_settings,
                seeded_config=seeded_config,
                selection_split=config.selection_split,
                selection_metric=config.selection_metric,
                seed=seed,
                split_seed=split_seed,
            )
            if existing_row is not None:
                rows.append(existing_row)
                continue
        result = run_oasis_experiment(seeded_config, settings=resolved_settings)
        metrics_by_split = _evaluation_metrics(result)
        rows.append(
            SeedStudyRow(
                seed=seed,
                split_seed=split_seed,
                run_name=result.training.run_name,
                experiment_name=seeded_config.experiment_name or result.training.run_name,
                best_checkpoint_path=None
                if result.training.best_checkpoint_path is None
                else str(result.training.best_checkpoint_path),
                tracked_experiment_root=None
                if result.tracked_experiment is None
                else str(result.tracked_experiment.paths.experiment_root),
                selection_score=_selection_score(
                    metrics_by_split=metrics_by_split,
                    split=config.selection_split,
                    metric_name=config.selection_metric,
                ),
                split_metrics=metrics_by_split,
            )
        )

    best_row = max(rows, key=lambda row: row.selection_score) if maximize_metric else min(rows, key=lambda row: row.selection_score)
    study_root = _study_root(resolved_settings, config.study_name)
    runs_csv_path = study_root / "seed_runs.csv"
    summary_json_path = study_root / "study_summary.json"
    best_run_json_path = study_root / "best_experiment.json"
    aggregate_json_path = study_root / "aggregate_metrics.json"

    _write_rows_csv(runs_csv_path, rows)
    metric_aggregates = _aggregate_metrics(rows)
    _write_json(
        aggregate_json_path,
        {
            "study_name": config.study_name,
            "aggregates": [aggregate.to_payload() for aggregate in metric_aggregates],
        },
    )
    _write_json(
        best_run_json_path,
        {
            "study_name": config.study_name,
            "selection_split": config.selection_split,
            "selection_metric": config.selection_metric,
            "maximize_metric": maximize_metric,
            "best_run": best_row.to_row(),
            "best_run_split_metrics": best_row.split_metrics,
        },
    )
    _write_json(
        summary_json_path,
        {
            "study_name": config.study_name,
            "seeds": list(config.seeds),
            "split_seeds": list(config.split_seeds),
            "pair_seed_and_split_seed": config.pair_seed_and_split_seed,
            "resume_existing": config.resume_existing,
            "selection_split": config.selection_split,
            "selection_metric": config.selection_metric,
            "maximize_metric": maximize_metric,
            "rows_csv_path": str(runs_csv_path),
            "best_run_json_path": str(best_run_json_path),
            "aggregate_json_path": str(aggregate_json_path),
            "best_seed": best_row.seed,
            "best_split_seed": best_row.split_seed,
            "best_experiment_name": best_row.experiment_name,
            "best_selection_score": best_row.selection_score,
            "best_checkpoint_path": best_row.best_checkpoint_path,
            "runs": [row.to_row() for row in rows],
            "aggregate_metrics": [aggregate.to_payload() for aggregate in metric_aggregates],
            "notes": [
                "Best model selection is based on the configured validation metric, not on the test split.",
                "Multi-seed summaries reduce the risk of over-trusting one lucky random seed.",
                "When split_seeds are provided, subject-safe repeated split validation is included in the study.",
                "This study remains OASIS-only and is for research decision support, not diagnosis.",
            ],
        },
    )
    return OASISModelSelectionStudyResult(
        config=config,
        study_root=study_root,
        runs_csv_path=runs_csv_path,
        summary_json_path=summary_json_path,
        best_run_json_path=best_run_json_path,
        aggregate_json_path=aggregate_json_path,
        rows=rows,
        best_row=best_row,
        metric_aggregates=metric_aggregates,
    )
