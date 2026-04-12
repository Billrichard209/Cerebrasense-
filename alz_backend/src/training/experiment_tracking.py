"""Experiment tracking helpers for reproducible OASIS research runs."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.configs.runtime import AppSettings, get_app_settings
from src.evaluation.metrics import build_confusion_matrix, compute_binary_roc_curve
from src.evaluation.oasis_run import OASISRunEvaluationResult
from src.evaluation.plots import save_publication_figures
from src.storage import ExperimentMetadataRecord, persist_experiment_record
from src.training.oasis_research import ResearchTrainingResult
from src.utils.io_utils import ensure_directory

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ExperimentTrackingConfig:
    """Configuration for exporting experiment-tracking artifacts."""

    experiment_name: str
    run_name: str
    tags: tuple[str, ...] = ()
    primary_split: str | None = None


@dataclass(slots=True)
class ExperimentTrackingPaths:
    """Resolved experiment output paths."""

    experiment_root: Path
    config_yaml_path: Path
    epoch_metrics_csv_path: Path
    epoch_metrics_json_path: Path
    final_metrics_path: Path
    summary_path: Path
    comparison_row_path: Path
    confusion_matrix_png_path: Path
    roc_curve_png_path: Path


@dataclass(slots=True)
class TrackedExperimentResult:
    """Result of exporting one tracked experiment."""

    config: ExperimentTrackingConfig
    paths: ExperimentTrackingPaths
    primary_split: str
    final_metrics: dict[str, Any]
    summary_payload: dict[str, Any]


def build_experiment_paths(
    cfg: ExperimentTrackingConfig,
    *,
    settings: AppSettings | None = None,
) -> ExperimentTrackingPaths:
    """Build the experiment folder layout under outputs/experiments."""

    resolved_settings = settings or get_app_settings()
    safe_name = cfg.experiment_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    experiment_root = ensure_directory(resolved_settings.outputs_root / "experiments" / safe_name)
    return ExperimentTrackingPaths(
        experiment_root=experiment_root,
        config_yaml_path=experiment_root / "config.yaml",
        epoch_metrics_csv_path=experiment_root / "epoch_metrics.csv",
        epoch_metrics_json_path=experiment_root / "epoch_metrics.json",
        final_metrics_path=experiment_root / "final_metrics.json",
        summary_path=experiment_root / "experiment_summary.json",
        comparison_row_path=experiment_root / "comparison_row.json",
        confusion_matrix_png_path=experiment_root / "confusion_matrix.png",
        roc_curve_png_path=experiment_root / "roc_curve.png",
    )


def _select_primary_evaluation(
    evaluations: list[OASISRunEvaluationResult],
    *,
    requested_split: str | None = None,
) -> OASISRunEvaluationResult:
    """Select the evaluation used for final experiment figures and metrics."""

    if not evaluations:
        raise ValueError("At least one evaluation result is required for experiment tracking.")
    if requested_split is not None:
        for evaluation in evaluations:
            if evaluation.config.split == requested_split:
                return evaluation
    for evaluation in evaluations:
        if evaluation.config.split == "val":
            return evaluation
    return evaluations[0]


def _copy_if_exists(source: Path, destination: Path) -> None:
    """Copy a file when it exists, preserving metadata."""

    if source.exists():
        shutil.copy2(source, destination)


def _prediction_arrays(evaluation: OASISRunEvaluationResult) -> tuple[list[int], list[int], list[float]]:
    """Extract y_true, y_pred, and positive-class scores from an evaluation result."""

    y_true = [int(record.true_label) for record in evaluation.result.predictions if record.true_label is not None]
    y_pred = [int(record.predicted_label) for record in evaluation.result.predictions if record.true_label is not None]
    y_score = [
        float(record.calibrated_probability_score if record.calibrated_probability_score is not None else record.probabilities[1])
        for record in evaluation.result.predictions
        if record.true_label is not None
    ]
    return y_true, y_pred, y_score


def _write_config_yaml(
    cfg: ExperimentTrackingConfig,
    *,
    training: ResearchTrainingResult,
    evaluations: list[OASISRunEvaluationResult],
    destination: Path,
) -> None:
    """Write resolved experiment configuration as YAML."""

    training_config_payload: dict[str, Any] = {}
    if training.resolved_config_path.exists():
        training_config_payload = json.loads(training.resolved_config_path.read_text(encoding="utf-8"))
    payload = {
        "experiment_name": cfg.experiment_name,
        "run_name": cfg.run_name,
        "tags": list(cfg.tags),
        "primary_split": cfg.primary_split,
        "training": training_config_payload,
        "evaluations": [
            {
                "split": evaluation.config.split,
                "checkpoint_path": str(evaluation.checkpoint.path),
                "metrics_json_path": str(evaluation.paths.metrics_json_path),
                "predictions_csv_path": str(evaluation.paths.predictions_csv_path),
            }
            for evaluation in evaluations
        ],
    }
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_summary(
    *,
    cfg: ExperimentTrackingConfig,
    training: ResearchTrainingResult,
    evaluations: list[OASISRunEvaluationResult],
    primary: OASISRunEvaluationResult,
    destination: Path,
) -> dict[str, Any]:
    """Write a JSON experiment summary."""

    payload = {
        "experiment_name": cfg.experiment_name,
        "run_name": cfg.run_name,
        "tags": list(cfg.tags),
        "primary_split": primary.config.split,
        "training": {
            "run_root": str(training.run_root),
            "best_checkpoint_path": str(training.best_checkpoint_path) if training.best_checkpoint_path else None,
            "last_checkpoint_path": str(training.last_checkpoint_path) if training.last_checkpoint_path else None,
            "best_epoch": training.best_epoch,
            "best_monitor_value": training.best_monitor_value,
            "stopped_early": training.stopped_early,
            "final_metrics": dict(training.final_metrics),
        },
        "evaluations": {
            evaluation.config.split: {
                "metrics": dict(evaluation.result.metrics),
                "evaluation_root": str(evaluation.paths.evaluation_root),
                "predictions_csv_path": str(evaluation.paths.predictions_csv_path),
                "metrics_json_path": str(evaluation.paths.metrics_json_path),
            }
            for evaluation in evaluations
        },
        "notes": [
            "Experiments are research development artifacts for decision support only.",
            "OASIS and Kaggle remain fully separate in experiment tracking.",
        ],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _comparison_row(
    *,
    cfg: ExperimentTrackingConfig,
    primary_split: str,
    final_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Return the compact comparison row used by experiment comparison tooling."""

    return {
        "experiment_name": cfg.experiment_name,
        "run_name": cfg.run_name,
        "tags": list(cfg.tags),
        "primary_split": primary_split,
        "accuracy": float(final_metrics.get("accuracy", 0.0)),
        "auroc": float(final_metrics.get("auroc", 0.0)),
        "sensitivity": float(final_metrics.get("sensitivity", 0.0)),
        "specificity": float(final_metrics.get("specificity", 0.0)),
    }


def export_tracked_oasis_experiment(
    cfg: ExperimentTrackingConfig,
    *,
    training: ResearchTrainingResult,
    evaluations: list[OASISRunEvaluationResult],
    settings: AppSettings | None = None,
) -> TrackedExperimentResult:
    """Export experiment-tracking artifacts under outputs/experiments."""

    resolved_settings = settings or get_app_settings()
    paths = build_experiment_paths(cfg, settings=resolved_settings)
    primary = _select_primary_evaluation(evaluations, requested_split=cfg.primary_split)

    _copy_if_exists(training.epoch_metrics_csv_path, paths.epoch_metrics_csv_path)
    _copy_if_exists(training.epoch_metrics_json_path, paths.epoch_metrics_json_path)
    _write_config_yaml(cfg, training=training, evaluations=evaluations, destination=paths.config_yaml_path)

    y_true, y_pred, y_score = _prediction_arrays(primary)
    confusion_matrix = build_confusion_matrix(y_true, y_pred)
    roc_curve = compute_binary_roc_curve(y_true, y_score)
    save_publication_figures(
        roc_curve=roc_curve,
        confusion_matrix=confusion_matrix,
        class_names=tuple(primary.result.class_names[:2]),
        output_root=paths.experiment_root,
        title_prefix=f"{cfg.experiment_name} {primary.config.split}",
    )

    final_metrics = {
        **dict(primary.result.metrics),
        "experiment_name": cfg.experiment_name,
        "run_name": cfg.run_name,
        "tags": list(cfg.tags),
        "primary_split": primary.config.split,
        "best_checkpoint_path": str(training.best_checkpoint_path) if training.best_checkpoint_path else None,
    }
    paths.final_metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    comparison_row = _comparison_row(cfg=cfg, primary_split=primary.config.split, final_metrics=final_metrics)
    paths.comparison_row_path.write_text(json.dumps(comparison_row, indent=2), encoding="utf-8")
    summary_payload = _write_summary(
        cfg=cfg,
        training=training,
        evaluations=evaluations,
        primary=primary,
        destination=paths.summary_path,
    )
    persist_experiment_record(
        ExperimentMetadataRecord(
            experiment_name=cfg.experiment_name,
            run_name=cfg.run_name,
            primary_split=primary.config.split,
            tags=list(cfg.tags),
            best_checkpoint_path=str(training.best_checkpoint_path) if training.best_checkpoint_path else None,
            metrics=final_metrics,
            summary_path=str(paths.summary_path),
        ),
        settings=resolved_settings,
    )
    LOGGER.info(
        "Tracked experiment '%s' at %s using split=%s.",
        cfg.experiment_name,
        paths.experiment_root,
        primary.config.split,
    )
    return TrackedExperimentResult(
        config=cfg,
        paths=paths,
        primary_split=primary.config.split,
        final_metrics=final_metrics,
        summary_payload=summary_payload,
    )


def load_experiment_rows(
    experiments_root: Path,
) -> pd.DataFrame:
    """Load compact experiment rows for comparison reports."""

    rows: list[dict[str, Any]] = []
    for experiment_dir in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
        comparison_path = experiment_dir / "comparison_row.json"
        final_metrics_path = experiment_dir / "final_metrics.json"
        payload_path = comparison_path if comparison_path.exists() else final_metrics_path
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_name": payload.get("experiment_name", experiment_dir.name),
                "accuracy": float(payload.get("accuracy", 0.0)),
                "auroc": float(payload.get("auroc", 0.0)),
                "sensitivity": float(payload.get("sensitivity", 0.0)),
                "specificity": float(payload.get("specificity", 0.0)),
                "primary_split": payload.get("primary_split", ""),
                "tags": ",".join(payload.get("tags", [])),
            }
        )
    return pd.DataFrame(rows)
