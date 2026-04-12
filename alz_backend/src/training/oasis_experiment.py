"""Train-then-evaluate orchestration for OASIS-1 research experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.evaluation.oasis_run import OASISRunEvaluationConfig, OASISRunEvaluationResult, evaluate_oasis_run_checkpoint
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_torch_symbols

from .experiment_tracking import ExperimentTrackingConfig, TrackedExperimentResult, export_tracked_oasis_experiment
from .oasis_research import ResearchOASISTrainingConfig, ResearchTrainingResult, run_research_oasis_training

_load_torch_symbols = load_torch_symbols


@dataclass(slots=True, frozen=True)
class OASISExperimentConfig:
    """Config for one train-then-evaluate OASIS research experiment."""

    training: ResearchOASISTrainingConfig = field(default_factory=ResearchOASISTrainingConfig)
    evaluate_splits: tuple[str, ...] = ("val",)
    checkpoint_name: str = "best_model.pt"
    max_eval_batches: int | None = None
    evaluation_output_prefix: str = "post_train"
    experiment_name: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class OASISExperimentResult:
    """Training result, evaluation results, and combined summary artifacts."""

    training: ResearchTrainingResult
    evaluations: list[OASISRunEvaluationResult]
    summary_json_path: Path
    summary_report_path: Path
    tracked_experiment: TrackedExperimentResult | None = None


def _resolve_eval_device(training_cfg: ResearchOASISTrainingConfig) -> str:
    """Resolve an evaluation device compatible with checkpoint loading."""

    if training_cfg.device != "auto":
        return training_cfg.device
    torch = _load_torch_symbols()["torch"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_evaluation_config(
    *,
    cfg: OASISExperimentConfig,
    split: str,
) -> OASISRunEvaluationConfig:
    """Build a run-local evaluation config from the training config."""

    training_cfg = cfg.training
    return OASISRunEvaluationConfig(
        run_name=training_cfg.run_name,
        split=split,
        checkpoint_name=cfg.checkpoint_name,
        model_config_path=training_cfg.model_config_path,
        batch_size=training_cfg.data.batch_size,
        num_workers=training_cfg.data.num_workers,
        cache_rate=training_cfg.data.cache_rate,
        image_size=training_cfg.data.image_size,
        seed=training_cfg.data.seed,
        device=_resolve_eval_device(training_cfg),
        max_batches=cfg.max_eval_batches,
        output_name=f"{cfg.evaluation_output_prefix}_{split}_{Path(cfg.checkpoint_name).stem}",
    )


def _write_experiment_summary(
    *,
    cfg: OASISExperimentConfig,
    result: OASISExperimentResult,
    elapsed_seconds: float,
) -> None:
    """Write combined JSON and Markdown summaries for a full experiment."""

    evaluation_payloads = [
        {
            "split": evaluation.config.split,
            "checkpoint": str(evaluation.checkpoint.path),
            "metrics": evaluation.result.metrics,
            "evaluation_root": str(evaluation.paths.evaluation_root),
            "report_json": str(evaluation.paths.report_json_path),
            "predictions_csv": str(evaluation.paths.predictions_csv_path),
        }
        for evaluation in result.evaluations
    ]
    payload: dict[str, Any] = {
        "run_name": cfg.training.run_name,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "experiment_tracking_root": str(result.tracked_experiment.paths.experiment_root)
        if result.tracked_experiment is not None
        else None,
        "training": {
            "run_root": str(result.training.run_root),
            "best_checkpoint": str(result.training.best_checkpoint_path)
            if result.training.best_checkpoint_path is not None
            else None,
            "last_checkpoint": str(result.training.last_checkpoint_path)
            if result.training.last_checkpoint_path is not None
            else None,
            "best_epoch": result.training.best_epoch,
            "best_monitor_value": result.training.best_monitor_value,
            "stopped_early": result.training.stopped_early,
            "final_metrics": result.training.final_metrics,
        },
        "evaluations": evaluation_payloads,
        "config": asdict(cfg),
        "notes": [
            "This is a research experiment summary for decision-support development, not diagnosis.",
            "Dry-run or tiny-batch results validate the pipeline only and should not be interpreted as model quality.",
        ],
    }
    result.summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# OASIS Experiment: {cfg.training.run_name}",
        "",
        "This experiment summary is for research decision-support development, not diagnosis.",
        "",
        "## Training",
        "",
        f"- run_root: {result.training.run_root}",
        f"- best_checkpoint: {result.training.best_checkpoint_path}",
        f"- last_checkpoint: {result.training.last_checkpoint_path}",
        f"- experiment_tracking_root: {result.tracked_experiment.paths.experiment_root if result.tracked_experiment else None}",
        f"- best_epoch: {result.training.best_epoch}",
        f"- best_monitor_value: {result.training.best_monitor_value}",
        f"- stopped_early: {result.training.stopped_early}",
        f"- elapsed_seconds: {round(elapsed_seconds, 2)}",
        "",
        "## Evaluations",
        "",
    ]
    for evaluation in result.evaluations:
        metrics = evaluation.result.metrics
        lines.extend(
            [
                f"### {evaluation.config.split}",
                "",
                f"- sample_count: {metrics.get('sample_count', 0)}",
                f"- accuracy: {metrics.get('accuracy', 0.0):.6f}",
                f"- auroc: {metrics.get('auroc', 0.0):.6f}",
                f"- f1: {metrics.get('f1', 0.0):.6f}",
                f"- mean_confidence: {metrics.get('mean_confidence', 0.0):.6f}",
                f"- evaluation_root: {evaluation.paths.evaluation_root}",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- Tiny runs are pipeline checks and should not be interpreted as model quality.",
            "- OASIS and Kaggle remain separate; this experiment uses only OASIS-1.",
        ]
    )
    result.summary_report_path.write_text("\n".join(lines), encoding="utf-8")


def run_oasis_experiment(
    cfg: OASISExperimentConfig | None = None,
    *,
    settings: AppSettings | None = None,
) -> OASISExperimentResult:
    """Train an OASIS model, evaluate selected splits, and write a combined summary."""

    resolved_cfg = cfg or OASISExperimentConfig()
    resolved_settings = settings or get_app_settings()
    start_time = perf_counter()
    training_result = run_research_oasis_training(resolved_cfg.training, settings=resolved_settings)

    evaluations = [
        evaluate_oasis_run_checkpoint(
            _build_evaluation_config(cfg=resolved_cfg, split=split),
            settings=resolved_settings,
        )
        for split in resolved_cfg.evaluate_splits
    ]
    tracked_experiment = export_tracked_oasis_experiment(
        ExperimentTrackingConfig(
            experiment_name=resolved_cfg.experiment_name or resolved_cfg.training.run_name,
            run_name=resolved_cfg.training.run_name,
            tags=resolved_cfg.tags,
        ),
        training=training_result,
        evaluations=evaluations,
        settings=resolved_settings,
    )

    reports_root = ensure_directory(training_result.run_root / "reports")
    result = OASISExperimentResult(
        training=training_result,
        evaluations=evaluations,
        summary_json_path=reports_root / "experiment_summary.json",
        summary_report_path=reports_root / "experiment_summary.md",
        tracked_experiment=tracked_experiment,
    )
    _write_experiment_summary(
        cfg=resolved_cfg,
        result=result,
        elapsed_seconds=perf_counter() - start_time,
    )
    return result
