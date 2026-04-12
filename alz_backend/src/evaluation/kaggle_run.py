"""Run-level Kaggle checkpoint evaluation for the separate 2D/3D branch."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.models.kaggle_model import KaggleMonaiModelConfig, build_kaggle_monai_network
from src.training.kaggle_research import (
    KAGGLE_RESEARCH_WARNINGS,
    ResearchKaggleTrainingConfig,
    _build_loaders,
    _evaluate_loader,
    _json_safe_metric_summary,
    _resolve_device,
    load_research_kaggle_training_config,
)
from src.training.trainer_utils import build_classification_loss
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_torch_symbols

_load_torch_symbols = load_torch_symbols


class KaggleRunEvaluationError(ValueError):
    """Raised when a saved Kaggle checkpoint cannot be evaluated safely."""


@dataclass(slots=True, frozen=True)
class KaggleRunEvaluationConfig:
    """Configuration for evaluating a saved Kaggle checkpoint."""

    run_name: str
    config_path: Path | None = None
    checkpoint_name: str = "best_model.pt"
    checkpoint_path: Path | None = None
    device: str = "auto"
    max_val_batches: int | None = None
    max_test_batches: int | None = None
    output_name: str | None = None


@dataclass(slots=True)
class LoadedKaggleCheckpoint:
    """Loaded checkpoint state and metadata."""

    path: Path
    model_state_dict: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(slots=True)
class KaggleRunEvaluationPaths:
    """Saved artifact paths for a Kaggle checkpoint evaluation."""

    evaluation_root: Path
    report_json_path: Path
    final_metrics_path: Path
    val_predictions_path: Path
    test_predictions_path: Path
    val_confusion_matrix_path: Path
    test_confusion_matrix_path: Path
    summary_report_path: Path


@dataclass(slots=True)
class KaggleRunEvaluationResult:
    """Evaluation result plus saved artifact paths."""

    config: KaggleRunEvaluationConfig
    checkpoint: LoadedKaggleCheckpoint
    final_metrics: dict[str, Any]
    paths: KaggleRunEvaluationPaths


def resolve_kaggle_run_root(settings: AppSettings, run_name: str) -> Path:
    """Resolve a Kaggle run folder."""

    run_root = settings.outputs_root / "runs" / "kaggle" / run_name
    if not run_root.exists():
        raise FileNotFoundError(f"Kaggle run folder not found: {run_root}")
    return run_root


def resolve_kaggle_checkpoint_path(
    cfg: KaggleRunEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> Path:
    """Resolve the checkpoint path for a Kaggle run evaluation."""

    if cfg.checkpoint_path is not None:
        checkpoint_path = Path(cfg.checkpoint_path)
    else:
        resolved_settings = settings or get_app_settings()
        checkpoint_path = resolve_kaggle_run_root(resolved_settings, cfg.run_name) / "checkpoints" / cfg.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Kaggle checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_kaggle_checkpoint(checkpoint_path: str | Path, *, device: str = "cpu") -> LoadedKaggleCheckpoint:
    """Load a research-style Kaggle checkpoint."""

    resolved_path = Path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Kaggle checkpoint not found: {resolved_path}")

    torch = _load_torch_symbols()["torch"]
    try:
        payload = torch.load(resolved_path, map_location=device)
    except Exception as error:
        message = str(error)
        if "Weights only load failed" not in message:
            raise
        payload = torch.load(resolved_path, map_location=device, weights_only=False)

    if not isinstance(payload, dict):
        raise KaggleRunEvaluationError(f"Unsupported checkpoint payload type: {type(payload)!r}")

    if "model_state_dict" in payload:
        model_state_dict = payload["model_state_dict"]
        metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    else:
        model_state_dict = payload
        metadata = {"checkpoint_format": "raw_state_dict"}

    if not isinstance(model_state_dict, dict):
        raise KaggleRunEvaluationError("Checkpoint model_state_dict must be a dictionary-like state dict.")

    return LoadedKaggleCheckpoint(
        path=resolved_path,
        model_state_dict=model_state_dict,
        metadata=metadata,
    )


def _evaluation_folder_name(cfg: KaggleRunEvaluationConfig) -> str:
    """Return the folder name for one checkpoint evaluation."""

    checkpoint_stem = Path(cfg.checkpoint_name).stem
    safe_name = (cfg.output_name or f"{checkpoint_stem}_eval").replace(" ", "_").replace("/", "_").replace("\\", "_")
    return safe_name


def build_kaggle_run_evaluation_paths(
    cfg: KaggleRunEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> KaggleRunEvaluationPaths:
    """Build the output paths for a run-level Kaggle evaluation."""

    resolved_settings = settings or get_app_settings()
    run_root = resolve_kaggle_run_root(resolved_settings, cfg.run_name)
    evaluation_root = ensure_directory(run_root / "evaluation" / _evaluation_folder_name(cfg))
    return KaggleRunEvaluationPaths(
        evaluation_root=evaluation_root,
        report_json_path=evaluation_root / "evaluation_report.json",
        final_metrics_path=evaluation_root / "final_metrics.json",
        val_predictions_path=evaluation_root / "val_predictions.csv",
        test_predictions_path=evaluation_root / "test_predictions.csv",
        val_confusion_matrix_path=evaluation_root / "val_confusion_matrix.json",
        test_confusion_matrix_path=evaluation_root / "test_confusion_matrix.json",
        summary_report_path=evaluation_root / "summary_report.md",
    )


def _checkpoint_json_safe(value: Any) -> Any:
    """Convert nested checkpoint values into JSON-safe summaries."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _checkpoint_json_safe(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_json_safe(item) for item in value]
    return str(value)


def _checkpoint_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build a compact checkpoint metadata summary."""

    summary: dict[str, Any] = {}
    for key in ("checkpoint_format", "epoch", "best_monitor_value", "best_epoch", "dataset_type", "class_names", "config"):
        if key in metadata:
            summary[key] = _checkpoint_json_safe(metadata[key])
    return summary


def _save_confusion_matrix(path: Path, *, split: str, class_names: tuple[str, ...], metrics: dict[str, Any]) -> None:
    """Save one split confusion matrix JSON payload."""

    path.write_text(
        json.dumps(
            {
                "split": split,
                "label_order": list(range(len(class_names))),
                "label_names": list(class_names),
                "row_axis": "true_label",
                "column_axis": "predicted_label",
                "confusion_matrix": metrics.get("confusion_matrix"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _save_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    """Save prediction rows to CSV."""

    pd.DataFrame(predictions).to_csv(path, index=False)


def _save_report_artifacts(
    *,
    cfg: KaggleRunEvaluationConfig,
    checkpoint: LoadedKaggleCheckpoint,
    class_names: tuple[str, ...],
    dataset_type: str,
    final_payload: dict[str, Any],
    val_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
    paths: KaggleRunEvaluationPaths,
) -> None:
    """Save JSON, CSV, and Markdown artifacts for the evaluation."""

    paths.final_metrics_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    _save_predictions_csv(paths.val_predictions_path, val_predictions)
    _save_predictions_csv(paths.test_predictions_path, test_predictions)
    _save_confusion_matrix(paths.val_confusion_matrix_path, split="val", class_names=class_names, metrics=final_payload["validation"])
    _save_confusion_matrix(paths.test_confusion_matrix_path, split="test", class_names=class_names, metrics=final_payload["test"])

    report_payload = {
        "run": {
            "run_name": cfg.run_name,
            "checkpoint_path": str(checkpoint.path),
            "dataset_type": dataset_type,
            "class_names": list(class_names),
            "device": cfg.device,
            "checkpoint_metadata": _checkpoint_metadata_summary(checkpoint.metadata),
        },
        "final_metrics": final_payload,
        "paths": {
            "final_metrics": str(paths.final_metrics_path),
            "val_predictions": str(paths.val_predictions_path),
            "test_predictions": str(paths.test_predictions_path),
            "val_confusion_matrix": str(paths.val_confusion_matrix_path),
            "test_confusion_matrix": str(paths.test_confusion_matrix_path),
        },
    }
    paths.report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    lines = [
        "# Kaggle Checkpoint Evaluation",
        "",
        "This is a secondary Kaggle branch checkpoint evaluation. It is not an OASIS-equivalent medical validation result.",
        "",
        "## Run",
        "",
        f"- run_name: {cfg.run_name}",
        f"- checkpoint_path: {checkpoint.path}",
        f"- dataset_type: {dataset_type}",
        f"- class_count: {len(class_names)}",
        "",
        "## Validation Metrics",
        "",
        f"- sample_count: {final_payload['validation'].get('sample_count')}",
        f"- accuracy: {final_payload['validation'].get('accuracy')}",
        f"- balanced_accuracy: {final_payload['validation'].get('balanced_accuracy')}",
        f"- macro_f1: {final_payload['validation'].get('macro_f1')}",
        f"- macro_ovr_auroc: {final_payload['validation'].get('macro_ovr_auroc')}",
        "",
        "## Test Metrics",
        "",
        f"- sample_count: {final_payload['test'].get('sample_count')}",
        f"- accuracy: {final_payload['test'].get('accuracy')}",
        f"- balanced_accuracy: {final_payload['test'].get('balanced_accuracy')}",
        f"- macro_f1: {final_payload['test'].get('macro_f1')}",
        f"- macro_ovr_auroc: {final_payload['test'].get('macro_ovr_auroc')}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in final_payload.get("warnings", []))
    paths.summary_report_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_kaggle_run_checkpoint(
    cfg: KaggleRunEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> KaggleRunEvaluationResult:
    """Evaluate a saved Kaggle checkpoint on val/test splits."""

    resolved_settings = settings or get_app_settings()
    training_cfg = load_research_kaggle_training_config(cfg.config_path)
    training_cfg = replace(
        training_cfg,
        run_name=cfg.run_name,
        dry_run=False,
        device=cfg.device,
        data=replace(
            training_cfg.data,
            max_val_batches=cfg.max_val_batches if cfg.max_val_batches is not None else training_cfg.data.max_val_batches,
            max_test_batches=cfg.max_test_batches if cfg.max_test_batches is not None else training_cfg.data.max_test_batches,
        ),
    )
    dataloaders = _build_loaders(training_cfg, resolved_settings)
    checkpoint_path = resolve_kaggle_checkpoint_path(cfg, settings=resolved_settings)

    torch = _load_torch_symbols()["torch"]
    device = _resolve_device(training_cfg.device, torch)
    amp_enabled = bool(training_cfg.mixed_precision and device.startswith("cuda"))
    checkpoint = load_kaggle_checkpoint(checkpoint_path, device=device)

    model = build_kaggle_monai_network(
        KaggleMonaiModelConfig(
            dataset_type=dataloaders.dataset_type,
            in_channels=1,
            out_channels=len(dataloaders.class_names),
            dropout_prob=training_cfg.dropout_prob,
        )
    ).to(device)
    model.load_state_dict(checkpoint.model_state_dict)
    loss_function = build_classification_loss(training_cfg.loss.name)

    val_evaluation = _evaluate_loader(
        split="val",
        loader=dataloaders.val_loader,
        model=model,
        loss_function=loss_function,
        torch=torch,
        device=device,
        amp_enabled=amp_enabled,
        max_batches=training_cfg.data.max_val_batches,
        class_names=dataloaders.class_names,
        dataset_type=dataloaders.dataset_type,
    )
    test_evaluation = _evaluate_loader(
        split="test",
        loader=dataloaders.test_loader,
        model=model,
        loss_function=loss_function,
        torch=torch,
        device=device,
        amp_enabled=amp_enabled,
        max_batches=training_cfg.data.max_test_batches,
        class_names=dataloaders.class_names,
        dataset_type=dataloaders.dataset_type,
    )

    final_payload = {
        "run_name": cfg.run_name,
        "dataset": "kaggle_alz",
        "dataset_type": dataloaders.dataset_type,
        "class_names": list(dataloaders.class_names),
        "runtime_label_map": dataloaders.runtime_label_map,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_metadata": _checkpoint_metadata_summary(checkpoint.metadata),
        "evaluation_type": "checkpoint_only",
        "warnings": list(KAGGLE_RESEARCH_WARNINGS),
        "validation": _json_safe_metric_summary(val_evaluation.metrics),
        "test": _json_safe_metric_summary(test_evaluation.metrics),
    }

    paths = build_kaggle_run_evaluation_paths(cfg, settings=resolved_settings)
    _save_report_artifacts(
        cfg=cfg,
        checkpoint=checkpoint,
        class_names=dataloaders.class_names,
        dataset_type=dataloaders.dataset_type,
        final_payload=final_payload,
        val_predictions=val_evaluation.predictions,
        test_predictions=test_evaluation.predictions,
        paths=paths,
    )
    return KaggleRunEvaluationResult(
        config=cfg,
        checkpoint=checkpoint,
        final_metrics=final_payload,
        paths=paths,
    )
