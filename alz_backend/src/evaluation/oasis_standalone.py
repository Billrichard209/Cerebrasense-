"""Standalone OASIS-1 checkpoint evaluation with publication-friendly artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.data.loaders import OASISLoaderConfig, build_oasis_dataloaders
from src.models.factory import OASIS_BINARY_CLASS_NAMES, build_model, load_oasis_model_config
from src.transforms.oasis_transforms import OASISSpatialConfig, OASISTransformConfig, load_oasis_transform_config
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_torch_symbols

from .calibration import ConfidenceBandConfig, summarize_calibrated_confidence
from .metrics import (
    build_confusion_matrix,
    compute_binary_classification_metrics,
    compute_binary_roc_curve,
    threshold_binary_scores,
)
from .oasis_run import load_oasis_checkpoint
from .plots import save_publication_figures

_load_torch_symbols = load_torch_symbols


class StandaloneOASISEvaluationError(ValueError):
    """Raised when standalone OASIS evaluation cannot proceed safely."""


@dataclass(slots=True, frozen=True)
class StandaloneOASISEvaluationConfig:
    """Configuration for standalone OASIS checkpoint evaluation."""

    checkpoint_path: Path
    split: str = "val"
    output_name: str = "oasis_standalone_evaluation"
    model_config_path: Path | None = None
    threshold: float | None = 0.5
    batch_size: int = 1
    num_workers: int = 0
    cache_rate: float = 0.0
    image_size: tuple[int, int, int] = (64, 64, 64)
    seed: int = 42
    device: str = "cpu"
    max_batches: int | None = None
    confidence_temperature: float = 1.0
    use_active_serving_policy: bool = False
    serving_config_path: Path | None = None
    registry_path: Path | None = None


@dataclass(slots=True)
class StandaloneOASISEvaluationPaths:
    """Output paths for standalone OASIS evaluation."""

    output_root: Path
    metrics_json_path: Path
    metrics_csv_path: Path
    predictions_csv_path: Path
    confusion_matrix_json_path: Path
    confusion_matrix_csv_path: Path
    roc_curve_csv_path: Path
    roc_curve_png_path: Path
    confusion_matrix_png_path: Path
    summary_report_path: Path
    resolved_config_path: Path


@dataclass(slots=True)
class StandaloneOASISEvaluationResult:
    """Result payload for standalone OASIS evaluation."""

    config: StandaloneOASISEvaluationConfig
    metrics: dict[str, Any]
    confusion_matrix: list[list[int]]
    roc_curve: dict[str, Any]
    predictions: pd.DataFrame
    paths: StandaloneOASISEvaluationPaths


def build_standalone_oasis_evaluation_paths(
    cfg: StandaloneOASISEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> StandaloneOASISEvaluationPaths:
    """Build publication-friendly evaluation output paths."""

    resolved_settings = settings or get_app_settings()
    safe_name = cfg.output_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_root = ensure_directory(resolved_settings.outputs_root / "evaluations" / "oasis" / safe_name)
    figures_root = ensure_directory(output_root / "figures")
    return StandaloneOASISEvaluationPaths(
        output_root=output_root,
        metrics_json_path=output_root / "metrics.json",
        metrics_csv_path=output_root / "metrics.csv",
        predictions_csv_path=output_root / "predictions.csv",
        confusion_matrix_json_path=output_root / "confusion_matrix.json",
        confusion_matrix_csv_path=output_root / "confusion_matrix.csv",
        roc_curve_csv_path=output_root / "roc_curve.csv",
        roc_curve_png_path=figures_root / "roc_curve.png",
        confusion_matrix_png_path=figures_root / "confusion_matrix.png",
        summary_report_path=output_root / "summary_report.md",
        resolved_config_path=output_root / "resolved_config.json",
    )


def _build_oasis_eval_loader(cfg: StandaloneOASISEvaluationConfig) -> object:
    """Build a deterministic OASIS validation or test loader."""

    if cfg.split not in {"val", "test"}:
        raise StandaloneOASISEvaluationError(f"split must be 'val' or 'test', got {cfg.split!r}.")
    transform_cfg = load_oasis_transform_config()
    transform_cfg = OASISTransformConfig(
        load=transform_cfg.load,
        orientation=transform_cfg.orientation,
        spacing=transform_cfg.spacing,
        intensity=transform_cfg.intensity,
        skull_strip=transform_cfg.skull_strip,
        spatial=OASISSpatialConfig(spatial_size=cfg.image_size),
        augmentation=transform_cfg.augmentation,
    )
    loader_cfg = OASISLoaderConfig(
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cache_rate=cfg.cache_rate,
        transform_config=transform_cfg,
    )
    loaders = build_oasis_dataloaders(loader_cfg)
    return loaders.val_loader if cfg.split == "val" else loaders.test_loader


def _extract_batch_value(batch: dict[str, Any], key: str, item_index: int, default: str = "") -> str:
    """Extract a batch metadata value for one item."""

    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, (list, tuple)) and item_index < len(value):
        return str(value[item_index])
    if item_index == 0:
        return str(value)
    return default


def _collect_predictions(
    *,
    model: object,
    loader: object,
    cfg: StandaloneOASISEvaluationConfig,
    class_names: tuple[str, ...],
    decision_policy: object,
) -> pd.DataFrame:
    """Run model inference and return one row per evaluated sample."""

    torch = _load_torch_symbols()["torch"]
    model = model.to(cfg.device)
    model.eval()
    rows: list[dict[str, Any]] = []
    confidence_config = decision_policy.confidence_config
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if cfg.max_batches is not None and batch_index >= cfg.max_batches:
                break
            images = batch["image"].to(cfg.device)
            labels = batch["label"].to(cfg.device).long() if hasattr(batch["label"], "to") else torch.as_tensor(
                batch["label"],
                device=cfg.device,
            ).long()
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            positive_scores = probabilities[:, 1].detach().cpu().tolist()
            predicted_labels = threshold_binary_scores(
                [float(score) for score in positive_scores],
                threshold=decision_policy.threshold,
            )
            calibrated_rows = summarize_calibrated_confidence(
                probabilities.detach().cpu().tolist(),
                config=confidence_config,
            )
            true_labels = labels.detach().cpu().tolist()
            batch_size = int(labels.numel())
            for item_index in range(batch_size):
                true_label = int(true_labels[item_index])
                predicted_label = int(predicted_labels[item_index])
                probability = float(positive_scores[item_index])
                calibrated = calibrated_rows[item_index]
                rows.append(
                    {
                        "dataset": "oasis1",
                        "dataset_name": "oasis1",
                        "split": cfg.split,
                        "subject_id": _extract_batch_value(batch, "subject_id", item_index),
                        "session_id": _extract_batch_value(batch, "session_id", item_index),
                        "source_path": _extract_batch_value(batch, "image_path", item_index),
                        "true_label": true_label,
                        "true_label_name": class_names[true_label] if 0 <= true_label < len(class_names) else str(true_label),
                        "predicted_label": predicted_label,
                        "predicted_label_name": class_names[predicted_label]
                        if 0 <= predicted_label < len(class_names)
                        else str(predicted_label),
                        "probability": probability,
                        "probability_demented": probability,
                        "calibrated_probability": calibrated.calibrated_probability_score,
                        "confidence": calibrated.confidence_score,
                        "confidence_level": calibrated.confidence_level,
                        "review_flag": calibrated.review_flag,
                        "threshold": decision_policy.threshold,
                    }
                )
    if not rows:
        raise StandaloneOASISEvaluationError("No samples were evaluated. Check split, max_batches, and dataloader setup.")
    return pd.DataFrame(rows)


def _write_outputs(
    *,
    cfg: StandaloneOASISEvaluationConfig,
    metrics: dict[str, Any],
    confusion_matrix: list[list[int]],
    roc_curve: dict[str, Any],
    predictions: pd.DataFrame,
    paths: StandaloneOASISEvaluationPaths,
    class_names: tuple[str, str],
) -> None:
    """Write standalone evaluation artifacts to disk."""

    paths.resolved_config_path.write_text(
        json.dumps(
            {
                **asdict(cfg),
                "checkpoint_path": str(cfg.checkpoint_path),
                "model_config_path": str(cfg.model_config_path) if cfg.model_config_path else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(paths.metrics_csv_path, index=False)
    predictions.to_csv(paths.predictions_csv_path, index=False)
    paths.confusion_matrix_json_path.write_text(
        json.dumps(
            {
                "label_order": [0, 1],
                "class_names": list(class_names),
                "row_axis": "true_label",
                "column_axis": "predicted_label",
                "layout": "[[true_negative, false_positive], [false_negative, true_positive]]",
                "matrix": confusion_matrix,
                "confusion_counts": metrics["confusion_counts"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).to_csv(paths.confusion_matrix_csv_path)
    pd.DataFrame(
        {
            "false_positive_rate": roc_curve["fpr"],
            "true_positive_rate": roc_curve["tpr"],
            "threshold": roc_curve["thresholds"],
        }
    ).to_csv(paths.roc_curve_csv_path, index=False)
    save_publication_figures(
        roc_curve=roc_curve,
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        output_root=paths.roc_curve_png_path.parent,
        title_prefix=f"OASIS {cfg.split}",
    )
    paths.summary_report_path.write_text(
        "\n".join(
            [
                f"# OASIS Standalone Evaluation: {cfg.output_name}",
                "",
                "This evaluation is for research decision-support development, not diagnosis.",
                "",
                "## Configuration",
                "",
                f"- split: {cfg.split}",
                f"- threshold: {metrics['threshold']}",
                f"- checkpoint: {cfg.checkpoint_path}",
                f"- sample_count: {metrics['sample_count']}",
                "",
                "## Metrics",
                "",
                f"- accuracy: {metrics['accuracy']:.6f}",
                f"- auroc: {metrics['auroc']:.6f}",
                f"- precision: {metrics['precision']:.6f}",
                f"- recall: {metrics['recall_sensitivity']:.6f}",
                f"- f1: {metrics['f1']:.6f}",
                f"- sensitivity: {metrics['sensitivity']:.6f}",
                f"- specificity: {metrics['specificity']:.6f}",
                "",
                "## Artifacts",
                "",
                f"- metrics_json: {paths.metrics_json_path}",
                f"- predictions_csv: {paths.predictions_csv_path}",
                f"- roc_curve_png: {paths.roc_curve_png_path}",
                f"- confusion_matrix_png: {paths.confusion_matrix_png_path}",
            ]
        ),
        encoding="utf-8",
    )


def evaluate_oasis_standalone(
    cfg: StandaloneOASISEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> StandaloneOASISEvaluationResult:
    """Evaluate an OASIS checkpoint on val/test and save standalone artifacts."""

    resolved_settings = settings or get_app_settings()
    model_cfg = load_oasis_model_config(cfg.model_config_path)
    class_names = tuple(model_cfg.class_names) if model_cfg.class_names else OASIS_BINARY_CLASS_NAMES
    if len(class_names) != 2:
        raise StandaloneOASISEvaluationError("Standalone OASIS binary evaluation expects exactly two class names.")
    model = build_model(model_cfg)
    checkpoint = load_oasis_checkpoint(cfg.checkpoint_path, device=cfg.device)
    model.load_state_dict(checkpoint.model_state_dict)
    loader = _build_oasis_eval_loader(cfg)
    from src.inference.serving import OASISDecisionPolicy, resolve_oasis_decision_policy

    if cfg.use_active_serving_policy or cfg.registry_path is not None or cfg.serving_config_path is not None:
        decision_policy = resolve_oasis_decision_policy(
            explicit_threshold=cfg.threshold,
            serving_config_path=cfg.serving_config_path,
            registry_path=cfg.registry_path,
            settings=resolved_settings,
        )
    else:
        decision_policy = OASISDecisionPolicy(
            threshold=float(cfg.threshold if cfg.threshold is not None else 0.5),
            confidence_config=ConfidenceBandConfig(temperature=cfg.confidence_temperature),
            serving_config=resolve_oasis_decision_policy(settings=resolved_settings).serving_config,
            registry_entry=None,
        )
    predictions = _collect_predictions(
        model=model,
        loader=loader,
        cfg=cfg,
        class_names=class_names,
        decision_policy=decision_policy,
    )
    y_true = [int(value) for value in predictions["true_label"].tolist()]
    y_score = [float(value) for value in predictions["probability"].tolist()]
    y_pred = [int(value) for value in predictions["predicted_label"].tolist()]
    metrics = compute_binary_classification_metrics(y_true, y_pred, y_score=y_score)
    metrics["threshold"] = float(decision_policy.threshold)
    if not predictions.empty:
        metrics["mean_calibrated_confidence"] = float(predictions["confidence"].mean())
        metrics["confidence_level_counts"] = {
            "high": int((predictions["confidence_level"] == "high").sum()),
            "medium": int((predictions["confidence_level"] == "medium").sum()),
            "low": int((predictions["confidence_level"] == "low").sum()),
        }
        metrics["review_required_count"] = int(predictions["review_flag"].sum())
    confusion_matrix = build_confusion_matrix(y_true, y_pred)
    roc_curve = compute_binary_roc_curve(y_true, y_score)
    metrics["auroc"] = roc_curve["auroc"]
    metrics["roc_curve_defined"] = roc_curve["is_defined"]
    if roc_curve["warning"]:
        metrics["roc_warning"] = roc_curve["warning"]
    paths = build_standalone_oasis_evaluation_paths(cfg, settings=resolved_settings)
    _write_outputs(
        cfg=cfg,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        roc_curve=roc_curve,
        predictions=predictions,
        paths=paths,
        class_names=(str(class_names[0]), str(class_names[1])),
    )
    return StandaloneOASISEvaluationResult(
        config=cfg,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        roc_curve=roc_curve,
        predictions=predictions,
        paths=paths,
    )
