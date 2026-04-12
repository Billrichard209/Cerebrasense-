"""Separate external-cohort evaluation path for non-OASIS evidence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.data.base_dataset import build_monai_dataloader, build_monai_dataset
from src.data.external_cohort import ExternalCohortManifestSummary, build_external_cohort_records
from src.models.factory import OASIS_BINARY_CLASS_NAMES, build_model, load_oasis_model_config
from src.transforms.oasis_transforms import (
    OASISSpatialConfig,
    OASISTransformConfig,
    build_oasis_infer_transforms,
    load_oasis_transform_config,
)
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_torch_symbols

from .calibration import ConfidenceBandConfig, summarize_calibrated_confidence
from .metrics import (
    build_confusion_matrix,
    compute_binary_classification_metrics,
    compute_binary_roc_curve,
    compute_uncertainty_from_probabilities,
    threshold_binary_scores,
)
from .oasis_run import load_oasis_checkpoint
from .plots import save_publication_figures

_load_torch_symbols = load_torch_symbols
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class ExternalCohortEvaluationError(ValueError):
    """Raised when external-cohort evaluation cannot proceed safely."""


@dataclass(slots=True, frozen=True)
class ExternalCohortEvaluationConfig:
    """Configuration for evaluating the OASIS binary model on an external 3D cohort."""

    manifest_path: Path
    checkpoint_path: Path
    output_name: str = "external_cohort_evaluation"
    model_config_path: Path | None = None
    threshold: float | None = 0.5
    batch_size: int = 1
    num_workers: int = 0
    cache_rate: float = 0.0
    image_size: tuple[int, int, int] = (64, 64, 64)
    device: str = "cpu"
    max_batches: int | None = None
    confidence_temperature: float = 1.0
    use_active_serving_policy: bool = False
    serving_config_path: Path | None = None
    registry_path: Path | None = None


@dataclass(slots=True)
class ExternalCohortEvaluationPaths:
    """Output paths for one external-cohort evaluation run."""

    output_root: Path
    report_json_path: Path
    manifest_summary_path: Path
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
class ExternalCohortEvaluationResult:
    """External-cohort evaluation result plus saved artifact paths."""

    config: ExternalCohortEvaluationConfig
    manifest_summary: ExternalCohortManifestSummary
    metrics: dict[str, Any]
    confusion_matrix: list[list[int]]
    roc_curve: dict[str, Any]
    predictions: pd.DataFrame
    paths: ExternalCohortEvaluationPaths
    warnings: list[str]
    notes: list[str]


def _safe_name(value: str) -> str:
    """Return a filesystem-friendly name."""

    normalized = _SAFE_NAME_PATTERN.sub("_", value.strip())
    return normalized.strip("._") or "external_cohort"


def build_external_cohort_evaluation_paths(
    cfg: ExternalCohortEvaluationConfig,
    *,
    cohort_name: str,
    settings: AppSettings | None = None,
) -> ExternalCohortEvaluationPaths:
    """Build a dedicated output folder for one external cohort."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(
        resolved_settings.outputs_root
        / "evaluations"
        / "external"
        / _safe_name(cohort_name)
        / _safe_name(cfg.output_name)
    )
    figures_root = ensure_directory(output_root / "figures")
    return ExternalCohortEvaluationPaths(
        output_root=output_root,
        report_json_path=output_root / "evaluation_report.json",
        manifest_summary_path=output_root / "manifest_summary.json",
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


def _extract_batch_value(batch: dict[str, Any], key: str, item_index: int, default: str = "") -> str:
    """Extract one metadata field from a MONAI batch."""

    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, (list, tuple)) and item_index < len(value):
        return str(value[item_index])
    if item_index == 0:
        return str(value)
    return default


def _build_external_eval_loader(
    cfg: ExternalCohortEvaluationConfig,
) -> tuple[object, ExternalCohortManifestSummary]:
    """Build a deterministic loader for one validated external 3D cohort manifest."""

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
    records, manifest_summary = build_external_cohort_records(
        cfg.manifest_path,
        expected_dataset_type="3d_volumes",
    )
    dataset = build_monai_dataset(
        records,
        build_oasis_infer_transforms(transform_cfg),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    loader = build_monai_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return loader, manifest_summary


def _collect_predictions(
    *,
    model: object,
    loader: object,
    cfg: ExternalCohortEvaluationConfig,
    class_names: tuple[str, ...],
    manifest_summary: ExternalCohortManifestSummary,
    decision_policy: object,
) -> pd.DataFrame:
    """Run deterministic external-cohort inference and return per-sample rows."""

    torch = _load_torch_symbols()["torch"]
    confidence_config = decision_policy.confidence_config
    rows: list[dict[str, Any]] = []
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
            probability_rows = probabilities.detach().cpu().tolist()
            positive_scores = probabilities[:, 1].detach().cpu().tolist()
            predicted_labels = threshold_binary_scores(
                [float(score) for score in positive_scores],
                threshold=decision_policy.threshold,
            )
            calibrated_rows = summarize_calibrated_confidence(
                probability_rows,
                config=confidence_config,
            )
            uncertainty_rows = compute_uncertainty_from_probabilities(probability_rows)
            true_labels = labels.detach().cpu().tolist()
            batch_size = int(labels.numel())
            for item_index in range(batch_size):
                true_label = int(true_labels[item_index])
                predicted_label = int(predicted_labels[item_index])
                probability = float(positive_scores[item_index])
                calibrated = calibrated_rows[item_index]
                uncertainty = uncertainty_rows[item_index]
                rows.append(
                    {
                        "dataset_name": manifest_summary.dataset_name,
                        "dataset_type": manifest_summary.dataset_type,
                        "sample_id": _extract_batch_value(batch, "subject_id", item_index)
                        or _extract_batch_value(batch, "session_id", item_index)
                        or f"sample_{batch_index:04d}_{item_index:02d}",
                        "subject_id": _extract_batch_value(batch, "subject_id", item_index),
                        "session_id": _extract_batch_value(batch, "session_id", item_index),
                        "source_path": _extract_batch_value(batch, "image_path", item_index),
                        "scan_timestamp": _extract_batch_value(batch, "scan_timestamp", item_index),
                        "source_label_name": _extract_batch_value(batch, "label_name", item_index, default="") or None,
                        "true_label": true_label,
                        "true_label_name": class_names[true_label] if 0 <= true_label < len(class_names) else str(true_label),
                        "predicted_label": predicted_label,
                        "predicted_label_name": class_names[predicted_label]
                        if 0 <= predicted_label < len(class_names)
                        else str(predicted_label),
                        "probability": probability,
                        "calibrated_probability_score": calibrated.calibrated_probability_score,
                        "confidence": calibrated.confidence_score,
                        "confidence_level": calibrated.confidence_level,
                        "review_flag": calibrated.review_flag,
                        "entropy": uncertainty["entropy"],
                        "normalized_entropy": uncertainty["normalized_entropy"],
                        "probability_margin": uncertainty["probability_margin"],
                        "uncertainty_score": uncertainty["uncertainty_score"],
                    }
                )
    return pd.DataFrame(rows)


def _write_outputs(
    *,
    cfg: ExternalCohortEvaluationConfig,
    manifest_summary: ExternalCohortManifestSummary,
    metrics: dict[str, Any],
    confusion_matrix: list[list[int]],
    roc_curve: dict[str, Any],
    predictions: pd.DataFrame,
    paths: ExternalCohortEvaluationPaths,
    class_names: tuple[str, str],
    warnings: list[str],
    notes: list[str],
) -> None:
    """Write separate external-cohort evaluation artifacts to disk."""

    paths.resolved_config_path.write_text(
        json.dumps(
            {
                **asdict(cfg),
                "manifest_path": str(cfg.manifest_path),
                "checkpoint_path": str(cfg.checkpoint_path),
                "model_config_path": str(cfg.model_config_path) if cfg.model_config_path else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.manifest_summary_path.write_text(json.dumps(manifest_summary.to_dict(), indent=2), encoding="utf-8")
    paths.metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(paths.metrics_csv_path, index=False)
    predictions.to_csv(paths.predictions_csv_path, index=False)
    paths.confusion_matrix_json_path.write_text(
        json.dumps(
            {
                "dataset_name": manifest_summary.dataset_name,
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
    pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).to_csv(
        paths.confusion_matrix_csv_path
    )
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
        title_prefix=f"External {manifest_summary.dataset_name}",
    )
    paths.report_json_path.write_text(
        json.dumps(
            {
                "dataset_name": manifest_summary.dataset_name,
                "dataset_type": manifest_summary.dataset_type,
                "manifest_summary": manifest_summary.to_dict(),
                "metrics": metrics,
                "warnings": warnings,
                "notes": notes,
                "paths": {
                    "metrics_json": str(paths.metrics_json_path),
                    "predictions_csv": str(paths.predictions_csv_path),
                    "roc_curve_png": str(paths.roc_curve_png_path),
                    "confusion_matrix_png": str(paths.confusion_matrix_png_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        f"# External Cohort Evaluation: {manifest_summary.dataset_name}",
        "",
        "This report is for research decision-support development only. It is not a diagnosis.",
        "",
        "## Cohort",
        "",
        f"- dataset_name: {manifest_summary.dataset_name}",
        f"- dataset_type: {manifest_summary.dataset_type}",
        f"- manifest_path: {manifest_summary.manifest_path}",
        f"- manifest_hash_sha256: {manifest_summary.manifest_hash_sha256}",
        f"- sample_count: {manifest_summary.sample_count}",
        f"- subject_count: {manifest_summary.subject_count if manifest_summary.subject_count is not None else 'unknown'}",
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
        f"- threshold: {metrics['threshold']:.2f}",
        "",
        "## Notes",
        "",
        "- External evidence is reported separately from OASIS internal validation.",
        "- No automatic label harmonization was performed before evaluation.",
        "- Treat this as research evidence, not clinical validation.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    paths.summary_report_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_external_cohort(
    cfg: ExternalCohortEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> ExternalCohortEvaluationResult:
    """Evaluate the OASIS binary model on one separate external 3D cohort manifest."""

    resolved_settings = settings or get_app_settings()
    model_cfg = load_oasis_model_config(cfg.model_config_path)
    class_names = tuple(model_cfg.class_names) if model_cfg.class_names else OASIS_BINARY_CLASS_NAMES
    if len(class_names) != 2:
        raise ExternalCohortEvaluationError(
            "External evaluation for the current baseline expects exactly two output classes."
        )
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

    loader, manifest_summary = _build_external_eval_loader(cfg)
    model = build_model(model_cfg)
    checkpoint = load_oasis_checkpoint(cfg.checkpoint_path, device=cfg.device)
    model.load_state_dict(checkpoint.model_state_dict)

    torch = _load_torch_symbols()["torch"]
    model = model.to(cfg.device)
    model.eval()
    predictions_frame = _collect_predictions(
        model=model,
        loader=loader,
        cfg=cfg,
        class_names=class_names,
        manifest_summary=manifest_summary,
        decision_policy=decision_policy,
    )
    if predictions_frame.empty:
        raise ExternalCohortEvaluationError(
            "No external cohort samples were evaluated. Check the manifest, dataloader, or max_batches setting."
        )
    y_true = [int(value) for value in predictions_frame["true_label"].tolist()]
    y_score = [float(value) for value in predictions_frame["probability"].tolist()]
    y_pred = [int(value) for value in predictions_frame["predicted_label"].tolist()]
    metrics = compute_binary_classification_metrics(y_true, y_pred, y_score=y_score)
    metrics["threshold"] = float(decision_policy.threshold)
    metrics["dataset_name"] = manifest_summary.dataset_name
    metrics["dataset_type"] = manifest_summary.dataset_type
    metrics["manifest_hash_sha256"] = manifest_summary.manifest_hash_sha256
    metrics["mean_confidence"] = float(predictions_frame["confidence"].mean())
    metrics["mean_calibrated_confidence"] = float(predictions_frame["calibrated_probability_score"].mean())
    metrics["mean_normalized_entropy"] = float(predictions_frame["normalized_entropy"].mean())
    metrics["mean_uncertainty_score"] = float(predictions_frame["uncertainty_score"].mean())
    metrics["confidence_level_counts"] = {
        "high": int((predictions_frame["confidence_level"] == "high").sum()),
        "medium": int((predictions_frame["confidence_level"] == "medium").sum()),
        "low": int((predictions_frame["confidence_level"] == "low").sum()),
    }
    metrics["review_required_count"] = int(predictions_frame["review_flag"].sum())
    confusion_matrix = build_confusion_matrix(y_true, y_pred)
    roc_curve = compute_binary_roc_curve(y_true, y_score)
    metrics["auroc"] = roc_curve["auroc"]
    metrics["roc_curve_defined"] = roc_curve["is_defined"]
    if roc_curve["warning"]:
        metrics["roc_warning"] = roc_curve["warning"]

    warnings = list(manifest_summary.warnings)
    if roc_curve["warning"]:
        warnings.append(str(roc_curve["warning"]))
    notes = list(manifest_summary.notes)
    notes.extend(
        [
            "External cohort results are stored under outputs/evaluations/external so they are not mixed with OASIS reports.",
            "This evaluation reuses deterministic MONAI inference preprocessing and avoids train-time augmentation leakage.",
        ]
    )
    paths = build_external_cohort_evaluation_paths(
        cfg,
        cohort_name=manifest_summary.dataset_name,
        settings=resolved_settings,
    )
    _write_outputs(
        cfg=cfg,
        manifest_summary=manifest_summary,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        roc_curve=roc_curve,
        predictions=predictions_frame,
        paths=paths,
        class_names=(str(class_names[0]), str(class_names[1])),
        warnings=warnings,
        notes=notes,
    )
    return ExternalCohortEvaluationResult(
        config=cfg,
        manifest_summary=manifest_summary,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        roc_curve=roc_curve,
        predictions=predictions_frame,
        paths=paths,
        warnings=warnings,
        notes=notes,
    )
