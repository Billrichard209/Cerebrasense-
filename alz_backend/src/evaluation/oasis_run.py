"""Run-level OASIS checkpoint evaluation for research training outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.evaluation.calibration import ConfidenceBandConfig
from src.data.loaders import OASISLoaderConfig, build_oasis_dataloaders
from src.models.factory import OASISModelConfig, build_model, load_oasis_model_config
from src.transforms.oasis_transforms import OASISSpatialConfig, OASISTransformConfig, load_oasis_transform_config
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_torch_symbols

from .evaluate_oasis import OASISEvaluationResult, evaluate_oasis_model_on_loader

_load_torch_symbols = load_torch_symbols


class OASISRunEvaluationError(ValueError):
    """Raised when a run-level OASIS evaluation cannot proceed safely."""


@dataclass(slots=True, frozen=True)
class OASISRunEvaluationConfig:
    """Configuration for evaluating an OASIS research training checkpoint."""

    run_name: str
    split: str = "val"
    checkpoint_name: str = "best_model.pt"
    checkpoint_path: Path | None = None
    model_config_path: Path | None = None
    threshold: float | None = None
    batch_size: int = 1
    num_workers: int = 0
    cache_rate: float = 0.0
    image_size: tuple[int, int, int] = (64, 64, 64)
    seed: int = 42
    device: str = "cpu"
    max_batches: int | None = None
    output_name: str | None = None
    use_active_serving_policy: bool = False
    serving_config_path: Path | None = None
    registry_path: Path | None = None


@dataclass(slots=True)
class LoadedCheckpoint:
    """Loaded checkpoint state and metadata."""

    path: Path
    model_state_dict: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(slots=True)
class OASISRunEvaluationPaths:
    """Output paths for a run-level checkpoint evaluation."""

    evaluation_root: Path
    report_json_path: Path
    predictions_csv_path: Path
    metrics_json_path: Path
    summary_report_path: Path


@dataclass(slots=True)
class OASISRunEvaluationResult:
    """Evaluation result plus saved artifact paths."""

    config: OASISRunEvaluationConfig
    checkpoint: LoadedCheckpoint
    result: OASISEvaluationResult
    paths: OASISRunEvaluationPaths


def resolve_oasis_run_root(settings: AppSettings, run_name: str) -> Path:
    """Resolve an OASIS research run folder."""

    run_root = settings.outputs_root / "runs" / "oasis" / run_name
    if not run_root.exists():
        raise FileNotFoundError(f"OASIS run folder not found: {run_root}")
    return run_root


def resolve_oasis_checkpoint_path(
    cfg: OASISRunEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> Path:
    """Resolve the checkpoint path for an OASIS run evaluation."""

    if cfg.checkpoint_path is not None:
        checkpoint_path = Path(cfg.checkpoint_path)
    else:
        resolved_settings = settings or get_app_settings()
        checkpoint_path = (
            resolve_oasis_run_root(resolved_settings, cfg.run_name)
            / "checkpoints"
            / cfg.checkpoint_name
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"OASIS checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_oasis_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> LoadedCheckpoint:
    """Load raw state-dict or research-training checkpoint payloads."""

    resolved_path = Path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"OASIS checkpoint not found: {resolved_path}")

    torch = _load_torch_symbols()["torch"]
    try:
        payload = torch.load(resolved_path, map_location=device)
    except Exception as error:
        message = str(error)
        if "Weights only load failed" not in message:
            raise
        # Local research checkpoints from older runs may include pathlib
        # metadata. Retry with full pickle loading only for these trusted
        # backend-generated checkpoints so legacy runs remain evaluable.
        payload = torch.load(resolved_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise OASISRunEvaluationError(f"Unsupported checkpoint payload type: {type(payload)!r}")

    if "model_state_dict" in payload:
        model_state_dict = payload["model_state_dict"]
        metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    else:
        model_state_dict = payload
        metadata = {"checkpoint_format": "raw_state_dict"}

    if not isinstance(model_state_dict, dict):
        raise OASISRunEvaluationError("Checkpoint model_state_dict must be a dictionary-like state dict.")
    return LoadedCheckpoint(
        path=resolved_path,
        model_state_dict=model_state_dict,
        metadata=metadata,
    )


def load_oasis_model_for_evaluation(
    cfg: OASISRunEvaluationConfig,
    *,
    model_cfg: OASISModelConfig | None = None,
    settings: AppSettings | None = None,
) -> tuple[object, OASISModelConfig, LoadedCheckpoint]:
    """Build the OASIS model and load a run checkpoint into it."""

    checkpoint_path = resolve_oasis_checkpoint_path(cfg, settings=settings)
    loaded_checkpoint = load_oasis_checkpoint(checkpoint_path, device=cfg.device)
    resolved_model_cfg = model_cfg or load_oasis_model_config(cfg.model_config_path)
    model = build_model(resolved_model_cfg)
    model.load_state_dict(loaded_checkpoint.model_state_dict)
    return model, resolved_model_cfg, loaded_checkpoint


def _build_loader(cfg: OASISRunEvaluationConfig) -> object:
    """Build the requested OASIS split loader."""

    if cfg.split not in {"val", "test"}:
        raise OASISRunEvaluationError(f"OASIS run evaluation supports split='val' or 'test', got {cfg.split!r}.")

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
    dataloaders = build_oasis_dataloaders(loader_cfg)
    return dataloaders.val_loader if cfg.split == "val" else dataloaders.test_loader


def _resolve_run_decision_policy(
    cfg: OASISRunEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> "OASISDecisionPolicy":
    """Resolve threshold/confidence policy for run-level evaluation."""

    from src.inference.serving import OASISDecisionPolicy, resolve_oasis_decision_policy

    resolved_settings = settings or get_app_settings()
    if cfg.use_active_serving_policy or cfg.registry_path is not None or cfg.serving_config_path is not None:
        return resolve_oasis_decision_policy(
            explicit_threshold=cfg.threshold,
            serving_config_path=cfg.serving_config_path,
            registry_path=cfg.registry_path,
            settings=resolved_settings,
        )

    return OASISDecisionPolicy(
        threshold=float(cfg.threshold if cfg.threshold is not None else 0.5),
        confidence_config=ConfidenceBandConfig(),
        serving_config=resolve_oasis_decision_policy(settings=resolved_settings).serving_config,
        registry_entry=None,
    )


def _evaluation_folder_name(cfg: OASISRunEvaluationConfig) -> str:
    """Return the folder name for one checkpoint/split evaluation."""

    checkpoint_stem = Path(cfg.checkpoint_name).stem
    output_name = cfg.output_name or f"{cfg.split}_{checkpoint_stem}"
    safe_name = output_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return safe_name


def build_oasis_run_evaluation_paths(
    cfg: OASISRunEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> OASISRunEvaluationPaths:
    """Build the output paths for a run-level evaluation."""

    resolved_settings = settings or get_app_settings()
    run_root = resolve_oasis_run_root(resolved_settings, cfg.run_name)
    evaluation_root = ensure_directory(run_root / "evaluation" / _evaluation_folder_name(cfg))
    return OASISRunEvaluationPaths(
        evaluation_root=evaluation_root,
        report_json_path=evaluation_root / "evaluation_report.json",
        predictions_csv_path=evaluation_root / "predictions.csv",
        metrics_json_path=evaluation_root / "metrics.json",
        summary_report_path=evaluation_root / "summary_report.md",
    )


def _prediction_rows(result: OASISEvaluationResult) -> list[dict[str, Any]]:
    """Flatten prediction records for CSV export."""

    rows: list[dict[str, Any]] = []
    resolved_threshold = result.metrics.get("threshold")
    for prediction in result.predictions:
        row = {
            "sample_id": prediction.sample_id,
            "true_label": prediction.true_label,
            "true_label_name": prediction.true_label_name,
            "predicted_label": prediction.predicted_label,
            "predicted_label_name": prediction.predicted_label_name,
            "threshold": resolved_threshold,
            "confidence": prediction.confidence,
            "calibrated_probability_score": prediction.calibrated_probability_score,
            "confidence_level": prediction.confidence_level,
            "review_flag": prediction.review_flag,
            "entropy": prediction.entropy,
            "normalized_entropy": prediction.normalized_entropy,
            "probability_margin": prediction.probability_margin,
            "uncertainty_score": prediction.uncertainty_score,
        }
        for index, probability in enumerate(prediction.probabilities):
            row[f"probability_class_{index}"] = probability
        for key, value in prediction.meta.items():
            row[f"meta_{key}"] = value
        rows.append(row)
    return rows


def _json_safe(value: Any) -> Any:
    """Convert nested checkpoint values into JSON-safe summaries."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _checkpoint_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build a compact JSON-safe checkpoint metadata summary."""

    summary: dict[str, Any] = {}
    for key in ("checkpoint_format", "epoch", "best_monitor_value", "best_epoch", "config"):
        if key in metadata:
            summary[key] = _json_safe(metadata[key])
    for key in ("optimizer_state_dict", "scheduler_state_dict"):
        if key in metadata and isinstance(metadata[key], dict):
            summary[f"{key}_keys"] = sorted(str(state_key) for state_key in metadata[key].keys())
    return summary


def save_oasis_run_evaluation(
    *,
    cfg: OASISRunEvaluationConfig,
    checkpoint: LoadedCheckpoint,
    result: OASISEvaluationResult,
    paths: OASISRunEvaluationPaths,
) -> None:
    """Save JSON, CSV, and Markdown reports for a run-level evaluation."""

    payload = result.to_payload()
    payload["run"] = {
        "run_name": cfg.run_name,
        "split": cfg.split,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_metadata": _checkpoint_metadata_summary(checkpoint.metadata),
        "config": {
            **asdict(cfg),
            "checkpoint_path": str(cfg.checkpoint_path) if cfg.checkpoint_path else None,
            "model_config_path": str(cfg.model_config_path) if cfg.model_config_path else None,
        },
    }
    paths.report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.metrics_json_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    pd.DataFrame(_prediction_rows(result)).to_csv(paths.predictions_csv_path, index=False)

    metrics = result.metrics
    lines = [
        f"# OASIS Run Evaluation: {cfg.run_name}",
        "",
        "This evaluation report is for research decision-support development, not diagnosis.",
        "",
        "## Run",
        "",
        f"- split: {cfg.split}",
        f"- checkpoint: {checkpoint.path}",
        f"- sample_count: {metrics.get('sample_count', 0)}",
        f"- threshold: {metrics.get('threshold', 0.5):.6f}",
        "",
        "## Metrics",
        "",
        f"- accuracy: {metrics.get('accuracy', 0.0):.6f}",
        f"- auroc: {metrics.get('auroc', 0.0):.6f}",
        f"- precision: {metrics.get('precision', 0.0):.6f}",
        f"- recall_sensitivity: {metrics.get('recall_sensitivity', 0.0):.6f}",
        f"- specificity: {metrics.get('specificity', 0.0):.6f}",
        f"- f1: {metrics.get('f1', 0.0):.6f}",
        f"- mean_confidence: {metrics.get('mean_confidence', 0.0):.6f}",
        f"- mean_uncertainty_score: {metrics.get('mean_uncertainty_score', 0.0):.6f}",
        "",
        "## Artifacts",
        "",
        f"- evaluation_report: {paths.report_json_path}",
        f"- metrics_json: {paths.metrics_json_path}",
        f"- predictions_csv: {paths.predictions_csv_path}",
    ]
    paths.summary_report_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_oasis_run_checkpoint(
    cfg: OASISRunEvaluationConfig,
    *,
    settings: AppSettings | None = None,
) -> OASISRunEvaluationResult:
    """Evaluate a trained OASIS run checkpoint and save run-local artifacts."""

    resolved_settings = settings or get_app_settings()
    model, model_cfg, checkpoint = load_oasis_model_for_evaluation(cfg, settings=resolved_settings)
    loader = _build_loader(cfg)
    decision_policy = _resolve_run_decision_policy(cfg, settings=resolved_settings)
    result = evaluate_oasis_model_on_loader(
        model=model,
        loader=loader,
        device=cfg.device,
        class_names=model_cfg.class_names,
        max_batches=cfg.max_batches,
        calibration_config=decision_policy.confidence_config,
        decision_threshold=decision_policy.threshold,
    )
    paths = build_oasis_run_evaluation_paths(cfg, settings=resolved_settings)
    save_oasis_run_evaluation(cfg=cfg, checkpoint=checkpoint, result=result, paths=paths)
    return OASISRunEvaluationResult(
        config=cfg,
        checkpoint=checkpoint,
        result=result,
        paths=paths,
    )
