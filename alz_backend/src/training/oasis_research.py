"""Research-grade MONAI training pipeline for OASIS-1 binary 3D MRI classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import yaml

from src.configs.runtime import AppSettings, get_app_settings
from src.data.loaders import OASISLoaderConfig, build_oasis_dataloaders
from src.evaluation.metrics import build_confusion_matrix, compute_binary_classification_metrics
from src.models.factory import OASISModelConfig, build_model, describe_model_config, load_oasis_model_config
from src.transforms.oasis_transforms import OASISSpatialConfig, OASISTransformConfig, load_oasis_transform_config
from src.utils.io_utils import ensure_directory, resolve_project_root
from src.utils.monai_utils import load_torch_symbols
from src.utils.seed import build_seed_snapshot, set_global_seed

from .trainer_utils import build_classification_loss, build_optimizer, build_scheduler

_load_torch_symbols = load_torch_symbols


class ResearchTrainingError(ValueError):
    """Raised when a research training run cannot be configured or executed safely."""


@dataclass(slots=True, frozen=True)
class ResearchDataConfig:
    """Data and transform settings for research-grade OASIS training."""

    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    cache_rate: float = 0.0
    image_size: tuple[int, int, int] = (64, 64, 64)
    seed: int = 42
    split_seed: int | None = None
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    weighted_sampling: bool = False
    max_train_batches: int | None = None
    max_val_batches: int | None = None


@dataclass(slots=True, frozen=True)
class OptimizerConfig:
    """Optimizer settings."""

    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9


@dataclass(slots=True, frozen=True)
class SchedulerConfig:
    """Learning-rate scheduler settings."""

    name: str = "none"
    step_size: int = 5
    gamma: float = 0.5
    patience: int = 2
    factor: float = 0.5


@dataclass(slots=True, frozen=True)
class LossConfig:
    """Loss-function settings."""

    name: str = "cross_entropy"


@dataclass(slots=True, frozen=True)
class EarlyStoppingConfig:
    """Early stopping settings."""

    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.0
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass(slots=True, frozen=True)
class CheckpointConfig:
    """Checkpointing settings."""

    resume_from: Path | None = None
    save_best: bool = True
    save_last: bool = True


@dataclass(slots=True, frozen=True)
class ResearchOASISTrainingConfig:
    """Top-level config for the research-grade OASIS training runner."""

    run_name: str = "oasis_research_baseline"
    epochs: int = 5
    device: str = "auto"
    mixed_precision: bool = False
    deterministic: bool = True
    dry_run: bool = False
    model_config_path: Path | None = None
    data: ResearchDataConfig = field(default_factory=ResearchDataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass(slots=True)
class ResearchRunPaths:
    """Resolved run folder paths."""

    run_root: Path
    checkpoint_root: Path
    metrics_root: Path
    reports_root: Path
    config_root: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    epoch_metrics_csv_path: Path
    epoch_metrics_json_path: Path
    confusion_matrix_path: Path
    summary_report_path: Path
    resolved_config_path: Path


@dataclass(slots=True)
class ResearchTrainingResult:
    """Artifacts and headline metrics produced by the research runner."""

    run_name: str
    run_root: Path
    best_checkpoint_path: Path | None
    last_checkpoint_path: Path | None
    epoch_metrics_csv_path: Path
    epoch_metrics_json_path: Path
    confusion_matrix_path: Path
    summary_report_path: Path
    resolved_config_path: Path
    best_epoch: int
    best_monitor_value: float
    stopped_early: bool
    final_metrics: dict[str, Any]


def default_oasis_train_config_path() -> Path:
    """Return the default research training YAML path."""

    return resolve_project_root() / "configs" / "oasis_train.yaml"


def _as_tuple(values: Any, *, cast_type: type, expected_length: int) -> tuple[Any, ...]:
    """Normalize YAML sequences into fixed-length tuples."""

    if not isinstance(values, (list, tuple)):
        raise ResearchTrainingError(f"Expected a sequence of length {expected_length}, got {values!r}")
    if len(values) != expected_length:
        raise ResearchTrainingError(f"Expected length {expected_length}, got {len(values)} for {values!r}")
    return tuple(cast_type(value) for value in values)


def _optional_path(raw_value: Any) -> Path | None:
    """Normalize optional path values from YAML."""

    if raw_value in {None, ""}:
        return None
    return Path(raw_value)


def _merge_training_config(default_config: ResearchOASISTrainingConfig, overrides: dict[str, Any]) -> ResearchOASISTrainingConfig:
    """Merge YAML overrides into a typed research training config."""

    data_section = dict(asdict(default_config.data))
    data_section.update(overrides.get("data", {}))
    if "image_size" in data_section:
        data_section["image_size"] = _as_tuple(data_section["image_size"], cast_type=int, expected_length=3)

    optimizer_section = dict(asdict(default_config.optimizer))
    optimizer_section.update(overrides.get("optimizer", {}))

    scheduler_section = dict(asdict(default_config.scheduler))
    scheduler_section.update(overrides.get("scheduler", {}))

    loss_section = dict(asdict(default_config.loss))
    loss_section.update(overrides.get("loss", {}))

    early_stopping_section = dict(asdict(default_config.early_stopping))
    early_stopping_section.update(overrides.get("early_stopping", {}))

    checkpoint_section = dict(asdict(default_config.checkpoint))
    checkpoint_section.update(overrides.get("checkpoint", {}))
    checkpoint_section["resume_from"] = _optional_path(checkpoint_section.get("resume_from"))

    return ResearchOASISTrainingConfig(
        run_name=str(overrides.get("run_name", default_config.run_name)),
        epochs=int(overrides.get("epochs", default_config.epochs)),
        device=str(overrides.get("device", default_config.device)),
        mixed_precision=bool(overrides.get("mixed_precision", default_config.mixed_precision)),
        deterministic=bool(overrides.get("deterministic", default_config.deterministic)),
        dry_run=bool(overrides.get("dry_run", default_config.dry_run)),
        model_config_path=_optional_path(overrides.get("model_config_path", default_config.model_config_path)),
        data=ResearchDataConfig(**data_section),
        optimizer=OptimizerConfig(**optimizer_section),
        scheduler=SchedulerConfig(**scheduler_section),
        loss=LossConfig(**loss_section),
        early_stopping=EarlyStoppingConfig(**early_stopping_section),
        checkpoint=CheckpointConfig(**checkpoint_section),
    )


def load_research_oasis_training_config(config_path: str | Path | None = None) -> ResearchOASISTrainingConfig:
    """Load the research OASIS training config from YAML."""

    resolved_path = Path(config_path) if config_path is not None else default_oasis_train_config_path()
    if not resolved_path.exists():
        raise FileNotFoundError(f"OASIS training config not found: {resolved_path}")
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ResearchTrainingError("OASIS training config YAML must decode to a dictionary.")
    return _merge_training_config(ResearchOASISTrainingConfig(), payload)


def _apply_dry_run_overrides(cfg: ResearchOASISTrainingConfig) -> ResearchOASISTrainingConfig:
    """Force tiny safe settings for dry-run mode."""

    if not cfg.dry_run:
        return cfg
    return ResearchOASISTrainingConfig(
        run_name=cfg.run_name,
        epochs=min(cfg.epochs, 1),
        device=cfg.device,
        mixed_precision=cfg.mixed_precision,
        deterministic=cfg.deterministic,
        dry_run=True,
        model_config_path=cfg.model_config_path,
        data=ResearchDataConfig(
            batch_size=cfg.data.batch_size,
            gradient_accumulation_steps=cfg.data.gradient_accumulation_steps,
            num_workers=cfg.data.num_workers,
            cache_rate=cfg.data.cache_rate,
            image_size=cfg.data.image_size,
            seed=cfg.data.seed,
            split_seed=cfg.data.split_seed,
            train_fraction=cfg.data.train_fraction,
            val_fraction=cfg.data.val_fraction,
            test_fraction=cfg.data.test_fraction,
            weighted_sampling=cfg.data.weighted_sampling,
            max_train_batches=cfg.data.max_train_batches or 2,
            max_val_batches=cfg.data.max_val_batches or 2,
        ),
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        loss=cfg.loss,
        early_stopping=cfg.early_stopping,
        checkpoint=cfg.checkpoint,
    )


def build_run_paths(settings: AppSettings, run_name: str) -> ResearchRunPaths:
    """Create and return the run folder structure under outputs/runs/oasis."""

    run_root = ensure_directory(settings.outputs_root / "runs" / "oasis" / run_name)
    checkpoint_root = ensure_directory(run_root / "checkpoints")
    metrics_root = ensure_directory(run_root / "metrics")
    reports_root = ensure_directory(run_root / "reports")
    config_root = ensure_directory(run_root / "configs")
    return ResearchRunPaths(
        run_root=run_root,
        checkpoint_root=checkpoint_root,
        metrics_root=metrics_root,
        reports_root=reports_root,
        config_root=config_root,
        best_checkpoint_path=checkpoint_root / "best_model.pt",
        last_checkpoint_path=checkpoint_root / "last_model.pt",
        epoch_metrics_csv_path=metrics_root / "epoch_metrics.csv",
        epoch_metrics_json_path=metrics_root / "epoch_metrics.json",
        confusion_matrix_path=metrics_root / "confusion_matrix.json",
        summary_report_path=reports_root / "summary_report.md",
        resolved_config_path=config_root / "resolved_config.json",
    )


def _resolve_device(requested_device: str, torch: object) -> str:
    """Resolve a device string."""

    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _save_resolved_config(
    *,
    cfg: ResearchOASISTrainingConfig,
    model_cfg: OASISModelConfig,
    paths: ResearchRunPaths,
) -> None:
    """Persist resolved training and model config for reproducibility."""

    payload = {
        "training": asdict(cfg),
        "model": describe_model_config(model_cfg),
        "seed": build_seed_snapshot(cfg.data.seed, deterministic=cfg.deterministic),
    }
    payload["training"]["model_config_path"] = str(cfg.model_config_path) if cfg.model_config_path else None
    payload["training"]["checkpoint"]["resume_from"] = (
        str(cfg.checkpoint.resume_from) if cfg.checkpoint.resume_from else None
    )
    paths.resolved_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_transform_config(cfg: ResearchOASISTrainingConfig) -> OASISTransformConfig:
    """Build the OASIS transform config with the training image size."""

    transform_cfg = load_oasis_transform_config()
    return OASISTransformConfig(
        load=transform_cfg.load,
        orientation=transform_cfg.orientation,
        spacing=transform_cfg.spacing,
        intensity=transform_cfg.intensity,
        skull_strip=transform_cfg.skull_strip,
        spatial=OASISSpatialConfig(spatial_size=cfg.data.image_size),
        augmentation=transform_cfg.augmentation,
    )


def _build_loaders(cfg: ResearchOASISTrainingConfig):
    """Build reproducible OASIS loaders for training and validation."""

    return build_oasis_dataloaders(
        OASISLoaderConfig(
            seed=cfg.data.seed,
            split_seed=cfg.data.split_seed,
            train_fraction=cfg.data.train_fraction,
            val_fraction=cfg.data.val_fraction,
            test_fraction=cfg.data.test_fraction,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            cache_rate=cfg.data.cache_rate,
            weighted_sampling=cfg.data.weighted_sampling,
            transform_config=_build_transform_config(cfg),
        )
    )


def _coerce_labels(raw_labels: object, torch: object, device: str) -> object:
    """Convert labels to a torch long tensor."""

    if hasattr(raw_labels, "to"):
        return raw_labels.to(device).long()
    return torch.as_tensor(raw_labels, device=device).long()


def _build_grad_scaler(torch: object, *, amp_enabled: bool) -> object:
    """Build a CUDA AMP grad scaler using the current PyTorch API when available."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def _autocast_context(torch: object, *, device: str, amp_enabled: bool) -> object:
    """Return an autocast context without triggering PyTorch deprecation warnings."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        return torch.amp.autocast(device_type=device_type, enabled=amp_enabled)
    return torch.cuda.amp.autocast(enabled=amp_enabled)


def _run_epoch(
    *,
    loader: object,
    model: object,
    loss_function: object,
    optimizer: object | None,
    torch: object,
    device: str,
    scaler: object | None,
    amp_enabled: bool,
    max_batches: int | None,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    """Run one train or validation epoch and return metrics-ready predictions."""

    training = optimizer is not None
    accumulation_steps = max(int(gradient_accumulation_steps), 1)
    if training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    total_loss = 0.0
    batch_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = batch["image"].to(device)
            labels = _coerce_labels(batch["label"], torch, device)

            with _autocast_context(torch, device=device, amp_enabled=amp_enabled):
                logits = model(inputs)
                loss = loss_function(logits, labels)
                backward_loss = loss / accumulation_steps if training and accumulation_steps > 1 else loss

            if training:
                should_step = (batch_index + 1) % accumulation_steps == 0
                if scaler is not None and amp_enabled:
                    scaler.scale(backward_loss).backward()
                    if should_step:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    backward_loss.backward()
                    if should_step:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            probabilities = torch.softmax(logits.detach(), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            total_loss += float(loss.item())
            batch_count += 1
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
            y_score.extend(probabilities[:, 1].detach().cpu().tolist())

    if training and batch_count % accumulation_steps != 0:
        if scaler is not None and amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if batch_count == 0:
        raise ResearchTrainingError("No batches were processed during an epoch.")

    metrics = compute_binary_classification_metrics(
        [int(label) for label in y_true],
        [int(prediction) for prediction in y_pred],
        y_score=[float(score) for score in y_score],
    )
    metrics["loss"] = total_loss / batch_count
    metrics["batch_count"] = batch_count
    metrics["y_true"] = [int(label) for label in y_true]
    metrics["y_pred"] = [int(prediction) for prediction in y_pred]
    metrics["y_score"] = [float(score) for score in y_score]
    return metrics


def _step_scheduler(scheduler: object | None, scheduler_name: str, val_loss: float) -> None:
    """Advance an optional scheduler."""

    if scheduler is None:
        return
    if scheduler_name.strip().lower() == "reduce_on_plateau":
        scheduler.step(val_loss)
        return
    scheduler.step()


def _is_improvement(value: float, best_value: float, *, mode: str, min_delta: float) -> bool:
    """Return whether an early-stopping/checkpoint metric improved."""

    if mode == "min":
        return value < (best_value - min_delta)
    if mode == "max":
        return value > (best_value + min_delta)
    raise ResearchTrainingError(f"Unsupported early stopping mode: {mode}")


def _resolve_monitor_value(metrics: dict[str, Any], monitor: str) -> float:
    """Resolve a monitor name from validation metrics using user-friendly aliases."""

    aliases = {
        "val_loss": "loss",
        "val_accuracy": "accuracy",
        "val_auroc": "auroc",
        "val_f1": "f1",
        "val_precision": "precision",
        "val_recall": "recall_sensitivity",
        "val_sensitivity": "sensitivity",
        "val_specificity": "specificity",
    }
    metric_key = aliases.get(monitor, monitor)
    if metric_key not in metrics:
        available_keys = sorted(key for key, value in metrics.items() if isinstance(value, (int, float)))
        raise ResearchTrainingError(
            f"Early stopping monitor {monitor!r} is not available. "
            f"Numeric metric keys include: {available_keys}"
        )
    return float(metrics[metric_key])


def _initial_best_value(mode: str) -> float:
    """Return the initial best monitor value."""

    if mode == "min":
        return float("inf")
    if mode == "max":
        return float("-inf")
    raise ResearchTrainingError(f"Unsupported early stopping mode: {mode}")


def _checkpoint_json_safe(value: Any) -> Any:
    """Convert checkpoint metadata into torch-safe primitive values."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _checkpoint_json_safe(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_json_safe(item) for item in value]
    return value


def _checkpoint_payload(
    *,
    epoch: int,
    model: object,
    optimizer: object,
    scheduler: object | None,
    best_monitor_value: float,
    best_epoch: int,
    cfg: ResearchOASISTrainingConfig,
) -> dict[str, Any]:
    """Build a checkpoint payload."""

    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_monitor_value": best_monitor_value,
        "best_epoch": best_epoch,
        "config": _checkpoint_json_safe(asdict(cfg)),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any], torch: object) -> None:
    """Save one checkpoint file."""

    ensure_directory(path.parent)
    torch.save(payload, path)


def _load_resume_checkpoint(
    *,
    path: Path,
    model: object,
    optimizer: object,
    scheduler: object | None,
    torch: object,
    device: str,
) -> tuple[int, float, int]:
    """Load a checkpoint and return start epoch, best value, and best epoch."""

    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_monitor_value"]), int(checkpoint["best_epoch"])


def _epoch_row(epoch: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any], learning_rate: float) -> dict[str, Any]:
    """Create one epoch metrics row."""

    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "accuracy": val_metrics["accuracy"],
        "auroc": val_metrics["auroc"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall_sensitivity"],
        "f1": val_metrics["f1"],
        "sensitivity": val_metrics["sensitivity"],
        "specificity": val_metrics["specificity"],
        "train_batches": train_metrics["batch_count"],
        "val_batches": val_metrics["batch_count"],
    }


def _write_epoch_metrics(rows: list[dict[str, Any]], paths: ResearchRunPaths) -> None:
    """Write epoch metrics as CSV and JSON."""

    pd.DataFrame(rows).to_csv(paths.epoch_metrics_csv_path, index=False)
    paths.epoch_metrics_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_confusion_matrix(val_metrics: dict[str, Any], paths: ResearchRunPaths) -> None:
    """Write final validation confusion matrix."""

    payload = {
        "label_order": [0, 1],
        "row_axis": "true_label",
        "column_axis": "predicted_label",
        "layout": "[[true_negative, false_positive], [false_negative, true_positive]]",
        "confusion_counts": val_metrics["confusion_counts"],
        "confusion_matrix": build_confusion_matrix(val_metrics["y_true"], val_metrics["y_pred"]),
    }
    paths.confusion_matrix_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary_report(
    *,
    cfg: ResearchOASISTrainingConfig,
    paths: ResearchRunPaths,
    rows: list[dict[str, Any]],
    best_epoch: int,
    best_monitor_value: float,
    stopped_early: bool,
    elapsed_seconds: float,
) -> None:
    """Write a human-readable final experiment report."""

    final_row = rows[-1]
    lines = [
        f"# {cfg.run_name}",
        "",
        "This is a research-grade OASIS-1 MONAI training run for backend development.",
        "It is decision-support research software, not diagnosis software.",
        "",
        "## Run",
        "",
        f"- epochs_requested: {cfg.epochs}",
        f"- epochs_completed: {len(rows)}",
        f"- dry_run: {cfg.dry_run}",
        f"- device: {cfg.device}",
        f"- mixed_precision: {cfg.mixed_precision}",
        f"- gradient_accumulation_steps: {cfg.data.gradient_accumulation_steps}",
        f"- stopped_early: {stopped_early}",
        f"- best_epoch: {best_epoch}",
        f"- best_monitor_value: {best_monitor_value}",
        f"- elapsed_seconds: {round(elapsed_seconds, 2)}",
        "",
        "## Final Validation Metrics",
        "",
        f"- val_loss: {final_row['val_loss']:.6f}",
        f"- accuracy: {final_row['accuracy']:.6f}",
        f"- auroc: {final_row['auroc']:.6f}",
        f"- precision: {final_row['precision']:.6f}",
        f"- recall: {final_row['recall']:.6f}",
        f"- f1: {final_row['f1']:.6f}",
        f"- sensitivity: {final_row['sensitivity']:.6f}",
        f"- specificity: {final_row['specificity']:.6f}",
        "",
        "## Artifacts",
        "",
        f"- best_checkpoint: {paths.best_checkpoint_path}",
        f"- last_checkpoint: {paths.last_checkpoint_path}",
        f"- epoch_metrics_csv: {paths.epoch_metrics_csv_path}",
        f"- epoch_metrics_json: {paths.epoch_metrics_json_path}",
        f"- confusion_matrix: {paths.confusion_matrix_path}",
        f"- resolved_config: {paths.resolved_config_path}",
        "",
        "## Failure Notes",
        "",
        "- Tiny dry runs are pipeline checks and should not be interpreted as model quality.",
        "- AUROC is reported as 0.0 when the evaluated split contains only one class.",
        "- A clinically meaningful baseline requires larger subject-safe training, calibration checks, and held-out test evaluation.",
    ]
    paths.summary_report_path.write_text("\n".join(lines), encoding="utf-8")


def run_research_oasis_training(
    config: ResearchOASISTrainingConfig | None = None,
    *,
    settings: AppSettings | None = None,
) -> ResearchTrainingResult:
    """Run the config-driven OASIS research training pipeline."""

    resolved_settings = settings or get_app_settings()
    cfg = _apply_dry_run_overrides(config or load_research_oasis_training_config())
    set_global_seed(cfg.data.seed, deterministic=cfg.deterministic)
    paths = build_run_paths(resolved_settings, cfg.run_name)
    model_cfg = load_oasis_model_config(cfg.model_config_path)
    _save_resolved_config(cfg=cfg, model_cfg=model_cfg, paths=paths)

    torch = _load_torch_symbols()["torch"]
    device = _resolve_device(cfg.device, torch)
    amp_enabled = bool(cfg.mixed_precision and device.startswith("cuda"))
    model = build_model(model_cfg).to(device)
    optimizer = build_optimizer(
        model,
        name=cfg.optimizer.name,
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        momentum=cfg.optimizer.momentum,
    )
    scheduler = build_scheduler(
        optimizer,
        name=cfg.scheduler.name,
        step_size=cfg.scheduler.step_size,
        gamma=cfg.scheduler.gamma,
        patience=cfg.scheduler.patience,
        factor=cfg.scheduler.factor,
    )
    loss_function = build_classification_loss(cfg.loss.name)
    scaler = _build_grad_scaler(torch, amp_enabled=amp_enabled)
    dataloaders = _build_loaders(cfg)

    start_epoch = 1
    best_monitor_value = _initial_best_value(cfg.early_stopping.mode)
    best_epoch = 0
    if cfg.checkpoint.resume_from is not None:
        start_epoch, best_monitor_value, best_epoch = _load_resume_checkpoint(
            path=cfg.checkpoint.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            torch=torch,
            device=device,
        )

    rows: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    stopped_early = False
    final_val_metrics: dict[str, Any] | None = None
    start_time = perf_counter()
    print(
        f"Starting OASIS training run {cfg.run_name} on device={device} "
        f"for up to {cfg.epochs} epochs. run_root={paths.run_root}"
    )
    print(f"Epoch metrics will be written to {paths.epoch_metrics_csv_path}")
    if cfg.checkpoint.resume_from is not None:
        print(f"Resuming from checkpoint {cfg.checkpoint.resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = _run_epoch(
            loader=dataloaders.train_loader,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            torch=torch,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_batches=cfg.data.max_train_batches,
            gradient_accumulation_steps=cfg.data.gradient_accumulation_steps,
        )
        val_metrics = _run_epoch(
            loader=dataloaders.val_loader,
            model=model,
            loss_function=loss_function,
            optimizer=None,
            torch=torch,
            device=device,
            scaler=None,
            amp_enabled=amp_enabled,
            max_batches=cfg.data.max_val_batches,
            gradient_accumulation_steps=1,
        )
        final_val_metrics = val_metrics
        learning_rate = float(optimizer.param_groups[0]["lr"])
        rows.append(_epoch_row(epoch, train_metrics, val_metrics, learning_rate))
        _write_epoch_metrics(rows, paths)
        _step_scheduler(scheduler, cfg.scheduler.name, float(val_metrics["loss"]))

        monitor_value = _resolve_monitor_value(val_metrics, cfg.early_stopping.monitor)
        improved = _is_improvement(
            monitor_value,
            best_monitor_value,
            mode=cfg.early_stopping.mode,
            min_delta=cfg.early_stopping.min_delta,
        )
        if improved:
            best_monitor_value = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_payload = _checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_monitor_value=best_monitor_value,
                best_epoch=best_epoch,
                cfg=cfg,
            )
            if cfg.checkpoint.save_best:
                _save_checkpoint(paths.best_checkpoint_path, checkpoint_payload, torch)
        else:
            epochs_without_improvement += 1

        checkpoint_payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_monitor_value=best_monitor_value,
            best_epoch=best_epoch,
            cfg=cfg,
        )
        if cfg.checkpoint.save_last:
            _save_checkpoint(paths.last_checkpoint_path, checkpoint_payload, torch)

        improvement_status = "improved" if improved else f"no_improve={epochs_without_improvement}"
        print(
            f"[{cfg.run_name}] epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"best_epoch={best_epoch} "
            f"best_{cfg.early_stopping.monitor}={best_monitor_value:.4f} "
            f"{improvement_status}"
        )

        if cfg.early_stopping.enabled and epochs_without_improvement >= cfg.early_stopping.patience:
            stopped_early = True
            print(
                f"Early stopping triggered for run {cfg.run_name} at epoch {epoch}. "
                f"Best epoch={best_epoch} best_{cfg.early_stopping.monitor}={best_monitor_value:.4f}"
            )
            break

    if not rows or final_val_metrics is None:
        raise ResearchTrainingError("Training finished without producing epoch metrics.")

    elapsed_seconds = perf_counter() - start_time
    _write_confusion_matrix(final_val_metrics, paths)
    _write_summary_report(
        cfg=cfg,
        paths=paths,
        rows=rows,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor_value,
        stopped_early=stopped_early,
        elapsed_seconds=elapsed_seconds,
    )

    return ResearchTrainingResult(
        run_name=cfg.run_name,
        run_root=paths.run_root,
        best_checkpoint_path=paths.best_checkpoint_path if paths.best_checkpoint_path.exists() else None,
        last_checkpoint_path=paths.last_checkpoint_path if paths.last_checkpoint_path.exists() else None,
        epoch_metrics_csv_path=paths.epoch_metrics_csv_path,
        epoch_metrics_json_path=paths.epoch_metrics_json_path,
        confusion_matrix_path=paths.confusion_matrix_path,
        summary_report_path=paths.summary_report_path,
        resolved_config_path=paths.resolved_config_path,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor_value,
        stopped_early=stopped_early,
        final_metrics=rows[-1],
    )
