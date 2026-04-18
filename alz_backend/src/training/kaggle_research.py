"""Research-style MONAI training pipeline for the separate Kaggle Alzheimer branch.

This module keeps the Kaggle branch isolated from OASIS:
- Kaggle remains a separate slice-based or volume-based experiment track.
- Numeric labels are used only inside the Kaggle workflow.
- Reports clearly warn when the dataset is 2D slice-based and not equivalent to
  subject-level 3D OASIS MRI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.configs.runtime import AppSettings, get_app_settings
from src.data.kaggle_dataset import build_kaggle_monai_dataloader, infer_kaggle_dataset_type, load_kaggle_manifest
from src.models.kaggle_model import KaggleMonaiModelConfig, build_kaggle_monai_network
from src.transforms.kaggle_transforms import KaggleSpatialConfig, KaggleTransformConfig, load_kaggle_transform_config
from src.utils.io_utils import ensure_directory, resolve_project_root
from src.utils.monai_utils import load_torch_symbols
from src.utils.seed import build_seed_snapshot, set_global_seed

from .oasis_research import CheckpointConfig, EarlyStoppingConfig, LossConfig, OptimizerConfig, SchedulerConfig
from .trainer_utils import build_classification_loss, build_optimizer, build_scheduler

logger = logging.getLogger(__name__)
_load_torch_symbols = load_torch_symbols


class KaggleResearchTrainingError(ValueError):
    """Raised when the Kaggle research runner cannot execute safely."""


KAGGLE_RESEARCH_WARNINGS = [
    "Kaggle remains a separate research branch and is not automatically equivalent to OASIS labels.",
    "Current local Kaggle evidence is slice-based (`2d_slices`) rather than subject-level 3D MRI.",
    "Validation and test evidence should be interpreted as Kaggle-only performance, not OASIS generalization.",
]


@dataclass(slots=True, frozen=True)
class ResearchKaggleDataConfig:
    """Data and transform settings for Kaggle research runs."""

    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    cache_rate: float = 0.0
    image_size_2d: tuple[int, int] = (224, 224)
    image_size_3d: tuple[int, int, int] = (128, 128, 128)
    seed: int = 42
    label_map: dict[str, int] | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None


@dataclass(slots=True, frozen=True)
class ResearchKaggleTrainingConfig:
    """Top-level config for Kaggle research training."""

    run_name: str = "kaggle_baseline_2d"
    epochs: int = 3
    device: str = "auto"
    mixed_precision: bool = True
    deterministic: bool = True
    dry_run: bool = False
    dropout_prob: float = 0.0
    data: ResearchKaggleDataConfig = field(default_factory=ResearchKaggleDataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    early_stopping: EarlyStoppingConfig = field(
        default_factory=lambda: EarlyStoppingConfig(
            enabled=True,
            patience=3,
            min_delta=0.0,
            monitor="val_macro_f1",
            mode="max",
        )
    )
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass(slots=True)
class KaggleResearchRunPaths:
    """Resolved run-folder structure for Kaggle experiments."""

    run_root: Path
    checkpoint_root: Path
    metrics_root: Path
    reports_root: Path
    config_root: Path
    evaluation_root: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    epoch_metrics_csv_path: Path
    epoch_metrics_json_path: Path
    final_metrics_path: Path
    val_confusion_matrix_path: Path
    test_confusion_matrix_path: Path
    val_predictions_path: Path
    test_predictions_path: Path
    summary_report_path: Path
    resolved_config_path: Path


@dataclass(slots=True)
class ResearchKaggleTrainingResult:
    """Artifacts and metrics produced by the Kaggle research runner."""

    run_name: str
    run_root: Path
    best_checkpoint_path: Path | None
    last_checkpoint_path: Path | None
    epoch_metrics_csv_path: Path
    epoch_metrics_json_path: Path
    final_metrics_path: Path
    val_confusion_matrix_path: Path
    test_confusion_matrix_path: Path
    val_predictions_path: Path
    test_predictions_path: Path
    summary_report_path: Path
    resolved_config_path: Path
    best_epoch: int
    best_monitor_value: float
    stopped_early: bool
    final_metrics: dict[str, Any]


@dataclass(slots=True)
class _KaggleLoaderBundle:
    """Resolved dataloaders and label metadata for a Kaggle run."""

    train_loader: object
    val_loader: object
    test_loader: object
    dataset_type: str
    class_names: tuple[str, ...]
    runtime_label_map: dict[str, int] | None = None


@dataclass(slots=True)
class _KaggleEvaluationResult:
    """Metrics and prediction rows for one Kaggle split."""

    split: str
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]


def default_kaggle_train_config_path() -> Path:
    """Return the default Kaggle research training YAML path."""

    return resolve_project_root() / "configs" / "kaggle_train.yaml"


def _as_tuple(values: Any, *, cast_type: type, expected_length: int) -> tuple[Any, ...]:
    """Normalize YAML sequences into fixed-length tuples."""

    if not isinstance(values, (list, tuple)):
        raise KaggleResearchTrainingError(f"Expected a sequence of length {expected_length}, got {values!r}")
    if len(values) != expected_length:
        raise KaggleResearchTrainingError(f"Expected length {expected_length}, got {len(values)} for {values!r}")
    return tuple(cast_type(value) for value in values)


def _optional_path(raw_value: Any) -> Path | None:
    """Normalize optional path values from YAML."""

    if raw_value in {None, ""}:
        return None
    return Path(raw_value)


def _optional_label_map(raw_value: Any) -> dict[str, int] | None:
    """Normalize an optional YAML label map."""

    if raw_value in {None, ""}:
        return None
    if not isinstance(raw_value, dict):
        raise KaggleResearchTrainingError("Kaggle `data.label_map` must decode to a dictionary when provided.")
    return {str(key): int(value) for key, value in raw_value.items()}


def _merge_training_config(
    default_config: ResearchKaggleTrainingConfig,
    overrides: dict[str, Any],
) -> ResearchKaggleTrainingConfig:
    """Merge YAML overrides into a strongly typed Kaggle config."""

    data_section = dict(asdict(default_config.data))
    data_section.update(overrides.get("data", {}))
    if "image_size_2d" in data_section:
        data_section["image_size_2d"] = _as_tuple(data_section["image_size_2d"], cast_type=int, expected_length=2)
    if "image_size_3d" in data_section:
        data_section["image_size_3d"] = _as_tuple(data_section["image_size_3d"], cast_type=int, expected_length=3)
    if "label_map" in data_section:
        data_section["label_map"] = _optional_label_map(data_section.get("label_map"))

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

    return ResearchKaggleTrainingConfig(
        run_name=str(overrides.get("run_name", default_config.run_name)),
        epochs=int(overrides.get("epochs", default_config.epochs)),
        device=str(overrides.get("device", default_config.device)),
        mixed_precision=bool(overrides.get("mixed_precision", default_config.mixed_precision)),
        deterministic=bool(overrides.get("deterministic", default_config.deterministic)),
        dry_run=bool(overrides.get("dry_run", default_config.dry_run)),
        dropout_prob=float(overrides.get("dropout_prob", default_config.dropout_prob)),
        data=ResearchKaggleDataConfig(**data_section),
        optimizer=OptimizerConfig(**optimizer_section),
        scheduler=SchedulerConfig(**scheduler_section),
        loss=LossConfig(**loss_section),
        early_stopping=EarlyStoppingConfig(**early_stopping_section),
        checkpoint=CheckpointConfig(**checkpoint_section),
    )


def load_research_kaggle_training_config(config_path: str | Path | None = None) -> ResearchKaggleTrainingConfig:
    """Load the Kaggle research training config from YAML."""

    resolved_path = Path(config_path) if config_path is not None else default_kaggle_train_config_path()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Kaggle training config not found: {resolved_path}")
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise KaggleResearchTrainingError("Kaggle training config YAML must decode to a dictionary.")
    return _merge_training_config(ResearchKaggleTrainingConfig(), payload)


def _apply_dry_run_overrides(cfg: ResearchKaggleTrainingConfig) -> ResearchKaggleTrainingConfig:
    """Force tiny safe settings for Kaggle dry runs."""

    if not cfg.dry_run:
        return cfg
    return ResearchKaggleTrainingConfig(
        run_name=cfg.run_name,
        epochs=min(cfg.epochs, 1),
        device=cfg.device,
        mixed_precision=cfg.mixed_precision,
        deterministic=cfg.deterministic,
        dry_run=True,
        dropout_prob=cfg.dropout_prob,
        data=ResearchKaggleDataConfig(
            batch_size=cfg.data.batch_size,
            gradient_accumulation_steps=cfg.data.gradient_accumulation_steps,
            num_workers=cfg.data.num_workers,
            cache_rate=cfg.data.cache_rate,
            image_size_2d=cfg.data.image_size_2d,
            image_size_3d=cfg.data.image_size_3d,
            seed=cfg.data.seed,
            label_map=cfg.data.label_map,
            max_train_batches=cfg.data.max_train_batches or 4,
            max_val_batches=cfg.data.max_val_batches or 2,
            max_test_batches=cfg.data.max_test_batches or 2,
        ),
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        loss=cfg.loss,
        early_stopping=cfg.early_stopping,
        checkpoint=cfg.checkpoint,
    )


def build_kaggle_run_paths(settings: AppSettings, run_name: str) -> KaggleResearchRunPaths:
    """Create and return the run-folder structure under outputs/runs/kaggle."""

    run_root = ensure_directory(settings.outputs_root / "runs" / "kaggle" / run_name)
    checkpoint_root = ensure_directory(run_root / "checkpoints")
    metrics_root = ensure_directory(run_root / "metrics")
    reports_root = ensure_directory(run_root / "reports")
    config_root = ensure_directory(run_root / "configs")
    evaluation_root = ensure_directory(run_root / "evaluation")
    return KaggleResearchRunPaths(
        run_root=run_root,
        checkpoint_root=checkpoint_root,
        metrics_root=metrics_root,
        reports_root=reports_root,
        config_root=config_root,
        evaluation_root=evaluation_root,
        best_checkpoint_path=checkpoint_root / "best_model.pt",
        last_checkpoint_path=checkpoint_root / "last_model.pt",
        epoch_metrics_csv_path=metrics_root / "epoch_metrics.csv",
        epoch_metrics_json_path=metrics_root / "epoch_metrics.json",
        final_metrics_path=metrics_root / "final_metrics.json",
        val_confusion_matrix_path=evaluation_root / "val_confusion_matrix.json",
        test_confusion_matrix_path=evaluation_root / "test_confusion_matrix.json",
        val_predictions_path=evaluation_root / "val_predictions.csv",
        test_predictions_path=evaluation_root / "test_predictions.csv",
        summary_report_path=reports_root / "summary_report.md",
        resolved_config_path=config_root / "resolved_config.json",
    )


def _resolve_device(requested_device: str, torch: object) -> str:
    """Resolve an execution device string."""

    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_transform_config(cfg: ResearchKaggleTrainingConfig) -> KaggleTransformConfig:
    """Load the base Kaggle transform config and override spatial sizes."""

    transform_cfg = load_kaggle_transform_config()
    return KaggleTransformConfig(
        load=transform_cfg.load,
        orientation=transform_cfg.orientation,
        spacing=transform_cfg.spacing,
        intensity=transform_cfg.intensity,
        channel=transform_cfg.channel,
        foreground=transform_cfg.foreground,
        spatial=KaggleSpatialConfig(
            image_size_2d=cfg.data.image_size_2d,
            image_size_3d=cfg.data.image_size_3d,
            final_op_2d=transform_cfg.spatial.final_op_2d,
            final_op_3d=transform_cfg.spatial.final_op_3d,
        ),
        augmentation=transform_cfg.augmentation,
    )


def _resolve_class_names_and_label_map(
    manifest_frame: pd.DataFrame,
    explicit_label_map: dict[str, int] | None,
) -> tuple[tuple[str, ...], dict[str, int] | None]:
    """Resolve the Kaggle class ordering and any runtime label map."""

    if explicit_label_map:
        ordered_pairs = sorted(explicit_label_map.items(), key=lambda item: int(item[1]))
        ordered_indices = [int(index) for _, index in ordered_pairs]
        if ordered_indices != list(range(len(ordered_indices))):
            raise KaggleResearchTrainingError(
                "Explicit Kaggle label_map indices must be contiguous starting at 0. "
                f"Observed indices: {ordered_indices}"
            )
        return tuple(label_name for label_name, _ in ordered_pairs), dict(explicit_label_map)

    labeled = manifest_frame.loc[manifest_frame["label"].notna() & manifest_frame["label_name"].notna(), ["label", "label_name"]]
    if not labeled.empty:
        mapping: dict[int, str] = {}
        for row in labeled.itertuples(index=False):
            label_index = int(float(row.label))
            label_name = str(row.label_name).strip()
            if label_index in mapping and mapping[label_index] != label_name:
                raise KaggleResearchTrainingError(
                    f"Numeric Kaggle label {label_index} maps to multiple class names: "
                    f"{mapping[label_index]!r} and {label_name!r}"
                )
            mapping[label_index] = label_name

        ordered_indices = sorted(mapping)
        expected_indices = list(range(len(ordered_indices)))
        if ordered_indices != expected_indices:
            raise KaggleResearchTrainingError(
                "Kaggle numeric labels must be contiguous starting at 0 for the current training pipeline. "
                f"Observed indices: {ordered_indices}"
            )
        return tuple(mapping[index] for index in ordered_indices), None

    label_names = sorted(
        {
            str(value).strip()
            for value in manifest_frame["label_name"].dropna().tolist()
            if str(value).strip()
        }
    )
    if not label_names:
        raise KaggleResearchTrainingError(
            "Kaggle training requires either numeric labels or non-empty label_name values in the train manifest."
        )
    runtime_label_map = {label_name: index for index, label_name in enumerate(label_names)}
    return tuple(label_names), runtime_label_map


def _build_loaders(
    cfg: ResearchKaggleTrainingConfig,
    settings: AppSettings,
) -> _KaggleLoaderBundle:
    """Build reproducible Kaggle train/val/test dataloaders."""

    train_manifest = load_kaggle_manifest(settings, split="train")
    val_manifest = load_kaggle_manifest(settings, split="val")
    test_manifest = load_kaggle_manifest(settings, split="test")

    dataset_type = infer_kaggle_dataset_type(train_manifest)
    class_names, runtime_label_map = _resolve_class_names_and_label_map(train_manifest, cfg.data.label_map)
    transform_config = _build_transform_config(cfg)

    logger.info("Building Kaggle dataloaders for dataset_type=%s with %d classes", dataset_type, len(class_names))

    train_loader = build_kaggle_monai_dataloader(
        settings,
        split="train",
        training=True,
        batch_size=cfg.data.batch_size,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
        transform_config=transform_config,
        label_map=runtime_label_map,
        require_labels=True,
    )
    val_loader = build_kaggle_monai_dataloader(
        settings,
        split="val",
        training=False,
        batch_size=cfg.data.batch_size,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
        transform_config=transform_config,
        label_map=runtime_label_map,
        require_labels=True,
    )
    test_loader = build_kaggle_monai_dataloader(
        settings,
        split="test",
        training=False,
        batch_size=cfg.data.batch_size,
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
        transform_config=transform_config,
        label_map=runtime_label_map,
        require_labels=True,
    )

    if infer_kaggle_dataset_type(val_manifest) != dataset_type or infer_kaggle_dataset_type(test_manifest) != dataset_type:
        raise KaggleResearchTrainingError("Kaggle train/val/test manifests must share one dataset_type.")

    return _KaggleLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_type=dataset_type,
        class_names=class_names,
        runtime_label_map=runtime_label_map,
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


def _compute_macro_ovr_auroc(
    y_true: list[int],
    y_score: list[list[float]],
    *,
    labels: list[int],
) -> float:
    """Compute macro one-vs-rest AUROC for binary or multiclass logits."""

    if not y_true or not y_score or len(set(y_true)) < 2:
        return 0.0

    probabilities = np.asarray(y_score, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise KaggleResearchTrainingError(
            f"Expected probability rows shaped (N, {len(labels)}), got {probabilities.shape}."
        )
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        return 0.0
    probabilities = probabilities / row_sums

    try:
        if len(labels) == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:
        return 0.0


def _compute_multiclass_metrics(
    y_true: list[int],
    y_pred: list[int],
    *,
    y_score: list[list[float]],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compute multiclass classification metrics for Kaggle runs."""

    if len(y_true) != len(y_pred):
        raise KaggleResearchTrainingError(
            f"Expected y_true and y_pred to have the same length, got {len(y_true)} and {len(y_pred)}."
        )

    labels = list(range(len(class_names)))
    sample_count = len(y_true)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "macro_ovr_auroc": 0.0,
            "per_class_metrics": [],
            "confusion_matrix": [[0 for _ in class_names] for _ in class_names],
            "label_order": labels,
            "class_names": list(class_names),
        }

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    class_precision, class_recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    confusion = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="pred"),
        dropna=False,
    )
    confusion = confusion.reindex(index=labels, columns=labels, fill_value=0)

    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    balanced_accuracy = 0.0
    if len(set(y_true)) >= 2:
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "sample_count": sample_count,
        "accuracy": correct / sample_count,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_ovr_auroc": _compute_macro_ovr_auroc(y_true, y_score, labels=labels),
        "per_class_metrics": [
            {
                "label": label_index,
                "label_name": class_names[label_index],
                "precision": float(class_precision[label_index]),
                "recall": float(class_recall[label_index]),
                "f1": float(class_f1[label_index]),
                "support": int(support[label_index]),
            }
            for label_index in labels
        ],
        "confusion_matrix": confusion.astype(int).values.tolist(),
        "label_order": labels,
        "class_names": list(class_names),
    }


def _expand_batch_values(raw_value: Any, batch_size: int) -> list[Any]:
    """Expand a batched value into one item per sample."""

    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    if hasattr(raw_value, "tolist") and not isinstance(raw_value, str):
        converted = raw_value.tolist()
        if isinstance(converted, list):
            return converted
    return [raw_value for _ in range(batch_size)]


def _extract_meta_rows(raw_meta: Any, batch_size: int) -> list[dict[str, Any]]:
    """Convert a collated metadata payload back into row-wise dictionaries."""

    if raw_meta is None:
        return [{} for _ in range(batch_size)]
    if isinstance(raw_meta, list):
        return [dict(item) if isinstance(item, dict) else {} for item in raw_meta]
    if isinstance(raw_meta, dict):
        rows = [{} for _ in range(batch_size)]
        for key, value in raw_meta.items():
            for row_index, row_value in enumerate(_expand_batch_values(value, batch_size)):
                if row_index < batch_size:
                    rows[row_index][str(key)] = row_value
        return rows
    return [{} for _ in range(batch_size)]


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
    class_names: tuple[str, ...],
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
    y_score: list[list[float]] = []

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

            if training:
                scaled_loss = loss / accumulation_steps
                if scaler is not None and amp_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = ((batch_index + 1) % accumulation_steps == 0) or (
                    max_batches is not None and batch_index + 1 >= max_batches
                )
                if should_step:
                    if scaler is not None and amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            probabilities = torch.softmax(logits.detach(), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            total_loss += float(loss.detach().item())
            batch_count += 1
            y_true.extend(int(item) for item in labels.detach().cpu().tolist())
            y_pred.extend(int(item) for item in predictions.detach().cpu().tolist())
            y_score.extend([[float(value) for value in row] for row in probabilities.detach().cpu().tolist()])

    if batch_count == 0:
        raise KaggleResearchTrainingError("A Kaggle epoch produced zero batches. Check split manifests and max_*_batches.")

    metrics = _compute_multiclass_metrics(y_true, y_pred, y_score=y_score, class_names=class_names)
    metrics["loss"] = total_loss / batch_count
    metrics["batch_count"] = batch_count
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_score"] = y_score
    return metrics


def _evaluate_loader(
    *,
    split: str,
    loader: object,
    model: object,
    loss_function: object,
    torch: object,
    device: str,
    amp_enabled: bool,
    max_batches: int | None,
    class_names: tuple[str, ...],
    dataset_type: str,
) -> _KaggleEvaluationResult:
    """Evaluate one Kaggle loader and collect per-sample predictions."""

    model.eval()
    total_loss = 0.0
    batch_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[list[float]] = []
    predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = batch["image"].to(device)
            labels = _coerce_labels(batch["label"], torch, device)
            with _autocast_context(torch, device=device, amp_enabled=amp_enabled):
                logits = model(inputs)
                loss = loss_function(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predicted_indices = torch.argmax(probabilities, dim=1)

            label_rows = labels.detach().cpu().tolist()
            probability_rows = probabilities.detach().cpu().tolist()
            predicted_rows = predicted_indices.detach().cpu().tolist()
            batch_size = len(label_rows)
            subject_rows = _expand_batch_values(batch.get("subject_id"), batch_size)
            path_rows = _expand_batch_values(batch.get("image_path"), batch_size)
            label_name_rows = _expand_batch_values(batch.get("label_name"), batch_size)
            meta_rows = _extract_meta_rows(batch.get("meta"), batch_size)

            total_loss += float(loss.detach().item())
            batch_count += 1
            for item_index in range(batch_size):
                true_label = int(label_rows[item_index])
                predicted_label = int(predicted_rows[item_index])
                probability_row = [float(value) for value in probability_rows[item_index]]
                y_true.append(true_label)
                y_pred.append(predicted_label)
                y_score.append(probability_row)
                predictions.append(
                    {
                        "split": split,
                        "image_path": str(path_rows[item_index]) if path_rows[item_index] is not None else "",
                        "subject_id": None if subject_rows[item_index] in {"", None} else str(subject_rows[item_index]),
                        "true_label": true_label,
                        "true_label_name": class_names[true_label] if true_label < len(class_names) else str(label_name_rows[item_index]),
                        "predicted_label": predicted_label,
                        "predicted_label_name": class_names[predicted_label] if predicted_label < len(class_names) else f"class_{predicted_label}",
                        "confidence": float(max(probability_row)),
                        "probabilities_json": json.dumps(probability_row),
                        "dataset": "kaggle_alz",
                        "dataset_type": dataset_type,
                        "subset_source": meta_rows[item_index].get("subset"),
                    }
                )

    if batch_count == 0:
        raise KaggleResearchTrainingError(f"Kaggle evaluation for split={split!r} produced zero batches.")

    metrics = _compute_multiclass_metrics(y_true, y_pred, y_score=y_score, class_names=class_names)
    metrics["loss"] = total_loss / batch_count
    metrics["batch_count"] = batch_count
    return _KaggleEvaluationResult(split=split, metrics=metrics, predictions=predictions)


def _step_scheduler(scheduler: object | None, scheduler_name: str, val_loss: float) -> None:
    """Advance an optional scheduler."""

    if scheduler is None:
        return
    if scheduler_name.strip().lower() == "reduce_on_plateau":
        scheduler.step(val_loss)
        return
    scheduler.step()


def _is_improvement(value: float, best_value: float, *, mode: str, min_delta: float) -> bool:
    """Return whether a checkpoint monitor improved."""

    if mode == "min":
        return value < (best_value - min_delta)
    if mode == "max":
        return value > (best_value + min_delta)
    raise KaggleResearchTrainingError(f"Unsupported early stopping mode: {mode}")


def _resolve_monitor_value(metrics: dict[str, Any], monitor: str) -> float:
    """Resolve a monitor name from validation metrics using friendly aliases."""

    aliases = {
        "val_loss": "loss",
        "val_accuracy": "accuracy",
        "val_balanced_accuracy": "balanced_accuracy",
        "val_macro_precision": "macro_precision",
        "val_macro_recall": "macro_recall",
        "val_macro_f1": "macro_f1",
        "val_macro_ovr_auroc": "macro_ovr_auroc",
    }
    metric_key = aliases.get(monitor, monitor)
    if metric_key not in metrics:
        available_keys = sorted(key for key, value in metrics.items() if isinstance(value, (int, float)))
        raise KaggleResearchTrainingError(
            f"Early stopping monitor {monitor!r} is not available. Numeric metric keys include: {available_keys}"
        )
    return float(metrics[metric_key])


def _initial_best_value(mode: str) -> float:
    """Return the initial best monitor value."""

    if mode == "min":
        return float("inf")
    if mode == "max":
        return float("-inf")
    raise KaggleResearchTrainingError(f"Unsupported early stopping mode: {mode}")


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
    cfg: ResearchKaggleTrainingConfig,
    dataset_type: str,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build one checkpoint payload."""

    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_monitor_value": best_monitor_value,
        "best_epoch": best_epoch,
        "dataset_type": dataset_type,
        "class_names": list(class_names),
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


def _save_resolved_config(
    *,
    cfg: ResearchKaggleTrainingConfig,
    class_names: tuple[str, ...],
    dataset_type: str,
    runtime_label_map: dict[str, int] | None,
    paths: KaggleResearchRunPaths,
) -> None:
    """Persist resolved training config for reproducibility."""

    payload = {
        "training": asdict(cfg),
        "dataset_type": dataset_type,
        "class_names": list(class_names),
        "runtime_label_map": runtime_label_map,
        "seed": build_seed_snapshot(cfg.data.seed, deterministic=cfg.deterministic),
        "notes": list(KAGGLE_RESEARCH_WARNINGS),
    }
    payload["training"]["checkpoint"]["resume_from"] = (
        str(cfg.checkpoint.resume_from) if cfg.checkpoint.resume_from else None
    )
    paths.resolved_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _epoch_row(epoch: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any], learning_rate: float) -> dict[str, Any]:
    """Create one epoch metrics row."""

    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "accuracy": val_metrics["accuracy"],
        "balanced_accuracy": val_metrics["balanced_accuracy"],
        "macro_precision": val_metrics["macro_precision"],
        "macro_recall": val_metrics["macro_recall"],
        "macro_f1": val_metrics["macro_f1"],
        "weighted_f1": val_metrics["weighted_f1"],
        "macro_ovr_auroc": val_metrics["macro_ovr_auroc"],
        "train_batches": train_metrics["batch_count"],
        "val_batches": val_metrics["batch_count"],
    }


def _write_epoch_metrics(rows: list[dict[str, Any]], paths: KaggleResearchRunPaths) -> None:
    """Write epoch metrics as CSV and JSON."""

    pd.DataFrame(rows).to_csv(paths.epoch_metrics_csv_path, index=False)
    paths.epoch_metrics_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_confusion_matrix(
    evaluation: _KaggleEvaluationResult,
    *,
    paths: KaggleResearchRunPaths,
    class_names: tuple[str, ...],
) -> None:
    """Write one split confusion matrix JSON."""

    payload = {
        "split": evaluation.split,
        "label_order": list(range(len(class_names))),
        "label_names": list(class_names),
        "row_axis": "true_label",
        "column_axis": "predicted_label",
        "confusion_matrix": evaluation.metrics["confusion_matrix"],
    }
    output_path = paths.val_confusion_matrix_path if evaluation.split == "val" else paths.test_confusion_matrix_path
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_predictions_csv(evaluation: _KaggleEvaluationResult, *, paths: KaggleResearchRunPaths) -> None:
    """Write split prediction rows as CSV."""

    output_path = paths.val_predictions_path if evaluation.split == "val" else paths.test_predictions_path
    pd.DataFrame(evaluation.predictions).to_csv(output_path, index=False)


def _json_safe_metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove raw probability arrays before persisting metric summaries."""

    disallowed = {"y_true", "y_pred", "y_score"}
    return {key: value for key, value in metrics.items() if key not in disallowed}


def _write_final_metrics(
    *,
    cfg: ResearchKaggleTrainingConfig,
    paths: KaggleResearchRunPaths,
    dataset_type: str,
    class_names: tuple[str, ...],
    runtime_label_map: dict[str, int] | None,
    best_epoch: int,
    best_monitor_value: float,
    val_evaluation: _KaggleEvaluationResult,
    test_evaluation: _KaggleEvaluationResult,
    stopped_early: bool,
) -> dict[str, Any]:
    """Write the final nested metrics JSON and return the payload."""

    payload = {
        "run_name": cfg.run_name,
        "dataset": "kaggle_alz",
        "dataset_type": dataset_type,
        "class_names": list(class_names),
        "runtime_label_map": runtime_label_map,
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor_value,
        "stopped_early": stopped_early,
        "warnings": list(KAGGLE_RESEARCH_WARNINGS),
        "validation": _json_safe_metric_summary(val_evaluation.metrics),
        "test": _json_safe_metric_summary(test_evaluation.metrics),
    }
    paths.final_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_summary_report(
    *,
    cfg: ResearchKaggleTrainingConfig,
    paths: KaggleResearchRunPaths,
    rows: list[dict[str, Any]],
    dataset_type: str,
    class_names: tuple[str, ...],
    runtime_label_map: dict[str, int] | None,
    best_epoch: int,
    best_monitor_value: float,
    stopped_early: bool,
    elapsed_seconds: float,
    final_payload: dict[str, Any],
) -> None:
    """Write a human-readable final experiment report."""

    final_row = rows[-1]
    val_metrics = final_payload["validation"]
    test_metrics = final_payload["test"]
    lines = [
        f"# {cfg.run_name}",
        "",
        "This is a separate Kaggle MONAI training run for backend research support.",
        "It is not an OASIS-equivalent medical validation result and must stay isolated from the OASIS branch.",
        "",
        "## Run",
        "",
        f"- epochs_requested: {cfg.epochs}",
        f"- epochs_completed: {len(rows)}",
        f"- dataset_type: {dataset_type}",
        f"- class_count: {len(class_names)}",
        f"- runtime_label_map_used: {runtime_label_map is not None}",
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
        f"- accuracy: {val_metrics['accuracy']:.6f}",
        f"- balanced_accuracy: {val_metrics['balanced_accuracy']:.6f}",
        f"- macro_f1: {val_metrics['macro_f1']:.6f}",
        f"- macro_ovr_auroc: {val_metrics['macro_ovr_auroc']:.6f}",
        "",
        "## Final Test Metrics",
        "",
        f"- test_loss: {test_metrics['loss']:.6f}",
        f"- accuracy: {test_metrics['accuracy']:.6f}",
        f"- balanced_accuracy: {test_metrics['balanced_accuracy']:.6f}",
        f"- macro_f1: {test_metrics['macro_f1']:.6f}",
        f"- macro_ovr_auroc: {test_metrics['macro_ovr_auroc']:.6f}",
        "",
        "## Artifacts",
        "",
        f"- best_checkpoint: {paths.best_checkpoint_path}",
        f"- last_checkpoint: {paths.last_checkpoint_path}",
        f"- epoch_metrics_csv: {paths.epoch_metrics_csv_path}",
        f"- epoch_metrics_json: {paths.epoch_metrics_json_path}",
        f"- final_metrics: {paths.final_metrics_path}",
        f"- val_predictions: {paths.val_predictions_path}",
        f"- test_predictions: {paths.test_predictions_path}",
        f"- resolved_config: {paths.resolved_config_path}",
        "",
        "## Warnings",
        "",
    ]
    if runtime_label_map:
        lines.extend(
            [
                "",
                "## Runtime Label Map",
                "",
                f"- mapping: {json.dumps(runtime_label_map, sort_keys=True)}",
            ]
        )
    lines.extend([f"- {warning}" for warning in KAGGLE_RESEARCH_WARNINGS])
    paths.summary_report_path.write_text("\n".join(lines), encoding="utf-8")


def run_research_kaggle_training(
    config: ResearchKaggleTrainingConfig | None = None,
    *,
    settings: AppSettings | None = None,
) -> ResearchKaggleTrainingResult:
    """Run the config-driven Kaggle research training pipeline."""

    resolved_settings = settings or get_app_settings()
    cfg = _apply_dry_run_overrides(config or load_research_kaggle_training_config())
    set_global_seed(cfg.data.seed, deterministic=cfg.deterministic)
    paths = build_kaggle_run_paths(resolved_settings, cfg.run_name)
    dataloaders = _build_loaders(cfg, resolved_settings)
    _save_resolved_config(
        cfg=cfg,
        class_names=dataloaders.class_names,
        dataset_type=dataloaders.dataset_type,
        runtime_label_map=dataloaders.runtime_label_map,
        paths=paths,
    )

    torch = _load_torch_symbols()["torch"]
    device = _resolve_device(cfg.device, torch)
    amp_enabled = bool(cfg.mixed_precision and device.startswith("cuda"))

    model = build_kaggle_monai_network(
        KaggleMonaiModelConfig(
            dataset_type=dataloaders.dataset_type,
            in_channels=1,
            out_channels=len(dataloaders.class_names),
            dropout_prob=cfg.dropout_prob,
        )
    ).to(device)
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
    start_time = perf_counter()
    print(
        f"Starting Kaggle training run {cfg.run_name} on device={device} "
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
            class_names=dataloaders.class_names,
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
            class_names=dataloaders.class_names,
        )

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
                dataset_type=dataloaders.dataset_type,
                class_names=dataloaders.class_names,
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
            dataset_type=dataloaders.dataset_type,
            class_names=dataloaders.class_names,
        )
        if cfg.checkpoint.save_last:
            _save_checkpoint(paths.last_checkpoint_path, checkpoint_payload, torch)

        improvement_status = "improved" if improved else f"no_improve={epochs_without_improvement}"
        print(
            f"[{cfg.run_name}] epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_macro_ovr_auroc={val_metrics['macro_ovr_auroc']:.4f} "
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

    if not rows:
        raise KaggleResearchTrainingError("Kaggle training finished without producing epoch metrics.")

    best_checkpoint_path = (
        paths.best_checkpoint_path
        if paths.best_checkpoint_path.exists()
        else paths.last_checkpoint_path if paths.last_checkpoint_path.exists() else None
    )
    if best_checkpoint_path is None:
        raise KaggleResearchTrainingError("Kaggle training did not produce any checkpoint.")

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_evaluation = _evaluate_loader(
        split="val",
        loader=dataloaders.val_loader,
        model=model,
        loss_function=loss_function,
        torch=torch,
        device=device,
        amp_enabled=amp_enabled,
        max_batches=cfg.data.max_val_batches,
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
        max_batches=cfg.data.max_test_batches,
        class_names=dataloaders.class_names,
        dataset_type=dataloaders.dataset_type,
    )

    _write_predictions_csv(val_evaluation, paths=paths)
    _write_predictions_csv(test_evaluation, paths=paths)
    _write_confusion_matrix(val_evaluation, paths=paths, class_names=dataloaders.class_names)
    _write_confusion_matrix(test_evaluation, paths=paths, class_names=dataloaders.class_names)

    final_payload = _write_final_metrics(
        cfg=cfg,
        paths=paths,
        dataset_type=dataloaders.dataset_type,
        class_names=dataloaders.class_names,
        runtime_label_map=dataloaders.runtime_label_map,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor_value,
        val_evaluation=val_evaluation,
        test_evaluation=test_evaluation,
        stopped_early=stopped_early,
    )

    elapsed_seconds = perf_counter() - start_time
    _write_summary_report(
        cfg=cfg,
        paths=paths,
        rows=rows,
        dataset_type=dataloaders.dataset_type,
        class_names=dataloaders.class_names,
        runtime_label_map=dataloaders.runtime_label_map,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor_value,
        stopped_early=stopped_early,
        elapsed_seconds=elapsed_seconds,
        final_payload=final_payload,
    )

    return ResearchKaggleTrainingResult(
        run_name=cfg.run_name,
        run_root=paths.run_root,
        best_checkpoint_path=paths.best_checkpoint_path if paths.best_checkpoint_path.exists() else None,
        last_checkpoint_path=paths.last_checkpoint_path if paths.last_checkpoint_path.exists() else None,
        epoch_metrics_csv_path=paths.epoch_metrics_csv_path,
        epoch_metrics_json_path=paths.epoch_metrics_json_path,
        final_metrics_path=paths.final_metrics_path,
        val_confusion_matrix_path=paths.val_confusion_matrix_path,
        test_confusion_matrix_path=paths.test_confusion_matrix_path,
        val_predictions_path=paths.val_predictions_path,
        test_predictions_path=paths.test_predictions_path,
        summary_report_path=paths.summary_report_path,
        resolved_config_path=paths.resolved_config_path,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor_value,
        stopped_early=stopped_early,
        final_metrics=final_payload,
    )
