"""Reusable MONAI-oriented helpers for experiment naming and training bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.monai_utils import load_monai_inferer_symbols, load_torch_symbols

_load_monai_inferer_symbols = load_monai_inferer_symbols
_load_torch_symbols = load_torch_symbols


@dataclass(slots=True)
class TrainingArtifacts:
    """Paths that a training run is expected to populate."""

    checkpoint_path: Path
    metrics_path: Path
    report_path: Path


@dataclass(slots=True)
class MonaiTrainingComponents:
    """MONAI-aligned training bundle for classification workflows."""

    train_loader: object
    val_loader: object
    model: object
    loss_function: object
    optimizer: object
    inferer: object
    class_names: tuple[str, ...]
    framework: str = "monai"


def build_training_artifacts(run_name: str) -> TrainingArtifacts:
    """Construct standard output paths for a named experiment run."""

    return TrainingArtifacts(
        checkpoint_path=Path("outputs/checkpoints") / f"{run_name}.pt",
        metrics_path=Path("outputs/metrics") / f"{run_name}.json",
        report_path=Path("outputs/reports") / f"{run_name}.md",
    )


def build_monai_classification_loss() -> object:
    """Return the default classification loss for MONAI training loops."""

    nn = _load_torch_symbols()["nn"]
    return nn.CrossEntropyLoss()


def build_classification_loss(name: str = "cross_entropy", **kwargs: Any) -> object:
    """Build a configurable classification loss function."""

    nn = _load_torch_symbols()["nn"]
    normalized_name = name.strip().lower()
    if normalized_name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(**kwargs)
    if normalized_name in {"nll_loss", "nll"}:
        return nn.NLLLoss(**kwargs)
    raise ValueError(f"Unsupported classification loss: {name}")


def build_monai_adam_optimizer(
    model: object,
    *,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
) -> object:
    """Return the default optimizer for MONAI classification experiments."""

    optim = _load_torch_symbols()["optim"]
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_optimizer(
    model: object,
    *,
    name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    momentum: float = 0.9,
) -> object:
    """Build a configurable optimizer for research training."""

    optim = _load_torch_symbols()["optim"]
    normalized_name = name.strip().lower()
    if normalized_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if normalized_name == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if normalized_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: object,
    *,
    name: str = "none",
    step_size: int = 5,
    gamma: float = 0.5,
    patience: int = 2,
    factor: float = 0.5,
) -> object | None:
    """Build an optional learning-rate scheduler."""

    optim = _load_torch_symbols()["optim"]
    normalized_name = name.strip().lower()
    if normalized_name in {"none", "off", ""}:
        return None
    if normalized_name == "step_lr":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if normalized_name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor)
    if normalized_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(step_size, 1))
    raise ValueError(f"Unsupported scheduler: {name}")


def build_monai_simple_inferer() -> object:
    """Return MONAI's default simple inferer for classification."""

    inferer_cls = _load_monai_inferer_symbols()["SimpleInferer"]
    return inferer_cls()


def build_supervised_batch_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim manifest-derived records down to the fields that batch cleanly for training."""

    return [{"image": record["image"], "label": record["label"]} for record in records]
