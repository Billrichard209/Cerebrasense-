"""MONAI-oriented training entry points for the primary OASIS-1 workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from src.configs.runtime import AppSettings
from src.data.base_dataset import build_monai_dataloader, build_monai_dataset
from src.data.oasis_dataset import build_oasis_monai_records
from src.models.oasis_model import OASISMonaiModelConfig, build_oasis_class_names, build_oasis_monai_network
from src.transforms.oasis_transforms import (
    OASISSpatialConfig,
    OASISTransformConfig,
    build_oasis_train_transforms,
    build_oasis_val_transforms,
)
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_torch_symbols

from .trainer_utils import (
    MonaiTrainingComponents,
    build_supervised_batch_records,
    build_training_artifacts,
    build_monai_adam_optimizer,
    build_monai_classification_loss,
    build_monai_simple_inferer,
)

_load_torch_symbols = load_torch_symbols


@dataclass(slots=True, frozen=True)
class OASISTrainingConfig:
    """Default MONAI training settings for OASIS."""

    batch_size: int = 1
    num_workers: int = 0
    cache_rate: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 1
    device: str = "auto"
    max_train_samples: int | None = 8
    max_val_samples: int | None = 4
    image_size: tuple[int, int, int] = (96, 96, 96)
    save_checkpoint: bool = True


@dataclass(slots=True)
class OASISRunResult:
    """Artifacts and metrics produced by a real OASIS MONAI training run."""

    run_name: str
    checkpoint_path: Path | None
    metrics_path: Path
    report_path: Path
    train_batches: int
    val_batches: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    device: str


def build_oasis_training_run_name() -> str:
    """Return a deterministic MONAI-aligned run name for OASIS experiments."""

    return "oasis_monai_densenet121"


def build_oasis_monai_training_components(
    settings: AppSettings | None = None,
    *,
    config: OASISTrainingConfig | None = None,
) -> MonaiTrainingComponents:
    """Build MONAI-aligned loaders, model, and optimization components for OASIS."""

    resolved_settings = settings or AppSettings.from_env()
    resolved_config = config or OASISTrainingConfig()
    transform_config = OASISTransformConfig(
        spatial=OASISSpatialConfig(spatial_size=resolved_config.image_size)
    )
    train_records = build_oasis_monai_records(resolved_settings, split="train")
    val_records = build_oasis_monai_records(resolved_settings, split="val")
    if resolved_config.max_train_samples is not None:
        train_records = train_records[: resolved_config.max_train_samples]
    if resolved_config.max_val_samples is not None:
        val_records = val_records[: resolved_config.max_val_samples]
    train_records = build_supervised_batch_records(train_records)
    val_records = build_supervised_batch_records(val_records)

    train_dataset = build_monai_dataset(
        train_records,
        build_oasis_train_transforms(transform_config),
        cache_rate=resolved_config.cache_rate,
        num_workers=resolved_config.num_workers,
    )
    val_dataset = build_monai_dataset(
        val_records,
        build_oasis_val_transforms(transform_config),
        cache_rate=resolved_config.cache_rate,
        num_workers=resolved_config.num_workers,
    )
    train_loader = build_monai_dataloader(
        train_dataset,
        batch_size=resolved_config.batch_size,
        shuffle=True,
        num_workers=resolved_config.num_workers,
    )
    val_loader = build_monai_dataloader(
        val_dataset,
        batch_size=resolved_config.batch_size,
        shuffle=False,
        num_workers=resolved_config.num_workers,
    )
    model = build_oasis_monai_network()
    return MonaiTrainingComponents(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_function=build_monai_classification_loss(),
        optimizer=build_monai_adam_optimizer(
            model,
            learning_rate=resolved_config.learning_rate,
            weight_decay=resolved_config.weight_decay,
        ),
        inferer=build_monai_simple_inferer(),
        class_names=build_oasis_class_names(),
    )


def _resolve_device(requested_device: str) -> str:
    """Resolve an execution device string."""

    if requested_device != "auto":
        return requested_device
    torch = _load_torch_symbols()["torch"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def _coerce_label_tensor(raw_labels: object, torch: object, device: str) -> object:
    """Convert dataloader labels to a torch long tensor."""

    if hasattr(raw_labels, "to"):
        return raw_labels.to(device).long()
    return torch.as_tensor(raw_labels, device=device).long()


def _run_epoch(
    *,
    loader: object,
    model: object,
    loss_function: object,
    inferer: object,
    torch: object,
    device: str,
    optimizer: object | None,
) -> tuple[float, float, int]:
    """Run one train or validation epoch."""

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batch_count = 0
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = _coerce_label_tensor(batch["label"], torch, device)

            if training:
                optimizer.zero_grad()

            logits = inferer(inputs=inputs, network=model)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            total_loss += float(loss.item())
            total_correct += int((predictions == labels).sum().item())
            total_examples += int(labels.numel())
            batch_count += 1

    average_loss = total_loss / max(batch_count, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy, batch_count


def run_oasis_monai_training_epoch(
    settings: AppSettings | None = None,
    *,
    config: OASISTrainingConfig | None = None,
    model_config: OASISMonaiModelConfig | None = None,
    run_name: str | None = None,
) -> OASISRunResult:
    """Run a real MONAI-based OASIS training job for the configured number of epochs."""

    resolved_settings = settings or AppSettings.from_env()
    resolved_config = config or OASISTrainingConfig()
    resolved_run_name = run_name or build_oasis_training_run_name()
    device = _resolve_device(resolved_config.device)
    torch = _load_torch_symbols()["torch"]

    components = build_oasis_monai_training_components(resolved_settings, config=resolved_config)
    model = build_oasis_monai_network(model_config).to(device)
    optimizer = build_monai_adam_optimizer(
        model,
        learning_rate=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )
    loss_function = build_monai_classification_loss()
    inferer = build_monai_simple_inferer()

    train_loss = 0.0
    val_loss = 0.0
    val_accuracy = 0.0
    train_batches = 0
    val_batches = 0
    start_time = perf_counter()

    for _ in range(resolved_config.epochs):
        train_loss, _, train_batches = _run_epoch(
            loader=components.train_loader,
            model=model,
            loss_function=loss_function,
            inferer=inferer,
            torch=torch,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy, val_batches = _run_epoch(
            loader=components.val_loader,
            model=model,
            loss_function=loss_function,
            inferer=inferer,
            torch=torch,
            device=device,
            optimizer=None,
        )

    artifacts = build_training_artifacts(resolved_run_name)
    checkpoint_path = resolved_settings.project_root / artifacts.checkpoint_path
    metrics_path = resolved_settings.project_root / artifacts.metrics_path
    report_path = resolved_settings.project_root / artifacts.report_path
    ensure_directory(checkpoint_path.parent)
    ensure_directory(metrics_path.parent)
    ensure_directory(report_path.parent)

    if resolved_config.save_checkpoint:
        torch.save(model.state_dict(), checkpoint_path)
        saved_checkpoint_path: Path | None = checkpoint_path
    else:
        saved_checkpoint_path = None

    elapsed_seconds = round(perf_counter() - start_time, 2)
    metrics_payload = {
        "run_name": resolved_run_name,
        "framework": "monai",
        "epochs": resolved_config.epochs,
        "device": device,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "class_names": list(build_oasis_class_names()),
        "config": asdict(resolved_config),
        "elapsed_seconds": elapsed_seconds,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                f"# {resolved_run_name}",
                "",
                f"- framework: monai",
                f"- device: {device}",
                f"- train_loss: {train_loss:.6f}",
                f"- val_loss: {val_loss:.6f}",
                f"- val_accuracy: {val_accuracy:.4f}",
                f"- elapsed_seconds: {elapsed_seconds}",
            ]
        ),
        encoding="utf-8",
    )

    return OASISRunResult(
        run_name=resolved_run_name,
        checkpoint_path=saved_checkpoint_path,
        metrics_path=metrics_path,
        report_path=report_path,
        train_batches=train_batches,
        val_batches=val_batches,
        train_loss=train_loss,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        device=device,
    )
