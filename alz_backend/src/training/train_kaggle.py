"""MONAI-oriented training entry points for the separate Kaggle workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.configs.runtime import AppSettings
from src.data.base_dataset import build_monai_dataloader, build_monai_dataset
from src.data.kaggle_dataset import build_kaggle_monai_records, infer_kaggle_class_names, infer_kaggle_dataset_type, load_kaggle_manifest
from src.models.kaggle_model import KaggleMonaiModelConfig, build_kaggle_monai_network
from src.transforms.kaggle_transforms import (
    KaggleTransformConfig,
    build_kaggle_train_transforms,
    build_kaggle_val_transforms,
    load_kaggle_transform_config,
)

from .trainer_utils import (
    MonaiTrainingComponents,
    build_supervised_batch_records,
    build_monai_adam_optimizer,
    build_monai_classification_loss,
    build_monai_simple_inferer,
)


@dataclass(slots=True, frozen=True)
class KaggleTrainingConfig:
    """Default MONAI training settings for Kaggle experiments."""

    batch_size: int = 16
    num_workers: int = 0
    cache_rate: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    label_map: dict[str, int] | None = field(default=None)
    transform_config: KaggleTransformConfig = field(default_factory=load_kaggle_transform_config)


def build_kaggle_training_run_name(*, dataset_type: str = "2d_slices") -> str:
    """Return a deterministic MONAI-aligned run name for Kaggle experiments."""

    return f"kaggle_monai_densenet121_{dataset_type}"


def build_kaggle_monai_training_components(
    settings: AppSettings | None = None,
    *,
    config: KaggleTrainingConfig | None = None,
) -> MonaiTrainingComponents:
    """Build MONAI-aligned loaders, model, and optimization components for Kaggle."""

    resolved_settings = settings or AppSettings.from_env()
    resolved_config = config or KaggleTrainingConfig()
    manifest_frame = load_kaggle_manifest(resolved_settings, split="train")
    dataset_type = infer_kaggle_dataset_type(manifest_frame)
    class_names = infer_kaggle_class_names(manifest_frame, label_map=resolved_config.label_map)
    if resolved_config.label_map:
        out_channels = len(set(resolved_config.label_map.values()))
    else:
        out_channels = len(class_names)

    train_records = build_supervised_batch_records(
        build_kaggle_monai_records(
            resolved_settings,
            split="train",
            label_map=resolved_config.label_map,
            require_labels=True,
        )
    )
    val_records = build_supervised_batch_records(
        build_kaggle_monai_records(
            resolved_settings,
            split="val",
            label_map=resolved_config.label_map,
            require_labels=True,
        )
    )
    train_loader = build_monai_dataloader(
        build_monai_dataset(
            train_records,
            build_kaggle_train_transforms(resolved_config.transform_config, dataset_type=dataset_type),
            cache_rate=resolved_config.cache_rate,
            num_workers=resolved_config.num_workers,
        ),
        batch_size=resolved_config.batch_size,
        shuffle=True,
        num_workers=resolved_config.num_workers,
    )
    val_loader = build_monai_dataloader(
        build_monai_dataset(
            val_records,
            build_kaggle_val_transforms(resolved_config.transform_config, dataset_type=dataset_type),
            cache_rate=resolved_config.cache_rate,
            num_workers=resolved_config.num_workers,
        ),
        batch_size=resolved_config.batch_size,
        shuffle=False,
        num_workers=resolved_config.num_workers,
    )
    model = build_kaggle_monai_network(
        KaggleMonaiModelConfig(
            dataset_type=dataset_type,
            out_channels=max(out_channels, 1),
        )
    )
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
        class_names=class_names,
    )
