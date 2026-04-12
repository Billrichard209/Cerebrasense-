"""Config-driven MONAI preprocessing builders for Kaggle Alzheimer datasets.

The Kaggle path is dataset-type aware:
- `2d_slices` uses 2D image resizing and conservative 2D augmentations
- `3d_volumes` uses MONAI-style 3D orientation/spacing normalization

This module keeps train/validation/inference aligned on the same core
preprocessing steps and only adds mild train-time augmentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import radians
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.utils.io_utils import resolve_project_root
from src.utils.monai_utils import load_monai_transform_symbols

_load_monai_transform_symbols = load_monai_transform_symbols


@dataclass(slots=True, frozen=True)
class KaggleLoadConfig:
    """I/O configuration for MONAI image loading."""

    keys: tuple[str, ...] = ("image",)
    reader_2d: str | None = None
    reader_3d: str = "NibabelReader"
    ensure_channel_first: bool = True


@dataclass(slots=True, frozen=True)
class KaggleOrientationConfig:
    """Orientation normalization settings for 3D data."""

    enabled_for_3d: bool = True
    axcodes: str = "RAS"
    labels: tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = (
        ("L", "R"),
        ("P", "A"),
        ("I", "S"),
    )


@dataclass(slots=True, frozen=True)
class KaggleSpacingConfig:
    """Voxel-spacing normalization settings for 3D inputs."""

    enabled_for_3d: bool = True
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mode: str = "bilinear"


@dataclass(slots=True, frozen=True)
class KaggleIntensityConfig:
    """Intensity normalization settings shared across Kaggle workflows."""

    percentile_lower: float = 1.0
    percentile_upper: float = 99.0
    output_min: float = 0.0
    output_max: float = 1.0
    clip: bool = True
    normalize_nonzero: bool = False


@dataclass(slots=True, frozen=True)
class KaggleChannelConfig:
    """Standardize mixed JPG exports into one MRI-like intensity channel."""

    force_single_channel: bool = True
    reduction: str = "mean"


@dataclass(slots=True, frozen=True)
class KaggleForegroundConfig:
    """Optional foreground cropping for already-centered slices or volumes."""

    enabled: bool = False
    source_key: str = "image"
    margin_2d: tuple[int, int] = (4, 4)
    margin_3d: tuple[int, int, int] = (4, 4, 4)
    intensity_threshold: float = 0.0
    allow_smaller: bool = True


@dataclass(slots=True, frozen=True)
class KaggleSpatialConfig:
    """Spatial output settings for 2D and 3D Kaggle data."""

    image_size_2d: tuple[int, int] = (224, 224)
    image_size_3d: tuple[int, int, int] = (128, 128, 128)
    final_op_2d: str = "resize"
    final_op_3d: str = "resize_with_pad_or_crop"


@dataclass(slots=True, frozen=True)
class KaggleAugmentationConfig:
    """Conservative train-time augmentation settings."""

    enabled: bool = True
    affine_probability_2d: float = 0.2
    affine_probability_3d: float = 0.15
    rotate_range_degrees_2d: float = 4.0
    rotate_range_degrees_3d: tuple[float, float, float] = (3.0, 3.0, 3.0)
    scale_range_2d: tuple[float, float] = (0.02, 0.02)
    scale_range_3d: tuple[float, float, float] = (0.02, 0.02, 0.02)
    gaussian_noise_probability: float = 0.05
    gaussian_noise_std: float = 0.01
    horizontal_flip_probability_2d: float = 0.0
    vertical_flip_probability_2d: float = 0.0


@dataclass(slots=True, frozen=True)
class KaggleTransformConfig:
    """Top-level config object for Kaggle MONAI transform pipelines."""

    load: KaggleLoadConfig = field(default_factory=KaggleLoadConfig)
    orientation: KaggleOrientationConfig = field(default_factory=KaggleOrientationConfig)
    spacing: KaggleSpacingConfig = field(default_factory=KaggleSpacingConfig)
    intensity: KaggleIntensityConfig = field(default_factory=KaggleIntensityConfig)
    channel: KaggleChannelConfig = field(default_factory=KaggleChannelConfig)
    foreground: KaggleForegroundConfig = field(default_factory=KaggleForegroundConfig)
    spatial: KaggleSpatialConfig = field(default_factory=KaggleSpatialConfig)
    augmentation: KaggleAugmentationConfig = field(default_factory=KaggleAugmentationConfig)


def default_kaggle_transform_config_path() -> Path:
    """Return the default YAML path for Kaggle transform settings."""

    return resolve_project_root() / "configs" / "kaggle_transforms.yaml"


def _as_tuple(values: Any, *, cast_type: type, expected_length: int) -> tuple[Any, ...]:
    """Normalize a config sequence into a fixed-length tuple."""

    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list or tuple with length {expected_length}, got {values!r}")
    if len(values) != expected_length:
        raise ValueError(f"Expected length {expected_length}, got {len(values)} for {values!r}")
    return tuple(cast_type(value) for value in values)


def _merge_dataclass_config(default_config: KaggleTransformConfig, overrides: dict[str, Any]) -> KaggleTransformConfig:
    """Merge YAML overrides into the strongly typed Kaggle transform config."""

    if not overrides:
        return default_config

    load_section = dict(asdict(default_config.load))
    load_section.update(overrides.get("load", {}))

    orientation_section = dict(asdict(default_config.orientation))
    orientation_section.update(overrides.get("orientation", {}))
    if "labels" in orientation_section:
        orientation_section["labels"] = tuple(tuple(pair) for pair in orientation_section["labels"])

    spacing_section = dict(asdict(default_config.spacing))
    spacing_section.update(overrides.get("spacing", {}))
    if "pixdim" in spacing_section:
        spacing_section["pixdim"] = _as_tuple(spacing_section["pixdim"], cast_type=float, expected_length=3)

    intensity_section = dict(asdict(default_config.intensity))
    intensity_section.update(overrides.get("intensity", {}))

    channel_section = dict(asdict(default_config.channel))
    channel_section.update(overrides.get("channel", {}))

    foreground_section = dict(asdict(default_config.foreground))
    foreground_section.update(overrides.get("foreground", {}))
    if "margin_2d" in foreground_section:
        foreground_section["margin_2d"] = _as_tuple(foreground_section["margin_2d"], cast_type=int, expected_length=2)
    if "margin_3d" in foreground_section:
        foreground_section["margin_3d"] = _as_tuple(foreground_section["margin_3d"], cast_type=int, expected_length=3)

    spatial_section = dict(asdict(default_config.spatial))
    spatial_section.update(overrides.get("spatial", {}))
    if "image_size_2d" in spatial_section:
        spatial_section["image_size_2d"] = _as_tuple(spatial_section["image_size_2d"], cast_type=int, expected_length=2)
    if "image_size_3d" in spatial_section:
        spatial_section["image_size_3d"] = _as_tuple(spatial_section["image_size_3d"], cast_type=int, expected_length=3)

    augmentation_section = dict(asdict(default_config.augmentation))
    augmentation_section.update(overrides.get("augmentation", {}))
    if "rotate_range_degrees_3d" in augmentation_section:
        augmentation_section["rotate_range_degrees_3d"] = _as_tuple(
            augmentation_section["rotate_range_degrees_3d"],
            cast_type=float,
            expected_length=3,
        )
    if "scale_range_2d" in augmentation_section:
        augmentation_section["scale_range_2d"] = _as_tuple(
            augmentation_section["scale_range_2d"],
            cast_type=float,
            expected_length=2,
        )
    if "scale_range_3d" in augmentation_section:
        augmentation_section["scale_range_3d"] = _as_tuple(
            augmentation_section["scale_range_3d"],
            cast_type=float,
            expected_length=3,
        )

    return KaggleTransformConfig(
        load=KaggleLoadConfig(**load_section),
        orientation=KaggleOrientationConfig(**orientation_section),
        spacing=KaggleSpacingConfig(**spacing_section),
        intensity=KaggleIntensityConfig(**intensity_section),
        channel=KaggleChannelConfig(**channel_section),
        foreground=KaggleForegroundConfig(**foreground_section),
        spatial=KaggleSpatialConfig(**spatial_section),
        augmentation=KaggleAugmentationConfig(**augmentation_section),
    )


def load_kaggle_transform_config(config_path: str | Path | None = None) -> KaggleTransformConfig:
    """Load the Kaggle transform YAML config into a typed object."""

    resolved_path = Path(config_path) if config_path is not None else default_kaggle_transform_config_path()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Kaggle transform config not found: {resolved_path}")
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Kaggle transform config YAML must decode to a dictionary.")
    return _merge_dataclass_config(KaggleTransformConfig(), payload)


def _foreground_select_fn(threshold: float):
    """Build a MONAI-compatible foreground selector."""

    return lambda image: image > threshold


def _standardize_single_channel(image: Any, reduction: str = "mean") -> Any:
    """Collapse mixed grayscale/RGB exports to one channel after channel-first loading."""

    if hasattr(image, "ndim"):
        ndim = int(image.ndim)
    else:
        image = np.asarray(image)
        ndim = int(image.ndim)

    if ndim < 3:
        return image

    channel_count = int(image.shape[0])
    if channel_count <= 1:
        return image
    if reduction != "mean":
        raise ValueError(f"Unsupported Kaggle channel reduction mode: {reduction}")
    return image.mean(axis=0, keepdims=True)


def _load_transform(keys: list[str], reader: str | None) -> object:
    """Create the MONAI dictionary loader with an optional explicit reader."""

    symbols = _load_monai_transform_symbols()
    if reader is None:
        return symbols["LoadImaged"](keys=keys)
    return symbols["LoadImaged"](keys=keys, reader=reader)


def _build_common_kaggle_steps(cfg: KaggleTransformConfig, *, dataset_type: str) -> list[tuple[str, object]]:
    """Build the shared deterministic preprocessing steps for train/val/infer."""

    if dataset_type not in {"2d_slices", "3d_volumes"}:
        raise ValueError(
            f"Kaggle transform builders only support `2d_slices` or `3d_volumes`, got {dataset_type}."
        )

    symbols = _load_monai_transform_symbols()
    keys = list(cfg.load.keys)
    reader = cfg.load.reader_3d if dataset_type == "3d_volumes" else cfg.load.reader_2d
    steps: list[tuple[str, object]] = [("load_image", _load_transform(keys, reader))]
    if cfg.load.ensure_channel_first:
        steps.append(("ensure_channel_first", symbols["EnsureChannelFirstd"](keys=keys)))
    if cfg.channel.force_single_channel:
        steps.append(
            (
                "single_channel_standardization",
                symbols["Lambdad"](
                    keys=keys,
                    func=lambda image, reduction=cfg.channel.reduction: _standardize_single_channel(
                        image,
                        reduction=reduction,
                    ),
                ),
            )
        )
    if dataset_type == "3d_volumes" and cfg.orientation.enabled_for_3d:
        steps.append(
            (
                "orientation_normalization",
                symbols["Orientationd"](
                    keys=keys,
                    axcodes=cfg.orientation.axcodes,
                    labels=cfg.orientation.labels,
                ),
            )
        )
    if dataset_type == "3d_volumes" and cfg.spacing.enabled_for_3d:
        steps.append(
            (
                "spacing_normalization",
                symbols["Spacingd"](
                    keys=keys,
                    pixdim=cfg.spacing.pixdim,
                    mode=cfg.spacing.mode,
                ),
            )
        )
    steps.append(
        (
            "intensity_scaling",
            symbols["ScaleIntensityRangePercentilesd"](
                keys=keys,
                lower=cfg.intensity.percentile_lower,
                upper=cfg.intensity.percentile_upper,
                b_min=cfg.intensity.output_min,
                b_max=cfg.intensity.output_max,
                clip=cfg.intensity.clip,
            ),
        )
    )
    if cfg.foreground.enabled:
        margin = cfg.foreground.margin_2d if dataset_type == "2d_slices" else cfg.foreground.margin_3d
        steps.append(
            (
                "optional_foreground_crop",
                symbols["CropForegroundd"](
                    keys=keys,
                    source_key=cfg.foreground.source_key,
                    select_fn=_foreground_select_fn(cfg.foreground.intensity_threshold),
                    margin=margin,
                    allow_smaller=cfg.foreground.allow_smaller,
                ),
            )
        )
    if dataset_type == "2d_slices":
        if cfg.spatial.final_op_2d != "resize":
            raise ValueError(f"Unsupported Kaggle spatial.final_op_2d: {cfg.spatial.final_op_2d}")
        steps.append(
            (
                "resize_crop_pad",
                symbols["Resized"](keys=keys, spatial_size=cfg.spatial.image_size_2d),
            )
        )
    else:
        if cfg.spatial.final_op_3d != "resize_with_pad_or_crop":
            raise ValueError(f"Unsupported Kaggle spatial.final_op_3d: {cfg.spatial.final_op_3d}")
        steps.append(
            (
                "resize_crop_pad",
                symbols["ResizeWithPadOrCropd"](keys=keys, spatial_size=cfg.spatial.image_size_3d),
            )
        )
    steps.append(
        (
            "intensity_normalization",
            symbols["NormalizeIntensityd"](keys=keys, nonzero=cfg.intensity.normalize_nonzero),
        )
    )
    steps.append(("ensure_typed", symbols["EnsureTyped"](keys=keys)))
    return steps


def _build_train_aug_steps(cfg: KaggleTransformConfig, *, dataset_type: str) -> list[tuple[str, object]]:
    """Build conservative train-only augmentation steps."""

    if not cfg.augmentation.enabled:
        return []

    symbols = _load_monai_transform_symbols()
    keys = list(cfg.load.keys)
    if dataset_type == "2d_slices":
        rotate_range = radians(cfg.augmentation.rotate_range_degrees_2d)
        steps: list[tuple[str, object]] = [
            (
                "small_affine_augmentation",
                symbols["RandAffined"](
                    keys=keys,
                    prob=cfg.augmentation.affine_probability_2d,
                    rotate_range=rotate_range,
                    scale_range=cfg.augmentation.scale_range_2d,
                    padding_mode="border",
                    mode="bilinear",
                ),
            )
        ]
        if cfg.augmentation.horizontal_flip_probability_2d > 0:
            steps.append(
                (
                    "horizontal_flip_augmentation",
                    symbols["RandFlipd"](
                        keys=keys,
                        prob=cfg.augmentation.horizontal_flip_probability_2d,
                        spatial_axis=0,
                    ),
                )
            )
        if cfg.augmentation.vertical_flip_probability_2d > 0:
            steps.append(
                (
                    "vertical_flip_augmentation",
                    symbols["RandFlipd"](
                        keys=keys,
                        prob=cfg.augmentation.vertical_flip_probability_2d,
                        spatial_axis=1,
                    ),
                )
            )
    else:
        rotate_range = tuple(radians(value) for value in cfg.augmentation.rotate_range_degrees_3d)
        steps = [
            (
                "small_affine_augmentation",
                symbols["RandAffined"](
                    keys=keys,
                    prob=cfg.augmentation.affine_probability_3d,
                    rotate_range=rotate_range,
                    scale_range=cfg.augmentation.scale_range_3d,
                    padding_mode="border",
                    mode="bilinear",
                ),
            )
        ]
    steps.append(
        (
            "gaussian_noise_augmentation",
            symbols["RandGaussianNoised"](
                keys=keys,
                prob=cfg.augmentation.gaussian_noise_probability,
                mean=0.0,
                std=cfg.augmentation.gaussian_noise_std,
            ),
        )
    )
    return steps


def describe_kaggle_transform_pipeline(
    cfg: KaggleTransformConfig | None = None,
    *,
    dataset_type: str,
    mode: str = "val",
) -> list[dict[str, str]]:
    """Return a human-readable description of the selected Kaggle pipeline."""

    resolved_cfg = cfg or KaggleTransformConfig()
    descriptions = {
        "load_image": "Load the image or volume into a MONAI dictionary sample.",
        "ensure_channel_first": "Move the imaging channel to the first dimension for MONAI networks.",
        "single_channel_standardization": "Collapse repeated RGB exports to one intensity channel so MONAI models receive consistent MRI-like inputs.",
        "orientation_normalization": "Normalize anatomical orientation for 3D medical volumes.",
        "spacing_normalization": "Optionally resample 3D volumes to a common voxel spacing.",
        "intensity_scaling": "Rescale intensities robustly with percentiles to stabilize scanner and export differences.",
        "optional_foreground_crop": "Optionally crop around foreground signal when backgrounds are stable and non-informative.",
        "resize_crop_pad": "Resize or crop/pad the sample to the target model input size.",
        "intensity_normalization": "Normalize intensities after spatial preprocessing for stable optimization.",
        "small_affine_augmentation": "Apply mild affine variation that preserves medically meaningful structure.",
        "gaussian_noise_augmentation": "Add low-amplitude Gaussian noise for modest robustness gains.",
        "horizontal_flip_augmentation": "Optional horizontal flip, disabled by default because laterality may matter.",
        "vertical_flip_augmentation": "Optional vertical flip, disabled by default because anatomy orientation may matter.",
        "ensure_typed": "Convert outputs to MONAI/Torch tensor-compatible types.",
    }
    common_steps = [name for name, _ in _build_common_kaggle_steps(resolved_cfg, dataset_type=dataset_type)]
    if mode == "train":
        step_names = common_steps[:-1] + [name for name, _ in _build_train_aug_steps(resolved_cfg, dataset_type=dataset_type)] + [common_steps[-1]]
    elif mode in {"val", "infer"}:
        step_names = common_steps
    else:
        raise ValueError(f"Unsupported Kaggle transform mode: {mode}")
    return [{"step": name, "why": descriptions[name]} for name in step_names]


def _build_compose(step_pairs: list[tuple[str, object]]) -> object:
    """Compose MONAI steps into one transform pipeline."""

    compose = _load_monai_transform_symbols()["Compose"]
    return compose([transform for _, transform in step_pairs])


def build_kaggle_train_transforms(cfg: KaggleTransformConfig, *, dataset_type: str) -> object:
    """Build the MONAI dictionary pipeline for Kaggle training."""

    common_steps = _build_common_kaggle_steps(cfg, dataset_type=dataset_type)
    train_aug_steps = _build_train_aug_steps(cfg, dataset_type=dataset_type)
    step_pairs = common_steps[:-1] + train_aug_steps + [common_steps[-1]]
    return _build_compose(step_pairs)


def build_kaggle_val_transforms(cfg: KaggleTransformConfig, *, dataset_type: str) -> object:
    """Build the deterministic MONAI dictionary pipeline for Kaggle validation."""

    return _build_compose(_build_common_kaggle_steps(cfg, dataset_type=dataset_type))


def build_kaggle_infer_transforms(cfg: KaggleTransformConfig, *, dataset_type: str) -> object:
    """Build the deterministic MONAI dictionary pipeline for Kaggle inference."""

    return _build_compose(_build_common_kaggle_steps(cfg, dataset_type=dataset_type))


def build_kaggle_monai_transforms(
    *,
    dataset_type: str,
    training: bool = False,
    config: KaggleTransformConfig | None = None,
) -> object:
    """Backward-compatible wrapper around the explicit Kaggle builders."""

    resolved_cfg = config or load_kaggle_transform_config()
    if training:
        return build_kaggle_train_transforms(resolved_cfg, dataset_type=dataset_type)
    return build_kaggle_val_transforms(resolved_cfg, dataset_type=dataset_type)
