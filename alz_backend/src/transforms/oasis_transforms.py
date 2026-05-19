"""Config-driven MONAI preprocessing builders for OASIS-1 3D MRI classification.

These builders follow MONAI dictionary-transform conventions and keep train,
validation, and inference pipelines aligned:
- train uses the same core preprocessing as val/infer plus conservative MRI augmentations
- val is deterministic and mirrors train without randomness
- infer is deterministic and identical to val unless explicitly overridden

Notes on skull stripping:
- this module does not assume a dedicated brain mask is always available
- the optional skull-strip step is implemented as a conservative foreground crop
  around the brain signal, which is useful when scans are already masked or have
  stable empty-background margins
- if the dataset is already skull stripped, this step can be disabled safely
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import radians
from pathlib import Path
from typing import Any

import yaml

from src.utils.io_utils import resolve_project_root
from src.utils.monai_utils import load_monai_transform_symbols

from src.utils.monai_utils import load_monai_transform_symbols

_load_monai_transform_symbols = load_monai_transform_symbols

class ExtractClinicalFeaturesd:
    """Extract clinical features (Age, Sex, MMSE) from the meta dictionary into a tensor."""
    def __init__(self, keys=("meta",), output_key="clinical"):
        self.keys = keys
        self.output_key = output_key

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)
        meta = d.get(self.keys[0], {})
        
        # Safely handle missing metadata by providing defaults
        oasis_meta = meta.get("oasis2_metadata", {}) if isinstance(meta, dict) else {}
        
        # If meta is missing entirely or not a dict, we might still have keys in the top-level 
        # but the spec says it should be in oasis2_metadata.
        
        age = float(oasis_meta.get("age_at_visit") or 70.0) / 100.0
        sex_val = oasis_meta.get("sex")
        sex_str = str(sex_val).lower() if sex_val is not None else "f"
        sex = 1.0 if sex_str == "m" else 0.0
        mmse = float(oasis_meta.get("mmse") or 27.0) / 30.0
        
        # New: Genetic Risk (APOE-ε4 status: 0, 1, or 2 copies)
        apoe = float(oasis_meta.get("apoe") or 0.0) / 2.0
        
        import torch
        d[self.output_key] = torch.tensor([age, sex, mmse, apoe], dtype=torch.float32)
        
        # IMPORTANT: Remove the variable-length meta dictionary to avoid KeyError 
        # during batch collation when batch_size > 1.
        for key in self.keys:
            if key in d:
                del d[key]
                
        return d


@dataclass(slots=True, frozen=True)
class OASISLoadConfig:
    """I/O configuration for MONAI image loading."""

    keys: tuple[str, ...] = ("image",)
    reader: str = "NibabelReader"
    ensure_channel_first: bool = True


@dataclass(slots=True, frozen=True)
class OASISOrientationConfig:
    """Orientation normalization settings."""

    enabled: bool = True
    axcodes: str = "RAS"
    labels: tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = (
        ("L", "R"),
        ("P", "A"),
        ("I", "S"),
    )


@dataclass(slots=True, frozen=True)
class OASISSpacingConfig:
    """Voxel-spacing normalization settings."""

    enabled: bool = True
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mode: str = "bilinear"


@dataclass(slots=True, frozen=True)
class OASISIntensityConfig:
    """Intensity normalization settings."""

    percentile_lower: float = 1.0
    percentile_upper: float = 99.0
    output_min: float = 0.0
    output_max: float = 1.0
    clip: bool = True
    normalize_nonzero: bool = False


@dataclass(slots=True, frozen=True)
class OASISSkullStripConfig:
    """Optional foreground-based skull-strip proxy settings."""

    enabled: bool = True
    strategy: str = "crop_foreground"
    source_key: str = "image"
    margin: tuple[int, int, int] = (4, 4, 4)
    intensity_threshold: float = 0.0
    allow_smaller: bool = True


@dataclass(slots=True, frozen=True)
class OASISSpatialConfig:
    """Crop / pad / resize settings for model-ready volumes."""

    spatial_size: tuple[int, int, int] = (128, 128, 128)
    final_op: str = "resize_with_pad_or_crop"


@dataclass(slots=True, frozen=True)
class OASISAugmentationConfig:
    """Conservative MRI augmentation settings for training only."""

    enabled: bool = True
    affine_probability: float = 0.2
    rotate_range_degrees: tuple[float, float, float] = (5.0, 5.0, 5.0)
    scale_range: tuple[float, float, float] = (0.02, 0.02, 0.02)
    gaussian_noise_probability: float = 0.1
    gaussian_noise_std: float = 0.01
    bias_field_probability: float = 0.0
    bias_field_coeff_range: tuple[float, float] = (0.0, 0.1)
    gibbs_noise_probability: float = 0.0
    gibbs_alpha: tuple[float, float] = (0.0, 0.3)
    contrast_probability: float = 0.0
    contrast_gamma: tuple[float, float] = (0.8, 1.2)


@dataclass(slots=True, frozen=True)
class OASISTransformConfig:
    """Top-level config object for OASIS MONAI transform pipelines."""

    load: OASISLoadConfig = field(default_factory=OASISLoadConfig)
    orientation: OASISOrientationConfig = field(default_factory=OASISOrientationConfig)
    spacing: OASISSpacingConfig = field(default_factory=OASISSpacingConfig)
    intensity: OASISIntensityConfig = field(default_factory=OASISIntensityConfig)
    skull_strip: OASISSkullStripConfig = field(default_factory=OASISSkullStripConfig)
    spatial: OASISSpatialConfig = field(default_factory=OASISSpatialConfig)
    augmentation: OASISAugmentationConfig = field(default_factory=OASISAugmentationConfig)


def default_oasis_transform_config_path() -> Path:
    """Return the default YAML path for OASIS transform settings."""

    return resolve_project_root() / "configs" / "oasis_transforms.yaml"


def _as_tuple(values: Any, *, cast_type: type, expected_length: int) -> tuple[Any, ...]:
    """Normalize a config sequence into a fixed-length tuple."""

    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list or tuple with length {expected_length}, got {values!r}")
    if len(values) != expected_length:
        raise ValueError(f"Expected length {expected_length}, got {len(values)} for {values!r}")
    return tuple(cast_type(value) for value in values)


def _merge_dataclass_config(default_config: OASISTransformConfig, overrides: dict[str, Any]) -> OASISTransformConfig:
    """Merge YAML overrides into the strongly typed transform config."""

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

    skull_strip_section = dict(asdict(default_config.skull_strip))
    skull_strip_section.update(overrides.get("skull_strip", {}))
    if "margin" in skull_strip_section:
        skull_strip_section["margin"] = _as_tuple(skull_strip_section["margin"], cast_type=int, expected_length=3)

    spatial_section = dict(asdict(default_config.spatial))
    spatial_section.update(overrides.get("spatial", {}))
    if "spatial_size" in spatial_section:
        spatial_section["spatial_size"] = _as_tuple(spatial_section["spatial_size"], cast_type=int, expected_length=3)

    augmentation_section = dict(asdict(default_config.augmentation))
    augmentation_section.update(overrides.get("augmentation", {}))
    if "rotate_range_degrees" in augmentation_section:
        augmentation_section["rotate_range_degrees"] = _as_tuple(
            augmentation_section["rotate_range_degrees"],
            cast_type=float,
            expected_length=3,
        )
    if "scale_range" in augmentation_section:
        augmentation_section["scale_range"] = _as_tuple(
            augmentation_section["scale_range"],
            cast_type=float,
            expected_length=3,
        )
    for key in ("bias_field_coeff_range", "gibbs_alpha", "contrast_gamma"):
        if key in augmentation_section:
            augmentation_section[key] = _as_tuple(
                augmentation_section[key],
                cast_type=float,
                expected_length=2,
            )

    return OASISTransformConfig(
        load=OASISLoadConfig(**load_section),
        orientation=OASISOrientationConfig(**orientation_section),
        spacing=OASISSpacingConfig(**spacing_section),
        intensity=OASISIntensityConfig(**intensity_section),
        skull_strip=OASISSkullStripConfig(**skull_strip_section),
        spatial=OASISSpatialConfig(**spatial_section),
        augmentation=OASISAugmentationConfig(**augmentation_section),
    )


def load_oasis_transform_config(config_path: str | Path | None = None) -> OASISTransformConfig:
    """Load the OASIS transform YAML config into a typed object."""

    resolved_path = Path(config_path) if config_path is not None else default_oasis_transform_config_path()
    if not resolved_path.exists():
        raise FileNotFoundError(f"OASIS transform config not found: {resolved_path}")
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("OASIS transform config YAML must decode to a dictionary.")
    return _merge_dataclass_config(OASISTransformConfig(), payload)


class ForegroundThresholdSelector:
    """Pickle-safe foreground selector for MONAI multiprocessing workers."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, image: object) -> object:
        return image > self.threshold


def _foreground_select_fn(threshold: float) -> ForegroundThresholdSelector:
    """Build a MONAI-compatible foreground selector."""

    return ForegroundThresholdSelector(threshold)


def _build_common_oasis_steps(cfg: OASISTransformConfig) -> list[tuple[str, object]]:
    """Build the shared deterministic preprocessing steps for train/val/infer."""

    symbols = _load_monai_transform_symbols()
    steps: list[tuple[str, object]] = [
        (
            "load_image",
            symbols["LoadImaged"](
                keys=list(cfg.load.keys),
                reader=cfg.load.reader,
            ),
        ),
    ]
    if cfg.load.ensure_channel_first:
        steps.append(("ensure_channel_first", symbols["EnsureChannelFirstd"](keys=list(cfg.load.keys))))
    if cfg.orientation.enabled:
        steps.append(
            (
                "orientation_normalization",
                symbols["Orientationd"](
                    keys=list(cfg.load.keys),
                    axcodes=cfg.orientation.axcodes,
                    labels=cfg.orientation.labels,
                ),
            )
        )
    if cfg.spacing.enabled:
        steps.append(
            (
                "spacing_normalization",
                symbols["Spacingd"](
                    keys=list(cfg.load.keys),
                    pixdim=cfg.spacing.pixdim,
                    mode=cfg.spacing.mode,
                ),
            )
        )
    steps.append(
        (
            "intensity_scaling",
            symbols["ScaleIntensityRangePercentilesd"](
                keys=list(cfg.load.keys),
                lower=cfg.intensity.percentile_lower,
                upper=cfg.intensity.percentile_upper,
                b_min=cfg.intensity.output_min,
                b_max=cfg.intensity.output_max,
                clip=cfg.intensity.clip,
            ),
        )
    )
    if cfg.skull_strip.enabled:
        if cfg.skull_strip.strategy != "crop_foreground":
            raise ValueError(
                "OASIS skull_strip.strategy currently supports only `crop_foreground`."
            )
        steps.append(
            (
                "optional_skull_strip_crop",
                symbols["CropForegroundd"](
                    keys=list(cfg.load.keys),
                    source_key=cfg.skull_strip.source_key,
                    select_fn=_foreground_select_fn(cfg.skull_strip.intensity_threshold),
                    margin=cfg.skull_strip.margin,
                    allow_smaller=cfg.skull_strip.allow_smaller,
                ),
            )
        )
    if cfg.spatial.final_op != "resize_with_pad_or_crop":
        raise ValueError(
            f"Unsupported OASIS spatial.final_op: {cfg.spatial.final_op}. "
            "Use `resize_with_pad_or_crop`."
        )
    steps.append(
        (
            "resize_crop_pad",
            symbols["ResizeWithPadOrCropd"](
                keys=list(cfg.load.keys),
                spatial_size=cfg.spatial.spatial_size,
            ),
        )
    )
    steps.append(
        (
            "intensity_normalization",
            symbols["NormalizeIntensityd"](
                keys=list(cfg.load.keys),
                nonzero=cfg.intensity.normalize_nonzero,
            ),
        )
    )
    steps.append(("ensure_typed", symbols["EnsureTyped"](keys=list(cfg.load.keys))))
    steps.append(("extract_clinical", ExtractClinicalFeaturesd()))
    return steps


def _build_train_aug_steps(cfg: OASISTransformConfig) -> list[tuple[str, object]]:
    """Build conservative training-only MRI augmentations."""

    if not cfg.augmentation.enabled:
        return []

    symbols = _load_monai_transform_symbols()
    rotate_range_radians = tuple(radians(value) for value in cfg.augmentation.rotate_range_degrees)
    steps = [
        (
            "small_affine_augmentation",
            symbols["RandAffined"](
                keys=list(cfg.load.keys),
                prob=cfg.augmentation.affine_probability,
                rotate_range=rotate_range_radians,
                scale_range=cfg.augmentation.scale_range,
                padding_mode="border",
                mode="bilinear",
            ),
        ),
        (
            "gaussian_noise_augmentation",
            symbols["RandGaussianNoised"](
                keys=list(cfg.load.keys),
                prob=cfg.augmentation.gaussian_noise_probability,
                mean=0.0,
                std=cfg.augmentation.gaussian_noise_std,
            ),
        ),
    ]
    if cfg.augmentation.bias_field_probability > 0:
        steps.append(
            (
                "bias_field_augmentation",
                symbols["RandBiasFieldd"](
                    keys=list(cfg.load.keys),
                    prob=cfg.augmentation.bias_field_probability,
                    coeff_range=cfg.augmentation.bias_field_coeff_range,
                ),
            )
        )
    if cfg.augmentation.gibbs_noise_probability > 0:
        steps.append(
            (
                "gibbs_noise_augmentation",
                symbols["RandGibbsNoised"](
                    keys=list(cfg.load.keys),
                    prob=cfg.augmentation.gibbs_noise_probability,
                    alpha=cfg.augmentation.gibbs_alpha,
                ),
            )
        )
    if cfg.augmentation.contrast_probability > 0:
        steps.append(
            (
                "contrast_augmentation",
                symbols["RandAdjustContrastd"](
                    keys=list(cfg.load.keys),
                    prob=cfg.augmentation.contrast_probability,
                    gamma=cfg.augmentation.contrast_gamma,
                ),
            )
        )
    return steps


def describe_oasis_transform_pipeline(
    cfg: OASISTransformConfig | None = None,
    *,
    mode: str = "val",
) -> list[dict[str, str]]:
    """Return a human-readable description of the selected OASIS pipeline."""

    resolved_cfg = cfg or OASISTransformConfig()
    descriptions = {
        "load_image": "Load the MRI volume and convert it into a MONAI dictionary sample.",
        "ensure_channel_first": "Move the imaging channel to the first dimension for MONAI networks.",
        "orientation_normalization": "Normalize anatomical orientation so every volume follows the same axis convention.",
        "spacing_normalization": "Optionally resample voxels to a common spacing for shape consistency.",
        "intensity_scaling": "Robustly rescale intensities using percentiles to reduce scanner-specific range differences.",
        "optional_skull_strip_crop": "Optionally crop to foreground signal as a conservative skull-strip proxy when appropriate.",
        "resize_crop_pad": "Crop or pad the volume to the target model input size without unrealistic geometric distortion.",
        "intensity_normalization": "Normalize image intensities after spatial preprocessing for stable optimization.",
        "small_affine_augmentation": "Apply small spatial perturbations that are plausible for MRI classification training.",
        "gaussian_noise_augmentation": "Add mild Gaussian noise to improve robustness without over-augmenting anatomy.",
        "bias_field_augmentation": "Simulate MRI scanner field inhomogeneity during training.",
        "gibbs_noise_augmentation": "Simulate MRI ringing artifacts during training.",
        "contrast_augmentation": "Vary T1 contrast response to improve scanner robustness.",
        "ensure_typed": "Convert outputs to MONAI/Torch tensor-compatible types.",
    }
    common_steps = [name for name, _ in _build_common_oasis_steps(resolved_cfg)]
    if mode == "train":
        step_names = common_steps[:-1] + [name for name, _ in _build_train_aug_steps(resolved_cfg)] + [common_steps[-1]]
    elif mode in {"val", "infer"}:
        step_names = common_steps
    else:
        raise ValueError(f"Unsupported OASIS transform mode: {mode}")

    return [{"step": name, "why": descriptions[name]} for name in step_names]


def _build_compose(step_pairs: list[tuple[str, object]]) -> object:
    """Compose MONAI steps into one transform pipeline."""

    compose = _load_monai_transform_symbols()["Compose"]
    return compose([transform for _, transform in step_pairs])


def build_oasis_train_transforms(cfg: OASISTransformConfig) -> object:
    """Build the MONAI dictionary pipeline for OASIS training."""

    common_steps = _build_common_oasis_steps(cfg)
    train_aug_steps = _build_train_aug_steps(cfg)
    step_pairs = common_steps[:-1] + train_aug_steps + [common_steps[-1]]
    return _build_compose(step_pairs)


def build_oasis_val_transforms(cfg: OASISTransformConfig) -> object:
    """Build the deterministic MONAI dictionary pipeline for OASIS validation."""

    return _build_compose(_build_common_oasis_steps(cfg))


def build_oasis_infer_transforms(cfg: OASISTransformConfig) -> object:
    """Build the deterministic MONAI dictionary pipeline for OASIS inference."""

    return _build_compose(_build_common_oasis_steps(cfg))


def build_oasis_monai_transforms(
    *,
    training: bool = False,
    config: OASISTransformConfig | None = None,
) -> object:
    """Backward-compatible wrapper around the new train/val builders."""

    resolved_cfg = config or load_oasis_transform_config()
    if training:
        return build_oasis_train_transforms(resolved_cfg)
    return build_oasis_val_transforms(resolved_cfg)
