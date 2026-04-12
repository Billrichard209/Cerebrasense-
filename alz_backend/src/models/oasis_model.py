"""MONAI model definitions for OASIS-1 structural MRI classification."""

from __future__ import annotations

from dataclasses import dataclass

from .base_model import ModelMetadata, MonaiClassificationModelConfig, build_monai_densenet121
from .factory import OASIS_BINARY_CLASS_NAMES, OASISModelConfig, build_model


@dataclass(slots=True, frozen=True)
class OASISMonaiModelConfig(MonaiClassificationModelConfig):
    """Default MONAI model settings for OASIS-1."""

    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 2
    dropout_prob: float = 0.0


def build_oasis_class_names() -> tuple[str, str]:
    """Return the default binary OASIS class ordering."""

    return OASIS_BINARY_CLASS_NAMES


def build_oasis_model_metadata() -> ModelMetadata:
    """Describe the starter OASIS MONAI model family."""

    return ModelMetadata(
        name="oasis_monai_densenet121",
        modality="structural_mri_3d",
        framework="monai",
        tasks=("classification", "feature_extraction"),
    )


def build_oasis_monai_network(config: OASISMonaiModelConfig | None = None) -> object:
    """Build the default MONAI network for OASIS classification."""

    if config is None:
        return build_model()
    return build_monai_densenet121(config)


def build_oasis_baseline_model(config: OASISModelConfig | None = None) -> object:
    """Build the config-driven OASIS baseline model via the model factory."""

    return build_model(config)
