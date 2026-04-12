"""MONAI model definitions for the isolated Kaggle Alzheimer experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .base_model import ModelMetadata, MonaiClassificationModelConfig, build_monai_densenet121


@dataclass(slots=True, frozen=True)
class KaggleMonaiModelConfig:
    """Default MONAI model settings for Kaggle image or volume experiments."""

    dataset_type: str = "2d_slices"
    in_channels: int = 1
    out_channels: int = 4
    dropout_prob: float = 0.0

    @property
    def spatial_dims(self) -> int:
        """Map Kaggle dataset types to MONAI spatial dimensions."""

        if self.dataset_type == "2d_slices":
            return 2
        if self.dataset_type == "3d_volumes":
            return 3
        raise ValueError(f"Unsupported Kaggle dataset_type for MONAI network construction: {self.dataset_type}")


def build_kaggle_model_metadata(
    *,
    dataset_type: str = "2d_slices",
    out_channels: int = 4,
) -> ModelMetadata:
    """Describe the starter Kaggle MONAI model family while preserving dataset boundaries."""

    modality = "slice_imaging_2d" if dataset_type == "2d_slices" else "medical_volume_3d"
    return ModelMetadata(
        name=f"kaggle_monai_densenet121_{dataset_type}",
        modality=modality,
        framework="monai",
        tasks=("classification",),
    )


def build_kaggle_monai_network(config: KaggleMonaiModelConfig | None = None) -> object:
    """Build the default MONAI network for Kaggle classification experiments."""

    resolved_config = config or KaggleMonaiModelConfig()
    return build_monai_densenet121(
        MonaiClassificationModelConfig(
            spatial_dims=resolved_config.spatial_dims,
            in_channels=resolved_config.in_channels,
            out_channels=resolved_config.out_channels,
            dropout_prob=resolved_config.dropout_prob,
        )
    )
