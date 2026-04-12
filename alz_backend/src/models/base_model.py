"""Shared MONAI-oriented model metadata and network builders."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.monai_utils import load_monai_network_symbols

_load_monai_network_symbols = load_monai_network_symbols


@dataclass(slots=True)
class ModelMetadata:
    """Lightweight metadata for tracking model family, framework, input type, and outputs."""

    name: str
    modality: str
    framework: str = "monai"
    tasks: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class MonaiClassificationModelConfig:
    """Generic MONAI classifier settings shared across datasets."""

    spatial_dims: int
    in_channels: int = 1
    out_channels: int = 2
    dropout_prob: float = 0.0


def build_monai_densenet121(config: MonaiClassificationModelConfig) -> object:
    """Build a MONAI DenseNet121 classifier for 2D or 3D medical imaging."""

    dense_net_cls = _load_monai_network_symbols()["DenseNet121"]
    return dense_net_cls(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dropout_prob=config.dropout_prob,
    )
