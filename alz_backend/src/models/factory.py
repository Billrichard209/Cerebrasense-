"""Config-driven model factory for OASIS-1 3D MRI classification.

Expected OASIS tensor shape:
- MONAI dictionary transforms should produce image tensors as ``(C, D, H, W)`` per sample.
- MONAI/PyTorch dataloaders batch those samples into ``(B, C, D, H, W)``.
- The default OASIS preprocessing config currently targets ``(B, 1, 128, 128, 128)``.

The first supported baseline is intentionally simple and reliable:
- MONAI DenseNet121
- 3D spatial dimensions
- binary output classes ordered as ``("nondemented", "demented")``
- optional penultimate embedding capture via a forward hook

Future architectures such as MONAI ResNet or a custom 3D CNN should be added
behind the same ``build_model(cfg)`` entrypoint without changing callers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.io_utils import resolve_project_root
from src.utils.monai_utils import load_monai_network_symbols, load_torch_symbols

_load_monai_network_symbols = load_monai_network_symbols
_load_torch_symbols = load_torch_symbols

OASIS_BINARY_CLASS_NAMES = ("nondemented", "demented")
SUPPORTED_MODEL_ARCHITECTURES = ("densenet121_3d",)


class ModelFactoryError(ValueError):
    """Raised when a model config cannot be built safely."""


@dataclass(slots=True, frozen=True)
class DenseNet3DConfig:
    """DenseNet-specific settings for the OASIS-1 3D baseline."""

    name: str = "densenet121"
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 2
    dropout_prob: float = 0.0


@dataclass(slots=True, frozen=True)
class EmbeddingConfig:
    """Optional embedding extraction settings."""

    enabled: bool = False
    hook_module_name: str = "class_layers.flatten"


@dataclass(slots=True, frozen=True)
class OASISModelConfig:
    """Top-level model factory config for OASIS-1 binary 3D MRI classification."""

    dataset: str = "oasis1"
    task: str = "binary_3d_mri_classification"
    architecture: str = "densenet121_3d"
    class_names: tuple[str, ...] = OASIS_BINARY_CLASS_NAMES
    expected_input_shape: tuple[int, int, int, int, int] = (1, 1, 128, 128, 128)
    densenet: DenseNet3DConfig = field(default_factory=DenseNet3DConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)


class EmbeddingEnabledModel:
    """Thin wrapper that returns logits by default and embeddings when requested."""

    def __init__(self, backbone: object, *, hook_module_name: str) -> None:
        torch = _load_torch_symbols()["torch"]
        self._module_cls = torch.nn.Module
        if not isinstance(backbone, self._module_cls):
            raise TypeError("EmbeddingEnabledModel expects a torch.nn.Module backbone.")

        class _Wrapper(torch.nn.Module):
            def __init__(self, resolved_backbone: object, resolved_hook_name: str) -> None:
                super().__init__()
                self.backbone = resolved_backbone
                self.hook_module_name = resolved_hook_name
                self.last_embedding = None
                self._hook_handle = self._resolve_hook_module().register_forward_hook(self._capture_embedding)

            def _resolve_hook_module(self) -> object:
                modules = dict(self.backbone.named_modules())
                if self.hook_module_name not in modules:
                    available = sorted(name for name in modules if name)
                    raise ModelFactoryError(
                        f"Embedding hook module {self.hook_module_name!r} was not found. "
                        f"Available module examples: {available[:20]}"
                    )
                return modules[self.hook_module_name]

            def _capture_embedding(self, _module: object, _inputs: tuple[object, ...], output: object) -> None:
                self.last_embedding = output

            def forward(self, image: object, *, return_embeddings: bool = False) -> object:
                logits = self.backbone(image)
                if return_embeddings:
                    if self.last_embedding is None:
                        raise ModelFactoryError("Embedding hook did not capture a representation.")
                    return logits, self.last_embedding
                return logits

        self.module = _Wrapper(backbone, hook_module_name)

    def __getattr__(self, name: str) -> object:
        """Delegate torch module behavior to the wrapped module."""

        return getattr(self.module, name)


def default_oasis_model_config_path() -> Path:
    """Return the default OASIS model YAML config path."""

    return resolve_project_root() / "configs" / "oasis_model.yaml"


def _as_tuple(values: Any, *, cast_type: type, expected_length: int) -> tuple[Any, ...]:
    """Normalize YAML sequences into fixed-length tuples."""

    if not isinstance(values, (list, tuple)):
        raise ModelFactoryError(f"Expected a sequence with length {expected_length}, got {values!r}")
    if len(values) != expected_length:
        raise ModelFactoryError(f"Expected length {expected_length}, got {len(values)} for {values!r}")
    return tuple(cast_type(value) for value in values)


def _merge_oasis_model_config(default_config: OASISModelConfig, overrides: dict[str, Any]) -> OASISModelConfig:
    """Merge YAML overrides into the typed model config."""

    if not overrides:
        return default_config

    densenet_section = dict(asdict(default_config.densenet))
    densenet_section.update(overrides.get("densenet", {}))

    embedding_section = dict(asdict(default_config.embeddings))
    embedding_section.update(overrides.get("embeddings", {}))

    class_names = overrides.get("class_names", default_config.class_names)
    expected_input_shape = overrides.get("expected_input_shape", default_config.expected_input_shape)

    return OASISModelConfig(
        dataset=str(overrides.get("dataset", default_config.dataset)),
        task=str(overrides.get("task", default_config.task)),
        architecture=str(overrides.get("architecture", default_config.architecture)),
        class_names=tuple(str(value) for value in class_names),
        expected_input_shape=_as_tuple(expected_input_shape, cast_type=int, expected_length=5),
        densenet=DenseNet3DConfig(**densenet_section),
        embeddings=EmbeddingConfig(**embedding_section),
    )


def load_oasis_model_config(config_path: str | Path | None = None) -> OASISModelConfig:
    """Load the OASIS model factory config from YAML."""

    resolved_path = Path(config_path) if config_path is not None else default_oasis_model_config_path()
    if not resolved_path.exists():
        raise FileNotFoundError(f"OASIS model config not found: {resolved_path}")
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ModelFactoryError("OASIS model config YAML must decode to a dictionary.")
    return _merge_oasis_model_config(OASISModelConfig(), payload)


def _validate_oasis_model_config(cfg: OASISModelConfig) -> None:
    """Validate assumptions that must hold for the first OASIS baseline."""

    if cfg.dataset != "oasis1":
        raise ModelFactoryError(f"This factory currently supports only dataset='oasis1', got {cfg.dataset!r}")
    if cfg.task != "binary_3d_mri_classification":
        raise ModelFactoryError(f"Unsupported OASIS model task: {cfg.task!r}")
    if cfg.architecture not in SUPPORTED_MODEL_ARCHITECTURES:
        raise ModelFactoryError(
            f"Unsupported model architecture {cfg.architecture!r}. "
            "Extension point is available for ResNet3D or custom 3D CNNs later."
        )
    if cfg.densenet.spatial_dims != 3:
        raise ModelFactoryError("OASIS DenseNet baseline must use spatial_dims=3.")
    if cfg.densenet.out_channels != len(cfg.class_names):
        raise ModelFactoryError(
            f"out_channels={cfg.densenet.out_channels} must match class_names={len(cfg.class_names)}."
        )
    if cfg.expected_input_shape[1] != cfg.densenet.in_channels:
        raise ModelFactoryError(
            "expected_input_shape channel dimension must match densenet.in_channels. "
            f"Got shape={cfg.expected_input_shape}, in_channels={cfg.densenet.in_channels}."
        )


def build_densenet3d(cfg: OASISModelConfig | DenseNet3DConfig | None = None) -> object:
    """Build the default MONAI DenseNet-based 3D OASIS classifier."""

    if cfg is None:
        dense_cfg = OASISModelConfig().densenet
    elif isinstance(cfg, OASISModelConfig):
        _validate_oasis_model_config(cfg)
        dense_cfg = cfg.densenet
    else:
        dense_cfg = cfg

    if dense_cfg.spatial_dims != 3:
        raise ModelFactoryError("build_densenet3d requires spatial_dims=3.")

    dense_net_cls = _load_monai_network_symbols()["DenseNet121"]
    return dense_net_cls(
        spatial_dims=dense_cfg.spatial_dims,
        in_channels=dense_cfg.in_channels,
        out_channels=dense_cfg.out_channels,
        dropout_prob=dense_cfg.dropout_prob,
    )


def with_embedding_output(model: object, *, hook_module_name: str = "class_layers.flatten") -> object:
    """Wrap a torch model so callers can request penultimate embeddings."""

    return EmbeddingEnabledModel(model, hook_module_name=hook_module_name).module


def build_model(cfg: OASISModelConfig | None = None) -> object:
    """Build an OASIS-1 baseline model from the factory config."""

    resolved_cfg = cfg or load_oasis_model_config()
    _validate_oasis_model_config(resolved_cfg)

    if resolved_cfg.architecture == "densenet121_3d":
        model = build_densenet3d(resolved_cfg)
    else:
        raise ModelFactoryError(f"Unsupported OASIS model architecture: {resolved_cfg.architecture}")

    if resolved_cfg.embeddings.enabled:
        model = with_embedding_output(
            model,
            hook_module_name=resolved_cfg.embeddings.hook_module_name,
        )
    return model


def describe_model_config(cfg: OASISModelConfig | None = None) -> dict[str, Any]:
    """Return a JSON-safe description of the active OASIS model config."""

    resolved_cfg = cfg or load_oasis_model_config()
    _validate_oasis_model_config(resolved_cfg)
    return {
        "dataset": resolved_cfg.dataset,
        "task": resolved_cfg.task,
        "architecture": resolved_cfg.architecture,
        "framework": "monai",
        "class_names": list(resolved_cfg.class_names),
        "expected_input_shape": list(resolved_cfg.expected_input_shape),
        "input_shape_note": "Batched OASIS tensors should be shaped (B, C, D, H, W).",
        "densenet": asdict(resolved_cfg.densenet),
        "embeddings": asdict(resolved_cfg.embeddings),
        "extension_notes": [
            "Add ResNet3D or custom 3D CNN support behind build_model(cfg) by extending architecture dispatch.",
            "Keep OASIS input tensors aligned with MONAI dictionary transforms and 3D spatial dimensions.",
            "Do not mix Kaggle and OASIS model configs unless an explicit dataset harmonization plan exists.",
        ],
    }
