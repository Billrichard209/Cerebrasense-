"""Serving-time configuration and cached model loading helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.configs.runtime import AppSettings, get_app_settings
from src.evaluation.calibration import ConfidenceBandConfig
from src.evaluation.oasis_run import LoadedCheckpoint, load_oasis_checkpoint
from src.models.factory import OASISModelConfig, build_model, load_oasis_model_config
from src.models.registry import ModelRegistryEntry, load_current_oasis_model_entry
from src.utils.io_utils import resolve_project_root


@dataclass(slots=True, frozen=True)
class ThresholdPolicyConfig:
    """Serving-time threshold policy."""

    use_registry_recommended_threshold: bool = True
    fallback_threshold: float = 0.5


@dataclass(slots=True, frozen=True)
class ScanValidationPolicy:
    """Serving-time scan-validation policy."""

    max_file_size_mb: int = 512
    allowed_suffixes: tuple[str, ...] = (".hdr", ".img", ".nii", ".nii.gz")
    enforce_workspace_boundary: bool = False


@dataclass(slots=True, frozen=True)
class ExplanationPolicyConfig:
    """Serving-time explainability policy."""

    default_target_layer: str = "auto"
    default_save_saliency: bool = True
    max_slice_overlays: int = 9


@dataclass(slots=True, frozen=True)
class BackendServingConfig:
    """Backend serving configuration."""

    active_oasis_model_registry: Path
    default_device: str = "cpu"
    threshold_policy: ThresholdPolicyConfig = ThresholdPolicyConfig()
    confidence_policy: ConfidenceBandConfig = ConfidenceBandConfig()
    scan_validation: ScanValidationPolicy = ScanValidationPolicy()
    explanation_policy: ExplanationPolicyConfig = ExplanationPolicyConfig()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        payload = asdict(self)
        payload["active_oasis_model_registry"] = str(self.active_oasis_model_registry)
        payload["scan_validation"]["allowed_suffixes"] = list(self.scan_validation.allowed_suffixes)
        return payload


@dataclass(slots=True, frozen=True)
class OASISDecisionPolicy:
    """Resolved threshold/confidence policy for one OASIS evaluation or inference run."""

    threshold: float
    confidence_config: ConfidenceBandConfig
    serving_config: BackendServingConfig
    registry_entry: ModelRegistryEntry | None = None


@dataclass(slots=True)
class CachedModelBundle:
    """Cached model, config, and checkpoint metadata."""

    model: object
    model_config: OASISModelConfig
    checkpoint: LoadedCheckpoint


def default_serving_config_path() -> Path:
    """Return the default backend serving config path."""

    return resolve_project_root() / "configs" / "backend_serving.yaml"


def _as_path(value: str | Path | None, *, default: Path, base_dir: Path | None = None) -> Path:
    """Normalize a config path."""

    if value in {None, ""}:
        return default
    resolved = Path(value)
    if resolved.is_absolute() or base_dir is None:
        return resolved
    return base_dir / resolved


def _as_tuple(values: Any, *, default: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize suffix lists from YAML."""

    if values is None:
        return default
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list/tuple of suffixes, got {values!r}.")
    return tuple(str(value) for value in values)


def load_backend_serving_config(
    config_path: str | Path | None = None,
    *,
    settings: AppSettings | None = None,
) -> BackendServingConfig:
    """Load backend serving config from YAML."""

    resolved_settings = settings or get_app_settings()
    resolved_path = (
        Path(config_path)
        if config_path is not None
        else resolved_settings.serving_config_path or default_serving_config_path()
    )
    if not resolved_path.exists():
        return BackendServingConfig(
            active_oasis_model_registry=resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json",
        )
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Backend serving config YAML must decode to a dictionary.")

    threshold_section = dict(payload.get("threshold_policy", {}))
    confidence_section = dict(payload.get("confidence_policy", {}))
    scan_validation_section = dict(payload.get("scan_validation", {}))
    explanation_section = dict(payload.get("explanation_policy", {}))

    return BackendServingConfig(
        active_oasis_model_registry=_as_path(
            payload.get("active_oasis_model_registry"),
            default=resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json",
            base_dir=resolved_settings.project_root,
        ),
        default_device=str(payload.get("default_device", "cpu")),
        threshold_policy=ThresholdPolicyConfig(
            use_registry_recommended_threshold=bool(
                threshold_section.get("use_registry_recommended_threshold", True)
            ),
            fallback_threshold=float(threshold_section.get("fallback_threshold", 0.5)),
        ),
        confidence_policy=ConfidenceBandConfig(
            temperature=float(confidence_section.get("temperature", 1.0)),
            high_confidence_min=float(confidence_section.get("high_confidence_min", 0.85)),
            medium_confidence_min=float(confidence_section.get("medium_confidence_min", 0.65)),
            high_entropy_max=float(confidence_section.get("high_entropy_max", 0.35)),
            medium_entropy_max=float(confidence_section.get("medium_entropy_max", 0.90)),
        ),
        scan_validation=ScanValidationPolicy(
            max_file_size_mb=int(scan_validation_section.get("max_file_size_mb", 512)),
            allowed_suffixes=_as_tuple(
                scan_validation_section.get("allowed_suffixes"),
                default=(".hdr", ".img", ".nii", ".nii.gz"),
            ),
            enforce_workspace_boundary=bool(scan_validation_section.get("enforce_workspace_boundary", False)),
        ),
        explanation_policy=ExplanationPolicyConfig(
            default_target_layer=str(explanation_section.get("default_target_layer", "auto")),
            default_save_saliency=bool(explanation_section.get("default_save_saliency", True)),
            max_slice_overlays=int(explanation_section.get("max_slice_overlays", 9)),
        ),
    )


def resolve_inference_threshold(
    *,
    explicit_threshold: float | None,
    registry_entry: ModelRegistryEntry | None,
    serving_config: BackendServingConfig,
) -> float:
    """Resolve the serving threshold with registry-aware policy."""

    if explicit_threshold is not None:
        return float(explicit_threshold)
    if (
        registry_entry is not None
        and serving_config.threshold_policy.use_registry_recommended_threshold
    ):
        return float(registry_entry.recommended_threshold)
    return float(serving_config.threshold_policy.fallback_threshold)


def resolve_confidence_config(
    *,
    serving_config: BackendServingConfig,
    registry_entry: ModelRegistryEntry | None = None,
) -> ConfidenceBandConfig:
    """Resolve the confidence policy, allowing registry overrides later."""

    if registry_entry is None:
        return serving_config.confidence_policy
    scaling = dict(registry_entry.temperature_scaling)
    policy = dict(registry_entry.confidence_policy)
    return ConfidenceBandConfig(
        temperature=float(scaling.get("temperature", serving_config.confidence_policy.temperature)),
        high_confidence_min=float(policy.get("high_confidence_min", serving_config.confidence_policy.high_confidence_min)),
        medium_confidence_min=float(policy.get("medium_confidence_min", serving_config.confidence_policy.medium_confidence_min)),
        high_entropy_max=float(policy.get("high_entropy_max", serving_config.confidence_policy.high_entropy_max)),
        medium_entropy_max=float(policy.get("medium_entropy_max", serving_config.confidence_policy.medium_entropy_max)),
    )


def resolve_oasis_decision_policy(
    *,
    explicit_threshold: float | None = None,
    explicit_confidence_config: ConfidenceBandConfig | None = None,
    serving_config_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    settings: AppSettings | None = None,
) -> OASISDecisionPolicy:
    """Resolve a shared threshold/confidence policy for OASIS evaluation or serving."""

    resolved_settings = settings or get_app_settings()
    serving_config = load_backend_serving_config(serving_config_path, settings=resolved_settings)
    if registry_path is not None:
        registry_entry = load_current_oasis_model_entry(Path(registry_path), settings=resolved_settings)
    else:
        try:
            registry_entry = load_active_oasis_registry_entry(serving_config)
        except FileNotFoundError:
            registry_entry = None

    threshold = resolve_inference_threshold(
        explicit_threshold=explicit_threshold,
        registry_entry=registry_entry,
        serving_config=serving_config,
    )
    confidence_config = explicit_confidence_config or resolve_confidence_config(
        serving_config=serving_config,
        registry_entry=registry_entry,
    )
    return OASISDecisionPolicy(
        threshold=float(threshold),
        confidence_config=confidence_config,
        serving_config=serving_config,
        registry_entry=registry_entry,
    )


def load_active_oasis_registry_entry(
    serving_config: BackendServingConfig,
) -> ModelRegistryEntry:
    """Load the active OASIS registry entry from serving config."""

    return load_current_oasis_model_entry(serving_config.active_oasis_model_registry)


@lru_cache(maxsize=4)
def _load_cached_model_bundle(
    checkpoint_path: str,
    model_config_path: str | None,
    device: str,
) -> CachedModelBundle:
    """Cache the heavy checkpoint/model loading work for serving."""

    resolved_model_config_path = Path(model_config_path) if model_config_path else None
    model_config = load_oasis_model_config(resolved_model_config_path)
    checkpoint = load_oasis_checkpoint(checkpoint_path, device=device)
    model = build_model(model_config)
    model.load_state_dict(checkpoint.model_state_dict)
    model = model.to(device)
    model.eval()
    return CachedModelBundle(model=model, model_config=model_config, checkpoint=checkpoint)


def load_cached_oasis_model_bundle(
    *,
    checkpoint_path: str | Path,
    model_config_path: str | Path | None,
    device: str,
) -> CachedModelBundle:
    """Public wrapper around cached model loading."""

    return _load_cached_model_bundle(
        str(Path(checkpoint_path)),
        None if model_config_path is None else str(Path(model_config_path)),
        str(device),
    )
