"""Central dataset registry builders for OASIS-first backend workflows."""

from __future__ import annotations

from src.configs.runtime import AppSettings

from .kaggle_dataset import KaggleDatasetSpec, build_kaggle_dataset_spec
from .oasis_dataset import OASISDatasetSpec, build_oasis_dataset_spec


def build_dataset_registry(
    settings: AppSettings | None = None,
) -> dict[str, OASISDatasetSpec | KaggleDatasetSpec]:
    """Build the live dataset registry while preserving explicit dataset separation."""

    resolved_settings = settings or AppSettings.from_env()
    return {
        "oasis": build_oasis_dataset_spec(resolved_settings),
        "kaggle": build_kaggle_dataset_spec(resolved_settings),
    }


def build_dataset_registry_snapshot(settings: AppSettings | None = None) -> dict[str, object]:
    """Return a JSON-safe dataset registry snapshot for the API layer."""

    resolved_settings = settings or AppSettings.from_env()
    registry = build_dataset_registry(resolved_settings)
    return {
        "primary_dataset": resolved_settings.primary_dataset,
        "datasets": {name: spec.to_snapshot() for name, spec in registry.items()},
    }
