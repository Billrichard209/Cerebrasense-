"""System, policy, dataset, and model metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import (
    DatasetRegistryResponse,
    HealthResponse,
    ModelMetadataResponse,
    PolicyResponse,
    RootResponse,
)
from src.api.services import (
    build_dataset_registry_payload,
    build_health_payload,
    build_model_metadata_payload,
    build_policy_payload,
    build_root_payload,
)

router = APIRouter(tags=["system"])


@router.get("/", response_model=RootResponse)
def root() -> RootResponse:
    """Return startup metadata and safety posture."""

    return RootResponse(**build_root_payload())


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a basic health payload for orchestration checks."""

    return HealthResponse(**build_health_payload())


@router.get("/policy", response_model=PolicyResponse)
def policy() -> PolicyResponse:
    """Return the active governance posture for decision-support use."""

    return PolicyResponse(**build_policy_payload())


@router.get("/datasets", response_model=DatasetRegistryResponse)
def datasets() -> DatasetRegistryResponse:
    """Expose the separate OASIS and Kaggle dataset registrations."""

    return DatasetRegistryResponse(**build_dataset_registry_payload())


@router.get("/models/oasis/metadata", response_model=ModelMetadataResponse)
def oasis_model_metadata(model_config_path: str | None = None) -> ModelMetadataResponse:
    """Return metadata for the OASIS baseline model factory."""

    return ModelMetadataResponse(**build_model_metadata_payload(model_config_path=model_config_path))

