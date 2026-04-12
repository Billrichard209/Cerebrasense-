"""Shared domain primitives for auditability and de-identification."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RecordStatus(str, Enum):
    """Lifecycle state for stored domain records."""

    active = "active"
    archived = "archived"
    deleted = "deleted"


class AuditFields(BaseModel):
    """Audit metadata intended to map cleanly to PostgreSQL columns."""

    model_config = ConfigDict(extra="forbid")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    created_by: UUID | None = None
    updated_by: UUID | None = None
    source_system: str | None = None
    status: RecordStatus = RecordStatus.active
    version: int = Field(default=1, ge=1)


class DataProvenance(BaseModel):
    """Where a record or measurement came from."""

    model_config = ConfigDict(extra="forbid")

    source_dataset: str | None = None
    source_record_id: str | None = None
    source_path: str | None = None
    ingestion_run_id: UUID | None = None
    model_run_id: str | None = None
    notes: str | None = None


class DeIdentificationInfo(BaseModel):
    """De-identification metadata for research records."""

    model_config = ConfigDict(extra="forbid")

    is_deidentified: bool = True
    method: str = "pseudonymized_subject_id"
    deidentified_at: datetime | None = None
    source_patient_hash: str | None = None
    phi_removed: bool = True
    fields_removed: list[str] = Field(default_factory=list)
    notes: str | None = None


class FlexibleMetadata(BaseModel):
    """Optional JSONB-style extension metadata."""

    model_config = ConfigDict(extra="allow")

    values: dict[str, Any] = Field(default_factory=dict)

