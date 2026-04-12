"""Anonymized research cohort schemas separated from patient-identifiable data."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import AuditFields, DataProvenance, DeIdentificationInfo


class ResearchDataUse(str, Enum):
    """Permitted research data use category."""

    internal_validation = "internal_validation"
    retrospective_research = "retrospective_research"
    external_collaboration = "external_collaboration"


class ResearchFeatureSnapshot(BaseModel):
    """Anonymized feature snapshot for analytics or fusion models."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: UUID = Field(default_factory=uuid4)
    research_subject_id: str
    session_id: str | None = None
    scan_timestamp_shifted: date | None = Field(
        default=None,
        description="Optional shifted/bucketed date, not the direct clinical scan timestamp.",
    )
    imaging_features: dict[str, float] = Field(default_factory=dict)
    cognitive_features: dict[str, float] = Field(default_factory=dict)
    model_outputs: dict[str, float] = Field(default_factory=dict)
    longitudinal_features: dict[str, float] = Field(default_factory=dict)
    provenance: DataProvenance = Field(default_factory=DataProvenance)
    deidentification: DeIdentificationInfo = Field(default_factory=DeIdentificationInfo)
    audit: AuditFields = Field(default_factory=AuditFields)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchCohortRecord(BaseModel):
    """Anonymized cohort membership record for research analytics."""

    model_config = ConfigDict(extra="forbid")

    cohort_record_id: UUID = Field(default_factory=uuid4)
    cohort_id: UUID
    research_subject_id: str
    source_patient_hash: str | None = Field(
        default=None,
        description="One-way hash for linkage auditing; never a patient_id or MRN.",
    )
    age_bucket: str | None = Field(default=None, description="Example: 65-69; avoid exact DOB.")
    sex_at_birth: str | None = None
    dataset_source: str | None = None
    consent_scope: ResearchDataUse = ResearchDataUse.internal_validation
    feature_snapshot_ids: list[UUID] = Field(default_factory=list)
    deidentification: DeIdentificationInfo = Field(default_factory=DeIdentificationInfo)
    provenance: DataProvenance = Field(default_factory=DataProvenance)
    audit: AuditFields = Field(default_factory=AuditFields)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("research_subject_id")
    @classmethod
    def _no_obvious_phi_subject_id(cls, value: str) -> str:
        """Reject obvious patient-identifiable subject IDs in research records."""

        lowered = value.lower()
        if "patient" in lowered or "mrn" in lowered:
            raise ValueError("research_subject_id must be de-identified and must not contain patient/MRN identifiers.")
        return value

