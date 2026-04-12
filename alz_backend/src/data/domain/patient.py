"""Patient-identifiable domain schemas.

These schemas are intentionally separate from anonymized research records.
They are backend-ready Pydantic models, not database ORM classes yet.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import AuditFields, DataProvenance


class SexAtBirth(str, Enum):
    """Sex-at-birth values for clinical recordkeeping."""

    female = "female"
    male = "male"
    intersex = "intersex"
    unknown = "unknown"


class PatientContact(BaseModel):
    """Contact details kept only in identifiable patient tables."""

    model_config = ConfigDict(extra="forbid")

    phone: str | None = None
    email: str | None = None
    address_line_1: str | None = None
    address_line_2: str | None = None
    city: str | None = None
    region: str | None = None
    postal_code: str | None = None
    country: str | None = None
    emergency_contact_name: str | None = None
    emergency_contact_phone: str | None = None


class PatientProfile(BaseModel):
    """Patient-identifiable profile for future patient management workflows."""

    model_config = ConfigDict(extra="forbid")

    patient_id: UUID = Field(default_factory=uuid4)
    external_patient_id: str | None = None
    medical_record_number: str | None = None
    first_name: str
    last_name: str
    preferred_name: str | None = None
    date_of_birth: date | None = None
    sex_at_birth: SexAtBirth = SexAtBirth.unknown
    primary_language: str | None = None
    contact: PatientContact | None = None
    care_team_ids: list[UUID] = Field(default_factory=list)
    primary_caregiver_id: UUID | None = None
    consent_research_contact: bool = False
    consent_data_use: bool = False
    audit: AuditFields = Field(default_factory=AuditFields)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScanHistoryEntry(BaseModel):
    """One scan event linked to an identifiable patient profile."""

    model_config = ConfigDict(extra="forbid")

    scan_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID
    subject_id: str | None = None
    session_id: str | None = None
    scan_timestamp: datetime | None = None
    modality: str = "MRI"
    body_region: str = "brain"
    scan_format: str | None = Field(default=None, description="Examples: NIfTI, Analyze, DICOM.")
    storage_uri: str
    checksum_sha256: str | None = None
    dataset_source: str | None = None
    inference_report_uri: str | None = None
    explanation_report_uri: str | None = None
    structural_report_uri: str | None = None
    longitudinal_report_uri: str | None = None
    provenance: DataProvenance = Field(default_factory=DataProvenance)
    audit: AuditFields = Field(default_factory=AuditFields)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("scan_format")
    @classmethod
    def _normalize_scan_format(cls, value: str | None) -> str | None:
        """Normalize optional scan format values for storage/search."""

        return value.upper() if value else None

