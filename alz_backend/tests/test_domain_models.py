"""Tests for patient-management and research domain schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.data.domain import (
    AlertRecord,
    AlertStatus,
    CognitiveTestScore,
    PatientContact,
    PatientProfile,
    ResearchCohortRecord,
    ResearchFeatureSnapshot,
    ScanHistoryEntry,
    SexAtBirth,
    SymptomLog,
)


def test_patient_profile_keeps_identifiable_data_in_clinical_schema() -> None:
    """Patient profile should support PHI only in the identifiable domain."""

    profile = PatientProfile(
        first_name="Jane",
        last_name="Doe",
        date_of_birth="1945-01-01",
        sex_at_birth=SexAtBirth.female,
        contact=PatientContact(email="jane@example.test", phone="555-0100"),
        consent_data_use=True,
    )

    assert profile.patient_id is not None
    assert profile.contact is not None
    assert profile.contact.email == "jane@example.test"
    assert profile.audit.version == 1


def test_scan_history_entry_normalizes_scan_format() -> None:
    """Scan history should preserve storage/provenance metadata."""

    patient_id = uuid4()
    scan = ScanHistoryEntry(
        patient_id=patient_id,
        subject_id="OAS1_0001",
        session_id="OAS1_0001_MR1",
        scan_timestamp=datetime.now(timezone.utc),
        scan_format="nifti",
        storage_uri="s3://bucket/scan.nii.gz",
        checksum_sha256="abc123",
    )

    assert scan.patient_id == patient_id
    assert scan.scan_format == "NIFTI"
    assert scan.storage_uri.endswith("scan.nii.gz")


def test_cognitive_symptom_and_alert_workflow_models() -> None:
    """Care-team workflow schemas should share patient IDs and audit fields."""

    patient_id = uuid4()
    score = CognitiveTestScore(patient_id=patient_id, test_name="MoCA", score=22.0, max_score=30.0)
    symptom = SymptomLog(
        patient_id=patient_id,
        symptom_name="memory_concern",
        severity=6,
        observed_at=datetime.now(timezone.utc),
    )
    alert = AlertRecord(
        patient_id=patient_id,
        alert_type="longitudinal_trend",
        title="Trend review",
        message="Model trend signal changed and should be reviewed by the care team.",
        generated_at=datetime.now(timezone.utc),
        generated_by="longitudinal_engine",
        related_report_uri="outputs/reports/longitudinal/demo.json",
    )

    assert score.patient_id == symptom.patient_id == alert.patient_id
    assert alert.status == AlertStatus.open
    assert symptom.severity == 6


def test_alert_rejects_diagnosis_claim_wording() -> None:
    """Alerts should avoid diagnosis claims."""

    with pytest.raises(ValidationError):
        AlertRecord(
            patient_id=uuid4(),
            alert_type="model_output",
            title="Unsafe wording",
            message="Patient has Alzheimer based on the model.",
            generated_at=datetime.now(timezone.utc),
            generated_by="model_inference",
        )


def test_research_records_are_deidentified_and_feature_ready() -> None:
    """Research cohort records should not require patient identifiers."""

    cohort = ResearchCohortRecord(
        cohort_id=uuid4(),
        research_subject_id="RS-0001",
        source_patient_hash="sha256:example",
        age_bucket="75-79",
        dataset_source="oasis1",
    )
    snapshot = ResearchFeatureSnapshot(
        research_subject_id=cohort.research_subject_id,
        imaging_features={"left_hippocampus_volume_mm3": 3200.0},
        cognitive_features={"moca": 22.0},
        model_outputs={"ad_like_probability": 0.32},
        longitudinal_features={"hippocampal_slope": -6.0},
    )

    assert cohort.deidentification.is_deidentified is True
    assert cohort.research_subject_id == snapshot.research_subject_id
    assert snapshot.model_outputs["ad_like_probability"] == 0.32


def test_research_subject_id_rejects_obvious_phi() -> None:
    """Research IDs should not look like direct patient or MRN identifiers."""

    with pytest.raises(ValidationError):
        ResearchCohortRecord(cohort_id=uuid4(), research_subject_id="patient-123")

