# Backend Domain Model Notes

These schemas are Pydantic domain models for future patient management, nurse
workflows, research analytics, and PostgreSQL storage. They are not CRUD or ORM
models yet.

## Entity Relationship Summary

Identifiable clinical domain:

```text
PatientProfile 1---N ScanHistoryEntry
PatientProfile 1---N CognitiveTestScore
PatientProfile 1---N MedicationRecord
PatientProfile 1---N AppointmentRecord
PatientProfile 1---N CaregiverNote
PatientProfile 1---N SymptomLog
PatientProfile 1---N AlertRecord

ScanHistoryEntry 1---N AlertRecord
ScanHistoryEntry 1---N CognitiveTestScore optional relation
```

Research domain:

```text
ResearchCohortRecord 1---N ResearchFeatureSnapshot
ResearchCohortRecord uses research_subject_id, not patient_id
ResearchFeatureSnapshot stores anonymized imaging/cognitive/model/longitudinal features
```

The clinical domain may contain PHI such as name, contact, medical record
number, and appointment details. The research domain must not contain direct
patient identifiers and should use pseudonymized or hashed linkage fields only.

## PostgreSQL-Ready Design Notes

- Use `uuid` primary keys for patient-facing tables.
- Use `timestamptz` for audit fields and event timestamps.
- Use `jsonb` for flexible `metadata`, feature dictionaries, and model outputs.
- Add indexes on `patient_id`, `scan_id`, `research_subject_id`, `cohort_id`,
  `scan_timestamp`, `observed_at`, `scheduled_start`, and `alert status/priority`.
- Store large files in object storage or filesystem-backed storage and keep
  only `storage_uri`, checksums, and provenance in PostgreSQL.
- Consider table-level separation or schemas:
  - `clinical.patient_profiles`
  - `clinical.scan_history`
  - `clinical.care_notes`
  - `research.cohort_records`
  - `research.feature_snapshots`
- Add row-level security before production if multiple care teams or research
  groups share one database.
- Add append-only audit logs later for regulatory-grade traceability; the
  current `AuditFields` are starter fields only.

## De-Identification Boundary

- `PatientProfile`, `PatientContact`, and clinical workflow records are
  patient-identifiable.
- `ResearchCohortRecord` and `ResearchFeatureSnapshot` are anonymized research
  records.
- Research records use `research_subject_id` and optional `source_patient_hash`,
  never direct `patient_id`, names, contact details, or MRNs.

