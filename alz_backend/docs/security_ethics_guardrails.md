# Security & Ethics Guardrails

This is a foundational backend security and ethics layer. It is not
production-grade hospital security and does not replace HIPAA/GDPR compliance
review, threat modeling, penetration testing, or institutional governance.

## Standard Disclaimer

All model-facing outputs should reuse:

```text
This output is for research and clinical decision support only. It is not a diagnosis, does not replace qualified clinical judgment, and must be reviewed with the source imaging and clinical context.
```

The source constant is `STANDARD_DECISION_SUPPORT_DISCLAIMER` in
`src/security/disclaimers.py`.

## PHI / PII Rules For Research Exports

Do not store these in anonymized research exports:

- patient IDs
- names
- medical record numbers
- dates of birth
- phone numbers
- emails
- direct addresses
- emergency contact details

Use `research_subject_id`, optional one-way `source_patient_hash`, and
`DeIdentificationInfo` metadata instead.

## Starter Guardrail Modules

- `src/security/deidentification.py`: pseudonymization, redaction, PHI-key checks.
- `src/security/access_control.py`: minimal role/action access hooks for future RBAC.
- `src/security/audit.py`: structured audit-event logging with metadata redaction.
- `src/security/disclaimers.py`: shared decision-support wording and output injection helpers.

## Audit Logging Pattern

Example:

```python
from src.security.audit import audit_sensitive_action

audit_sensitive_action(
    action="predict_scan",
    subject_id="OAS1_0001",
    metadata={"scan_file_name": "scan.hdr", "email": "person@example.com"},
)
```

The audit payload redacts PHI-like metadata before logging.

