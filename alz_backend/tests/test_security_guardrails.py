"""Tests for foundational security and ethics guardrails."""

from __future__ import annotations

import logging

import pytest

from src.security import (
    STANDARD_DECISION_SUPPORT_DISCLAIMER,
    AccessControlError,
    ActorContext,
    add_decision_support_disclaimer,
    assert_no_phi_keys,
    audit_sensitive_action,
    build_ai_summary,
    deidentify_mapping,
    pseudonymize_identifier,
    redact_text,
    require_action_allowed,
)


def test_standard_disclaimer_is_decision_support_only() -> None:
    """The shared disclaimer should avoid diagnosis claims."""

    assert "not a diagnosis" in STANDARD_DECISION_SUPPORT_DISCLAIMER.lower()
    assert "clinical judgment" in STANDARD_DECISION_SUPPORT_DISCLAIMER.lower()


def test_build_ai_summary_uses_standard_disclaimer() -> None:
    """AI summary wording should include the shared disclaimer."""

    summary = build_ai_summary(label_name="demented", probability_score=0.67, confidence_score=0.8)

    assert STANDARD_DECISION_SUPPORT_DISCLAIMER in summary
    assert "not a diagnosis" in summary.lower()


def test_add_decision_support_disclaimer_injects_fields() -> None:
    """Payloads should get a standard disclaimer without mutating by default."""

    original = {"notes": []}
    updated = add_decision_support_disclaimer(original)

    assert "clinical_disclaimer" in updated
    assert updated["decision_support_only"] is True
    assert original == {"notes": []}


def test_deidentify_mapping_redacts_phi_keys_and_text_patterns() -> None:
    """Research export helpers should remove obvious PHI/PII values."""

    cleaned = deidentify_mapping(
        {
            "patient_id": "abc",
            "nested": {"email": "person@example.com"},
            "note": "Call +1 555 123 4567 or person@example.com",
            "safe_feature": 0.42,
        }
    )

    assert cleaned["patient_id"] == "[REDACTED]"
    assert cleaned["nested"]["email"] == "[REDACTED]"
    assert "person@example.com" not in cleaned["note"]
    assert cleaned["safe_feature"] == 0.42


def test_assert_no_phi_keys_fails_for_research_exports() -> None:
    """Obvious PHI key names should fail research export validation."""

    with pytest.raises(ValueError, match="patient_id"):
        assert_no_phi_keys({"research_subject_id": "RS-1", "patient_id": "abc"})


def test_pseudonymize_identifier_is_deterministic() -> None:
    """Pseudonymous IDs should be deterministic with the same salt."""

    assert pseudonymize_identifier("patient-1", salt="unit") == pseudonymize_identifier("patient-1", salt="unit")
    assert pseudonymize_identifier("patient-1", salt="unit") != pseudonymize_identifier("patient-2", salt="unit")


def test_access_control_hook_allows_and_denies_roles() -> None:
    """Starter access-control hooks should enforce action roles."""

    require_action_allowed(ActorContext(actor_id="u1", roles=frozenset({"researcher"})), "export_research_record")

    with pytest.raises(AccessControlError):
        require_action_allowed(ActorContext(actor_id="u2", roles=frozenset({"nurse"})), "export_research_record")


def test_audit_sensitive_action_redacts_metadata(caplog: pytest.LogCaptureFixture) -> None:
    """Audit events should log structured and redacted metadata."""

    caplog.set_level(logging.INFO, logger="alz_backend.security.audit")
    payload = audit_sensitive_action(
        action="predict_scan",
        subject_id="OAS1_0001",
        metadata={"email": "person@example.com", "scan_file_name": "scan.hdr"},
    )

    assert payload["metadata"]["email"] == "[REDACTED]"
    assert payload["metadata"]["scan_file_name"] == "scan.hdr"
    assert any("security_audit_event" in record.message for record in caplog.records)


def test_redact_text_removes_email_and_phone_patterns() -> None:
    """Free-text redaction should remove common contact patterns."""

    redacted = redact_text("Email person@example.com or call 555-123-4567")

    assert "person@example.com" not in redacted
    assert "555-123-4567" not in redacted

