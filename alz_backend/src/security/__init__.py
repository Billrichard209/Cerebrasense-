"""Security, privacy, and ethics helpers for safe backend behavior."""

from .access_control import AccessControlError, ActorContext, require_action_allowed
from .audit import AuditEvent, audit_sensitive_action, emit_audit_event
from .deidentification import (
    DEFAULT_PHI_KEYS,
    assert_no_phi_keys,
    deidentify_mapping,
    pseudonymize_identifier,
    redact_text,
)
from .disclaimers import (
    EXPLANATION_LIMITATION,
    STANDARD_DECISION_SUPPORT_DISCLAIMER,
    add_decision_support_disclaimer,
    build_ai_summary,
)
from .governance import DEFAULT_POLICY, PROJECT_POLICY, ProjectPolicy, assert_decision_support_policy, get_policy_snapshot

__all__ = [
    "AccessControlError",
    "ActorContext",
    "AuditEvent",
    "DEFAULT_PHI_KEYS",
    "DEFAULT_POLICY",
    "EXPLANATION_LIMITATION",
    "PROJECT_POLICY",
    "ProjectPolicy",
    "STANDARD_DECISION_SUPPORT_DISCLAIMER",
    "add_decision_support_disclaimer",
    "assert_decision_support_policy",
    "assert_no_phi_keys",
    "audit_sensitive_action",
    "build_ai_summary",
    "deidentify_mapping",
    "emit_audit_event",
    "get_policy_snapshot",
    "pseudonymize_identifier",
    "redact_text",
    "require_action_allowed",
]
