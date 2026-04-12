"""Audit logging hooks for sensitive backend actions."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .deidentification import deidentify_mapping

SECURITY_AUDIT_LOGGER_NAME = "alz_backend.security.audit"
logger = logging.getLogger(SECURITY_AUDIT_LOGGER_NAME)


@dataclass(slots=True)
class AuditEvent:
    """Structured audit event for sensitive backend actions."""

    action: str
    actor_id: str | None = None
    subject_id: str | None = None
    patient_id_hash: str | None = None
    outcome: str = "success"
    purpose: str = "clinical_decision_support"
    event_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return an audit-safe JSON payload."""

        payload = asdict(self)
        payload["metadata"] = deidentify_mapping(payload.get("metadata", {}))
        return payload


def emit_audit_event(event: AuditEvent) -> dict[str, Any]:
    """Emit an audit event through the standard Python logger."""

    payload = event.to_payload()
    logger.info("security_audit_event=%s", json.dumps(payload, sort_keys=True))
    try:
        from src.storage import AuditMetadataRecord, persist_audit_record

        persist_audit_record(
            AuditMetadataRecord(
                action=event.action,
                actor_id=event.actor_id,
                subject_id=event.subject_id,
                patient_id_hash=event.patient_id_hash,
                outcome=event.outcome,
                purpose=event.purpose,
                payload=payload,
                event_time=event.event_time,
            )
        )
    except Exception as error:  # pragma: no cover - defensive persistence hook
        logger.warning("audit_persistence_warning=%s", error)
    return payload


def audit_sensitive_action(
    *,
    action: str,
    actor_id: str | None = None,
    subject_id: str | None = None,
    patient_id_hash: str | None = None,
    outcome: str = "success",
    purpose: str = "clinical_decision_support",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience hook for service/API layers to log sensitive actions."""

    return emit_audit_event(
        AuditEvent(
            action=action,
            actor_id=actor_id,
            subject_id=subject_id,
            patient_id_hash=patient_id_hash,
            outcome=outcome,
            purpose=purpose,
            metadata=metadata or {},
        )
    )
