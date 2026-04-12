"""Starter access-control hooks for API and service layers."""

from __future__ import annotations

from dataclasses import dataclass, field


class AccessControlError(PermissionError):
    """Raised when an actor is not allowed to perform an action."""


@dataclass(slots=True, frozen=True)
class ActorContext:
    """Minimal actor context for future RBAC integration."""

    actor_id: str | None
    roles: frozenset[str] = field(default_factory=frozenset)
    purpose: str = "clinical_decision_support"


ACTION_ROLE_POLICY: dict[str, frozenset[str]] = {
    "predict_scan": frozenset({"clinician", "nurse", "researcher", "admin", "system"}),
    "explain_scan": frozenset({"clinician", "researcher", "admin", "system"}),
    "build_longitudinal_report": frozenset({"clinician", "nurse", "researcher", "admin", "system"}),
    "export_research_record": frozenset({"researcher", "admin", "system"}),
}


def require_action_allowed(actor: ActorContext, action: str) -> None:
    """Check a minimal role/action policy.

    This is scaffolding, not production-grade hospital access control.
    """

    allowed_roles = ACTION_ROLE_POLICY.get(action)
    if allowed_roles is None:
        raise AccessControlError(f"Unknown protected action: {action}")
    if actor.roles & allowed_roles:
        return
    raise AccessControlError(f"Actor is not allowed to perform action {action!r}.")

