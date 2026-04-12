"""Operational hold rules driven by post-promotion review outcomes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.configs.runtime import AppSettings, get_app_settings
from src.models.factory import load_oasis_model_config
from src.models.registry import ModelRegistryEntry, load_current_oasis_model_entry, save_oasis_model_entry
from src.models.review_analytics import ReviewAnalyticsSummary, analyze_review_records
from src.security.audit import audit_sensitive_action
from src.storage import list_review_records
from src.utils.io_utils import ensure_directory, resolve_project_root


def _utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True, frozen=True)
class OperationalHoldPolicy:
    """Thresholds for switching an approved model into an operational hold."""

    policy_name: str = "oasis_operational_hold_v1"
    minimum_total_reviews: int = 5
    minimum_adjudicated_reviews: int = 5
    override_rate_threshold: float = 0.40
    false_negative_count_threshold: int = 3
    false_negative_fraction_threshold: float = 0.50
    critical_signal_levels: tuple[str, ...] = ("critical",)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe policy payload."""

        payload = asdict(self)
        payload["critical_signal_levels"] = list(self.critical_signal_levels)
        return payload


@dataclass(slots=True)
class OperationalHoldDecision:
    """Structured decision produced by operational hold evaluation."""

    assessed_at_utc: str
    policy_name: str
    previous_status: str
    operational_status: str
    hold_applied: bool
    status_changed: bool
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    trigger_codes: list[str] = field(default_factory=list)
    summary: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe decision payload."""

        return asdict(self)


@dataclass(slots=True)
class ActiveModelHoldAssessment:
    """Artifacts produced when assessing the active model for hold status."""

    registry_entry: ModelRegistryEntry
    registry_path: Path
    review_monitoring: ReviewAnalyticsSummary
    decision: OperationalHoldDecision
    history_path: Path


def default_oasis_hold_policy_path() -> Path:
    """Return the default YAML config for active-model hold rules."""

    return resolve_project_root() / "configs" / "oasis_operational_hold.yaml"


def load_oasis_hold_policy(
    config_path: str | Path | None = None,
    *,
    settings: AppSettings | None = None,
) -> OperationalHoldPolicy:
    """Load operational hold policy config or fall back to defaults."""

    _ = settings or get_app_settings()
    resolved_path = Path(config_path) if config_path is not None else default_oasis_hold_policy_path()
    if not resolved_path.exists():
        return OperationalHoldPolicy()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Operational hold YAML must decode to a dictionary.")
    critical_signal_levels = payload.get("critical_signal_levels")
    if critical_signal_levels is not None:
        payload["critical_signal_levels"] = tuple(str(value) for value in critical_signal_levels)
    return OperationalHoldPolicy(**payload)


def _make_check(*, passed: bool, observed: Any, expected: Any, comparator: str) -> dict[str, Any]:
    """Return one structured hold-evaluation check."""

    return {
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "comparator": comparator,
    }


def evaluate_operational_hold(
    review_monitoring: ReviewAnalyticsSummary,
    *,
    policy: OperationalHoldPolicy,
    previous_status: str = "active",
) -> OperationalHoldDecision:
    """Evaluate whether the active model should be put on operational hold."""

    false_negative_count = int(review_monitoring.error_breakdown.get("false_negative", 0))
    override_rate = review_monitoring.override_rate
    override_rate_observed = None if override_rate is None else round(float(override_rate), 6)
    false_negative_fraction = (
        None
        if review_monitoring.overridden_reviews <= 0
        else round(false_negative_count / review_monitoring.overridden_reviews, 6)
    )
    critical_signal_codes = [
        signal.code
        for signal in review_monitoring.risk_signals
        if signal.level in set(policy.critical_signal_levels)
    ]

    checks = {
        "minimum_total_reviews": _make_check(
            passed=review_monitoring.total_reviews >= policy.minimum_total_reviews,
            observed=review_monitoring.total_reviews,
            expected=policy.minimum_total_reviews,
            comparator=">=",
        ),
        "minimum_adjudicated_reviews": _make_check(
            passed=review_monitoring.adjudicated_reviews >= policy.minimum_adjudicated_reviews,
            observed=review_monitoring.adjudicated_reviews,
            expected=policy.minimum_adjudicated_reviews,
            comparator=">=",
        ),
        "override_rate": _make_check(
            passed=(override_rate is not None) and override_rate >= policy.override_rate_threshold,
            observed=override_rate_observed,
            expected=policy.override_rate_threshold,
            comparator=">=",
        ),
        "false_negative_count": _make_check(
            passed=false_negative_count >= policy.false_negative_count_threshold,
            observed=false_negative_count,
            expected=policy.false_negative_count_threshold,
            comparator=">=",
        ),
        "false_negative_fraction": _make_check(
            passed=(false_negative_fraction is not None)
            and false_negative_fraction >= policy.false_negative_fraction_threshold,
            observed=false_negative_fraction,
            expected=policy.false_negative_fraction_threshold,
            comparator=">=",
        ),
        "critical_review_signals": _make_check(
            passed=bool(critical_signal_codes),
            observed=critical_signal_codes,
            expected=list(policy.critical_signal_levels),
            comparator="contains",
        ),
    }

    sufficient_evidence = checks["minimum_total_reviews"]["passed"] and checks["minimum_adjudicated_reviews"]["passed"]
    override_hold = sufficient_evidence and checks["override_rate"]["passed"]
    false_negative_hold = (
        sufficient_evidence
        and checks["false_negative_count"]["passed"]
        and checks["false_negative_fraction"]["passed"]
    )
    critical_signal_hold = sufficient_evidence and checks["critical_review_signals"]["passed"]

    trigger_codes: list[str] = []
    if override_hold:
        trigger_codes.append("override_rate_threshold")
    if false_negative_hold:
        trigger_codes.append("false_negative_pattern")
    if critical_signal_hold:
        trigger_codes.extend(code for code in critical_signal_codes if code not in trigger_codes)

    hold_applied = bool(trigger_codes)
    operational_status = "hold" if hold_applied else "active"
    notes = [
        "Operational hold is a post-promotion safeguard derived from human review outcomes.",
        "A hold does not erase original benchmark approval evidence; it changes serving posture.",
    ]
    if not sufficient_evidence:
        notes.append("Review evidence is still limited, so the model remains active unless strong triggers emerge later.")
    summary = (
        "Active model placed on operational hold due to review-monitoring triggers: "
        + ", ".join(trigger_codes)
        if hold_applied
        else "No operational hold applied from current review-monitoring evidence."
    )

    return OperationalHoldDecision(
        assessed_at_utc=_utc_now(),
        policy_name=policy.policy_name,
        previous_status=previous_status,
        operational_status=operational_status,
        hold_applied=hold_applied,
        status_changed=previous_status != operational_status,
        checks=checks,
        trigger_codes=trigger_codes,
        summary=summary,
        notes=notes,
    )


def _resolve_model_name(entry: ModelRegistryEntry) -> str | None:
    """Load the architecture name for the active registry entry."""

    try:
        config = load_oasis_model_config(Path(entry.model_config_path) if entry.model_config_path else None)
    except FileNotFoundError:
        return None
    return config.architecture


def _update_registry_entry_for_hold(
    entry: ModelRegistryEntry,
    *,
    review_monitoring: ReviewAnalyticsSummary,
    decision: OperationalHoldDecision,
) -> ModelRegistryEntry:
    """Apply hold metadata and serving restrictions to the active registry entry."""

    entry.operational_status = decision.operational_status
    entry.hold_decision = decision.to_dict()
    entry.review_monitoring = review_monitoring.to_dict()
    entry.serving_restrictions = {
        "force_manual_review": decision.operational_status == "hold",
        "allow_prediction_output": True,
        "block_as_operational_default": decision.operational_status == "hold",
    }
    notes = [
        note
        for note in entry.notes
        if "operational hold" not in note.lower()
    ]
    if decision.operational_status == "hold":
        notes.append(
            "Active model is currently under operational hold due to post-promotion review monitoring."
        )
    else:
        notes.append("Operational hold monitor last assessed the active model without triggering a hold.")
    entry.notes = notes
    return entry


def assess_active_oasis_model_hold(
    *,
    registry_path: str | Path | None = None,
    policy_config_path: str | Path | None = None,
    actor_id: str = "system",
    settings: AppSettings | None = None,
) -> ActiveModelHoldAssessment:
    """Assess the active OASIS model and update its operational hold status."""

    resolved_settings = settings or get_app_settings()
    resolved_registry_path = (
        Path(registry_path)
        if registry_path is not None
        else resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    )
    entry = load_current_oasis_model_entry(path=resolved_registry_path, settings=resolved_settings)
    model_name = _resolve_model_name(entry)
    records = list_review_records(limit=1000, status=None, settings=resolved_settings)
    review_monitoring = analyze_review_records(
        records,
        model_name=model_name,
        active_model_id=entry.model_id,
        run_name=entry.run_name,
    )
    policy = load_oasis_hold_policy(policy_config_path, settings=resolved_settings)
    decision = evaluate_operational_hold(
        review_monitoring,
        policy=policy,
        previous_status=entry.operational_status,
    )
    entry = _update_registry_entry_for_hold(
        entry,
        review_monitoring=review_monitoring,
        decision=decision,
    )
    save_oasis_model_entry(entry, path=resolved_registry_path, settings=resolved_settings)

    history_root = ensure_directory(resolved_settings.outputs_root / "model_registry" / "hold_history")
    history_path = history_root / f"{entry.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    history_payload = {
        "registry_path": str(resolved_registry_path),
        "policy": policy.to_dict(),
        "review_monitoring": review_monitoring.to_dict(),
        "decision": decision.to_dict(),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    if decision.status_changed:
        audit_sensitive_action(
            action="update_model_operational_hold",
            actor_id=actor_id,
            metadata={
                "model_id": entry.model_id,
                "run_name": entry.run_name,
                "previous_status": decision.previous_status,
                "operational_status": decision.operational_status,
                "trigger_codes": decision.trigger_codes,
                "history_path": str(history_path),
            },
        )

    return ActiveModelHoldAssessment(
        registry_entry=entry,
        registry_path=resolved_registry_path,
        review_monitoring=review_monitoring,
        decision=decision,
        history_path=history_path,
    )
