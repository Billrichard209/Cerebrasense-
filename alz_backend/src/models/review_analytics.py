"""Post-promotion review analytics for model governance and operations."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Iterable, Sequence

from src.storage.schemas import ReviewQueueRecord

LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    """Convert a value to ``float`` when possible."""

    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert a value to ``int`` when possible."""

    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a safe proportion or ``None`` when the denominator is empty."""

    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    """Return a regular dictionary sorted by descending count then key."""

    return {
        key: int(counter[key])
        for key in sorted(counter.keys(), key=lambda item: (-counter[item], item))
    }


@dataclass(slots=True)
class ReviewRiskSignal:
    """One review-monitoring signal for model governance."""

    level: str
    code: str
    message: str
    metric: str | None = None
    value: float | int | None = None
    threshold: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return asdict(self)


@dataclass(slots=True)
class ModelReviewBreakdown:
    """Model-specific review summary."""

    model_name: str
    total_reviews: int
    pending_reviews: int
    resolved_reviews: int
    adjudicated_reviews: int
    overridden_reviews: int
    confirmed_reviews: int
    dismissed_reviews: int
    override_rate: float | None = None
    confirmation_rate: float | None = None
    error_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return asdict(self)


@dataclass(slots=True)
class ReviewAnalyticsSummary:
    """Aggregated view of the review queue for operators and governance."""

    generated_at_utc: str
    scope: str
    total_reviews: int
    pending_reviews: int
    resolved_reviews: int
    adjudicated_reviews: int
    overridden_reviews: int
    confirmed_reviews: int
    dismissed_reviews: int
    override_rate: float | None
    confirmation_rate: float | None
    status_counts: dict[str, int] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    reviewer_counts: dict[str, int] = field(default_factory=dict)
    confidence_level_counts: dict[str, int] = field(default_factory=dict)
    error_breakdown: dict[str, int] = field(default_factory=dict)
    error_confidence_distribution: dict[str, int] = field(default_factory=dict)
    label_override_pairs: dict[str, int] = field(default_factory=dict)
    model_breakdown: list[ModelReviewBreakdown] = field(default_factory=list)
    average_error_probability_score: float | None = None
    high_risk: bool = False
    reviewer_agreement_available: bool = False
    reviewer_agreement_note: str = ""
    risk_signals: list[ReviewRiskSignal] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return asdict(self)


def _record_matches_filters(
    record: ReviewQueueRecord,
    *,
    model_name: str | None,
    active_model_id: str | None,
    run_name: str | None,
) -> bool:
    """Return whether a review record matches the requested filter scope."""

    payload = record.payload if isinstance(record.payload, dict) else {}
    has_explicit_scope_match = False
    if active_model_id:
        payload_active_model_id = payload.get("active_model_id")
        if payload_active_model_id not in {None, ""}:
            has_explicit_scope_match = True
            if payload_active_model_id != active_model_id:
                return False
    if run_name:
        payload_run_name = payload.get("model_run_name")
        if payload_run_name not in {None, ""}:
            has_explicit_scope_match = True
            if payload_run_name != run_name:
                return False
    if model_name and record.model_name != model_name:
        return False
    if (active_model_id or run_name) and model_name is None and not has_explicit_scope_match:
        return False
    return True


def _resolve_scope(
    *,
    model_name: str | None,
    active_model_id: str | None,
    run_name: str | None,
) -> str:
    """Build a compact scope description for the analytics payload."""

    parts: list[str] = []
    if active_model_id:
        parts.append(f"active_model_id={active_model_id}")
    if run_name:
        parts.append(f"run_name={run_name}")
    if model_name:
        parts.append(f"model_name={model_name}")
    return ", ".join(parts) if parts else "all_review_queue_records"


def _classification_direction(record: ReviewQueueRecord) -> str | None:
    """Classify an override as false positive, false negative, or unknown."""

    payload = record.payload if isinstance(record.payload, dict) else {}
    resolution = payload.get("resolution", {})
    if not isinstance(resolution, dict):
        return None
    if resolution.get("action") != "override_prediction":
        return None
    predicted_label = _safe_int(payload.get("predicted_label"))
    resolved_label = _safe_int(resolution.get("resolved_label"))
    if predicted_label is None or resolved_label is None:
        return "unknown_direction"
    if predicted_label == 1 and resolved_label == 0:
        return "false_positive"
    if predicted_label == 0 and resolved_label == 1:
        return "false_negative"
    return "label_flip_other"


def _label_override_pair(record: ReviewQueueRecord) -> str | None:
    """Build a compact label transition string for one override."""

    payload = record.payload if isinstance(record.payload, dict) else {}
    resolution = payload.get("resolution", {})
    if not isinstance(resolution, dict) or resolution.get("action") != "override_prediction":
        return None
    predicted_name = str(payload.get("label_name", f"class_{payload.get('predicted_label', 'unknown')}"))
    resolved_name = str(
        resolution.get("resolved_label_name", f"class_{resolution.get('resolved_label', 'unknown')}")
    )
    return f"{predicted_name} -> {resolved_name}"


def _build_risk_signals(
    *,
    total_reviews: int,
    pending_reviews: int,
    adjudicated_reviews: int,
    dismissed_reviews: int,
    override_rate: float | None,
    error_breakdown: Counter[str],
    confidence_level_counts: Counter[str],
) -> list[ReviewRiskSignal]:
    """Generate compact review-monitoring warnings from aggregated counts."""

    signals: list[ReviewRiskSignal] = []
    low_confidence_reviews = int(confidence_level_counts.get("low", 0))

    if total_reviews < 5:
        signals.append(
            ReviewRiskSignal(
                level="info",
                code="limited_review_evidence",
                metric="total_reviews",
                value=total_reviews,
                threshold=5,
                message="Review analytics are still based on a small number of queued cases.",
            )
        )
    if override_rate is not None and adjudicated_reviews >= 5 and override_rate >= 0.30:
        signals.append(
            ReviewRiskSignal(
                level="warning" if override_rate < 0.50 else "critical",
                code="high_override_rate",
                metric="override_rate",
                value=round(override_rate, 4),
                threshold=0.30,
                message="Reviewed cases show a high model-override rate; re-check calibration and promotion evidence.",
            )
        )
    pending_rate = _rate(pending_reviews, total_reviews)
    if pending_rate is not None and pending_reviews >= 5 and pending_rate >= 0.40:
        signals.append(
            ReviewRiskSignal(
                level="warning",
                code="review_backlog",
                metric="pending_rate",
                value=round(pending_rate, 4),
                threshold=0.40,
                message="A large fraction of queued cases remain unresolved, which weakens post-promotion monitoring.",
            )
        )
    if low_confidence_reviews >= 5 and low_confidence_reviews == total_reviews:
        signals.append(
            ReviewRiskSignal(
                level="info",
                code="all_reviews_low_confidence",
                metric="low_confidence_reviews",
                value=low_confidence_reviews,
                threshold=5,
                message="All reviewed cases were low-confidence escalations; operating thresholds may still be conservative.",
            )
        )
    false_positive_count = int(error_breakdown.get("false_positive", 0))
    false_negative_count = int(error_breakdown.get("false_negative", 0))
    if false_negative_count >= 3 and false_negative_count > false_positive_count:
        signals.append(
            ReviewRiskSignal(
                level="warning",
                code="false_negative_pattern",
                metric="false_negative_count",
                value=false_negative_count,
                threshold=3,
                message="Overrides currently lean toward false negatives; sensitivity should be re-checked on the approved threshold.",
            )
        )
    if false_positive_count >= 3 and false_positive_count > false_negative_count:
        signals.append(
            ReviewRiskSignal(
                level="warning",
                code="false_positive_pattern",
                metric="false_positive_count",
                value=false_positive_count,
                threshold=3,
                message="Overrides currently lean toward false positives; specificity and operating thresholds should be reviewed.",
            )
        )
    if dismissed_reviews >= 3 and dismissed_reviews >= adjudicated_reviews:
        signals.append(
            ReviewRiskSignal(
                level="info",
                code="dismissal_pattern",
                metric="dismissed_reviews",
                value=dismissed_reviews,
                threshold=3,
                message="Many review cases are being dismissed, which may indicate data-quality or queue-triage issues.",
            )
        )
    return signals


def analyze_review_records(
    records: Sequence[ReviewQueueRecord],
    *,
    model_name: str | None = None,
    active_model_id: str | None = None,
    run_name: str | None = None,
) -> ReviewAnalyticsSummary:
    """Aggregate stored review records into an operator-friendly summary."""

    selected_records = [
        record
        for record in records
        if _record_matches_filters(
            record,
            model_name=model_name,
            active_model_id=active_model_id,
            run_name=run_name,
        )
    ]
    scope = _resolve_scope(model_name=model_name, active_model_id=active_model_id, run_name=run_name)
    notes: list[str] = []
    if selected_records and active_model_id:
        missing_active_ids = sum(
            1
            for record in selected_records
            if not isinstance(record.payload, dict) or record.payload.get("active_model_id") in {None, ""}
        )
        if missing_active_ids:
            notes.append(
                "Some review records predate active-model identifiers, so filtering may fall back to architecture/run matching."
            )
    if not selected_records:
        notes.append("No review records matched the requested scope yet.")

    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reviewer_counts: Counter[str] = Counter()
    confidence_level_counts: Counter[str] = Counter()
    error_breakdown: Counter[str] = Counter()
    error_confidence_distribution: Counter[str] = Counter()
    label_override_pairs: Counter[str] = Counter()
    probability_values_for_errors: list[float] = []
    model_totals: Counter[str] = Counter()
    model_status_counts: dict[str, Counter[str]] = {}
    model_error_counts: dict[str, Counter[str]] = {}

    for record in selected_records:
        payload = record.payload if isinstance(record.payload, dict) else {}
        resolution = payload.get("resolution", {})
        if not isinstance(resolution, dict):
            resolution = {}

        status_counts[record.status] += 1
        confidence_level_counts[str(record.confidence_level or "unknown")] += 1
        model_totals[record.model_name] += 1
        model_status_counts.setdefault(record.model_name, Counter())[record.status] += 1
        model_error_counts.setdefault(record.model_name, Counter())

        action = resolution.get("action")
        if action:
            action_counts[str(action)] += 1
        elif record.status == "confirmed":
            action_counts["confirm_model_output"] += 1
        elif record.status == "overridden":
            action_counts["override_prediction"] += 1
        elif record.status == "dismissed":
            action_counts["dismiss"] += 1

        reviewer_id = resolution.get("reviewer_id")
        if reviewer_id:
            reviewer_counts[str(reviewer_id)] += 1

        direction = _classification_direction(record)
        if direction is not None:
            error_breakdown[direction] += 1
            model_error_counts[record.model_name][direction] += 1
            error_confidence_distribution[str(record.confidence_level or "unknown")] += 1
            probability_value = _safe_float(record.probability_score)
            if probability_value is not None:
                probability_values_for_errors.append(probability_value)
            transition = _label_override_pair(record)
            if transition:
                label_override_pairs[transition] += 1

    total_reviews = len(selected_records)
    pending_reviews = int(status_counts.get("pending", 0))
    confirmed_reviews = int(status_counts.get("confirmed", 0))
    overridden_reviews = int(status_counts.get("overridden", 0))
    dismissed_reviews = int(status_counts.get("dismissed", 0))
    resolved_reviews = total_reviews - pending_reviews
    adjudicated_reviews = confirmed_reviews + overridden_reviews
    override_rate = _rate(overridden_reviews, adjudicated_reviews)
    confirmation_rate = _rate(confirmed_reviews, adjudicated_reviews)

    model_breakdown: list[ModelReviewBreakdown] = []
    for current_model_name in sorted(model_totals.keys()):
        current_status_counts = model_status_counts.get(current_model_name, Counter())
        current_confirmed = int(current_status_counts.get("confirmed", 0))
        current_overridden = int(current_status_counts.get("overridden", 0))
        current_pending = int(current_status_counts.get("pending", 0))
        current_dismissed = int(current_status_counts.get("dismissed", 0))
        current_adjudicated = current_confirmed + current_overridden
        current_total = int(model_totals[current_model_name])
        model_breakdown.append(
            ModelReviewBreakdown(
                model_name=current_model_name,
                total_reviews=current_total,
                pending_reviews=current_pending,
                resolved_reviews=current_total - current_pending,
                adjudicated_reviews=current_adjudicated,
                overridden_reviews=current_overridden,
                confirmed_reviews=current_confirmed,
                dismissed_reviews=current_dismissed,
                override_rate=_rate(current_overridden, current_adjudicated),
                confirmation_rate=_rate(current_confirmed, current_adjudicated),
                error_breakdown=_sorted_counts(model_error_counts.get(current_model_name, Counter())),
            )
        )

    risk_signals = _build_risk_signals(
        total_reviews=total_reviews,
        pending_reviews=pending_reviews,
        adjudicated_reviews=adjudicated_reviews,
        dismissed_reviews=dismissed_reviews,
        override_rate=override_rate,
        error_breakdown=error_breakdown,
        confidence_level_counts=confidence_level_counts,
    )

    if not reviewer_counts:
        notes.append("No resolved reviewer identities have been recorded yet.")
    notes.append(
        "Reviewer agreement is not available yet because the current queue stores one primary reviewer resolution per case."
    )

    summary = ReviewAnalyticsSummary(
        generated_at_utc=_utc_now(),
        scope=scope,
        total_reviews=total_reviews,
        pending_reviews=pending_reviews,
        resolved_reviews=resolved_reviews,
        adjudicated_reviews=adjudicated_reviews,
        overridden_reviews=overridden_reviews,
        confirmed_reviews=confirmed_reviews,
        dismissed_reviews=dismissed_reviews,
        override_rate=override_rate,
        confirmation_rate=confirmation_rate,
        status_counts=_sorted_counts(status_counts),
        action_counts=_sorted_counts(action_counts),
        reviewer_counts=_sorted_counts(reviewer_counts),
        confidence_level_counts=_sorted_counts(confidence_level_counts),
        error_breakdown=_sorted_counts(error_breakdown),
        error_confidence_distribution=_sorted_counts(error_confidence_distribution),
        label_override_pairs=_sorted_counts(label_override_pairs),
        model_breakdown=model_breakdown,
        average_error_probability_score=(round(mean(probability_values_for_errors), 6) if probability_values_for_errors else None),
        high_risk=any(signal.level in {"warning", "critical"} for signal in risk_signals),
        reviewer_agreement_available=False,
        reviewer_agreement_note=(
            "Single-review resolution workflow only; inter-reviewer agreement is not measurable yet."
        ),
        risk_signals=risk_signals,
        notes=notes,
    )
    LOGGER.debug("Built review analytics summary for scope %s with %d records", scope, total_reviews)
    return summary


def summarize_review_records(
    records: Iterable[ReviewQueueRecord],
    *,
    model_name: str | None = None,
    active_model_id: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning a JSON-safe payload."""

    return analyze_review_records(
        list(records),
        model_name=model_name,
        active_model_id=active_model_id,
        run_name=run_name,
    ).to_dict()
