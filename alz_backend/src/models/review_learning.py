"""Advisory reviewer-outcome learning loop for threshold and retraining guidance."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from src.evaluation.metrics import compute_binary_classification_metrics, threshold_binary_scores
from src.storage.schemas import ReviewQueueRecord

LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible."""

    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert a value to int when possible."""

    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a safe rate or ``None`` when denominator is empty."""

    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


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


def _resolve_scope(*, model_name: str | None, active_model_id: str | None, run_name: str | None) -> str:
    """Build a compact scope description for the report."""

    parts: list[str] = []
    if active_model_id:
        parts.append(f"active_model_id={active_model_id}")
    if run_name:
        parts.append(f"run_name={run_name}")
    if model_name:
        parts.append(f"model_name={model_name}")
    return ", ".join(parts) if parts else "all_review_queue_records"


def _resolved_final_label(record: ReviewQueueRecord) -> int | None:
    """Return the reviewer-adjudicated final label for a record when available."""

    payload = record.payload if isinstance(record.payload, dict) else {}
    resolution = payload.get("resolution", {})
    if not isinstance(resolution, dict):
        resolution = {}
    if record.status == "confirmed":
        return _safe_int(payload.get("predicted_label"))
    if record.status == "overridden":
        return _safe_int(resolution.get("resolved_label"))
    return None


def _classification_direction(record: ReviewQueueRecord) -> str | None:
    """Classify one adjudicated review case as FP, FN, or non-error."""

    payload = record.payload if isinstance(record.payload, dict) else {}
    predicted_label = _safe_int(payload.get("predicted_label"))
    final_label = _resolved_final_label(record)
    if predicted_label is None or final_label is None:
        return None
    if predicted_label == 1 and final_label == 0:
        return "false_positive"
    if predicted_label == 0 and final_label == 1:
        return "false_negative"
    if predicted_label == final_label:
        return "confirmed_correct"
    return "other_label_flip"


def _candidate_thresholds(step: float) -> list[float]:
    """Build a small stable threshold grid from 0.0 to 1.0 inclusive."""

    if step <= 0 or step > 1:
        raise ValueError(f"Threshold step must be in (0, 1], got {step}")
    values: list[float] = []
    current = 0.0
    while current < 1.0:
        values.append(round(current, 10))
        current += step
    values.append(1.0)
    return values


@dataclass(slots=True)
class ConfidenceBandReviewSummary:
    """Reviewer outcome summary for one confidence band."""

    confidence_level: str
    total_cases: int
    adjudicated_cases: int
    confirmed_cases: int
    overridden_cases: int
    false_positive_count: int
    false_negative_count: int
    override_rate: float | None = None
    mean_probability_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ThresholdTuningRecommendation:
    """Advisory threshold suggestion derived from reviewed cases."""

    current_threshold: float
    suggested_threshold: float
    selection_metric: str
    direction: str
    support_sample_count: int
    current_threshold_score: float | None = None
    suggested_threshold_score: float | None = None
    threshold_delta: float = 0.0
    evidence_strength: str = "insufficient"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewLearningSignal:
    """One advisory signal for retraining or threshold review."""

    level: str
    code: str
    message: str
    metric: str | None = None
    value: float | int | None = None
    threshold: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewLearningReport:
    """Advisory learning report derived from resolved reviewer outcomes."""

    generated_at_utc: str
    scope: str
    current_threshold: float
    total_reviews: int
    resolved_reviews: int
    adjudicated_reviews: int
    confirmed_reviews: int
    overridden_reviews: int
    dismissed_reviews: int
    reviewer_labeled_samples: int
    override_rate: float | None
    false_positive_count: int
    false_negative_count: int
    medium_or_high_confidence_overrides: int
    recommended_action: str
    confidence_band_summary: list[ConfidenceBandReviewSummary] = field(default_factory=list)
    threshold_recommendation: ThresholdTuningRecommendation | None = None
    threshold_grid: list[dict[str, Any]] = field(default_factory=list)
    retraining_signals: list[ReviewLearningSignal] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "confidence_band_summary": [item.to_dict() for item in self.confidence_band_summary],
            "threshold_recommendation": None
            if self.threshold_recommendation is None
            else self.threshold_recommendation.to_dict(),
            "retraining_signals": [item.to_dict() for item in self.retraining_signals],
        }


def _build_confidence_band_summary(records: Sequence[ReviewQueueRecord]) -> list[ConfidenceBandReviewSummary]:
    """Aggregate reviewed outcomes by stored confidence band."""

    band_order = {"low": 0, "medium": 1, "high": 2}
    buckets: dict[str, list[ReviewQueueRecord]] = {}
    for record in records:
        level = str(record.confidence_level or "unknown")
        buckets.setdefault(level, []).append(record)

    summaries: list[ConfidenceBandReviewSummary] = []
    for level in sorted(buckets.keys(), key=lambda value: (band_order.get(value, 99), value)):
        bucket_records = buckets[level]
        adjudicated = [record for record in bucket_records if record.status in {"confirmed", "overridden"}]
        confirmed_cases = sum(1 for record in adjudicated if record.status == "confirmed")
        overridden_cases = sum(1 for record in adjudicated if record.status == "overridden")
        false_positive_count = sum(
            1 for record in adjudicated if _classification_direction(record) == "false_positive"
        )
        false_negative_count = sum(
            1 for record in adjudicated if _classification_direction(record) == "false_negative"
        )
        probability_values = [
            probability
            for probability in (_safe_float(record.probability_score) for record in bucket_records)
            if probability is not None
        ]
        summaries.append(
            ConfidenceBandReviewSummary(
                confidence_level=level,
                total_cases=len(bucket_records),
                adjudicated_cases=len(adjudicated),
                confirmed_cases=confirmed_cases,
                overridden_cases=overridden_cases,
                false_positive_count=false_positive_count,
                false_negative_count=false_negative_count,
                override_rate=_rate(overridden_cases, len(adjudicated)),
                mean_probability_score=None
                if not probability_values
                else round(sum(probability_values) / len(probability_values), 6),
            )
        )
    return summaries


def _build_reviewed_frame(records: Sequence[ReviewQueueRecord]) -> pd.DataFrame:
    """Convert adjudicated review records into a threshold-evaluable frame."""

    rows: list[dict[str, Any]] = []
    for record in records:
        if record.status not in {"confirmed", "overridden"}:
            continue
        probability_score = _safe_float(record.probability_score)
        final_label = _resolved_final_label(record)
        payload = record.payload if isinstance(record.payload, dict) else {}
        predicted_label = _safe_int(payload.get("predicted_label"))
        if probability_score is None or final_label is None or predicted_label is None:
            continue
        rows.append(
            {
                "review_id": record.review_id,
                "true_label": final_label,
                "probability_class_1": probability_score,
                "predicted_label": predicted_label,
                "confidence_level": str(record.confidence_level or "unknown"),
                "status": record.status,
                "error_direction": _classification_direction(record),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_threshold_grid(
    reviewed_frame: pd.DataFrame,
    *,
    current_threshold: float,
    selection_metric: str,
    threshold_step: float,
) -> tuple[list[dict[str, Any]], ThresholdTuningRecommendation]:
    """Evaluate a small threshold grid on reviewer-labeled cases."""

    supported_metrics = {
        "balanced_accuracy",
        "accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "precision",
        "auroc",
    }
    if selection_metric not in supported_metrics:
        raise ValueError(
            "Unsupported selection_metric for reviewer learning. "
            "Use one of: accuracy, auroc, balanced_accuracy, f1, precision, sensitivity, specificity."
        )
    if reviewed_frame.empty:
        return [], ThresholdTuningRecommendation(
            current_threshold=current_threshold,
            suggested_threshold=current_threshold,
            selection_metric=selection_metric,
            direction="keep_threshold",
            support_sample_count=0,
            evidence_strength="insufficient",
            note=(
                "No adjudicated review cases with usable probability scores are available yet, "
                "so reviewer-guided threshold tuning would be premature."
            ),
        )

    y_true = [int(value) for value in reviewed_frame["true_label"].tolist()]
    y_score = [float(value) for value in reviewed_frame["probability_class_1"].tolist()]
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    current_row: dict[str, Any] | None = None
    for threshold in _candidate_thresholds(threshold_step):
        y_pred = threshold_binary_scores(y_score, threshold=threshold)
        metrics = compute_binary_classification_metrics(y_true, y_pred, y_score=y_score)
        balanced_accuracy = (metrics["sensitivity"] + metrics["specificity"]) / 2.0
        if selection_metric == "balanced_accuracy":
            selection_score = balanced_accuracy
        else:
            selection_score = float(metrics.get(selection_metric, 0.0))
        row = {
            "threshold": threshold,
            "selection_score": selection_score,
            "balanced_accuracy": balanced_accuracy,
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
            "sensitivity": float(metrics["sensitivity"]),
            "specificity": float(metrics["specificity"]),
            "sample_count": int(metrics["sample_count"]),
        }
        rows.append(row)
        if abs(threshold - current_threshold) < 1e-9:
            current_row = row
        if best_row is None or row["selection_score"] > best_row["selection_score"]:
            best_row = row

    if best_row is None:
        raise ValueError("Threshold grid evaluation unexpectedly produced no rows.")

    if current_row is None:
        current_threshold = round(current_threshold, 10)
        nearest = min(rows, key=lambda row: abs(float(row["threshold"]) - current_threshold))
        current_row = nearest
        current_threshold = float(nearest["threshold"])

    support_count = int(len(reviewed_frame))
    threshold_delta = round(float(best_row["threshold"]) - float(current_row["threshold"]), 6)
    if support_count < 5:
        evidence_strength = "insufficient"
    elif support_count < 10:
        evidence_strength = "limited"
    else:
        evidence_strength = "moderate"

    if abs(threshold_delta) < max(threshold_step, 0.05) or evidence_strength == "insufficient":
        direction = "keep_threshold"
    elif threshold_delta > 0:
        direction = "raise_threshold"
    else:
        direction = "lower_threshold"

    note = (
        "Suggestion is computed only on reviewed cases, which are selection-biased toward uncertain failures. "
        "Use this as a cue for validation-only recalibration, not as a direct production threshold change."
    )
    return rows, ThresholdTuningRecommendation(
        current_threshold=float(current_row["threshold"]),
        suggested_threshold=float(best_row["threshold"]),
        selection_metric=selection_metric,
        direction=direction,
        support_sample_count=support_count,
        current_threshold_score=float(current_row["selection_score"]),
        suggested_threshold_score=float(best_row["selection_score"]),
        threshold_delta=threshold_delta,
        evidence_strength=evidence_strength,
        note=note,
    )


def _build_learning_signals(
    *,
    adjudicated_reviews: int,
    override_rate: float | None,
    false_positive_count: int,
    false_negative_count: int,
    medium_or_high_confidence_overrides: int,
    threshold_recommendation: ThresholdTuningRecommendation,
    confidence_band_summary: Sequence[ConfidenceBandReviewSummary],
) -> list[ReviewLearningSignal]:
    """Generate advisory retraining and threshold-review signals."""

    signals: list[ReviewLearningSignal] = []
    if adjudicated_reviews < 5:
        signals.append(
            ReviewLearningSignal(
                level="info",
                code="limited_feedback_volume",
                metric="adjudicated_reviews",
                value=adjudicated_reviews,
                threshold=5,
                message="Reviewer learning is still based on a small number of adjudicated cases.",
            )
        )
    if override_rate is not None and adjudicated_reviews >= 8 and override_rate >= 0.30:
        signals.append(
            ReviewLearningSignal(
                level="warning",
                code="high_override_rate",
                metric="override_rate",
                value=round(override_rate, 4),
                threshold=0.30,
                message="Reviewer overrides remain high enough to justify a focused failure-pattern review.",
            )
        )
    if false_negative_count >= 3 and false_negative_count > false_positive_count:
        signals.append(
            ReviewLearningSignal(
                level="warning",
                code="false_negative_pattern",
                metric="false_negative_count",
                value=false_negative_count,
                threshold=3,
                message="Reviewer outcomes lean toward false negatives; sensitivity and threshold choice should be re-checked.",
            )
        )
    if false_positive_count >= 3 and false_positive_count > false_negative_count:
        signals.append(
            ReviewLearningSignal(
                level="warning",
                code="false_positive_pattern",
                metric="false_positive_count",
                value=false_positive_count,
                threshold=3,
                message="Reviewer outcomes lean toward false positives; specificity and threshold choice should be re-checked.",
            )
        )
    if medium_or_high_confidence_overrides >= 3:
        signals.append(
            ReviewLearningSignal(
                level="warning",
                code="confident_override_pattern",
                metric="medium_or_high_confidence_overrides",
                value=medium_or_high_confidence_overrides,
                threshold=3,
                message="Several overridden cases were not low-confidence escalations, which is a stronger retraining signal than low-confidence misses alone.",
            )
        )
    if (
        threshold_recommendation.direction != "keep_threshold"
        and threshold_recommendation.evidence_strength in {"limited", "moderate"}
    ):
        signals.append(
            ReviewLearningSignal(
                level="info",
                code="threshold_recalibration_candidate",
                metric="threshold_delta",
                value=threshold_recommendation.threshold_delta,
                threshold=0.05,
                message="Reviewed outcomes suggest a different operating threshold may deserve validation-only recalibration testing.",
            )
        )
    for summary in confidence_band_summary:
        if summary.override_rate is not None and summary.adjudicated_cases >= 3 and summary.override_rate >= 0.50:
            signals.append(
                ReviewLearningSignal(
                    level="warning",
                    code=f"{summary.confidence_level}_band_instability",
                    metric="override_rate",
                    value=round(summary.override_rate, 4),
                    threshold=0.50,
                    message=(
                        f"Reviewer overrides are high in the {summary.confidence_level} confidence band; "
                        "inspect that band before trusting it operationally."
                    ),
                )
            )
    return signals


def analyze_review_learning(
    records: Sequence[ReviewQueueRecord],
    *,
    model_name: str | None = None,
    active_model_id: str | None = None,
    run_name: str | None = None,
    current_threshold: float = 0.5,
    selection_metric: str = "balanced_accuracy",
    threshold_step: float = 0.05,
) -> ReviewLearningReport:
    """Build an advisory learning report from reviewed outcomes."""

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
    total_reviews = len(selected_records)
    resolved_reviews = sum(1 for record in selected_records if record.status != "pending")
    confirmed_reviews = sum(1 for record in selected_records if record.status == "confirmed")
    overridden_reviews = sum(1 for record in selected_records if record.status == "overridden")
    dismissed_reviews = sum(1 for record in selected_records if record.status == "dismissed")
    adjudicated_reviews = confirmed_reviews + overridden_reviews
    override_rate = _rate(overridden_reviews, adjudicated_reviews)

    reviewed_frame = _build_reviewed_frame(selected_records)
    false_positive_count = int((reviewed_frame["error_direction"] == "false_positive").sum()) if not reviewed_frame.empty else 0
    false_negative_count = int((reviewed_frame["error_direction"] == "false_negative").sum()) if not reviewed_frame.empty else 0
    medium_or_high_confidence_overrides = int(
        (
            (reviewed_frame["status"] == "overridden")
            & (reviewed_frame["confidence_level"].isin(["medium", "high"]))
        ).sum()
    ) if not reviewed_frame.empty else 0

    confidence_band_summary = _build_confidence_band_summary(
        [record for record in selected_records if record.status != "pending"]
    )
    threshold_grid, threshold_recommendation = _evaluate_threshold_grid(
        reviewed_frame,
        current_threshold=current_threshold,
        selection_metric=selection_metric,
        threshold_step=threshold_step,
    )
    retraining_signals = _build_learning_signals(
        adjudicated_reviews=adjudicated_reviews,
        override_rate=override_rate,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        medium_or_high_confidence_overrides=medium_or_high_confidence_overrides,
        threshold_recommendation=threshold_recommendation,
        confidence_band_summary=confidence_band_summary,
    )

    if any(signal.level == "warning" for signal in retraining_signals):
        recommended_action = (
            "Investigate reviewer failure patterns before any promotion or threshold change; use the threshold suggestion only as validation-only input."
        )
    elif threshold_recommendation.direction != "keep_threshold" and threshold_recommendation.evidence_strength != "insufficient":
        recommended_action = (
            "Run a validation-only threshold recalibration experiment using the reviewer-guided suggestion, then compare against the current approved setting."
        )
    else:
        recommended_action = "Keep the current threshold and continue collecting reviewed cases for a stronger feedback signal."

    notes = [
        "Reviewer feedback is selection-biased toward queued cases and should not replace held-out validation or external evaluation.",
        "Any threshold suggestion here is advisory only and should be re-tested on validation data before changing the active model.",
    ]
    if not selected_records:
        notes.append("No review records matched the requested scope yet.")
    if reviewed_frame.empty:
        notes.append("No adjudicated review cases contained enough data for threshold-learning analysis.")
    elif len(reviewed_frame) < 10:
        notes.append("Reviewer-labeled sample count is still small, so tuning and retraining signals should be treated cautiously.")

    report = ReviewLearningReport(
        generated_at_utc=_utc_now(),
        scope=scope,
        current_threshold=current_threshold,
        total_reviews=total_reviews,
        resolved_reviews=resolved_reviews,
        adjudicated_reviews=adjudicated_reviews,
        confirmed_reviews=confirmed_reviews,
        overridden_reviews=overridden_reviews,
        dismissed_reviews=dismissed_reviews,
        reviewer_labeled_samples=int(len(reviewed_frame)),
        override_rate=override_rate,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        medium_or_high_confidence_overrides=medium_or_high_confidence_overrides,
        recommended_action=recommended_action,
        confidence_band_summary=confidence_band_summary,
        threshold_recommendation=threshold_recommendation,
        threshold_grid=threshold_grid,
        retraining_signals=retraining_signals,
        notes=notes,
    )
    LOGGER.debug("Built review learning report for scope %s with %d records", scope, total_reviews)
    return report


def summarize_review_learning(
    records: Iterable[ReviewQueueRecord],
    *,
    model_name: str | None = None,
    active_model_id: str | None = None,
    run_name: str | None = None,
    current_threshold: float = 0.5,
    selection_metric: str = "balanced_accuracy",
    threshold_step: float = 0.05,
) -> dict[str, Any]:
    """Convenience wrapper returning a JSON-safe review learning payload."""

    return analyze_review_learning(
        list(records),
        model_name=model_name,
        active_model_id=active_model_id,
        run_name=run_name,
        current_threshold=current_threshold,
        selection_metric=selection_metric,
        threshold_step=threshold_step,
    ).to_dict()
