"""Tests for advisory reviewer-outcome learning and threshold guidance."""

from __future__ import annotations

from types import SimpleNamespace

from src.api.services import build_review_learning_payload
from src.models.review_learning import analyze_review_learning
from src.models.registry import ModelRegistryEntry
from src.storage import ReviewQueueRecord


def _resolution(
    *,
    action: str,
    reviewer_id: str,
    resolved_label: int | None = None,
    resolved_label_name: str | None = None,
) -> dict[str, object]:
    return {
        "action": action,
        "reviewer_id": reviewer_id,
        "resolved_label": resolved_label,
        "resolved_label_name": resolved_label_name,
        "final_status": {
            "confirm_model_output": "confirmed",
            "override_prediction": "overridden",
            "dismiss": "dismissed",
        }[action],
    }


def _sample_learning_records() -> list[ReviewQueueRecord]:
    """Return reviewer outcomes that suggest a slightly higher threshold."""

    def _record(
        review_id: str,
        *,
        probability_score: float,
        predicted_label: int,
        status: str,
        confidence_level: str,
        resolution: dict[str, object] | None = None,
    ) -> ReviewQueueRecord:
        payload = {
            "predicted_label": predicted_label,
            "label_name": "demented" if predicted_label == 1 else "nondemented",
            "active_model_id": "oasis_current_baseline",
            "model_run_name": "run42",
        }
        if resolution is not None:
            payload["resolution"] = resolution
        return ReviewQueueRecord(
            review_id=review_id,
            inference_id=f"inf-{review_id}",
            trace_id=f"trace-{review_id}",
            model_name="densenet121_3d",
            confidence_level=confidence_level,
            probability_score=probability_score,
            status=status,
            payload=payload,
        )

    return [
        _record(
            "confirmed-neg-1",
            probability_score=0.10,
            predicted_label=0,
            status="confirmed",
            confidence_level="high",
            resolution=_resolution(action="confirm_model_output", reviewer_id="reviewer_a"),
        ),
        _record(
            "confirmed-neg-2",
            probability_score=0.20,
            predicted_label=0,
            status="confirmed",
            confidence_level="medium",
            resolution=_resolution(action="confirm_model_output", reviewer_id="reviewer_b"),
        ),
        _record(
            "confirmed-pos-1",
            probability_score=0.70,
            predicted_label=1,
            status="confirmed",
            confidence_level="medium",
            resolution=_resolution(action="confirm_model_output", reviewer_id="reviewer_c"),
        ),
        _record(
            "confirmed-pos-2",
            probability_score=0.82,
            predicted_label=1,
            status="confirmed",
            confidence_level="high",
            resolution=_resolution(action="confirm_model_output", reviewer_id="reviewer_a"),
        ),
        _record(
            "fp-1",
            probability_score=0.52,
            predicted_label=1,
            status="overridden",
            confidence_level="medium",
            resolution=_resolution(
                action="override_prediction",
                reviewer_id="reviewer_a",
                resolved_label=0,
                resolved_label_name="nondemented",
            ),
        ),
        _record(
            "fp-2",
            probability_score=0.56,
            predicted_label=1,
            status="overridden",
            confidence_level="medium",
            resolution=_resolution(
                action="override_prediction",
                reviewer_id="reviewer_b",
                resolved_label=0,
                resolved_label_name="nondemented",
            ),
        ),
        _record(
            "fp-3",
            probability_score=0.58,
            predicted_label=1,
            status="overridden",
            confidence_level="high",
            resolution=_resolution(
                action="override_prediction",
                reviewer_id="reviewer_c",
                resolved_label=0,
                resolved_label_name="nondemented",
            ),
        ),
        _record(
            "pending",
            probability_score=0.54,
            predicted_label=1,
            status="pending",
            confidence_level="low",
        ),
    ]


def test_analyze_review_learning_builds_threshold_guidance_and_failure_signals() -> None:
    """Learning report should surface threshold suggestions and retraining patterns."""

    report = analyze_review_learning(
        _sample_learning_records(),
        model_name="densenet121_3d",
        active_model_id="oasis_current_baseline",
        run_name="run42",
        current_threshold=0.5,
        selection_metric="balanced_accuracy",
        threshold_step=0.05,
    )

    assert report.total_reviews == 8
    assert report.resolved_reviews == 7
    assert report.adjudicated_reviews == 7
    assert report.override_rate == 3 / 7
    assert report.false_positive_count == 3
    assert report.false_negative_count == 0
    assert report.medium_or_high_confidence_overrides == 3
    assert report.threshold_recommendation is not None
    assert report.threshold_recommendation.direction == "raise_threshold"
    assert report.threshold_recommendation.suggested_threshold > 0.5
    assert report.threshold_recommendation.evidence_strength == "limited"
    assert any(signal.code == "false_positive_pattern" for signal in report.retraining_signals)
    assert any(signal.code == "threshold_recalibration_candidate" for signal in report.retraining_signals)
    assert "validation-only" in report.recommended_action.lower()


def test_build_review_learning_payload_can_scope_to_active_model(monkeypatch) -> None:
    """Service payload should use the active model threshold when requested."""

    records = _sample_learning_records()
    monkeypatch.setattr("src.api.services.list_review_records", lambda **_: records)
    monkeypatch.setattr(
        "src.api.services.load_backend_serving_config",
        lambda *_, **__: SimpleNamespace(active_oasis_model_registry="ignored.json"),
    )
    monkeypatch.setattr(
        "src.api.services.load_current_oasis_model_entry",
        lambda *_args, **_kwargs: ModelRegistryEntry(
            registry_version="1.0",
            model_id="oasis_current_baseline",
            dataset="oasis1",
            run_name="run42",
            checkpoint_path="checkpoint.pt",
            model_config_path="configs/oasis_model.yaml",
            preprocessing_config_path="configs/oasis_transforms.yaml",
            image_size=[64, 64, 64],
            promoted_at_utc="2026-01-01T00:00:00+00:00",
            decision_support_only=True,
            clinical_disclaimer="Decision-support only.",
            recommended_threshold=0.45,
        ),
    )
    monkeypatch.setattr(
        "src.api.services.load_oasis_model_config",
        lambda *_args, **_kwargs: SimpleNamespace(architecture="densenet121_3d"),
    )

    payload = build_review_learning_payload(active_model_only=True, limit=50)

    assert payload["scope"].startswith("active_model_id=oasis_current_baseline")
    assert payload["current_threshold"] == 0.45
    assert payload["threshold_recommendation"]["direction"] == "raise_threshold"
    assert payload["threshold_recommendation"]["suggested_threshold"] >= 0.5
