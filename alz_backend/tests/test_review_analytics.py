"""Tests for post-promotion review analytics and monitoring helpers."""

from __future__ import annotations

from types import SimpleNamespace

from src.api.services import build_review_analytics_payload
from src.models.review_analytics import analyze_review_records
from src.models.registry import ModelRegistryEntry
from src.storage import ReviewQueueRecord


def _resolution(
    *,
    action: str,
    reviewer_id: str,
    resolved_label: int | None = None,
    resolved_label_name: str | None = None,
) -> dict[str, object]:
    """Build a minimal stored resolution payload."""

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


def _sample_review_records() -> list[ReviewQueueRecord]:
    """Return mixed review records for analytics tests."""

    return [
        ReviewQueueRecord(
            review_id="review-pending",
            inference_id="inf-pending",
            trace_id="trace-pending",
            model_name="densenet121_3d",
            confidence_level="low",
            probability_score=0.55,
            status="pending",
            payload={
                "predicted_label": 1,
                "label_name": "demented",
                "active_model_id": "oasis_current_baseline",
                "model_run_name": "run42",
            },
        ),
        ReviewQueueRecord(
            review_id="review-fp",
            inference_id="inf-fp",
            trace_id="trace-fp",
            model_name="densenet121_3d",
            confidence_level="low",
            probability_score=0.56,
            status="overridden",
            payload={
                "predicted_label": 1,
                "label_name": "demented",
                "active_model_id": "oasis_current_baseline",
                "model_run_name": "run42",
                "resolution": _resolution(
                    action="override_prediction",
                    reviewer_id="reviewer_a",
                    resolved_label=0,
                    resolved_label_name="nondemented",
                ),
            },
        ),
        ReviewQueueRecord(
            review_id="review-fn",
            inference_id="inf-fn",
            trace_id="trace-fn",
            model_name="densenet121_3d",
            confidence_level="medium",
            probability_score=0.44,
            status="overridden",
            payload={
                "predicted_label": 0,
                "label_name": "nondemented",
                "active_model_id": "oasis_current_baseline",
                "model_run_name": "run42",
                "resolution": _resolution(
                    action="override_prediction",
                    reviewer_id="reviewer_b",
                    resolved_label=1,
                    resolved_label_name="demented",
                ),
            },
        ),
        ReviewQueueRecord(
            review_id="review-confirm-1",
            inference_id="inf-confirm-1",
            trace_id="trace-confirm-1",
            model_name="densenet121_3d",
            confidence_level="low",
            probability_score=0.62,
            status="confirmed",
            payload={
                "predicted_label": 1,
                "label_name": "demented",
                "active_model_id": "oasis_current_baseline",
                "model_run_name": "run42",
                "resolution": _resolution(
                    action="confirm_model_output",
                    reviewer_id="reviewer_a",
                ),
            },
        ),
        ReviewQueueRecord(
            review_id="review-confirm-2",
            inference_id="inf-confirm-2",
            trace_id="trace-confirm-2",
            model_name="densenet121_3d",
            confidence_level="low",
            probability_score=0.68,
            status="confirmed",
            payload={
                "predicted_label": 1,
                "label_name": "demented",
                "active_model_id": "oasis_current_baseline",
                "model_run_name": "run42",
                "resolution": _resolution(
                    action="confirm_model_output",
                    reviewer_id="reviewer_c",
                ),
            },
        ),
        ReviewQueueRecord(
            review_id="review-confirm-3",
            inference_id="inf-confirm-3",
            trace_id="trace-confirm-3",
            model_name="densenet121_3d",
            confidence_level="medium",
            probability_score=0.73,
            status="confirmed",
            payload={
                "predicted_label": 1,
                "label_name": "demented",
                "active_model_id": "oasis_current_baseline",
                "model_run_name": "run42",
                "resolution": _resolution(
                    action="confirm_model_output",
                    reviewer_id="reviewer_b",
                ),
            },
        ),
        ReviewQueueRecord(
            review_id="review-other-model",
            inference_id="inf-other-model",
            trace_id="trace-other-model",
            model_name="custom3d",
            confidence_level="low",
            probability_score=0.51,
            status="overridden",
            payload={
                "predicted_label": 1,
                "label_name": "demented",
                "active_model_id": "legacy_model",
                "model_run_name": "legacy_run",
                "resolution": _resolution(
                    action="override_prediction",
                    reviewer_id="reviewer_z",
                    resolved_label=0,
                    resolved_label_name="nondemented",
                ),
            },
        ),
    ]


def test_analyze_review_records_reports_override_patterns_for_active_model_scope() -> None:
    """Analytics should summarize overrides, directions, and governance warnings."""

    summary = analyze_review_records(
        _sample_review_records(),
        model_name="densenet121_3d",
        active_model_id="oasis_current_baseline",
        run_name="run42",
    )

    assert summary.total_reviews == 6
    assert summary.pending_reviews == 1
    assert summary.adjudicated_reviews == 5
    assert summary.override_rate == 0.4
    assert summary.error_breakdown["false_positive"] == 1
    assert summary.error_breakdown["false_negative"] == 1
    assert summary.label_override_pairs["demented -> nondemented"] == 1
    assert summary.label_override_pairs["nondemented -> demented"] == 1
    assert summary.high_risk is True
    assert any(signal.code == "high_override_rate" for signal in summary.risk_signals)
    assert summary.model_breakdown[0].model_name == "densenet121_3d"
    assert summary.reviewer_agreement_available is False


def test_build_review_analytics_payload_can_scope_to_active_model(monkeypatch) -> None:
    """Service analytics should filter to the active model when requested."""

    records = _sample_review_records()
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
        ),
    )
    monkeypatch.setattr(
        "src.api.services.load_oasis_model_config",
        lambda *_args, **_kwargs: SimpleNamespace(architecture="densenet121_3d"),
    )

    payload = build_review_analytics_payload(active_model_only=True, limit=50)

    assert payload["scope"].startswith("active_model_id=oasis_current_baseline")
    assert payload["total_reviews"] == 6
    assert payload["overridden_reviews"] == 2
    assert payload["high_risk"] is True
