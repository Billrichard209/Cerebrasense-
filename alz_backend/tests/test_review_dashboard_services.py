"""Tests for reviewer dashboard service helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.api.services import (
    build_hold_history_payload,
    build_review_detail_payload,
    build_resolved_review_queue_payload,
    build_review_dashboard_payload,
)
from src.configs.runtime import AppSettings
from src.storage import ReviewQueueRecord, persist_review_record


def _build_settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    storage_root = project_root / "storage"
    data_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)
    storage_root.mkdir(parents=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=project_root.parent,
        oasis_source_root=project_root.parent / "OASIS",
        storage_root=storage_root,
        database_path=storage_root / "backend.sqlite3",
    )


def test_build_resolved_review_queue_payload_excludes_pending_items(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Resolved review payloads should only return non-pending review states."""

    settings = _build_settings(tmp_path)
    persist_review_record(
        ReviewQueueRecord(
            review_id="review-pending",
            inference_id="inf-pending",
            trace_id="trace-pending",
            model_name="densenet121_3d",
            status="pending",
            payload={"review_flag": True},
        ),
        settings=settings,
    )
    persist_review_record(
        ReviewQueueRecord(
            review_id="review-confirmed",
            inference_id="inf-confirmed",
            trace_id="trace-confirmed",
            model_name="densenet121_3d",
            status="confirmed",
            payload={"review_flag": False, "resolution": {"reviewer_id": "reviewer_a"}},
        ),
        settings=settings,
    )
    monkeypatch.setattr("src.api.services.get_app_settings", lambda: settings)

    payload = build_resolved_review_queue_payload(limit=10)

    assert payload["total"] == 1
    assert payload["items"][0]["review_id"] == "review-confirmed"
    assert payload["items"][0]["status"] == "confirmed"


def test_build_review_detail_payload_returns_one_review_case(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Review detail payload should include stored resolution metadata for one case."""

    settings = _build_settings(tmp_path)
    persist_review_record(
        ReviewQueueRecord(
            review_id="review-detail",
            inference_id="inf-detail",
            trace_id="trace-detail",
            model_name="densenet121_3d",
            status="confirmed",
            payload={
                "review_flag": False,
                "resolution": {"reviewer_id": "reviewer_a", "action": "confirm_model_output"},
            },
        ),
        settings=settings,
    )
    monkeypatch.setattr("src.api.services.get_app_settings", lambda: settings)

    payload = build_review_detail_payload("review-detail")

    assert payload["review_id"] == "review-detail"
    assert payload["status"] == "confirmed"
    assert payload["resolution"]["reviewer_id"] == "reviewer_a"


def test_build_hold_history_payload_reads_recent_history_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Hold-history payload should parse saved hold assessment artifacts."""

    settings = _build_settings(tmp_path)
    history_root = settings.outputs_root / "model_registry" / "hold_history"
    history_root.mkdir(parents=True, exist_ok=True)
    history_path = history_root / "unit_run_20260411T000000.json"
    history_path.write_text(
        json.dumps(
            {
                "decision": {
                    "assessed_at_utc": "2026-04-11T00:00:00+00:00",
                    "policy_name": "oasis_operational_hold_v1",
                    "operational_status": "hold",
                    "hold_applied": True,
                    "status_changed": True,
                    "trigger_codes": ["override_rate_threshold"],
                    "summary": "Active model placed on hold.",
                },
                "review_monitoring": {
                    "high_risk": True,
                    "total_reviews": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.api.services.get_app_settings", lambda: settings)

    payload = build_hold_history_payload(limit=5)

    assert payload["total"] == 1
    assert payload["items"][0]["operational_status"] == "hold"
    assert payload["items"][0]["hold_applied"] is True
    assert payload["items"][0]["trigger_codes"] == ["override_rate_threshold"]


def test_build_review_dashboard_payload_combines_active_model_queue_and_history(monkeypatch) -> None:
    """Dashboard payload should compose the key reviewer-operations blocks."""

    monkeypatch.setattr(
        "src.api.services.build_active_oasis_model_payload",
        lambda: {
            "model_id": "oasis_current_baseline",
            "dataset": "oasis1",
            "run_name": "unit_run",
            "checkpoint_path": "checkpoint.pt",
            "promoted_at_utc": "2026-01-01T00:00:00+00:00",
            "decision_support_only": True,
            "clinical_disclaimer": "Decision-support only.",
            "default_threshold": 0.5,
            "recommended_threshold": 0.45,
            "threshold_calibration": {},
            "temperature_scaling": {},
            "confidence_policy": {},
            "approval_status": "approved",
            "operational_status": "active",
            "benchmark": {},
            "promotion_decision": {},
            "evidence": {},
            "validation_metrics": {},
            "test_metrics": {},
            "serving_restrictions": {"force_manual_review": False},
            "hold_decision": {},
            "review_monitoring": {},
            "notes": [],
        },
    )
    monkeypatch.setattr("src.api.services.build_pending_review_queue_payload", lambda limit=10: {"total": 2, "items": []})
    monkeypatch.setattr("src.api.services.build_resolved_review_queue_payload", lambda limit=10, status=None: {"total": 5, "items": []})
    monkeypatch.setattr(
        "src.api.services.build_review_analytics_payload",
        lambda limit=200, model_name=None, active_model_only=True: {
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "scope": "active_model_id=oasis_current_baseline",
            "total_reviews": 7,
            "pending_reviews": 2,
            "resolved_reviews": 5,
            "adjudicated_reviews": 5,
            "overridden_reviews": 1,
            "confirmed_reviews": 4,
            "dismissed_reviews": 0,
            "override_rate": 0.2,
            "confirmation_rate": 0.8,
            "status_counts": {},
            "action_counts": {},
            "reviewer_counts": {},
            "confidence_level_counts": {},
            "error_breakdown": {},
            "error_confidence_distribution": {},
            "label_override_pairs": {},
            "model_breakdown": [],
            "average_error_probability_score": None,
            "high_risk": False,
            "reviewer_agreement_available": False,
            "reviewer_agreement_note": "Single-review resolution workflow only.",
            "risk_signals": [],
            "notes": [],
        },
    )
    monkeypatch.setattr("src.api.services.build_hold_history_payload", lambda limit=10: {"total": 3, "items": []})

    payload = build_review_dashboard_payload()

    assert payload["summary"]["pending_reviews"] == 2
    assert payload["summary"]["resolved_reviews"] == 5
    assert payload["summary"]["hold_history_entries"] == 3
    assert payload["summary"]["operational_status"] == "active"
