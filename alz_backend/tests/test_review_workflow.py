"""Tests for review-queue resolution workflow."""

from __future__ import annotations

from pathlib import Path

from src.api.schemas import ReviewResolutionRequest
from src.api.services import resolve_review_queue_item_payload
from src.configs.runtime import AppSettings
from src.storage import ReviewQueueRecord, get_review_record, persist_review_record


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


def test_resolve_review_queue_item_payload_updates_status_and_audits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Resolving a review item should persist status, resolution metadata, and audit context."""

    settings = _build_settings(tmp_path)
    persist_review_record(
        ReviewQueueRecord(
            review_id="review-1",
            inference_id="inf-1",
            trace_id="trace-1",
            subject_id="OAS1_0001",
            source_path="scan.nii.gz",
            model_name="densenet121_3d",
            confidence_level="low",
            probability_score=0.54,
            payload={
                "predicted_label": 1,
                "label_name": "demented",
                "review_flag": True,
            },
        ),
        settings=settings,
    )
    audit_calls: list[dict[str, object]] = []
    monkeypatch.setattr("src.api.services.get_app_settings", lambda: settings)
    monkeypatch.setattr("src.api.services.audit_sensitive_action", lambda **kwargs: audit_calls.append(kwargs))
    hold_assessment_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.api.services.assess_active_oasis_model_hold",
        lambda **kwargs: hold_assessment_calls.append(kwargs),
    )

    response = resolve_review_queue_item_payload(
        "review-1",
        ReviewResolutionRequest(
            reviewer_id="reviewer_a",
            action="override_prediction",
            resolved_label=0,
            resolution_note="False positive after source-image review.",
        ),
    )

    updated = get_review_record("review-1", settings=settings)
    assert updated is not None
    assert updated.status == "overridden"
    assert updated.payload["resolution"]["reviewer_id"] == "reviewer_a"
    assert updated.payload["resolution"]["resolved_label"] == 0
    assert updated.payload["resolution"]["resolved_label_name"] == "nondemented"
    assert response["status"] == "overridden"
    assert response["item"]["resolution"]["final_status"] == "overridden"
    assert hold_assessment_calls
    assert hold_assessment_calls[0]["actor_id"] == "reviewer_a"
    assert audit_calls
    assert audit_calls[0]["action"] == "resolve_review_case"
