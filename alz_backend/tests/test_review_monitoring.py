"""Tests for operational hold evaluation from review-monitoring evidence."""

from __future__ import annotations

from pathlib import Path

from src.configs.runtime import AppSettings
from src.models.registry import ModelRegistryEntry, save_oasis_model_entry
from src.models.review_monitoring import assess_active_oasis_model_hold
from src.storage import ReviewQueueRecord, persist_review_record


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=project_root / "data",
        outputs_root=project_root / "outputs",
        oasis_source_root=project_root.parent / "OASIS",
        kaggle_source_root=project_root.parent / "archive (1)",
        storage_root=project_root / "storage",
        database_path=project_root / "storage" / "backend.sqlite3",
    )


def _resolution(action: str, reviewer_id: str, resolved_label: int | None = None, resolved_label_name: str | None = None) -> dict[str, object]:
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


def test_assess_active_oasis_model_hold_updates_registry_when_override_rate_is_high(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """High override evidence should move the active registry entry into hold status."""

    settings = _settings(tmp_path)
    settings.outputs_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    registry_entry = ModelRegistryEntry(
        registry_version="1.0",
        model_id="oasis_current_baseline",
        dataset="oasis1",
        run_name="run42",
        checkpoint_path=str(checkpoint_path),
        model_config_path="configs/oasis_model.yaml",
        preprocessing_config_path="configs/oasis_transforms.yaml",
        image_size=[64, 64, 64],
        promoted_at_utc="2026-01-01T00:00:00+00:00",
        decision_support_only=True,
        clinical_disclaimer="Decision-support only.",
    )
    save_oasis_model_entry(registry_entry, settings=settings)
    monkeypatch.setattr(
        "src.models.review_monitoring.load_oasis_model_config",
        lambda *_args, **_kwargs: type("Cfg", (), {"architecture": "densenet121_3d"})(),
    )
    audit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.models.review_monitoring.audit_sensitive_action",
        lambda **kwargs: audit_calls.append(kwargs),
    )

    records = [
        ReviewQueueRecord(
            review_id=f"review-{index}",
            inference_id=f"inf-{index}",
            trace_id=f"trace-{index}",
            model_name="densenet121_3d",
            confidence_level="low",
            probability_score=0.55 + (index * 0.01),
            status="overridden" if index < 3 else "confirmed",
            payload={
                "predicted_label": 1,
                "label_name": "demented",
                "active_model_id": "oasis_current_baseline",
                "model_run_name": "run42",
                "resolution": _resolution(
                    "override_prediction" if index < 3 else "confirm_model_output",
                    reviewer_id=f"reviewer_{index}",
                    resolved_label=0 if index < 3 else None,
                    resolved_label_name="nondemented" if index < 3 else None,
                ),
            },
        )
        for index in range(5)
    ]
    for record in records:
        persist_review_record(record, settings=settings)

    result = assess_active_oasis_model_hold(actor_id="reviewer_ops", settings=settings)

    assert result.registry_entry.operational_status == "hold"
    assert result.registry_entry.serving_restrictions["force_manual_review"] is True
    assert result.decision.hold_applied is True
    assert "override_rate_threshold" in result.decision.trigger_codes
    assert result.history_path.exists()
    assert audit_calls
    assert audit_calls[0]["action"] == "update_model_operational_hold"
