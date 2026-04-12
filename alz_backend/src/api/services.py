"""Service helpers that connect API routes with backend business modules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.configs.runtime import get_app_settings
from src.data.registry import build_dataset_registry_snapshot
from src.explainability.gradcam import ExplainScanConfig, explain_scan
from src.inference.serving import load_backend_serving_config
from src.inference.pipeline import PredictScanOptions, predict_scan
from src.longitudinal.service import build_and_save_longitudinal_report
from src.longitudinal.structural import (
    build_oasis_structural_longitudinal_summary,
    save_structural_longitudinal_report,
)
from src.longitudinal.tracker import LongitudinalRecord, TrendFeatureConfig
from src.models.factory import OASIS_BINARY_CLASS_NAMES, describe_model_config, load_oasis_model_config
from src.models.promotion_workflow import (
    load_promotion_candidates,
    load_promotion_history_entries,
    load_promotion_studies,
)
from src.models.review_analytics import summarize_review_records
from src.models.review_learning import summarize_review_learning
from src.models.review_monitoring import assess_active_oasis_model_hold
from src.models.registry import load_current_oasis_model_entry
from src.models.validation_depth import build_validation_depth_dashboard, load_validation_depth_studies
from src.security.audit import audit_sensitive_action
from src.security.disclaimers import STANDARD_DECISION_SUPPORT_DISCLAIMER
from src.security.governance import get_policy_snapshot
from src.storage import get_review_record, list_review_records, persist_review_record
from src.utils.io_utils import ensure_directory
from src.volumetrics.service import analyze_oasis_volume

from .schemas import (
    LongitudinalReportRequest,
    ReviewResolutionRequest,
    ScanExplanationRequest,
    ScanPredictionRequest,
)

RESOLVED_REVIEW_STATUSES = frozenset({"confirmed", "overridden", "dismissed"})


def build_root_payload() -> dict[str, str]:
    """Return the root endpoint payload."""

    settings = get_app_settings()
    return {
        "message": "Structural MRI backend core is running.",
        "mode": "decision_support",
        "primary_dataset": settings.primary_dataset,
    }


def build_health_payload() -> dict[str, object]:
    """Return the health endpoint payload."""

    settings = get_app_settings()
    return {
        "status": "ok",
        "primary_dataset": settings.primary_dataset,
        "decision_support_only": settings.decision_support_only,
    }


def build_policy_payload() -> dict[str, object]:
    """Return the active project policy."""

    return get_policy_snapshot()


def build_dataset_registry_payload() -> dict[str, object]:
    """Return the dataset registry snapshot for API serialization."""

    return build_dataset_registry_snapshot(get_app_settings())


def build_model_metadata_payload(*, model_config_path: str | None = None) -> dict[str, object]:
    """Return active model-factory metadata for API clients."""

    model_config = load_oasis_model_config(Path(model_config_path) if model_config_path is not None else None)
    payload = describe_model_config(model_config)
    serving_config = load_backend_serving_config(settings=get_app_settings())
    try:
        registry_entry = load_current_oasis_model_entry(serving_config.active_oasis_model_registry)
    except FileNotFoundError:
        registry_entry = None
    payload["active_model_id"] = None if registry_entry is None else registry_entry.model_id
    payload["active_checkpoint_path"] = None if registry_entry is None else registry_entry.checkpoint_path
    payload["recommended_threshold"] = None if registry_entry is None else registry_entry.recommended_threshold
    payload["default_threshold"] = None if registry_entry is None else registry_entry.default_threshold
    payload["serving_policy"] = serving_config.to_dict()
    payload["decision_support_only"] = True
    payload["medical_wording"] = STANDARD_DECISION_SUPPORT_DISCLAIMER
    return payload


def build_active_oasis_model_payload() -> dict[str, object]:
    """Return the active registry entry with promotion and benchmark evidence."""

    serving_config = load_backend_serving_config(settings=get_app_settings())
    entry = load_current_oasis_model_entry(serving_config.active_oasis_model_registry)
    payload = entry.to_dict()
    review_monitoring = _build_active_review_monitoring_payload(entry)
    if review_monitoring.get("high_risk"):
        notes = list(payload.get("notes", []))
        warning_note = (
            "Post-promotion review monitoring has flagged elevated risk patterns; "
            "re-check calibration and held-out evaluation before further rollout."
        )
        if warning_note not in notes:
            notes.append(warning_note)
        payload["notes"] = notes
    if payload.get("operational_status") == "hold":
        notes = list(payload.get("notes", []))
        hold_note = (
            "Operational status is currently on hold; predictions may still be generated "
            "for review workflows but should not be treated as the default operating model."
        )
        if hold_note not in notes:
            notes.append(hold_note)
        payload["notes"] = notes
    payload["approval_status"] = (
        "approved"
        if bool(entry.promotion_decision.get("approved"))
        else ("legacy_active" if not entry.promotion_decision else "rejected")
    )
    payload["review_monitoring"] = review_monitoring
    return payload


def _review_record_payload(record) -> dict[str, object]:
    """Normalize a review record for API serialization."""

    payload = record.to_payload()
    nested_payload = dict(payload.get("payload", {}))
    payload["payload"] = nested_payload
    payload["resolution"] = dict(nested_payload.get("resolution", {}))
    return payload


def build_pending_review_queue_payload(*, limit: int = 20) -> dict[str, object]:
    """Return recent pending review-queue items for human follow-up."""

    items = _list_review_payloads(limit=limit, status="pending")
    return {
        "total": len(items),
        "items": items,
    }


def build_review_detail_payload(review_id: str) -> dict[str, object]:
    """Return one review case with normalized resolution details."""

    settings = get_app_settings()
    record = get_review_record(review_id, settings=settings)
    if record is None:
        raise LookupError(f"Review item not found: {review_id}")
    return _review_record_payload(record)


def _list_review_payloads(
    *,
    limit: int,
    status: str | None = None,
    include_statuses: set[str] | None = None,
) -> list[dict[str, object]]:
    """Load review records and normalize them for API serialization."""

    settings = get_app_settings()
    fetch_limit = limit
    if status is None and include_statuses is not None:
        fetch_limit = max(limit * 5, 200)
    records = list_review_records(limit=fetch_limit, status=status, settings=settings)
    if include_statuses is not None:
        records = [record for record in records if record.status in include_statuses]
    return [_review_record_payload(record) for record in records[:limit]]


def build_resolved_review_queue_payload(
    *,
    limit: int = 20,
    status: str | None = None,
) -> dict[str, object]:
    """Return recently resolved review cases for reviewer follow-up."""

    if status is None:
        items = _list_review_payloads(limit=limit, status=None, include_statuses=set(RESOLVED_REVIEW_STATUSES))
    else:
        if status not in RESOLVED_REVIEW_STATUSES:
            raise ValueError(
                f"Resolved review status must be one of: {', '.join(sorted(RESOLVED_REVIEW_STATUSES))}."
            )
        items = _list_review_payloads(limit=limit, status=status)
    return {
        "total": len(items),
        "items": items,
    }


def build_review_analytics_payload(
    *,
    limit: int = 200,
    model_name: str | None = None,
    active_model_only: bool = False,
) -> dict[str, object]:
    """Return post-promotion review analytics for governance follow-up."""

    settings = get_app_settings()
    records = list_review_records(limit=limit, status=None, settings=settings)
    if not active_model_only:
        return summarize_review_records(records, model_name=model_name)

    serving_config = load_backend_serving_config(settings=settings)
    entry = load_current_oasis_model_entry(serving_config.active_oasis_model_registry)
    architecture = _active_model_architecture(entry)
    return summarize_review_records(
        records,
        model_name=architecture,
        active_model_id=entry.model_id,
        run_name=entry.run_name,
    )


def build_review_learning_payload(
    *,
    limit: int = 200,
    model_name: str | None = None,
    active_model_only: bool = False,
    selection_metric: str = "balanced_accuracy",
    threshold_step: float = 0.05,
) -> dict[str, object]:
    """Return an advisory learning report built from reviewer outcomes."""

    settings = get_app_settings()
    records = list_review_records(limit=limit, status=None, settings=settings)
    if not active_model_only:
        return summarize_review_learning(
            records,
            model_name=model_name,
            current_threshold=0.5,
            selection_metric=selection_metric,
            threshold_step=threshold_step,
        )

    serving_config = load_backend_serving_config(settings=settings)
    entry = load_current_oasis_model_entry(serving_config.active_oasis_model_registry)
    architecture = _active_model_architecture(entry)
    return summarize_review_learning(
        records,
        model_name=architecture,
        active_model_id=entry.model_id,
        run_name=entry.run_name,
        current_threshold=float(entry.recommended_threshold),
        selection_metric=selection_metric,
        threshold_step=threshold_step,
    )


def build_hold_history_payload(*, limit: int = 20) -> dict[str, object]:
    """Return recent operational-hold assessment history entries."""

    settings = get_app_settings()
    history_root = settings.outputs_root / "model_registry" / "hold_history"
    if not history_root.exists():
        return {"total": 0, "items": []}

    items: list[dict[str, object]] = []
    history_files = sorted(
        history_root.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for history_path in history_files:
        if len(items) >= limit:
            break
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        decision = dict(payload.get("decision", {}))
        monitoring = dict(payload.get("review_monitoring", {}))
        items.append(
            {
                "history_path": str(history_path),
                "assessed_at_utc": decision.get("assessed_at_utc", ""),
                "policy_name": decision.get("policy_name"),
                "operational_status": decision.get("operational_status", "unknown"),
                "hold_applied": bool(decision.get("hold_applied", False)),
                "status_changed": bool(decision.get("status_changed", False)),
                "trigger_codes": list(decision.get("trigger_codes", [])),
                "run_name": payload.get("run_name")
                or payload.get("registry_entry", {}).get("run_name")
                or _history_run_name_from_path(history_path),
                "model_id": payload.get("model_id") or payload.get("registry_entry", {}).get("model_id"),
                "summary": decision.get("summary"),
                "high_risk": bool(monitoring.get("high_risk", False)),
                "total_reviews": int(monitoring.get("total_reviews", 0)),
            }
        )
    return {
        "total": len(items),
        "items": items,
    }


def _history_run_name_from_path(history_path: Path) -> str:
    """Infer a run-like name from a hold-history file name when metadata is missing."""

    name = history_path.stem
    parts = name.rsplit("_", 1)
    return parts[0] if parts else name


def build_review_dashboard_payload(
    *,
    pending_limit: int = 10,
    resolved_limit: int = 10,
    history_limit: int = 10,
) -> dict[str, object]:
    """Return a compact reviewer-operations dashboard payload."""

    active_model = build_active_oasis_model_payload()
    pending_reviews = build_pending_review_queue_payload(limit=pending_limit)
    resolved_reviews = build_resolved_review_queue_payload(limit=resolved_limit)
    analytics = build_review_analytics_payload(limit=200, active_model_only=True)
    hold_history = build_hold_history_payload(limit=history_limit)

    operational_status = str(active_model.get("operational_status", "unknown"))
    high_risk = bool(analytics.get("high_risk", False))
    if operational_status == "hold":
        recommended_action = "Keep the active model in manual-review mode until hold triggers are cleared."
    elif high_risk:
        recommended_action = "Review the active model closely; analytics show elevated risk patterns."
    elif pending_reviews["total"] > 0:
        recommended_action = "Continue manual review of queued cases and monitor for new override patterns."
    else:
        recommended_action = "Reviewer queue is currently clear; continue routine monitoring."

    notes = [
        "This dashboard is for reviewer operations and governance follow-up, not for diagnosis decisions.",
        "Hold history preserves past reassessments without overwriting original promotion evidence.",
    ]
    if analytics.get("reviewer_agreement_available") is False:
        notes.append("Inter-reviewer agreement is not yet available because the workflow stores one primary resolution per case.")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "operational_status": operational_status,
            "high_risk": high_risk,
            "pending_reviews": pending_reviews["total"],
            "resolved_reviews": resolved_reviews["total"],
            "hold_history_entries": hold_history["total"],
            "recommended_action": recommended_action,
        },
        "active_model": active_model,
        "pending_reviews": pending_reviews,
        "resolved_reviews": resolved_reviews,
        "analytics": analytics,
        "hold_history": hold_history,
        "notes": notes,
    }


def build_promotion_candidates_payload(*, limit: int = 10) -> dict[str, object]:
    """Return tracked experiment candidates for promotion review."""

    items = [candidate.to_dict() for candidate in load_promotion_candidates(limit=limit, settings=get_app_settings())]
    return {
        "total": len(items),
        "items": items,
    }


def build_promotion_dashboard_payload(
    *,
    candidate_limit: int = 5,
    study_limit: int = 5,
    history_limit: int = 5,
) -> dict[str, object]:
    """Return a compact promotion workflow payload for the active model."""

    active_model = build_active_oasis_model_payload()
    candidates_payload = build_promotion_candidates_payload(limit=candidate_limit)
    studies = [study.to_dict() for study in load_promotion_studies(limit=study_limit, settings=get_app_settings())]
    history = [
        entry.to_dict()
        for entry in load_promotion_history_entries(limit=history_limit, settings=get_app_settings())
    ]

    ready_candidates = [
        item
        for item in candidates_payload["items"]
        if bool(item.get("promotion_preflight", {}).get("approved"))
    ]
    top_candidate = candidates_payload["items"][0] if candidates_payload["items"] else None
    if not ready_candidates:
        recommended_action = "No tracked candidate currently clears the promotion preflight; keep the active model and continue experiments."
    elif top_candidate is not None and bool(top_candidate.get("current_active")):
        recommended_action = "The current active model remains the strongest preflight-ready candidate from tracked experiments."
    else:
        top_name = None if top_candidate is None else top_candidate.get("experiment_name")
        recommended_action = (
            f"Candidate {top_name!r} is preflight-ready; run the explicit promotion workflow before changing the active model."
            if top_name
            else "A tracked candidate is preflight-ready; run the explicit promotion workflow before changing the active model."
        )

    notes = [
        "Promotion dashboard is read-only and advisory; it does not replace the explicit benchmarked promotion workflow.",
        "Tracked candidates are compared against the active model using saved experiment artifacts and the current research gate.",
    ]
    if studies:
        notes.append("Model-selection studies are included to show validation depth, not just one-off experiment wins.")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "active_run_name": active_model["run_name"],
            "candidate_count": candidates_payload["total"],
            "promotion_ready_candidates": len(ready_candidates),
            "top_candidate_experiment": None if top_candidate is None else top_candidate.get("experiment_name"),
            "recommended_action": recommended_action,
        },
        "active_model": active_model,
        "candidates": candidates_payload,
        "studies": studies,
        "recent_promotion_decisions": history,
        "notes": notes,
    }


def build_validation_studies_payload(*, limit: int = 10) -> dict[str, object]:
    """Return saved validation-depth studies for the active model family."""

    items = [study.to_dict() for study in load_validation_depth_studies(limit=limit, settings=get_app_settings())]
    return {
        "total": len(items),
        "items": items,
    }


def build_validation_depth_payload(*, limit: int = 10) -> dict[str, object]:
    """Return a compact validation-depth dashboard for the active OASIS model family."""

    dashboard = build_validation_depth_dashboard(limit=limit, settings=get_app_settings())
    return {
        "generated_at_utc": dashboard.generated_at_utc,
        "summary": {
            "active_model_id": dashboard.active_model_id,
            "active_run_name": dashboard.active_run_name,
            "active_run_family": dashboard.active_run_family,
            "total_studies": dashboard.total_studies,
            "repeated_split_studies": dashboard.repeated_split_studies,
            "direct_active_run_studies": dashboard.direct_active_run_studies,
            "related_family_studies": dashboard.related_family_studies,
            "repeated_split_family_studies": dashboard.repeated_split_family_studies,
            "overall_validation_depth": dashboard.overall_validation_depth,
            "recommended_action": dashboard.recommended_action,
            "strongest_study_name": dashboard.strongest_study_name,
            "strongest_stability_status": dashboard.strongest_stability_status,
        },
        "studies": [study.to_dict() for study in dashboard.studies],
        "notes": list(dashboard.notes),
    }


def _resolved_label_name(*, resolved_label: int | None, provided_name: str | None, current_payload: dict[str, object]) -> str | None:
    """Resolve a human-readable label name for a reviewer outcome."""

    if provided_name:
        return provided_name
    if resolved_label is None:
        return None
    if 0 <= int(resolved_label) < len(OASIS_BINARY_CLASS_NAMES):
        return OASIS_BINARY_CLASS_NAMES[int(resolved_label)]
    current_label_name = current_payload.get("label_name")
    if current_payload.get("predicted_label") == resolved_label and isinstance(current_label_name, str):
        return current_label_name
    return f"class_{resolved_label}"


def _active_model_architecture(entry) -> str | None:
    """Load the architecture name for the active model registry entry."""

    try:
        model_config = load_oasis_model_config(
            Path(entry.model_config_path) if entry.model_config_path else None
        )
    except FileNotFoundError:
        return None
    return model_config.architecture


def _build_active_review_monitoring_payload(entry) -> dict[str, object]:
    """Build review analytics scoped to the active model when possible."""

    settings = get_app_settings()
    architecture = _active_model_architecture(entry)
    records = list_review_records(limit=200, status=None, settings=settings)
    monitoring = summarize_review_records(
        records,
        model_name=architecture,
        active_model_id=entry.model_id,
        run_name=entry.run_name,
    )
    monitoring["active_model_id"] = entry.model_id
    monitoring["run_name"] = entry.run_name
    monitoring["model_name"] = architecture
    return monitoring


def resolve_review_queue_item_payload(review_id: str, request: ReviewResolutionRequest) -> dict[str, object]:
    """Resolve one pending review-queue item with audit-friendly metadata."""

    settings = get_app_settings()
    record = get_review_record(review_id, settings=settings)
    if record is None:
        raise LookupError(f"Review item not found: {review_id}")
    if record.status != "pending":
        raise ValueError(f"Review item {review_id} is already resolved with status '{record.status}'.")

    status_map = {
        "confirm_model_output": "confirmed",
        "override_prediction": "overridden",
        "dismiss": "dismissed",
    }
    resolved_status = status_map[request.action]
    payload = dict(record.payload)
    resolved_label_name = _resolved_label_name(
        resolved_label=request.resolved_label,
        provided_name=request.resolved_label_name,
        current_payload=payload,
    )
    resolution = {
        "action": request.action,
        "reviewer_id": request.reviewer_id,
        "resolution_note": request.resolution_note,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "resolved_label": request.resolved_label,
        "resolved_label_name": resolved_label_name,
        "final_status": resolved_status,
    }
    payload["review_flag"] = False
    payload["review_status"] = resolved_status
    payload["resolution"] = resolution

    record.status = resolved_status
    record.payload = payload
    persist_review_record(record, settings=settings)
    try:
        assess_active_oasis_model_hold(actor_id=request.reviewer_id, settings=settings)
    except FileNotFoundError:
        pass
    audit_sensitive_action(
        action="resolve_review_case",
        actor_id=request.reviewer_id,
        subject_id=record.subject_id,
        metadata={
            "review_id": record.review_id,
            "inference_id": record.inference_id,
            "trace_id": record.trace_id,
            "action": request.action,
            "resolved_status": resolved_status,
            "resolved_label": request.resolved_label,
        },
    )
    return {
        "review_id": record.review_id,
        "status": resolved_status,
        "message": f"Review item {record.review_id} marked as {resolved_status}.",
        "item": _review_record_payload(record),
    }


def build_oasis_volumetric_payload(
    *,
    image_path: str | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    scan_timestamp: str | None = None,
    split: str | None = None,
    row_index: int = 0,
    manifest_path: str | None = None,
) -> dict[str, object]:
    """Build a volumetric proxy payload for one OASIS MRI volume."""

    result = analyze_oasis_volume(
        image_path=image_path,
        subject_id=subject_id,
        session_id=session_id,
        scan_timestamp=scan_timestamp,
        split=split,
        row_index=row_index,
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
        settings=get_app_settings(),
    )
    return result.to_report_payload()


def build_oasis_longitudinal_structural_payload(
    *,
    subject_id: str,
    split: str | None = None,
    manifest_path: str | None = None,
    max_timepoints: int | None = None,
) -> dict[str, object]:
    """Build a subject-level OASIS structural longitudinal payload."""

    summary = build_oasis_structural_longitudinal_summary(
        subject_id,
        settings=get_app_settings(),
        split=split,
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
        max_timepoints=max_timepoints,
    )
    return summary.to_payload()


def build_saved_longitudinal_report_payload(request: LongitudinalReportRequest) -> dict[str, object]:
    """Build and save a subject-level longitudinal report payload."""

    if request.records:
        feature_configs = [
            TrendFeatureConfig(
                feature_name=config.feature_name,
                source=config.source,
                decline_direction=config.decline_direction,
                normalization=config.normalization,
                stable_slope_threshold=config.stable_slope_threshold,
                rapid_slope_threshold=config.rapid_slope_threshold,
                display_name=config.display_name,
                unit=config.unit,
                notes=config.notes,
            )
            for config in request.feature_configs
        ]
        records = [
            LongitudinalRecord(
                subject_id=record.subject_id,
                session_id=record.session_id,
                visit_order=record.visit_order,
                summary_label=record.summary_label,
                scan_timestamp=record.scan_timestamp,
                source_path=record.source_path,
                dataset=record.dataset,
                volumetric_features=dict(record.volumetric_features),
                model_probabilities=dict(record.model_probabilities),
                metadata=dict(record.metadata),
            )
            for record in request.records
        ]
        report, output_path = build_and_save_longitudinal_report(
            records,
            subject_id=request.subject_id,
            feature_configs=feature_configs or None,
            settings=get_app_settings(),
            file_stem=request.output_name,
        )
        payload = report.to_payload()
        payload["output_path"] = str(output_path)
        return payload

    summary = build_oasis_structural_longitudinal_summary(
        request.subject_id,
        settings=get_app_settings(),
        split=request.split,
        manifest_path=Path(request.manifest_path) if request.manifest_path is not None else None,
        max_timepoints=request.max_timepoints,
    )
    output_path = save_structural_longitudinal_report(
        summary,
        settings=get_app_settings(),
        file_stem=request.output_name or f"{request.subject_id}_api_longitudinal_report",
    )
    payload = summary.to_payload()
    payload["output_path"] = str(output_path)
    payload["report_type"] = "oasis_structural_proxy_longitudinal"
    return payload


def build_scan_prediction_payload(request: ScanPredictionRequest) -> dict[str, object]:
    """Run the reusable scan inference pipeline for an API request."""

    return predict_scan(
        request.scan_path,
        request.checkpoint_path,
        request.config_path,
        options=PredictScanOptions(
            output_name=request.output_name,
            threshold=request.threshold,
            device=request.device,
            model_config_path=Path(request.model_config_path) if request.model_config_path else None,
            save_debug_slices=request.save_debug_slices,
            subject_id=request.subject_id,
            session_id=request.session_id,
            scan_timestamp=request.scan_timestamp,
        ),
        settings=get_app_settings(),
    )


def _safe_upload_name(file_name: str) -> str:
    """Return a path-safe upload file name."""

    name = Path(file_name).name.strip()
    if not name:
        raise ValueError("Uploaded scan file_name cannot be empty.")
    return name.replace(" ", "_")


def _validate_raw_upload_file_name(file_name: str) -> None:
    """Validate raw upload format support.

    Analyze ``.hdr``/``.img`` pairs are intentionally not supported through the
    raw-byte endpoint because a safe upload needs both files together. Use the
    path-based endpoint for those scans.
    """

    lower_name = file_name.lower()
    if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
        raise ValueError("Raw upload endpoint supports .nii and .nii.gz only. Use scan_path for Analyze pairs.")


def save_raw_scan_upload(*, file_name: str, content: bytes, output_name: str) -> Path:
    """Save a raw binary scan upload under the backend outputs folder."""

    if not content:
        raise ValueError("Uploaded scan payload is empty.")
    safe_name = _safe_upload_name(file_name)
    _validate_raw_upload_file_name(safe_name)
    serving_config = load_backend_serving_config(settings=get_app_settings())
    max_bytes = int(serving_config.scan_validation.max_file_size_mb) * 1024 * 1024
    if len(content) > max_bytes:
        raise ValueError(
            f"Uploaded scan exceeds the configured serving limit of {serving_config.scan_validation.max_file_size_mb} MB."
        )
    safe_output_name = output_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    upload_root = ensure_directory(get_app_settings().outputs_root / "uploads" / "api" / safe_output_name)
    output_path = upload_root / safe_name
    output_path.write_bytes(content)
    return output_path


def build_scan_prediction_upload_payload(
    *,
    file_name: str,
    content: bytes,
    checkpoint_path: str,
    config_path: str | None,
    model_config_path: str | None,
    output_name: str,
    threshold: float | None,
    device: str,
    save_debug_slices: bool,
    subject_id: str | None,
    session_id: str | None,
    scan_timestamp: str | None,
) -> dict[str, object]:
    """Save a raw upload and run prediction with the same inference pipeline."""

    scan_path = save_raw_scan_upload(file_name=file_name, content=content, output_name=output_name)
    request = ScanPredictionRequest(
        scan_path=str(scan_path),
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        model_config_path=model_config_path,
        output_name=output_name,
        threshold=threshold,
        device=device,
        save_debug_slices=save_debug_slices,
        subject_id=subject_id,
        session_id=session_id,
        scan_timestamp=scan_timestamp,
    )
    payload = build_scan_prediction_payload(request)
    payload.setdefault("input_metadata", {})["upload_mode"] = "raw_binary_octet_stream"
    return payload


def build_scan_explanation_payload(request: ScanExplanationRequest) -> dict[str, object]:
    """Run Grad-CAM-style explainability for an API request."""

    result = explain_scan(
        ExplainScanConfig(
            scan_path=Path(request.scan_path),
            checkpoint_path=Path(request.checkpoint_path),
            preprocessing_config_path=Path(request.preprocessing_config_path)
            if request.preprocessing_config_path
            else None,
            model_config_path=Path(request.model_config_path) if request.model_config_path else None,
            output_name=request.output_name,
            target_layer=request.target_layer,
            target_class=request.target_class,
            device=request.device,
            image_size=tuple(request.image_size) if request.image_size is not None else None,
            slice_axis=request.slice_axis,
            slice_indices=tuple(request.slice_indices) if request.slice_indices is not None else None,
            save_saliency=request.save_saliency,
            true_label=request.true_label,
        ),
        settings=get_app_settings(),
    )
    return result.payload
