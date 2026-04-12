"""OASIS-first longitudinal focus reporting for the narrowed project goal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.data.oasis2_readiness import build_oasis2_readiness_report
from src.evaluation.evidence_reporting import resolve_scope_evidence_paths
from src.inference.serving import load_backend_serving_config
from src.utils.io_utils import ensure_directory


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


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


def _resolve_registry_linked_path(path_value: str | Path | None, settings: AppSettings) -> Path | None:
    """Resolve a path stored inside registry JSON relative to the project/workspace."""

    if path_value in {None, ""}:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    workspace_candidate = settings.workspace_root / candidate
    if workspace_candidate.exists():
        return workspace_candidate
    project_candidate = settings.project_root / candidate
    if project_candidate.exists():
        return project_candidate
    return workspace_candidate


@dataclass(slots=True, frozen=True)
class OASISLongitudinalFocusPaths:
    """Resolved paths for the OASIS longitudinal-focus report."""

    oasis_registry_path: Path
    oasis_split_summary_path: Path | None
    oasis_longitudinal_subject_summary_path: Path | None
    oasis_repeated_split_study_path: Path | None


def resolve_oasis_longitudinal_focus_paths(
    settings: AppSettings | None = None,
    *,
    oasis_registry_path: Path | None = None,
) -> OASISLongitudinalFocusPaths:
    """Resolve the active OASIS evidence files for longitudinal planning."""

    resolved_settings = settings or get_app_settings()
    resolved_registry_path = (
        oasis_registry_path
        if oasis_registry_path is not None
        else load_backend_serving_config(settings=resolved_settings).active_oasis_model_registry
    )
    split_summary_path = None
    longitudinal_subject_summary_path = None
    if resolved_registry_path.exists():
        registry_payload = _load_json(resolved_registry_path)
        benchmark = dict(registry_payload.get("benchmark", {}))
        manifest_path_value = benchmark.get("manifest_path")
        if manifest_path_value:
            manifest_path = _resolve_registry_linked_path(manifest_path_value, resolved_settings)
            manifest_root = manifest_path.parent
            candidate_split_summary = manifest_root / "oasis_split_summary.json"
            candidate_longitudinal_subject_summary = manifest_root / "oasis_longitudinal_subject_summary.csv"
            if candidate_split_summary.exists():
                split_summary_path = candidate_split_summary
            if candidate_longitudinal_subject_summary.exists():
                longitudinal_subject_summary_path = candidate_longitudinal_subject_summary

    scope_paths = resolve_scope_evidence_paths(resolved_settings)
    return OASISLongitudinalFocusPaths(
        oasis_registry_path=resolved_registry_path,
        oasis_split_summary_path=split_summary_path,
        oasis_longitudinal_subject_summary_path=longitudinal_subject_summary_path,
        oasis_repeated_split_study_path=scope_paths.oasis_repeated_split_study_path,
    )


def _build_current_oasis_section(
    *,
    registry_payload: dict[str, Any],
    split_summary: dict[str, Any] | None,
    subject_summary_frame: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build the current OASIS evidence section."""

    test_metrics = dict(registry_payload.get("test_metrics", {}))
    validation_metrics = dict(registry_payload.get("validation_metrics", {}))
    benchmark = dict(registry_payload.get("benchmark", {}))
    split_longitudinal = dict((split_summary or {}).get("longitudinal", {}))
    subject_counts = dict((split_summary or {}).get("subject_counts", {}))
    row_counts = dict((split_summary or {}).get("row_counts", {}))

    if subject_summary_frame is None or subject_summary_frame.empty:
        subject_summary_payload = {
            "subject_count": 0,
            "multi_session_subject_count": 0,
            "multi_session_fraction": None,
            "max_sessions_per_subject": None,
            "timestamp_coverage_fraction": None,
            "label_distribution_for_multi_session_subjects": {},
            "example_multi_session_subjects": [],
        }
    else:
        multi_session_frame = subject_summary_frame.loc[subject_summary_frame["is_longitudinal_subject"].fillna(False)]
        timestamp_available = (
            subject_summary_frame["first_scan_timestamp"].fillna("").astype(str).str.strip().ne("")
        )
        subject_count = int(len(subject_summary_frame))
        multi_session_subject_count = int(len(multi_session_frame))
        label_counts = (
            multi_session_frame["label_name"].fillna("").astype(str).value_counts().to_dict()
            if not multi_session_frame.empty
            else {}
        )
        subject_summary_payload = {
            "subject_count": subject_count,
            "multi_session_subject_count": multi_session_subject_count,
            "multi_session_fraction": (
                round(multi_session_subject_count / subject_count, 6) if subject_count else None
            ),
            "max_sessions_per_subject": int(subject_summary_frame["subject_session_count"].max()),
            "timestamp_coverage_fraction": round(float(timestamp_available.mean()), 6),
            "label_distribution_for_multi_session_subjects": {
                str(label): int(count) for label, count in label_counts.items()
            },
            "example_multi_session_subjects": multi_session_frame["subject_id"].astype(str).tolist()[:5],
        }

    return {
        "dataset": registry_payload.get("dataset"),
        "run_name": registry_payload.get("run_name"),
        "benchmark_subject_safe": bool(benchmark.get("subject_safe", False)),
        "benchmark_subject_count": _safe_int(benchmark.get("subject_count")),
        "benchmark_sample_count": _safe_int(benchmark.get("sample_count")),
        "validation_metrics": {
            "accuracy": _safe_float(validation_metrics.get("accuracy")),
            "auroc": _safe_float(validation_metrics.get("auroc")),
            "sensitivity": _safe_float(validation_metrics.get("sensitivity")),
            "specificity": _safe_float(validation_metrics.get("specificity")),
            "f1": _safe_float(validation_metrics.get("f1")),
        },
        "test_metrics": {
            "accuracy": _safe_float(test_metrics.get("accuracy")),
            "auroc": _safe_float(test_metrics.get("auroc")),
            "sensitivity": _safe_float(test_metrics.get("sensitivity")),
            "specificity": _safe_float(test_metrics.get("specificity")),
            "f1": _safe_float(test_metrics.get("f1")),
            "review_required_count": _safe_int(test_metrics.get("review_required_count")),
            "sample_count": _safe_int(test_metrics.get("sample_count")),
        },
        "split_summary": {
            "subject_counts": {str(key): int(value) for key, value in subject_counts.items()},
            "row_counts": {str(key): int(value) for key, value in row_counts.items()},
            "subjects_with_multiple_sessions": _safe_int(split_longitudinal.get("subjects_with_multiple_sessions")),
            "subjects_with_single_sessions": _safe_int(split_longitudinal.get("subjects_with_single_session")),
            "max_sessions_per_subject": _safe_int(split_longitudinal.get("max_sessions_per_subject")),
            "visit_order_sources": dict(split_longitudinal.get("visit_order_sources", {})),
            "session_id_sources": dict(split_longitudinal.get("session_id_sources", {})),
        },
        "subject_summary": subject_summary_payload,
        "notes": list(registry_payload.get("notes", [])),
    }


def _build_repeated_split_section(study_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Build the strongest repeated-split OASIS support summary."""

    if study_payload is None:
        return {
            "available": False,
            "notes": ["No repeated-split OASIS study summary was found."],
        }

    def _aggregate(metric_name: str) -> dict[str, float | None]:
        for aggregate in study_payload.get("aggregate_metrics", []):
            if aggregate.get("split") == "test" and aggregate.get("metric_name") == metric_name:
                return {
                    "mean": _safe_float(aggregate.get("mean")),
                    "std": _safe_float(aggregate.get("std")),
                    "ci95_low": _safe_float(aggregate.get("ci95_low")),
                    "ci95_high": _safe_float(aggregate.get("ci95_high")),
                }
        return {}

    return {
        "available": True,
        "study_name": study_payload.get("study_name"),
        "selection_metric": study_payload.get("selection_metric"),
        "run_count": len(study_payload.get("runs", [])),
        "split_seed_count": len(set(value for value in study_payload.get("split_seeds", []) if value is not None)),
        "test_aggregate": {
            "accuracy": _aggregate("accuracy"),
            "auroc": _aggregate("auroc"),
            "sensitivity": _aggregate("sensitivity"),
            "specificity": _aggregate("specificity"),
            "f1": _aggregate("f1"),
            "review_required_count": _aggregate("review_required_count"),
        },
        "notes": list(study_payload.get("notes", [])),
    }


def _build_oasis2_readiness_section(report: dict[str, Any]) -> dict[str, Any]:
    """Compress the OASIS-2 readiness report into the planning view."""

    dataset_summary = dict(report.get("dataset_summary", {}))
    return {
        "overall_status": report.get("overall_status"),
        "source_root": report.get("source_root"),
        "source_resolution": report.get("source_resolution"),
        "supported_volume_file_count": _safe_int(dataset_summary.get("supported_volume_file_count")),
        "metadata_file_count": _safe_int(dataset_summary.get("metadata_file_count")),
        "unique_subject_id_count": _safe_int(dataset_summary.get("unique_subject_id_count")),
        "unique_session_id_count": _safe_int(dataset_summary.get("unique_session_id_count")),
        "longitudinal_subject_count": _safe_int(dataset_summary.get("longitudinal_subject_count")),
        "notes": list(report.get("notes", [])),
        "recommendations": list(report.get("recommendations", [])),
    }


def _focus_recommendations(
    *,
    current_oasis: dict[str, Any],
    repeated_splits: dict[str, Any],
    oasis2_readiness: dict[str, Any],
) -> list[str]:
    """Create longitudinal-focus recommendations from current evidence."""

    recommendations = [
        "Keep OASIS-1 as the primary 3D evidence track for structural MRI decision-support research.",
        "Use the active OASIS model and repeated-split evidence as the main benchmark for future work, not Kaggle.",
    ]
    multi_session_subject_count = _safe_int(current_oasis.get("subject_summary", {}).get("multi_session_subject_count"))
    if multi_session_subject_count == 0:
        recommendations.append(
            "Do not overclaim longitudinal trend evidence from the current OASIS-1 manifest: the local split summary shows no multi-session subjects right now."
        )
    else:
        recommendations.append(
            "Current OASIS-1 data already contains some repeated-subject structure; reuse it for conservative within-dataset longitudinal experiments."
        )

    if repeated_splits.get("available"):
        recommendations.append(
            "Anchor future model comparisons to the repeated-split OASIS study before adding more architecture complexity."
        )
    else:
        recommendations.append(
            "Create or preserve an OASIS repeated-split study before expanding the model family."
        )

    if oasis2_readiness.get("overall_status") == "pass":
        recommendations.append(
            "OASIS-2 looks locally available enough for onboarding; the next concrete step should be a dedicated OASIS-2 manifest adapter."
        )
    else:
        recommendations.append(
            "OASIS-2 is still the correct next dataset priority for real longitudinal evidence once it becomes available locally."
        )
    return recommendations


def build_oasis_longitudinal_focus_report(
    settings: AppSettings | None = None,
    *,
    oasis_registry_path: Path | None = None,
) -> dict[str, Any]:
    """Build the OASIS-first longitudinal focus report."""

    resolved_settings = settings or get_app_settings()
    paths = resolve_oasis_longitudinal_focus_paths(
        resolved_settings,
        oasis_registry_path=oasis_registry_path,
    )
    registry_payload = _load_json(paths.oasis_registry_path)
    split_summary_payload = _load_json(paths.oasis_split_summary_path) if paths.oasis_split_summary_path else None
    subject_summary_frame = (
        pd.read_csv(paths.oasis_longitudinal_subject_summary_path)
        if paths.oasis_longitudinal_subject_summary_path and paths.oasis_longitudinal_subject_summary_path.exists()
        else None
    )
    repeated_split_payload = (
        _load_json(paths.oasis_repeated_split_study_path)
        if paths.oasis_repeated_split_study_path and paths.oasis_repeated_split_study_path.exists()
        else None
    )
    oasis2_readiness_payload = build_oasis2_readiness_report(resolved_settings).to_payload()

    current_oasis = _build_current_oasis_section(
        registry_payload=registry_payload,
        split_summary=split_summary_payload,
        subject_summary_frame=subject_summary_frame,
    )
    repeated_splits = _build_repeated_split_section(repeated_split_payload)
    oasis2_readiness = _build_oasis2_readiness_section(oasis2_readiness_payload)
    recommendations = _focus_recommendations(
        current_oasis=current_oasis,
        repeated_splits=repeated_splits,
        oasis2_readiness=oasis2_readiness,
    )

    longitudinal_evidence_status = (
        "limited"
        if _safe_int(current_oasis.get("subject_summary", {}).get("multi_session_subject_count")) == 0
        else "present"
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_statement": (
            "Center future work on OASIS-first 3D structural MRI evidence, repeated-split internal validation, "
            "and future OASIS-2 longitudinal onboarding."
        ),
        "focus_assessment": {
            "classification_evidence_status": "strong_baseline_present",
            "longitudinal_evidence_status": longitudinal_evidence_status,
            "next_dataset_priority": "oasis2",
            "secondary_branch_status": "kaggle_should_remain_secondary",
        },
        "paths": {
            "oasis_registry_path": str(paths.oasis_registry_path),
            "oasis_split_summary_path": None if paths.oasis_split_summary_path is None else str(paths.oasis_split_summary_path),
            "oasis_longitudinal_subject_summary_path": None
            if paths.oasis_longitudinal_subject_summary_path is None
            else str(paths.oasis_longitudinal_subject_summary_path),
            "oasis_repeated_split_study_path": None
            if paths.oasis_repeated_split_study_path is None
            else str(paths.oasis_repeated_split_study_path),
        },
        "current_oasis": current_oasis,
        "repeated_splits": repeated_splits,
        "oasis2_readiness": oasis2_readiness,
        "recommendations": recommendations,
        "notes": [
            "This report is intentionally OASIS-first and longitudinal-focused.",
            "It does not treat Kaggle as interchangeable longitudinal evidence.",
            "Longitudinal workflow scaffolding can exist before longitudinal evidence becomes strong; the report distinguishes those two ideas.",
        ],
    }


def save_oasis_longitudinal_focus_report(
    report: dict[str, Any],
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis_longitudinal_focus_report",
) -> tuple[Path, Path]:
    """Save the OASIS longitudinal-focus report as JSON and Markdown."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "evidence")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    current_oasis = report.get("current_oasis", {})
    subject_summary = current_oasis.get("subject_summary", {})
    repeated_splits = report.get("repeated_splits", {})
    oasis2_readiness = report.get("oasis2_readiness", {})
    lines = [
        "# OASIS Longitudinal Focus Report",
        "",
        report["goal_statement"],
        "",
        "## Current OASIS Baseline",
        "",
        f"- run_name: {current_oasis.get('run_name')}",
        f"- benchmark_subject_safe: {current_oasis.get('benchmark_subject_safe')}",
        f"- test_accuracy: {current_oasis.get('test_metrics', {}).get('accuracy')}",
        f"- test_auroc: {current_oasis.get('test_metrics', {}).get('auroc')}",
        f"- test_sensitivity: {current_oasis.get('test_metrics', {}).get('sensitivity')}",
        "",
        "## Current Longitudinal Coverage",
        "",
        f"- subject_count: {subject_summary.get('subject_count')}",
        f"- multi_session_subject_count: {subject_summary.get('multi_session_subject_count')}",
        f"- multi_session_fraction: {subject_summary.get('multi_session_fraction')}",
        f"- max_sessions_per_subject: {subject_summary.get('max_sessions_per_subject')}",
        f"- timestamp_coverage_fraction: {subject_summary.get('timestamp_coverage_fraction')}",
        "",
        "## Repeated-Split Support",
        "",
        f"- available: {repeated_splits.get('available')}",
        f"- study_name: {repeated_splits.get('study_name')}",
        f"- test_auroc_mean: {repeated_splits.get('test_aggregate', {}).get('auroc', {}).get('mean')}",
        f"- test_accuracy_mean: {repeated_splits.get('test_aggregate', {}).get('accuracy', {}).get('mean')}",
        "",
        "## OASIS-2 Readiness",
        "",
        f"- overall_status: {oasis2_readiness.get('overall_status')}",
        f"- source_root: {oasis2_readiness.get('source_root')}",
        f"- supported_volume_file_count: {oasis2_readiness.get('supported_volume_file_count')}",
        f"- unique_subject_id_count: {oasis2_readiness.get('unique_subject_id_count')}",
        f"- longitudinal_subject_count: {oasis2_readiness.get('longitudinal_subject_count')}",
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {recommendation}" for recommendation in report.get("recommendations", []))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
