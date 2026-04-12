"""Scope-aligned evidence reporting for the current backend goal.

This module deliberately treats the OASIS 3D branch as the primary evidence
track and the Kaggle 2D branch as a secondary comparison branch. It does not
claim the two datasets are directly interchangeable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.inference.serving import load_backend_serving_config
from src.utils.io_utils import ensure_directory


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _safe_float(value: Any) -> float | None:
    """Convert a metric value to float when possible."""

    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert a metric value to int when possible."""

    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True, frozen=True)
class ScopeEvidencePaths:
    """Resolved evidence paths used for the scope-aligned report."""

    oasis_registry_path: Path
    oasis_repeated_split_study_path: Path | None
    kaggle_final_metrics_path: Path | None
    kaggle_split_summary_path: Path | None
    kaggle_manifest_summary_path: Path | None
    training_device_profile_path: Path | None


def _resolve_latest_kaggle_metrics_path(settings: AppSettings, run_name: str | None) -> Path | None:
    """Resolve the latest Kaggle final_metrics.json path or one explicit run."""

    kaggle_root = settings.outputs_root / "runs" / "kaggle"
    if run_name:
        candidates = [
            kaggle_root / run_name / "metrics" / "final_metrics.json",
            *list((kaggle_root / run_name / "evaluation").glob("*/final_metrics.json")),
        ]
        existing = [candidate for candidate in candidates if candidate.exists()]
        if not existing:
            return None
        return max(existing, key=lambda path: path.stat().st_mtime)
    candidates = sorted(
        [
            *kaggle_root.glob("*/metrics/final_metrics.json"),
            *kaggle_root.glob("*/evaluation/*/final_metrics.json"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_best_repeated_split_study_path(settings: AppSettings) -> Path | None:
    """Pick the strongest repeated-split OASIS study summary available."""

    study_paths = sorted((settings.outputs_root / "model_selection").glob("*/study_summary.json"))
    if not study_paths:
        return None

    best_path: Path | None = None
    best_score: tuple[int, int, float] | None = None
    for study_path in study_paths:
        try:
            payload = _load_json(study_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        split_seeds = [value for value in payload.get("split_seeds", []) if value is not None]
        repeated_split = len(set(split_seeds)) > 1
        if not repeated_split:
            continue
        runs = list(payload.get("runs", []))
        aggregate_metrics = list(payload.get("aggregate_metrics", []))
        test_auroc_mean = None
        test_sample_count_mean = None
        for aggregate in aggregate_metrics:
            if aggregate.get("split") == "test" and aggregate.get("metric_name") == "auroc":
                test_auroc_mean = _safe_float(aggregate.get("mean"))
            if aggregate.get("split") == "test" and aggregate.get("metric_name") == "sample_count":
                test_sample_count_mean = _safe_float(aggregate.get("mean"))
        score = (
            int(test_sample_count_mean or 0),
            len(runs),
            len(set(split_seeds)),
            test_auroc_mean or 0.0,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_path = study_path
    return best_path


def resolve_scope_evidence_paths(
    settings: AppSettings | None = None,
    *,
    kaggle_run_name: str | None = None,
) -> ScopeEvidencePaths:
    """Resolve default evidence inputs from the current workspace."""

    resolved_settings = settings or get_app_settings()
    serving_config = load_backend_serving_config(settings=resolved_settings)
    oasis_registry_path = serving_config.active_oasis_model_registry
    return ScopeEvidencePaths(
        oasis_registry_path=oasis_registry_path,
        oasis_repeated_split_study_path=_resolve_best_repeated_split_study_path(resolved_settings),
        kaggle_final_metrics_path=_resolve_latest_kaggle_metrics_path(resolved_settings, kaggle_run_name),
        kaggle_split_summary_path=(
            resolved_settings.data_root / "interim" / "kaggle_alz_split_summary.json"
            if (resolved_settings.data_root / "interim" / "kaggle_alz_split_summary.json").exists()
            else None
        ),
        kaggle_manifest_summary_path=(
            resolved_settings.data_root / "interim" / "kaggle_alz_manifest_summary.json"
            if (resolved_settings.data_root / "interim" / "kaggle_alz_manifest_summary.json").exists()
            else None
        ),
        training_device_profile_path=(
            resolved_settings.outputs_root / "reports" / "readiness" / "training_device_profile.json"
            if (resolved_settings.outputs_root / "reports" / "readiness" / "training_device_profile.json").exists()
            else None
        ),
    )


def _metric_triplet(metrics: dict[str, Any], *names: str) -> dict[str, float | int | None]:
    """Extract a compact comparable metrics payload."""

    payload: dict[str, float | int | None] = {}
    for name in names:
        if "count" in name or "sample_count" in name:
            payload[name] = _safe_int(metrics.get(name))
        else:
            payload[name] = _safe_float(metrics.get(name))
    return payload


def _build_oasis_primary_section(oasis_registry: dict[str, Any]) -> dict[str, Any]:
    """Create the primary OASIS evidence section."""

    validation_metrics = dict(oasis_registry.get("validation_metrics", {}))
    test_metrics = dict(oasis_registry.get("test_metrics", {}))
    review_monitoring = dict(oasis_registry.get("review_monitoring", {}))
    threshold_calibration = dict(oasis_registry.get("threshold_calibration", {}))
    return {
        "dataset": oasis_registry.get("dataset"),
        "role": "primary_evidence_track",
        "modality": "3d_structural_mri",
        "run_name": oasis_registry.get("run_name"),
        "model_id": oasis_registry.get("model_id"),
        "approval_status": "approved" if oasis_registry.get("promotion_decision", {}).get("approved") else "not_approved",
        "operational_status": oasis_registry.get("operational_status"),
        "image_size": list(oasis_registry.get("image_size", [])),
        "recommended_threshold": _safe_float(oasis_registry.get("recommended_threshold")),
        "validation_metrics": _metric_triplet(
            validation_metrics,
            "sample_count",
            "accuracy",
            "auroc",
            "sensitivity",
            "specificity",
            "f1",
            "review_required_count",
        ),
        "test_metrics": _metric_triplet(
            test_metrics,
            "sample_count",
            "accuracy",
            "auroc",
            "sensitivity",
            "specificity",
            "f1",
            "review_required_count",
        ),
        "threshold_calibration": {
            "selection_metric": threshold_calibration.get("selection_metric"),
            "threshold": _safe_float(threshold_calibration.get("threshold")),
        },
        "review_monitoring": {
            "total_reviews": _safe_int(review_monitoring.get("total_reviews")),
            "override_rate": _safe_float(review_monitoring.get("override_rate")),
            "high_risk": bool(review_monitoring.get("high_risk", False)),
            "risk_signals": list(review_monitoring.get("risk_signals", [])),
        },
        "notes": list(oasis_registry.get("notes", [])),
    }


def _build_oasis_repeated_split_section(study_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Create the repeated-split summary section when available."""

    if study_payload is None:
        return {
            "available": False,
            "role": "validation_depth_support",
            "notes": ["No repeated-split OASIS study summary was found in outputs/model_selection."],
        }

    def _aggregate(split: str, metric_name: str) -> dict[str, float | None]:
        for aggregate in study_payload.get("aggregate_metrics", []):
            if aggregate.get("split") == split and aggregate.get("metric_name") == metric_name:
                return {
                    "mean": _safe_float(aggregate.get("mean")),
                    "std": _safe_float(aggregate.get("std")),
                    "ci95_low": _safe_float(aggregate.get("ci95_low")),
                    "ci95_high": _safe_float(aggregate.get("ci95_high")),
                }
        return {}

    return {
        "available": True,
        "role": "validation_depth_support",
        "study_name": study_payload.get("study_name"),
        "best_run_name": next(
            (
                run.get("run_name")
                for run in study_payload.get("runs", [])
                if run.get("experiment_name") == study_payload.get("best_experiment_name")
            ),
            None,
        ),
        "run_count": len(study_payload.get("runs", [])),
        "seed_count": len(set(study_payload.get("seeds", []))),
        "split_seed_count": len(set(value for value in study_payload.get("split_seeds", []) if value is not None)),
        "selection_metric": study_payload.get("selection_metric"),
        "best_selection_score": _safe_float(study_payload.get("best_selection_score")),
        "validation_aggregate": {
            "auroc": _aggregate("val", "auroc"),
            "accuracy": _aggregate("val", "accuracy"),
            "f1": _aggregate("val", "f1"),
        },
        "test_aggregate": {
            "auroc": _aggregate("test", "auroc"),
            "accuracy": _aggregate("test", "accuracy"),
            "sensitivity": _aggregate("test", "sensitivity"),
            "specificity": _aggregate("test", "specificity"),
            "f1": _aggregate("test", "f1"),
            "review_required_count": _aggregate("test", "review_required_count"),
        },
        "notes": list(study_payload.get("notes", [])),
    }


def _build_kaggle_secondary_section(
    kaggle_metrics: dict[str, Any] | None,
    *,
    kaggle_split_summary: dict[str, Any] | None,
    kaggle_manifest_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create the secondary Kaggle evidence section."""

    if kaggle_metrics is None:
        return {
            "available": False,
            "role": "secondary_comparison_branch",
            "notes": ["No Kaggle final_metrics.json artifact was found yet."],
        }

    validation_metrics = dict(kaggle_metrics.get("validation", {}))
    test_metrics = dict(kaggle_metrics.get("test", {}))
    split_summary = kaggle_split_summary or {}
    manifest_summary = kaggle_manifest_summary or {}
    return {
        "available": True,
        "role": "secondary_comparison_branch",
        "dataset": kaggle_metrics.get("dataset"),
        "dataset_type": kaggle_metrics.get("dataset_type"),
        "run_name": kaggle_metrics.get("run_name"),
        "class_names": list(kaggle_metrics.get("class_names", [])),
        "validation_metrics": _metric_triplet(
            validation_metrics,
            "sample_count",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_ovr_auroc",
            "loss",
        ),
        "test_metrics": _metric_triplet(
            test_metrics,
            "sample_count",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_ovr_auroc",
            "loss",
        ),
        "split_summary": {
            "train_rows": _safe_int(split_summary.get("train_rows")),
            "val_rows": _safe_int(split_summary.get("val_rows")),
            "test_rows": _safe_int(split_summary.get("test_rows")),
            "subset_distribution_by_split": dict(split_summary.get("subset_distribution_by_split", {})),
            "warnings": list(split_summary.get("warnings", [])),
        },
        "manifest_summary": {
            "dataset_type": manifest_summary.get("dataset_type"),
            "manifest_row_count": _safe_int(manifest_summary.get("manifest_row_count")),
            "organization": manifest_summary.get("organization"),
            "warnings": list(manifest_summary.get("warnings", [])),
        },
        "notes": list(kaggle_metrics.get("warnings", [])),
    }


def _build_device_section(device_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Create the laptop/device context section."""

    if device_profile is None:
        return {
            "available": False,
            "notes": ["No saved training device profile was found."],
        }
    return {
        "available": True,
        "recommended_device": device_profile.get("recommended_device"),
        "cuda_available": bool(device_profile.get("cuda_available", False)),
        "cuda_device_name": device_profile.get("cuda_device_name"),
        "total_memory_gb": _safe_float(device_profile.get("total_memory_gb")),
        "available_memory_gb": _safe_float(device_profile.get("available_memory_gb")),
        "warnings": list(device_profile.get("warnings", [])),
        "recommendations": list(device_profile.get("recommendations", [])),
    }


def _recommendations(
    *,
    oasis_primary: dict[str, Any],
    oasis_repeated_splits: dict[str, Any],
    kaggle_secondary: dict[str, Any],
) -> list[str]:
    """Create practical next-step recommendations from the current evidence mix."""

    recommendations = [
        "Keep OASIS as the primary evidence track for the project goal; Kaggle remains a secondary comparison branch.",
        "Do not use Kaggle 2D results to promote or replace the active OASIS 3D decision-support model.",
    ]
    if oasis_repeated_splits.get("available"):
        recommendations.append(
            "Use the repeated-split OASIS study as the main internal stability reference until an outside 3D cohort is available."
        )
    else:
        recommendations.append(
            "Run or preserve a repeated-split OASIS study before making stronger model claims."
        )

    kaggle_available = bool(kaggle_secondary.get("available"))
    kaggle_test = dict(kaggle_secondary.get("test_metrics", {}))
    kaggle_auroc = _safe_float(kaggle_test.get("macro_ovr_auroc"))
    if kaggle_available and kaggle_auroc is not None and kaggle_auroc >= 0.75:
        recommendations.append(
            "Treat Kaggle as a useful engineering and slice-based comparison branch, while keeping all conclusions dataset-specific."
        )
    elif kaggle_available:
        recommendations.append(
            "Improve the Kaggle branch only as a secondary experiment; weak 2D performance should not redirect the main OASIS-first roadmap."
        )
    else:
        recommendations.append(
            "Produce one full non-dry Kaggle run only if you want a secondary 2D comparison branch in the evidence report."
        )

    oasis_test = dict(oasis_primary.get("test_metrics", {}))
    review_required_count = _safe_int(oasis_test.get("review_required_count"))
    sample_count = _safe_int(oasis_test.get("sample_count"))
    if review_required_count is not None and sample_count:
        review_rate = review_required_count / sample_count if sample_count else None
        if review_rate is not None and review_rate > 0.2:
            recommendations.append(
                "A meaningful fraction of OASIS test cases still trigger review; keep review/governance workflows active during demos and future experiments."
            )
    recommendations.append(
        "If OASIS-2 becomes available later, onboard it for longitudinal strength before spending effort on larger architectures."
    )
    return recommendations


def build_scope_aligned_evidence_report(
    settings: AppSettings | None = None,
    *,
    kaggle_run_name: str | None = None,
    oasis_registry_path: Path | None = None,
    oasis_repeated_split_study_path: Path | None = None,
    kaggle_final_metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Build a scope-aligned evidence report for the current project goal."""

    resolved_settings = settings or get_app_settings()
    resolved_paths = resolve_scope_evidence_paths(resolved_settings, kaggle_run_name=kaggle_run_name)
    oasis_registry_payload = _load_json(oasis_registry_path or resolved_paths.oasis_registry_path)
    repeated_split_payload = None
    resolved_repeated_split_path = oasis_repeated_split_study_path or resolved_paths.oasis_repeated_split_study_path
    if resolved_repeated_split_path is not None and resolved_repeated_split_path.exists():
        repeated_split_payload = _load_json(resolved_repeated_split_path)

    resolved_kaggle_metrics_path = kaggle_final_metrics_path or resolved_paths.kaggle_final_metrics_path
    kaggle_metrics_payload = (
        _load_json(resolved_kaggle_metrics_path)
        if resolved_kaggle_metrics_path is not None and resolved_kaggle_metrics_path.exists()
        else None
    )
    kaggle_split_summary = (
        _load_json(resolved_paths.kaggle_split_summary_path)
        if resolved_paths.kaggle_split_summary_path is not None
        else None
    )
    kaggle_manifest_summary = (
        _load_json(resolved_paths.kaggle_manifest_summary_path)
        if resolved_paths.kaggle_manifest_summary_path is not None
        else None
    )
    device_profile = (
        _load_json(resolved_paths.training_device_profile_path)
        if resolved_paths.training_device_profile_path is not None
        else None
    )

    oasis_primary = _build_oasis_primary_section(oasis_registry_payload)
    oasis_repeated_splits = _build_oasis_repeated_split_section(repeated_split_payload)
    kaggle_secondary = _build_kaggle_secondary_section(
        kaggle_metrics_payload,
        kaggle_split_summary=kaggle_split_summary,
        kaggle_manifest_summary=kaggle_manifest_summary,
    )
    device_context = _build_device_section(device_profile)
    recommendations = _recommendations(
        oasis_primary=oasis_primary,
        oasis_repeated_splits=oasis_repeated_splits,
        kaggle_secondary=kaggle_secondary,
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_statement": (
            "OASIS-first structural brain MRI decision-support research backend for dementia-related pattern analysis, "
            "with Kaggle retained only as a separate 2D comparison branch."
        ),
        "scope_alignment": {
            "primary_goal": "structural_mri_decision_support_research",
            "primary_dataset_role": "oasis_primary_3d",
            "secondary_dataset_role": "kaggle_secondary_2d_comparison",
            "diagnosis_claim_allowed": False,
            "implicit_dataset_merge_allowed": False,
        },
        "paths": {
            "oasis_registry_path": str(oasis_registry_path or resolved_paths.oasis_registry_path),
            "oasis_repeated_split_study_path": None
            if resolved_repeated_split_path is None
            else str(resolved_repeated_split_path),
            "kaggle_final_metrics_path": None
            if resolved_kaggle_metrics_path is None
            else str(resolved_kaggle_metrics_path),
            "kaggle_split_summary_path": None
            if resolved_paths.kaggle_split_summary_path is None
            else str(resolved_paths.kaggle_split_summary_path),
            "training_device_profile_path": None
            if resolved_paths.training_device_profile_path is None
            else str(resolved_paths.training_device_profile_path),
        },
        "oasis_primary": oasis_primary,
        "oasis_repeated_splits": oasis_repeated_splits,
        "kaggle_secondary": kaggle_secondary,
        "device_context": device_context,
        "comparison_notes": [
            "OASIS metrics and Kaggle metrics are both useful, but they are not a like-for-like medical validation comparison.",
            "OASIS reflects subject-level 3D structural MRI evidence; Kaggle reflects slice-based 2D comparison evidence.",
            "Kaggle should influence engineering intuition and secondary experiments, not replace the OASIS promotion/governance path.",
        ],
        "recommendations": recommendations,
    }


def save_scope_aligned_evidence_report(
    report: dict[str, Any],
    settings: AppSettings | None = None,
    *,
    file_stem: str = "scope_aligned_evidence_report",
) -> tuple[Path, Path]:
    """Save the scope-aligned evidence report as JSON and Markdown."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "evidence")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    oasis_test = report.get("oasis_primary", {}).get("test_metrics", {})
    kaggle_test = report.get("kaggle_secondary", {}).get("test_metrics", {})
    oasis_repeated = report.get("oasis_repeated_splits", {})
    lines = [
        "# Scope-Aligned Evidence Report",
        "",
        report["goal_statement"],
        "",
        "## OASIS Primary Evidence",
        "",
        f"- run_name: {report['oasis_primary'].get('run_name')}",
        f"- approval_status: {report['oasis_primary'].get('approval_status')}",
        f"- operational_status: {report['oasis_primary'].get('operational_status')}",
        f"- test_accuracy: {oasis_test.get('accuracy')}",
        f"- test_auroc: {oasis_test.get('auroc')}",
        f"- test_sensitivity: {oasis_test.get('sensitivity')}",
        f"- test_specificity: {oasis_test.get('specificity')}",
        f"- test_f1: {oasis_test.get('f1')}",
        "",
        "## OASIS Repeated-Split Support",
        "",
        f"- available: {oasis_repeated.get('available')}",
        f"- study_name: {oasis_repeated.get('study_name')}",
        f"- run_count: {oasis_repeated.get('run_count')}",
        f"- split_seed_count: {oasis_repeated.get('split_seed_count')}",
        f"- test_auroc_mean: {oasis_repeated.get('test_aggregate', {}).get('auroc', {}).get('mean')}",
        f"- test_accuracy_mean: {oasis_repeated.get('test_aggregate', {}).get('accuracy', {}).get('mean')}",
        "",
        "## Kaggle Secondary Branch",
        "",
        f"- available: {report['kaggle_secondary'].get('available')}",
        f"- dataset_type: {report['kaggle_secondary'].get('dataset_type')}",
        f"- run_name: {report['kaggle_secondary'].get('run_name')}",
        f"- test_accuracy: {kaggle_test.get('accuracy')}",
        f"- test_balanced_accuracy: {kaggle_test.get('balanced_accuracy')}",
        f"- test_macro_f1: {kaggle_test.get('macro_f1')}",
        f"- test_macro_ovr_auroc: {kaggle_test.get('macro_ovr_auroc')}",
        "",
        "## Comparison Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in report.get("comparison_notes", []))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {recommendation}" for recommendation in report.get("recommendations", []))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
