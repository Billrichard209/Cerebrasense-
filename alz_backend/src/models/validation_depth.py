"""Validation-depth and stability reporting from saved OASIS study artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.models.registry import load_current_oasis_model_entry


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


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _run_family_name(run_name: str | None) -> str | None:
    """Collapse seed/split suffixes into a model-family identifier."""

    if not run_name:
        return None
    if "_seed" in run_name:
        return run_name.split("_seed", 1)[0]
    return run_name


def _aggregate_lookup(aggregates: list[dict[str, Any]], split: str, metric_name: str) -> dict[str, Any] | None:
    """Return one aggregate metric block for a split/metric pair."""

    for aggregate in aggregates:
        if aggregate.get("split") == split and aggregate.get("metric_name") == metric_name:
            return aggregate
    return None


def _aggregate_payload(aggregate: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize an aggregate metric block into a compact payload."""

    if aggregate is None:
        return {}
    return {
        "mean": _safe_float(aggregate.get("mean")),
        "std": _safe_float(aggregate.get("std")),
        "minimum": _safe_float(aggregate.get("minimum")),
        "maximum": _safe_float(aggregate.get("maximum")),
        "ci95_low": _safe_float(aggregate.get("ci95_low")),
        "ci95_high": _safe_float(aggregate.get("ci95_high")),
        "values": list(aggregate.get("values", [])),
    }


@dataclass(slots=True)
class ValidationStudySummary:
    """One saved multi-seed or repeated-split study summarized for operators."""

    study_name: str
    study_root: str
    evaluation_type: str
    run_count: int
    seed_count: int
    split_seed_count: int
    repeated_split: bool
    pair_seed_and_split_seed: bool
    active_run_included: bool
    active_family_included: bool
    selection_split: str
    selection_metric: str
    best_experiment_name: str | None
    best_run_name: str | None
    best_selection_score: float | None
    validation_depth_level: str
    stability_status: str
    promotion_confidence_support: str
    aggregate_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationDepthDashboard:
    """High-level validation-depth summary for the active OASIS model family."""

    generated_at_utc: str
    active_model_id: str | None
    active_run_name: str
    active_run_family: str | None
    total_studies: int
    repeated_split_studies: int
    direct_active_run_studies: int
    related_family_studies: int
    repeated_split_family_studies: int
    overall_validation_depth: str
    recommended_action: str
    strongest_study_name: str | None = None
    strongest_stability_status: str | None = None
    notes: list[str] = field(default_factory=list)
    studies: list[ValidationStudySummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "studies": [study.to_dict() for study in self.studies],
        }


def _stability_status(
    *,
    run_count: int,
    repeated_split: bool,
    sample_count_mean: float | None,
    val_auroc_mean: float | None,
    test_auroc_mean: float | None,
    test_auroc_std: float | None,
    test_sensitivity_std: float | None,
) -> tuple[str, list[str]]:
    """Return a simple stability label plus explanatory warnings."""

    warnings: list[str] = []
    if run_count < 3:
        warnings.append("Study has fewer than three runs, so stability evidence is limited.")
        return "insufficient", warnings
    if sample_count_mean is None or sample_count_mean < 10:
        warnings.append("Held-out sample count is too small for trustworthy stability claims.")
        return "insufficient", warnings

    val_test_gap = None
    if val_auroc_mean is not None and test_auroc_mean is not None:
        val_test_gap = abs(val_auroc_mean - test_auroc_mean)

    if test_auroc_std is not None and test_auroc_std >= 0.10:
        warnings.append("Test AUROC varies substantially across runs.")
        return "unstable", warnings
    if test_sensitivity_std is not None and test_sensitivity_std >= 0.15:
        warnings.append("Test sensitivity varies substantially across runs.")
        return "unstable", warnings
    if val_test_gap is not None and val_test_gap >= 0.12:
        warnings.append("Mean validation/test AUROC drift is large across the study.")
        return "unstable", warnings

    if repeated_split and run_count >= 4:
        if (
            (test_auroc_std is None or test_auroc_std <= 0.05)
            and (test_sensitivity_std is None or test_sensitivity_std <= 0.08)
            and (val_test_gap is None or val_test_gap <= 0.08)
        ):
            return "stable", warnings
        warnings.append("Repeated-split evidence exists, but variability is still noticeable.")
        return "moderate", warnings

    if (
        (test_auroc_std is None or test_auroc_std <= 0.07)
        and (val_test_gap is None or val_test_gap <= 0.10)
    ):
        return "moderate", warnings
    warnings.append("Fixed-split multi-seed evidence is present, but variance is still material.")
    return "unstable", warnings


def _validation_depth_level(
    *,
    repeated_split: bool,
    split_seed_count: int,
    run_count: int,
    sample_count_mean: float | None,
) -> str:
    """Return a coarse validation-depth label."""

    if run_count < 3 or sample_count_mean is None or sample_count_mean < 10:
        return "insufficient"
    if repeated_split and split_seed_count >= 2 and run_count >= 4 and sample_count_mean >= 20:
        return "strong"
    if run_count >= 3 and sample_count_mean >= 20:
        return "moderate"
    return "limited"


def _promotion_confidence_support(validation_depth_level: str, stability_status: str) -> str:
    """Translate study evidence into a promotion-confidence label."""

    if validation_depth_level == "strong" and stability_status == "stable":
        return "strong"
    if validation_depth_level in {"strong", "moderate"} and stability_status in {"stable", "moderate"}:
        return "moderate"
    if validation_depth_level == "insufficient":
        return "insufficient"
    return "weak"


def _best_run_name(payload: dict[str, Any]) -> str | None:
    """Resolve the best run name from a study summary."""

    best_experiment_name = payload.get("best_experiment_name")
    for run in payload.get("runs", []):
        if run.get("experiment_name") == best_experiment_name:
            return run.get("run_name")
    return None


def _study_summary_from_payload(
    payload: dict[str, Any],
    *,
    study_root: Path,
    active_run_name: str,
) -> ValidationStudySummary:
    """Convert one study-summary JSON payload into a structured summary."""

    runs = list(payload.get("runs", []))
    split_seed_values = [value for value in payload.get("split_seeds", []) if value is not None]
    repeated_split = len(set(split_seed_values)) > 1
    active_run_family = _run_family_name(active_run_name)
    run_names = [str(run.get("run_name")) for run in runs if run.get("run_name")]
    active_run_included = active_run_name in run_names
    active_family_included = any(_run_family_name(run_name) == active_run_family for run_name in run_names)
    aggregates = list(payload.get("aggregate_metrics", []))

    val_auroc = _aggregate_lookup(aggregates, "val", "auroc")
    test_auroc = _aggregate_lookup(aggregates, "test", "auroc")
    test_sensitivity = _aggregate_lookup(aggregates, "test", "sensitivity")
    test_specificity = _aggregate_lookup(aggregates, "test", "specificity")
    test_review_required = _aggregate_lookup(aggregates, "test", "review_required_count")
    test_sample_count = _aggregate_lookup(aggregates, "test", "sample_count")
    sample_count_mean = None if test_sample_count is None else _safe_float(test_sample_count.get("mean"))
    val_auroc_mean = None if val_auroc is None else _safe_float(val_auroc.get("mean"))
    test_auroc_mean = None if test_auroc is None else _safe_float(test_auroc.get("mean"))
    test_auroc_std = None if test_auroc is None else _safe_float(test_auroc.get("std"))
    test_sensitivity_std = None if test_sensitivity is None else _safe_float(test_sensitivity.get("std"))

    stability_status, warnings = _stability_status(
        run_count=len(runs),
        repeated_split=repeated_split,
        sample_count_mean=sample_count_mean,
        val_auroc_mean=val_auroc_mean,
        test_auroc_mean=test_auroc_mean,
        test_auroc_std=test_auroc_std,
        test_sensitivity_std=test_sensitivity_std,
    )
    validation_depth_level = _validation_depth_level(
        repeated_split=repeated_split,
        split_seed_count=len(set(split_seed_values)),
        run_count=len(runs),
        sample_count_mean=sample_count_mean,
    )
    support = _promotion_confidence_support(validation_depth_level, stability_status)

    notes = list(payload.get("notes", []))
    if repeated_split:
        notes.append("Study includes multiple subject-safe split seeds for broader validation depth.")
    else:
        notes.append("Study is multi-seed but not repeated across multiple subject-safe split seeds.")
    if active_family_included and not active_run_included:
        notes.append("Study covers the active model family but not the exact promoted run.")

    return ValidationStudySummary(
        study_name=str(payload.get("study_name", study_root.name)),
        study_root=str(study_root),
        evaluation_type="repeated_subject_safe_splits" if repeated_split else "multi_seed_fixed_split",
        run_count=len(runs),
        seed_count=len(set(payload.get("seeds", []))),
        split_seed_count=len(set(split_seed_values)),
        repeated_split=repeated_split,
        pair_seed_and_split_seed=bool(payload.get("pair_seed_and_split_seed", False)),
        active_run_included=active_run_included,
        active_family_included=active_family_included,
        selection_split=str(payload.get("selection_split", "val")),
        selection_metric=str(payload.get("selection_metric", "auroc")),
        best_experiment_name=payload.get("best_experiment_name"),
        best_run_name=_best_run_name(payload),
        best_selection_score=_safe_float(payload.get("best_selection_score")),
        validation_depth_level=validation_depth_level,
        stability_status=stability_status,
        promotion_confidence_support=support,
        aggregate_summary={
            "val_auroc": _aggregate_payload(val_auroc),
            "test_auroc": _aggregate_payload(test_auroc),
            "test_sensitivity": _aggregate_payload(test_sensitivity),
            "test_specificity": _aggregate_payload(test_specificity),
            "test_review_required_count": _aggregate_payload(test_review_required),
            "test_sample_count": _aggregate_payload(test_sample_count),
            "mean_val_test_auroc_gap": None
            if val_auroc_mean is None or test_auroc_mean is None
            else round(abs(val_auroc_mean - test_auroc_mean), 6),
        },
        warnings=warnings,
        notes=notes,
    )


def load_validation_depth_studies(
    *,
    limit: int = 10,
    settings: AppSettings | None = None,
) -> list[ValidationStudySummary]:
    """Load saved validation-depth studies for the active OASIS model family."""

    resolved_settings = settings or get_app_settings()
    active_entry = load_current_oasis_model_entry(settings=resolved_settings)
    studies_root = resolved_settings.outputs_root / "model_selection"
    if not studies_root.exists():
        return []
    studies: list[ValidationStudySummary] = []
    for study_dir in sorted(path for path in studies_root.iterdir() if path.is_dir()):
        summary_path = study_dir / "study_summary.json"
        if not summary_path.exists():
            continue
        payload = _load_json(summary_path)
        studies.append(
            _study_summary_from_payload(
                payload,
                study_root=study_dir,
                active_run_name=active_entry.run_name,
            )
        )
    depth_rank = {"strong": 3, "moderate": 2, "limited": 1, "insufficient": 0}
    stability_rank = {"stable": 3, "moderate": 2, "unstable": 1, "insufficient": 0}
    return sorted(
        studies,
        key=lambda study: (
            depth_rank.get(study.validation_depth_level, -1),
            stability_rank.get(study.stability_status, -1),
            study.active_run_included,
            study.active_family_included,
        ),
        reverse=True,
    )[:limit]


def build_validation_depth_dashboard(
    *,
    limit: int = 10,
    settings: AppSettings | None = None,
) -> ValidationDepthDashboard:
    """Build a compact validation-depth dashboard for the active model family."""

    resolved_settings = settings or get_app_settings()
    active_entry = load_current_oasis_model_entry(settings=resolved_settings)
    studies = load_validation_depth_studies(limit=limit, settings=resolved_settings)
    repeated_split_studies = [study for study in studies if study.repeated_split]
    direct_active_studies = [study for study in studies if study.active_run_included]
    related_family_studies = [study for study in studies if study.active_family_included]
    repeated_family_studies = [study for study in repeated_split_studies if study.active_family_included]

    overall_validation_depth = "insufficient"
    if any(study.validation_depth_level == "strong" and study.stability_status == "stable" for study in repeated_family_studies):
        overall_validation_depth = "strong"
    elif any(study.validation_depth_level in {"strong", "moderate"} for study in related_family_studies):
        overall_validation_depth = "moderate"
    elif related_family_studies:
        overall_validation_depth = "limited"

    strongest_study = studies[0] if studies else None
    if overall_validation_depth == "strong":
        recommended_action = "Validation depth is comparatively strong for the active model family; continue monitoring while prioritizing external validation."
    elif overall_validation_depth == "moderate":
        recommended_action = "Validation depth is improving, but repeat subject-safe studies and external validation should stay high priority."
    elif overall_validation_depth == "limited":
        recommended_action = "Current validation evidence is limited; add more repeated subject-safe studies before treating the baseline as stable."
    else:
        recommended_action = "Validation depth is insufficient; run repeated subject-safe studies before trusting one split or one seed."

    notes = [
        "Validation depth is about stability and generalization evidence, not just one good metric snapshot.",
        "Repeated subject-safe split evidence is stronger than multi-seed evidence on a single fixed split.",
    ]
    if not repeated_family_studies:
        notes.append("No repeated subject-safe study currently covers the active model family directly.")
    if direct_active_studies:
        notes.append("At least one saved study includes the exact promoted run.")
    elif related_family_studies:
        notes.append("Saved studies cover the active model family indirectly through related seed/split variants.")

    return ValidationDepthDashboard(
        generated_at_utc=_utc_now(),
        active_model_id=active_entry.model_id,
        active_run_name=active_entry.run_name,
        active_run_family=_run_family_name(active_entry.run_name),
        total_studies=len(studies),
        repeated_split_studies=len(repeated_split_studies),
        direct_active_run_studies=len(direct_active_studies),
        related_family_studies=len(related_family_studies),
        repeated_split_family_studies=len(repeated_family_studies),
        overall_validation_depth=overall_validation_depth,
        recommended_action=recommended_action,
        strongest_study_name=None if strongest_study is None else strongest_study.study_name,
        strongest_stability_status=None if strongest_study is None else strongest_study.stability_status,
        notes=notes,
        studies=studies,
    )
