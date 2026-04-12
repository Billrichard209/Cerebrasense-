"""Read-only promotion workflow helpers for candidate-vs-active model review."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.models.governance import (
    BenchmarkRegistryEntry,
    evaluate_oasis_promotion_candidate,
    load_oasis_promotion_policy,
)
from src.models.registry import ModelRegistryEntry, load_current_oasis_model_entry


def _safe_float(value: Any) -> float | None:
    """Convert a metric value to ``float`` when possible."""

    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class CandidatePreflightSummary:
    """Read-only promotion preflight result for one tracked experiment."""

    evaluable: bool
    approved: bool | None
    benchmark_name: str | None
    benchmark_reused_from_active: bool
    policy_name: str | None
    checks: dict[str, Any] = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return asdict(self)


@dataclass(slots=True)
class PromotionCandidate:
    """Tracked experiment surfaced as a promotion candidate."""

    experiment_name: str
    run_name: str
    experiment_root: str
    tags: list[str] = field(default_factory=list)
    primary_split: str | None = None
    best_checkpoint_path: str | None = None
    current_active: bool = False
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    comparison_to_active: dict[str, dict[str, float | None]] = field(default_factory=dict)
    promotion_preflight: CandidatePreflightSummary = field(
        default_factory=lambda: CandidatePreflightSummary(
            evaluable=False,
            approved=None,
            benchmark_name=None,
            benchmark_reused_from_active=False,
            policy_name=None,
            notes=[],
        )
    )
    tracked_artifacts: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return {
            **asdict(self),
            "promotion_preflight": self.promotion_preflight.to_dict(),
        }


@dataclass(slots=True)
class PromotionStudySummary:
    """Compact model-selection study summary for promotion review."""

    study_name: str
    study_root: str
    selection_split: str
    selection_metric: str
    best_experiment_name: str | None
    best_run_name: str | None
    best_selection_score: float | None
    best_checkpoint_path: str | None
    aggregate_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromotionHistoryEntry:
    """Compact summary of one saved promotion decision."""

    decision_id: str | None
    run_name: str
    checked_at_utc: str | None
    approved: bool
    benchmark_name: str | None
    policy_name: str | None
    failed_checks: list[str] = field(default_factory=list)
    output_path: str | None = None
    history_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _active_benchmark_entry(active_entry: ModelRegistryEntry) -> BenchmarkRegistryEntry | None:
    """Return the active benchmark entry when present."""

    benchmark = dict(active_entry.benchmark)
    if not benchmark:
        return None
    try:
        return BenchmarkRegistryEntry(**benchmark)
    except TypeError:
        return None


def _metric_delta(candidate: dict[str, Any], active: dict[str, Any], key: str) -> float | None:
    """Return the candidate-minus-active metric delta when both values exist."""

    candidate_value = _safe_float(candidate.get(key))
    active_value = _safe_float(active.get(key))
    if candidate_value is None or active_value is None:
        return None
    return round(candidate_value - active_value, 6)


def _comparison_to_active(
    *,
    candidate_validation: dict[str, Any],
    candidate_test: dict[str, Any],
    active_entry: ModelRegistryEntry,
) -> dict[str, dict[str, float | None]]:
    """Build validation/test metric deltas versus the active model."""

    tracked_keys = (
        "accuracy",
        "auroc",
        "f1",
        "sensitivity",
        "specificity",
        "mean_calibrated_confidence",
        "review_required_count",
    )
    return {
        "validation": {
            key: _metric_delta(candidate_validation, active_entry.validation_metrics, key)
            for key in tracked_keys
        },
        "test": {
            key: _metric_delta(candidate_test, active_entry.test_metrics, key)
            for key in tracked_keys
        },
    }


def _candidate_preflight(
    *,
    run_name: str,
    candidate_validation: dict[str, Any],
    candidate_test: dict[str, Any],
    active_entry: ModelRegistryEntry,
    settings: AppSettings,
) -> CandidatePreflightSummary:
    """Run a read-only policy preflight for one experiment candidate."""

    if not candidate_validation or not candidate_test:
        return CandidatePreflightSummary(
            evaluable=False,
            approved=None,
            benchmark_name=None,
            benchmark_reused_from_active=False,
            policy_name=None,
            notes=["Candidate is missing validation or test metrics, so promotion preflight is unavailable."],
        )
    benchmark_entry = _active_benchmark_entry(active_entry)
    if benchmark_entry is None:
        return CandidatePreflightSummary(
            evaluable=False,
            approved=None,
            benchmark_name=None,
            benchmark_reused_from_active=False,
            policy_name=None,
            notes=["Active registry does not expose a benchmark entry to reuse for preflight comparison."],
        )
    decision = evaluate_oasis_promotion_candidate(
        run_name=run_name,
        benchmark_entry=benchmark_entry,
        validation_metrics=candidate_validation,
        test_metrics=candidate_test,
        policy=load_oasis_promotion_policy(settings=settings),
    )
    notes = list(decision.notes)
    notes.append(
        "Preflight reuses the currently active benchmark as a reference proxy. Run the explicit promotion workflow before changing the active model."
    )
    return CandidatePreflightSummary(
        evaluable=True,
        approved=decision.approved,
        benchmark_name=benchmark_entry.benchmark_name,
        benchmark_reused_from_active=True,
        policy_name=decision.policy_name,
        checks=dict(decision.checks),
        failed_checks=list(decision.failed_checks),
        evidence_summary=dict(decision.evidence_summary),
        notes=notes,
    )


def _candidate_from_experiment_dir(
    experiment_dir: Path,
    *,
    active_entry: ModelRegistryEntry,
    settings: AppSettings,
) -> PromotionCandidate | None:
    """Parse one tracked experiment into a promotion candidate payload."""

    summary_path = experiment_dir / "experiment_summary.json"
    if not summary_path.exists():
        return None
    summary = _load_json(summary_path)
    experiment_name = str(summary.get("experiment_name", experiment_dir.name))
    run_name = str(summary.get("run_name", experiment_name))
    training_payload = dict(summary.get("training", {}))
    evaluations = dict(summary.get("evaluations", {}))
    validation_metrics = dict(evaluations.get("val", {}).get("metrics", {}))
    test_metrics = dict(evaluations.get("test", {}).get("metrics", {}))
    best_checkpoint_path = training_payload.get("best_checkpoint_path")
    current_active = run_name == active_entry.run_name
    comparison_to_active = _comparison_to_active(
        candidate_validation=validation_metrics,
        candidate_test=test_metrics,
        active_entry=active_entry,
    )
    preflight = _candidate_preflight(
        run_name=run_name,
        candidate_validation=validation_metrics,
        candidate_test=test_metrics,
        active_entry=active_entry,
        settings=settings,
    )
    notes = [
        "Candidate surfaces tracked experiment outputs for research promotion review.",
        "Promotion preflight is advisory and does not replace explicit benchmark registration plus full promotion workflow.",
    ]
    if current_active:
        notes.append("This candidate matches the current active model run.")
    return PromotionCandidate(
        experiment_name=experiment_name,
        run_name=run_name,
        experiment_root=str(experiment_dir),
        tags=list(summary.get("tags", [])),
        primary_split=summary.get("primary_split"),
        best_checkpoint_path=best_checkpoint_path,
        current_active=current_active,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        comparison_to_active=comparison_to_active,
        promotion_preflight=preflight,
        tracked_artifacts={
            "summary_path": str(summary_path),
            "config_path": str(experiment_dir / "config.yaml") if (experiment_dir / "config.yaml").exists() else None,
            "final_metrics_path": str(experiment_dir / "final_metrics.json")
            if (experiment_dir / "final_metrics.json").exists()
            else None,
            "comparison_row_path": str(experiment_dir / "comparison_row.json")
            if (experiment_dir / "comparison_row.json").exists()
            else None,
        },
        notes=notes,
    )


def _candidate_sort_key(candidate: PromotionCandidate) -> tuple[int, float, float]:
    """Sort candidates by readiness first, then test AUROC, then validation AUROC."""

    ready_score = 1 if candidate.promotion_preflight.approved else 0
    test_auroc = _safe_float(candidate.test_metrics.get("auroc")) or float("-inf")
    val_auroc = _safe_float(candidate.validation_metrics.get("auroc")) or float("-inf")
    return (ready_score, test_auroc, val_auroc)


def load_promotion_candidates(
    *,
    limit: int = 10,
    settings: AppSettings | None = None,
) -> list[PromotionCandidate]:
    """Load tracked experiment candidates sorted by promotion readiness and performance."""

    resolved_settings = settings or get_app_settings()
    active_entry = load_current_oasis_model_entry(settings=resolved_settings)
    experiments_root = resolved_settings.outputs_root / "experiments"
    if not experiments_root.exists():
        return []
    candidates: list[PromotionCandidate] = []
    for experiment_dir in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
        candidate = _candidate_from_experiment_dir(
            experiment_dir,
            active_entry=active_entry,
            settings=resolved_settings,
        )
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=_candidate_sort_key, reverse=True)[:limit]


def _aggregate_lookup(aggregates: list[dict[str, Any]], split: str, metric_name: str) -> dict[str, Any] | None:
    """Return one aggregate metric block from a study summary."""

    for aggregate in aggregates:
        if aggregate.get("split") == split and aggregate.get("metric_name") == metric_name:
            return aggregate
    return None


def load_promotion_studies(
    *,
    limit: int = 10,
    settings: AppSettings | None = None,
) -> list[PromotionStudySummary]:
    """Load recent model-selection studies relevant to promotion review."""

    resolved_settings = settings or get_app_settings()
    studies_root = resolved_settings.outputs_root / "model_selection"
    if not studies_root.exists():
        return []
    studies: list[PromotionStudySummary] = []
    for study_dir in sorted(path for path in studies_root.iterdir() if path.is_dir()):
        summary_path = study_dir / "study_summary.json"
        if not summary_path.exists():
            continue
        payload = _load_json(summary_path)
        aggregates = list(payload.get("aggregate_metrics", []))
        val_auroc = _aggregate_lookup(aggregates, "val", "auroc")
        test_auroc = _aggregate_lookup(aggregates, "test", "auroc")
        studies.append(
            PromotionStudySummary(
                study_name=str(payload.get("study_name", study_dir.name)),
                study_root=str(study_dir),
                selection_split=str(payload.get("selection_split", "val")),
                selection_metric=str(payload.get("selection_metric", "auroc")),
                best_experiment_name=payload.get("best_experiment_name"),
                best_run_name=next(
                    (
                        str(run.get("run_name"))
                        for run in payload.get("runs", [])
                        if run.get("experiment_name") == payload.get("best_experiment_name")
                    ),
                    None,
                ),
                best_selection_score=_safe_float(payload.get("best_selection_score")),
                best_checkpoint_path=payload.get("best_checkpoint_path"),
                aggregate_summary={
                    "val_auroc_mean": None if val_auroc is None else _safe_float(val_auroc.get("mean")),
                    "val_auroc_ci95": None
                    if val_auroc is None
                    else [_safe_float(val_auroc.get("ci95_low")), _safe_float(val_auroc.get("ci95_high"))],
                    "test_auroc_mean": None if test_auroc is None else _safe_float(test_auroc.get("mean")),
                    "test_auroc_ci95": None
                    if test_auroc is None
                    else [_safe_float(test_auroc.get("ci95_low")), _safe_float(test_auroc.get("ci95_high"))],
                },
                notes=list(payload.get("notes", [])),
            )
        )
    return sorted(
        studies,
        key=lambda study: study.best_selection_score if study.best_selection_score is not None else float("-inf"),
        reverse=True,
    )[:limit]


def load_promotion_history_entries(
    *,
    limit: int = 10,
    settings: AppSettings | None = None,
) -> list[PromotionHistoryEntry]:
    """Load recent saved promotion decisions."""

    resolved_settings = settings or get_app_settings()
    history_root = resolved_settings.outputs_root / "model_registry" / "promotion_history"
    if not history_root.exists():
        return []
    entries: list[PromotionHistoryEntry] = []
    history_files = sorted(history_root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for history_path in history_files[:limit]:
        payload = _load_json(history_path)
        decision = dict(payload.get("decision", {}))
        benchmark = dict(payload.get("benchmark", {}))
        entries.append(
            PromotionHistoryEntry(
                decision_id=decision.get("decision_id"),
                run_name=str(payload.get("run_name", decision.get("run_name", history_path.stem))),
                checked_at_utc=decision.get("checked_at_utc"),
                approved=bool(decision.get("approved", False)),
                benchmark_name=benchmark.get("benchmark_name"),
                policy_name=decision.get("policy_name"),
                failed_checks=list(decision.get("failed_checks", [])),
                output_path=payload.get("active_registry_path"),
                history_path=str(history_path),
                notes=list(decision.get("notes", [])),
            )
        )
    return entries
