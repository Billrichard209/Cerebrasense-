"""Benchmark registration and promotion gating for active OASIS models."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.configs.runtime import AppSettings, get_app_settings
from src.evaluation.thresholds import ThresholdCalibrationResult, calibrate_binary_threshold
from src.models.registry import (
    ModelRegistryEntry,
    load_current_oasis_model_entry,
    promote_oasis_checkpoint,
    save_oasis_model_entry,
)
from src.storage import (
    BenchmarkMetadataRecord,
    PromotionMetadataRecord,
    persist_benchmark_record,
    persist_promotion_record,
)
from src.utils.io_utils import ensure_directory, resolve_project_root

CANONICAL_METRIC_KEYS = {
    "sample_count",
    "accuracy",
    "auroc",
    "precision",
    "recall_sensitivity",
    "sensitivity",
    "specificity",
    "f1",
    "mean_confidence",
    "mean_calibrated_confidence",
    "mean_normalized_entropy",
    "mean_uncertainty_score",
    "review_required_count",
}


def _utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class BenchmarkRegistryEntry:
    """Frozen benchmark manifest registration for one evaluation split."""

    benchmark_id: str
    benchmark_name: str
    dataset: str
    split_name: str
    manifest_path: str
    manifest_hash_sha256: str
    sample_count: int
    subject_count: int | None
    subject_id_column: str | None
    subject_safe: bool
    label_distribution: dict[str, int] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=_utc_now)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe benchmark payload."""

        return asdict(self)


@dataclass(slots=True, frozen=True)
class PromotionPolicy:
    """Explicit pass/fail thresholds for OASIS checkpoint promotion."""

    policy_name: str = "oasis_research_gate_v1"
    dataset: str = "oasis1"
    require_subject_safe_benchmark: bool = True
    minimum_validation_auroc: float = 0.75
    minimum_test_auroc: float = 0.78
    minimum_test_sensitivity: float = 0.80
    maximum_review_required_rate: float = 0.50
    minimum_mean_calibrated_confidence: float = 0.65
    minimum_test_sample_count: int = 30

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe policy payload."""

        return asdict(self)


@dataclass(slots=True)
class PromotionDecision:
    """Evaluated checkpoint promotion decision with explicit evidence checks."""

    decision_id: str
    run_name: str
    policy_name: str
    approved: bool
    checked_at_utc: str
    benchmark_id: str | None
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe decision payload."""

        return asdict(self)


@dataclass(slots=True)
class PromotionWorkflowResult:
    """Artifacts produced by benchmark registration and promotion gating."""

    benchmark_entry: BenchmarkRegistryEntry
    benchmark_path: Path
    decision: PromotionDecision
    decision_path: Path
    active_registry_entry: ModelRegistryEntry | None
    active_registry_path: Path | None


@dataclass(slots=True)
class ActiveModelCalibrationResult:
    """Threshold calibration artifacts for the active approved OASIS model."""

    registry_entry: ModelRegistryEntry
    registry_path: Path
    calibration_result: ThresholdCalibrationResult
    validation_predictions_path: Path
    test_predictions_path: Path | None


def default_oasis_promotion_policy_path() -> Path:
    """Return the default OASIS promotion policy config path."""

    return resolve_project_root() / "configs" / "oasis_promotion_policy.yaml"


def _resolve_workspace_relative_path(path_value: str | Path, *, settings: AppSettings) -> Path:
    """Resolve absolute or workspace-relative artifact paths."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return settings.workspace_root / path


def _predictions_path_from_metrics_path(metrics_path: Path) -> Path:
    """Resolve a sibling predictions CSV next to a metrics JSON file."""

    predictions_path = metrics_path.with_name("predictions.csv")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Could not find predictions.csv next to metrics file: {metrics_path}")
    return predictions_path


def _default_calibration_output_dir(*, entry: ModelRegistryEntry, selection_metric: str, settings: AppSettings) -> Path:
    """Choose a stable calibration artifact directory for the active model."""

    checkpoint_path = _resolve_workspace_relative_path(entry.checkpoint_path, settings=settings)
    if checkpoint_path.parent.name == "checkpoints" and checkpoint_path.parent.parent.exists():
        return checkpoint_path.parent.parent / "calibration" / f"threshold_{selection_metric}"
    return settings.outputs_root / "model_registry" / "calibration_history" / entry.run_name / selection_metric


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Read a CSV or JSONL manifest into dictionaries."""

    lower_name = path.name.lower()
    if lower_name.endswith(".csv"):
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            return [dict(row) for row in csv.DictReader(file_handle)]
    if lower_name.endswith(".jsonl"):
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if stripped:
                    payload = json.loads(stripped)
                    if not isinstance(payload, dict):
                        raise ValueError(f"JSONL manifest rows must be objects: {path}")
                    rows.append(payload)
        return rows
    if lower_name.endswith(".json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        raise ValueError(f"JSON manifest must contain a list of samples: {path}")
    raise ValueError(f"Unsupported benchmark manifest format: {path.suffix}")


def _count_values(rows: list[dict[str, Any]], column_name: str) -> dict[str, int]:
    """Count manifest column values without extra dependencies."""

    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(column_name)
        key = "null" if value in {None, ""} else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def register_benchmark(
    *,
    manifest_path: str | Path,
    benchmark_name: str,
    dataset: str = "oasis1",
    split_name: str = "test",
    subject_id_column: str = "subject_id",
    label_column: str = "label",
    output_path: str | Path | None = None,
    settings: AppSettings | None = None,
) -> tuple[BenchmarkRegistryEntry, Path]:
    """Register one frozen benchmark manifest with a hash and summary stats."""

    resolved_settings = settings or get_app_settings()
    resolved_manifest_path = Path(manifest_path)
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {resolved_manifest_path}")

    rows = _read_manifest_rows(resolved_manifest_path)
    if not rows:
        raise ValueError(f"Benchmark manifest is empty: {resolved_manifest_path}")

    subject_values = {
        str(row.get(subject_id_column)).strip()
        for row in rows
        if row.get(subject_id_column) not in {None, ""}
    }
    subject_safe = any(row.get(subject_id_column) not in {None, ""} for row in rows)
    notes: list[str] = []
    if not subject_safe:
        notes.append(
            f"Manifest does not expose {subject_id_column!r}; subject-safe leakage checks are weaker for this benchmark."
        )

    benchmark_prefix = f"{dataset}_{split_name}_"
    benchmark_id = benchmark_name if benchmark_name.startswith(benchmark_prefix) else f"{benchmark_prefix}{benchmark_name}"

    entry = BenchmarkRegistryEntry(
        benchmark_id=benchmark_id,
        benchmark_name=benchmark_name,
        dataset=dataset,
        split_name=split_name,
        manifest_path=str(resolved_manifest_path),
        manifest_hash_sha256=_sha256_file(resolved_manifest_path),
        sample_count=len(rows),
        subject_count=len(subject_values) if subject_safe else None,
        subject_id_column=subject_id_column if subject_safe else None,
        subject_safe=subject_safe,
        label_distribution=_count_values(rows, label_column),
        notes=notes,
    )

    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else resolved_settings.outputs_root / "benchmarks" / f"{benchmark_name}.json"
    )
    ensure_directory(resolved_output_path.parent)
    resolved_output_path.write_text(json.dumps(entry.to_dict(), indent=2), encoding="utf-8")
    persist_benchmark_record(
        BenchmarkMetadataRecord(
            benchmark_id=entry.benchmark_id,
            benchmark_name=entry.benchmark_name,
            dataset=entry.dataset,
            split_name=entry.split_name,
            manifest_path=entry.manifest_path,
            manifest_hash_sha256=entry.manifest_hash_sha256,
            sample_count=entry.sample_count,
            subject_safe=entry.subject_safe,
            payload=entry.to_dict(),
        ),
        settings=resolved_settings,
    )
    return entry, resolved_output_path


def load_oasis_promotion_policy(
    config_path: str | Path | None = None,
    *,
    settings: AppSettings | None = None,
) -> PromotionPolicy:
    """Load the OASIS promotion policy from YAML or return defaults."""

    resolved_path = Path(config_path) if config_path is not None else default_oasis_promotion_policy_path()
    if not resolved_path.exists():
        return PromotionPolicy()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Promotion policy YAML must decode to a dictionary.")
    return PromotionPolicy(**payload)


def _load_metrics_payload(source: str | Path | dict[str, Any], *, split: str | None = None) -> dict[str, Any]:
    """Load metrics from several artifact shapes without caller-specific logic."""

    payload = source
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Metrics source must decode to a JSON object.")

    if split and isinstance(payload.get("evaluations"), list):
        for item in payload["evaluations"]:
            if isinstance(item, dict) and item.get("split") == split and isinstance(item.get("metrics"), dict):
                return dict(item["metrics"])
    if isinstance(payload.get("metrics"), dict):
        return dict(payload["metrics"])
    if CANONICAL_METRIC_KEYS & set(payload.keys()):
        return dict(payload)
    raise ValueError("Could not locate a metrics object in the provided source.")


def _make_check(*, passed: bool, observed: Any, expected: Any, comparator: str) -> dict[str, Any]:
    """Create one structured promotion check payload."""

    return {
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "comparator": comparator,
    }


def evaluate_oasis_promotion_candidate(
    *,
    run_name: str,
    benchmark_entry: BenchmarkRegistryEntry,
    validation_metrics: str | Path | dict[str, Any],
    test_metrics: str | Path | dict[str, Any],
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """Evaluate one OASIS checkpoint against an explicit promotion policy."""

    resolved_policy = policy or PromotionPolicy()
    val_metrics = _load_metrics_payload(validation_metrics, split="val")
    test_metrics_payload = _load_metrics_payload(test_metrics, split="test")
    test_sample_count = int(test_metrics_payload.get("sample_count", 0))
    test_sensitivity = float(
        test_metrics_payload.get("sensitivity", test_metrics_payload.get("recall_sensitivity", 0.0))
    )
    review_required_count = float(test_metrics_payload.get("review_required_count", 0.0))
    review_required_rate = review_required_count / test_sample_count if test_sample_count > 0 else 1.0
    mean_calibrated_confidence = float(
        test_metrics_payload.get("mean_calibrated_confidence", test_metrics_payload.get("mean_confidence", 0.0))
    )

    checks = {
        "dataset_match": _make_check(
            passed=benchmark_entry.dataset == resolved_policy.dataset,
            observed=benchmark_entry.dataset,
            expected=resolved_policy.dataset,
            comparator="==",
        ),
        "subject_safe_benchmark": _make_check(
            passed=(not resolved_policy.require_subject_safe_benchmark) or benchmark_entry.subject_safe,
            observed=benchmark_entry.subject_safe,
            expected=resolved_policy.require_subject_safe_benchmark,
            comparator="subject_safe_required",
        ),
        "test_sample_count": _make_check(
            passed=test_sample_count >= resolved_policy.minimum_test_sample_count,
            observed=test_sample_count,
            expected=resolved_policy.minimum_test_sample_count,
            comparator=">=",
        ),
        "validation_auroc": _make_check(
            passed=float(val_metrics.get("auroc", 0.0)) >= resolved_policy.minimum_validation_auroc,
            observed=float(val_metrics.get("auroc", 0.0)),
            expected=resolved_policy.minimum_validation_auroc,
            comparator=">=",
        ),
        "test_auroc": _make_check(
            passed=float(test_metrics_payload.get("auroc", 0.0)) >= resolved_policy.minimum_test_auroc,
            observed=float(test_metrics_payload.get("auroc", 0.0)),
            expected=resolved_policy.minimum_test_auroc,
            comparator=">=",
        ),
        "test_sensitivity": _make_check(
            passed=test_sensitivity >= resolved_policy.minimum_test_sensitivity,
            observed=test_sensitivity,
            expected=resolved_policy.minimum_test_sensitivity,
            comparator=">=",
        ),
        "review_required_rate": _make_check(
            passed=review_required_rate <= resolved_policy.maximum_review_required_rate,
            observed=review_required_rate,
            expected=resolved_policy.maximum_review_required_rate,
            comparator="<=",
        ),
        "mean_calibrated_confidence": _make_check(
            passed=mean_calibrated_confidence >= resolved_policy.minimum_mean_calibrated_confidence,
            observed=mean_calibrated_confidence,
            expected=resolved_policy.minimum_mean_calibrated_confidence,
            comparator=">=",
        ),
    }

    failed_checks = [name for name, check in checks.items() if not check["passed"]]
    notes = [
        "Promotion checks use held-out evaluation artifacts and a lightweight confidence proxy.",
        "This gate supports research governance; it is not a clinical validation protocol.",
    ]
    val_test_gap = abs(float(val_metrics.get("auroc", 0.0)) - float(test_metrics_payload.get("auroc", 0.0)))
    if val_test_gap > 0.10:
        notes.append(
            f"Validation/test AUROC drift is {val_test_gap:.3f}; confirm the checkpoint is stable across repeated splits."
        )
    if test_sample_count < 50:
        notes.append("Held-out benchmark remains small; external validation is still needed.")
    if review_required_rate > 0.25:
        notes.append("A meaningful fraction of cases still require manual review under the current confidence policy.")

    return PromotionDecision(
        decision_id=f"promotion_{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
        run_name=run_name,
        policy_name=resolved_policy.policy_name,
        approved=not failed_checks,
        checked_at_utc=_utc_now(),
        benchmark_id=benchmark_entry.benchmark_id,
        checks=checks,
        failed_checks=failed_checks,
        evidence_summary={
            "benchmark_name": benchmark_entry.benchmark_name,
            "validation_auroc": float(val_metrics.get("auroc", 0.0)),
            "test_auroc": float(test_metrics_payload.get("auroc", 0.0)),
            "test_sensitivity": test_sensitivity,
            "review_required_rate": review_required_rate,
            "mean_calibrated_confidence": mean_calibrated_confidence,
            "test_sample_count": test_sample_count,
        },
        notes=notes,
    )


def run_oasis_promotion_workflow(
    *,
    run_name: str,
    checkpoint_path: str | Path,
    validation_metrics_path: str | Path,
    test_metrics_path: str | Path,
    manifest_path: str | Path,
    benchmark_name: str,
    split_name: str = "test",
    model_config_path: str | Path | None = None,
    preprocessing_config_path: str | Path | None = None,
    image_size: tuple[int, int, int] = (64, 64, 64),
    threshold_calibration_path: str | Path | None = None,
    recommended_threshold: float | None = None,
    policy_config_path: str | Path | None = None,
    settings: AppSettings | None = None,
) -> PromotionWorkflowResult:
    """Register a benchmark, evaluate policy gates, and promote if approved."""

    resolved_settings = settings or get_app_settings()
    benchmark_entry, benchmark_path = register_benchmark(
        manifest_path=manifest_path,
        benchmark_name=benchmark_name,
        dataset="oasis1",
        split_name=split_name,
        settings=resolved_settings,
    )
    policy = load_oasis_promotion_policy(policy_config_path, settings=resolved_settings)
    decision = evaluate_oasis_promotion_candidate(
        run_name=run_name,
        benchmark_entry=benchmark_entry,
        validation_metrics=validation_metrics_path,
        test_metrics=test_metrics_path,
        policy=policy,
    )

    decision_payload = {
        "run_name": run_name,
        "benchmark": benchmark_entry.to_dict(),
        "policy": policy.to_dict(),
        "decision": decision.to_dict(),
        "evidence": {
            "checkpoint_path": str(Path(checkpoint_path)),
            "validation_metrics_path": str(Path(validation_metrics_path)),
            "test_metrics_path": str(Path(test_metrics_path)),
            "threshold_calibration_path": None
            if threshold_calibration_path is None
            else str(Path(threshold_calibration_path)),
            "benchmark_registry_path": str(benchmark_path),
        },
        "active_registry_updated": False,
        "active_registry_path": None,
    }
    decision_root = ensure_directory(resolved_settings.outputs_root / "model_registry" / "promotion_history")
    decision_path = decision_root / f"{decision.decision_id}.json"

    active_registry_entry: ModelRegistryEntry | None = None
    active_registry_path: Path | None = None
    if decision.approved:
        active_registry_entry, active_registry_path = promote_oasis_checkpoint(
            run_name=run_name,
            checkpoint_path=checkpoint_path,
            val_metrics_path=validation_metrics_path,
            test_metrics_path=test_metrics_path,
            model_config_path=model_config_path,
            preprocessing_config_path=preprocessing_config_path,
            image_size=image_size,
            recommended_threshold=recommended_threshold,
            threshold_calibration_path=threshold_calibration_path,
            benchmark=benchmark_entry.to_dict(),
            promotion_decision=decision.to_dict(),
            evidence=decision_payload["evidence"],
            settings=resolved_settings,
        )
        decision_payload["active_registry_updated"] = True
        decision_payload["active_registry_path"] = str(active_registry_path)

    decision_path.write_text(json.dumps(decision_payload, indent=2), encoding="utf-8")
    persist_promotion_record(
        PromotionMetadataRecord(
            model_id="oasis_current_baseline",
            run_name=run_name,
            benchmark_id=benchmark_entry.benchmark_id,
            policy_name=policy.policy_name,
            approved=decision.approved,
            output_path=str(active_registry_path) if active_registry_path is not None else None,
            payload=decision_payload,
        ),
        settings=resolved_settings,
    )

    return PromotionWorkflowResult(
        benchmark_entry=benchmark_entry,
        benchmark_path=benchmark_path,
        decision=decision,
        decision_path=decision_path,
        active_registry_entry=active_registry_entry,
        active_registry_path=active_registry_path,
    )


def calibrate_active_oasis_model(
    *,
    registry_path: str | Path | None = None,
    validation_predictions_path: str | Path | None = None,
    test_predictions_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    selection_metric: str = "balanced_accuracy",
    threshold_step: float = 0.01,
    settings: AppSettings | None = None,
) -> ActiveModelCalibrationResult:
    """Calibrate the active approved OASIS model and update its registry entry."""

    resolved_settings = settings or get_app_settings()
    entry = load_current_oasis_model_entry(path=registry_path, settings=resolved_settings)
    resolved_registry_path = (
        Path(registry_path)
        if registry_path is not None
        else resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    )
    validation_metrics_path = entry.evidence.get("validation_metrics_path")
    test_metrics_path = entry.evidence.get("test_metrics_path")

    resolved_val_predictions_path = (
        Path(validation_predictions_path)
        if validation_predictions_path is not None
        else _predictions_path_from_metrics_path(
            _resolve_workspace_relative_path(validation_metrics_path, settings=resolved_settings)
        )
    )
    resolved_test_predictions_path = (
        None
        if test_predictions_path is None and not test_metrics_path
        else (
            Path(test_predictions_path)
            if test_predictions_path is not None
            else _predictions_path_from_metrics_path(
                _resolve_workspace_relative_path(test_metrics_path, settings=resolved_settings)
            )
        )
    )
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_calibration_output_dir(
            entry=entry,
            selection_metric=selection_metric,
            settings=resolved_settings,
        )
    )
    calibration_result = calibrate_binary_threshold(
        validation_predictions_path=resolved_val_predictions_path,
        test_predictions_path=resolved_test_predictions_path,
        output_dir=resolved_output_dir,
        selection_metric=selection_metric,
        threshold_step=threshold_step,
    )

    entry.recommended_threshold = float(calibration_result.threshold)
    entry.threshold_calibration = json.loads(
        calibration_result.calibration_report_path.read_text(encoding="utf-8")
    )
    entry.evidence = {
        **dict(entry.evidence),
        "threshold_calibration_path": str(calibration_result.calibration_report_path),
        "threshold_grid_path": str(calibration_result.threshold_grid_path),
        "calibration_selection_metric": selection_metric,
    }
    notes = list(entry.notes)
    calibration_note = (
        f"Serving threshold recalibrated on validation predictions using {selection_metric} "
        f"with threshold={calibration_result.threshold:.2f}."
    )
    if calibration_note not in notes:
        notes.append(calibration_note)
    entry.notes = notes
    save_oasis_model_entry(entry, path=resolved_registry_path, settings=resolved_settings)

    return ActiveModelCalibrationResult(
        registry_entry=entry,
        registry_path=resolved_registry_path,
        calibration_result=calibration_result,
        validation_predictions_path=resolved_val_predictions_path,
        test_predictions_path=resolved_test_predictions_path,
    )
