"""Small model registry helpers for promoted research checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.security.disclaimers import STANDARD_DECISION_SUPPORT_DISCLAIMER
from src.utils.io_utils import ensure_directory


@dataclass(slots=True)
class ModelRegistryEntry:
    """Metadata for a promoted model checkpoint used by backend inference."""

    registry_version: str
    model_id: str
    dataset: str
    run_name: str
    checkpoint_path: str
    model_config_path: str | None
    preprocessing_config_path: str | None
    image_size: list[int]
    promoted_at_utc: str
    decision_support_only: bool
    clinical_disclaimer: str
    default_threshold: float = 0.5
    recommended_threshold: float = 0.5
    threshold_calibration: dict[str, Any] = field(default_factory=dict)
    temperature_scaling: dict[str, Any] = field(default_factory=lambda: {"enabled": False, "temperature": 1.0})
    confidence_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "high_confidence_min": 0.85,
            "medium_confidence_min": 0.65,
            "high_entropy_max": 0.35,
            "medium_entropy_max": 0.90,
        }
    )
    benchmark: dict[str, Any] = field(default_factory=dict)
    promotion_decision: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    operational_status: str = "active"
    serving_restrictions: dict[str, Any] = field(
        default_factory=lambda: {
            "force_manual_review": False,
            "allow_prediction_output": True,
            "block_as_operational_default": False,
        }
    )
    hold_decision: dict[str, Any] = field(default_factory=dict)
    review_monitoring: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe registry payload."""

        return asdict(self)


def _load_metrics(path: Path | None) -> dict[str, Any]:
    """Load a metrics JSON file if supplied."""

    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics payload must be a JSON object: {path}")
    return payload


def _load_threshold_calibration(path: Path | None) -> dict[str, Any]:
    """Load a threshold calibration JSON report if supplied."""

    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Threshold calibration report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Threshold calibration payload must be a JSON object: {path}")
    return payload


def promote_oasis_checkpoint(
    *,
    run_name: str,
    checkpoint_path: str | Path,
    val_metrics_path: str | Path | None = None,
    test_metrics_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    preprocessing_config_path: str | Path | None = None,
    image_size: tuple[int, int, int] = (64, 64, 64),
    default_threshold: float = 0.5,
    recommended_threshold: float | None = None,
    threshold_calibration_path: str | Path | None = None,
    benchmark: dict[str, Any] | None = None,
    promotion_decision: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    settings: AppSettings | None = None,
) -> tuple[ModelRegistryEntry, Path]:
    """Promote an OASIS checkpoint into a registry JSON entry."""

    resolved_settings = settings or get_app_settings()
    resolved_checkpoint = Path(checkpoint_path)
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

    resolved_model_config = Path(model_config_path) if model_config_path is not None else resolved_settings.project_root / "configs" / "oasis_model.yaml"
    resolved_preprocessing_config = (
        Path(preprocessing_config_path)
        if preprocessing_config_path is not None
        else resolved_settings.project_root / "configs" / "oasis_transforms.yaml"
    )
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    )
    threshold_calibration = _load_threshold_calibration(
        Path(threshold_calibration_path) if threshold_calibration_path is not None else None
    )
    resolved_recommended_threshold = (
        recommended_threshold
        if recommended_threshold is not None
        else float(threshold_calibration.get("threshold", default_threshold))
    )

    entry = ModelRegistryEntry(
        registry_version="1.0",
        model_id="oasis_current_baseline",
        dataset="oasis1",
        run_name=run_name,
        checkpoint_path=str(resolved_checkpoint),
        model_config_path=str(resolved_model_config) if resolved_model_config.exists() else None,
        preprocessing_config_path=str(resolved_preprocessing_config) if resolved_preprocessing_config.exists() else None,
        image_size=[int(value) for value in image_size],
        promoted_at_utc=datetime.now(timezone.utc).isoformat(),
        decision_support_only=True,
        clinical_disclaimer=STANDARD_DECISION_SUPPORT_DISCLAIMER,
        default_threshold=float(default_threshold),
        recommended_threshold=float(resolved_recommended_threshold),
        threshold_calibration=threshold_calibration,
        temperature_scaling={"enabled": False, "temperature": 1.0},
        confidence_policy={
            "high_confidence_min": 0.85,
            "medium_confidence_min": 0.65,
            "high_entropy_max": 0.35,
            "medium_entropy_max": 0.90,
        },
        benchmark=dict(benchmark or {}),
        promotion_decision=dict(promotion_decision or {}),
        evidence=dict(evidence or {}),
        validation_metrics=_load_metrics(Path(val_metrics_path) if val_metrics_path is not None else None),
        test_metrics=_load_metrics(Path(test_metrics_path) if test_metrics_path is not None else None),
        operational_status="active",
        serving_restrictions={
            "force_manual_review": False,
            "allow_prediction_output": True,
            "block_as_operational_default": False,
        },
        hold_decision={},
        review_monitoring={},
        notes=[
            "Promoted checkpoint is a research baseline for decision support only.",
            "Use held-out metrics as development evidence, not clinical validation.",
            "OASIS and Kaggle remain separate; this registry entry is OASIS-only.",
        ],
    )
    ensure_directory(resolved_output_path.parent)
    resolved_output_path.write_text(json.dumps(entry.to_dict(), indent=2), encoding="utf-8")
    return entry, resolved_output_path


def load_current_oasis_model_entry(
    path: str | Path | None = None,
    *,
    settings: AppSettings | None = None,
) -> ModelRegistryEntry:
    """Load the current promoted OASIS baseline registry entry."""

    resolved_settings = settings or get_app_settings()
    resolved_path = (
        Path(path)
        if path is not None
        else resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    )
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    return ModelRegistryEntry(**payload)


def save_oasis_model_entry(
    entry: ModelRegistryEntry,
    path: str | Path | None = None,
    *,
    settings: AppSettings | None = None,
) -> Path:
    """Persist a fully resolved OASIS model registry entry."""

    resolved_settings = settings or get_app_settings()
    resolved_path = (
        Path(path)
        if path is not None
        else resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    )
    ensure_directory(resolved_path.parent)
    resolved_path.write_text(json.dumps(entry.to_dict(), indent=2), encoding="utf-8")
    return resolved_path
