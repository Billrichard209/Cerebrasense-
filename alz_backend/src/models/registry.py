"""Small model registry helpers for promoted research checkpoints."""

from __future__ import annotations

import json
import shutil
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


@dataclass(slots=True)
class ImportedOASISRunResult:
    """Result of importing a promoted Colab/Drive run into the local backend."""

    run_name: str
    local_run_root: Path
    local_checkpoint_path: Path
    local_registry_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary."""

        return {
            "run_name": self.run_name,
            "local_run_root": str(self.local_run_root),
            "local_checkpoint_path": str(self.local_checkpoint_path),
            "local_registry_path": None if self.local_registry_path is None else str(self.local_registry_path),
        }


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


def _load_registry_payload(path: Path) -> dict[str, Any]:
    """Load a model registry payload from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Registry payload must be a JSON object: {path}")
    return payload


def _resolve_registry_path_value(raw_path: str | None, *, settings: AppSettings) -> str | None:
    """Resolve a registry path value into an absolute local path when possible."""

    if raw_path in {None, ""}:
        return raw_path
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return str(candidate)

    parts = candidate.parts
    if parts and parts[0].lower() == settings.project_root.name.lower():
        return str((settings.workspace_root / candidate).resolve())
    return str((settings.project_root / candidate).resolve())


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


def import_promoted_oasis_run(
    *,
    source_run_root: str | Path | None = None,
    source_registry_path: str | Path | None = None,
    source_runtime_root: str | Path | None = None,
    run_name: str | None = None,
    registry_output_path: str | Path | None = None,
    overwrite: bool = False,
    settings: AppSettings | None = None,
) -> ImportedOASISRunResult:
    """Copy a Drive-exported OASIS run locally and rewrite registry paths for local serving."""

    resolved_settings = settings or get_app_settings()
    resolved_source_runtime_root = Path(source_runtime_root).expanduser().resolve() if source_runtime_root is not None else None
    if resolved_source_runtime_root is not None and not resolved_source_runtime_root.exists():
        raise FileNotFoundError(
            f"Source runtime root not found: {resolved_source_runtime_root}. "
            "Provide the real local Google Drive sync path to Cerebrasensecloud/backend_runtime."
        )
    if resolved_source_runtime_root is not None and not resolved_source_runtime_root.is_dir():
        raise ValueError(f"Source runtime root must be a directory: {resolved_source_runtime_root}")
    resolved_source_registry_path = (
        Path(source_registry_path).expanduser().resolve()
        if source_registry_path is not None
        else (
            resolved_source_runtime_root / "outputs" / "model_registry" / "oasis_current_baseline.json"
            if resolved_source_runtime_root is not None
            else None
        )
    )

    resolved_run_name = run_name
    if resolved_run_name is None and resolved_source_registry_path is not None:
        if not resolved_source_registry_path.exists():
            raise FileNotFoundError(
                f"Source runtime registry not found: {resolved_source_registry_path}. "
                "Ensure backend_runtime/outputs/model_registry/oasis_current_baseline.json is synced locally "
                "or provide --run-name explicitly with --source-run-root."
            )
        resolved_run_name = str(_load_registry_payload(resolved_source_registry_path).get("run_name") or "").strip() or None

    if source_run_root is None:
        if resolved_source_runtime_root is None:
            raise ValueError("Provide source_run_root or source_runtime_root when importing a promoted OASIS run.")
        if resolved_run_name is None:
            raise ValueError(
                "Could not infer the run name from the runtime registry. "
                "Provide run_name explicitly or ensure the runtime registry exists."
            )
        resolved_source_run_root = resolved_source_runtime_root / "outputs" / "runs" / "oasis" / resolved_run_name
    else:
        resolved_source_run_root = Path(source_run_root).expanduser().resolve()

    if not resolved_source_run_root.exists():
        raise FileNotFoundError(f"Source run root not found: {resolved_source_run_root}")
    if not resolved_source_run_root.is_dir():
        raise ValueError(f"Source run root must be a directory: {resolved_source_run_root}")

    resolved_run_name = resolved_run_name or resolved_source_run_root.name
    local_run_root = resolved_settings.outputs_root / "runs" / "oasis" / resolved_run_name
    if local_run_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Local run root already exists: {local_run_root}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(local_run_root)

    ensure_directory(local_run_root.parent)
    shutil.copytree(resolved_source_run_root, local_run_root)

    if resolved_source_registry_path is not None and resolved_source_registry_path.exists():
        registry_payload = _load_registry_payload(resolved_source_registry_path)
        checkpoint_name = Path(str(registry_payload.get("checkpoint_path", "best_model.pt"))).name
    else:
        checkpoint_name = "best_model.pt"

    local_checkpoint_path = local_run_root / "checkpoints" / checkpoint_name
    if not local_checkpoint_path.exists():
        raise FileNotFoundError(f"Expected imported checkpoint not found: {local_checkpoint_path}")

    local_registry_path: Path | None = None
    if resolved_source_registry_path is not None and resolved_source_registry_path.exists():
        registry_payload = _load_registry_payload(resolved_source_registry_path)
        local_model_config = resolved_settings.project_root / "configs" / "oasis_model.yaml"
        local_preprocessing_config = resolved_settings.project_root / "configs" / "oasis_transforms.yaml"
        registry_payload["run_name"] = resolved_run_name
        registry_payload["checkpoint_path"] = str(local_checkpoint_path)
        registry_payload["model_config_path"] = (
            str(local_model_config) if local_model_config.exists() else registry_payload.get("model_config_path")
        )
        registry_payload["preprocessing_config_path"] = (
            str(local_preprocessing_config)
            if local_preprocessing_config.exists()
            else registry_payload.get("preprocessing_config_path")
        )

        local_registry_path = (
            Path(registry_output_path)
            if registry_output_path is not None
            else resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
        )
        ensure_directory(local_registry_path.parent)
        local_registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

    return ImportedOASISRunResult(
        run_name=resolved_run_name,
        local_run_root=local_run_root,
        local_checkpoint_path=local_checkpoint_path,
        local_registry_path=local_registry_path,
    )


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
    if not isinstance(payload, dict):
        raise ValueError(f"Registry payload must be a JSON object: {resolved_path}")
    payload["checkpoint_path"] = _resolve_registry_path_value(payload.get("checkpoint_path"), settings=resolved_settings)
    payload["model_config_path"] = _resolve_registry_path_value(payload.get("model_config_path"), settings=resolved_settings)
    payload["preprocessing_config_path"] = _resolve_registry_path_value(
        payload.get("preprocessing_config_path"),
        settings=resolved_settings,
    )
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
