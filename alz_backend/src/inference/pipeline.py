"""Clean reusable inference pipeline for Alzheimer decision-support workflows.

Returned ``predict_scan`` schema:
- ``predicted_label``: integer class index selected by the configured threshold
- ``label_name``: model class name for the selected class
- ``probability_score``: positive-class probability for binary OASIS classification
- ``confidence_score``: max class probability
- ``model_name``: model architecture loaded from the model factory
- ``preprocessing_config``: YAML transform config path used to preprocess the scan
- ``input_metadata``: scan path, file format, size, and optional IDs
- ``ai_summary``: decision-support wording that avoids diagnosis claims
- ``outputs``: saved report and optional processed-slice debug paths

The first implementation targets OASIS-style 3D MRI volumes and keeps Kaggle
separate. Future multimodal or dataset-specific routers can call the same
function and branch based on explicit dataset/config inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from src.configs.runtime import AppSettings, get_app_settings
from src.evaluation.calibration import ConfidenceBandConfig, summarize_calibrated_confidence
from src.evaluation.metrics import compute_uncertainty_from_probabilities, threshold_binary_scores
from src.evaluation.oasis_run import load_oasis_checkpoint
from src.inference.serving import (
    BackendServingConfig,
    load_backend_serving_config,
    load_cached_oasis_model_bundle,
    load_active_oasis_registry_entry,
    resolve_oasis_decision_policy,
    resolve_confidence_config,
    resolve_inference_threshold,
)
from src.models.factory import OASIS_BINARY_CLASS_NAMES, build_model, load_oasis_model_config
from src.models.registry import ModelRegistryEntry
from src.security.audit import audit_sensitive_action
from src.security.disclaimers import STANDARD_DECISION_SUPPORT_DISCLAIMER, build_ai_summary
from src.storage import (
    InferenceMetadataRecord,
    ReviewQueueRecord,
    ScanRegistryRecord,
    persist_inference_record,
    persist_review_record,
    persist_scan_record,
)
from src.transforms.oasis_transforms import build_oasis_infer_transforms, load_oasis_transform_config
from src.transforms.oasis_transforms import OASISSpatialConfig, OASISTransformConfig
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_monai_inferer_symbols, load_torch_symbols

_load_monai_inferer_symbols = load_monai_inferer_symbols
_load_torch_symbols = load_torch_symbols


class PredictScanError(ValueError):
    """Raised when generic scan inference cannot proceed safely."""


@dataclass(slots=True, frozen=True)
class PredictScanOptions:
    """Optional runtime settings for ``predict_scan``."""

    output_name: str = "scan_prediction"
    threshold: float | None = None
    device: str = "cpu"
    model_config_path: Path | None = None
    save_debug_slices: bool = False
    subject_id: str | None = None
    session_id: str | None = None
    scan_timestamp: str | None = None
    confidence_config: ConfidenceBandConfig | None = None
    serving_config_path: Path | None = None
    use_cached_model: bool = True
    trace_id: str | None = None
    prediction_id: str | None = None


SUPPORTED_3D_SCAN_SUFFIXES = (".hdr", ".img", ".nii", ".nii.gz")


def _scan_suffix(path: Path) -> str:
    """Return a normalized medical image suffix."""

    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def validate_scan_path(
    scan_path: str | Path,
    *,
    max_file_size_bytes: int | None = None,
    allowed_suffixes: tuple[str, ...] | None = None,
) -> Path:
    """Validate that a scan path exists and uses a supported 3D MRI format."""

    resolved_path = Path(scan_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Scan path does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise PredictScanError(f"Scan path must point to a file: {resolved_path}")
    suffix = _scan_suffix(resolved_path)
    supported_suffixes = allowed_suffixes or SUPPORTED_3D_SCAN_SUFFIXES
    if suffix not in supported_suffixes:
        raise PredictScanError(
            f"Unsupported scan format {suffix!r}. Supported formats: {', '.join(supported_suffixes)}"
        )
    if max_file_size_bytes is not None and resolved_path.stat().st_size > max_file_size_bytes:
        raise PredictScanError(
            f"Scan file exceeds the serving safety limit of {max_file_size_bytes} bytes: {resolved_path}"
        )
    return resolved_path


def _build_input_metadata(
    scan_path: Path,
    *,
    options: PredictScanOptions,
) -> dict[str, Any]:
    """Build JSON-safe input metadata for reports and API responses."""

    return {
        "scan_path": str(scan_path),
        "source_path": str(scan_path),
        "file_name": scan_path.name,
        "file_format": _scan_suffix(scan_path),
        "file_size_bytes": int(scan_path.stat().st_size),
        "subject_id": options.subject_id,
        "session_id": options.session_id,
        "scan_timestamp": options.scan_timestamp,
        "dataset_assumption": "oasis1_3d_volume",
    }


def _apply_spatial_override(
    transform_config: OASISTransformConfig,
    *,
    spatial_size_override: tuple[int, int, int] | None,
) -> OASISTransformConfig:
    """Apply a registry-aware spatial-size override to the transform config."""

    if spatial_size_override is None:
        return transform_config
    return OASISTransformConfig(
        load=transform_config.load,
        orientation=transform_config.orientation,
        spacing=transform_config.spacing,
        intensity=transform_config.intensity,
        skull_strip=transform_config.skull_strip,
        spatial=OASISSpatialConfig(spatial_size=spatial_size_override),
        augmentation=transform_config.augmentation,
    )


def _load_preprocessed_sample(
    scan_path: Path,
    config_path: str | Path | None,
    *,
    spatial_size_override: tuple[int, int, int] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Apply deterministic MONAI inference transforms and return the sample."""

    transform_config_path = Path(config_path) if config_path is not None else None
    transform_config = load_oasis_transform_config(transform_config_path)
    transform_config = _apply_spatial_override(
        transform_config,
        spatial_size_override=spatial_size_override,
    )
    transforms = build_oasis_infer_transforms(transform_config)
    return transforms({"image": str(scan_path)}), transform_config_path


def _tensor_to_numpy(image: Any) -> np.ndarray:
    """Convert a MONAI/torch image tensor into a NumPy array."""

    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    if hasattr(image, "numpy"):
        array = image.numpy()
    else:
        array = np.asarray(image)
    return np.asarray(array)


def _save_debug_slices(image: Any, output_root: Path) -> list[str]:
    """Save representative processed slices for debugging preprocessing output."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slices_root = ensure_directory(output_root / "processed_slices")
    array = _tensor_to_numpy(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise PredictScanError(f"Expected processed image shaped (D, H, W) or (C, D, H, W), got {array.shape}.")
    slice_specs = {
        "axial_mid.png": array[array.shape[0] // 2, :, :],
        "coronal_mid.png": array[:, array.shape[1] // 2, :],
        "sagittal_mid.png": array[:, :, array.shape[2] // 2],
    }
    saved_paths: list[str] = []
    for file_name, slice_array in slice_specs.items():
        output_path = slices_root / file_name
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(np.rot90(slice_array), cmap="gray")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        saved_paths.append(str(output_path))
    return saved_paths


def _build_output_root(settings: AppSettings, output_name: str) -> Path:
    """Build the prediction output folder."""

    safe_name = output_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return ensure_directory(settings.outputs_root / "predictions" / safe_name)


def _ai_summary(*, label_name: str, probability_score: float, confidence_score: float) -> str:
    """Build decision-support wording without diagnosis claims."""

    return build_ai_summary(
        label_name=label_name,
        probability_score=probability_score,
        confidence_score=confidence_score,
    )


def _save_prediction_report(payload: dict[str, Any], output_root: Path) -> Path:
    """Save the prediction payload as JSON."""

    report_path = output_root / "prediction.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def _resolve_registry_entry(serving_config: BackendServingConfig) -> ModelRegistryEntry | None:
    """Load the active registry entry when available."""

    try:
        return load_active_oasis_registry_entry(serving_config)
    except FileNotFoundError:
        return None


def _max_file_size_bytes(serving_config: BackendServingConfig) -> int:
    """Convert the serving max file size from MB to bytes."""

    return int(serving_config.scan_validation.max_file_size_mb) * 1024 * 1024


def _review_reason(
    *,
    confidence_level: str | None,
    uncertainty: dict[str, Any],
    governance_hold: bool = False,
) -> str:
    """Build a compact reason for a queued human review."""

    if governance_hold:
        return "model_governance_hold"
    if confidence_level == "low":
        return "low_confidence_prediction"
    if float(uncertainty.get("normalized_entropy", 0.0)) >= 0.90:
        return "high_uncertainty_prediction"
    return "policy_review_required"


def _is_operational_hold_active(registry_entry: ModelRegistryEntry | None) -> bool:
    """Return whether the active registry entry is currently on governance hold."""

    return registry_entry is not None and str(registry_entry.operational_status).lower() == "hold"


def predict_scan(
    scan_path: str,
    checkpoint_path: str,
    config_path: str | None = None,
    *,
    options: PredictScanOptions | None = None,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """Run the clean backend inference pipeline for one structural MRI scan.

    Parameters are intentionally simple so this function can be reused by CLI,
    FastAPI, or background jobs:
    - ``scan_path``: path to NIfTI or Analyze MRI scan
    - ``checkpoint_path``: trained OASIS model checkpoint
    - ``config_path``: MONAI preprocessing YAML; when omitted, the default OASIS config is used
    """

    resolved_options = options or PredictScanOptions()
    resolved_settings = settings or get_app_settings()
    decision_policy = resolve_oasis_decision_policy(
        explicit_threshold=resolved_options.threshold,
        explicit_confidence_config=resolved_options.confidence_config,
        serving_config_path=resolved_options.serving_config_path,
        settings=resolved_settings,
    )
    serving_config = decision_policy.serving_config
    registry_entry = decision_policy.registry_entry
    resolved_threshold = decision_policy.threshold
    confidence_config = decision_policy.confidence_config
    resolved_scan_path = validate_scan_path(
        scan_path,
        max_file_size_bytes=_max_file_size_bytes(serving_config),
        allowed_suffixes=tuple(serving_config.scan_validation.allowed_suffixes),
    )
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved_checkpoint_path}")

    registry_image_size = None
    if registry_entry is not None and getattr(registry_entry, "image_size", None):
        registry_image_size = tuple(int(value) for value in registry_entry.image_size)

    sample, transform_config_path = _load_preprocessed_sample(
        resolved_scan_path,
        config_path,
        spatial_size_override=registry_image_size,
    )
    if resolved_options.use_cached_model:
        model_bundle = load_cached_oasis_model_bundle(
            checkpoint_path=resolved_checkpoint_path,
            model_config_path=resolved_options.model_config_path,
            device=resolved_options.device,
        )
        model_config = model_bundle.model_config
        checkpoint = model_bundle.checkpoint
        model = model_bundle.model
    else:
        model_config = load_oasis_model_config(resolved_options.model_config_path)
        checkpoint = load_oasis_checkpoint(resolved_checkpoint_path, device=resolved_options.device)
        model = build_model(model_config)
        model.load_state_dict(checkpoint.model_state_dict)
        model = model.to(resolved_options.device)
        model.eval()

    image = sample["image"]
    batch = image.unsqueeze(0)
    torch = _load_torch_symbols()["torch"]
    inferer = _load_monai_inferer_symbols()["SimpleInferer"]()
    model = model.to(resolved_options.device)
    model.eval()
    with torch.no_grad():
        logits = inferer(inputs=batch.to(resolved_options.device), network=model)
        probabilities_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()
    probabilities = [float(value) for value in probabilities_tensor.tolist()]
    calibrated = summarize_calibrated_confidence([probabilities], config=confidence_config)[0]
    probability_score = float(probabilities[1]) if len(probabilities) > 1 else float(max(probabilities))
    predicted_label = threshold_binary_scores([probability_score], threshold=resolved_threshold)[0]
    class_names = tuple(model_config.class_names) if model_config.class_names else OASIS_BINARY_CLASS_NAMES
    label_name = class_names[predicted_label] if predicted_label < len(class_names) else f"class_{predicted_label}"
    confidence_score = float(calibrated.confidence_score)
    uncertainty = compute_uncertainty_from_probabilities([probabilities])[0]
    governance_hold_active = _is_operational_hold_active(registry_entry)
    serving_restrictions = dict(registry_entry.serving_restrictions) if registry_entry is not None else {}
    review_required = bool(calibrated.review_flag or governance_hold_active or serving_restrictions.get("force_manual_review"))
    output_root = _build_output_root(resolved_settings, resolved_options.output_name)
    debug_slice_paths = _save_debug_slices(image, output_root) if resolved_options.save_debug_slices else []
    preprocessing_reference = str(transform_config_path or "default:oasis_transforms.yaml")
    prediction_id = resolved_options.prediction_id or str(uuid4())
    trace_id = resolved_options.trace_id or str(uuid4())

    payload: dict[str, Any] = {
        "prediction_id": prediction_id,
        "trace_id": trace_id,
        "predicted_label": predicted_label,
        "label_name": label_name,
        "probability_score": probability_score,
        "calibrated_probability_score": calibrated.calibrated_probability_score,
        "confidence_score": confidence_score,
        "confidence_level": calibrated.confidence_level,
        "review_flag": review_required,
        "risk_category": label_name,
        "model_name": model_config.architecture,
        "preprocessing_config": preprocessing_reference,
        "preprocessing_overrides": {
            "spatial_size": list(registry_image_size),
        }
        if registry_image_size is not None
        else {},
        "input_metadata": _build_input_metadata(resolved_scan_path, options=resolved_options),
        "checkpoint_path": str(resolved_checkpoint_path),
        "active_model_id": registry_entry.model_id if registry_entry is not None else None,
        "operational_status": None if registry_entry is None else registry_entry.operational_status,
        "serving_restrictions": serving_restrictions,
        "ai_summary": _ai_summary(
            label_name=label_name,
            probability_score=probability_score,
            confidence_score=confidence_score,
        ),
        "probabilities": {
            class_names[index] if index < len(class_names) else f"class_{index}": probability
            for index, probability in enumerate(probabilities)
        },
        "uncertainty": uncertainty,
        "decision_support_only": True,
        "clinical_disclaimer": STANDARD_DECISION_SUPPORT_DISCLAIMER,
        "abnormal_regions": [],
        "heatmap_visualization": None,
        "outputs": {
            "output_root": str(output_root),
            "prediction_json": None,
            "processed_slices": debug_slice_paths,
        },
        "notes": [
            STANDARD_DECISION_SUPPORT_DISCLAIMER,
            "Heatmap and abnormal-region localization are placeholders until explainability/segmentation modules are validated.",
            "Low-confidence outputs should be reviewed before operational use.",
        ],
    }
    if registry_image_size is not None:
        payload["notes"].append(
            f"Inference preprocessing used the active registry spatial size override {list(registry_image_size)} to stay aligned with the promoted model input shape."
        )
    if governance_hold_active:
        payload["notes"].append(
            "Active model is currently under governance hold; every prediction requires manual review and should not be used as the default operating output."
        )
        payload["ai_summary"] = (
            payload["ai_summary"]
            + " The active model is currently under governance hold and requires manual review before any operational use."
        )
    report_path = _save_prediction_report(payload, output_root)
    payload["outputs"]["prediction_json"] = str(report_path)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    scan_record = ScanRegistryRecord(
        subject_id=resolved_options.subject_id,
        session_id=resolved_options.session_id,
        source_path=str(resolved_scan_path),
        file_format=_scan_suffix(resolved_scan_path),
        file_size_bytes=int(resolved_scan_path.stat().st_size),
        dataset="oasis1",
        metadata=payload["input_metadata"],
    )
    persist_scan_record(scan_record, settings=resolved_settings)
    persist_inference_record(
        InferenceMetadataRecord(
            inference_id=prediction_id,
            trace_id=trace_id,
            scan_id=scan_record.scan_id,
            subject_id=resolved_options.subject_id,
            session_id=resolved_options.session_id,
            model_name=model_config.architecture,
            checkpoint_path=str(resolved_checkpoint_path),
            output_path=str(report_path),
            predicted_label=predicted_label,
            label_name=label_name,
            confidence_level=calibrated.confidence_level,
            review_flag=review_required,
            payload=payload,
        ),
        settings=resolved_settings,
    )
    if review_required:
        persist_review_record(
            ReviewQueueRecord(
                inference_id=prediction_id,
                trace_id=trace_id,
                scan_id=scan_record.scan_id,
                subject_id=resolved_options.subject_id,
                session_id=resolved_options.session_id,
                source_path=str(resolved_scan_path),
                model_name=model_config.architecture,
                confidence_level=calibrated.confidence_level,
                probability_score=probability_score,
                output_path=str(report_path),
                reason=_review_reason(
                    confidence_level=calibrated.confidence_level,
                    uncertainty=uncertainty,
                    governance_hold=governance_hold_active or bool(serving_restrictions.get("force_manual_review")),
                ),
                payload={
                    "prediction_id": prediction_id,
                    "trace_id": trace_id,
                    "predicted_label": predicted_label,
                    "label_name": label_name,
                    "model_name": model_config.architecture,
                    "model_run_name": registry_entry.run_name if registry_entry is not None else None,
                    "active_model_id": registry_entry.model_id if registry_entry is not None else None,
                    "checkpoint_path": str(resolved_checkpoint_path),
                    "operational_status": None if registry_entry is None else registry_entry.operational_status,
                    "serving_restrictions": serving_restrictions,
                    "confidence_level": calibrated.confidence_level,
                    "review_flag": review_required,
                    "probability_score": probability_score,
                    "calibrated_probability_score": calibrated.calibrated_probability_score,
                },
            ),
            settings=resolved_settings,
        )
    audit_sensitive_action(
        action="predict_scan",
        subject_id=resolved_options.subject_id,
        metadata={
            "output_name": resolved_options.output_name,
            "scan_file_name": resolved_scan_path.name,
            "checkpoint_name": resolved_checkpoint_path.name,
            "trace_id": trace_id,
        },
    )
    return payload
