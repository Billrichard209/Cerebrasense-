"""MONAI-oriented prediction helpers for OASIS-1 MRI decision-support inference."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.evaluation.metrics import compute_uncertainty_from_probabilities, threshold_binary_scores
from src.evaluation.oasis_run import LoadedCheckpoint, load_oasis_checkpoint
from src.models.factory import OASISModelConfig, build_model, load_oasis_model_config
from src.models.oasis_model import build_oasis_class_names
from src.transforms.oasis_transforms import (
    OASISSpatialConfig,
    OASISTransformConfig,
    build_oasis_infer_transforms,
    load_oasis_transform_config,
)
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_monai_inferer_symbols, load_torch_symbols

_load_monai_inferer_symbols = load_monai_inferer_symbols
_load_torch_symbols = load_torch_symbols


@dataclass(slots=True)
class PredictionResult:
    """Prediction container for downstream reporting and auditing."""

    predicted_index: int
    label: str
    confidence: float
    probabilities: list[float]
    source_dataset: str
    dataset_type: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OASISInferenceConfig:
    """Configuration for one auditable OASIS checkpoint inference call."""

    checkpoint_path: Path
    image_path: Path
    output_name: str = "oasis_prediction"
    model_config_path: Path | None = None
    threshold: float = 0.5
    image_size: tuple[int, int, int] = (64, 64, 64)
    device: str = "cpu"
    subject_id: str | None = None
    session_id: str | None = None
    scan_timestamp: str | None = None
    save_report: bool = True


@dataclass(slots=True)
class OASISCheckpointPrediction:
    """A checkpoint-backed OASIS inference result with report paths."""

    prediction: PredictionResult
    checkpoint: LoadedCheckpoint
    model_config: OASISModelConfig
    output_root: Path | None = None
    prediction_json_path: Path | None = None
    summary_report_path: Path | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize the prediction for API and JSON report outputs."""

        return {
            "dataset": self.prediction.source_dataset,
            "dataset_type": self.prediction.dataset_type,
            "predicted_label": self.prediction.predicted_index,
            "predicted_label_name": self.prediction.label,
            "confidence": self.prediction.confidence,
            "probabilities": self.prediction.probabilities,
            "meta": dict(self.prediction.meta),
            "checkpoint_path": str(self.checkpoint.path),
            "model": {
                "architecture": self.model_config.architecture,
                "class_names": list(self.model_config.class_names),
                "expected_input_shape": list(self.model_config.expected_input_shape),
            },
            "outputs": {
                "output_root": str(self.output_root) if self.output_root else None,
                "prediction_json": str(self.prediction_json_path) if self.prediction_json_path else None,
                "summary_report": str(self.summary_report_path) if self.summary_report_path else None,
            },
            "notes": [
                "This is decision-support model output, not a diagnosis.",
                "OASIS checkpoint inference is separate from Kaggle pipelines.",
            ],
        }


def _tensor_to_probability_list(tensor: Any) -> list[float]:
    """Convert a torch-like tensor or list to a Python probability list."""

    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "tolist"):
        values = tensor.tolist()
    else:
        values = list(tensor)
    return [float(value) for value in values]


def predict_oasis_image(
    image_path: str | Path,
    *,
    model: object,
    class_names: tuple[str, ...] | None = None,
    device: str = "cpu",
    transforms: object | None = None,
    transform_config: OASISTransformConfig | None = None,
    threshold: float | None = None,
    meta: dict[str, Any] | None = None,
) -> PredictionResult:
    """Run MONAI-style inference for one OASIS MRI image."""

    resolved_path = Path(image_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"OASIS inference image path does not exist: {resolved_path}")

    resolved_class_names = class_names or build_oasis_class_names()
    resolved_transforms = transforms or build_oasis_infer_transforms(transform_config or OASISTransformConfig())

    sample = resolved_transforms({"image": str(resolved_path)})
    image = sample["image"]
    batch = image.unsqueeze(0)

    symbols = _load_torch_symbols()
    torch = symbols["torch"]
    inferer = _load_monai_inferer_symbols()["SimpleInferer"]()

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = inferer(inputs=batch.to(device), network=model)
        probabilities = torch.softmax(logits, dim=1)[0]

    probability_list = _tensor_to_probability_list(probabilities)
    if threshold is not None and len(probability_list) == 2:
        predicted_index = threshold_binary_scores([probability_list[1]], threshold=threshold)[0]
    else:
        predicted_index = int(torch.argmax(probabilities, dim=0).item())
    predicted_label = (
        resolved_class_names[predicted_index]
        if predicted_index < len(resolved_class_names)
        else f"class_{predicted_index}"
    )
    uncertainty = compute_uncertainty_from_probabilities([probability_list])[0]
    result_meta = {"image": str(resolved_path), **(meta or {})}
    if threshold is not None:
        result_meta["threshold"] = threshold
    result_meta.update(uncertainty)
    return PredictionResult(
        predicted_index=predicted_index,
        label=predicted_label,
        confidence=float(max(probability_list)),
        probabilities=probability_list,
        source_dataset="oasis1",
        dataset_type="3d_volumes",
        meta=result_meta,
    )


def _build_inference_transform_config(cfg: OASISInferenceConfig) -> OASISTransformConfig:
    """Build deterministic OASIS inference transforms with the requested image size."""

    transform_cfg = load_oasis_transform_config()
    return OASISTransformConfig(
        load=transform_cfg.load,
        orientation=transform_cfg.orientation,
        spacing=transform_cfg.spacing,
        intensity=transform_cfg.intensity,
        skull_strip=transform_cfg.skull_strip,
        spatial=OASISSpatialConfig(spatial_size=cfg.image_size),
        augmentation=transform_cfg.augmentation,
    )


def load_oasis_checkpoint_model(
    cfg: OASISInferenceConfig,
) -> tuple[object, OASISModelConfig, LoadedCheckpoint]:
    """Load an OASIS model factory config and checkpoint for inference."""

    model_cfg = load_oasis_model_config(cfg.model_config_path)
    checkpoint = load_oasis_checkpoint(cfg.checkpoint_path, device=cfg.device)
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint.model_state_dict)
    return model, model_cfg, checkpoint


def _prediction_output_root(cfg: OASISInferenceConfig, settings: AppSettings) -> Path:
    """Resolve the output folder for a saved prediction."""

    safe_name = cfg.output_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return ensure_directory(settings.outputs_root / "predictions" / "oasis" / safe_name)


def save_oasis_prediction_report(
    result: OASISCheckpointPrediction,
    *,
    output_root: Path,
) -> OASISCheckpointPrediction:
    """Save JSON and Markdown reports for one OASIS inference result."""

    ensure_directory(output_root)
    prediction_json_path = output_root / "prediction.json"
    summary_report_path = output_root / "summary_report.md"
    saved_result = OASISCheckpointPrediction(
        prediction=result.prediction,
        checkpoint=result.checkpoint,
        model_config=result.model_config,
        output_root=output_root,
        prediction_json_path=prediction_json_path,
        summary_report_path=summary_report_path,
    )
    prediction_json_path.write_text(json.dumps(saved_result.to_payload(), indent=2), encoding="utf-8")
    summary_report_path.write_text(
        "\n".join(
            [
                f"# OASIS Prediction: {result.prediction.label}",
                "",
                "This output is for research decision-support development, not diagnosis.",
                "",
                "## Prediction",
                "",
                f"- predicted_label: {result.prediction.predicted_index}",
                f"- predicted_label_name: {result.prediction.label}",
                f"- confidence: {result.prediction.confidence:.6f}",
                f"- probabilities: {result.prediction.probabilities}",
                f"- threshold: {result.prediction.meta.get('threshold')}",
                "",
                "## Source",
                "",
                f"- image: {result.prediction.meta.get('image')}",
                f"- checkpoint: {result.checkpoint.path}",
            ]
        ),
        encoding="utf-8",
    )
    return saved_result


def predict_oasis_checkpoint(
    cfg: OASISInferenceConfig,
    *,
    settings: AppSettings | None = None,
) -> OASISCheckpointPrediction:
    """Load a trained OASIS checkpoint and run deterministic inference for one MRI volume."""

    resolved_settings = settings or get_app_settings()
    model, model_cfg, checkpoint = load_oasis_checkpoint_model(cfg)
    prediction = predict_oasis_image(
        cfg.image_path,
        model=model,
        class_names=tuple(model_cfg.class_names),
        device=cfg.device,
        transform_config=_build_inference_transform_config(cfg),
        threshold=cfg.threshold,
        meta={
            "subject_id": cfg.subject_id,
            "session_id": cfg.session_id,
            "scan_timestamp": cfg.scan_timestamp,
            "checkpoint_path": str(checkpoint.path),
        },
    )
    result = OASISCheckpointPrediction(
        prediction=prediction,
        checkpoint=checkpoint,
        model_config=model_cfg,
    )
    if not cfg.save_report:
        return result
    return save_oasis_prediction_report(
        result,
        output_root=_prediction_output_root(cfg, resolved_settings),
    )
