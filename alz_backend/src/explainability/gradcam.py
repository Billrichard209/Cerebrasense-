"""Grad-CAM style explainability for OASIS 3D MRI classifiers.

Limitations:
- Grad-CAM heatmaps are approximate model-attribution visualizations, not
  anatomical segmentations or proof of disease.
- Region importance values are coarse array-region summaries of the heatmap,
  not validated neuroanatomical measurements.
- This module is separated from training and reuses deterministic inference
  preprocessing to avoid train-time augmentation leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.configs.runtime import AppSettings, get_app_settings
from src.evaluation.calibration import ConfidenceBandConfig, summarize_calibrated_confidence
from src.evaluation.metrics import compute_uncertainty_from_probabilities
from src.evaluation.oasis_run import load_oasis_checkpoint
from src.inference.pipeline import validate_scan_path
from src.inference.serving import load_backend_serving_config
from src.models.factory import OASIS_BINARY_CLASS_NAMES, build_model, load_oasis_model_config
from src.models.registry import load_current_oasis_model_entry
from src.security.audit import audit_sensitive_action
from src.security.disclaimers import EXPLANATION_LIMITATION, STANDARD_DECISION_SUPPORT_DISCLAIMER
from src.transforms.oasis_transforms import (
    OASISSpatialConfig,
    OASISTransformConfig,
    build_oasis_infer_transforms,
    load_oasis_transform_config,
)
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_torch_symbols

from .reporting import (
    classify_explanation_quality,
    describe_highlighted_regions,
    interpret_confidence,
    summarize_heatmap_intensity,
)

_load_torch_symbols = load_torch_symbols


class ExplainabilityError(ValueError):
    """Raised when an explanation cannot be produced safely."""


@dataclass(slots=True, frozen=True)
class ExplainScanConfig:
    """Configuration for one Grad-CAM style scan explanation."""

    scan_path: Path
    checkpoint_path: Path
    preprocessing_config_path: Path | None = None
    model_config_path: Path | None = None
    output_name: str = "scan_explanation"
    target_layer: str = "auto"
    target_class: int | None = None
    device: str = "cpu"
    image_size: tuple[int, int, int] | None = None
    slice_axis: str = "axial"
    slice_indices: tuple[int, ...] | None = None
    save_saliency: bool = True
    true_label: int | None = None
    confidence_config: ConfidenceBandConfig | None = None


@dataclass(slots=True)
class ExplanationResult:
    """Saved explanation output paths and report payload."""

    output_root: Path
    report_path: Path
    overlay_paths: list[Path]
    saliency_paths: list[Path]
    payload: dict[str, Any]


def _load_transform_config(
    cfg: ExplainScanConfig,
    *,
    image_size_override: tuple[int, int, int] | None = None,
) -> OASISTransformConfig:
    """Load deterministic inference transform config with optional image-size override."""

    transform_cfg = load_oasis_transform_config(cfg.preprocessing_config_path)
    resolved_image_size = image_size_override or cfg.image_size
    if resolved_image_size is None:
        return transform_cfg
    return OASISTransformConfig(
        load=transform_cfg.load,
        orientation=transform_cfg.orientation,
        spacing=transform_cfg.spacing,
        intensity=transform_cfg.intensity,
        skull_strip=transform_cfg.skull_strip,
        spatial=OASISSpatialConfig(spatial_size=resolved_image_size),
        augmentation=transform_cfg.augmentation,
    )


def _resolve_registry_image_size(
    cfg: ExplainScanConfig,
    *,
    settings: AppSettings | None = None,
) -> tuple[int, int, int] | None:
    """Return an active-registry image-size override when the checkpoint matches."""

    resolved_settings = settings or get_app_settings()
    try:
        serving_config = load_backend_serving_config(settings=resolved_settings)
        registry_entry = load_current_oasis_model_entry(serving_config.active_oasis_model_registry, settings=resolved_settings)
    except FileNotFoundError:
        return None

    registry_checkpoint_path = Path(registry_entry.checkpoint_path)
    if not registry_checkpoint_path.is_absolute():
        registry_checkpoint_path = resolved_settings.project_root / registry_checkpoint_path
    try:
        if registry_checkpoint_path.resolve() != cfg.checkpoint_path.resolve():
            return None
    except FileNotFoundError:
        return None
    if not getattr(registry_entry, "image_size", None):
        return None
    return tuple(int(value) for value in registry_entry.image_size)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert torch/MONAI tensor-like values to NumPy arrays."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _normalize_array(value: Any) -> np.ndarray:
    """Normalize an array into [0, 1] for visualization."""

    array = _tensor_to_numpy(value).astype(float)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value <= min_value:
        return np.zeros_like(array, dtype=float)
    return (array - min_value) / (max_value - min_value)


def _resolve_target_module(model: object, target_layer: str, torch: object) -> tuple[str, object]:
    """Resolve a configured target layer or select the last Conv3d module."""

    modules = dict(model.named_modules())
    if target_layer != "auto":
        if target_layer not in modules:
            examples = sorted(name for name in modules if name)[:30]
            raise ExplainabilityError(f"Target layer {target_layer!r} not found. Available examples: {examples}")
        return target_layer, modules[target_layer]

    conv3d_cls = torch.nn.Conv3d
    conv_candidates = [(name, module) for name, module in modules.items() if isinstance(module, conv3d_cls)]
    if not conv_candidates:
        examples = sorted(name for name in modules if name)[:30]
        raise ExplainabilityError(f"Could not auto-select a Conv3d target layer. Available examples: {examples}")
    return conv_candidates[-1]


def _class_name(class_index: int, class_names: tuple[str, ...]) -> str:
    """Resolve a model class name."""

    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    return f"class_{class_index}"


def _compute_gradcam(
    *,
    model: object,
    image: object,
    target_layer: str,
    target_class: int | None,
    device: str,
) -> dict[str, Any]:
    """Compute Grad-CAM and input saliency arrays for one preprocessed image."""

    torch = _load_torch_symbols()["torch"]
    model = model.to(device)
    model.eval()
    resolved_layer_name, target_module = _resolve_target_module(model, target_layer, torch)
    activations: dict[str, object] = {}
    gradients: dict[str, object] = {}

    def forward_hook(_module: object, _inputs: tuple[object, ...], output: object) -> None:
        activations["value"] = output

    def backward_hook(_module: object, _grad_input: tuple[object, ...], grad_output: tuple[object, ...]) -> None:
        gradients["value"] = grad_output[0]

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    try:
        batch = image.unsqueeze(0).to(device)
        batch.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0]
        selected_class = int(target_class) if target_class is not None else int(torch.argmax(probabilities).item())
        score = logits[0, selected_class]
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise ExplainabilityError("Grad-CAM hooks did not capture activations and gradients.")
        activation = activations["value"]
        gradient = gradients["value"]
        if activation.ndim != 5 or gradient.ndim != 5:
            raise ExplainabilityError(
                f"Expected 5D target activations/gradients, got {tuple(activation.shape)} and {tuple(gradient.shape)}."
            )
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=tuple(batch.shape[2:]),
            mode="trilinear",
            align_corners=False,
        )[0, 0]
        saliency = batch.grad.detach().abs().amax(dim=1)[0]
        return {
            "cam": _normalize_array(cam),
            "saliency": _normalize_array(saliency),
            "probabilities": [float(value) for value in probabilities.detach().cpu().tolist()],
            "selected_class": selected_class,
            "target_layer": resolved_layer_name,
        }
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _volume_from_image(image: object) -> np.ndarray:
    """Return a normalized 3D base image array from a channel-first tensor."""

    array = _normalize_array(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ExplainabilityError(f"Expected image shaped (D, H, W) or (C, D, H, W), got {array.shape}.")
    return array


def _selected_indices(volume_shape: tuple[int, int, int], axis: str, requested: tuple[int, ...] | None) -> list[int]:
    """Resolve selected slice indices."""

    axis_to_dim = {"axial": 0, "coronal": 1, "sagittal": 2}
    if axis not in axis_to_dim:
        raise ExplainabilityError(f"Unsupported slice axis {axis!r}. Use axial, coronal, or sagittal.")
    size = volume_shape[axis_to_dim[axis]]
    if requested:
        return [int(index) for index in requested if 0 <= int(index) < size]
    return sorted(set([size // 4, size // 2, (3 * size) // 4]))


def _slice(array: np.ndarray, *, axis: str, index: int) -> np.ndarray:
    """Extract one slice from a 3D volume."""

    if axis == "axial":
        return array[index, :, :]
    if axis == "coronal":
        return array[:, index, :]
    if axis == "sagittal":
        return array[:, :, index]
    raise ExplainabilityError(f"Unsupported slice axis {axis!r}.")


def _save_overlay(
    *,
    base_slice: np.ndarray,
    heatmap_slice: np.ndarray,
    output_path: Path,
    title: str,
) -> Path:
    """Save one heatmap overlay figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(np.rot90(base_slice), cmap="gray")
    ax.imshow(np.rot90(heatmap_slice), cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.05)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def _save_explanation_slices(
    *,
    image: object,
    cam: np.ndarray,
    saliency: np.ndarray,
    cfg: ExplainScanConfig,
    output_root: Path,
) -> tuple[list[Path], list[Path]]:
    """Save Grad-CAM and saliency overlays for selected slices."""

    base = _volume_from_image(image)
    indices = _selected_indices(tuple(base.shape), cfg.slice_axis, cfg.slice_indices)
    overlay_paths: list[Path] = []
    saliency_paths: list[Path] = []
    for index in indices:
        base_slice = _slice(base, axis=cfg.slice_axis, index=index)
        cam_slice = _slice(cam, axis=cfg.slice_axis, index=index)
        overlay_paths.append(
            _save_overlay(
                base_slice=base_slice,
                heatmap_slice=cam_slice,
                output_path=output_root / "gradcam_overlays" / f"{cfg.slice_axis}_{index:03d}.png",
                title=f"Grad-CAM {cfg.slice_axis} slice {index}",
            )
        )
        if cfg.save_saliency:
            saliency_slice = _slice(saliency, axis=cfg.slice_axis, index=index)
            saliency_paths.append(
                _save_overlay(
                    base_slice=base_slice,
                    heatmap_slice=saliency_slice,
                    output_path=output_root / "saliency_overlays" / f"{cfg.slice_axis}_{index:03d}.png",
                    title=f"Saliency {cfg.slice_axis} slice {index}",
                )
            )
    return overlay_paths, saliency_paths


def _region_importance(cam: np.ndarray) -> dict[str, float]:
    """Compute coarse heatmap-region proxy importance values."""

    depth, height, width = cam.shape
    regions = {
        "first_depth_half_proxy": cam[: depth // 2, :, :],
        "second_depth_half_proxy": cam[depth // 2 :, :, :],
        "first_height_half_proxy": cam[:, : height // 2, :],
        "second_height_half_proxy": cam[:, height // 2 :, :],
        "first_width_half_proxy": cam[:, :, : width // 2],
        "second_width_half_proxy": cam[:, :, width // 2 :],
    }
    return {name: float(np.mean(values)) for name, values in regions.items()}


def _output_root(settings: AppSettings, output_name: str) -> Path:
    """Resolve the explanation output folder."""

    safe_name = output_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return ensure_directory(settings.outputs_root / "visualizations" / "explanations" / safe_name)


def explain_scan(
    cfg: ExplainScanConfig,
    *,
    settings: AppSettings | None = None,
) -> ExplanationResult:
    """Generate Grad-CAM-style explanation artifacts for one MRI scan."""

    resolved_settings = settings or get_app_settings()
    scan_path = validate_scan_path(cfg.scan_path)
    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")
    registry_image_size = None if cfg.image_size is not None else _resolve_registry_image_size(cfg, settings=resolved_settings)
    transform_config = _load_transform_config(cfg, image_size_override=registry_image_size)
    sample = build_oasis_infer_transforms(transform_config)({"image": str(scan_path)})
    image = sample["image"]

    model_cfg = load_oasis_model_config(cfg.model_config_path)
    class_names = tuple(model_cfg.class_names) if model_cfg.class_names else OASIS_BINARY_CLASS_NAMES
    checkpoint = load_oasis_checkpoint(cfg.checkpoint_path, device=cfg.device)
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint.model_state_dict)

    gradcam = _compute_gradcam(
        model=model,
        image=image,
        target_layer=cfg.target_layer,
        target_class=cfg.target_class,
        device=cfg.device,
    )
    output_root = _output_root(resolved_settings, cfg.output_name)
    overlay_paths, saliency_paths = _save_explanation_slices(
        image=image,
        cam=gradcam["cam"],
        saliency=gradcam["saliency"],
        cfg=cfg,
        output_root=output_root,
    )
    selected_class = int(gradcam["selected_class"])
    probabilities = gradcam["probabilities"]
    uncertainty = compute_uncertainty_from_probabilities([probabilities])[0]
    calibrated = summarize_calibrated_confidence([probabilities], config=cfg.confidence_config)[0]
    heatmap_summary = summarize_heatmap_intensity(gradcam["cam"])
    region_proxy = _region_importance(gradcam["cam"])
    prediction_correctness = (
        "correct" if cfg.true_label is not None and int(cfg.true_label) == selected_class else
        "incorrect" if cfg.true_label is not None else
        None
    )
    payload = {
        "method": "grad_cam_style_3d",
        "dataset_assumption": "oasis1_3d_volume",
        "scan_path": str(scan_path),
        "checkpoint_path": str(cfg.checkpoint_path),
        "preprocessing_config": str(cfg.preprocessing_config_path or "default:oasis_transforms.yaml"),
        "preprocessing_overrides": {
            "spatial_size": list(registry_image_size),
        }
        if registry_image_size is not None
        else {},
        "model_name": model_cfg.architecture,
        "target_layer": gradcam["target_layer"],
        "target_class": selected_class,
        "target_class_name": class_names[selected_class] if selected_class < len(class_names) else f"class_{selected_class}",
        "true_label": cfg.true_label,
        "prediction_correctness": prediction_correctness,
        "probabilities": {
            class_names[index] if index < len(class_names) else f"class_{index}": probability
            for index, probability in enumerate(probabilities)
        },
        "uncertainty": uncertainty,
        "confidence_level": calibrated.confidence_level,
        "review_flag": calibrated.review_flag,
        "region_importance_proxy": region_proxy,
        "heatmap_intensity_summary": heatmap_summary,
        "highlighted_regions": describe_highlighted_regions(region_proxy),
        "confidence_interpretation": interpret_confidence(
            uncertainty,
            confidence_level=calibrated.confidence_level,
        ),
        "explanation_quality": classify_explanation_quality(
            heatmap_summary,
            confidence_level=calibrated.confidence_level,
        ),
        "decision_support_only": True,
        "clinical_disclaimer": STANDARD_DECISION_SUPPORT_DISCLAIMER,
        "artifacts": {
            "gradcam_overlays": [str(path) for path in overlay_paths],
            "saliency_overlays": [str(path) for path in saliency_paths],
            "report_json": str(output_root / "explanation_report.json"),
        },
        "limitations": [
            EXPLANATION_LIMITATION,
            "Region importance values are coarse array-region proxies, not validated anatomical measurements.",
            "Explanation quality labels describe heatmap clarity only; they do not validate model correctness.",
            STANDARD_DECISION_SUPPORT_DISCLAIMER,
        ],
    }
    if registry_image_size is not None:
        payload["limitations"].append(
            f"Explainability preprocessing used the active registry spatial size override {list(registry_image_size)} to stay aligned with the promoted model input shape."
        )
    report_path = output_root / "explanation_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    audit_sensitive_action(
        action="explain_scan",
        metadata={
            "output_name": cfg.output_name,
            "scan_file_name": scan_path.name,
            "checkpoint_name": cfg.checkpoint_path.name,
            "target_layer": gradcam["target_layer"],
        },
    )
    return ExplanationResult(
        output_root=output_root,
        report_path=report_path,
        overlay_paths=overlay_paths,
        saliency_paths=saliency_paths,
        payload=payload,
    )
