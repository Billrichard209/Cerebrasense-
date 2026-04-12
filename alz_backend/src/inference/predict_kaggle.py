"""MONAI-oriented prediction helpers for the separate Kaggle inference path."""

from __future__ import annotations

from pathlib import Path

from src.transforms.kaggle_transforms import KaggleTransformConfig, build_kaggle_infer_transforms, load_kaggle_transform_config

from .predict_oasis import PredictionResult, _load_monai_inferer_symbols, _load_torch_symbols, _tensor_to_probability_list


def predict_kaggle_image(
    image_path: str | Path,
    *,
    model: object,
    dataset_type: str,
    class_names: tuple[str, ...],
    device: str = "cpu",
    transforms: object | None = None,
    transform_config: KaggleTransformConfig | None = None,
) -> PredictionResult:
    """Run MONAI-style inference for one Kaggle image or volume."""

    resolved_path = Path(image_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Kaggle inference image path does not exist: {resolved_path}")

    resolved_transforms = transforms or build_kaggle_infer_transforms(
        transform_config or load_kaggle_transform_config(),
        dataset_type=dataset_type,
    )

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

    predicted_index = int(torch.argmax(probabilities, dim=0).item())
    probability_list = _tensor_to_probability_list(probabilities)
    predicted_label = class_names[predicted_index] if predicted_index < len(class_names) else f"class_{predicted_index}"
    return PredictionResult(
        predicted_index=predicted_index,
        label=predicted_label,
        confidence=float(max(probability_list)),
        probabilities=probability_list,
        source_dataset="kaggle_alz",
        dataset_type=dataset_type,
        meta={"image": str(resolved_path)},
    )
