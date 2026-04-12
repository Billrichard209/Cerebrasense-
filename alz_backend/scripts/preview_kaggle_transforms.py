"""Preview Kaggle MONAI preprocessing pipelines and save transformed sample figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from monai.utils import set_determinism
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kaggle_dataset import build_kaggle_monai_records, infer_kaggle_dataset_type, load_kaggle_manifest
from src.transforms.kaggle_transforms import (
    build_kaggle_infer_transforms,
    build_kaggle_train_transforms,
    build_kaggle_val_transforms,
    describe_kaggle_transform_pipeline,
    load_kaggle_transform_config,
)
from src.utils.io_utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Preview Kaggle MONAI preprocessing transforms.")
    parser.add_argument("--mode", choices=("train", "val", "infer"), default="val")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "kaggle_transforms.yaml")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "visualizations" / "kaggle_transforms",
    )
    return parser


def _load_2d_image(path: Path) -> np.ndarray:
    """Load a Kaggle 2D image for visualization."""

    return np.asarray(Image.open(path).convert("L"))


def _load_3d_volume(path: Path) -> np.ndarray:
    """Load a Kaggle 3D volume for visualization."""

    return np.asarray(nib.load(str(path)).get_fdata()).squeeze()


def _extract_three_planes(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract center slices from a 3D volume."""

    array = np.asarray(volume).squeeze()
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume for preview, got shape {array.shape}")
    return (
        array[array.shape[0] // 2, :, :],
        array[:, array.shape[1] // 2, :],
        array[:, :, array.shape[2] // 2],
    )


def _build_transform(mode: str, cfg: object, *, dataset_type: str) -> object:
    """Select the correct Kaggle transform builder."""

    if mode == "train":
        return build_kaggle_train_transforms(cfg, dataset_type=dataset_type)
    if mode == "val":
        return build_kaggle_val_transforms(cfg, dataset_type=dataset_type)
    return build_kaggle_infer_transforms(cfg, dataset_type=dataset_type)


def _plot_2d_preview(original: np.ndarray, transformed: np.ndarray, title: str, output_path: Path) -> None:
    """Save a side-by-side 2D Kaggle preview."""

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(np.asarray(transformed).squeeze(), cmap="gray")
    axes[1].set_title("Transformed")
    axes[1].axis("off")
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_3d_preview(original: np.ndarray, transformed: np.ndarray, title: str, output_path: Path) -> None:
    """Save a three-plane before/after 3D Kaggle preview."""

    original_planes = _extract_three_planes(original)
    transformed_planes = _extract_three_planes(np.asarray(transformed).squeeze())
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    plane_titles = ("Sagittal", "Coronal", "Axial")
    for axis_index, plane in enumerate(original_planes):
        axes[0, axis_index].imshow(np.rot90(plane), cmap="gray")
        axes[0, axis_index].set_title(f"Original {plane_titles[axis_index]}")
        axes[0, axis_index].axis("off")
    for axis_index, plane in enumerate(transformed_planes):
        axes[1, axis_index].imshow(np.rot90(plane), cmap="gray")
        axes[1, axis_index].set_title(f"Transformed {plane_titles[axis_index]}")
        axes[1, axis_index].axis("off")
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Preview the configured Kaggle preprocessing pipeline on real samples."""

    args = build_parser().parse_args()
    ensure_directory(args.output_dir)
    set_determinism(seed=args.seed)

    cfg = load_kaggle_transform_config(args.config)
    manifest_frame = load_kaggle_manifest(split=args.split)
    dataset_type = infer_kaggle_dataset_type(manifest_frame)
    transform = _build_transform(args.mode, cfg, dataset_type=dataset_type)
    records = build_kaggle_monai_records(split=args.split)
    selected_records = records[: args.max_samples]

    if not selected_records:
        raise ValueError("No Kaggle records were available for transform preview.")

    description_path = args.output_dir / f"kaggle_{dataset_type}_{args.mode}_transform_description.json"
    description_path.write_text(
        json.dumps(
            describe_kaggle_transform_pipeline(cfg, dataset_type=dataset_type, mode=args.mode),
            indent=2,
        ),
        encoding="utf-8",
    )

    saved_paths: list[Path] = []
    for index, record in enumerate(selected_records, start=1):
        image_path = Path(record["image"])
        transformed = transform({"image": str(image_path)})["image"]
        output_path = args.output_dir / f"kaggle_{dataset_type}_{args.mode}_preview_{index:02d}.png"
        title = f"Kaggle {args.mode} preview: {image_path.name}"
        if dataset_type == "2d_slices":
            _plot_2d_preview(_load_2d_image(image_path), np.asarray(transformed), title, output_path)
        else:
            _plot_3d_preview(_load_3d_volume(image_path), np.asarray(transformed), title, output_path)
        saved_paths.append(output_path)

    print(f"description={description_path}")
    for path in saved_paths:
        print(f"preview={path}")


if __name__ == "__main__":
    main()
