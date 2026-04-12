"""Preview OASIS MONAI preprocessing pipelines and save transformed sample figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from monai.utils import set_determinism

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis_dataset import build_oasis_monai_records
from src.transforms.oasis_transforms import (
    build_oasis_infer_transforms,
    build_oasis_train_transforms,
    build_oasis_val_transforms,
    describe_oasis_transform_pipeline,
    load_oasis_transform_config,
)
from src.utils.io_utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Preview OASIS MONAI preprocessing transforms.")
    parser.add_argument("--mode", choices=("train", "val", "infer"), default="val")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "oasis_transforms.yaml")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "visualizations" / "oasis_transforms",
    )
    return parser


def _load_volume(path: Path) -> np.ndarray:
    """Load an OASIS image volume for visualization."""

    volume = nib.load(str(path)).get_fdata()
    return np.asarray(volume).squeeze()


def _extract_three_planes(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sagittal, coronal, and axial center slices."""

    array = np.asarray(volume).squeeze()
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume for preview, got shape {array.shape}")
    sagittal = array[array.shape[0] // 2, :, :]
    coronal = array[:, array.shape[1] // 2, :]
    axial = array[:, :, array.shape[2] // 2]
    return sagittal, coronal, axial


def _build_transform(mode: str, cfg: object) -> object:
    """Select the correct OASIS transform builder."""

    if mode == "train":
        return build_oasis_train_transforms(cfg)
    if mode == "val":
        return build_oasis_val_transforms(cfg)
    return build_oasis_infer_transforms(cfg)


def main() -> None:
    """Preview the configured OASIS preprocessing pipeline on real samples."""

    args = build_parser().parse_args()
    ensure_directory(args.output_dir)
    set_determinism(seed=args.seed)

    cfg = load_oasis_transform_config(args.config)
    transform = _build_transform(args.mode, cfg)
    records = build_oasis_monai_records(split=args.split)
    selected_records = records[: args.max_samples]

    if not selected_records:
        raise ValueError("No OASIS records were available for transform preview.")

    description_path = args.output_dir / f"oasis_{args.mode}_transform_description.json"
    description_path.write_text(
        json.dumps(describe_oasis_transform_pipeline(cfg, mode=args.mode), indent=2),
        encoding="utf-8",
    )

    saved_paths: list[Path] = []
    for index, record in enumerate(selected_records, start=1):
        image_path = Path(record["image"])
        original_volume = _load_volume(image_path)
        transformed_volume = np.asarray(transform({"image": str(image_path)})["image"]).squeeze()

        original_planes = _extract_three_planes(original_volume)
        transformed_planes = _extract_three_planes(transformed_volume)

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

        figure.suptitle(f"OASIS {args.mode} preview: {image_path.name}", fontsize=12)
        figure.tight_layout()
        output_path = args.output_dir / f"oasis_{args.mode}_preview_{index:02d}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    print(f"description={description_path}")
    for path in saved_paths:
        print(f"preview={path}")


if __name__ == "__main__":
    main()
