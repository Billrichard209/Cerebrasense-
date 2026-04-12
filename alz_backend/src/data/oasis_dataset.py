"""Dataset definitions and MONAI data loaders for the primary OASIS-1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.oasis_config import OASISConfig
from src.configs.runtime import AppSettings
from src.transforms.oasis_transforms import (
    OASISTransformConfig,
    build_oasis_infer_transforms,
    build_oasis_train_transforms,
    build_oasis_val_transforms,
)

from .base_dataset import (
    DatasetSample,
    build_monai_dataloader,
    build_monai_dataset,
    canonicalize_optional_string,
    load_manifest_frame,
    parse_manifest_meta,
)

OASIS_MANIFEST_COLUMNS = {
    "image",
    "label",
    "label_name",
    "subject_id",
    "scan_timestamp",
    "dataset",
    "meta",
}
OASIS_DEFAULT_DATASET_TYPE = "3d_volumes"


@dataclass(slots=True)
class OASISDatasetSpec:
    """Specification for the primary OASIS-1 dataset pipeline."""

    name: str
    priority: str
    source_root: Path
    raw_root: Path
    interim_root: Path
    processed_root: Path
    metadata_root: Path
    checkpoint_root: Path
    label_column: str
    subject_id_column: str
    visit_id_column: str
    class_names: tuple[str, ...]

    def to_snapshot(self) -> dict[str, object]:
        """Return a JSON-safe dictionary for API responses and reports."""

        return {
            "name": self.name,
            "priority": self.priority,
            "source_root": str(self.source_root),
            "source_exists": self.source_root.exists(),
            "raw_root": str(self.raw_root),
            "interim_root": str(self.interim_root),
            "processed_root": str(self.processed_root),
            "metadata_root": str(self.metadata_root),
            "checkpoint_root": str(self.checkpoint_root),
            "label_column": self.label_column,
            "subject_id_column": self.subject_id_column,
            "visit_id_column": self.visit_id_column,
            "class_names": list(self.class_names),
        }


def build_oasis_dataset_spec(settings: AppSettings | None = None) -> OASISDatasetSpec:
    """Build the primary OASIS dataset specification from central app settings."""

    resolved_settings = settings or AppSettings.from_env()
    config = OASISConfig()
    paths = config.build_paths(resolved_settings)
    return OASISDatasetSpec(
        name=config.dataset_name,
        priority=config.priority,
        source_root=paths.external_source_root,
        raw_root=paths.raw_dir,
        interim_root=paths.interim_dir,
        processed_root=paths.processed_dir,
        metadata_root=paths.metadata_dir,
        checkpoint_root=paths.checkpoint_dir,
        label_column=config.label_column,
        subject_id_column=config.subject_id_column,
        visit_id_column=config.session_id_column,
        class_names=config.class_names,
    )


def resolve_oasis_manifest_path(
    settings: AppSettings,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
) -> Path:
    """Resolve the OASIS manifest path, optionally for a specific split."""

    if manifest_path is not None:
        return manifest_path
    if split is None:
        return settings.data_root / "interim" / "oasis1_manifest.csv"
    return settings.data_root / "interim" / f"oasis1_{split}_manifest.csv"


def load_oasis_manifest(
    settings: AppSettings | None = None,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Load an OASIS manifest and align it to MONAI dataset conventions."""

    resolved_settings = settings or AppSettings.from_env()
    resolved_path = resolve_oasis_manifest_path(
        resolved_settings,
        split=split,
        manifest_path=manifest_path,
    )
    return load_manifest_frame(
        resolved_path,
        required_columns=OASIS_MANIFEST_COLUMNS,
        default_dataset_type=OASIS_DEFAULT_DATASET_TYPE,
    )


def build_oasis_monai_records(
    settings: AppSettings | None = None,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
    require_labels: bool = True,
) -> list[dict[str, Any]]:
    """Convert OASIS manifest rows into MONAI dictionary records."""

    frame = load_oasis_manifest(settings, split=split, manifest_path=manifest_path)
    records: list[dict[str, Any]] = []

    for row in frame.itertuples(index=False):
        image_path = Path(row.image)
        if not image_path.exists():
            raise FileNotFoundError(f"OASIS manifest image path does not exist: {image_path}")

        label_value: int | None = None
        if not pd.isna(row.label):
            label_value = int(float(row.label))
        elif require_labels:
            raise ValueError(f"OASIS record is missing a numeric label: {image_path}")

        sample = DatasetSample(
            sample_id=canonicalize_optional_string(getattr(row, "subject_id", None)) or image_path.stem,
            image_path=image_path,
            label=label_value,
            label_name=canonicalize_optional_string(getattr(row, "label_name", None)),
            subject_id=canonicalize_optional_string(getattr(row, "subject_id", None)),
            session_id=parse_manifest_meta(getattr(row, "meta")).get("session_id"),
            dataset=canonicalize_optional_string(getattr(row, "dataset", None)),
            dataset_type=canonicalize_optional_string(getattr(row, "dataset_type", None)) or OASIS_DEFAULT_DATASET_TYPE,
            scan_timestamp=canonicalize_optional_string(getattr(row, "scan_timestamp", None)),
            meta=parse_manifest_meta(getattr(row, "meta")),
        )
        records.append(sample.to_monai_record())

    return records


def build_oasis_monai_dataset(
    settings: AppSettings | None = None,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
    training: bool = False,
    cache_rate: float = 0.0,
    num_workers: int = 0,
    transform: object | None = None,
    transform_config: OASISTransformConfig | None = None,
) -> object:
    """Build a MONAI Dataset or CacheDataset for OASIS."""

    records = build_oasis_monai_records(
        settings,
        split=split,
        manifest_path=manifest_path,
        require_labels=True,
    )
    resolved_cfg = transform_config or OASISTransformConfig()
    if transform is not None:
        resolved_transform = transform
    elif training:
        resolved_transform = build_oasis_train_transforms(resolved_cfg)
    elif split == "test":
        resolved_transform = build_oasis_infer_transforms(resolved_cfg)
    else:
        resolved_transform = build_oasis_val_transforms(resolved_cfg)
    return build_monai_dataset(
        records,
        resolved_transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )


def build_oasis_monai_dataloader(
    settings: AppSettings | None = None,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
    training: bool = False,
    batch_size: int = 1,
    cache_rate: float = 0.0,
    num_workers: int = 0,
    transform: object | None = None,
    transform_config: OASISTransformConfig | None = None,
) -> object:
    """Build a MONAI dataloader for OASIS manifest-driven training or inference."""

    dataset = build_oasis_monai_dataset(
        settings,
        split=split,
        manifest_path=manifest_path,
        training=training,
        cache_rate=cache_rate,
        num_workers=num_workers,
        transform=transform,
        transform_config=transform_config,
    )
    return build_monai_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
    )
