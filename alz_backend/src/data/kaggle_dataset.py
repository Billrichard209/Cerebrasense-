"""Dataset definitions and MONAI data loaders for the separate Kaggle Alzheimer dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.kaggle_config import KaggleConfig
from src.configs.runtime import AppSettings
from src.transforms.kaggle_transforms import (
    KaggleTransformConfig,
    build_kaggle_infer_transforms,
    build_kaggle_train_transforms,
    build_kaggle_val_transforms,
    load_kaggle_transform_config,
)

from .base_dataset import (
    DatasetSample,
    build_monai_dataloader,
    build_monai_dataset,
    canonicalize_optional_string,
    load_manifest_frame,
    parse_manifest_meta,
)

KAGGLE_MANIFEST_COLUMNS = {
    "image",
    "label",
    "label_name",
    "subject_id",
    "scan_timestamp",
    "dataset",
    "dataset_type",
    "meta",
}


@dataclass(slots=True)
class KaggleDatasetSpec:
    """Specification for Kaggle dataset handling without implicit harmonization."""

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


def build_kaggle_dataset_spec(settings: AppSettings | None = None) -> KaggleDatasetSpec:
    """Build the separate Kaggle dataset specification from central app settings."""

    resolved_settings = settings or AppSettings.from_env()
    config = KaggleConfig()
    paths = config.build_paths(resolved_settings)
    return KaggleDatasetSpec(
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


def resolve_kaggle_manifest_path(
    settings: AppSettings,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
) -> Path:
    """Resolve the Kaggle manifest path, optionally for a specific split."""

    if manifest_path is not None:
        return manifest_path
    if split is None:
        return settings.data_root / "interim" / "kaggle_alz_manifest.csv"
    return settings.data_root / "interim" / f"kaggle_alz_{split}_manifest.csv"


def load_kaggle_manifest(
    settings: AppSettings | None = None,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Load a Kaggle manifest and align it to MONAI dataset conventions."""

    resolved_settings = settings or AppSettings.from_env()
    resolved_path = resolve_kaggle_manifest_path(
        resolved_settings,
        split=split,
        manifest_path=manifest_path,
    )
    return load_manifest_frame(resolved_path, required_columns=KAGGLE_MANIFEST_COLUMNS)


def infer_kaggle_dataset_type(manifest_frame: pd.DataFrame) -> str:
    """Infer the unique Kaggle dataset type represented by a manifest frame."""

    dataset_types = sorted(
        {
            dataset_type
            for dataset_type in manifest_frame["dataset_type"].map(canonicalize_optional_string).tolist()
            if dataset_type is not None
        }
    )
    if not dataset_types:
        raise ValueError("Kaggle manifest does not define any dataset_type values.")
    if len(dataset_types) > 1:
        raise ValueError(f"Kaggle manifest mixes dataset types: {dataset_types}")
    return dataset_types[0]


def infer_kaggle_class_names(
    manifest_frame: pd.DataFrame,
    *,
    label_map: dict[str, int] | None = None,
) -> tuple[str, ...]:
    """Infer the visible Kaggle class names for MONAI model configuration."""

    if label_map:
        return tuple(sorted(label_map))
    class_names = sorted(
        {
            label_name
            for label_name in manifest_frame["label_name"].map(canonicalize_optional_string).tolist()
            if label_name is not None
        }
    )
    return tuple(class_names)


def build_kaggle_monai_records(
    settings: AppSettings | None = None,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
    label_map: dict[str, int] | None = None,
    require_labels: bool = False,
) -> list[dict[str, Any]]:
    """Convert Kaggle manifest rows into MONAI dictionary records."""

    frame = load_kaggle_manifest(settings, split=split, manifest_path=manifest_path)
    records: list[dict[str, Any]] = []

    for row in frame.itertuples(index=False):
        image_path = Path(row.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Kaggle manifest image path does not exist: {image_path}")

        numeric_label: int | None = None
        mapping_applied = False
        if not pd.isna(row.label):
            numeric_label = int(float(row.label))
        else:
            label_name = canonicalize_optional_string(getattr(row, "label_name", None))
            if label_name is not None and label_map is not None and label_name in label_map:
                numeric_label = int(label_map[label_name])
                mapping_applied = True

        if require_labels and numeric_label is None:
            raise ValueError(
                "Kaggle MONAI training requires numeric labels. "
                "Provide a manifest with explicit remapping or pass `label_map` explicitly."
            )

        meta = parse_manifest_meta(getattr(row, "meta"))
        if mapping_applied:
            meta["explicit_runtime_label_map_applied"] = True

        sample = DatasetSample(
            sample_id=canonicalize_optional_string(getattr(row, "subject_id", None)) or image_path.stem,
            image_path=image_path,
            label=numeric_label,
            label_name=canonicalize_optional_string(getattr(row, "label_name", None)),
            subject_id=canonicalize_optional_string(getattr(row, "subject_id", None)),
            dataset=canonicalize_optional_string(getattr(row, "dataset", None)),
            dataset_type=canonicalize_optional_string(getattr(row, "dataset_type", None)),
            scan_timestamp=canonicalize_optional_string(getattr(row, "scan_timestamp", None)),
            meta=meta,
        )
        record = sample.to_monai_record()
        for key in ("label_name", "subject_id", "session_id", "scan_timestamp", "dataset", "dataset_type"):
            if record.get(key) is None:
                record[key] = ""
        record["image_path"] = str(image_path)
        records.append(record)

    return records


def build_kaggle_monai_dataset(
    settings: AppSettings | None = None,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
    training: bool = False,
    cache_rate: float = 0.0,
    num_workers: int = 0,
    transform: object | None = None,
    transform_config: KaggleTransformConfig | None = None,
    label_map: dict[str, int] | None = None,
    require_labels: bool | None = None,
) -> object:
    """Build a MONAI Dataset or CacheDataset for Kaggle data."""

    frame = load_kaggle_manifest(settings, split=split, manifest_path=manifest_path)
    dataset_type = infer_kaggle_dataset_type(frame)
    resolved_require_labels = training if require_labels is None else require_labels
    records = build_kaggle_monai_records(
        settings,
        split=split,
        manifest_path=manifest_path,
        label_map=label_map,
        require_labels=resolved_require_labels,
    )
    resolved_cfg = transform_config or load_kaggle_transform_config()
    if transform is not None:
        resolved_transform = transform
    elif training:
        resolved_transform = build_kaggle_train_transforms(resolved_cfg, dataset_type=dataset_type)
    elif split == "test":
        resolved_transform = build_kaggle_infer_transforms(resolved_cfg, dataset_type=dataset_type)
    else:
        resolved_transform = build_kaggle_val_transforms(resolved_cfg, dataset_type=dataset_type)
    return build_monai_dataset(
        records,
        resolved_transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )


def build_kaggle_monai_dataloader(
    settings: AppSettings | None = None,
    *,
    split: str | None = None,
    manifest_path: Path | None = None,
    training: bool = False,
    batch_size: int = 1,
    cache_rate: float = 0.0,
    num_workers: int = 0,
    transform: object | None = None,
    transform_config: KaggleTransformConfig | None = None,
    label_map: dict[str, int] | None = None,
    require_labels: bool | None = None,
) -> object:
    """Build a MONAI dataloader for Kaggle manifest-driven training or inference."""

    dataset = build_kaggle_monai_dataset(
        settings,
        split=split,
        manifest_path=manifest_path,
        training=training,
        cache_rate=cache_rate,
        num_workers=num_workers,
        transform=transform,
        transform_config=transform_config,
        label_map=label_map,
        require_labels=require_labels,
    )
    return build_monai_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
    )
