"""Shared manifest and MONAI dataset primitives used by separate dataset adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.utils.monai_utils import load_monai_data_symbols

_load_monai_data_symbols = load_monai_data_symbols


@dataclass(slots=True)
class DatasetSample:
    """Minimal dataset sample contract for MRI or metadata-driven pipelines."""

    sample_id: str
    image_path: Path
    label: int | None = None
    label_name: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    dataset: str | None = None
    dataset_type: str | None = None
    scan_timestamp: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_monai_record(self) -> dict[str, Any]:
        """Convert the sample to MONAI dictionary-dataset format."""

        return {
            "image": str(self.image_path),
            "label": self.label,
            "label_name": self.label_name,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "scan_timestamp": self.scan_timestamp,
            "dataset": self.dataset,
            "dataset_type": self.dataset_type,
            "meta": dict(self.meta),
        }


def canonicalize_optional_string(raw_value: Any) -> str | None:
    """Normalize optional manifest text fields."""

    if raw_value is None or pd.isna(raw_value):
        return None
    text = str(raw_value).strip()
    return text or None


def parse_manifest_meta(raw_value: Any) -> dict[str, Any]:
    """Parse a manifest metadata column that may already be a dictionary or JSON string."""

    if isinstance(raw_value, dict):
        return dict(raw_value)
    text = canonicalize_optional_string(raw_value)
    if text is None:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not parse manifest metadata JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Manifest `meta` payloads must decode to dictionaries.")
    return payload


def load_manifest_frame(
    manifest_path: Path,
    *,
    required_columns: Iterable[str],
    default_dataset_type: str | None = None,
) -> pd.DataFrame:
    """Load a manifest CSV and validate its required schema."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    required = set(required_columns)
    if default_dataset_type is not None and "dataset_type" not in frame.columns:
        frame = frame.assign(dataset_type=default_dataset_type)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Manifest {manifest_path} is empty.")
    return frame


def build_monai_dataset(
    records: list[dict[str, Any]],
    transform: object,
    *,
    cache_rate: float = 0.0,
    num_workers: int = 0,
) -> object:
    """Build a MONAI Dataset or CacheDataset from manifest-derived records."""

    if not records:
        raise ValueError("Cannot build a MONAI dataset from zero records.")

    symbols = _load_monai_data_symbols()
    if cache_rate > 0:
        cache_dataset_cls = symbols["CacheDataset"]
        return cache_dataset_cls(
            data=records,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    dataset_cls = symbols["Dataset"]
    return dataset_cls(data=records, transform=transform)


def build_monai_dataloader(
    dataset: object,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> object:
    """Build a MONAI-compatible dataloader from a dataset object."""

    data_loader_cls = _load_monai_data_symbols()["DataLoader"]
    return data_loader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
