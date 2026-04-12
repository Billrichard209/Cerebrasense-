"""Dataclasses used by the dataset inspection subsystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ImagingFileRecord:
    """Represents one loadable imaging asset discovered during inspection."""

    dataset_name: str
    load_path: Path
    fingerprint_path: Path
    format_name: str
    relative_path: str
    subset: str | None = None
    label: str | None = None
    subject_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary for logging and reports."""

        payload = asdict(self)
        payload["load_path"] = str(self.load_path)
        payload["fingerprint_path"] = str(self.fingerprint_path)
        return payload


@dataclass(slots=True)
class DatasetInspectionReport:
    """Top-level dataset inspection report serialized to JSON."""

    dataset_name: str
    source_root: str
    inspected_at: str
    total_file_count: int
    primary_image_count: int
    file_formats: dict[str, int]
    data_type: str
    is_3d_volume_dataset: bool
    monai_3d_suitability: str
    monai_3d_reason: str
    shape_distribution: list[dict[str, Any]]
    voxel_spacing_distribution: list[dict[str, Any]]
    intensity_statistics: list[dict[str, Any]]
    label_distribution: dict[str, int]
    label_source: str
    subject_id_summary: dict[str, Any]
    scan_timestamp_summary: dict[str, Any]
    missing_files: list[str]
    corrupt_files: list[dict[str, str]]
    duplicate_risk_summary: dict[str, Any]
    metadata_summary: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""

        return asdict(self)
