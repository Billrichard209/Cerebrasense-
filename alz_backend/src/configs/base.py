"""Shared path containers for dataset-specific configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class DatasetPaths:
    """Resolved source and working directories for a single dataset."""

    external_source_root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    metadata_dir: Path
    checkpoint_dir: Path
