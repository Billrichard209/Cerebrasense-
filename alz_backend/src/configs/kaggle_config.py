"""Configuration objects for the separate Kaggle Alzheimer workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .base import DatasetPaths

if TYPE_CHECKING:
    from .runtime import AppSettings


@dataclass(slots=True)
class KaggleConfig:
    """Default configuration for Kaggle experiments kept isolated from OASIS."""

    dataset_name: str = "kaggle"
    priority: str = "secondary"
    label_column: str = "label"
    subject_id_column: str = "sample_id"
    session_id_column: str = "session_id"
    class_names: tuple[str, ...] = field(default_factory=tuple)

    def build_paths(self, settings: AppSettings) -> DatasetPaths:
        """Resolve Kaggle-specific source and working directories from app settings."""

        return DatasetPaths(
            external_source_root=settings.kaggle_source_root,
            raw_dir=settings.data_root / "raw" / self.dataset_name,
            interim_dir=settings.data_root / "interim" / self.dataset_name,
            processed_dir=settings.data_root / "processed" / self.dataset_name,
            metadata_dir=settings.data_root / "metadata" / self.dataset_name,
            checkpoint_dir=settings.outputs_root / "checkpoints" / self.dataset_name,
        )
