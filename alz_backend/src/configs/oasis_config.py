"""Configuration objects for OASIS-1 training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .base import DatasetPaths

if TYPE_CHECKING:
    from .runtime import AppSettings


@dataclass(slots=True)
class OASISConfig:
    """Default configuration for the primary OASIS-1 backend pipeline."""

    dataset_name: str = "oasis"
    priority: str = "primary"
    label_column: str = "clinical_status"
    subject_id_column: str = "subject_id"
    session_id_column: str = "session_id"
    class_names: tuple[str, ...] = field(default_factory=lambda: ("nondemented", "demented"))

    def build_paths(self, settings: AppSettings) -> DatasetPaths:
        """Resolve OASIS-specific source and working directories from app settings."""

        return DatasetPaths(
            external_source_root=settings.oasis_source_root,
            raw_dir=settings.data_root / "raw" / self.dataset_name,
            interim_dir=settings.data_root / "interim" / self.dataset_name,
            processed_dir=settings.data_root / "processed" / self.dataset_name,
            metadata_dir=settings.data_root / "metadata" / self.dataset_name,
            checkpoint_dir=settings.outputs_root / "checkpoints" / self.dataset_name,
        )
