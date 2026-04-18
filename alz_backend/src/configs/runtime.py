"""Central runtime settings for project roots, dataset sources, and safety defaults."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from os import getenv
from pathlib import Path


def _resolve_path(env_var: str, default: Path) -> Path:
    """Resolve a path from an environment variable, falling back to a default."""

    raw_value = getenv(env_var)
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return default.resolve()


@dataclass(slots=True, frozen=True)
class AppSettings:
    """Central settings object for the local structural MRI backend workspace."""

    project_root: Path
    workspace_root: Path
    collection_root: Path
    data_root: Path
    outputs_root: Path
    kaggle_source_root: Path
    oasis_source_root: Path
    primary_dataset: str = "oasis"
    decision_support_only: bool = True
    storage_root: Path | None = None
    database_path: Path | None = None
    serving_config_path: Path | None = None

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Build settings from the repository location and optional environment overrides."""

        project_root = Path(__file__).resolve().parents[2]
        workspace_root = project_root.parent
        collection_root = workspace_root.parent
        return cls(
            project_root=project_root,
            workspace_root=workspace_root,
            collection_root=collection_root,
            data_root=_resolve_path("ALZ_DATA_ROOT", project_root / "data"),
            outputs_root=_resolve_path("ALZ_OUTPUTS_ROOT", project_root / "outputs"),
            kaggle_source_root=_resolve_path("ALZ_KAGGLE_SOURCE_DIR", workspace_root),
            oasis_source_root=_resolve_path("ALZ_OASIS_SOURCE_DIR", collection_root / "OASIS"),
            storage_root=_resolve_path("ALZ_STORAGE_ROOT", project_root / "storage"),
            database_path=_resolve_path("ALZ_DATABASE_PATH", project_root / "storage" / "alz_backend.sqlite3"),
            serving_config_path=_resolve_path("ALZ_SERVING_CONFIG_PATH", project_root / "configs" / "backend_serving.yaml"),
        )


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    """Return a cached app settings instance for the running process."""

    return AppSettings.from_env()
