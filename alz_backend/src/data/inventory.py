"""Inventory helpers for describing the current OASIS and Kaggle source roots."""

from __future__ import annotations

from pathlib import Path

from src.configs.runtime import AppSettings

from .registry import build_dataset_registry

IGNORED_NAMES = {".git", ".venv", ".sixth", "alz_backend", "__pycache__"}


def _visible_children(root: Path) -> list[Path]:
    """Return top-level children excluding hidden and project-maintenance folders."""

    if not root.exists():
        return []
    return [child for child in root.iterdir() if child.name not in IGNORED_NAMES and not child.name.startswith(".")]


def _preview_children(root: Path, limit: int = 8) -> list[str]:
    """Return a small, stable preview of top-level child names for a dataset root."""

    return sorted(path.name for path in _visible_children(root))[:limit]


def _shallow_inventory(root: Path) -> dict[str, object]:
    """Build a lightweight non-recursive inventory for a dataset source root."""

    if not root.exists():
        return {
            "exists": False,
            "directory_count": 0,
            "file_count": 0,
            "preview": [],
        }

    children = _visible_children(root)
    directory_count = sum(1 for child in children if child.is_dir())
    file_count = sum(1 for child in children if child.is_file())
    return {
        "exists": True,
        "directory_count": directory_count,
        "file_count": file_count,
        "preview": _preview_children(root),
    }


def build_dataset_inventory_snapshot(settings: AppSettings | None = None) -> dict[str, object]:
    """Build a lightweight inventory snapshot for the registered dataset sources."""

    resolved_settings = settings or AppSettings.from_env()
    registry = build_dataset_registry(resolved_settings)
    return {
        "primary_dataset": resolved_settings.primary_dataset,
        "datasets": {
            name: {
                "source_root": str(spec.source_root),
                "inventory": _shallow_inventory(spec.source_root),
            }
            for name, spec in registry.items()
        },
    }
