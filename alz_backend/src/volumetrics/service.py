"""Service helpers for structural and volumetric summary generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.data.base_dataset import parse_manifest_meta
from src.data.oasis_dataset import load_oasis_manifest
from src.utils.io_utils import ensure_directory

from .measurements import VolumetricAnalysisResult, VolumetricMeasurement, analyze_mri_volume, summarize_volumetrics


def build_volumetric_report(measurements: list[VolumetricMeasurement]) -> dict[str, object]:
    """Build a lightweight volumetric report payload for API or report generation."""

    return {
        "measurement_count": len(measurements),
        "regions": summarize_volumetrics(measurements),
    }


def _resolve_oasis_manifest_row(
    *,
    settings: AppSettings,
    split: str | None = None,
    row_index: int = 0,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve one manifest-backed OASIS sample by split and row index."""

    frame = load_oasis_manifest(settings, split=split, manifest_path=manifest_path)
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"Requested OASIS manifest row {row_index} is out of range for {len(frame)} rows.")
    row = frame.iloc[row_index]
    meta = parse_manifest_meta(row["meta"])
    return {
        "image": str(row["image"]),
        "subject_id": None if pd.isna(row["subject_id"]) else str(row["subject_id"]),
        "session_id": meta.get("session_id"),
        "scan_timestamp": None if pd.isna(row.get("scan_timestamp")) else str(row["scan_timestamp"]),
        "dataset": None if pd.isna(row.get("dataset")) else str(row["dataset"]),
        "dataset_type": None if pd.isna(row.get("dataset_type")) else str(row["dataset_type"]),
        "meta": meta,
    }


def analyze_oasis_volume(
    *,
    image_path: str | Path | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    scan_timestamp: str | None = None,
    split: str | None = None,
    row_index: int = 0,
    manifest_path: str | Path | None = None,
    settings: AppSettings | None = None,
) -> VolumetricAnalysisResult:
    """Analyze one OASIS MRI volume by path or manifest row."""

    resolved_settings = settings or get_app_settings()
    resolved_image_path = Path(image_path) if image_path is not None else None
    if resolved_image_path is None:
        resolved_row = _resolve_oasis_manifest_row(
            settings=resolved_settings,
            split=split,
            row_index=row_index,
            manifest_path=Path(manifest_path) if manifest_path is not None else None,
        )
        resolved_image_path = Path(resolved_row["image"])
        subject_id = subject_id or resolved_row["subject_id"]
        session_id = session_id or resolved_row["session_id"]
        scan_timestamp = scan_timestamp or resolved_row["scan_timestamp"]

    result = analyze_mri_volume(
        resolved_image_path,
        dataset="oasis1",
        dataset_type="3d_volumes",
        subject_id=subject_id,
        session_id=session_id,
        scan_timestamp=scan_timestamp,
    )
    if session_id is None:
        result.warnings.append("No session_id was supplied, so source-session linkage is limited for this report.")
    return result


def save_oasis_volumetric_report(
    report: dict[str, Any],
    *,
    settings: AppSettings | None = None,
    file_stem: str | None = None,
) -> Path:
    """Save an OASIS volumetric report under outputs/reports/volumetrics."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "volumetrics")
    resolved_stem = file_stem or "oasis_volumetric_report"
    output_path = output_root / f"{resolved_stem}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
