"""Parsers for externally generated FreeSurfer structural statistics.

The parser reads files that already exist on disk, usually from a FreeSurfer
subject directory. It does not run FreeSurfer and does not fabricate missing
measurements.
"""

from __future__ import annotations

import re
from pathlib import Path

from .schemas import BrainRegionVolume, CorticalThicknessSummary, GlobalStructuralMeasure

HEMISPHERE_PREFIXES = {
    "Left-": "left",
    "Right-": "right",
}


class FreeSurferParseError(ValueError):
    """Raised when a FreeSurfer stats file cannot be parsed safely."""


def infer_hemisphere_from_region(region_name: str) -> str | None:
    """Infer left/right hemisphere from a FreeSurfer region name."""

    for prefix, hemisphere in HEMISPHERE_PREFIXES.items():
        if region_name.startswith(prefix):
            return hemisphere
    return None


def _read_existing_file(path: str | Path) -> tuple[Path, list[str]]:
    """Read an existing text file."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"FreeSurfer stats file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise FreeSurferParseError(f"FreeSurfer stats path must be a file: {resolved_path}")
    return resolved_path, resolved_path.read_text(encoding="utf-8", errors="replace").splitlines()


def _parse_colheaders(line: str) -> list[str]:
    """Parse a FreeSurfer ``# ColHeaders`` line."""

    return line.replace("#", "", 1).strip().split()[1:]


def _parse_float(raw_value: str, *, file_path: Path, line_number: int, column_name: str) -> float:
    """Parse a float with a helpful error."""

    try:
        return float(raw_value)
    except ValueError as exc:
        raise FreeSurferParseError(
            f"Could not parse {column_name}={raw_value!r} in {file_path} line {line_number}."
        ) from exc


def _parse_int(raw_value: str, *, file_path: Path, line_number: int, column_name: str) -> int:
    """Parse an integer with a helpful error."""

    try:
        return int(float(raw_value))
    except ValueError as exc:
        raise FreeSurferParseError(
            f"Could not parse {column_name}={raw_value!r} in {file_path} line {line_number}."
        ) from exc


def parse_aseg_stats(path: str | Path) -> list[BrainRegionVolume]:
    """Parse FreeSurfer ``aseg.stats`` region-volume rows.

    Supported rows use the common ``# ColHeaders Index SegId NVoxels
    Volume_mm3 StructName ...`` format. If the header is missing, the parser
    falls back to the standard FreeSurfer column order.
    """

    file_path, lines = _read_existing_file(path)
    headers: list[str] | None = None
    volumes: list[BrainRegionVolume] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# ColHeaders"):
            headers = _parse_colheaders(line)
            continue
        if line.startswith("#"):
            continue

        parts = line.split()
        row = dict(zip(headers, parts)) if headers is not None else {}
        region_name = row.get("StructName") if row else (parts[4] if len(parts) >= 5 else None)
        volume_raw = row.get("Volume_mm3") if row else (parts[3] if len(parts) >= 4 else None)
        seg_id_raw = row.get("SegId") if row else (parts[1] if len(parts) >= 2 else None)
        if region_name is None or volume_raw is None:
            continue

        volumes.append(
            BrainRegionVolume(
                region_name=region_name,
                value_mm3=_parse_float(volume_raw, file_path=file_path, line_number=line_number, column_name="Volume_mm3"),
                segmentation_label=_parse_int(seg_id_raw, file_path=file_path, line_number=line_number, column_name="SegId")
                if seg_id_raw is not None
                else None,
                hemisphere=infer_hemisphere_from_region(region_name),
                source_file=file_path,
            )
        )
    return volumes


def _feature_name_from_measure_id(measure_id: str) -> str:
    """Convert a FreeSurfer measure id into a stable feature key."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", measure_id.strip()).strip("_")
    return normalized.lower()


def parse_global_measures(path: str | Path) -> list[GlobalStructuralMeasure]:
    """Parse FreeSurfer ``# Measure ...`` summary rows when present."""

    file_path, lines = _read_existing_file(path)
    measures: list[GlobalStructuralMeasure] = []
    prefix = "# Measure "
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue
        parts = [part.strip() for part in line[len(prefix) :].split(",", maxsplit=4)]
        if len(parts) < 4:
            continue
        measure_id, feature_id, display_name, value_raw = parts[:4]
        unit = parts[4] if len(parts) == 5 else None
        measures.append(
            GlobalStructuralMeasure(
                measure_id=measure_id,
                feature_name=_feature_name_from_measure_id(feature_id or measure_id),
                display_name=display_name or measure_id,
                value=float(value_raw),
                unit=unit,
                source_file=file_path,
            )
        )
    return measures


def _hemisphere_from_aparc_path(path: Path) -> str | None:
    """Infer hemisphere from an aparc stats file name."""

    name = path.name.lower()
    if name.startswith("lh."):
        return "left"
    if name.startswith("rh."):
        return "right"
    return None


def parse_aparc_stats(path: str | Path, *, hemisphere: str | None = None) -> list[CorticalThicknessSummary]:
    """Parse FreeSurfer ``lh.aparc.stats`` or ``rh.aparc.stats`` cortical rows."""

    file_path, lines = _read_existing_file(path)
    resolved_hemisphere = hemisphere or _hemisphere_from_aparc_path(file_path) or "unknown"
    headers: list[str] | None = None
    summaries: list[CorticalThicknessSummary] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# ColHeaders"):
            headers = _parse_colheaders(line)
            continue
        if line.startswith("#"):
            continue

        parts = line.split()
        row = dict(zip(headers, parts)) if headers is not None else {}
        region_name = row.get("StructName") if row else (parts[0] if parts else None)
        thickness_raw = row.get("ThickAvg") if row else (parts[4] if len(parts) >= 5 else None)
        if region_name is None or thickness_raw is None:
            continue

        summaries.append(
            CorticalThicknessSummary(
                hemisphere=resolved_hemisphere,
                region_name=region_name,
                mean_thickness_mm=_parse_float(
                    thickness_raw,
                    file_path=file_path,
                    line_number=line_number,
                    column_name="ThickAvg",
                ),
                thickness_std_mm=_parse_float(
                    row["ThickStd"],
                    file_path=file_path,
                    line_number=line_number,
                    column_name="ThickStd",
                )
                if row.get("ThickStd") is not None
                else None,
                surface_area_mm2=_parse_float(
                    row["SurfArea"],
                    file_path=file_path,
                    line_number=line_number,
                    column_name="SurfArea",
                )
                if row.get("SurfArea") is not None
                else None,
                gray_matter_volume_mm3=_parse_float(
                    row["GrayVol"],
                    file_path=file_path,
                    line_number=line_number,
                    column_name="GrayVol",
                )
                if row.get("GrayVol") is not None
                else None,
                source_file=file_path,
            )
        )
    return summaries


def parse_freesurfer_version(lines: list[str]) -> str | None:
    """Extract a FreeSurfer version string when present."""

    version_patterns = (
        re.compile(r"freesurfer\s+version\s*[:=]\s*(.+)$", re.IGNORECASE),
        re.compile(r"generated by freesurfer\s+(.+)$", re.IGNORECASE),
    )
    for line in lines:
        for pattern in version_patterns:
            match = pattern.search(line)
            if match:
                return match.group(1).strip()
    return None


def read_freesurfer_version_from_file(path: str | Path) -> str | None:
    """Read a FreeSurfer version hint from a stats file if available."""

    _, lines = _read_existing_file(path)
    return parse_freesurfer_version(lines)
