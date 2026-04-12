"""Raw OASIS-2 source discovery and inventory helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings
from src.utils.io_utils import ensure_directory

OASIS2_SOURCE_ENV_VAR = "ALZ_OASIS2_SOURCE_DIR"
OASIS2_PART_PATTERN = re.compile(r"OAS2_RAW_PART\d+", re.IGNORECASE)
OASIS2_SUBJECT_PATTERN = re.compile(r"(OAS2[_-]\d{4})", re.IGNORECASE)
OASIS2_SESSION_PATTERN = re.compile(r"(OAS2[_-]\d{4}_MR\d+)", re.IGNORECASE)
OASIS2_ACQUISITION_PATTERN = re.compile(r"(mpr-\d+)", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class OASIS2SourceLayout:
    """Resolved OASIS-2 raw source layout."""

    source_root: Path
    inspection_roots: tuple[Path, ...]
    candidate_roots: tuple[Path, ...]
    source_resolution: str


@dataclass(slots=True)
class OASIS2RawInventoryResult:
    """Artifacts produced by the OASIS-2 raw inventory builder."""

    inventory_path: Path
    dropped_rows_path: Path
    summary_path: Path
    session_row_count: int
    unique_subject_count: int
    unique_session_count: int
    part_roots: tuple[Path, ...]


@dataclass(slots=True)
class OASIS2SessionManifestResult:
    """Artifacts produced by the unlabeled OASIS-2 session manifest builder."""

    manifest_path: Path
    longitudinal_records_path: Path
    subject_summary_path: Path
    summary_path: Path
    session_row_count: int
    unique_subject_count: int
    source_part_roots: tuple[Path, ...]


def _normalize_suffix(path: Path) -> str:
    """Return a normalized suffix, including compound suffixes like `.nii.gz`."""

    suffixes = [part.lower() for part in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] == [".nii", ".gz"]:
        return ".nii.gz"
    return path.suffix.lower()


def _normalize_identifier(value: str) -> str:
    """Normalize OASIS identifiers to stable upper-case underscore forms."""

    return value.upper().replace("-", "_")


def _detect_part_roots(parent: Path) -> tuple[Path, ...]:
    """Return immediate child directories that look like OAS2 raw parts."""

    if not parent.exists() or not parent.is_dir():
        return ()
    return tuple(
        sorted(
            (
                child.resolve()
                for child in parent.iterdir()
                if child.is_dir() and OASIS2_PART_PATTERN.fullmatch(child.name)
            ),
            key=lambda path: path.name,
        )
    )


def resolve_oasis2_source_layout(
    settings: AppSettings | None = None,
    *,
    source_root: Path | None = None,
) -> OASIS2SourceLayout:
    """Resolve the most likely OASIS-2 source layout, including split raw parts."""

    resolved_settings = settings or AppSettings.from_env()
    if source_root is not None:
        explicit_root = source_root.expanduser().resolve()
        if explicit_root.name and OASIS2_PART_PATTERN.fullmatch(explicit_root.name):
            part_roots = _detect_part_roots(explicit_root.parent)
            inspection_roots = part_roots or (explicit_root,)
            return OASIS2SourceLayout(
                source_root=explicit_root.parent if part_roots else explicit_root,
                inspection_roots=inspection_roots,
                candidate_roots=(),
                source_resolution="argument_part_root" if part_roots else "argument",
            )
        explicit_parts = _detect_part_roots(explicit_root)
        return OASIS2SourceLayout(
            source_root=explicit_root,
            inspection_roots=explicit_parts or ((explicit_root,) if explicit_root.exists() else ()),
            candidate_roots=(),
            source_resolution="argument_part_roots" if explicit_parts else "argument",
        )

    env_value = getenv(OASIS2_SOURCE_ENV_VAR)
    if env_value:
        env_root = Path(env_value).expanduser().resolve()
        if env_root.name and OASIS2_PART_PATTERN.fullmatch(env_root.name):
            part_roots = _detect_part_roots(env_root.parent)
            inspection_roots = part_roots or (env_root,)
            return OASIS2SourceLayout(
                source_root=env_root.parent if part_roots else env_root,
                inspection_roots=inspection_roots,
                candidate_roots=(),
                source_resolution="env_part_root" if part_roots else "env",
            )
        env_parts = _detect_part_roots(env_root)
        return OASIS2SourceLayout(
            source_root=env_root,
            inspection_roots=env_parts or ((env_root,) if env_root.exists() else ()),
            candidate_roots=(),
            source_resolution="env_part_roots" if env_parts else "env",
        )

    candidate_roots = (
        (resolved_settings.collection_root / "OASIS2").resolve(),
        (resolved_settings.collection_root / "OASIS-2").resolve(),
        (resolved_settings.workspace_root / "OASIS2").resolve(),
        (resolved_settings.workspace_root / "OASIS-2").resolve(),
    )
    for candidate in candidate_roots:
        if candidate.exists():
            explicit_parts = _detect_part_roots(candidate)
            return OASIS2SourceLayout(
                source_root=candidate,
                inspection_roots=explicit_parts or (candidate,),
                candidate_roots=candidate_roots,
                source_resolution="auto_existing_candidate_parts" if explicit_parts else "auto_existing_candidate",
            )

    part_parent_candidates = (
        resolved_settings.collection_root.resolve(),
        resolved_settings.workspace_root.resolve(),
    )
    for parent in part_parent_candidates:
        part_roots = _detect_part_roots(parent)
        if part_roots:
            return OASIS2SourceLayout(
                source_root=parent,
                inspection_roots=part_roots,
                candidate_roots=candidate_roots,
                source_resolution="auto_detected_part_roots",
            )

    return OASIS2SourceLayout(
        source_root=candidate_roots[0],
        inspection_roots=(),
        candidate_roots=candidate_roots,
        source_resolution="auto_default_candidate",
    )


def _session_directories(part_roots: tuple[Path, ...]) -> list[tuple[Path, Path]]:
    """Return `(part_root, session_dir)` pairs for OASIS-2-like session folders."""

    pairs: list[tuple[Path, Path]] = []
    for part_root in part_roots:
        if not part_root.exists():
            continue
        for child in sorted(part_root.iterdir(), key=lambda path: path.name):
            if child.is_dir() and OASIS2_SESSION_PATTERN.fullmatch(child.name):
                pairs.append((part_root, child))
    return pairs


def _build_volume_rows(
    part_roots: tuple[Path, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build raw OASIS-2 volume rows and dropped-row audit entries."""

    rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []

    for part_root, session_dir in _session_directories(part_roots):
        session_id = _normalize_identifier(session_dir.name)
        subject_id = session_id.split("_MR", 1)[0]
        visit_number = int(session_id.split("_MR", 1)[1])
        raw_dir = session_dir / "RAW"
        if not raw_dir.exists():
            dropped_rows.append(
                {
                    "part_root": part_root.name,
                    "session_id": session_id,
                    "reason": "missing_raw_directory",
                }
            )
            continue

        volume_paths = sorted(path for path in raw_dir.iterdir() if path.is_file())
        if not volume_paths:
            dropped_rows.append(
                {
                    "part_root": part_root.name,
                    "session_id": session_id,
                    "reason": "no_raw_files",
                }
            )
            continue

        for path in volume_paths:
            suffix = _normalize_suffix(path)
            if suffix == ".img":
                continue
            acquisition_match = OASIS2_ACQUISITION_PATTERN.search(path.name)
            acquisition_id = acquisition_match.group(1).lower() if acquisition_match else path.stem.lower()
            pair_path = None
            volume_format = suffix
            if suffix == ".hdr":
                paired_img = path.with_suffix(".img")
                if not paired_img.exists():
                    dropped_rows.append(
                        {
                            "part_root": part_root.name,
                            "session_id": session_id,
                            "image": str(path),
                            "reason": "missing_img_pair",
                        }
                    )
                    continue
                pair_path = paired_img
                volume_format = "analyze_pair"

            rows.append(
                {
                    "image": str(path),
                    "paired_image": str(pair_path) if pair_path else None,
                    "label": None,
                    "label_name": None,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "visit_number": visit_number,
                    "scan_timestamp": None,
                    "dataset": "oasis2_raw",
                    "dataset_type": "3d_volumes",
                    "source_part": part_root.name,
                    "acquisition_id": acquisition_id,
                    "volume_format": volume_format,
                    "meta": json.dumps(
                        {
                            "raw_dir": str(raw_dir),
                            "source_part": part_root.name,
                            "source_root": str(part_root.parent),
                            "session_dir": str(session_dir),
                            "volume_suffix": suffix,
                            "has_pair": pair_path is not None,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                }
            )

    return rows, dropped_rows


def build_oasis2_raw_inventory(
    settings: AppSettings | None = None,
    *,
    source_root: Path | None = None,
    output_root: Path | None = None,
) -> OASIS2RawInventoryResult:
    """Build a raw OASIS-2 inventory from split-part session folders."""

    resolved_settings = settings or AppSettings.from_env()
    layout = resolve_oasis2_source_layout(resolved_settings, source_root=source_root)
    if not layout.inspection_roots:
        raise FileNotFoundError(
            f"No OASIS-2 raw roots were found from {layout.source_root}. "
            f"Set {OASIS2_SOURCE_ENV_VAR} or pass --source-root."
        )

    rows, dropped_rows = _build_volume_rows(layout.inspection_roots)
    inventory_frame = pd.DataFrame(rows)
    dropped_frame = pd.DataFrame(dropped_rows)
    destination_root = ensure_directory(output_root or (resolved_settings.data_root / "interim"))

    inventory_path = destination_root / "oasis2_raw_inventory.csv"
    dropped_rows_path = destination_root / "oasis2_raw_inventory_dropped_rows.csv"
    summary_path = destination_root / "oasis2_raw_inventory_summary.json"

    inventory_frame.to_csv(inventory_path, index=False)
    dropped_frame.to_csv(dropped_rows_path, index=False)

    subject_counts = inventory_frame["subject_id"].value_counts().to_dict() if not inventory_frame.empty else {}
    session_counts = inventory_frame["session_id"].value_counts().to_dict() if not inventory_frame.empty else {}
    longitudinal_subjects = sum(1 for count in pd.Series(session_counts).groupby(lambda key: key.split("_MR", 1)[0]).size().tolist() if count > 1) if session_counts else 0
    volume_format_counts = inventory_frame["volume_format"].value_counts().to_dict() if not inventory_frame.empty else {}
    part_counts = inventory_frame["source_part"].value_counts().to_dict() if not inventory_frame.empty else {}

    summary_payload = {
        "source_root": str(layout.source_root),
        "source_resolution": layout.source_resolution,
        "inspection_roots": [str(path) for path in layout.inspection_roots],
        "volume_row_count": int(len(inventory_frame)),
        "dropped_row_count": int(len(dropped_frame)),
        "unique_subject_count": int(inventory_frame["subject_id"].nunique()) if not inventory_frame.empty else 0,
        "unique_session_count": int(inventory_frame["session_id"].nunique()) if not inventory_frame.empty else 0,
        "longitudinal_subject_count": int(longitudinal_subjects),
        "volume_format_counts": {str(key): int(value) for key, value in volume_format_counts.items()},
        "part_counts": {str(key): int(value) for key, value in part_counts.items()},
        "subject_examples": sorted(subject_counts)[:5],
        "session_examples": sorted(session_counts)[:5],
        "notes": [
            "This is a raw OASIS-2 structural MRI inventory, not a labeled training manifest.",
            "Clinical labels and visit metadata are not inferred here and should remain explicit in future onboarding work.",
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return OASIS2RawInventoryResult(
        inventory_path=inventory_path,
        dropped_rows_path=dropped_rows_path,
        summary_path=summary_path,
        session_row_count=int(len(inventory_frame)),
        unique_subject_count=summary_payload["unique_subject_count"],
        unique_session_count=summary_payload["unique_session_count"],
        part_roots=layout.inspection_roots,
    )


def _acquisition_sort_key(acquisition_id: str | None) -> tuple[int, str]:
    """Return a stable sort key that prefers the lowest MPR acquisition index."""

    normalized = (acquisition_id or "").strip().lower()
    match = OASIS2_ACQUISITION_PATTERN.search(normalized)
    if match:
        try:
            return int(match.group(1).split("-", 1)[1]), normalized
        except (IndexError, ValueError):
            return 10**9, normalized
    return 10**9, normalized


def _load_or_build_inventory_frame(
    settings: AppSettings,
    *,
    inventory_path: Path | None = None,
    source_root: Path | None = None,
) -> tuple[pd.DataFrame, OASIS2RawInventoryResult]:
    """Load an existing OASIS-2 raw inventory or build one if missing."""

    if inventory_path is not None and inventory_path.exists():
        frame = pd.read_csv(inventory_path)
        layout = resolve_oasis2_source_layout(settings, source_root=source_root)
        return (
            frame,
            OASIS2RawInventoryResult(
                inventory_path=inventory_path,
                dropped_rows_path=settings.data_root / "interim" / "oasis2_raw_inventory_dropped_rows.csv",
                summary_path=settings.data_root / "interim" / "oasis2_raw_inventory_summary.json",
                session_row_count=int(len(frame)),
                unique_subject_count=int(frame["subject_id"].nunique()) if not frame.empty else 0,
                unique_session_count=int(frame["session_id"].nunique()) if not frame.empty else 0,
                part_roots=layout.inspection_roots,
            ),
        )

    result = build_oasis2_raw_inventory(settings=settings, source_root=source_root)
    frame = pd.read_csv(result.inventory_path)
    return frame, result


def build_oasis2_session_manifest(
    settings: AppSettings | None = None,
    *,
    inventory_path: Path | None = None,
    source_root: Path | None = None,
    output_root: Path | None = None,
) -> OASIS2SessionManifestResult:
    """Build an unlabeled OASIS-2 session manifest and longitudinal-ready records."""

    resolved_settings = settings or AppSettings.from_env()
    inventory_frame, inventory_result = _load_or_build_inventory_frame(
        resolved_settings,
        inventory_path=inventory_path,
        source_root=source_root,
    )
    if inventory_frame.empty:
        raise ValueError("OASIS-2 raw inventory is empty; cannot build a session manifest.")

    destination_root = ensure_directory(output_root or (resolved_settings.data_root / "interim"))
    manifest_path = destination_root / "oasis2_session_manifest.csv"
    longitudinal_records_path = destination_root / "oasis2_longitudinal_records.csv"
    subject_summary_path = destination_root / "oasis2_subject_summary.csv"
    summary_path = destination_root / "oasis2_session_manifest_summary.json"

    working = inventory_frame.copy()
    working["acquisition_sort_key"] = working["acquisition_id"].map(_acquisition_sort_key)
    working = working.sort_values(
        by=["subject_id", "visit_number", "session_id", "acquisition_sort_key", "source_part", "image"],
        kind="stable",
    ).reset_index(drop=True)

    manifest_rows: list[dict[str, Any]] = []
    for session_id, session_frame in working.groupby("session_id", sort=False):
        selected = session_frame.iloc[0]
        available_acquisitions = session_frame["acquisition_id"].dropna().astype(str).tolist()
        meta_payload = json.loads(selected["meta"]) if isinstance(selected["meta"], str) else dict(selected["meta"])
        meta_payload.update(
            {
                "available_acquisition_ids": available_acquisitions,
                "acquisition_count": int(len(session_frame)),
                "selected_acquisition_id": str(selected["acquisition_id"]),
                "selection_strategy": "lowest_mpr_index",
                "paired_image": None if pd.isna(selected.get("paired_image")) else str(selected.get("paired_image")),
            }
        )
        manifest_rows.append(
            {
                "image": str(selected["image"]),
                "label": None,
                "label_name": None,
                "subject_id": str(selected["subject_id"]),
                "session_id": str(session_id),
                "visit_number": int(selected["visit_number"]),
                "scan_timestamp": None,
                "dataset": "oasis2_raw",
                "dataset_type": "3d_volumes",
                "meta": json.dumps(meta_payload, ensure_ascii=True, sort_keys=True),
            }
        )

    manifest_frame = pd.DataFrame(manifest_rows).sort_values(
        by=["subject_id", "visit_number", "session_id"],
        kind="stable",
    )
    manifest_frame.to_csv(manifest_path, index=False)

    longitudinal_records = manifest_frame.rename(
        columns={
            "image": "source_path",
            "visit_number": "visit_order",
        }
    )
    longitudinal_records.insert(0, "record_type", "oasis2_session")
    longitudinal_records.to_csv(longitudinal_records_path, index=False)

    subject_summary = (
        manifest_frame.groupby("subject_id", sort=True)
        .agg(
            session_count=("session_id", "nunique"),
            first_visit=("visit_number", "min"),
            last_visit=("visit_number", "max"),
        )
        .reset_index()
    )
    session_lists = (
        manifest_frame.groupby("subject_id", sort=True)["session_id"]
        .apply(lambda values: "|".join(str(value) for value in values))
        .reset_index(name="session_ids")
    )
    subject_summary = subject_summary.merge(session_lists, on="subject_id", how="left")
    subject_summary.to_csv(subject_summary_path, index=False)

    summary_payload = {
        "source_root": str(inventory_result.part_roots[0].parent) if inventory_result.part_roots else None,
        "part_roots": [str(path) for path in inventory_result.part_roots],
        "inventory_path": str(inventory_result.inventory_path),
        "manifest_path": str(manifest_path),
        "longitudinal_records_path": str(longitudinal_records_path),
        "subject_summary_path": str(subject_summary_path),
        "session_row_count": int(len(manifest_frame)),
        "unique_subject_count": int(manifest_frame["subject_id"].nunique()),
        "longitudinal_subject_count": int((subject_summary["session_count"] > 1).sum()),
        "selection_strategy": "lowest_mpr_index",
        "notes": [
            "This OASIS-2 session manifest is unlabeled and should not be treated as a supervised training manifest.",
            "One representative structural acquisition is selected per session using the lowest available MPR index.",
            "This file is suitable for preprocessing, structural pipelines, and future longitudinal tracking preparation.",
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return OASIS2SessionManifestResult(
        manifest_path=manifest_path,
        longitudinal_records_path=longitudinal_records_path,
        subject_summary_path=subject_summary_path,
        summary_path=summary_path,
        session_row_count=int(len(manifest_frame)),
        unique_subject_count=int(manifest_frame["subject_id"].nunique()),
        source_part_roots=inventory_result.part_roots,
    )
