"""Dedicated OASIS-1 adapter and manifest builder.

Assumptions:
- OASIS-1 is treated as a session-level dataset where one manifest row maps to one MRI session.
- The primary metadata table is the largest OASIS tabular file that contains a session identifier and label column.
- For binary classification, CDR `0` maps to nondemented/control and CDR `> 0` maps to demented/AD-like.
- Preferred image selection is explicit and ordered: `MASKED` > `T88` > `SUBJ` > `MPRAGE_RAW`.
- The adapter does not harmonize labels beyond the requested binary mapping and stores source label details in `meta`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

from .inspection_utils import DATE_PATTERN, DEFAULT_IGNORED_NAMES, collect_files, get_extension

SESSION_COLUMN_CANDIDATES = ("id", "sessionid", "session_id", "mrid", "mriid")
LABEL_COLUMN_CANDIDATES = ("cdr", "group", "diagnosis", "clinicalstatus", "clinical_status", "label")
TIMESTAMP_COLUMN_CANDIDATES = ("scantimestamp", "scan_timestamp", "scandate", "acquisitiondate", "date", "timestamp")
PREFERRED_CONTENT_ORDER = ("MASKED", "T88", "SUBJ", "MPRAGE_RAW")
LABEL_NAME_MAP = {0: "nondemented", 1: "demented"}


class OASIS1ManifestError(ValueError):
    """Raised when OASIS-1 metadata is ambiguous or structurally invalid."""


@dataclass(slots=True, frozen=True)
class OASIS1ImageCandidate:
    """A candidate MRI resource extracted from an OASIS session XML file."""

    session_id: str
    subject_id: str
    content: str
    format_name: str
    xml_path: Path
    image_path: Path
    uri: str
    dimensions: tuple[int, ...]
    voxel_spacing: tuple[float, ...]


@dataclass(slots=True)
class OASIS1ManifestResult:
    """Paths and counts produced by a manifest build run."""

    manifest_csv_path: Path | None
    manifest_jsonl_path: Path | None
    dropped_rows_path: Path
    manifest_row_count: int
    dropped_row_count: int


def _normalized_column_name(column_name: str) -> str:
    """Normalize a column name for heuristic matching."""

    return "".join(character for character in column_name.lower() if character.isalnum())


def _matching_columns(columns: list[str], candidates: tuple[str, ...]) -> list[str]:
    """Return all columns whose normalized names match the candidate set."""

    candidate_set = set(candidates)
    return [column for column in columns if _normalized_column_name(column) in candidate_set]


def _resolve_unique_column(
    columns: list[str],
    candidates: tuple[str, ...],
    *,
    field_name: str,
    required: bool,
) -> str | None:
    """Resolve one metadata column or fail loudly when ambiguous."""

    matches = _matching_columns(columns, candidates)
    if not matches:
        if required:
            raise OASIS1ManifestError(
                f"Could not find a required {field_name} column. Available columns: {columns}"
            )
        return None
    if len(matches) > 1:
        raise OASIS1ManifestError(
            f"Ambiguous {field_name} columns detected: {matches}. Please clean the metadata selection logic explicitly."
        )
    return matches[0]


def resolve_oasis1_metadata_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    """Resolve the session, label, and optional timestamp columns for OASIS-1."""

    columns = [str(column) for column in frame.columns]
    return {
        "session_id": _resolve_unique_column(
            columns,
            SESSION_COLUMN_CANDIDATES,
            field_name="session identifier",
            required=True,
        ),
        "label": _resolve_unique_column(
            columns,
            LABEL_COLUMN_CANDIDATES,
            field_name="label",
            required=True,
        ),
        "scan_timestamp": _resolve_unique_column(
            columns,
            TIMESTAMP_COLUMN_CANDIDATES,
            field_name="scan timestamp",
            required=False,
        ),
    }


def extract_subject_id(session_id: str) -> str:
    """Extract a reliable OASIS subject ID from a session identifier."""

    clean_session = str(session_id).strip()
    if "_MR" not in clean_session:
        raise OASIS1ManifestError(f"Session ID does not look like an OASIS MRI session: {session_id}")
    subject_id = clean_session.split("_MR", maxsplit=1)[0]
    if not subject_id.startswith("OAS1_"):
        raise OASIS1ManifestError(f"Could not derive a reliable OASIS subject ID from session: {session_id}")
    return subject_id


def map_oasis1_binary_label(raw_value: Any) -> tuple[int, str]:
    """Map OASIS source labels to the requested binary target labels."""

    if raw_value is None or pd.isna(raw_value):
        raise OASIS1ManifestError("Missing OASIS label.")

    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        string_map = {
            "0": 0,
            "0.0": 0,
            "control": 0,
            "cn": 0,
            "nondemented": 0,
            "non-demented": 0,
            "1": 1,
            "1.0": 1,
            "0.5": 1,
            "2": 1,
            "2.0": 1,
            "demented": 1,
            "ad": 1,
            "ad-like": 1,
            "alzheimers": 1,
            "alzheimer": 1,
        }
        if normalized in string_map:
            mapped = string_map[normalized]
            return mapped, LABEL_NAME_MAP[mapped]
        try:
            numeric_value = float(normalized)
        except ValueError as error:
            raise OASIS1ManifestError(f"Unsupported OASIS label value: {raw_value}") from error
    else:
        numeric_value = float(raw_value)

    if numeric_value == 0.0:
        return 0, LABEL_NAME_MAP[0]
    if numeric_value > 0.0:
        return 1, LABEL_NAME_MAP[1]
    raise OASIS1ManifestError(f"Unsupported OASIS numeric label value: {raw_value}")


def _load_tabular_metadata(path: Path) -> pd.DataFrame:
    """Load an OASIS metadata table from CSV or Excel."""

    extension = get_extension(path)
    if extension == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, engine="openpyxl")


def find_primary_oasis1_metadata_table(source_root: Path) -> tuple[Path, pd.DataFrame]:
    """Select the best OASIS metadata table for manifest construction."""

    candidate_files = [
        path
        for path in collect_files(source_root, DEFAULT_IGNORED_NAMES)
        if get_extension(path) in {".csv", ".xls", ".xlsx"}
    ]
    if not candidate_files:
        raise OASIS1ManifestError(f"No OASIS tabular metadata files were found under {source_root}.")

    successful_tables: list[tuple[Path, pd.DataFrame]] = []
    parse_errors: list[str] = []
    for path in candidate_files:
        try:
            frame = _load_tabular_metadata(path)
            resolve_oasis1_metadata_columns(frame)
        except Exception as error:
            parse_errors.append(f"{path}: {error}")
            continue
        successful_tables.append((path, frame))

    if not successful_tables:
        raise OASIS1ManifestError(
            "Could not find a usable OASIS metadata table with unambiguous session and label columns. "
            f"Errors: {parse_errors}"
        )

    successful_tables.sort(key=lambda item: (-len(item[1]), str(item[0])))
    return successful_tables[0]


def _resolve_oasis_uri_to_path(uri: str, source_root: Path, session_root: Path) -> Path:
    """Resolve an OASIS XML URI to a concrete load path on disk."""

    relative_parts = [part for part in uri.split("/") if part]
    if session_root.name in relative_parts:
        session_index = relative_parts.index(session_root.name)
        path = session_root.joinpath(*relative_parts[session_index + 1 :])
    else:
        if relative_parts and relative_parts[0].lower() == "cross-sectional":
            relative_parts = relative_parts[1:]
        path = source_root.joinpath(*relative_parts)
    if path.suffix.lower() == ".img":
        header_path = path.with_suffix(".hdr")
        if header_path.exists():
            return header_path
    return path


def parse_oasis1_xml_candidates(source_root: Path) -> dict[str, list[OASIS1ImageCandidate]]:
    """Parse OASIS XML files into session-level MRI image candidates."""

    xml_files = sorted(source_root.rglob("*.xml"))
    candidates_by_session: dict[str, list[OASIS1ImageCandidate]] = {}

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue

        session_id = xml_path.stem
        session_root = xml_path.parent
        try:
            subject_id = extract_subject_id(session_id)
        except OASIS1ManifestError:
            continue

        session_candidates: list[OASIS1ImageCandidate] = []
        for element in root.iter():
            tag = element.tag.lower()
            uri = element.attrib.get("URI")
            if not tag.endswith("file") or not uri:
                continue

            content = (element.attrib.get("content") or "").strip()
            if content not in {"MASKED", "T88", "SUBJ", "MPRAGE_RAW"}:
                continue

            dimensions: tuple[int, ...] = ()
            voxel_spacing: tuple[float, ...] = ()
            for child in list(element):
                child_tag = child.tag.lower()
                if child_tag.endswith("dimensions"):
                    values = [child.attrib.get(axis) for axis in ("x", "y", "z") if child.attrib.get(axis)]
                    if values:
                        dimensions = tuple(int(value) for value in values)
                if "voxelres" in child_tag:
                    values = [child.attrib.get(axis) for axis in ("x", "y", "z") if child.attrib.get(axis)]
                    if values:
                        voxel_spacing = tuple(float(value) for value in values)

            image_path = _resolve_oasis_uri_to_path(uri, source_root, session_root)
            session_candidates.append(
                OASIS1ImageCandidate(
                    session_id=session_id,
                    subject_id=subject_id,
                    content=content,
                    format_name=element.attrib.get("format", ""),
                    xml_path=xml_path,
                    image_path=image_path,
                    uri=uri,
                    dimensions=dimensions,
                    voxel_spacing=voxel_spacing,
                )
            )

        if session_candidates:
            candidates_by_session[session_id] = sorted(
                session_candidates,
                key=lambda candidate: (
                    PREFERRED_CONTENT_ORDER.index(candidate.content),
                    candidate.image_path.name,
                ),
            )

    return candidates_by_session


def choose_preferred_oasis1_image(candidates: list[OASIS1ImageCandidate]) -> OASIS1ImageCandidate | None:
    """Choose the preferred image candidate for one OASIS session."""

    for candidate in candidates:
        if candidate.image_path.exists():
            return candidate
    return None


def _parse_scan_timestamp(raw_value: Any) -> str | None:
    """Normalize a timestamp-like field to string when possible."""

    if raw_value is None or pd.isna(raw_value):
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if DATE_PATTERN.search(text):
        return text
    return None


def _build_manifest_rows(
    metadata_frame: pd.DataFrame,
    metadata_columns: dict[str, str | None],
    candidates_by_session: dict[str, list[OASIS1ImageCandidate]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build normalized manifest rows plus dropped-row audit records."""

    manifest_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()

    session_column = metadata_columns["session_id"]
    label_column = metadata_columns["label"]
    timestamp_column = metadata_columns["scan_timestamp"]
    assert session_column is not None
    assert label_column is not None

    for row_index, row in metadata_frame.iterrows():
        row_session_id = str(row[session_column]).strip() if not pd.isna(row[session_column]) else ""
        if not row_session_id:
            dropped_rows.append({"row_index": int(row_index), "reason": "missing_session_id"})
            continue

        if row_session_id in seen_sessions:
            dropped_rows.append(
                {"row_index": int(row_index), "session_id": row_session_id, "reason": "duplicate_session_in_metadata"}
            )
            continue

        seen_sessions.add(row_session_id)

        try:
            subject_id = extract_subject_id(row_session_id)
            label, label_name = map_oasis1_binary_label(row[label_column])
        except OASIS1ManifestError as error:
            dropped_rows.append(
                {"row_index": int(row_index), "session_id": row_session_id, "reason": str(error)}
            )
            continue

        candidates = candidates_by_session.get(row_session_id, [])
        if not candidates:
            dropped_rows.append(
                {"row_index": int(row_index), "session_id": row_session_id, "reason": "no_xml_image_candidates"}
            )
            continue

        preferred_candidate = choose_preferred_oasis1_image(candidates)
        if preferred_candidate is None:
            dropped_rows.append(
                {"row_index": int(row_index), "session_id": row_session_id, "reason": "no_existing_image_path"}
            )
            continue

        if not preferred_candidate.image_path.exists():
            dropped_rows.append(
                {
                    "row_index": int(row_index),
                    "session_id": row_session_id,
                    "reason": "image_path_missing_on_disk",
                    "image_path": str(preferred_candidate.image_path),
                }
            )
            continue

        scan_timestamp = _parse_scan_timestamp(row[timestamp_column]) if timestamp_column else None
        manifest_rows.append(
            {
                "image": str(preferred_candidate.image_path),
                "label": label,
                "label_name": label_name,
                "subject_id": subject_id,
                "scan_timestamp": scan_timestamp,
                "dataset": "oasis1",
                "meta": {
                    "session_id": row_session_id,
                    "source_label": None if pd.isna(row[label_column]) else row[label_column],
                    "source_label_column": label_column,
                    "selected_image_content": preferred_candidate.content,
                    "selected_image_format": preferred_candidate.format_name,
                    "selected_image_uri": preferred_candidate.uri,
                    "xml_path": str(preferred_candidate.xml_path),
                    "dimensions": list(preferred_candidate.dimensions),
                    "voxel_spacing": list(preferred_candidate.voxel_spacing),
                    "available_image_contents": [candidate.content for candidate in candidates],
                },
            }
        )

    return manifest_rows, dropped_rows


def save_oasis1_manifest(
    rows: list[dict[str, Any]],
    destination_root: Path,
    *,
    file_stem: str = "oasis1_manifest",
    output_format: str = "csv",
) -> tuple[Path | None, Path | None]:
    """Save manifest rows to CSV and/or JSONL."""

    ensure_directory(destination_root)
    manifest_frame = pd.DataFrame(
        [
            {
                **row,
                "meta": json.dumps(row["meta"], sort_keys=True),
            }
            for row in rows
        ]
    )

    csv_path: Path | None = None
    jsonl_path: Path | None = None

    if output_format in {"csv", "both"}:
        csv_path = destination_root / f"{file_stem}.csv"
        manifest_frame.to_csv(csv_path, index=False)

    if output_format in {"jsonl", "both"}:
        jsonl_path = destination_root / f"{file_stem}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    return csv_path, jsonl_path


def save_dropped_rows_log(
    dropped_rows: list[dict[str, Any]],
    destination_root: Path,
    *,
    file_name: str = "oasis1_manifest_dropped_rows.csv",
) -> Path:
    """Save dropped-row audit information to CSV."""

    ensure_directory(destination_root)
    output_path = destination_root / file_name
    pd.DataFrame(dropped_rows).to_csv(output_path, index=False)
    return output_path


def build_oasis1_manifest(
    settings: AppSettings | None = None,
    *,
    output_format: str = "csv",
) -> OASIS1ManifestResult:
    """Build and save a normalized OASIS-1 manifest."""

    if output_format not in {"csv", "jsonl", "both"}:
        raise OASIS1ManifestError(f"Unsupported output format: {output_format}")

    resolved_settings = settings or get_app_settings()
    source_root = resolved_settings.oasis_source_root
    destination_root = resolved_settings.data_root / "interim"

    metadata_path, metadata_frame = find_primary_oasis1_metadata_table(source_root)
    metadata_columns = resolve_oasis1_metadata_columns(metadata_frame)
    candidates_by_session = parse_oasis1_xml_candidates(source_root)
    manifest_rows, dropped_rows = _build_manifest_rows(metadata_frame, metadata_columns, candidates_by_session)

    csv_path, jsonl_path = save_oasis1_manifest(
        manifest_rows,
        destination_root,
        output_format=output_format,
    )
    dropped_rows_path = save_dropped_rows_log(dropped_rows, destination_root)

    summary_path = destination_root / "oasis1_manifest_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset": "oasis1",
                "source_root": str(source_root),
                "metadata_path": str(metadata_path),
                "metadata_columns": metadata_columns,
                "manifest_row_count": len(manifest_rows),
                "dropped_row_count": len(dropped_rows),
                "output_format": output_format,
                "assumptions": [
                    "Binary mapping uses CDR 0 -> nondemented and CDR > 0 -> demented.",
                    "Preferred image selection order is MASKED, then T88, then SUBJ, then MPRAGE_RAW.",
                    "Session IDs and subject IDs are preserved for future split and longitudinal logic.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return OASIS1ManifestResult(
        manifest_csv_path=csv_path,
        manifest_jsonl_path=jsonl_path,
        dropped_rows_path=dropped_rows_path,
        manifest_row_count=len(manifest_rows),
        dropped_row_count=len(dropped_rows),
    )
