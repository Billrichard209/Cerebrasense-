"""Metadata parsing helpers for OASIS and Kaggle dataset inspection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from .inspection_utils import (
    DATE_PATTERN,
    DEFAULT_IGNORED_NAMES,
    collect_files,
    find_date_like_strings,
    get_extension,
    infer_session_id_from_path,
    infer_subject_id_from_path,
)


def _normalized_column_name(column_name: str) -> str:
    """Normalize a column name for heuristic matching."""

    return "".join(character for character in column_name.lower() if character.isalnum())


def _select_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Pick the best matching column from a list of names."""

    normalized_lookup = {_normalized_column_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized_lookup:
            return normalized_lookup[candidate]
    return None


def parse_oasis_xml_metadata(xml_files: list[Path]) -> dict[str, Any]:
    """Parse OASIS XML files for subject IDs, session IDs, timestamps, and spacing hints."""

    subject_ids: set[str] = set()
    session_ids: set[str] = set()
    timestamp_examples: list[str] = []
    xml_errors: list[dict[str, str]] = []
    voxel_resolutions: Counter[str] = Counter()
    image_resources: list[dict[str, Any]] = []

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as error:
            xml_errors.append({"path": str(xml_path), "error": str(error)})
            continue

        inferred_subject = infer_subject_id_from_path(xml_path)
        inferred_session = infer_session_id_from_path(xml_path)
        if inferred_subject:
            subject_ids.add(inferred_subject)
        if inferred_session:
            session_ids.add(inferred_session)

        for element in root.iter():
            tag = element.tag.lower()
            text = (element.text or "").strip()
            if tag.endswith("subject_id") and text:
                subject_ids.add(text)
            if tag.endswith("mrsession_id") and text:
                session_ids.add(text)
            if "voxelres" in tag and element.attrib:
                values = [element.attrib.get(axis) for axis in ("x", "y", "z") if element.attrib.get(axis)]
                if values:
                    voxel_resolutions["x".join(values)] += 1
            if DATE_PATTERN.search(text):
                timestamp_examples.append(text)
            if tag.endswith("file") and element.attrib.get("URI"):
                dimensions: tuple[int, ...] | tuple[()] = ()
                spacing: tuple[float, ...] | tuple[()] = ()
                for child in list(element):
                    child_tag = child.tag.lower()
                    if child_tag.endswith("dimensions"):
                        values = [child.attrib.get(axis) for axis in ("x", "y", "z") if child.attrib.get(axis)]
                        if values:
                            dimensions = tuple(int(value) for value in values)
                    if "voxelres" in child_tag:
                        values = [child.attrib.get(axis) for axis in ("x", "y", "z") if child.attrib.get(axis)]
                        if values:
                            spacing = tuple(float(value) for value in values)
                image_resources.append(
                    {
                        "xml_path": str(xml_path),
                        "uri": element.attrib.get("URI"),
                        "content": element.attrib.get("content"),
                        "format": element.attrib.get("format"),
                        "dimensions": dimensions,
                        "voxel_spacing": spacing,
                    }
                )

    return {
        "subject_ids": sorted(subject_ids),
        "session_ids": sorted(session_ids),
        "timestamp_examples": timestamp_examples[:10],
        "xml_errors": xml_errors,
        "xml_voxel_spacing_distribution": dict(sorted(voxel_resolutions.items())),
        "image_resources": image_resources,
    }


def parse_oasis_tabular_metadata(root: Path) -> dict[str, Any]:
    """Parse OASIS spreadsheet metadata if available."""

    spreadsheets = [
        path
        for path in collect_files(root, DEFAULT_IGNORED_NAMES)
        if get_extension(path) in {".xls", ".xlsx", ".csv"}
    ]

    if not spreadsheets:
        return {
            "metadata_files": [],
            "spreadsheet_errors": [],
            "label_distribution": {},
            "label_source": "unavailable",
            "subject_ids": [],
            "timestamp_examples": [],
            "row_count": 0,
        }

    spreadsheet_errors: list[dict[str, str]] = []
    best_frame: pd.DataFrame | None = None
    best_path: Path | None = None

    for path in spreadsheets:
        try:
            if get_extension(path) == ".csv":
                frame = pd.read_csv(path)
            else:
                frame = pd.read_excel(path, engine="openpyxl")
        except Exception as error:  # pragma: no cover - defensive against local file variance
            spreadsheet_errors.append({"path": str(path), "error": str(error)})
            continue

        if best_frame is None or len(frame) > len(best_frame):
            best_frame = frame
            best_path = path

    if best_frame is None:
        return {
            "metadata_files": [str(path) for path in spreadsheets],
            "spreadsheet_errors": spreadsheet_errors,
            "label_distribution": {},
            "label_source": "unavailable",
            "subject_ids": [],
            "timestamp_examples": [],
            "row_count": 0,
        }

    columns = [str(column) for column in best_frame.columns]
    id_column = _select_column(columns, ("id", "subjectid", "subject_id"))
    label_column = _select_column(columns, ("group", "cdr", "clinicalstatus", "diagnosis", "label"))
    timestamp_column = _select_column(columns, ("scandate", "acquisitiondate", "date", "timestamp"))

    label_distribution: dict[str, int] = {}
    if label_column:
        series = best_frame[label_column].dropna().astype(str).str.strip()
        label_distribution = dict(sorted(series.value_counts().to_dict().items()))

    subject_ids = []
    if id_column:
        subject_ids = sorted(best_frame[id_column].dropna().astype(str).str.strip().unique().tolist())

    timestamp_examples = []
    if timestamp_column:
        timestamp_examples = find_date_like_strings(best_frame[timestamp_column].dropna().tolist())

    return {
        "metadata_files": [str(path) for path in spreadsheets],
        "primary_metadata_file": str(best_path) if best_path else None,
        "spreadsheet_errors": spreadsheet_errors,
        "label_distribution": label_distribution,
        "label_source": f"spreadsheet:{label_column}" if label_column else "unavailable",
        "subject_ids": subject_ids,
        "timestamp_examples": timestamp_examples,
        "row_count": int(len(best_frame)),
        "columns": columns,
    }


def parse_kaggle_tabular_metadata(root: Path) -> dict[str, Any]:
    """Parse Kaggle-side metadata files if any exist."""

    candidate_files = [
        path
        for path in collect_files(root, DEFAULT_IGNORED_NAMES)
        if get_extension(path) in {".xls", ".xlsx", ".csv", ".tsv", ".json", ".txt"}
    ]

    tabular_files = [path for path in candidate_files if get_extension(path) in {".xls", ".xlsx", ".csv", ".tsv"}]
    errors: list[dict[str, str]] = []
    dataframes: list[tuple[Path, pd.DataFrame]] = []

    for path in tabular_files:
        try:
            extension = get_extension(path)
            if extension == ".csv":
                frame = pd.read_csv(path)
            elif extension == ".tsv":
                frame = pd.read_csv(path, sep="\t")
            else:
                frame = pd.read_excel(path, engine="openpyxl")
        except Exception as error:  # pragma: no cover - depends on local files
            errors.append({"path": str(path), "error": str(error)})
            continue
        dataframes.append((path, frame))

    timestamp_examples: list[str] = []
    subject_ids: set[str] = set()
    for _, frame in dataframes:
        for column in frame.columns:
            normalized = _normalized_column_name(str(column))
            if normalized in {"id", "subjectid", "patientid"}:
                subject_ids.update(frame[column].dropna().astype(str).str.strip().tolist())
            if "date" in normalized or "time" in normalized:
                timestamp_examples.extend(find_date_like_strings(frame[column].dropna().tolist(), limit=3))

    return {
        "metadata_files": [str(path) for path in candidate_files],
        "spreadsheet_errors": errors,
        "subject_ids": sorted(subject_ids),
        "timestamp_examples": timestamp_examples[:10],
    }
