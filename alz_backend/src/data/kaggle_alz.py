"""Flexible Kaggle Alzheimer dataset adapter and manifest builder.

Assumptions:
- Kaggle data remain separate from OASIS and are never harmonized implicitly.
- If the dataset looks slice-based, the adapter warns clearly and marks rows as `2d_slices`.
- Explicit label remapping is optional. Without it, original labels are preserved and numeric labels stay unset.
- The adapter supports common layouts: class-folder datasets, metadata-table datasets, and unlabeled flat-file datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

from .inspection_utils import (
    DEFAULT_IGNORED_NAMES,
    IMAGE_EXTENSIONS,
    VOLUME_EXTENSIONS,
    collect_files,
    get_extension,
    sample_evenly,
)
from .kaggle_dataset import build_kaggle_dataset_spec

KAGGLE_IGNORED_NAMES = DEFAULT_IGNORED_NAMES | {"scripts"}
PATH_COLUMN_CANDIDATES = (
    "image",
    "imagepath",
    "image_path",
    "filepath",
    "file_path",
    "path",
    "scan",
    "scanpath",
    "scan_path",
    "volume",
    "volumepath",
    "volume_path",
)
LABEL_COLUMN_CANDIDATES = ("label", "class", "group", "diagnosis", "category", "target")
SUBJECT_COLUMN_CANDIDATES = ("subjectid", "subject_id", "patientid", "patient_id", "id", "scanid", "scan_id")
TIMESTAMP_COLUMN_CANDIDATES = ("scantimestamp", "scan_timestamp", "date", "timestamp", "scandate", "acquisitiondate")


class KaggleAlzManifestError(ValueError):
    """Raised when Kaggle manifest generation cannot proceed safely."""


@dataclass(slots=True, frozen=True)
class KaggleDatasetOrganization:
    """Represents the likely organization of a Kaggle Alzheimer dataset."""

    kind: str
    dataset_type: str
    source_root: Path
    notes: tuple[str, ...] = ()
    metadata_path: Path | None = None
    image_path_column: str | None = None
    label_column: str | None = None
    subject_id_column: str | None = None
    scan_timestamp_column: str | None = None
    subset_roots: tuple[str, ...] = ()


@dataclass(slots=True)
class KaggleManifestResult:
    """Artifacts and counts produced by a Kaggle manifest build run."""

    organization: KaggleDatasetOrganization
    manifest_csv_path: Path | None
    manifest_jsonl_path: Path | None
    dropped_rows_path: Path
    summary_path: Path
    manifest_row_count: int
    dropped_row_count: int
    warnings: list[str] = field(default_factory=list)


def _normalized_column_name(column_name: str) -> str:
    """Normalize column names for heuristic matching."""

    return "".join(character for character in str(column_name).lower() if character.isalnum())


def _matching_columns(columns: list[str], candidates: tuple[str, ...]) -> list[str]:
    """Return columns whose normalized forms match the candidate set."""

    candidate_set = set(candidates)
    return [column for column in columns if _normalized_column_name(column) in candidate_set]


def _first_matching_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column for a heuristic candidate list."""

    matches = _matching_columns(columns, candidates)
    return matches[0] if matches else None


def _load_tabular_file(path: Path) -> pd.DataFrame:
    """Load a CSV/TSV/Excel metadata file."""

    extension = get_extension(path)
    if extension == ".csv":
        return pd.read_csv(path)
    if extension == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_excel(path, engine="openpyxl")


def _resolve_existing_image_path(raw_value: Any, *, source_root: Path, metadata_path: Path | None = None) -> Path | None:
    """Resolve a metadata-provided path against likely roots."""

    if raw_value is None or pd.isna(raw_value):
        return None
    raw_path = Path(str(raw_value).strip())
    if not str(raw_path):
        return None

    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        if metadata_path is not None:
            candidates.append((metadata_path.parent / raw_path).resolve())
        candidates.append((source_root / raw_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix.lower() == ".img":
                header_path = candidate.with_suffix(".hdr")
                if header_path.exists():
                    return header_path
            return candidate
    return None


def _infer_dataset_type_from_samples(paths: list[Path]) -> tuple[str, list[str]]:
    """Infer whether a dataset is composed of 2D slices or 3D volumes."""

    if not paths:
        return "unknown", ["No imaging files were available to infer dataset type."]

    sampled_paths = sample_evenly(paths, 32)
    image_like = 0
    volume_like = 0
    notes: list[str] = []

    for path in sampled_paths:
        extension = get_extension(path)
        if extension in IMAGE_EXTENSIONS:
            image_like += 1
            continue
        if extension in VOLUME_EXTENSIONS:
            volume_like += 1
            if extension in {".hdr", ".nii", ".nii.gz"}:
                try:
                    image = nib.load(str(path))
                    shape = tuple(int(part) for part in image.shape)
                    if len(shape) >= 3:
                        volume_like += 1
                except Exception as error:  # pragma: no cover - depends on local file variance
                    notes.append(f"Could not inspect volume header for {path.name}: {error}")
            continue

    if volume_like > image_like and volume_like > 0:
        return "3d_volumes", notes
    if image_like > 0 and volume_like == 0:
        notes.append(
            "Detected slice-based image files. Treat this Kaggle dataset as 2D slices, not subject-level 3D MRI volumes."
        )
        return "2d_slices", notes
    if image_like > 0 and volume_like > 0:
        notes.append("Detected mixed 2D and 3D imaging formats. Review the manifest carefully before training.")
        return "mixed_unknown", notes
    return "unknown", notes


def _collect_imaging_files(source_root: Path) -> list[Path]:
    """Collect likely imaging files beneath the Kaggle source root."""

    return [
        path
        for path in collect_files(source_root, KAGGLE_IGNORED_NAMES)
        if get_extension(path) in IMAGE_EXTENSIONS | VOLUME_EXTENSIONS
    ]


def _detect_metadata_table_organization(source_root: Path) -> KaggleDatasetOrganization | None:
    """Detect a metadata-table-driven organization if one exists."""

    metadata_files = [
        path
        for path in collect_files(source_root, KAGGLE_IGNORED_NAMES)
        if get_extension(path) in {".csv", ".tsv", ".xls", ".xlsx"}
    ]
    table_candidates: list[tuple[Path, pd.DataFrame, str, str | None, str | None, str | None]] = []

    for path in metadata_files:
        try:
            frame = _load_tabular_file(path)
        except Exception:
            continue
        columns = [str(column) for column in frame.columns]
        image_path_column = _first_matching_column(columns, PATH_COLUMN_CANDIDATES)
        if image_path_column is None:
            continue
        label_column = _first_matching_column(columns, LABEL_COLUMN_CANDIDATES)
        subject_column = _first_matching_column(columns, SUBJECT_COLUMN_CANDIDATES)
        timestamp_column = _first_matching_column(columns, TIMESTAMP_COLUMN_CANDIDATES)
        table_candidates.append((path, frame, image_path_column, label_column, subject_column, timestamp_column))

    if not table_candidates:
        return None

    table_candidates.sort(key=lambda item: (-len(item[1]), str(item[0])))
    metadata_path, frame, image_path_column, label_column, subject_column, timestamp_column = table_candidates[0]

    resolved_paths = [
        _resolve_existing_image_path(value, source_root=source_root, metadata_path=metadata_path)
        for value in frame[image_path_column].dropna().tolist()
    ]
    existing_paths = [path for path in resolved_paths if path is not None]
    dataset_type, type_notes = _infer_dataset_type_from_samples(existing_paths)

    notes = [f"Detected metadata-table organization from {metadata_path.name}."] + type_notes
    if label_column is None:
        notes.append("No label column was detected in the metadata table; labels will remain unset unless inferred elsewhere.")

    return KaggleDatasetOrganization(
        kind="metadata_table",
        dataset_type=dataset_type,
        source_root=source_root,
        notes=tuple(notes),
        metadata_path=metadata_path,
        image_path_column=image_path_column,
        label_column=label_column,
        subject_id_column=subject_column,
        scan_timestamp_column=timestamp_column,
    )


def _detect_class_folder_organization(source_root: Path) -> KaggleDatasetOrganization | None:
    """Detect a class-folder organization."""

    imaging_files = _collect_imaging_files(source_root)
    if not imaging_files:
        return None

    label_names = sorted({path.parent.name for path in imaging_files if path.parent != source_root})
    if len(label_names) < 2:
        return None

    subset_roots = sorted(
        {
            path.relative_to(source_root).parts[0]
            for path in imaging_files
            if len(path.relative_to(source_root).parts) >= 3
        }
    )
    dataset_type, type_notes = _infer_dataset_type_from_samples(imaging_files)
    notes = ["Detected class-folder organization from directory structure."] + type_notes
    if dataset_type == "2d_slices":
        notes.append("Class folders appear slice-based; do not treat these labels as OASIS-equivalent without explicit remapping.")

    return KaggleDatasetOrganization(
        kind="class_folders",
        dataset_type=dataset_type,
        source_root=source_root,
        notes=tuple(notes),
        subset_roots=tuple(subset_roots),
    )


def detect_kaggle_dataset_organization(source_root: Path) -> KaggleDatasetOrganization:
    """Detect the most likely organization of a Kaggle Alzheimer dataset."""

    metadata_first = _detect_metadata_table_organization(source_root)
    if metadata_first is not None:
        return metadata_first

    class_folder = _detect_class_folder_organization(source_root)
    if class_folder is not None:
        return class_folder

    imaging_files = _collect_imaging_files(source_root)
    if imaging_files:
        dataset_type, type_notes = _infer_dataset_type_from_samples(imaging_files)
        return KaggleDatasetOrganization(
            kind="flat_files",
            dataset_type=dataset_type,
            source_root=source_root,
            notes=tuple(["Detected unlabeled flat-file organization."] + type_notes),
        )

    raise KaggleAlzManifestError(f"No Kaggle imaging files were found under {source_root}.")


def _normalize_label_remap_config(label_remap: dict[str, Any] | Path | None) -> dict[str, dict[str, Any]]:
    """Normalize an optional explicit label remap configuration."""

    if label_remap is None:
        return {}
    raw_config: dict[str, Any]
    if isinstance(label_remap, Path):
        raw_config = json.loads(label_remap.read_text(encoding="utf-8"))
    else:
        raw_config = label_remap

    normalized: dict[str, dict[str, Any]] = {}
    for original_label, target in raw_config.items():
        key = str(original_label).strip()
        if isinstance(target, int):
            normalized[key] = {"label": int(target), "label_name": key}
            continue
        if isinstance(target, dict) and "label" in target:
            normalized[key] = {
                "label": int(target["label"]),
                "label_name": str(target.get("label_name", key)),
            }
            continue
        raise KaggleAlzManifestError(
            "Label remap entries must be integers or objects containing at least a `label` field."
        )
    return normalized


def _apply_optional_label_remap(
    original_label: str | None,
    remap_config: dict[str, dict[str, Any]],
) -> tuple[int | None, str | None, bool]:
    """Apply explicit remapping only when provided."""

    if original_label is None:
        return None, None, False
    if not remap_config:
        return None, original_label, False
    remapped = remap_config.get(original_label)
    if remapped is None:
        return None, original_label, False
    return int(remapped["label"]), str(remapped["label_name"]), True


def _build_class_folder_rows(
    organization: KaggleDatasetOrganization,
    remap_config: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Build manifest rows for a class-folder Kaggle dataset."""

    rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    warnings = list(organization.notes)

    for path in _collect_imaging_files(organization.source_root):
        if not path.exists():
            dropped_rows.append({"image": str(path), "reason": "missing_image_path"})
            continue

        relative = path.relative_to(organization.source_root)
        original_label = relative.parts[-2] if len(relative.parts) >= 2 else None
        subset = relative.parts[-3] if len(relative.parts) >= 3 else None
        label, label_name, remap_applied = _apply_optional_label_remap(original_label, remap_config)

        rows.append(
            {
                "image": str(path),
                "label": label,
                "label_name": label_name,
                "subject_id": None,
                "scan_timestamp": None,
                "dataset": "kaggle_alz",
                "dataset_type": organization.dataset_type,
                "meta": {
                    "organization": organization.kind,
                    "subset": subset,
                    "original_class_name": original_label,
                    "original_label_value": original_label,
                    "label_mapping_applied": remap_applied,
                    "source_root": str(organization.source_root),
                },
            }
        )

    return rows, dropped_rows, warnings


def _build_metadata_table_rows(
    organization: KaggleDatasetOrganization,
    remap_config: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Build manifest rows for a metadata-table Kaggle dataset."""

    if organization.metadata_path is None or organization.image_path_column is None:
        raise KaggleAlzManifestError("Metadata-table organization is missing required metadata details.")

    frame = _load_tabular_file(organization.metadata_path)
    rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    warnings = list(organization.notes)

    for row_index, row in frame.iterrows():
        image_path = _resolve_existing_image_path(
            row[organization.image_path_column],
            source_root=organization.source_root,
            metadata_path=organization.metadata_path,
        )
        if image_path is None:
            dropped_rows.append(
                {
                    "row_index": int(row_index),
                    "reason": "missing_or_unresolvable_image_path",
                    "raw_image_value": None if pd.isna(row[organization.image_path_column]) else row[organization.image_path_column],
                }
            )
            continue

        original_label = None
        if organization.label_column is not None and not pd.isna(row[organization.label_column]):
            original_label = str(row[organization.label_column]).strip()

        label, label_name, remap_applied = _apply_optional_label_remap(original_label, remap_config)
        subject_id = None
        if organization.subject_id_column is not None and not pd.isna(row[organization.subject_id_column]):
            subject_id = str(row[organization.subject_id_column]).strip() or None
        scan_timestamp = None
        if organization.scan_timestamp_column is not None and not pd.isna(row[organization.scan_timestamp_column]):
            scan_timestamp = str(row[organization.scan_timestamp_column]).strip() or None

        rows.append(
            {
                "image": str(image_path),
                "label": label,
                "label_name": label_name if label_name is not None else original_label,
                "subject_id": subject_id,
                "scan_timestamp": scan_timestamp,
                "dataset": "kaggle_alz",
                "dataset_type": organization.dataset_type,
                "meta": {
                    "organization": organization.kind,
                    "metadata_path": str(organization.metadata_path),
                    "original_class_name": original_label,
                    "original_label_value": original_label,
                    "label_mapping_applied": remap_applied,
                    "row_index": int(row_index),
                    "source_root": str(organization.source_root),
                },
            }
        )

    return rows, dropped_rows, warnings


def _build_flat_file_rows(
    organization: KaggleDatasetOrganization,
    remap_config: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Build manifest rows for an unlabeled flat-file Kaggle dataset."""

    del remap_config
    rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    warnings = list(organization.notes)
    warnings.append("No class labels were detected. Label fields will remain unset.")

    for path in _collect_imaging_files(organization.source_root):
        rows.append(
            {
                "image": str(path),
                "label": None,
                "label_name": None,
                "subject_id": None,
                "scan_timestamp": None,
                "dataset": "kaggle_alz",
                "dataset_type": organization.dataset_type,
                "meta": {
                    "organization": organization.kind,
                    "original_class_name": None,
                    "original_label_value": None,
                    "label_mapping_applied": False,
                    "source_root": str(organization.source_root),
                },
            }
        )

    return rows, dropped_rows, warnings


def _serialize_manifest_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame representation suitable for CSV export."""

    return pd.DataFrame(
        [
            {
                **row,
                "meta": json.dumps(row["meta"], sort_keys=True),
            }
            for row in rows
        ]
    )


def _save_manifest_outputs(
    rows: list[dict[str, Any]],
    dropped_rows: list[dict[str, Any]],
    summary_payload: dict[str, Any],
    destination_root: Path,
    *,
    output_format: str,
) -> tuple[Path | None, Path | None, Path, Path]:
    """Save manifest, dropped rows, and summary artifacts."""

    ensure_directory(destination_root)
    manifest_csv_path: Path | None = None
    manifest_jsonl_path: Path | None = None
    dropped_rows_path = destination_root / "kaggle_alz_manifest_dropped_rows.csv"
    summary_path = destination_root / "kaggle_alz_manifest_summary.json"

    manifest_frame = _serialize_manifest_rows(rows)
    if output_format in {"csv", "both"}:
        manifest_csv_path = destination_root / "kaggle_alz_manifest.csv"
        manifest_frame.to_csv(manifest_csv_path, index=False)

    if output_format in {"jsonl", "both"}:
        manifest_jsonl_path = destination_root / "kaggle_alz_manifest.jsonl"
        with manifest_jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    pd.DataFrame(dropped_rows).to_csv(dropped_rows_path, index=False)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return manifest_csv_path, manifest_jsonl_path, dropped_rows_path, summary_path


def build_kaggle_manifest(
    settings: AppSettings | None = None,
    *,
    output_format: str = "csv",
    label_remap: dict[str, Any] | Path | None = None,
) -> KaggleManifestResult:
    """Detect, normalize, and save a Kaggle Alzheimer dataset manifest."""

    if output_format not in {"csv", "jsonl", "both"}:
        raise KaggleAlzManifestError(f"Unsupported output format: {output_format}")

    resolved_settings = settings or get_app_settings()
    spec = build_kaggle_dataset_spec(resolved_settings)
    organization = detect_kaggle_dataset_organization(spec.source_root)
    remap_config = _normalize_label_remap_config(label_remap)

    if organization.kind == "class_folders":
        rows, dropped_rows, warnings = _build_class_folder_rows(organization, remap_config)
    elif organization.kind == "metadata_table":
        rows, dropped_rows, warnings = _build_metadata_table_rows(organization, remap_config)
    else:
        rows, dropped_rows, warnings = _build_flat_file_rows(organization, remap_config)

    if organization.dataset_type == "2d_slices":
        slice_warning = (
            "Kaggle manifest is slice-based (`2d_slices`). Do not treat these rows as equivalent to subject-level 3D MRI volumes."
        )
        if slice_warning not in warnings:
            warnings.append(slice_warning)

    summary_payload = {
        "dataset": "kaggle_alz",
        "source_root": str(spec.source_root),
        "organization": organization.kind,
        "dataset_type": organization.dataset_type,
        "manifest_row_count": len(rows),
        "dropped_row_count": len(dropped_rows),
        "label_mapping_applied": bool(remap_config),
        "warnings": warnings,
        "notes": list(organization.notes),
        "metadata_path": str(organization.metadata_path) if organization.metadata_path else None,
    }

    manifest_csv_path, manifest_jsonl_path, dropped_rows_path, summary_path = _save_manifest_outputs(
        rows,
        dropped_rows,
        summary_payload,
        resolved_settings.data_root / "interim",
        output_format=output_format,
    )

    return KaggleManifestResult(
        organization=organization,
        manifest_csv_path=manifest_csv_path,
        manifest_jsonl_path=manifest_jsonl_path,
        dropped_rows_path=dropped_rows_path,
        summary_path=summary_path,
        manifest_row_count=len(rows),
        dropped_row_count=len(dropped_rows),
        warnings=warnings,
    )
