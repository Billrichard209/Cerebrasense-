"""Helpers for building external-cohort manifests from 3D MRI folders and optional metadata CSV files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .base_dataset import canonicalize_optional_string
from .external_cohort import RESERVED_INTERNAL_DATASET_NAMES

SUPPORTED_3D_IMAGE_GLOBS = ("*.nii.gz", "*.nii", "*.nrrd", "*.mha", "*.mhd")
DEFAULT_METADATA_IMAGE_COLUMNS = (
    "image",
    "image_path",
    "filepath",
    "file_path",
    "path",
    "filename",
    "file_name",
    "scan_path",
    "scan_file",
)


class ExternalManifestBuildError(ValueError):
    """Raised when an external-cohort manifest cannot be built safely."""


@dataclass(slots=True, frozen=True)
class ExternalManifestBuilderConfig:
    """Configuration for building an external-cohort manifest."""

    images_root: Path
    dataset_name: str
    output_path: Path
    dataset_type: str = "3d_volumes"
    metadata_csv_path: Path | None = None
    image_column: str | None = None
    label_column: str | None = None
    label_name_column: str | None = None
    subject_id_column: str | None = None
    session_id_column: str | None = None
    scan_timestamp_column: str | None = None
    meta_columns: tuple[str, ...] = ()
    image_globs: tuple[str, ...] = SUPPORTED_3D_IMAGE_GLOBS
    recursive: bool = True
    require_labels: bool = False


@dataclass(slots=True)
class ExternalManifestBuildResult:
    """Saved manifest plus a compact build report."""

    manifest_path: Path
    report_path: Path
    row_count: int
    discovered_image_count: int
    matched_image_count: int
    unmatched_image_count: int
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "report_path": str(self.report_path),
            "row_count": self.row_count,
            "discovered_image_count": self.discovered_image_count,
            "matched_image_count": self.matched_image_count,
            "unmatched_image_count": self.unmatched_image_count,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class _DiscoveredImageLookup:
    """Index of discovered cohort image files for robust metadata matching."""

    images: list[Path]
    by_absolute: dict[str, Path]
    by_relative: dict[str, Path]
    by_name: dict[str, list[Path]]
    by_stem: dict[str, list[Path]]


def _normalize_dataset_name(dataset_name: str) -> str:
    """Validate and normalize an external dataset name."""

    normalized = canonicalize_optional_string(dataset_name)
    if normalized is None:
        raise ExternalManifestBuildError("External manifest builder requires a non-empty dataset name.")
    if normalized.lower() in RESERVED_INTERNAL_DATASET_NAMES:
        raise ExternalManifestBuildError(
            f"External dataset name {normalized!r} collides with an internal dataset path. "
            "Use a true outside cohort name such as 'adni_pilot' or 'aibl_site_a'."
        )
    return normalized


def _normalize_text_path(value: str | Path) -> str:
    """Normalize a path-like value for case-insensitive comparisons."""

    return str(value).replace("\\", "/").strip().lower()


def _strip_all_suffixes(path: Path) -> str:
    """Return the lowercase filename without any suffixes."""

    name = path.name
    for suffix in path.suffixes:
        name = name[: -len(suffix)]
    return name.lower()


def _add_multi_value(mapping: dict[str, list[Path]], key: str, image_path: Path) -> None:
    """Append one discovered image to a multi-value lookup."""

    mapping.setdefault(key, []).append(image_path)


def discover_external_3d_images(
    images_root: Path,
    *,
    image_globs: tuple[str, ...] = SUPPORTED_3D_IMAGE_GLOBS,
    recursive: bool = True,
) -> _DiscoveredImageLookup:
    """Discover supported 3D image files and build matching indexes."""

    if not images_root.exists():
        raise FileNotFoundError(f"External images root not found: {images_root}")
    if not images_root.is_dir():
        raise ExternalManifestBuildError(f"External images root is not a directory: {images_root}")

    discovered: dict[str, Path] = {}
    for pattern in image_globs:
        candidates = images_root.rglob(pattern) if recursive else images_root.glob(pattern)
        for candidate in candidates:
            if candidate.is_file():
                discovered[_normalize_text_path(candidate.resolve())] = candidate.resolve()

    images = sorted(discovered.values())
    if not images:
        raise ExternalManifestBuildError(
            f"No supported 3D image files were found under {images_root} using patterns {list(image_globs)}."
        )

    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for image_path in images:
        relative_key = _normalize_text_path(image_path.relative_to(images_root))
        by_relative[relative_key] = image_path
        _add_multi_value(by_name, image_path.name.lower(), image_path)
        _add_multi_value(by_stem, _strip_all_suffixes(image_path), image_path)

    return _DiscoveredImageLookup(
        images=images,
        by_absolute={_normalize_text_path(path): path for path in images},
        by_relative=by_relative,
        by_name=by_name,
        by_stem=by_stem,
    )


def _infer_metadata_image_column(frame: pd.DataFrame) -> str:
    """Choose a likely metadata image column when the user does not specify one."""

    for column_name in DEFAULT_METADATA_IMAGE_COLUMNS:
        if column_name in frame.columns:
            return column_name
    raise ExternalManifestBuildError(
        "Could not infer which metadata column points to image files. "
        f"Expected one of {list(DEFAULT_METADATA_IMAGE_COLUMNS)} or pass --image-column explicitly."
    )


def _resolve_metadata_image_path(
    raw_value: Any,
    *,
    images_root: Path,
    metadata_root: Path | None,
    lookup: _DiscoveredImageLookup,
) -> Path:
    """Resolve one metadata image reference against discovered image files."""

    text_value = canonicalize_optional_string(raw_value)
    if text_value is None:
        raise ExternalManifestBuildError("Metadata contains an empty image reference.")

    path_value = Path(text_value)
    candidate_paths: list[Path] = []
    if path_value.is_absolute():
        matched = lookup.by_absolute.get(_normalize_text_path(path_value))
        if matched is not None:
            candidate_paths.append(matched)
    else:
        direct_relative = lookup.by_relative.get(_normalize_text_path(path_value))
        if direct_relative is not None:
            candidate_paths.append(direct_relative)
        rooted_candidate = lookup.by_absolute.get(_normalize_text_path((images_root / path_value).resolve()))
        if rooted_candidate is not None:
            candidate_paths.append(rooted_candidate)
        if metadata_root is not None:
            metadata_candidate = metadata_root / path_value
            if metadata_candidate.exists():
                matched = lookup.by_absolute.get(_normalize_text_path(metadata_candidate.resolve()))
                if matched is not None:
                    candidate_paths.append(matched)

    file_name_matches = lookup.by_name.get(path_value.name.lower(), [])
    candidate_paths.extend(file_name_matches)
    candidate_paths.extend(lookup.by_stem.get(_strip_all_suffixes(path_value), []))

    unique_candidates = list(dict.fromkeys(candidate_paths))
    if not unique_candidates:
        raise ExternalManifestBuildError(
            f"Could not match metadata image reference {text_value!r} to a discovered 3D image file."
        )
    if len(unique_candidates) > 1:
        raise ExternalManifestBuildError(
            f"Metadata image reference {text_value!r} is ambiguous. "
            "Use an explicit path-style column so one row maps to one discovered file."
        )
    return unique_candidates[0]


def _build_meta_payload(row: pd.Series, *, meta_columns: tuple[str, ...]) -> dict[str, Any]:
    """Extract selected metadata columns into the manifest meta payload."""

    payload: dict[str, Any] = {}
    for column_name in meta_columns:
        if column_name not in row.index:
            raise ExternalManifestBuildError(
                f"Requested meta column {column_name!r} is not present in the metadata CSV."
            )
        value = row[column_name]
        if value is None or pd.isna(value):
            continue
        payload[column_name] = value.item() if hasattr(value, "item") else value
    return payload


def _extract_optional_row_value(row: pd.Series, column_name: str | None) -> Any:
    """Return one optional column value from a metadata row."""

    if column_name is None:
        return None
    if column_name not in row.index:
        raise ExternalManifestBuildError(f"Metadata column {column_name!r} is not present in the CSV.")
    return row[column_name]


def _build_rows_from_metadata(
    cfg: ExternalManifestBuilderConfig,
    *,
    lookup: _DiscoveredImageLookup,
) -> tuple[list[dict[str, Any]], int]:
    """Build manifest rows from a metadata CSV matched to discovered images."""

    if cfg.metadata_csv_path is None:
        raise ExternalManifestBuildError("Metadata CSV path is required for metadata-backed manifest building.")
    if not cfg.metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {cfg.metadata_csv_path}")

    frame = pd.read_csv(cfg.metadata_csv_path)
    if frame.empty:
        raise ExternalManifestBuildError(f"Metadata CSV is empty: {cfg.metadata_csv_path}")
    image_column = cfg.image_column or _infer_metadata_image_column(frame)
    if image_column not in frame.columns:
        raise ExternalManifestBuildError(f"Metadata image column {image_column!r} is not present in the CSV.")
    if cfg.require_labels and cfg.label_column is None:
        raise ExternalManifestBuildError(
            "Labels are required for evaluation-ready external manifests. Pass --label-column."
        )

    rows: list[dict[str, Any]] = []
    matched_images: set[str] = set()
    metadata_root = cfg.metadata_csv_path.parent
    for _, row in frame.iterrows():
        image_path = _resolve_metadata_image_path(
            row[image_column],
            images_root=cfg.images_root,
            metadata_root=metadata_root,
            lookup=lookup,
        )
        image_key = _normalize_text_path(image_path)
        if image_key in matched_images:
            raise ExternalManifestBuildError(
                f"Image {image_path} was matched more than once from the metadata CSV. "
                "Keep one manifest row per image for evaluation traceability."
            )
        matched_images.add(image_key)

        label_value = _extract_optional_row_value(row, cfg.label_column)
        if cfg.require_labels and (label_value is None or pd.isna(label_value)):
            raise ExternalManifestBuildError(
                f"Metadata row for image {image_path.name!r} is missing a required label value."
            )

        rows.append(
            {
                "image": str(image_path),
                "label": None if label_value is None or pd.isna(label_value) else label_value,
                "label_name": _extract_optional_row_value(row, cfg.label_name_column),
                "subject_id": _extract_optional_row_value(row, cfg.subject_id_column),
                "session_id": _extract_optional_row_value(row, cfg.session_id_column),
                "scan_timestamp": _extract_optional_row_value(row, cfg.scan_timestamp_column),
                "dataset": cfg.dataset_name,
                "dataset_type": cfg.dataset_type,
                "meta": json.dumps(_build_meta_payload(row, meta_columns=cfg.meta_columns)) if cfg.meta_columns else "",
            }
        )
    return rows, len(matched_images)


def _build_rows_without_metadata(
    cfg: ExternalManifestBuilderConfig,
    *,
    lookup: _DiscoveredImageLookup,
) -> tuple[list[dict[str, Any]], int]:
    """Build an unlabeled manifest directly from discovered image paths."""

    if cfg.require_labels:
        raise ExternalManifestBuildError(
            "Cannot require labels when no metadata CSV is provided. Add metadata or disable required labels."
        )
    rows = [
        {
            "image": str(image_path),
            "label": None,
            "label_name": None,
            "subject_id": None,
            "session_id": None,
            "scan_timestamp": None,
            "dataset": cfg.dataset_name,
            "dataset_type": cfg.dataset_type,
            "meta": "",
        }
        for image_path in lookup.images
    ]
    return rows, len(rows)


def build_external_cohort_manifest(cfg: ExternalManifestBuilderConfig) -> ExternalManifestBuildResult:
    """Build and save an external-cohort manifest with a reproducible build report."""

    normalized_dataset_name = _normalize_dataset_name(cfg.dataset_name)
    normalized_cfg = ExternalManifestBuilderConfig(
        images_root=cfg.images_root,
        dataset_name=normalized_dataset_name,
        output_path=cfg.output_path,
        dataset_type=cfg.dataset_type,
        metadata_csv_path=cfg.metadata_csv_path,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        label_name_column=cfg.label_name_column,
        subject_id_column=cfg.subject_id_column,
        session_id_column=cfg.session_id_column,
        scan_timestamp_column=cfg.scan_timestamp_column,
        meta_columns=cfg.meta_columns,
        image_globs=cfg.image_globs,
        recursive=cfg.recursive,
        require_labels=cfg.require_labels,
    )

    lookup = discover_external_3d_images(
        normalized_cfg.images_root,
        image_globs=normalized_cfg.image_globs,
        recursive=normalized_cfg.recursive,
    )
    if normalized_cfg.metadata_csv_path is None:
        rows, matched_count = _build_rows_without_metadata(normalized_cfg, lookup=lookup)
    else:
        rows, matched_count = _build_rows_from_metadata(normalized_cfg, lookup=lookup)

    frame = pd.DataFrame(rows)
    ordered_columns = [
        "image",
        "label",
        "label_name",
        "subject_id",
        "session_id",
        "scan_timestamp",
        "dataset",
        "dataset_type",
        "meta",
    ]
    frame = frame[ordered_columns]
    normalized_cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(normalized_cfg.output_path, index=False)

    warnings: list[str] = []
    if normalized_cfg.metadata_csv_path is None:
        warnings.append(
            "Manifest was built without metadata labels, so it is suitable for cohort onboarding but not labeled evaluation yet."
        )
    if matched_count < len(lookup.images):
        warnings.append(
            "Some discovered images were not included because the metadata CSV did not match every file."
        )
    report_path = normalized_cfg.output_path.with_suffix(".build_report.json")
    result = ExternalManifestBuildResult(
        manifest_path=normalized_cfg.output_path,
        report_path=report_path,
        row_count=int(len(frame)),
        discovered_image_count=len(lookup.images),
        matched_image_count=matched_count,
        unmatched_image_count=len(lookup.images) - matched_count,
        warnings=warnings,
        notes=[
            "This manifest builder does not harmonize labels automatically.",
            "For the current OASIS external evaluation path, dataset_type should remain 3d_volumes.",
        ],
    )
    report_path.write_text(
        json.dumps(
            {
                "config": {
                    **asdict(normalized_cfg),
                    "images_root": str(normalized_cfg.images_root),
                    "output_path": str(normalized_cfg.output_path),
                    "metadata_csv_path": (
                        str(normalized_cfg.metadata_csv_path) if normalized_cfg.metadata_csv_path else None
                    ),
                },
                "result": result.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
