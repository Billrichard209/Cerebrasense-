"""Dataset-specific inspection entry points for OASIS and Kaggle."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

from .inspection_models import DatasetInspectionReport, ImagingFileRecord
from .inspection_utils import (
    DEFAULT_IGNORED_NAMES,
    IMAGE_EXTENSIONS,
    VOLUME_EXTENSIONS,
    build_dataframe,
    build_duplicate_risk_summary,
    build_file_format_table,
    collect_files,
    get_extension,
    infer_session_id_from_path,
    infer_subject_id_from_path,
    inspect_image_header,
    inspect_image_intensity,
    inspect_volume_header,
    inspect_volume_intensity,
    render_preview_image,
    sample_evenly,
    summarize_shape_distribution,
    summarize_spacing_distribution,
    try_open_image,
)
from .kaggle_dataset import build_kaggle_dataset_spec
from .metadata_parser import parse_kaggle_tabular_metadata, parse_oasis_tabular_metadata, parse_oasis_xml_metadata
from .oasis_dataset import build_oasis_dataset_spec


def _build_artifact_paths(settings: AppSettings, dataset_name: str) -> tuple[Path, Path]:
    """Resolve report and visualization roots for a dataset inspection run."""

    report_root = ensure_directory(settings.outputs_root / "reports" / dataset_name)
    visualization_root = ensure_directory(settings.outputs_root / "visualizations" / dataset_name)
    return report_root, visualization_root


def _save_table(rows: list[dict[str, Any]], output_path: Path, columns: list[str] | None = None) -> str:
    """Write a CSV summary table and return its path."""

    dataframe = build_dataframe(rows, columns=columns)
    dataframe.to_csv(output_path, index=False)
    return str(output_path)


def _save_report(report: DatasetInspectionReport, output_path: Path) -> str:
    """Serialize a report to JSON."""

    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return str(output_path)


def _save_visualizations(
    intensity_rows: list[dict[str, Any]],
    visualization_root: Path,
    dataset_name: str,
    *,
    is_volume_dataset: bool,
    max_visualizations: int,
) -> list[str]:
    """Save a small preview set for the inspected dataset."""

    saved_paths: list[str] = []
    for index, row in enumerate(intensity_rows[:max_visualizations], start=1):
        output_path = visualization_root / f"{dataset_name}_sample_{index:02d}.png"
        title = Path(row["relative_path"]).name
        render_preview_image(row["preview_array"], output_path, title, is_volume_slice=is_volume_dataset)
        saved_paths.append(str(output_path))
    return saved_paths


def _finalize_intensity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop non-serializable preview arrays before saving the report."""

    finalized: list[dict[str, Any]] = []
    for row in rows:
        clean_row = {key: value for key, value in row.items() if key != "preview_array"}
        if "shape" in clean_row and isinstance(clean_row["shape"], tuple):
            clean_row["shape"] = list(clean_row["shape"])
        finalized.append(clean_row)
    return finalized


def _select_oasis_volume_records(source_root: Path) -> tuple[list[ImagingFileRecord], list[str]]:
    """Select loadable OASIS volume records and flag missing Analyze pairs."""

    files = collect_files(source_root, DEFAULT_IGNORED_NAMES)
    by_path = {path: path for path in files}
    records: list[ImagingFileRecord] = []
    missing_files: list[str] = []

    for path in files:
        extension = get_extension(path)
        if extension not in VOLUME_EXTENSIONS:
            continue
        if extension == ".img":
            continue
        if extension == ".hdr":
            paired_img = path.with_suffix(".img")
            if paired_img not in by_path:
                missing_files.append(str(path.relative_to(source_root)))
                continue
            relative_path = str(path.relative_to(source_root))
            path_parts = path.relative_to(source_root).parts
            subset = path_parts[1] if len(path_parts) > 1 and path_parts[0].startswith("oasis_cross") else path_parts[0]
            records.append(
                ImagingFileRecord(
                    dataset_name="oasis",
                    load_path=path,
                    fingerprint_path=paired_img,
                    format_name="ANALYZE 7.5",
                    relative_path=relative_path,
                    subset=subset,
                    subject_id=infer_subject_id_from_path(path),
                    session_id=infer_session_id_from_path(path),
                )
            )
            continue
        if extension in {".nii", ".nii.gz"}:
            records.append(
                ImagingFileRecord(
                    dataset_name="oasis",
                    load_path=path,
                    fingerprint_path=path,
                    format_name="NIfTI",
                    relative_path=str(path.relative_to(source_root)),
                    subset=path.relative_to(source_root).parts[0],
                    subject_id=infer_subject_id_from_path(path),
                    session_id=infer_session_id_from_path(path),
                )
            )

    for path in files:
        if get_extension(path) == ".img" and path.with_suffix(".hdr") not in by_path:
            missing_files.append(str(path.relative_to(source_root)))

    return records, sorted(set(missing_files))


def _select_kaggle_image_records(source_root: Path) -> list[ImagingFileRecord]:
    """Select Kaggle imaging records from the original and augmented folders."""

    files = collect_files(source_root, DEFAULT_IGNORED_NAMES)
    records: list[ImagingFileRecord] = []
    for path in files:
        extension = get_extension(path)
        if extension not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(source_root)
        if len(relative.parts) < 3:
            continue
        subset, label = relative.parts[0], relative.parts[1]
        if subset not in {"OriginalDataset", "AugmentedAlzheimerDataset"}:
            continue
        records.append(
            ImagingFileRecord(
                dataset_name="kaggle",
                load_path=path,
                fingerprint_path=path,
                format_name=extension,
                relative_path=str(relative),
                subset=subset,
                label=label,
            )
        )
    return records


def inspect_oasis_dataset(
    settings: AppSettings | None = None,
    *,
    max_intensity_samples: int = 8,
    max_visualizations: int = 4,
) -> DatasetInspectionReport:
    """Inspect the OASIS dataset separately from Kaggle."""

    resolved_settings = settings or get_app_settings()
    spec = build_oasis_dataset_spec(resolved_settings)
    report_root, visualization_root = _build_artifact_paths(resolved_settings, "oasis")
    all_files = collect_files(spec.source_root, DEFAULT_IGNORED_NAMES)
    records, missing_files = _select_oasis_volume_records(spec.source_root)

    header_rows: list[dict[str, Any]] = []
    corrupt_files: list[dict[str, str]] = []

    intensity_rows: list[dict[str, Any]] = []
    for record in sample_evenly(records, max_intensity_samples):
        try:
            intensity_rows.append(inspect_volume_intensity(record))
        except Exception as error:  # pragma: no cover - defensive against local file variance
            corrupt_files.append({"path": record.relative_path, "error": str(error)})

    xml_files = [path for path in all_files if get_extension(path) == ".xml"]
    xml_summary = parse_oasis_xml_metadata(xml_files)
    spreadsheet_summary = parse_oasis_tabular_metadata(spec.source_root)
    xml_image_resources = xml_summary["image_resources"]
    header_rows = [
        {
            "relative_path": resource["uri"],
            "shape": tuple(resource["dimensions"]),
            "voxel_spacing": tuple(resource["voxel_spacing"]),
            "format_name": resource["format"],
        }
        for resource in xml_image_resources
        if resource["dimensions"]
    ]
    header_shapes = [row["shape"] for row in header_rows]
    header_spacings = [row["voxel_spacing"] for row in header_rows if row["voxel_spacing"]]
    duplicate_summary = build_duplicate_risk_summary(records, max_files=200)

    notes = [
        f"Detected {len(records)} loadable OASIS volumes from {len(all_files)} total files.",
        f"Shape and spacing distributions were derived from {len(xml_image_resources)} OASIS XML image-resource entries.",
        "OASIS appears suitable for 3D MONAI classification after choosing one consistent acquisition family.",
        "Raw and processed OASIS volumes were inspected separately in-place; no label harmonization was performed.",
    ]
    if spreadsheet_summary["spreadsheet_errors"]:
        notes.append("Some OASIS spreadsheet metadata files could not be parsed.")
    if xml_summary["xml_errors"]:
        notes.append("Some OASIS XML files could not be parsed.")

    visualization_paths = _save_visualizations(
        intensity_rows,
        visualization_root,
        "oasis",
        is_volume_dataset=True,
        max_visualizations=max_visualizations,
    )

    report = DatasetInspectionReport(
        dataset_name="oasis",
        source_root=str(spec.source_root),
        inspected_at=datetime.now(UTC).isoformat(),
        total_file_count=len(all_files),
        primary_image_count=len(records),
        file_formats={row["file_format"]: row["count"] for row in build_file_format_table(all_files)},
        data_type="3d_volumes",
        is_3d_volume_dataset=True,
        monai_3d_suitability="suitable",
        monai_3d_reason="OASIS contains true MRI volumes with 3D spatial structure and voxel spacing metadata.",
        shape_distribution=summarize_shape_distribution(header_shapes),
        voxel_spacing_distribution=summarize_spacing_distribution(header_spacings),
        intensity_statistics=_finalize_intensity_rows(intensity_rows),
        label_distribution=spreadsheet_summary["label_distribution"],
        label_source=spreadsheet_summary["label_source"],
        subject_id_summary={
            "available": True,
            "count": len(set(spreadsheet_summary["subject_ids"]) | set(xml_summary["subject_ids"])),
            "sample": sorted(set(spreadsheet_summary["subject_ids"]) | set(xml_summary["subject_ids"]))[:10],
        },
        scan_timestamp_summary={
            "available": bool(spreadsheet_summary["timestamp_examples"] or xml_summary["timestamp_examples"]),
            "count": len(spreadsheet_summary["timestamp_examples"]) + len(xml_summary["timestamp_examples"]),
            "sample": (spreadsheet_summary["timestamp_examples"] + xml_summary["timestamp_examples"])[:10],
        },
        missing_files=missing_files,
        corrupt_files=corrupt_files + xml_summary["xml_errors"] + spreadsheet_summary["spreadsheet_errors"],
        duplicate_risk_summary=duplicate_summary,
        metadata_summary={"spreadsheet": spreadsheet_summary, "xml": xml_summary},
        notes=notes,
    )

    report.artifacts = {
        "report_json": _save_report(report, report_root / "inspection_report.json"),
        "file_formats_csv": _save_table(
            build_file_format_table(all_files),
            report_root / "file_formats.csv",
            columns=["file_format", "count"],
        ),
        "shape_distribution_csv": _save_table(
            report.shape_distribution,
            report_root / "shape_distribution.csv",
            columns=["shape", "count"],
        ),
        "voxel_spacing_distribution_csv": _save_table(
            report.voxel_spacing_distribution,
            report_root / "voxel_spacing_distribution.csv",
            columns=["voxel_spacing", "count"],
        ),
        "label_distribution_csv": _save_table(
            [{"label": label, "count": count} for label, count in sorted(report.label_distribution.items())],
            report_root / "label_distribution.csv",
            columns=["label", "count"],
        ),
        "intensity_statistics_csv": _save_table(
            report.intensity_statistics,
            report_root / "intensity_statistics.csv",
        ),
        "visualizations": visualization_paths,
    }
    _save_report(report, report_root / "inspection_report.json")
    return report


def inspect_kaggle_dataset(
    settings: AppSettings | None = None,
    *,
    max_intensity_samples: int = 8,
    max_visualizations: int = 4,
) -> DatasetInspectionReport:
    """Inspect the Kaggle dataset separately from OASIS."""

    resolved_settings = settings or get_app_settings()
    spec = build_kaggle_dataset_spec(resolved_settings)
    report_root, visualization_root = _build_artifact_paths(resolved_settings, "kaggle")
    all_files = collect_files(spec.source_root, DEFAULT_IGNORED_NAMES)
    records = _select_kaggle_image_records(spec.source_root)
    header_inspection_records = sample_evenly(records, 2500)

    header_rows: list[dict[str, Any]] = []
    corrupt_files: list[dict[str, str]] = []
    for record in header_inspection_records:
        try:
            header_rows.append(inspect_image_header(record))
        except Exception as error:  # pragma: no cover - defensive against local image variance
            corrupt_files.append({"path": record.relative_path, "error": str(error)})

    intensity_rows: list[dict[str, Any]] = []
    for record in sample_evenly(records, max_intensity_samples):
        try:
            intensity_rows.append(inspect_image_intensity(record))
        except Exception as error:  # pragma: no cover - defensive against local image variance
            corrupt_files.append({"path": record.relative_path, "error": str(error)})

    label_distribution = dict(sorted(Counter(record.label or "unknown" for record in records).items()))
    subset_distribution = dict(sorted(Counter(record.subset or "unknown" for record in records).items()))
    duplicate_summary = build_duplicate_risk_summary(records, max_files=1500)
    metadata_summary = parse_kaggle_tabular_metadata(spec.source_root)

    visualization_paths = _save_visualizations(
        intensity_rows,
        visualization_root,
        "kaggle",
        is_volume_dataset=False,
        max_visualizations=max_visualizations,
    )

    notes = [
        f"Detected {len(records)} loadable Kaggle images from {len(all_files)} total files.",
        f"Shape distribution was estimated from {len(header_inspection_records)} sampled Kaggle images for practicality.",
        "Kaggle appears to be a 2D image dataset and is not directly suitable for native 3D MONAI classification.",
        "Class labels were inferred from directory names without any automatic remapping or harmonization.",
    ]
    if metadata_summary["metadata_files"]:
        notes.append("Additional Kaggle metadata files were discovered and summarized when parseable.")
    if duplicate_summary["cross_label_duplicate_group_count"] > 0:
        notes.append("Exact duplicate images were found across different Kaggle labels, which is a leakage risk.")

    report = DatasetInspectionReport(
        dataset_name="kaggle",
        source_root=str(spec.source_root),
        inspected_at=datetime.now(UTC).isoformat(),
        total_file_count=len(all_files),
        primary_image_count=len(records),
        file_formats={row["file_format"]: row["count"] for row in build_file_format_table(all_files)},
        data_type="2d_images",
        is_3d_volume_dataset=False,
        monai_3d_suitability="not_suitable",
        monai_3d_reason="Kaggle contains 2D class-folder images rather than consistent 3D MRI volumes.",
        shape_distribution=summarize_shape_distribution([row["shape"] for row in header_rows]),
        voxel_spacing_distribution=[],
        intensity_statistics=_finalize_intensity_rows(intensity_rows),
        label_distribution=label_distribution,
        label_source="directory_names",
        subject_id_summary={
            "available": bool(metadata_summary["subject_ids"]),
            "count": len(metadata_summary["subject_ids"]),
            "sample": metadata_summary["subject_ids"][:10],
        },
        scan_timestamp_summary={
            "available": bool(metadata_summary["timestamp_examples"]),
            "count": len(metadata_summary["timestamp_examples"]),
            "sample": metadata_summary["timestamp_examples"][:10],
        },
        missing_files=[],
        corrupt_files=corrupt_files + metadata_summary["spreadsheet_errors"],
        duplicate_risk_summary=duplicate_summary | {"subset_distribution": subset_distribution},
        metadata_summary=metadata_summary | {"subset_distribution": subset_distribution},
        notes=notes,
    )

    report.artifacts = {
        "report_json": _save_report(report, report_root / "inspection_report.json"),
        "file_formats_csv": _save_table(
            build_file_format_table(all_files),
            report_root / "file_formats.csv",
            columns=["file_format", "count"],
        ),
        "shape_distribution_csv": _save_table(
            report.shape_distribution,
            report_root / "shape_distribution.csv",
            columns=["shape", "count"],
        ),
        "label_distribution_csv": _save_table(
            [{"label": label, "count": count} for label, count in sorted(report.label_distribution.items())],
            report_root / "label_distribution.csv",
            columns=["label", "count"],
        ),
        "intensity_statistics_csv": _save_table(
            report.intensity_statistics,
            report_root / "intensity_statistics.csv",
        ),
        "visualizations": visualization_paths,
    }
    _save_report(report, report_root / "inspection_report.json")
    return report


def inspect_datasets(
    dataset_name: str = "all",
    settings: AppSettings | None = None,
    *,
    max_intensity_samples: int = 8,
    max_visualizations: int = 4,
) -> dict[str, DatasetInspectionReport]:
    """Inspect one or both datasets and return report objects."""

    resolved_settings = settings or get_app_settings()
    reports: dict[str, DatasetInspectionReport] = {}
    if dataset_name in {"all", "oasis"}:
        reports["oasis"] = inspect_oasis_dataset(
            resolved_settings,
            max_intensity_samples=max_intensity_samples,
            max_visualizations=max_visualizations,
        )
    if dataset_name in {"all", "kaggle"}:
        reports["kaggle"] = inspect_kaggle_dataset(
            resolved_settings,
            max_intensity_samples=max_intensity_samples,
            max_visualizations=max_visualizations,
        )
    return reports
