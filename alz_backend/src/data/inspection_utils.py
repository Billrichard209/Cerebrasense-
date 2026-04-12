"""Shared helpers for dataset inspection, profiling, and visualization."""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .inspection_models import ImagingFileRecord

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULT_IGNORED_NAMES = {".git", ".venv", ".sixth", "alz_backend", "__pycache__"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VOLUME_EXTENSIONS = {".nii", ".nii.gz", ".hdr", ".img", ".dcm", ".ima"}
SUBJECT_ID_PATTERN = re.compile(r"(OAS\d?_[0-9]{4})", re.IGNORECASE)
SESSION_ID_PATTERN = re.compile(r"(OAS\d?_[0-9]{4}_MR\d+)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}")


def get_extension(path: Path) -> str:
    """Return a normalized file extension, preserving `.nii.gz`."""

    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower() or "<no_extension>"


def collect_files(root: Path, ignored_names: set[str] | None = None) -> list[Path]:
    """Recursively collect visible files beneath a dataset root."""

    ignored = ignored_names or DEFAULT_IGNORED_NAMES
    if not root.exists():
        return []

    collected: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in ignored and not dirname.startswith(".")
        ]
        current_dir = Path(dirpath)
        for filename in filenames:
            if filename in ignored or filename.startswith("."):
                continue
            collected.append(current_dir / filename)
    return sorted(collected)


def summarize_counter(counter: Counter[str], key_name: str, value_name: str) -> list[dict[str, Any]]:
    """Convert a counter to a sorted table-friendly list of dictionaries."""

    return [
        {key_name: key, value_name: int(value)}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def summarize_shape_distribution(shapes: Iterable[tuple[int, ...]], column_name: str = "shape") -> list[dict[str, Any]]:
    """Summarize a list of shapes into a distribution."""

    counter = Counter("x".join(str(part) for part in shape) for shape in shapes if shape)
    return summarize_counter(counter, column_name, "count")


def summarize_spacing_distribution(spacings: Iterable[tuple[float, ...]]) -> list[dict[str, Any]]:
    """Summarize voxel spacings into a distribution."""

    counter = Counter(
        "x".join(f"{part:.4f}" for part in spacing)
        for spacing in spacings
        if spacing
    )
    return summarize_counter(counter, "voxel_spacing", "count")


def sample_evenly(items: list[Any], limit: int) -> list[Any]:
    """Return an evenly spaced subset while preserving deterministic ordering."""

    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / (limit - 1)
    indices = sorted({round(index * step) for index in range(limit)})
    return [items[index] for index in indices]


def normalize_volume_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Drop a trailing singleton channel dimension when present."""

    normalized = tuple(int(part) for part in shape)
    if len(normalized) == 4 and normalized[-1] == 1:
        return normalized[:-1]
    return normalized


def infer_subject_id_from_path(path: Path) -> str | None:
    """Extract a plausible subject identifier from a file path."""

    match = SUBJECT_ID_PATTERN.search(path.as_posix())
    return match.group(1).upper() if match else None


def infer_session_id_from_path(path: Path) -> str | None:
    """Extract a plausible session identifier from a file path."""

    match = SESSION_ID_PATTERN.search(path.as_posix())
    return match.group(1).upper() if match else None


def compute_file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a stable SHA1 fingerprint for duplicate detection."""

    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_volume_header(record: ImagingFileRecord) -> dict[str, Any]:
    """Read volume shape and spacing without forcing a full data load."""

    image = nib.load(str(record.load_path))
    shape = normalize_volume_shape(tuple(int(part) for part in image.shape))
    zooms = tuple(float(part) for part in image.header.get_zooms()[: len(shape)])
    return {
        "relative_path": record.relative_path,
        "shape": shape,
        "voxel_spacing": zooms,
        "format_name": record.format_name,
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "subset": record.subset,
    }


def inspect_volume_intensity(record: ImagingFileRecord) -> dict[str, Any]:
    """Load one volume and compute intensity statistics."""

    image = nib.load(str(record.load_path))
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    data = np.squeeze(data)
    middle_slice = np.take(data, indices=data.shape[-1] // 2, axis=-1) if data.ndim >= 3 else data
    return {
        "relative_path": record.relative_path,
        "shape": tuple(int(part) for part in data.shape),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "nonzero_fraction": float(np.count_nonzero(data) / data.size),
        "preview_array": middle_slice.astype(np.float32),
    }


def inspect_image_header(record: ImagingFileRecord) -> dict[str, Any]:
    """Read image shape and mode without forcing a full decode into memory."""

    with Image.open(record.load_path) as image:
        width, height = image.size
        channels = len(image.getbands())
        shape = (height, width) if channels == 1 else (height, width, channels)
        return {
            "relative_path": record.relative_path,
            "shape": shape,
            "mode": image.mode,
            "format_name": image.format or record.format_name,
            "label": record.label,
            "subset": record.subset,
        }


def inspect_image_intensity(record: ImagingFileRecord) -> dict[str, Any]:
    """Load one image and compute intensity statistics."""

    with Image.open(record.load_path) as image:
        array = np.asarray(image, dtype=np.float32)
    preview = array if array.ndim == 2 else array[..., :3]
    return {
        "relative_path": record.relative_path,
        "shape": tuple(int(part) for part in array.shape),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "nonzero_fraction": float(np.count_nonzero(array) / array.size),
        "preview_array": preview,
    }


def render_preview_image(
    preview_array: np.ndarray,
    output_path: Path,
    title: str,
    is_volume_slice: bool,
) -> None:
    """Save a preview visualization for a sample volume slice or 2D image."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 4))
    if preview_array.ndim == 2:
        axis.imshow(preview_array, cmap="gray")
    else:
        axis.imshow(preview_array.astype(np.uint8) if not is_volume_slice else preview_array)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_duplicate_risk_summary(
    records: list[ImagingFileRecord],
    *,
    max_files: int | None = None,
    max_groups: int = 25,
) -> dict[str, Any]:
    """Detect exact duplicate files and flag cross-label leakage risk."""

    grouped_paths: defaultdict[str, list[ImagingFileRecord]] = defaultdict(list)
    hash_errors: list[dict[str, str]] = []
    inspected_records = records if max_files is None else sample_evenly(records, max_files)

    for record in inspected_records:
        try:
            digest = compute_file_sha1(record.fingerprint_path)
        except OSError as error:
            hash_errors.append({"path": record.relative_path, "error": str(error)})
            continue
        grouped_paths[digest].append(record)

    duplicate_groups: list[dict[str, Any]] = []
    cross_label_groups = 0
    for digest, group in grouped_paths.items():
        if len(group) < 2:
            continue
        labels = sorted({record.label for record in group if record.label})
        subsets = sorted({record.subset for record in group if record.subset})
        if len(labels) > 1:
            cross_label_groups += 1
        duplicate_groups.append(
            {
                "sha1": digest,
                "count": len(group),
                "labels": labels,
                "subsets": subsets,
                "paths": [record.relative_path for record in group[:10]],
                "cross_label_leakage_risk": len(labels) > 1,
            }
        )

    duplicate_groups.sort(key=lambda item: (-item["count"], item["sha1"]))
    return {
        "duplicate_check_scope": "full" if len(inspected_records) == len(records) else "sampled",
        "duplicate_files_checked": len(inspected_records),
        "duplicate_files_available": len(records),
        "duplicate_group_count": len(duplicate_groups),
        "cross_label_duplicate_group_count": cross_label_groups,
        "duplicate_groups_preview": duplicate_groups[:max_groups],
        "hash_errors": hash_errors,
    }


def build_file_format_table(files: list[Path]) -> list[dict[str, Any]]:
    """Build a file format distribution across all discovered files."""

    return summarize_counter(Counter(get_extension(path) for path in files), "file_format", "count")


def build_dataframe(rows: list[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    """Create a DataFrame even when there are no rows."""

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=columns or [])


def find_date_like_strings(values: Iterable[Any], limit: int = 10) -> list[str]:
    """Extract a few date-like strings from arbitrary values."""

    results: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if DATE_PATTERN.search(text):
            results.append(text)
        if len(results) >= limit:
            break
    return results


def try_open_image(path: Path) -> str | None:
    """Validate that an image can be opened by Pillow."""

    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except (OSError, UnidentifiedImageError) as error:
        return str(error)
