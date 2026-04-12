"""Helpers to build a portable Kaggle upload bundle for Drive or Colab."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

from .base_dataset import parse_manifest_meta


class KaggleUploadBundleError(ValueError):
    """Raised when a Kaggle upload bundle cannot be built safely."""


@dataclass(slots=True)
class KaggleUploadBundleResult:
    """Artifacts and counts produced when building a Kaggle upload bundle."""

    bundle_root: Path
    relative_manifest_path: Path
    summary_path: Path
    file_index_path: Path
    included_subset_names: tuple[str, ...]
    materialized_file_count: int
    missing_reference_count: int
    materialize_mode: str


def _resolve_source_path(path_value: str | Path | None, source_root: Path) -> Path | None:
    """Resolve a source path that may be relative or absolute."""

    if path_value in {None, ""}:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return source_root / candidate


def _relative_to_root(path: Path, source_root: Path) -> Path:
    """Return a path relative to the Kaggle source root or fail loudly."""

    try:
        return path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise KaggleUploadBundleError(
            f"Expected {path} to be inside the Kaggle source root {source_root}"
        ) from error


def _materialize_file(source_path: Path, destination_path: Path, *, mode: str) -> None:
    """Create one destination file by copy or hardlink, with safe reruns."""

    ensure_directory(destination_path.parent)
    if destination_path.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source_path, destination_path)
            return
        except OSError:
            shutil.copy2(source_path, destination_path)
            return
    if mode == "copy":
        shutil.copy2(source_path, destination_path)
        return
    raise KaggleUploadBundleError(f"Unsupported materialize mode: {mode}")


def _collect_subset_names(manifest_frame: pd.DataFrame, source_root: Path) -> tuple[str, ...]:
    """Infer the top-level Kaggle subset folders that should be bundled."""

    subset_names: set[str] = set()
    for row in manifest_frame.to_dict(orient="records"):
        meta = parse_manifest_meta(row.get("meta"))
        subset_name = str(meta.get("subset", "")).strip()
        if subset_name:
            subset_names.add(subset_name)
            continue

        image_path = _resolve_source_path(row.get("image"), source_root)
        if image_path is None or not image_path.exists():
            continue
        relative = _relative_to_root(image_path, source_root)
        if relative.parts:
            subset_names.add(relative.parts[0])

    resolved_names = tuple(sorted(name for name in subset_names if (source_root / name).is_dir()))
    if not resolved_names:
        raise KaggleUploadBundleError(
            "Could not determine which Kaggle top-level folders to bundle from the current manifest."
        )
    return resolved_names


def _collect_subset_files(source_root: Path, subset_names: tuple[str, ...]) -> list[Path]:
    """Collect all files beneath the selected top-level Kaggle subset roots."""

    files: list[Path] = []
    for subset_name in subset_names:
        subset_root = source_root / subset_name
        if not subset_root.exists():
            raise KaggleUploadBundleError(f"Kaggle subset root not found: {subset_root}")
        files.extend(sorted(path for path in subset_root.rglob("*") if path.is_file()))
    return files


def _build_relative_manifest(
    manifest_frame: pd.DataFrame,
    *,
    source_root: Path,
    path_lookup: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Rewrite manifest image paths to be relative to the Kaggle bundle root."""

    portable_rows: list[dict[str, Any]] = []
    missing_references: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []

    for row in manifest_frame.to_dict(orient="records"):
        raw_image = None if row.get("image") is None else str(row.get("image"))
        relative_lookup = None if path_lookup is None or raw_image is None else path_lookup.get(raw_image)

        if relative_lookup is not None:
            relative_path = Path(relative_lookup)
        else:
            image_path = _resolve_source_path(row.get("image"), source_root)
            if image_path is None or not image_path.exists():
                missing_references.append(
                    {
                        "image": row.get("image"),
                        "reason": "missing_image_reference",
                        "subset": parse_manifest_meta(row.get("meta")).get("subset"),
                    }
                )
                continue
            relative_path = _relative_to_root(image_path, source_root)

        if not relative_path.parts:
            missing_references.append(
                {
                    "image": row.get("image"),
                    "reason": "missing_image_reference",
                    "subset": parse_manifest_meta(row.get("meta")).get("subset"),
                }
            )
            continue
        meta = parse_manifest_meta(row.get("meta"))
        portable_meta = dict(meta)
        portable_meta["source_root"] = "."
        portable_row = dict(row)
        portable_row["image"] = relative_path.as_posix()
        portable_row["meta"] = json.dumps(portable_meta, ensure_ascii=True, sort_keys=True)
        portable_rows.append(portable_row)
        index_rows.append(
            {
                "subset": relative_path.parts[0] if relative_path.parts else None,
                "class_name": relative_path.parts[-2] if len(relative_path.parts) >= 2 else None,
                "relative_path": relative_path.as_posix(),
                "label_name": row.get("label_name"),
            }
        )

    return pd.DataFrame(portable_rows), missing_references, index_rows


def _write_relative_split_assignments(
    source_path: Path,
    destination_path: Path,
    *,
    source_root: Path,
    path_lookup: dict[str, str] | None = None,
) -> None:
    """Rewrite the split assignments file so image paths are relative inside the bundle."""

    if not source_path.exists():
        return
    frame = pd.read_csv(source_path)
    if "image" in frame.columns:
        relative_images: list[str] = []
        for raw_image in frame["image"].tolist():
            raw_key = None if raw_image is None else str(raw_image)
            relative_lookup = None if path_lookup is None or raw_key is None else path_lookup.get(raw_key)
            if relative_lookup is not None:
                relative_images.append(relative_lookup)
                continue
            resolved = _resolve_source_path(raw_image, source_root)
            if resolved is None or not resolved.exists():
                relative_images.append(str(raw_image))
                continue
            relative_images.append(_relative_to_root(resolved, source_root).as_posix())
        frame["image"] = relative_images
    frame.to_csv(destination_path, index=False)


def _write_bundle_readme(bundle_root: Path, *, materialize_mode: str) -> None:
    """Write a short upload/readback guide into the bundle."""

    readme_path = bundle_root / "README.md"
    lines = [
        "# Kaggle Alzheimer Upload Bundle",
        "",
        "This bundle keeps the Kaggle 2D branch portable and separate from OASIS.",
        "",
        "## Contents",
        "",
        "- `OriginalDataset/`",
        "- `AugmentedAlzheimerDataset/`",
        "- `backend_reference/kaggle_alz_manifest_relative.csv`",
        "- `backend_reference/kaggle_alz_train_manifest_relative.csv`",
        "- `backend_reference/kaggle_alz_val_manifest_relative.csv`",
        "- `backend_reference/kaggle_alz_test_manifest_relative.csv`",
        "- `backend_reference/kaggle_alz_split_assignments_relative.csv`",
        "- `backend_reference/kaggle_alz_manifest_summary.json`",
        "- `backend_reference/kaggle_alz_split_summary.json`",
        "- `backend_reference/kaggle_file_index.csv`",
        "- `backend_reference/kaggle_upload_bundle_summary.json`",
        "",
        "## Upload Advice",
        "",
        "- Upload this whole bundle folder to Google Drive.",
        "- Keep it extracted on Drive for Colab use.",
        "- When using it later, point `ALZ_KAGGLE_SOURCE_DIR` to this bundle root.",
        "",
        "## Notes",
        "",
        f"- local_materialize_mode: {materialize_mode}",
        "- The relative manifests are included for portability and inspection.",
        "- Rebuilding the Kaggle manifest and splits from this bundle is the safest path in a new environment.",
        "- This dataset stays a separate 2D comparison branch and should not be merged implicitly with OASIS.",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def build_kaggle_upload_bundle(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
    materialize_mode: str = "hardlink",
) -> KaggleUploadBundleResult:
    """Build a portable Kaggle upload bundle suitable for Drive upload."""

    resolved_settings = settings or get_app_settings()
    source_root = resolved_settings.kaggle_source_root
    if not source_root.exists():
        raise FileNotFoundError(f"Kaggle source root not found: {source_root}")

    resolved_manifest_path = manifest_path or (resolved_settings.data_root / "interim" / "kaggle_alz_manifest.csv")
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"Kaggle manifest not found: {resolved_manifest_path}")

    manifest_frame = pd.read_csv(resolved_manifest_path)
    subset_names = _collect_subset_names(manifest_frame, source_root)
    bundle_root = output_root or (resolved_settings.outputs_root / "exports" / "kaggle_alz_upload_bundle")
    bundle_root = ensure_directory(bundle_root)
    backend_reference_root = ensure_directory(bundle_root / "backend_reference")

    source_files = _collect_subset_files(source_root, subset_names)
    for source_path in source_files:
        relative_path = _relative_to_root(source_path, source_root)
        destination_path = bundle_root / relative_path
        _materialize_file(source_path, destination_path, mode=materialize_mode)

    relative_manifest_frame, missing_references, index_rows = _build_relative_manifest(
        manifest_frame,
        source_root=source_root,
    )
    path_lookup: dict[str, str] | None = None
    if len(relative_manifest_frame) == len(manifest_frame):
        path_lookup = {
            str(manifest_frame.iloc[index]["image"]): str(relative_manifest_frame.iloc[index]["image"])
            for index in range(len(manifest_frame))
        }

    relative_manifest_path = backend_reference_root / "kaggle_alz_manifest_relative.csv"
    relative_manifest_frame.to_csv(relative_manifest_path, index=False)

    split_manifest_names = (
        "kaggle_alz_train_manifest.csv",
        "kaggle_alz_val_manifest.csv",
        "kaggle_alz_test_manifest.csv",
    )
    for manifest_name in split_manifest_names:
        source_manifest = resolved_settings.data_root / "interim" / manifest_name
        if not source_manifest.exists():
            continue
        split_frame = pd.read_csv(source_manifest)
        relative_split_frame, _, _ = _build_relative_manifest(
            split_frame,
            source_root=source_root,
            path_lookup=path_lookup,
        )
        relative_name = manifest_name.replace(".csv", "_relative.csv")
        relative_split_frame.to_csv(backend_reference_root / relative_name, index=False)

    assignments_source = resolved_settings.data_root / "interim" / "kaggle_alz_split_assignments.csv"
    _write_relative_split_assignments(
        assignments_source,
        backend_reference_root / "kaggle_alz_split_assignments_relative.csv",
        source_root=source_root,
        path_lookup=path_lookup,
    )

    reference_copy_names = (
        "kaggle_alz_manifest_summary.json",
        "kaggle_alz_manifest_dropped_rows.csv",
        "kaggle_alz_split_summary.json",
    )
    for file_name in reference_copy_names:
        source_path = resolved_settings.data_root / "interim" / file_name
        if source_path.exists():
            shutil.copy2(source_path, backend_reference_root / file_name)

    file_index_path = backend_reference_root / "kaggle_file_index.csv"
    pd.DataFrame(index_rows).to_csv(file_index_path, index=False)

    if missing_references:
        missing_path = backend_reference_root / "kaggle_upload_bundle_missing_references.csv"
        with missing_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in missing_references for key in row}))
            writer.writeheader()
            writer.writerows(missing_references)

    summary_path = backend_reference_root / "kaggle_upload_bundle_summary.json"
    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle_root),
        "source_root": str(source_root),
        "manifest_path": str(resolved_manifest_path),
        "materialize_mode": materialize_mode,
        "included_subset_names": list(subset_names),
        "materialized_file_count": len(source_files),
        "missing_reference_count": len(missing_references),
        "notes": [
            "This bundle preserves the current full Kaggle 2D comparison dataset roots used by the backend.",
            "Use the bundle root itself as ALZ_KAGGLE_SOURCE_DIR when rebuilding the manifest elsewhere.",
            "The relative manifests are included for inspection and portability, but rebuilding the manifest from the bundle is the safest path.",
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _write_bundle_readme(bundle_root, materialize_mode=materialize_mode)

    return KaggleUploadBundleResult(
        bundle_root=bundle_root,
        relative_manifest_path=relative_manifest_path,
        summary_path=summary_path,
        file_index_path=file_index_path,
        included_subset_names=subset_names,
        materialized_file_count=len(source_files),
        missing_reference_count=len(missing_references),
        materialize_mode=materialize_mode,
    )
