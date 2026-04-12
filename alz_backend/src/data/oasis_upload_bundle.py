"""Helpers to build a smaller OASIS upload bundle for Drive or Colab."""

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
from .oasis1 import find_primary_oasis1_metadata_table


class OASISUploadBundleError(ValueError):
    """Raised when a portable OASIS upload bundle cannot be built safely."""


@dataclass(slots=True)
class OASISUploadBundleResult:
    """Artifacts and counts produced when building an OASIS upload bundle."""

    bundle_root: Path
    oasis_subset_root: Path
    relative_manifest_path: Path
    summary_path: Path
    session_index_path: Path
    included_session_count: int
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
    """Return a path relative to the OASIS source root or fail loudly."""

    try:
        return path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise OASISUploadBundleError(
            f"Expected {path} to be inside the OASIS source root {source_root}"
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
    raise OASISUploadBundleError(f"Unsupported materialize mode: {mode}")


def _collect_bundle_rows(
    manifest_frame: pd.DataFrame,
    *,
    source_root: Path,
) -> tuple[list[dict[str, Any]], set[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect manifest rows, source files, and session-level audit entries."""

    portable_rows: list[dict[str, Any]] = []
    source_files: set[Path] = set()
    session_rows: list[dict[str, Any]] = []
    missing_references: list[dict[str, Any]] = []

    for row in manifest_frame.to_dict(orient="records"):
        meta = parse_manifest_meta(row.get("meta"))
        image_path = _resolve_source_path(row.get("image"), source_root)
        xml_path = _resolve_source_path(meta.get("xml_path"), source_root)
        session_id = str(meta.get("session_id", "")).strip()

        if image_path is None or not image_path.exists():
            missing_references.append(
                {
                    "session_id": session_id or None,
                    "reason": "missing_image_reference",
                    "image": row.get("image"),
                }
            )
            continue
        if xml_path is None or not xml_path.exists():
            missing_references.append(
                {
                    "session_id": session_id or None,
                    "reason": "missing_xml_reference",
                    "xml_path": meta.get("xml_path"),
                }
            )
            continue

        image_relative = _relative_to_root(image_path, source_root)
        xml_relative = _relative_to_root(xml_path, source_root)
        source_files.add(image_path)
        source_files.add(xml_path)
        if image_path.suffix.lower() == ".hdr":
            paired_img = image_path.with_suffix(".img")
            if not paired_img.exists():
                missing_references.append(
                    {
                        "session_id": session_id or None,
                        "reason": "missing_img_pair",
                        "hdr_path": str(image_path),
                    }
                )
                continue
            source_files.add(paired_img)

        portable_meta = dict(meta)
        if xml_path is not None:
            portable_meta["xml_path"] = xml_relative.as_posix()
        portable_row = dict(row)
        portable_row["image"] = image_relative.as_posix()
        portable_row["meta"] = json.dumps(portable_meta, ensure_ascii=True, sort_keys=True)
        portable_rows.append(portable_row)
        session_rows.append(
            {
                "session_id": session_id,
                "subject_id": row.get("subject_id"),
                "label": row.get("label"),
                "label_name": row.get("label_name"),
                "image_relative_path": image_relative.as_posix(),
                "xml_relative_path": xml_relative.as_posix(),
            }
        )

    return portable_rows, source_files, session_rows, missing_references


def _write_bundle_readme(bundle_root: Path, *, materialize_mode: str) -> None:
    """Write a short upload/readback guide into the bundle."""

    readme_path = bundle_root / "README.md"
    lines = [
        "# OASIS Upload Bundle",
        "",
        "This bundle contains only the labeled OASIS-1 sessions currently used by the backend.",
        "",
        "## Contents",
        "",
        "- `OASIS/`",
        "- `backend_reference/oasis1_manifest_relative.csv`",
        "- `backend_reference/oasis1_manifest_summary.json`",
        "- `backend_reference/oasis1_manifest_dropped_rows.csv`",
        "- `backend_reference/oasis_session_index.csv`",
        "- `backend_reference/oasis_upload_bundle_summary.json`",
        "",
        "## Upload Advice",
        "",
        "- Upload this whole bundle folder to Google Drive.",
        "- Keep it extracted on Drive for Colab use.",
        "- When using it later, point `ALZ_OASIS_SOURCE_DIR` to the `OASIS/` folder inside this bundle.",
        "",
        "## Notes",
        "",
        f"- local_materialize_mode: {materialize_mode}",
        "- The relative manifest is for reference and portability.",
        "- The backend can rebuild a fresh local manifest from the `OASIS/` subset using the copied metadata workbook and XML files.",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def build_oasis_upload_bundle(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
    materialize_mode: str = "hardlink",
) -> OASISUploadBundleResult:
    """Build a smaller labeled OASIS bundle suitable for Drive upload."""

    resolved_settings = settings or get_app_settings()
    source_root = resolved_settings.oasis_source_root
    if not source_root.exists():
        raise FileNotFoundError(f"OASIS source root not found: {source_root}")

    resolved_manifest_path = manifest_path or (resolved_settings.data_root / "interim" / "oasis1_manifest.csv")
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"OASIS manifest not found: {resolved_manifest_path}")

    manifest_frame = pd.read_csv(resolved_manifest_path)
    bundle_root = output_root or (resolved_settings.outputs_root / "exports" / "oasis1_upload_bundle")
    bundle_root = ensure_directory(bundle_root)
    oasis_subset_root = ensure_directory(bundle_root / "OASIS")
    backend_reference_root = ensure_directory(bundle_root / "backend_reference")

    portable_rows, source_files, session_rows, missing_references = _collect_bundle_rows(
        manifest_frame,
        source_root=source_root,
    )

    metadata_path, _ = find_primary_oasis1_metadata_table(source_root)
    source_files.add(metadata_path)

    for source_path in sorted(source_files):
        relative_path = _relative_to_root(source_path, source_root)
        destination_path = oasis_subset_root / relative_path
        _materialize_file(source_path, destination_path, mode=materialize_mode)

    relative_manifest_path = backend_reference_root / "oasis1_manifest_relative.csv"
    pd.DataFrame(portable_rows).to_csv(relative_manifest_path, index=False)

    session_index_path = backend_reference_root / "oasis_session_index.csv"
    pd.DataFrame(session_rows).to_csv(session_index_path, index=False)

    summary_source_paths = {
        "manifest_summary": resolved_settings.data_root / "interim" / "oasis1_manifest_summary.json",
        "dropped_rows": resolved_settings.data_root / "interim" / "oasis1_manifest_dropped_rows.csv",
    }
    for key, source_path in summary_source_paths.items():
        if source_path.exists():
            shutil.copy2(source_path, backend_reference_root / source_path.name)

    summary_path = backend_reference_root / "oasis_upload_bundle_summary.json"
    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle_root),
        "oasis_subset_root": str(oasis_subset_root),
        "source_root": str(source_root),
        "manifest_path": str(resolved_manifest_path),
        "metadata_path": str(metadata_path),
        "materialize_mode": materialize_mode,
        "included_session_count": len(session_rows),
        "materialized_file_count": len(source_files),
        "missing_reference_count": len(missing_references),
        "notes": [
            "This bundle preserves only the labeled sessions used by the current OASIS manifest.",
            "Use the OASIS folder inside this bundle as ALZ_OASIS_SOURCE_DIR when rebuilding the manifest elsewhere.",
            "The relative manifest is included for inspection and portability, but rebuilding the manifest from the subset is the safest path.",
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if missing_references:
        missing_path = backend_reference_root / "oasis_upload_bundle_missing_references.csv"
        with missing_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in missing_references for key in row}))
            writer.writeheader()
            writer.writerows(missing_references)

    _write_bundle_readme(bundle_root, materialize_mode=materialize_mode)

    return OASISUploadBundleResult(
        bundle_root=bundle_root,
        oasis_subset_root=oasis_subset_root,
        relative_manifest_path=relative_manifest_path,
        summary_path=summary_path,
        session_index_path=session_index_path,
        included_session_count=len(session_rows),
        materialized_file_count=len(source_files),
        missing_reference_count=len(missing_references),
        materialize_mode=materialize_mode,
    )
