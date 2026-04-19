"""Helpers to build a portable OASIS-2 upload bundle for Drive or Colab."""

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
from .oasis2 import OASIS2_PART_PATTERN, build_oasis2_session_manifest, resolve_oasis2_source_layout


class OASIS2UploadBundleError(ValueError):
    """Raised when an OASIS-2 upload bundle cannot be built safely."""


@dataclass(slots=True)
class OASIS2UploadBundleResult:
    """Artifacts and counts produced when building an OASIS-2 upload bundle."""

    bundle_root: Path
    relative_manifest_path: Path
    summary_path: Path
    session_index_path: Path
    included_session_count: int
    materialized_file_count: int
    missing_reference_count: int
    materialize_mode: str


@dataclass(slots=True, frozen=True)
class OASIS2UploadBundleCheck:
    """One validation finding for a built OASIS-2 upload bundle."""

    name: str
    status: str
    message: str
    details: dict[str, Any]


@dataclass(slots=True)
class OASIS2UploadBundleReport:
    """Serialized validation report for an OASIS-2 upload bundle."""

    generated_at: str
    bundle_root: str
    overall_status: str
    checks: list[OASIS2UploadBundleCheck]
    bundle_summary: dict[str, Any]
    notes: list[str]
    recommendations: list[str]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return {
            "generated_at": self.generated_at,
            "bundle_root": self.bundle_root,
            "overall_status": self.overall_status,
            "summary": _status_counts(self.checks),
            "bundle_summary": dict(self.bundle_summary),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "message": check.message,
                    "details": dict(check.details),
                }
                for check in self.checks
            ],
            "notes": list(self.notes),
            "recommendations": list(self.recommendations),
        }


def _resolve_source_path(path_value: str | Path | None, source_root: Path) -> Path | None:
    """Resolve a source path that may be relative or absolute."""

    if path_value in {None, ""}:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return source_root / candidate


def _relative_to_root(path: Path, source_root: Path) -> Path:
    """Return a path relative to the OASIS-2 source root or fail loudly."""

    try:
        return path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise OASIS2UploadBundleError(
            f"Expected {path} to be inside the OASIS-2 source root {source_root}"
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
    raise OASIS2UploadBundleError(f"Unsupported materialize mode: {mode}")


def _status_counts(checks: list[OASIS2UploadBundleCheck]) -> dict[str, int]:
    """Count pass/warn/fail states for JSON and Markdown summaries."""

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return counts


def _overall_status_from_checks(checks: list[OASIS2UploadBundleCheck]) -> str:
    """Collapse check states into one report-level status."""

    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OASIS2UploadBundleError(f"Expected JSON object at {path}")
    return payload


def _collect_bundle_rows(
    manifest_frame: pd.DataFrame,
    *,
    source_root: Path,
) -> tuple[list[dict[str, Any]], set[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect session-manifest rows, selected files, and audit entries."""

    portable_rows: list[dict[str, Any]] = []
    source_files: set[Path] = set()
    session_rows: list[dict[str, Any]] = []
    missing_references: list[dict[str, Any]] = []

    for row in manifest_frame.to_dict(orient="records"):
        meta = parse_manifest_meta(row.get("meta"))
        image_path = _resolve_source_path(row.get("image"), source_root)
        session_id = str(row.get("session_id", "")).strip()
        if image_path is None or not image_path.exists():
            missing_references.append(
                {
                    "session_id": session_id or None,
                    "reason": "missing_image_reference",
                    "image": row.get("image"),
                }
            )
            continue

        image_relative = _relative_to_root(image_path, source_root)
        source_files.add(image_path)

        paired_image = _resolve_source_path(meta.get("paired_image"), source_root)
        if paired_image is not None:
            if not paired_image.exists():
                missing_references.append(
                    {
                        "session_id": session_id or None,
                        "reason": "missing_paired_image",
                        "paired_image": meta.get("paired_image"),
                    }
                )
                continue
            source_files.add(paired_image)
            meta["paired_image"] = _relative_to_root(paired_image, source_root).as_posix()

        meta["source_root"] = "."
        portable_row = dict(row)
        portable_row["image"] = image_relative.as_posix()
        portable_row["meta"] = json.dumps(meta, ensure_ascii=True, sort_keys=True)
        portable_rows.append(portable_row)
        session_rows.append(
            {
                "subject_id": row.get("subject_id"),
                "session_id": session_id,
                "visit_number": row.get("visit_number"),
                "image_relative_path": image_relative.as_posix(),
                "selected_acquisition_id": meta.get("selected_acquisition_id"),
                "acquisition_count": meta.get("acquisition_count"),
            }
        )

    return portable_rows, source_files, session_rows, missing_references


def _write_relative_longitudinal_records(
    source_path: Path,
    destination_path: Path,
    *,
    source_root: Path,
) -> None:
    """Rewrite longitudinal record source paths to bundle-relative paths."""

    if not source_path.exists():
        return
    frame = pd.read_csv(source_path)
    if "source_path" in frame.columns:
        relative_paths: list[str] = []
        for raw_path in frame["source_path"].tolist():
            resolved = _resolve_source_path(raw_path, source_root)
            if resolved is None or not resolved.exists():
                relative_paths.append(str(raw_path))
                continue
            relative_paths.append(_relative_to_root(resolved, source_root).as_posix())
        frame["source_path"] = relative_paths
    frame.to_csv(destination_path, index=False)


def _write_bundle_readme(bundle_root: Path, *, materialize_mode: str) -> None:
    """Write a short upload/readback guide into the bundle."""

    readme_path = bundle_root / "README.md"
    lines = [
        "# OASIS-2 Upload Bundle",
        "",
        "This bundle contains the unlabeled OASIS-2 session subset prepared from the raw OAS2 split-part download.",
        "",
        "## Contents",
        "",
        "- `OAS2_RAW_PART1/` and `OAS2_RAW_PART2/` selected files only",
        "- `backend_reference/oasis2_session_manifest_relative.csv`",
        "- `backend_reference/oasis2_longitudinal_records_relative.csv`",
        "- `backend_reference/oasis2_subject_summary.csv`",
        "- `backend_reference/oasis2_raw_inventory.csv`",
        "- `backend_reference/oasis2_raw_inventory_summary.json`",
        "- `backend_reference/oasis2_session_manifest_summary.json`",
        "- `backend_reference/oasis2_metadata_template.csv`",
        "- `backend_reference/oasis2_metadata_template_summary.json`",
        "- `backend_reference/oasis2_labeled_prep_manifest.csv`",
        "- `backend_reference/oasis2_metadata_adapter_status.json`",
        "- `backend_reference/oasis2_subject_safe_split_plan.csv`",
        "- `backend_reference/oasis2_subject_safe_split_plan_summary.json`",
        "- `backend_reference/oasis2_session_index.csv`",
        "- `backend_reference/oasis2_upload_bundle_summary.json`",
        "",
        "## Upload Advice",
        "",
        "- Upload this whole bundle folder to Google Drive.",
        "- Keep it extracted on Drive for Colab use.",
        "- When using it later, point `ALZ_OASIS2_SOURCE_DIR` or --source-root to this bundle root.",
        "- The metadata template and split-plan files are planning artifacts; they do not make OASIS-2 training-ready by themselves.",
        "",
        "## Notes",
        "",
        f"- local_materialize_mode: {materialize_mode}",
        "- This is an unlabeled structural/longitudinal preparation bundle, not a supervised training set.",
        "- The backend can rebuild the unlabeled session manifest from this bundle if needed.",
        "- The bundle now includes metadata-mapping and subject-safe split-planning artifacts so remote review can start from the same local state.",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def build_oasis2_upload_bundle(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
    source_root: Path | None = None,
    output_root: Path | None = None,
    materialize_mode: str = "hardlink",
) -> OASIS2UploadBundleResult:
    """Build a portable OASIS-2 upload bundle from the unlabeled session manifest."""

    resolved_settings = settings or get_app_settings()
    layout = resolve_oasis2_source_layout(resolved_settings, source_root=source_root)
    if not layout.inspection_roots:
        raise FileNotFoundError(
            f"No OASIS-2 roots were found from {layout.source_root}. Set ALZ_OASIS2_SOURCE_DIR or pass --source-root."
        )
    resolved_source_root = layout.source_root

    resolved_manifest_path = manifest_path or (resolved_settings.data_root / "interim" / "oasis2_session_manifest.csv")
    if not resolved_manifest_path.exists():
        build_oasis2_session_manifest(settings=resolved_settings, source_root=source_root)
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"OASIS-2 session manifest not found: {resolved_manifest_path}")

    manifest_frame = pd.read_csv(resolved_manifest_path)
    bundle_root = output_root or (resolved_settings.outputs_root / "exports" / "oasis2_upload_bundle")
    bundle_root = ensure_directory(bundle_root)
    backend_reference_root = ensure_directory(bundle_root / "backend_reference")

    portable_rows, source_files, session_rows, missing_references = _collect_bundle_rows(
        manifest_frame,
        source_root=resolved_source_root,
    )

    for source_path in sorted(source_files):
        relative_path = _relative_to_root(source_path, resolved_source_root)
        destination_path = bundle_root / relative_path
        _materialize_file(source_path, destination_path, mode=materialize_mode)

    relative_manifest_path = backend_reference_root / "oasis2_session_manifest_relative.csv"
    pd.DataFrame(portable_rows).to_csv(relative_manifest_path, index=False)

    session_index_path = backend_reference_root / "oasis2_session_index.csv"
    pd.DataFrame(session_rows).to_csv(session_index_path, index=False)

    reference_copy_names = (
        "oasis2_raw_inventory.csv",
        "oasis2_raw_inventory_dropped_rows.csv",
        "oasis2_raw_inventory_summary.json",
        "oasis2_subject_summary.csv",
        "oasis2_session_manifest_summary.json",
        "oasis2_metadata_template.csv",
        "oasis2_metadata_template_summary.json",
        "oasis2_labeled_prep_manifest.csv",
        "oasis2_subject_safe_split_plan.csv",
        "oasis2_subject_safe_split_plan_summary.json",
    )
    for name in reference_copy_names:
        source_path = resolved_settings.data_root / "interim" / name
        if source_path.exists():
            shutil.copy2(source_path, backend_reference_root / name)

    longitudinal_source = resolved_settings.data_root / "interim" / "oasis2_longitudinal_records.csv"
    _write_relative_longitudinal_records(
        longitudinal_source,
        backend_reference_root / "oasis2_longitudinal_records_relative.csv",
        source_root=resolved_source_root,
    )

    readiness_json = resolved_settings.outputs_root / "reports" / "readiness" / "oasis2_readiness.json"
    readiness_md = resolved_settings.outputs_root / "reports" / "readiness" / "oasis2_readiness.md"
    onboarding_reference_names = (
        "oasis2_adapter_status.json",
        "oasis2_adapter_status.md",
        "oasis2_metadata_adapter_status.json",
        "oasis2_metadata_adapter_status.md",
        "oasis2_subject_safe_split_plan.md",
    )
    if readiness_json.exists():
        shutil.copy2(readiness_json, backend_reference_root / readiness_json.name)
    if readiness_md.exists():
        shutil.copy2(readiness_md, backend_reference_root / readiness_md.name)
    onboarding_root = resolved_settings.outputs_root / "reports" / "onboarding"
    for name in onboarding_reference_names:
        source_path = onboarding_root / name
        if source_path.exists():
            shutil.copy2(source_path, backend_reference_root / source_path.name)

    summary_path = backend_reference_root / "oasis2_upload_bundle_summary.json"
    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle_root),
        "source_root": str(resolved_source_root),
        "manifest_path": str(resolved_manifest_path),
        "materialize_mode": materialize_mode,
        "included_session_count": len(session_rows),
        "materialized_file_count": len(source_files),
        "missing_reference_count": len(missing_references),
        "inspection_roots": [str(path) for path in layout.inspection_roots],
        "backend_reference_files": sorted(path.name for path in backend_reference_root.iterdir() if path.is_file()),
        "notes": [
            "This bundle preserves one selected structural acquisition per OASIS-2 session.",
            "Use the bundle root as ALZ_OASIS2_SOURCE_DIR when rebuilding the unlabeled session manifest elsewhere.",
            "This bundle is intended for preprocessing, structural workflows, and longitudinal preparation, not supervised classification.",
            "Metadata and split-planning artifacts are included for remote review, but OASIS-2 still requires explicit labels and subject-safe split decisions before supervised training.",
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if missing_references:
        missing_path = backend_reference_root / "oasis2_upload_bundle_missing_references.csv"
        with missing_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in missing_references for key in row}))
            writer.writeheader()
            writer.writerows(missing_references)

    _write_bundle_readme(bundle_root, materialize_mode=materialize_mode)

    return OASIS2UploadBundleResult(
        bundle_root=bundle_root,
        relative_manifest_path=relative_manifest_path,
        summary_path=summary_path,
        session_index_path=session_index_path,
        included_session_count=len(session_rows),
        materialized_file_count=len(source_files),
        missing_reference_count=len(missing_references),
        materialize_mode=materialize_mode,
    )


def inspect_oasis2_upload_bundle(
    settings: AppSettings | None = None,
    *,
    bundle_root: Path | None = None,
    max_examples: int = 5,
) -> OASIS2UploadBundleReport:
    """Validate a built OASIS-2 upload bundle before remote use."""

    resolved_settings = settings or get_app_settings()
    resolved_bundle_root = (bundle_root or (resolved_settings.outputs_root / "exports" / "oasis2_upload_bundle")).expanduser().resolve()

    checks: list[OASIS2UploadBundleCheck] = []
    notes = [
        "This validator checks the bundle structure and reference files only. It does not turn OASIS-2 into a supervised training set.",
        "The intended use remains remote review, preprocessing, and longitudinal preparation until explicit visit and clinical metadata are filled.",
    ]
    recommendations: list[str] = []

    if not resolved_bundle_root.exists():
        checks.append(
            OASIS2UploadBundleCheck(
                name="bundle_root",
                status="fail",
                message="The OASIS-2 upload bundle root does not exist.",
                details={"bundle_root": str(resolved_bundle_root)},
            )
        )
        recommendations.append("Build the upload bundle first or point --bundle-root at an existing extracted OASIS-2 bundle.")
        return OASIS2UploadBundleReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            bundle_root=str(resolved_bundle_root),
            overall_status=_overall_status_from_checks(checks),
            checks=checks,
            bundle_summary={"bundle_root_exists": False},
            notes=notes,
            recommendations=recommendations,
        )

    if not resolved_bundle_root.is_dir():
        checks.append(
            OASIS2UploadBundleCheck(
                name="bundle_root",
                status="fail",
                message="The OASIS-2 upload bundle path exists but is not a directory.",
                details={"bundle_root": str(resolved_bundle_root)},
            )
        )
        recommendations.append("Point --bundle-root at the extracted bundle folder, not a single file.")
        return OASIS2UploadBundleReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            bundle_root=str(resolved_bundle_root),
            overall_status=_overall_status_from_checks(checks),
            checks=checks,
            bundle_summary={"bundle_root_exists": True, "bundle_root_is_directory": False},
            notes=notes,
            recommendations=recommendations,
        )

    backend_reference_root = resolved_bundle_root / "backend_reference"
    readme_path = resolved_bundle_root / "README.md"
    raw_part_roots = tuple(
        sorted(
            (
                child
                for child in resolved_bundle_root.iterdir()
                if child.is_dir() and OASIS2_PART_PATTERN.fullmatch(child.name)
            ),
            key=lambda path: path.name,
        )
    )
    top_level_missing = [
        name
        for name, path in {
            "backend_reference": backend_reference_root,
            "README.md": readme_path,
        }.items()
        if not path.exists()
    ]
    checks.append(
        OASIS2UploadBundleCheck(
            name="bundle_layout",
            status="pass" if not top_level_missing and raw_part_roots else "fail",
            message=(
                "The upload bundle has the expected top-level layout."
                if not top_level_missing and raw_part_roots
                else "The upload bundle is missing required top-level items."
            ),
            details={
                "bundle_root": str(resolved_bundle_root),
                "raw_part_roots": [path.name for path in raw_part_roots],
                "missing_top_level_items": top_level_missing,
            },
        )
    )

    core_reference_files = (
        "oasis2_session_manifest_relative.csv",
        "oasis2_longitudinal_records_relative.csv",
        "oasis2_subject_summary.csv",
        "oasis2_raw_inventory.csv",
        "oasis2_raw_inventory_summary.json",
        "oasis2_session_manifest_summary.json",
        "oasis2_session_index.csv",
        "oasis2_upload_bundle_summary.json",
    )
    planning_reference_files = (
        "oasis2_metadata_template.csv",
        "oasis2_metadata_template_summary.json",
        "oasis2_labeled_prep_manifest.csv",
        "oasis2_metadata_adapter_status.json",
        "oasis2_subject_safe_split_plan.csv",
        "oasis2_subject_safe_split_plan_summary.json",
    )
    core_missing = [name for name in core_reference_files if not (backend_reference_root / name).exists()]
    planning_missing = [name for name in planning_reference_files if not (backend_reference_root / name).exists()]
    checks.append(
        OASIS2UploadBundleCheck(
            name="backend_reference_core",
            status="pass" if not core_missing else "fail",
            message=(
                "The bundle contains the core backend reference files needed for remote reuse."
                if not core_missing
                else "The bundle is missing required backend reference files."
            ),
            details={"missing_core_files": core_missing},
        )
    )
    checks.append(
        OASIS2UploadBundleCheck(
            name="backend_reference_planning",
            status="pass" if not planning_missing else "warn",
            message=(
                "The bundle includes the metadata and split-planning artifacts for remote review."
                if not planning_missing
                else "The bundle is usable, but some metadata or split-planning artifacts are missing."
            ),
            details={"missing_planning_files": planning_missing},
        )
    )

    summary_path = backend_reference_root / "oasis2_upload_bundle_summary.json"
    summary_payload: dict[str, Any] = {}
    if summary_path.exists():
        summary_payload = _load_json_object(summary_path)
        missing_reference_count = int(summary_payload.get("missing_reference_count", 0) or 0)
        checks.append(
            OASIS2UploadBundleCheck(
                name="bundle_summary",
                status="pass" if missing_reference_count == 0 else "warn",
                message=(
                    "The bundle summary reports no missing source references."
                    if missing_reference_count == 0
                    else "The bundle summary reports missing source references."
                ),
                details={
                    "included_session_count": summary_payload.get("included_session_count"),
                    "materialized_file_count": summary_payload.get("materialized_file_count"),
                    "missing_reference_count": missing_reference_count,
                    "inspection_roots": summary_payload.get("inspection_roots", []),
                },
            )
        )
    else:
        checks.append(
            OASIS2UploadBundleCheck(
                name="bundle_summary",
                status="warn",
                message="The bundle summary JSON is missing, so portable counts could not be verified.",
                details={"summary_path": str(summary_path)},
            )
        )

    manifest_row_count = 0
    session_index_row_count = 0
    manifest_examples: list[str] = []
    missing_materialized_paths: list[str] = []
    manifest_path = backend_reference_root / "oasis2_session_manifest_relative.csv"
    session_index_path = backend_reference_root / "oasis2_session_index.csv"
    if manifest_path.exists():
        manifest_frame = pd.read_csv(manifest_path)
        manifest_row_count = int(len(manifest_frame))
        if not manifest_frame.empty:
            manifest_examples = [str(value) for value in manifest_frame["image"].head(max_examples).tolist()]
            for row in manifest_frame.head(max_examples).to_dict(orient="records"):
                relative_image = str(row.get("image", "")).strip()
                if relative_image and not (resolved_bundle_root / relative_image).exists():
                    missing_materialized_paths.append(relative_image)
                meta = parse_manifest_meta(row.get("meta"))
                paired_image = str(meta.get("paired_image", "")).strip()
                if paired_image and not (resolved_bundle_root / paired_image).exists():
                    missing_materialized_paths.append(paired_image)
        if session_index_path.exists():
            session_index_row_count = int(len(pd.read_csv(session_index_path)))
        count_matches_summary = (
            summary_payload.get("included_session_count") in {None, manifest_row_count}
            and session_index_row_count in {0, manifest_row_count}
        )
        checks.append(
            OASIS2UploadBundleCheck(
                name="manifest_consistency",
                status="pass" if manifest_row_count > 0 and count_matches_summary else "warn",
                message=(
                    "The relative manifest and session index counts are internally consistent."
                    if manifest_row_count > 0 and count_matches_summary
                    else "The relative manifest exists, but one or more bundle counts do not line up."
                ),
                details={
                    "manifest_row_count": manifest_row_count,
                    "session_index_row_count": session_index_row_count,
                    "summary_included_session_count": summary_payload.get("included_session_count"),
                    "manifest_examples": manifest_examples,
                },
            )
        )
    else:
        checks.append(
            OASIS2UploadBundleCheck(
                name="manifest_consistency",
                status="fail",
                message="The relative OASIS-2 session manifest is missing from backend_reference.",
                details={"manifest_path": str(manifest_path)},
            )
        )

    checks.append(
        OASIS2UploadBundleCheck(
            name="materialized_examples",
            status="pass" if not missing_materialized_paths else "warn",
            message=(
                "Sample relative manifest paths resolve inside the bundle."
                if not missing_materialized_paths
                else "Some sample relative manifest paths do not resolve inside the bundle."
            ),
            details={
                "checked_example_count": len(manifest_examples),
                "missing_materialized_paths": missing_materialized_paths[:max_examples],
            },
        )
    )

    bundle_summary = {
        "bundle_root_exists": True,
        "bundle_root_is_directory": True,
        "bundle_root": str(resolved_bundle_root),
        "raw_part_roots": [str(path) for path in raw_part_roots],
        "core_reference_files_present": len(core_reference_files) - len(core_missing),
        "planning_reference_files_present": len(planning_reference_files) - len(planning_missing),
        "manifest_row_count": manifest_row_count,
        "session_index_row_count": session_index_row_count,
        "included_session_count": summary_payload.get("included_session_count"),
        "materialized_file_count": summary_payload.get("materialized_file_count"),
        "missing_reference_count": summary_payload.get("missing_reference_count"),
    }

    recommendations.extend(
        [
            "Point ALZ_OASIS2_SOURCE_DIR or --source-root at this bundle root when rebuilding the OASIS-2 session manifest on another machine or in Colab.",
            "Use the bundle for remote preprocessing and longitudinal review only; do not treat it as a supervised training set until the metadata template is filled explicitly.",
            "Review oasis2_metadata_template.csv, oasis2_metadata_adapter_status.json, and oasis2_subject_safe_split_plan.csv before opening labeled OASIS-2 experiments.",
        ]
    )

    return OASIS2UploadBundleReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        bundle_root=str(resolved_bundle_root),
        overall_status=_overall_status_from_checks(checks),
        checks=checks,
        bundle_summary=bundle_summary,
        notes=notes,
        recommendations=recommendations,
    )


def save_oasis2_upload_bundle_report(
    report: OASIS2UploadBundleReport,
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis2_upload_bundle_status",
) -> tuple[Path, Path]:
    """Save the OASIS-2 upload bundle report as JSON and Markdown."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "onboarding")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    payload = report.to_payload()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload["summary"]
    lines = [
        "# OASIS-2 Upload Bundle Status",
        "",
        f"- overall_status: {report.overall_status}",
        f"- generated_at: {report.generated_at}",
        f"- bundle_root: {report.bundle_root}",
        "",
        "## Summary",
        "",
        f"- pass: {summary.get('pass', 0)}",
        f"- warn: {summary.get('warn', 0)}",
        f"- fail: {summary.get('fail', 0)}",
        f"- included_session_count: {report.bundle_summary.get('included_session_count')}",
        f"- materialized_file_count: {report.bundle_summary.get('materialized_file_count')}",
        f"- missing_reference_count: {report.bundle_summary.get('missing_reference_count')}",
        f"- manifest_row_count: {report.bundle_summary.get('manifest_row_count')}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {check.status.upper()}: {check.name} - {check.message}" for check in report.checks)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    if report.recommendations:
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"- {recommendation}" for recommendation in report.recommendations)

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
