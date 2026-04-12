"""Readiness helpers for future OASIS-2 longitudinal dataset onboarding.

This module is intentionally conservative. It does not claim that OASIS-2 is
present or fully supported. Instead, it inspects a candidate source root and
reports whether the backend has enough raw ingredients to begin a future
manifest/integration effort safely.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from os import getenv
from pathlib import Path
from typing import Any

from src.configs.runtime import AppSettings
from src.utils.io_utils import ensure_directory

OASIS2_SOURCE_ENV_VAR = "ALZ_OASIS2_SOURCE_DIR"
SUPPORTED_3D_VOLUME_SUFFIXES = (".nii", ".nii.gz", ".img", ".hdr", ".mgh", ".mgz")
DICOM_SUFFIXES = (".dcm", ".ima")
METADATA_SUFFIXES = (".csv", ".tsv", ".xls", ".xlsx", ".xml", ".json")
SUBJECT_ID_PATTERN = re.compile(r"(OAS2[_-]\d{4})", re.IGNORECASE)
SESSION_ID_PATTERN = re.compile(r"(OAS2[_-]\d{4}_MR\d+)", re.IGNORECASE)


def _normalize_suffix(path: Path) -> str:
    """Return a normalized suffix, including compound suffixes like ``.nii.gz``."""

    suffixes = [part.lower() for part in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] == [".nii", ".gz"]:
        return ".nii.gz"
    return path.suffix.lower()


def _normalize_oasis_identifier(value: str) -> str:
    """Normalize OASIS identifiers to a stable upper-case underscore form."""

    return value.upper().replace("-", "_")


def _status_counts(checks: list["OASIS2ReadinessCheck"]) -> dict[str, int]:
    """Count pass/warn/fail states for JSON and markdown summaries."""

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return counts


def resolve_oasis2_source_root(
    settings: AppSettings | None = None,
    *,
    source_root: Path | None = None,
) -> tuple[Path, list[Path], str]:
    """Resolve the most likely OASIS-2 source root and return the search context."""

    resolved_settings = settings or AppSettings.from_env()
    if source_root is not None:
        return source_root.expanduser().resolve(), [], "argument"

    env_value = getenv(OASIS2_SOURCE_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser().resolve(), [], "env"

    candidate_roots = [
        resolved_settings.collection_root / "OASIS2",
        resolved_settings.collection_root / "OASIS-2",
        resolved_settings.workspace_root / "OASIS2",
        resolved_settings.workspace_root / "OASIS-2",
    ]
    resolved_candidates = [candidate.resolve() for candidate in candidate_roots]
    for candidate in resolved_candidates:
        if candidate.exists():
            return candidate, resolved_candidates, "auto_existing_candidate"
    return resolved_candidates[0], resolved_candidates, "auto_default_candidate"


@dataclass(slots=True, frozen=True)
class OASIS2ReadinessCheck:
    """One readiness finding for future OASIS-2 integration."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OASIS2ReadinessReport:
    """Serialized readiness report for future OASIS-2 onboarding."""

    generated_at: str
    source_root: str
    source_resolution: str
    overall_status: str
    checks: list[OASIS2ReadinessCheck]
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Convert the report to JSON-safe data."""

        return {
            "generated_at": self.generated_at,
            "source_root": self.source_root,
            "source_resolution": self.source_resolution,
            "overall_status": self.overall_status,
            "summary": _status_counts(self.checks),
            "dataset_summary": dict(self.dataset_summary),
            "checks": [asdict(check) for check in self.checks],
            "notes": list(self.notes),
            "recommendations": list(self.recommendations),
        }


def _overall_status(checks: list[OASIS2ReadinessCheck]) -> str:
    """Collapse check states into one report-level status."""

    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _build_summary_for_missing_source(
    *,
    source_root: Path,
    candidate_roots: list[Path],
) -> dict[str, Any]:
    """Return a minimal summary when no OASIS-2 source root is available yet."""

    return {
        "source_exists": False,
        "source_is_directory": False,
        "candidate_roots": [str(path) for path in candidate_roots],
        "total_files": 0,
        "total_directories": 0,
        "format_counts": {},
        "supported_volume_file_count": 0,
        "dicom_file_count": 0,
        "metadata_file_count": 0,
        "unique_subject_id_count": 0,
        "unique_session_id_count": 0,
        "longitudinal_subject_count": 0,
        "subject_id_examples": [],
        "session_id_examples": [],
        "volume_examples": [],
        "metadata_examples": [],
        "source_root": str(source_root),
    }


def build_oasis2_readiness_report(
    settings: AppSettings | None = None,
    *,
    source_root: Path | None = None,
    max_examples: int = 5,
) -> OASIS2ReadinessReport:
    """Inspect a candidate OASIS-2 source root and summarize onboarding readiness."""

    resolved_settings = settings or AppSettings.from_env()
    resolved_root, candidate_roots, source_resolution = resolve_oasis2_source_root(
        resolved_settings,
        source_root=source_root,
    )
    checks: list[OASIS2ReadinessCheck] = []
    notes: list[str] = [
        "This report is a future-dataset readiness aid. It does not build an OASIS-2 manifest or claim model compatibility by itself.",
        "OASIS-2 is treated as a planned longitudinal extension, not as an already-integrated training source.",
    ]
    recommendations: list[str] = []

    if not resolved_root.exists():
        checks.append(
            OASIS2ReadinessCheck(
                name="source_root",
                status="warn",
                message="No OASIS-2 source directory was found yet.",
                details={
                    "source_root": str(resolved_root),
                    "source_resolution": source_resolution,
                    "candidate_roots": [str(path) for path in candidate_roots],
                    "environment_variable": OASIS2_SOURCE_ENV_VAR,
                },
            )
        )
        checks.append(
            OASIS2ReadinessCheck(
                name="onboarding_readiness",
                status="warn",
                message="Future OASIS-2 onboarding is blocked only because the dataset is not present locally yet.",
                details={"next_step": "Place OASIS-2 under one of the candidate roots or set ALZ_OASIS2_SOURCE_DIR."},
            )
        )
        recommendations.extend(
            [
                "When OASIS-2 becomes available, point the readiness check at the dataset root with --source-root or ALZ_OASIS2_SOURCE_DIR.",
                "Keep OASIS-2 separate from OASIS-1 until a dedicated manifest and split strategy is implemented.",
            ]
        )
        return OASIS2ReadinessReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            source_root=str(resolved_root),
            source_resolution=source_resolution,
            overall_status=_overall_status(checks),
            checks=checks,
            dataset_summary=_build_summary_for_missing_source(source_root=resolved_root, candidate_roots=candidate_roots),
            notes=notes,
            recommendations=recommendations,
        )

    if not resolved_root.is_dir():
        checks.append(
            OASIS2ReadinessCheck(
                name="source_root",
                status="fail",
                message="The resolved OASIS-2 source path exists but is not a directory.",
                details={"source_root": str(resolved_root), "source_resolution": source_resolution},
            )
        )
        recommendations.append("Point the readiness check at the dataset directory rather than a single file.")
        return OASIS2ReadinessReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            source_root=str(resolved_root),
            source_resolution=source_resolution,
            overall_status=_overall_status(checks),
            checks=checks,
            dataset_summary={
                "source_exists": True,
                "source_is_directory": False,
                "candidate_roots": [str(path) for path in candidate_roots],
                "source_root": str(resolved_root),
            },
            notes=notes,
            recommendations=recommendations,
        )

    total_directories = 0
    total_files = 0
    format_counts: Counter[str] = Counter()
    supported_volume_examples: list[str] = []
    metadata_examples: list[str] = []
    dicom_examples: list[str] = []
    subject_ids: set[str] = set()
    session_ids: set[str] = set()
    subject_to_sessions: dict[str, set[str]] = defaultdict(set)
    subject_to_volume_containers: dict[str, set[str]] = defaultdict(set)

    for path in resolved_root.rglob("*"):
        if path.is_dir():
            total_directories += 1
            continue
        if not path.is_file():
            continue
        total_files += 1
        relative_path = path.relative_to(resolved_root)
        relative_name = relative_path.as_posix()
        suffix = _normalize_suffix(path)
        format_counts[suffix] += 1

        subject_match = SUBJECT_ID_PATTERN.search(relative_name)
        session_match = SESSION_ID_PATTERN.search(relative_name)
        normalized_subject: str | None = None

        if subject_match:
            normalized_subject = _normalize_oasis_identifier(subject_match.group(1))
            subject_ids.add(normalized_subject)
        if session_match:
            normalized_session = _normalize_oasis_identifier(session_match.group(1))
            session_ids.add(normalized_session)
            session_subject = normalized_session.split("_MR", 1)[0]
            subject_ids.add(session_subject)
            subject_to_sessions[session_subject].add(normalized_session)
            normalized_subject = session_subject

        if suffix in SUPPORTED_3D_VOLUME_SUFFIXES:
            if len(supported_volume_examples) < max_examples:
                supported_volume_examples.append(relative_name)
            if normalized_subject is not None:
                subject_to_volume_containers[normalized_subject].add(relative_path.parent.as_posix())
        elif suffix in METADATA_SUFFIXES and len(metadata_examples) < max_examples:
            metadata_examples.append(relative_name)
        elif suffix in DICOM_SUFFIXES and len(dicom_examples) < max_examples:
            dicom_examples.append(relative_name)

    longitudinal_subjects = {
        subject_id
        for subject_id, sessions in subject_to_sessions.items()
        if len(sessions) > 1
    }
    if not longitudinal_subjects:
        longitudinal_subjects = {
            subject_id
            for subject_id, containers in subject_to_volume_containers.items()
            if len(containers) > 1
        }

    supported_volume_count = sum(format_counts.get(suffix, 0) for suffix in SUPPORTED_3D_VOLUME_SUFFIXES)
    dicom_count = sum(format_counts.get(suffix, 0) for suffix in DICOM_SUFFIXES)
    metadata_count = sum(format_counts.get(suffix, 0) for suffix in METADATA_SUFFIXES)

    checks.append(
        OASIS2ReadinessCheck(
            name="source_root",
            status="pass",
            message="An OASIS-2 candidate source directory is available for inspection.",
            details={
                "source_root": str(resolved_root),
                "source_resolution": source_resolution,
                "candidate_roots": [str(path) for path in candidate_roots],
            },
        )
    )
    checks.append(
        OASIS2ReadinessCheck(
            name="image_formats",
            status="pass" if supported_volume_count > 0 or dicom_count > 0 else "warn",
            message=(
                "Detected 3D image files that could support future OASIS-2 onboarding."
                if supported_volume_count > 0 or dicom_count > 0
                else "No likely OASIS-2 3D image files were detected yet."
            ),
            details={
                "supported_volume_file_count": supported_volume_count,
                "dicom_file_count": dicom_count,
                "format_counts": dict(sorted(format_counts.items())),
                "volume_examples": supported_volume_examples,
                "dicom_examples": dicom_examples,
            },
        )
    )
    checks.append(
        OASIS2ReadinessCheck(
            name="metadata_files",
            status="pass" if metadata_count > 0 else "warn",
            message=(
                "Detected metadata-like files that could support manifest building."
                if metadata_count > 0
                else "No metadata-like files were detected; future manifest building may need manual metadata handling."
            ),
            details={
                "metadata_file_count": metadata_count,
                "metadata_examples": metadata_examples,
            },
        )
    )
    checks.append(
        OASIS2ReadinessCheck(
            name="subject_identifiers",
            status="pass" if subject_ids else "warn",
            message=(
                "Detected OASIS-2-style subject identifiers in the candidate source."
                if subject_ids
                else "No OASIS-2-style subject identifiers were detected from file or directory names."
            ),
            details={
                "unique_subject_id_count": len(subject_ids),
                "subject_id_examples": sorted(subject_ids)[:max_examples],
            },
        )
    )
    checks.append(
        OASIS2ReadinessCheck(
            name="longitudinal_signals",
            status="pass" if longitudinal_subjects else "warn",
            message=(
                "Detected repeated-subject structure that looks promising for longitudinal expansion."
                if longitudinal_subjects
                else "No repeated-subject longitudinal pattern was detected yet from the available file names."
            ),
            details={
                "unique_session_id_count": len(session_ids),
                "session_id_examples": sorted(session_ids)[:max_examples],
                "longitudinal_subject_count": len(longitudinal_subjects),
                "longitudinal_subject_examples": sorted(longitudinal_subjects)[:max_examples],
            },
        )
    )

    if supported_volume_count > 0 and subject_ids:
        pipeline_status = "pass"
        pipeline_message = (
            "The available file patterns look compatible with a future OASIS-2 manifest builder once label/metadata mapping is defined."
        )
    elif dicom_count > 0 and subject_ids:
        pipeline_status = "warn"
        pipeline_message = (
            "OASIS-2 looks present, but the current backend would still need a dedicated DICOM intake or conversion step."
        )
    else:
        pipeline_status = "warn"
        pipeline_message = (
            "The backend cannot safely claim OASIS-2 onboarding readiness yet because key volume or identifier signals are missing."
        )

    checks.append(
        OASIS2ReadinessCheck(
            name="pipeline_fit",
            status=pipeline_status,
            message=pipeline_message,
            details={
                "supported_volume_file_count": supported_volume_count,
                "dicom_file_count": dicom_count,
                "unique_subject_id_count": len(subject_ids),
                "unique_session_id_count": len(session_ids),
            },
        )
    )

    if dicom_count > 0 and supported_volume_count == 0:
        notes.append(
            "Only DICOM-like files were detected. The current backend is stronger with converted 3D volume files (for example NIfTI or Analyze pairs)."
        )
    if not longitudinal_subjects:
        notes.append(
            "Longitudinal readiness is still uncertain because repeated-subject structure was not visible from the inspected file names alone."
        )
    if metadata_count == 0:
        notes.append(
            "A future OASIS-2 manifest builder will be easier to implement once metadata tables or XML sidecars are available."
        )

    recommendations.append("Keep OASIS-2 separate from OASIS-1 until a dedicated manifest and split workflow exists.")
    if supported_volume_count > 0 and subject_ids:
        recommendations.append("The next OASIS-2 step can be a dedicated manifest adapter once label and visit metadata are available.")
    elif dicom_count > 0:
        recommendations.append("Plan a DICOM-to-volume conversion or DICOM intake layer before treating OASIS-2 as model-ready.")
    else:
        recommendations.append("Confirm the exact OASIS-2 root and file layout before adding dataset-specific code.")

    dataset_summary = {
        "source_exists": True,
        "source_is_directory": True,
        "candidate_roots": [str(path) for path in candidate_roots],
        "source_root": str(resolved_root),
        "total_files": total_files,
        "total_directories": total_directories,
        "format_counts": dict(sorted(format_counts.items())),
        "supported_volume_file_count": supported_volume_count,
        "dicom_file_count": dicom_count,
        "metadata_file_count": metadata_count,
        "unique_subject_id_count": len(subject_ids),
        "unique_session_id_count": len(session_ids),
        "longitudinal_subject_count": len(longitudinal_subjects),
        "subject_id_examples": sorted(subject_ids)[:max_examples],
        "session_id_examples": sorted(session_ids)[:max_examples],
        "volume_examples": supported_volume_examples,
        "metadata_examples": metadata_examples,
    }

    return OASIS2ReadinessReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_root=str(resolved_root),
        source_resolution=source_resolution,
        overall_status=_overall_status(checks),
        checks=checks,
        dataset_summary=dataset_summary,
        notes=notes,
        recommendations=recommendations,
    )


def save_oasis2_readiness_report(
    report: OASIS2ReadinessReport,
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis2_readiness",
) -> tuple[Path, Path]:
    """Save the OASIS-2 readiness report as JSON and Markdown."""

    resolved_settings = settings or AppSettings.from_env()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "readiness")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"

    payload = report.to_payload()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# OASIS-2 Readiness Report",
        "",
        f"- overall_status: {report.overall_status}",
        f"- generated_at: {report.generated_at}",
        f"- source_root: {report.source_root}",
        f"- source_resolution: {report.source_resolution}",
        "",
        "## Summary",
        "",
    ]
    summary = payload["summary"]
    lines.extend(
        [
            f"- pass: {summary.get('pass', 0)}",
            f"- warn: {summary.get('warn', 0)}",
            f"- fail: {summary.get('fail', 0)}",
            f"- supported_volume_file_count: {report.dataset_summary.get('supported_volume_file_count', 0)}",
            f"- metadata_file_count: {report.dataset_summary.get('metadata_file_count', 0)}",
            f"- unique_subject_id_count: {report.dataset_summary.get('unique_subject_id_count', 0)}",
            f"- longitudinal_subject_count: {report.dataset_summary.get('longitudinal_subject_count', 0)}",
            "",
            "## Checks",
            "",
        ]
    )
    lines.extend(f"- {check.status.upper()}: {check.name} - {check.message}" for check in report.checks)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    if report.recommendations:
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"- {recommendation}" for recommendation in report.recommendations)

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path

