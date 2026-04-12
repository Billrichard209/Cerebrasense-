"""Subject-level longitudinal structural summaries built from OASIS volumetrics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.data.base_dataset import canonicalize_optional_string, parse_manifest_meta
from src.data.oasis_dataset import load_oasis_manifest
from src.utils.io_utils import ensure_directory
from src.volumetrics.measurements import VolumetricAnalysisResult, analyze_mri_volume, summarize_volumetrics

SESSION_VISIT_PATTERN = re.compile(r"_MR(\d+)\b", re.IGNORECASE)
CHANGE_METRICS = (
    "foreground_proxy_brain",
    "left_hemisphere_proxy",
    "right_hemisphere_proxy",
    "hemisphere_asymmetry_index",
)


class LongitudinalStructuralError(ValueError):
    """Raised when a longitudinal structural summary cannot be created safely."""


@dataclass(slots=True)
class StructuralTimepoint:
    """One subject-level structural timepoint."""

    subject_id: str
    session_id: str | None
    visit_order: int
    scan_timestamp: str | None
    image: str
    metrics: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the timepoint into a JSON-safe payload."""

        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "visit_order": self.visit_order,
            "scan_timestamp": self.scan_timestamp,
            "image": self.image,
            "metrics": dict(self.metrics),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class StructuralChange:
    """Change summary for one follow-up timepoint."""

    session_id: str | None
    visit_order: int
    delta_from_baseline: dict[str, float | None]
    percent_change_from_baseline: dict[str, float | None]
    delta_from_previous: dict[str, float | None]
    percent_change_from_previous: dict[str, float | None]

    def to_payload(self) -> dict[str, Any]:
        """Serialize the change summary into a JSON-safe payload."""

        return {
            "session_id": self.session_id,
            "visit_order": self.visit_order,
            "delta_from_baseline": dict(self.delta_from_baseline),
            "percent_change_from_baseline": dict(self.percent_change_from_baseline),
            "delta_from_previous": dict(self.delta_from_previous),
            "percent_change_from_previous": dict(self.percent_change_from_previous),
        }


@dataclass(slots=True)
class StructuralLongitudinalSummary:
    """Subject-level longitudinal structural report."""

    subject_id: str
    dataset: str
    dataset_type: str
    timepoint_count: int
    timepoints: list[StructuralTimepoint]
    changes: list[StructuralChange]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the report into a JSON-safe payload."""

        return {
            "subject_id": self.subject_id,
            "dataset": self.dataset,
            "dataset_type": self.dataset_type,
            "timepoint_count": self.timepoint_count,
            "timepoints": [timepoint.to_payload() for timepoint in self.timepoints],
            "changes": [change.to_payload() for change in self.changes],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def _extract_visit_number(session_id: str | None) -> int | None:
    """Extract an OASIS MR visit number when available."""

    if not session_id:
        return None
    match = SESSION_VISIT_PATTERN.search(session_id)
    if match is None:
        return None
    return int(match.group(1))


def _safe_timestamp(raw_value: Any) -> str | None:
    """Normalize an optional timestamp-like value."""

    return canonicalize_optional_string(raw_value)


def _prepare_subject_frame(frame: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """Filter and order one subject's OASIS manifest rows."""

    subject_frame = frame.loc[frame["subject_id"].astype(str) == subject_id].copy().reset_index(drop=True)
    if subject_frame.empty:
        raise LongitudinalStructuralError(f"No OASIS rows found for subject_id={subject_id!r}.")

    subject_frame["scan_timestamp"] = subject_frame["scan_timestamp"].apply(_safe_timestamp)
    subject_frame["scan_timestamp_parsed"] = pd.to_datetime(subject_frame["scan_timestamp"], errors="coerce")
    subject_frame["meta_payload"] = subject_frame["meta"].apply(parse_manifest_meta)
    subject_frame["session_id"] = subject_frame["meta_payload"].apply(
        lambda payload: canonicalize_optional_string(payload.get("session_id"))
    )
    subject_frame["session_id"] = subject_frame.apply(
        lambda row: row["session_id"] or Path(row["image"]).stem,
        axis=1,
    )
    subject_frame["session_visit_number"] = subject_frame["session_id"].apply(_extract_visit_number)
    subject_frame = subject_frame.assign(
        _timestamp_missing=subject_frame["scan_timestamp_parsed"].isna().astype(int),
        _timestamp_value=subject_frame["scan_timestamp_parsed"].fillna(pd.Timestamp.max),
        _visit_missing=subject_frame["session_visit_number"].isna().astype(int),
        _visit_value=subject_frame["session_visit_number"].fillna(10**9).astype(int),
        _stable_index=subject_frame.index,
    )
    subject_frame = subject_frame.sort_values(
        [
            "_timestamp_missing",
            "_timestamp_value",
            "_visit_missing",
            "_visit_value",
            "session_id",
            "_stable_index",
        ]
    ).reset_index(drop=True)
    subject_frame["visit_order"] = subject_frame.index + 1
    return subject_frame


def _timepoint_from_analysis(
    analysis: VolumetricAnalysisResult,
    *,
    subject_id: str,
    session_id: str | None,
    visit_order: int,
    scan_timestamp: str | None,
) -> StructuralTimepoint:
    """Convert a volumetric analysis into a longitudinal timepoint."""

    return StructuralTimepoint(
        subject_id=subject_id,
        session_id=session_id,
        visit_order=visit_order,
        scan_timestamp=scan_timestamp,
        image=str(analysis.image_path),
        metrics=summarize_volumetrics(analysis.measurements),
        warnings=list(analysis.warnings),
    )


def _percent_change(current_value: float | None, reference_value: float | None) -> float | None:
    """Compute percent change while guarding against missing or zero references."""

    if current_value is None or reference_value in {None, 0.0}:
        return None
    return float(((current_value - reference_value) / reference_value) * 100.0)


def _metric_delta(current: dict[str, float], reference: dict[str, float]) -> dict[str, float | None]:
    """Compute metric deltas for the standard longitudinal structural metrics."""

    delta: dict[str, float | None] = {}
    for metric_name in CHANGE_METRICS:
        current_value = current.get(metric_name)
        reference_value = reference.get(metric_name)
        delta[metric_name] = None if current_value is None or reference_value is None else float(current_value - reference_value)
    return delta


def _metric_percent_change(current: dict[str, float], reference: dict[str, float]) -> dict[str, float | None]:
    """Compute metric percent changes for the standard longitudinal structural metrics."""

    return {
        metric_name: _percent_change(current.get(metric_name), reference.get(metric_name))
        for metric_name in CHANGE_METRICS
    }


def build_structural_changes(timepoints: list[StructuralTimepoint]) -> list[StructuralChange]:
    """Build follow-up change summaries relative to baseline and prior timepoint."""

    if len(timepoints) < 2:
        return []

    baseline = timepoints[0]
    changes: list[StructuralChange] = []
    for index in range(1, len(timepoints)):
        current = timepoints[index]
        previous = timepoints[index - 1]
        changes.append(
            StructuralChange(
                session_id=current.session_id,
                visit_order=current.visit_order,
                delta_from_baseline=_metric_delta(current.metrics, baseline.metrics),
                percent_change_from_baseline=_metric_percent_change(current.metrics, baseline.metrics),
                delta_from_previous=_metric_delta(current.metrics, previous.metrics),
                percent_change_from_previous=_metric_percent_change(current.metrics, previous.metrics),
            )
        )
    return changes


def build_oasis_structural_longitudinal_summary(
    subject_id: str,
    *,
    settings: AppSettings | None = None,
    split: str | None = None,
    manifest_path: str | Path | None = None,
    max_timepoints: int | None = None,
) -> StructuralLongitudinalSummary:
    """Analyze one OASIS subject across available visits and summarize structural change."""

    resolved_settings = settings or get_app_settings()
    frame = load_oasis_manifest(
        resolved_settings,
        split=split,
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
    )
    subject_frame = _prepare_subject_frame(frame, subject_id)
    if max_timepoints is not None:
        subject_frame = subject_frame.head(max_timepoints).reset_index(drop=True)

    timepoints: list[StructuralTimepoint] = []
    warnings: list[str] = []
    for row in subject_frame.itertuples(index=False):
        analysis = analyze_mri_volume(
            row.image,
            dataset="oasis1",
            dataset_type="3d_volumes",
            subject_id=subject_id,
            session_id=row.session_id,
            scan_timestamp=row.scan_timestamp,
        )
        timepoints.append(
            _timepoint_from_analysis(
                analysis,
                subject_id=subject_id,
                session_id=row.session_id,
                visit_order=int(row.visit_order),
                scan_timestamp=row.scan_timestamp,
            )
        )
        warnings.extend(analysis.warnings)

    if len(timepoints) < 2:
        warnings.append("Only one timepoint is available, so no longitudinal change can be computed yet.")

    return StructuralLongitudinalSummary(
        subject_id=subject_id,
        dataset="oasis1",
        dataset_type="3d_volumes",
        timepoint_count=len(timepoints),
        timepoints=timepoints,
        changes=build_structural_changes(timepoints),
        warnings=warnings,
        notes=[
            "Longitudinal structural changes use foreground-proxy measurements, not region-specific segmentation.",
            "All visits for a subject should remain in the same model split to avoid leakage.",
            "This output is for decision support and research workflow auditing, not diagnosis.",
        ],
    )


def save_structural_longitudinal_report(
    summary: StructuralLongitudinalSummary,
    *,
    settings: AppSettings | None = None,
    file_stem: str | None = None,
) -> Path:
    """Save a subject-level structural longitudinal report."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "longitudinal_structural")
    resolved_stem = file_stem or f"{summary.subject_id}_structural_longitudinal"
    output_path = output_root / f"{resolved_stem}.json"
    output_path.write_text(json.dumps(summary.to_payload(), indent=2), encoding="utf-8")
    return output_path
