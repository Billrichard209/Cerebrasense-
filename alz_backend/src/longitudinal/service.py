"""Service helpers for longitudinal report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.storage import LongitudinalMetadataRecord, persist_longitudinal_record
from src.utils.io_utils import ensure_directory

from .tracker import (
    LongitudinalRecord,
    LongitudinalReport,
    LongitudinalTrackingError,
    TrendFeatureConfig,
    build_longitudinal_report,
    sort_records_by_visit,
)

VOL_PREFIX = "vol__"
PROB_PREFIX = "prob__"


def build_progression_summary(records: list[LongitudinalRecord]) -> dict[str, object]:
    """Return a compact progression summary suitable for API endpoints."""

    ordered_records = sort_records_by_visit(records)
    report = build_longitudinal_report(ordered_records) if ordered_records else None
    return {
        "subject_id": ordered_records[0].subject_id if ordered_records else None,
        "visit_count": len(ordered_records),
        "sessions": [record.session_id for record in ordered_records],
        "alerts": [] if report is None else [alert.to_payload() for alert in report.alerts],
        "overall_trend_classification": None
        if report is None
        else report.progression_overview.overall_trend_classification,
        "review_recommended": False if report is None else report.progression_overview.review_recommended,
        "narrative": None if report is None else report.progression_overview.narrative,
    }


def _is_missing(value: Any) -> bool:
    """Return whether a tabular value should be treated as missing."""

    return value is None or bool(pd.isna(value))


def _optional_str(value: Any) -> str | None:
    """Normalize optional string values from CSV rows."""

    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    """Normalize optional integer values from CSV rows."""

    if _is_missing(value):
        return None
    return int(value)


def _numeric_feature_dict(row: pd.Series, *, prefix: str) -> dict[str, float]:
    """Extract numeric feature columns with a given prefix."""

    features: dict[str, float] = {}
    for column_name, value in row.items():
        if not str(column_name).startswith(prefix) or _is_missing(value):
            continue
        try:
            features[str(column_name)[len(prefix) :]] = float(value)
        except (TypeError, ValueError):
            continue
    return features


def records_from_csv(csv_path: str | Path, *, subject_id: str | None = None) -> list[LongitudinalRecord]:
    """Load scan-history records from CSV.

    Feature columns are prefixed as ``vol__<name>`` or ``prob__<name>``.
    """

    frame = pd.read_csv(csv_path)
    if "subject_id" not in frame.columns:
        raise LongitudinalTrackingError("Longitudinal CSV must include a subject_id column.")
    if subject_id is not None:
        frame = frame.loc[frame["subject_id"].astype(str) == subject_id].copy()
    if frame.empty:
        raise LongitudinalTrackingError(f"No longitudinal rows found in {csv_path} for subject_id={subject_id!r}.")

    records: list[LongitudinalRecord] = []
    reserved = {
        "subject_id",
        "session_id",
        "visit_order",
        "scan_timestamp",
        "source_path",
        "image",
        "dataset",
        "summary_label",
    }
    for row in frame.to_dict(orient="records"):
        series = pd.Series(row)
        metadata = {
            key: value
            for key, value in row.items()
            if not str(key).startswith((VOL_PREFIX, PROB_PREFIX)) and key not in reserved and not _is_missing(value)
        }
        records.append(
            LongitudinalRecord(
                subject_id=str(row["subject_id"]).strip(),
                session_id=_optional_str(row.get("session_id")),
                visit_order=_optional_int(row.get("visit_order")),
                summary_label=_optional_str(row.get("summary_label")),
                scan_timestamp=_optional_str(row.get("scan_timestamp")),
                source_path=_optional_str(row.get("source_path")) or _optional_str(row.get("image")),
                dataset=_optional_str(row.get("dataset")),
                volumetric_features=_numeric_feature_dict(series, prefix=VOL_PREFIX),
                model_probabilities=_numeric_feature_dict(series, prefix=PROB_PREFIX),
                metadata=metadata,
            )
        )
    return records


def records_from_structural_summary_payload(payload: dict[str, Any]) -> list[LongitudinalRecord]:
    """Convert an existing structural longitudinal payload into generic records."""

    subject_id = str(payload.get("subject_id") or "").strip()
    if not subject_id:
        raise LongitudinalTrackingError("Structural summary payload must include subject_id.")
    records: list[LongitudinalRecord] = []
    for timepoint in payload.get("timepoints", []):
        records.append(
            LongitudinalRecord(
                subject_id=subject_id,
                session_id=_optional_str(timepoint.get("session_id")),
                visit_order=_optional_int(timepoint.get("visit_order")),
                scan_timestamp=_optional_str(timepoint.get("scan_timestamp")),
                source_path=_optional_str(timepoint.get("image")),
                dataset=str(payload.get("dataset") or "oasis1"),
                volumetric_features={
                    str(key): float(value)
                    for key, value in dict(timepoint.get("metrics", {})).items()
                    if value is not None
                },
                metadata={"source_report_type": payload.get("dataset_type", "structural_summary")},
            )
        )
    return records


def records_from_structural_summary_json(path: str | Path) -> list[LongitudinalRecord]:
    """Load records from a saved structural longitudinal JSON report."""

    return records_from_structural_summary_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def _feature_config_from_payload(payload: dict[str, Any]) -> TrendFeatureConfig:
    """Build a trend-feature config from JSON."""

    return TrendFeatureConfig(
        feature_name=str(payload["feature_name"]),
        source=str(payload["source"]),
        decline_direction=str(payload["decline_direction"]),
        normalization=str(payload.get("normalization", "percent_from_baseline")),
        stable_slope_threshold=float(payload.get("stable_slope_threshold", 1.0)),
        rapid_slope_threshold=float(payload.get("rapid_slope_threshold", 5.0)),
        display_name=_optional_str(payload.get("display_name")),
        unit=_optional_str(payload.get("unit")),
        notes=_optional_str(payload.get("notes")),
    )


def load_feature_configs(path: str | Path | None) -> list[TrendFeatureConfig] | None:
    """Load optional trend-feature configs from JSON."""

    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("features", [])
    if not isinstance(payload, list):
        raise LongitudinalTrackingError("Feature config JSON must be a list or an object with a features list.")
    return [_feature_config_from_payload(item) for item in payload]


def save_longitudinal_report(
    report: LongitudinalReport,
    *,
    settings: AppSettings | None = None,
    file_stem: str | None = None,
) -> Path:
    """Save a timeline-ready longitudinal report."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "longitudinal")
    safe_stem = (file_stem or f"{report.subject_id}_longitudinal_report").replace(" ", "_").replace("/", "_")
    output_path = output_root / f"{safe_stem}.json"
    payload = report.to_payload()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    persist_longitudinal_record(
        LongitudinalMetadataRecord(
            subject_id=report.subject_id,
            report_type=report.report_type,
            output_path=str(output_path),
            payload=payload,
        ),
        settings=resolved_settings,
    )
    return output_path


def build_and_save_longitudinal_report(
    records: list[LongitudinalRecord],
    *,
    subject_id: str | None = None,
    feature_configs: list[TrendFeatureConfig] | None = None,
    settings: AppSettings | None = None,
    file_stem: str | None = None,
) -> tuple[LongitudinalReport, Path]:
    """Build and save a longitudinal report from prepared records."""

    report = build_longitudinal_report(records, subject_id=subject_id, feature_configs=feature_configs)
    output_path = save_longitudinal_report(report, settings=settings, file_stem=file_stem)
    return report, output_path
