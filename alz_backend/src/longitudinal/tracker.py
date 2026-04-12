"""Interpretable longitudinal scan-history tracking.

This module computes timeline-ready summaries from supplied volumetric features
and model probabilities. Trend thresholds are configurable research defaults,
not clinical criteria.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SESSION_VISIT_PATTERN = re.compile(r"_MR(\d+)\b", re.IGNORECASE)


class LongitudinalTrackingError(ValueError):
    """Raised when a longitudinal report cannot be built safely."""


@dataclass(slots=True)
class LongitudinalRecord:
    """One scan-history timepoint grouped by subject."""

    subject_id: str
    session_id: str | None
    visit_order: int | None
    summary_label: str | None = None
    scan_timestamp: str | None = None
    source_path: str | None = None
    dataset: str | None = None
    volumetric_features: dict[str, float] = field(default_factory=dict)
    model_probabilities: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Serialize one timepoint."""

        return asdict(self)


ScanHistoryRecord = LongitudinalRecord


@dataclass(slots=True, frozen=True)
class TrendFeatureConfig:
    """Configurable trend feature and classification thresholds."""

    feature_name: str
    source: str
    decline_direction: str
    normalization: str = "percent_from_baseline"
    stable_slope_threshold: float = 1.0
    rapid_slope_threshold: float = 5.0
    display_name: str | None = None
    unit: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate trend config values."""

        if self.source not in {"volumetric", "model_probability"}:
            raise LongitudinalTrackingError(f"Unsupported trend source {self.source!r}.")
        if self.decline_direction not in {"increase", "decrease"}:
            raise LongitudinalTrackingError(f"Unsupported decline direction {self.decline_direction!r}.")
        if self.normalization not in {"raw", "percent_from_baseline"}:
            raise LongitudinalTrackingError(f"Unsupported normalization {self.normalization!r}.")
        if self.stable_slope_threshold < 0 or self.rapid_slope_threshold < self.stable_slope_threshold:
            raise LongitudinalTrackingError("Trend thresholds must be non-negative and rapid >= stable.")

    def to_payload(self) -> dict[str, Any]:
        """Serialize the config."""

        return asdict(self)


@dataclass(slots=True, frozen=True)
class IntervalChange:
    """Feature change between consecutive timepoints."""

    feature_name: str
    source: str
    from_session_id: str | None
    to_session_id: str | None
    from_visit_order: int | None
    to_visit_order: int | None
    from_value: float
    to_value: float
    absolute_change: float
    percent_change: float | None
    elapsed_days: float | None

    def to_payload(self) -> dict[str, Any]:
        """Serialize the interval change."""

        return asdict(self)


@dataclass(slots=True, frozen=True)
class TrendSummary:
    """Slope and classification for one feature."""

    feature_name: str
    source: str
    decline_direction: str
    normalization: str
    trend_classification: str
    slope_per_visit: float | None
    normalized_slope_per_visit: float | None
    normalized_slope_unit: str
    adverse_slope: float | None
    baseline_value: float | None
    latest_value: float | None
    timepoint_count: int
    thresholds: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the summary."""

        return asdict(self)


@dataclass(slots=True, frozen=True)
class FeatureDriftSummary:
    """Baseline-to-latest drift summary for one configured feature."""

    feature_name: str
    source: str
    classification: str
    baseline_value: float | None
    latest_value: float | None
    absolute_change: float | None
    percent_change: float | None
    adverse_change: float | None
    available_timepoints: int

    def to_payload(self) -> dict[str, Any]:
        """Serialize the feature drift."""

        return asdict(self)


@dataclass(slots=True, frozen=True)
class ProgressionOverview:
    """High-level longitudinal progression overview for one subject."""

    subject_id: str
    baseline_session_id: str | None
    latest_session_id: str | None
    baseline_timestamp: str | None
    latest_timestamp: str | None
    overall_trend_classification: str
    rapid_decline_features: list[str]
    mild_decline_features: list[str]
    stable_features: list[str]
    insufficient_data_features: list[str]
    review_recommended: bool
    narrative: str
    feature_drifts: list[FeatureDriftSummary] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the progression overview."""

        return {
            "subject_id": self.subject_id,
            "baseline_session_id": self.baseline_session_id,
            "latest_session_id": self.latest_session_id,
            "baseline_timestamp": self.baseline_timestamp,
            "latest_timestamp": self.latest_timestamp,
            "overall_trend_classification": self.overall_trend_classification,
            "rapid_decline_features": list(self.rapid_decline_features),
            "mild_decline_features": list(self.mild_decline_features),
            "stable_features": list(self.stable_features),
            "insufficient_data_features": list(self.insufficient_data_features),
            "review_recommended": self.review_recommended,
            "narrative": self.narrative,
            "feature_drifts": [item.to_payload() for item in self.feature_drifts],
        }


@dataclass(slots=True, frozen=True)
class ProgressionAlert:
    """Progression alert from a trend summary."""

    severity: str
    feature_name: str
    trend_classification: str
    message: str

    def to_payload(self) -> dict[str, Any]:
        """Serialize the alert."""

        return asdict(self)


@dataclass(slots=True)
class LongitudinalReport:
    """Frontend-ready subject-level longitudinal report."""

    subject_id: str
    timepoint_count: int
    timepoints: list[LongitudinalRecord]
    interval_changes: list[IntervalChange]
    trend_summaries: list[TrendSummary]
    alerts: list[ProgressionAlert]
    timeline: list[dict[str, Any]]
    progression_overview: ProgressionOverview
    feature_configs: list[TrendFeatureConfig]
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    report_type: str = "longitudinal_tracking"

    def to_payload(self) -> dict[str, Any]:
        """Serialize the report."""

        return {
            "report_type": self.report_type,
            "subject_id": self.subject_id,
            "generated_at": self.generated_at,
            "timepoint_count": self.timepoint_count,
            "timepoints": [record.to_payload() for record in self.timepoints],
            "timeline": list(self.timeline),
            "interval_changes": [change.to_payload() for change in self.interval_changes],
            "trend_summaries": [summary.to_payload() for summary in self.trend_summaries],
            "alerts": [alert.to_payload() for alert in self.alerts],
            "feature_configs": [config.to_payload() for config in self.feature_configs],
            "progression_overview": self.progression_overview.to_payload(),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


def _parse_timestamp(raw_value: str | None) -> datetime | None:
    """Parse an optional timestamp."""

    if not raw_value or not str(raw_value).strip():
        return None
    try:
        return datetime.fromisoformat(str(raw_value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_visit_number(session_id: str | None) -> int | None:
    """Extract visit number from OASIS-like session IDs."""

    if not session_id:
        return None
    match = SESSION_VISIT_PATTERN.search(session_id)
    return int(match.group(1)) if match else None


def _sort_key(index_and_record: tuple[int, LongitudinalRecord]) -> tuple[int, object, int, int, str]:
    """Sort by timestamp, visit number, session ID, then input order."""

    index, record = index_and_record
    timestamp = _parse_timestamp(record.scan_timestamp)
    visit_order = record.visit_order if record.visit_order is not None else _extract_visit_number(record.session_id)
    return (
        0 if timestamp is not None else 1,
        timestamp or datetime.max,
        0 if visit_order is not None else 1,
        int(visit_order) if visit_order is not None else 10**9,
        record.session_id or f"input_{index:06d}",
    )


def sort_records_by_visit(records: list[LongitudinalRecord]) -> list[LongitudinalRecord]:
    """Return records ordered for visit-to-visit comparison."""

    return [record for _, record in sorted(enumerate(records), key=_sort_key)]


def group_records_by_subject(records: list[LongitudinalRecord]) -> dict[str, list[LongitudinalRecord]]:
    """Group scan-history records by subject ID."""

    grouped: dict[str, list[LongitudinalRecord]] = {}
    for record in records:
        if not record.subject_id:
            raise LongitudinalTrackingError("Every longitudinal record must include subject_id.")
        grouped.setdefault(record.subject_id, []).append(record)
    return {subject_id: sort_records_by_visit(subject_records) for subject_id, subject_records in grouped.items()}


def _get_feature_value(record: LongitudinalRecord, config: TrendFeatureConfig) -> float | None:
    """Read a configured feature value from one record."""

    source = record.volumetric_features if config.source == "volumetric" else record.model_probabilities
    value = source.get(config.feature_name)
    try:
        numeric = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return numeric if numeric is not None and math.isfinite(numeric) else None


def _percent_change(current_value: float, reference_value: float) -> float | None:
    """Compute percent change while guarding against zero references."""

    if reference_value == 0.0:
        return None
    return float(((current_value - reference_value) / reference_value) * 100.0)


def _elapsed_days(previous: LongitudinalRecord, current: LongitudinalRecord) -> float | None:
    """Compute elapsed days between records when timestamps exist."""

    previous_timestamp = _parse_timestamp(previous.scan_timestamp)
    current_timestamp = _parse_timestamp(current.scan_timestamp)
    if previous_timestamp is None or current_timestamp is None:
        return None
    return float((current_timestamp - previous_timestamp).total_seconds() / 86400.0)


def build_interval_changes(
    records: list[LongitudinalRecord],
    feature_configs: list[TrendFeatureConfig],
) -> list[IntervalChange]:
    """Compute interval changes for consecutive visits."""

    ordered_records = sort_records_by_visit(records)
    changes: list[IntervalChange] = []
    for previous, current in zip(ordered_records, ordered_records[1:]):
        for config in feature_configs:
            previous_value = _get_feature_value(previous, config)
            current_value = _get_feature_value(current, config)
            if previous_value is None or current_value is None:
                continue
            changes.append(
                IntervalChange(
                    feature_name=config.feature_name,
                    source=config.source,
                    from_session_id=previous.session_id,
                    to_session_id=current.session_id,
                    from_visit_order=previous.visit_order,
                    to_visit_order=current.visit_order,
                    from_value=previous_value,
                    to_value=current_value,
                    absolute_change=float(current_value - previous_value),
                    percent_change=_percent_change(current_value, previous_value),
                    elapsed_days=_elapsed_days(previous, current),
                )
            )
    return changes


def _slope(values: list[float]) -> float | None:
    """Compute ordinary least-squares slope over visit index."""

    if len(values) < 2:
        return None
    x_values = [float(index) for index in range(len(values))]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(values) / len(values)
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    if denominator == 0.0:
        return None
    numerator = sum((x_value - x_mean) * (value - y_mean) for x_value, value in zip(x_values, values))
    return float(numerator / denominator)


def _normalized_values(values: list[float], config: TrendFeatureConfig) -> tuple[list[float] | None, str, list[str]]:
    """Normalize values before trend classification."""

    if config.normalization == "raw":
        return values, config.unit or "value per visit", []
    baseline = values[0]
    if baseline == 0.0:
        return (
            None,
            config.unit or "percent per visit",
            [f"Feature {config.feature_name!r} has a zero baseline; percent trend is unavailable."],
        )
    return [((value - baseline) / baseline) * 100.0 for value in values], config.unit or "percent per visit", []


def _classify_trend(adverse_slope: float | None, config: TrendFeatureConfig) -> str:
    """Classify trend using configurable research thresholds."""

    if adverse_slope is None:
        return "insufficient_data"
    if adverse_slope <= config.stable_slope_threshold:
        return "stable"
    if adverse_slope >= config.rapid_slope_threshold:
        return "rapid_decline"
    return "mild_decline"


def build_trend_summaries(
    records: list[LongitudinalRecord],
    feature_configs: list[TrendFeatureConfig],
) -> list[TrendSummary]:
    """Compute slope summaries for configured features."""

    ordered_records = sort_records_by_visit(records)
    summaries: list[TrendSummary] = []
    for config in feature_configs:
        valid_values = [value for record in ordered_records if (value := _get_feature_value(record, config)) is not None]
        warnings: list[str] = []
        if len(valid_values) < 2:
            warnings.append(f"Feature {config.feature_name!r} has fewer than two usable values.")
            summaries.append(
                TrendSummary(
                    feature_name=config.feature_name,
                    source=config.source,
                    decline_direction=config.decline_direction,
                    normalization=config.normalization,
                    trend_classification="insufficient_data",
                    slope_per_visit=None,
                    normalized_slope_per_visit=None,
                    normalized_slope_unit=config.unit or "value per visit",
                    adverse_slope=None,
                    baseline_value=valid_values[0] if valid_values else None,
                    latest_value=valid_values[-1] if valid_values else None,
                    timepoint_count=len(valid_values),
                    thresholds={
                        "stable_slope_threshold": config.stable_slope_threshold,
                        "rapid_slope_threshold": config.rapid_slope_threshold,
                    },
                    warnings=warnings,
                )
            )
            continue

        raw_slope = _slope(valid_values)
        normalized, normalized_unit, normalization_warnings = _normalized_values(valid_values, config)
        warnings.extend(normalization_warnings)
        normalized_slope = _slope(normalized) if normalized is not None else None
        adverse_slope = None
        if normalized_slope is not None:
            adverse_slope = normalized_slope if config.decline_direction == "increase" else -normalized_slope
        summaries.append(
            TrendSummary(
                feature_name=config.feature_name,
                source=config.source,
                decline_direction=config.decline_direction,
                normalization=config.normalization,
                trend_classification=_classify_trend(adverse_slope, config),
                slope_per_visit=raw_slope,
                normalized_slope_per_visit=normalized_slope,
                normalized_slope_unit=normalized_unit,
                adverse_slope=adverse_slope,
                baseline_value=valid_values[0],
                latest_value=valid_values[-1],
                timepoint_count=len(valid_values),
                thresholds={
                    "stable_slope_threshold": config.stable_slope_threshold,
                    "rapid_slope_threshold": config.rapid_slope_threshold,
                },
                warnings=warnings,
            )
        )
    return summaries


def build_progression_alerts(summaries: list[TrendSummary]) -> list[ProgressionAlert]:
    """Create interpretable alerts for mild or rapid decline trends."""

    alerts: list[ProgressionAlert] = []
    for summary in summaries:
        if summary.trend_classification not in {"mild_decline", "rapid_decline"}:
            continue
        severity = "high" if summary.trend_classification == "rapid_decline" else "moderate"
        alerts.append(
            ProgressionAlert(
                severity=severity,
                feature_name=summary.feature_name,
                trend_classification=summary.trend_classification,
                message=(
                    f"{summary.feature_name} shows a {summary.trend_classification.replace('_', ' ')} "
                    "trend under configured research thresholds. Review source scans and metrics before acting."
                ),
            )
        )
    return alerts


def _baseline_latest_value(
    records: list[LongitudinalRecord],
    config: TrendFeatureConfig,
) -> tuple[float | None, float | None, int]:
    """Return baseline/latest values for a configured feature."""

    values = [_get_feature_value(record, config) for record in records]
    available = [value for value in values if value is not None]
    return (
        available[0] if available else None,
        available[-1] if available else None,
        len(available),
    )


def _overall_trend_classification(summaries: list[TrendSummary]) -> str:
    """Fuse feature-level classifications into an interpretable overall class."""

    classes = {summary.trend_classification for summary in summaries}
    if "rapid_decline" in classes:
        return "rapid_decline"
    if "mild_decline" in classes:
        return "mild_decline"
    if "stable" in classes:
        return "stable"
    return "insufficient_data"


def build_progression_overview(
    records: list[LongitudinalRecord],
    feature_configs: list[TrendFeatureConfig],
    summaries: list[TrendSummary],
) -> ProgressionOverview:
    """Build a fused, timeline-aware overview from feature trends."""

    ordered_records = sort_records_by_visit(records)
    baseline_record = ordered_records[0]
    latest_record = ordered_records[-1]
    summary_by_feature = {summary.feature_name: summary for summary in summaries}
    feature_drifts: list[FeatureDriftSummary] = []
    rapid_features: list[str] = []
    mild_features: list[str] = []
    stable_features: list[str] = []
    insufficient_features: list[str] = []

    for config in feature_configs:
        summary = summary_by_feature.get(config.feature_name)
        baseline_value, latest_value, available_timepoints = _baseline_latest_value(ordered_records, config)
        absolute_change = (
            None
            if baseline_value is None or latest_value is None
            else float(latest_value - baseline_value)
        )
        percent_change = (
            None
            if baseline_value is None or latest_value is None
            else _percent_change(latest_value, baseline_value)
        )
        adverse_change = None
        if absolute_change is not None:
            adverse_change = absolute_change if config.decline_direction == "increase" else -absolute_change
        classification = "insufficient_data" if summary is None else summary.trend_classification
        if classification == "rapid_decline":
            rapid_features.append(config.feature_name)
        elif classification == "mild_decline":
            mild_features.append(config.feature_name)
        elif classification == "stable":
            stable_features.append(config.feature_name)
        else:
            insufficient_features.append(config.feature_name)
        feature_drifts.append(
            FeatureDriftSummary(
                feature_name=config.feature_name,
                source=config.source,
                classification=classification,
                baseline_value=baseline_value,
                latest_value=latest_value,
                absolute_change=absolute_change,
                percent_change=percent_change,
                adverse_change=adverse_change,
                available_timepoints=available_timepoints,
            )
        )

    overall = _overall_trend_classification(summaries)
    review_recommended = bool(rapid_features or mild_features or insufficient_features)
    if overall == "rapid_decline":
        narrative = (
            "Multiple longitudinal signals suggest a rapid adverse trend under the configured research thresholds. "
            "Review source scans, preprocessing quality, and volumetric/model inputs before using this output."
        )
    elif overall == "mild_decline":
        narrative = (
            "Longitudinal signals suggest a mild adverse trend under the configured research thresholds. "
            "This is a decision-support summary and should be reviewed in context."
        )
    elif overall == "stable":
        narrative = (
            "Tracked longitudinal signals appear stable under the configured research thresholds. "
            "Stability here does not rule out clinically meaningful change."
        )
    else:
        narrative = (
            "Longitudinal inputs were insufficient for a reliable overall trend classification. "
            "Missing timestamps or sparse features may limit interpretation."
        )
    return ProgressionOverview(
        subject_id=baseline_record.subject_id,
        baseline_session_id=baseline_record.session_id,
        latest_session_id=latest_record.session_id,
        baseline_timestamp=baseline_record.scan_timestamp,
        latest_timestamp=latest_record.scan_timestamp,
        overall_trend_classification=overall,
        rapid_decline_features=rapid_features,
        mild_decline_features=mild_features,
        stable_features=stable_features,
        insufficient_data_features=insufficient_features,
        review_recommended=review_recommended,
        narrative=narrative,
        feature_drifts=feature_drifts,
    )


def build_timeline(records: list[LongitudinalRecord]) -> list[dict[str, Any]]:
    """Build frontend-ready timeline points."""

    ordered_records = sort_records_by_visit(records)
    parsed_timestamps = [_parse_timestamp(record.scan_timestamp) for record in ordered_records]
    baseline_timestamp = next((timestamp for timestamp in parsed_timestamps if timestamp is not None), None)
    timeline: list[dict[str, Any]] = []
    for index, record in enumerate(ordered_records):
        timestamp = parsed_timestamps[index]
        elapsed_days = None
        if timestamp is not None and baseline_timestamp is not None:
            elapsed_days = float((timestamp - baseline_timestamp).total_seconds() / 86400.0)
        timeline.append(
            {
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "visit_order": record.visit_order or index + 1,
                "scan_timestamp": record.scan_timestamp,
                "timestamp_available": timestamp is not None,
                "elapsed_days_from_baseline": elapsed_days,
                "source_path": record.source_path,
                "dataset": record.dataset,
                "summary_label": record.summary_label,
                "volumetric_features": dict(record.volumetric_features),
                "model_probabilities": dict(record.model_probabilities),
            }
        )
    return timeline


def default_feature_configs_for_records(
    records: list[LongitudinalRecord],
    *,
    stable_volume_percent_per_visit: float = 1.0,
    rapid_volume_percent_per_visit: float = 5.0,
    stable_probability_per_visit: float = 0.02,
    rapid_probability_per_visit: float = 0.10,
) -> list[TrendFeatureConfig]:
    """Infer trend-feature configs from supplied features.

    Defaults are research placeholders, not clinical thresholds.
    """

    volumetric_features = sorted({key for record in records for key in record.volumetric_features})
    probability_features = sorted({key for record in records for key in record.model_probabilities})
    configs = [
        TrendFeatureConfig(
            feature_name=feature_name,
            source="volumetric",
            decline_direction="decrease",
            normalization="percent_from_baseline",
            stable_slope_threshold=stable_volume_percent_per_visit,
            rapid_slope_threshold=rapid_volume_percent_per_visit,
            unit="percent per visit",
            notes="Research default: decreasing structural volume is treated as adverse.",
        )
        for feature_name in volumetric_features
    ]
    configs.extend(
        TrendFeatureConfig(
            feature_name=feature_name,
            source="model_probability",
            decline_direction="increase",
            normalization="raw",
            stable_slope_threshold=stable_probability_per_visit,
            rapid_slope_threshold=rapid_probability_per_visit,
            unit="probability per visit",
            notes="Research default: increasing AD-like probability is treated as adverse.",
        )
        for feature_name in probability_features
    )
    return configs


def build_longitudinal_report(
    records: list[LongitudinalRecord],
    *,
    subject_id: str | None = None,
    feature_configs: list[TrendFeatureConfig] | None = None,
) -> LongitudinalReport:
    """Build a timeline-ready longitudinal report for one subject."""

    if not records:
        raise LongitudinalTrackingError("At least one scan-history record is required.")
    grouped = group_records_by_subject(records)
    if subject_id is None:
        if len(grouped) != 1:
            raise LongitudinalTrackingError("subject_id is required when records contain multiple subjects.")
        subject_id = next(iter(grouped))
    if subject_id not in grouped:
        raise LongitudinalTrackingError(f"No records found for subject_id={subject_id!r}.")

    ordered_records = grouped[subject_id]
    resolved_configs = feature_configs or default_feature_configs_for_records(ordered_records)
    warnings: list[str] = []
    if len(ordered_records) < 2:
        warnings.append("Only one timepoint is available, so trend slopes cannot be computed.")
    if any(_parse_timestamp(record.scan_timestamp) is None for record in ordered_records):
        warnings.append("One or more timestamps are missing or unparseable; ordering falls back to visit/session/input order.")
    if not resolved_configs:
        warnings.append("No volumetric or model-probability trend inputs were available.")

    trends = build_trend_summaries(ordered_records, resolved_configs)
    warnings.extend(warning for summary in trends for warning in summary.warnings)
    progression_overview = build_progression_overview(ordered_records, resolved_configs, trends)
    return LongitudinalReport(
        subject_id=subject_id,
        timepoint_count=len(ordered_records),
        timepoints=ordered_records,
        interval_changes=build_interval_changes(ordered_records, resolved_configs),
        trend_summaries=trends,
        alerts=build_progression_alerts(trends),
        timeline=build_timeline(ordered_records),
        progression_overview=progression_overview,
        feature_configs=resolved_configs,
        warnings=warnings,
        limitations=[
            "Trend categories use configurable research thresholds and are not clinical diagnostic criteria.",
            "Missing timestamps are handled gracefully by falling back to visit/session/input order.",
            "Volumetric trends are only as reliable as the supplied measurements or external segmentation outputs.",
            "Model-probability trends are decision-support signals and should not be interpreted as diagnosis.",
        ],
    )
