"""Structural metrics report generation from external neuroimaging outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

from .freesurfer import parse_aparc_stats, parse_aseg_stats, parse_global_measures, read_freesurfer_version_from_file
from .schemas import (
    STRUCTURAL_METRICS_LIMITATIONS,
    AsymmetryMetric,
    BrainRegionVolume,
    CorticalThicknessSummary,
    ExternalToolReference,
    GlobalStructuralMeasure,
    HippocampalVolume,
    StructuralReferenceComparison,
    StructuralReferenceRange,
    StructuralMetricsReport,
)


class StructuralMetricsError(ValueError):
    """Raised when structural metrics cannot be generated safely."""


def _hippocampal_volumes(region_volumes: Iterable[BrainRegionVolume]) -> list[HippocampalVolume]:
    """Extract hippocampal volumes from parsed region-volume rows."""

    hippocampal: list[HippocampalVolume] = []
    for volume in region_volumes:
        if "hippocampus" not in volume.region_name.lower():
            continue
        hippocampal.append(
            HippocampalVolume(
                hemisphere=volume.hemisphere or "unknown",
                region_name=volume.region_name,
                value_mm3=volume.value_mm3,
                source_file=volume.source_file,
                method=volume.method,
            )
        )
    return hippocampal


def _asymmetry_index(left_value: float, right_value: float) -> float:
    """Compute a symmetric left/right asymmetry index."""

    denominator = max((left_value + right_value) / 2.0, 1e-8)
    return float((right_value - left_value) / denominator)


def _hippocampal_asymmetry(hippocampal_volumes: list[HippocampalVolume]) -> list[AsymmetryMetric]:
    """Build hippocampal asymmetry when both hemispheres are available."""

    left = next((item for item in hippocampal_volumes if item.hemisphere == "left"), None)
    right = next((item for item in hippocampal_volumes if item.hemisphere == "right"), None)
    if left is None or right is None:
        return []
    return [
        AsymmetryMetric(
            metric_name="hippocampal_asymmetry_index",
            left_value=left.value_mm3,
            right_value=right.value_mm3,
            asymmetry_index=_asymmetry_index(left.value_mm3, right.value_mm3),
            unit="ratio",
            notes="Positive values indicate right volume is larger than left volume.",
        )
    ]


def _feature_key(*parts: str) -> str:
    """Build a stable feature key for downstream fusion."""

    return "_".join(part.lower().replace("-", "_").replace(" ", "_") for part in parts if part)


def build_structural_feature_vector(report: StructuralMetricsReport) -> dict[str, float]:
    """Create a flat feature vector for later fusion with model predictions."""

    features: dict[str, float] = {}
    for volume in report.hippocampal_volumes:
        features[_feature_key(volume.hemisphere, "hippocampus", "volume_mm3")] = volume.value_mm3
    for asymmetry in report.asymmetry_metrics:
        features[_feature_key(asymmetry.metric_name)] = asymmetry.asymmetry_index
    for thickness in report.cortical_thickness:
        features[_feature_key(thickness.hemisphere, thickness.region_name, "thickness_mm")] = thickness.mean_thickness_mm
    for region_volume in report.brain_region_volumes:
        features[_feature_key(region_volume.region_name, "volume_mm3")] = region_volume.value_mm3
    for measure in report.global_measures:
        features[_feature_key(measure.feature_name)] = measure.value
    return features


def load_structural_reference_ranges(path: str | Path) -> list[StructuralReferenceRange]:
    """Load reference-style structural ranges from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("features", [])
    if not isinstance(payload, list):
        raise StructuralMetricsError("Reference ranges JSON must be a list or an object with a features list.")
    reference_ranges: list[StructuralReferenceRange] = []
    for item in payload:
        if not isinstance(item, dict):
            raise StructuralMetricsError("Each structural reference range must be a JSON object.")
        reference_ranges.append(
            StructuralReferenceRange(
                feature_name=str(item["feature_name"]),
                display_name=None if item.get("display_name") is None else str(item["display_name"]),
                min_value=None if item.get("min_value") is None else float(item["min_value"]),
                max_value=None if item.get("max_value") is None else float(item["max_value"]),
                unit=None if item.get("unit") is None else str(item["unit"]),
                notes=None if item.get("notes") is None else str(item["notes"]),
            )
        )
    return reference_ranges


def compare_report_to_reference_ranges(
    report: StructuralMetricsReport,
    reference_ranges: list[StructuralReferenceRange],
) -> list[StructuralReferenceComparison]:
    """Compare parsed structural features against provided reference-style ranges."""

    comparisons: list[StructuralReferenceComparison] = []
    for reference in reference_ranges:
        observed_value = report.feature_vector.get(reference.feature_name)
        if observed_value is None:
            continue
        status = "within_range"
        if reference.min_value is not None and observed_value < reference.min_value:
            status = "low"
        elif reference.max_value is not None and observed_value > reference.max_value:
            status = "high"
        comparisons.append(
            StructuralReferenceComparison(
                feature_name=reference.feature_name,
                display_name=reference.display_name or reference.feature_name,
                observed_value=float(observed_value),
                status=status,
                min_value=reference.min_value,
                max_value=reference.max_value,
                unit=reference.unit,
                notes=reference.notes,
            )
        )
    return comparisons


def _source_files(*paths: str | Path | None) -> tuple[Path, ...]:
    """Return existing optional source paths."""

    return tuple(Path(path) for path in paths if path is not None)


def build_freesurfer_structural_report(
    *,
    subject_id: str | None = None,
    session_id: str | None = None,
    aseg_stats_path: str | Path | None = None,
    lh_aparc_stats_path: str | Path | None = None,
    rh_aparc_stats_path: str | Path | None = None,
) -> StructuralMetricsReport:
    """Build a structural metrics report from available FreeSurfer stats files."""

    source_paths = _source_files(aseg_stats_path, lh_aparc_stats_path, rh_aparc_stats_path)
    if not source_paths:
        raise StructuralMetricsError("At least one external stats file path must be provided.")

    warnings: list[str] = []
    brain_region_volumes: list[BrainRegionVolume] = []
    cortical_thickness: list[CorticalThicknessSummary] = []
    global_measures: list[GlobalStructuralMeasure] = []

    if aseg_stats_path is not None:
        brain_region_volumes = parse_aseg_stats(aseg_stats_path)
        global_measures = parse_global_measures(aseg_stats_path)
        if not brain_region_volumes:
            warnings.append(f"No region-volume rows were parsed from {aseg_stats_path}.")
    else:
        warnings.append("No aseg.stats file was provided, so anatomical region volumes are unavailable.")

    if lh_aparc_stats_path is not None:
        cortical_thickness.extend(parse_aparc_stats(lh_aparc_stats_path, hemisphere="left"))
    else:
        warnings.append("No lh.aparc.stats file was provided, so left cortical thickness is unavailable.")

    if rh_aparc_stats_path is not None:
        cortical_thickness.extend(parse_aparc_stats(rh_aparc_stats_path, hemisphere="right"))
    else:
        warnings.append("No rh.aparc.stats file was provided, so right cortical thickness is unavailable.")

    hippocampal = _hippocampal_volumes(brain_region_volumes)
    if not hippocampal:
        warnings.append("No hippocampal volume rows were available in the supplied external files.")

    asymmetry = _hippocampal_asymmetry(hippocampal)
    if not asymmetry:
        warnings.append("Hippocampal asymmetry requires both left and right hippocampal volumes.")

    tool_version = None
    for path in source_paths:
        tool_version = read_freesurfer_version_from_file(path)
        if tool_version:
            break

    report = StructuralMetricsReport(
        subject_id=subject_id,
        session_id=session_id,
        source=ExternalToolReference(
            tool_name="freesurfer",
            tool_version=tool_version,
            source_files=source_paths,
            notes=(
                "Parsed from externally generated FreeSurfer statistics files.",
                "The backend did not run segmentation for this report.",
            ),
        ),
        hippocampal_volumes=hippocampal,
        asymmetry_metrics=asymmetry,
        cortical_thickness=cortical_thickness,
        brain_region_volumes=brain_region_volumes,
        global_measures=global_measures,
        warnings=warnings,
        limitations=list(STRUCTURAL_METRICS_LIMITATIONS),
    )
    report.feature_vector = build_structural_feature_vector(report)
    return report


def save_structural_metrics_report(
    report: StructuralMetricsReport,
    *,
    settings: AppSettings | None = None,
    file_stem: str | None = None,
) -> Path:
    """Save a structural metrics report under ``outputs/reports/structural_metrics``."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "structural_metrics")
    default_stem = "_".join(part for part in (report.subject_id, report.session_id, "structural_metrics") if part)
    output_path = output_root / f"{file_stem or default_stem or 'structural_metrics_report'}.json"
    output_path.write_text(json.dumps(report.to_payload(), indent=2), encoding="utf-8")
    return output_path
