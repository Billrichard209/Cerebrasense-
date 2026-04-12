"""Schemas for externally generated structural MRI metrics.

These schemas are designed for validated neuroimaging outputs such as
FreeSurfer statistics files. The backend does not infer hippocampal volume,
cortical thickness, or anatomical region segmentation from raw MRI unless an
explicit external or future validated segmentation source provides them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _json_safe_path(path: str | Path | None) -> str | None:
    """Normalize optional paths for JSON payloads."""

    return None if path is None else str(path)


@dataclass(slots=True, frozen=True)
class ExternalToolReference:
    """Provenance for structural metrics parsed from an external tool."""

    tool_name: str
    tool_version: str | None = None
    source_files: tuple[str | Path, ...] = ()
    notes: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Serialize the source reference."""

        return {
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "source_files": [_json_safe_path(path) for path in self.source_files],
            "notes": list(self.notes),
        }


@dataclass(slots=True, frozen=True)
class HippocampalVolume:
    """Hippocampal volume parsed from an anatomical segmentation output."""

    hemisphere: str
    region_name: str
    value_mm3: float
    source_file: str | Path
    method: str = "external_segmentation"

    def to_payload(self) -> dict[str, Any]:
        """Serialize the hippocampal volume."""

        return {
            "hemisphere": self.hemisphere,
            "region_name": self.region_name,
            "value_mm3": self.value_mm3,
            "source_file": _json_safe_path(self.source_file),
            "method": self.method,
        }


@dataclass(slots=True, frozen=True)
class AsymmetryMetric:
    """Left/right structural asymmetry summary."""

    metric_name: str
    left_value: float
    right_value: float
    asymmetry_index: float
    unit: str
    source: str = "derived_from_external_metrics"
    notes: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize the asymmetry metric."""

        return {
            "metric_name": self.metric_name,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "asymmetry_index": self.asymmetry_index,
            "unit": self.unit,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass(slots=True, frozen=True)
class CorticalThicknessSummary:
    """Per-region cortical thickness summary parsed from surface stats."""

    hemisphere: str
    region_name: str
    mean_thickness_mm: float
    source_file: str | Path
    thickness_std_mm: float | None = None
    surface_area_mm2: float | None = None
    gray_matter_volume_mm3: float | None = None
    method: str = "external_surface_stats"

    def to_payload(self) -> dict[str, Any]:
        """Serialize the cortical thickness summary."""

        return {
            "hemisphere": self.hemisphere,
            "region_name": self.region_name,
            "mean_thickness_mm": self.mean_thickness_mm,
            "thickness_std_mm": self.thickness_std_mm,
            "surface_area_mm2": self.surface_area_mm2,
            "gray_matter_volume_mm3": self.gray_matter_volume_mm3,
            "source_file": _json_safe_path(self.source_file),
            "method": self.method,
        }


@dataclass(slots=True, frozen=True)
class BrainRegionVolume:
    """Brain-region volume parsed from an external segmentation output."""

    region_name: str
    value_mm3: float
    source_file: str | Path
    hemisphere: str | None = None
    segmentation_label: int | None = None
    method: str = "external_segmentation"

    def to_payload(self) -> dict[str, Any]:
        """Serialize the brain-region volume."""

        return {
            "region_name": self.region_name,
            "hemisphere": self.hemisphere,
            "value_mm3": self.value_mm3,
            "segmentation_label": self.segmentation_label,
            "source_file": _json_safe_path(self.source_file),
            "method": self.method,
        }


@dataclass(slots=True, frozen=True)
class GlobalStructuralMeasure:
    """Global structural metric parsed from an external tool summary line."""

    measure_id: str
    feature_name: str
    display_name: str
    value: float
    unit: str | None = None
    source_file: str | Path | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize the global measure."""

        return {
            "measure_id": self.measure_id,
            "feature_name": self.feature_name,
            "display_name": self.display_name,
            "value": self.value,
            "unit": self.unit,
            "source_file": _json_safe_path(self.source_file),
        }


@dataclass(slots=True, frozen=True)
class StructuralReferenceRange:
    """Reference-style range for one structural feature."""

    feature_name: str
    display_name: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    unit: str | None = None
    notes: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize the reference range."""

        return {
            "feature_name": self.feature_name,
            "display_name": self.display_name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "unit": self.unit,
            "notes": self.notes,
        }


@dataclass(slots=True, frozen=True)
class StructuralReferenceComparison:
    """Comparison of an observed feature against a provided reference range."""

    feature_name: str
    display_name: str
    observed_value: float
    status: str
    min_value: float | None = None
    max_value: float | None = None
    unit: str | None = None
    notes: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize the comparison."""

        return {
            "feature_name": self.feature_name,
            "display_name": self.display_name,
            "observed_value": self.observed_value,
            "status": self.status,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "unit": self.unit,
            "notes": self.notes,
        }


@dataclass(slots=True)
class StructuralMetricsReport:
    """Fusion-ready structural metrics report for one subject/session."""

    subject_id: str | None
    session_id: str | None
    source: ExternalToolReference
    hippocampal_volumes: list[HippocampalVolume] = field(default_factory=list)
    asymmetry_metrics: list[AsymmetryMetric] = field(default_factory=list)
    cortical_thickness: list[CorticalThicknessSummary] = field(default_factory=list)
    brain_region_volumes: list[BrainRegionVolume] = field(default_factory=list)
    global_measures: list[GlobalStructuralMeasure] = field(default_factory=list)
    reference_comparisons: list[StructuralReferenceComparison] = field(default_factory=list)
    feature_vector: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the structural metrics report."""

        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "source": self.source.to_payload(),
            "hippocampal_volumes": [item.to_payload() for item in self.hippocampal_volumes],
            "asymmetry_metrics": [item.to_payload() for item in self.asymmetry_metrics],
            "cortical_thickness": [item.to_payload() for item in self.cortical_thickness],
            "brain_region_volumes": [item.to_payload() for item in self.brain_region_volumes],
            "global_measures": [item.to_payload() for item in self.global_measures],
            "reference_comparisons": [item.to_payload() for item in self.reference_comparisons],
            "feature_vector": dict(self.feature_vector),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


STRUCTURAL_METRICS_LIMITATIONS = (
    "Measurements are parsed from external segmentation/statistics outputs when available.",
    "This backend does not claim FreeSurfer or other external outputs exist unless files are provided.",
    "Hippocampal volume and cortical thickness are not inferred from raw MRI by this module.",
    "Structural metrics are decision-support research features, not diagnosis.",
)
