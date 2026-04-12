"""Structural and volumetric analysis helpers for MRI-derived measurements."""

from .freesurfer import FreeSurferParseError, parse_aparc_stats, parse_aseg_stats, parse_global_measures
from .measurements import (
    IntensityStatistics,
    VolumeGeometry,
    VolumetricAnalysisError,
    VolumetricAnalysisResult,
    VolumetricMeasurement,
    analyze_mri_volume,
    estimate_foreground_mask,
    summarize_volumetrics,
)
from .schemas import (
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
from .service import analyze_oasis_volume, build_volumetric_report, save_oasis_volumetric_report
from .structural import (
    StructuralMetricsError,
    build_freesurfer_structural_report,
    build_structural_feature_vector,
    compare_report_to_reference_ranges,
    load_structural_reference_ranges,
    save_structural_metrics_report,
)

__all__ = [
    "AsymmetryMetric",
    "BrainRegionVolume",
    "CorticalThicknessSummary",
    "ExternalToolReference",
    "FreeSurferParseError",
    "GlobalStructuralMeasure",
    "HippocampalVolume",
    "IntensityStatistics",
    "StructuralMetricsError",
    "StructuralMetricsReport",
    "StructuralReferenceComparison",
    "StructuralReferenceRange",
    "VolumeGeometry",
    "VolumetricAnalysisError",
    "VolumetricAnalysisResult",
    "VolumetricMeasurement",
    "analyze_mri_volume",
    "analyze_oasis_volume",
    "build_freesurfer_structural_report",
    "build_structural_feature_vector",
    "build_volumetric_report",
    "compare_report_to_reference_ranges",
    "estimate_foreground_mask",
    "parse_aparc_stats",
    "parse_aseg_stats",
    "parse_global_measures",
    "load_structural_reference_ranges",
    "save_oasis_volumetric_report",
    "save_structural_metrics_report",
    "summarize_volumetrics",
]
