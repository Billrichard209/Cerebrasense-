"""Foreground-proxy volumetric measurements for structural MRI workflows.

This module intentionally implements conservative baseline measurements rather
than region-specific neuroanatomical segmentation. The current outputs are
appropriate for backend prototyping and decision-support reporting:
- volume geometry from the MRI header
- robust intensity summaries
- a foreground-proxy brain mask estimate
- whole-foreground and hemisphere-proxy volume estimates

The measurements should not be presented as diagnostic or region-specific
atrophy biomarkers without a validated segmentation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


class VolumetricAnalysisError(ValueError):
    """Raised when a volumetric analysis request cannot be completed safely."""


@dataclass(slots=True, frozen=True)
class VolumetricMeasurement:
    """Container for one structural proxy measurement."""

    region_name: str
    value_mm3: float
    source_session: str | None
    unit: str = "mm3"
    method: str = "foreground_proxy"
    notes: str | None = None


@dataclass(slots=True, frozen=True)
class VolumeGeometry:
    """Geometry metadata derived from the MRI header."""

    shape: tuple[int, int, int]
    voxel_spacing_mm: tuple[float, float, float]
    voxel_volume_mm3: float
    affine_determinant: float


@dataclass(slots=True, frozen=True)
class IntensityStatistics:
    """Robust summary statistics for voxel intensities."""

    minimum: float
    maximum: float
    mean: float
    std: float
    p01: float
    p50: float
    p99: float


@dataclass(slots=True)
class VolumetricAnalysisResult:
    """Structured volumetric analysis output for one MRI volume."""

    image_path: Path
    dataset: str
    dataset_type: str
    subject_id: str | None
    session_id: str | None
    scan_timestamp: str | None
    geometry: VolumeGeometry
    intensity: IntensityStatistics
    foreground_intensity: IntensityStatistics | None
    foreground_threshold: float
    foreground_voxels: int
    nonzero_voxel_fraction: float
    bounding_box_shape_voxels: tuple[int, int, int]
    measurements: list[VolumetricMeasurement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_report_payload(self) -> dict[str, Any]:
        """Serialize the result into a JSON-safe report payload."""

        return {
            "image": str(self.image_path),
            "dataset": self.dataset,
            "dataset_type": self.dataset_type,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "scan_timestamp": self.scan_timestamp,
            "geometry": {
                "shape": list(self.geometry.shape),
                "voxel_spacing_mm": list(self.geometry.voxel_spacing_mm),
                "voxel_volume_mm3": self.geometry.voxel_volume_mm3,
                "affine_determinant": self.geometry.affine_determinant,
            },
            "intensity": {
                "minimum": self.intensity.minimum,
                "maximum": self.intensity.maximum,
                "mean": self.intensity.mean,
                "std": self.intensity.std,
                "p01": self.intensity.p01,
                "p50": self.intensity.p50,
                "p99": self.intensity.p99,
            },
            "foreground_intensity": None
            if self.foreground_intensity is None
            else {
                "minimum": self.foreground_intensity.minimum,
                "maximum": self.foreground_intensity.maximum,
                "mean": self.foreground_intensity.mean,
                "std": self.foreground_intensity.std,
                "p01": self.foreground_intensity.p01,
                "p50": self.foreground_intensity.p50,
                "p99": self.foreground_intensity.p99,
            },
            "foreground_threshold": self.foreground_threshold,
            "foreground_voxels": self.foreground_voxels,
            "nonzero_voxel_fraction": self.nonzero_voxel_fraction,
            "bounding_box_shape_voxels": list(self.bounding_box_shape_voxels),
            "measurements": [
                {
                    "region_name": measurement.region_name,
                    "value_mm3": measurement.value_mm3,
                    "source_session": measurement.source_session,
                    "unit": measurement.unit,
                    "method": measurement.method,
                    "notes": measurement.notes,
                }
                for measurement in self.measurements
            ],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def summarize_volumetrics(measurements: list[VolumetricMeasurement]) -> dict[str, float]:
    """Collapse measurements into a region-to-volume dictionary."""

    return {measurement.region_name: measurement.value_mm3 for measurement in measurements}


def _ensure_3d_image(data: np.ndarray) -> np.ndarray:
    """Validate that an image array is a 3D volume."""

    if data.ndim == 4 and 1 in data.shape:
        data = np.squeeze(data)
    if data.ndim != 3:
        raise VolumetricAnalysisError(f"Expected a 3D MRI volume, got shape {tuple(data.shape)}")
    return data


def _compute_intensity_statistics(values: np.ndarray) -> IntensityStatistics:
    """Compute robust summary statistics for a flat voxel array."""

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise VolumetricAnalysisError("No finite voxel values were available for intensity statistics.")
    return IntensityStatistics(
        minimum=float(np.min(finite_values)),
        maximum=float(np.max(finite_values)),
        mean=float(np.mean(finite_values)),
        std=float(np.std(finite_values)),
        p01=float(np.percentile(finite_values, 1)),
        p50=float(np.percentile(finite_values, 50)),
        p99=float(np.percentile(finite_values, 99)),
    )


def _estimate_foreground_threshold(data: np.ndarray) -> float:
    """Estimate a conservative foreground threshold from image edges."""

    finite = np.isfinite(data)
    positive = data[finite & (data > 0)]
    if positive.size == 0:
        return 0.0

    edge_values = np.concatenate(
        [
            data[0, :, :].ravel(),
            data[-1, :, :].ravel(),
            data[:, 0, :].ravel(),
            data[:, -1, :].ravel(),
            data[:, :, 0].ravel(),
            data[:, :, -1].ravel(),
        ]
    )
    edge_values = edge_values[np.isfinite(edge_values)]
    if edge_values.size == 0:
        return 0.0

    background_level = float(np.median(edge_values))
    background_std = float(np.std(edge_values))
    positive_floor = float(np.percentile(positive, 1))
    threshold = max(background_level + (2.5 * background_std), min(positive_floor, float(np.median(positive) * 0.05)))
    return float(max(threshold, 0.0))


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""

    labeled_mask, component_count = ndimage.label(mask)
    if component_count <= 1:
        return mask
    component_sizes = ndimage.sum(mask, labeled_mask, index=range(1, component_count + 1))
    largest_component = int(np.argmax(component_sizes)) + 1
    return labeled_mask == largest_component


def estimate_foreground_mask(data: np.ndarray) -> tuple[np.ndarray, float, list[str]]:
    """Estimate a foreground proxy mask for structural volume summaries."""

    threshold = _estimate_foreground_threshold(data)
    finite = np.isfinite(data)
    raw_mask = finite & (data > threshold)
    warnings: list[str] = []

    if raw_mask.mean() < 0.01:
        warnings.append(
            "Edge-derived threshold produced a very small foreground mask, so the analysis fell back to data > 0."
        )
        raw_mask = finite & (data > 0)
        threshold = 0.0

    if not raw_mask.any():
        raise VolumetricAnalysisError("Could not estimate a usable foreground mask from the MRI volume.")

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    cleaned_mask = ndimage.binary_closing(raw_mask, structure=structure, iterations=1)
    cleaned_mask = ndimage.binary_fill_holes(cleaned_mask)
    cleaned_mask = _keep_largest_component(cleaned_mask)

    if cleaned_mask.mean() > 0.95:
        warnings.append(
            "Foreground mask covers almost the whole volume. Interpret the structural proxy measurements carefully."
        )

    return cleaned_mask.astype(bool), float(threshold), warnings


def _bounding_box_shape(mask: np.ndarray) -> tuple[int, int, int]:
    """Compute the axis-aligned bounding-box shape for a mask."""

    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return (0, 0, 0)
    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0)
    spans = (maxs - mins) + 1
    return tuple(int(value) for value in spans)


def _build_measurements(
    mask: np.ndarray,
    *,
    voxel_volume_mm3: float,
    session_id: str | None,
) -> list[VolumetricMeasurement]:
    """Build volumetric proxy measurements from a foreground mask."""

    total_foreground_voxels = int(mask.sum())
    total_foreground_mm3 = float(total_foreground_voxels * voxel_volume_mm3)
    bounding_box_shape = _bounding_box_shape(mask)
    bounding_box_mm3 = float(np.prod(bounding_box_shape) * voxel_volume_mm3)

    split_index = mask.shape[0] // 2
    left_mask = mask[:split_index, :, :]
    right_mask = mask[split_index:, :, :]
    left_mm3 = float(int(left_mask.sum()) * voxel_volume_mm3)
    right_mm3 = float(int(right_mask.sum()) * voxel_volume_mm3)
    asymmetry = float(abs(left_mm3 - right_mm3) / max(total_foreground_mm3, 1.0))

    return [
        VolumetricMeasurement(
            region_name="foreground_proxy_brain",
            value_mm3=total_foreground_mm3,
            source_session=session_id,
            notes="Foreground-proxy volume estimated from the largest connected non-background component.",
        ),
        VolumetricMeasurement(
            region_name="foreground_proxy_bounding_box",
            value_mm3=bounding_box_mm3,
            source_session=session_id,
            notes="Axis-aligned bounding box enclosing the foreground-proxy mask.",
        ),
        VolumetricMeasurement(
            region_name="left_hemisphere_proxy",
            value_mm3=left_mm3,
            source_session=session_id,
            notes="Foreground-proxy volume in the left half of the canonicalized volume.",
        ),
        VolumetricMeasurement(
            region_name="right_hemisphere_proxy",
            value_mm3=right_mm3,
            source_session=session_id,
            notes="Foreground-proxy volume in the right half of the canonicalized volume.",
        ),
        VolumetricMeasurement(
            region_name="hemisphere_asymmetry_index",
            value_mm3=asymmetry,
            source_session=session_id,
            unit="ratio",
            notes="Absolute left-right difference divided by total foreground-proxy volume.",
        ),
    ]


def analyze_mri_volume(
    image_path: str | Path,
    *,
    dataset: str = "oasis1",
    dataset_type: str = "3d_volumes",
    subject_id: str | None = None,
    session_id: str | None = None,
    scan_timestamp: str | None = None,
) -> VolumetricAnalysisResult:
    """Analyze one 3D MRI volume and return structural proxy measurements."""

    resolved_path = Path(image_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Volume path does not exist: {resolved_path}")

    image = nib.load(str(resolved_path))
    canonical_image = nib.as_closest_canonical(image)
    data = np.asarray(canonical_image.get_fdata(dtype=np.float32))
    data = _ensure_3d_image(data)

    zooms = canonical_image.header.get_zooms()[:3]
    voxel_spacing = tuple(float(value) for value in zooms)
    voxel_volume_mm3 = float(np.prod(voxel_spacing))
    geometry = VolumeGeometry(
        shape=tuple(int(value) for value in data.shape),
        voxel_spacing_mm=voxel_spacing,
        voxel_volume_mm3=voxel_volume_mm3,
        affine_determinant=float(np.linalg.det(canonical_image.affine[:3, :3])),
    )

    intensity = _compute_intensity_statistics(data.ravel())
    foreground_mask, threshold, warnings = estimate_foreground_mask(data)
    foreground_values = data[foreground_mask]
    foreground_intensity = _compute_intensity_statistics(foreground_values) if foreground_values.size > 0 else None
    nonzero_voxel_fraction = float(int(foreground_mask.sum()) / max(int(np.isfinite(data).sum()), 1))
    measurements = _build_measurements(
        foreground_mask,
        voxel_volume_mm3=voxel_volume_mm3,
        session_id=session_id,
    )

    notes = [
        "This report is a structural foreground-proxy summary for decision support, not a diagnosis.",
        "Region-specific biomarkers such as hippocampal volume require validated segmentation models and are not inferred here.",
        "Left/right proxy volumes are computed after canonical orientation and should be interpreted as coarse asymmetry checks only.",
    ]
    return VolumetricAnalysisResult(
        image_path=resolved_path,
        dataset=dataset,
        dataset_type=dataset_type,
        subject_id=subject_id,
        session_id=session_id,
        scan_timestamp=scan_timestamp,
        geometry=geometry,
        intensity=intensity,
        foreground_intensity=foreground_intensity,
        foreground_threshold=threshold,
        foreground_voxels=int(foreground_mask.sum()),
        nonzero_voxel_fraction=nonzero_voxel_fraction,
        bounding_box_shape_voxels=_bounding_box_shape(foreground_mask),
        measurements=measurements,
        warnings=warnings,
        notes=notes,
    )
