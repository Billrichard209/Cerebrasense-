"""Tests for external-tool structural metrics parsing and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.configs.runtime import AppSettings
from src.volumetrics.freesurfer import parse_aparc_stats, parse_aseg_stats, parse_global_measures
from src.volumetrics.structural import (
    build_freesurfer_structural_report,
    compare_report_to_reference_ranges,
    load_structural_reference_ranges,
    save_structural_metrics_report,
)


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for structural metrics tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    data_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=project_root.parent,
        oasis_source_root=project_root.parent / "OASIS",
    )


def _write_aseg_stats(path: Path, *, include_hippocampus: bool = True) -> Path:
    """Write a small synthetic FreeSurfer aseg.stats file."""

    rows = [
        "# FreeSurfer version: synthetic-7.4.1",
        "# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1218597.2, mm^3",
        "# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange",
        "1 2 1000 1000.0 Left-Cerebral-White-Matter 0 0 0 0 0",
    ]
    if include_hippocampus:
        rows.extend(
            [
                "2 17 2100 3210.5 Left-Hippocampus 0 0 0 0 0",
                "3 53 2200 3302.1 Right-Hippocampus 0 0 0 0 0",
            ]
        )
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _write_aparc_stats(path: Path) -> Path:
    """Write a small synthetic FreeSurfer aparc.stats file."""

    path.write_text(
        "\n".join(
            [
                "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd",
                "entorhinal 150 310.0 842.0 2.71 0.18 0 0 0 0",
                "precuneus 300 640.0 1500.0 2.42 0.21 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_parse_aseg_stats_extracts_region_volumes_and_hemisphere(tmp_path: Path) -> None:
    """aseg.stats parsing should preserve region volumes and labels."""

    aseg_path = _write_aseg_stats(tmp_path / "aseg.stats")
    volumes = parse_aseg_stats(aseg_path)

    hippocampus = {volume.region_name: volume for volume in volumes if "Hippocampus" in volume.region_name}
    assert hippocampus["Left-Hippocampus"].value_mm3 == 3210.5
    assert hippocampus["Left-Hippocampus"].hemisphere == "left"
    assert hippocampus["Right-Hippocampus"].segmentation_label == 53


def test_parse_aparc_stats_extracts_cortical_thickness(tmp_path: Path) -> None:
    """aparc.stats parsing should extract cortical thickness summaries."""

    aparc_path = _write_aparc_stats(tmp_path / "lh.aparc.stats")
    summaries = parse_aparc_stats(aparc_path)

    assert summaries[0].hemisphere == "left"
    assert summaries[0].region_name == "entorhinal"
    assert summaries[0].mean_thickness_mm == 2.71
    assert summaries[0].surface_area_mm2 == 310.0
    assert summaries[0].gray_matter_volume_mm3 == 842.0


def test_parse_global_measures_extracts_measure_rows(tmp_path: Path) -> None:
    """Global ``# Measure`` rows should be preserved for later feature fusion."""

    aseg_path = _write_aseg_stats(tmp_path / "aseg.stats")
    measures = parse_global_measures(aseg_path)

    assert measures[0].measure_id == "BrainSeg"
    assert measures[0].feature_name == "brainsegvol"
    assert measures[0].value == 1218597.2


def test_build_freesurfer_structural_report_builds_fusion_ready_features(tmp_path: Path) -> None:
    """A full FreeSurfer report should include hippocampal, asymmetry, and thickness features."""

    aseg_path = _write_aseg_stats(tmp_path / "aseg.stats")
    lh_path = _write_aparc_stats(tmp_path / "lh.aparc.stats")
    rh_path = _write_aparc_stats(tmp_path / "rh.aparc.stats")

    report = build_freesurfer_structural_report(
        subject_id="OAS1_0002",
        session_id="OAS1_0002_MR1",
        aseg_stats_path=aseg_path,
        lh_aparc_stats_path=lh_path,
        rh_aparc_stats_path=rh_path,
    )

    assert report.subject_id == "OAS1_0002"
    assert report.source.tool_name == "freesurfer"
    assert report.source.tool_version == "synthetic-7.4.1"
    assert len(report.hippocampal_volumes) == 2
    assert report.asymmetry_metrics[0].metric_name == "hippocampal_asymmetry_index"
    assert "left_hippocampus_volume_mm3" in report.feature_vector
    assert "right_hippocampus_volume_mm3" in report.feature_vector
    assert "left_entorhinal_thickness_mm" in report.feature_vector
    assert "brainsegvol" in report.feature_vector
    assert report.global_measures[0].display_name == "Brain Segmentation Volume"
    assert any("not diagnosis" in limitation.lower() for limitation in report.limitations)


def test_reference_range_comparison_marks_low_within_and_high(tmp_path: Path) -> None:
    """Reference-style comparisons should classify observed values without faking conclusions."""

    aseg_path = _write_aseg_stats(tmp_path / "aseg.stats")
    report = build_freesurfer_structural_report(subject_id="OAS1_0010", aseg_stats_path=aseg_path)
    reference_path = tmp_path / "reference_ranges.json"
    reference_path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "feature_name": "left_hippocampus_volume_mm3",
                        "display_name": "Left Hippocampus",
                        "min_value": 3300.0,
                        "max_value": 4000.0,
                        "unit": "mm3",
                    },
                    {
                        "feature_name": "brainsegvol",
                        "display_name": "Brain Seg Volume",
                        "min_value": 1100000.0,
                        "max_value": 1300000.0,
                        "unit": "mm^3",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    comparisons = compare_report_to_reference_ranges(report, load_structural_reference_ranges(reference_path))

    status_by_feature = {item.feature_name: item.status for item in comparisons}
    assert status_by_feature["left_hippocampus_volume_mm3"] == "low"
    assert status_by_feature["brainsegvol"] == "within_range"


def test_structural_report_does_not_fabricate_missing_hippocampal_measurements(tmp_path: Path) -> None:
    """Missing hippocampal rows should produce warnings, not fake measurements."""

    aseg_path = _write_aseg_stats(tmp_path / "aseg.stats", include_hippocampus=False)
    report = build_freesurfer_structural_report(
        subject_id="OAS1_0003",
        aseg_stats_path=aseg_path,
    )

    assert report.hippocampal_volumes == []
    assert report.asymmetry_metrics == []
    assert "left_hippocampus_volume_mm3" not in report.feature_vector
    assert any("no hippocampal volume" in warning.lower() for warning in report.warnings)


def test_save_structural_metrics_report_writes_json_payload(tmp_path: Path) -> None:
    """The report generator should write JSON-safe structural metrics."""

    settings = _build_settings(tmp_path)
    aseg_path = _write_aseg_stats(tmp_path / "aseg.stats")
    report = build_freesurfer_structural_report(subject_id="OAS1_0004", aseg_stats_path=aseg_path)
    output_path = save_structural_metrics_report(report, settings=settings)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["subject_id"] == "OAS1_0004"
    assert payload["source"]["tool_name"] == "freesurfer"
    assert payload["brain_region_volumes"]
    assert payload["global_measures"]


def test_build_freesurfer_structural_report_requires_external_files() -> None:
    """The module should fail loudly instead of inventing measurements with no inputs."""

    with pytest.raises(ValueError, match="At least one external stats file"):
        build_freesurfer_structural_report(subject_id="OAS1_0005")
