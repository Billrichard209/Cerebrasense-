"""Tests for general longitudinal scan-history tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.configs.runtime import AppSettings
from src.longitudinal.service import (
    build_and_save_longitudinal_report,
    records_from_csv,
    records_from_structural_summary_payload,
)
from src.longitudinal.tracker import (
    LongitudinalRecord,
    TrendFeatureConfig,
    build_longitudinal_report,
    sort_records_by_visit,
)


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for longitudinal tests."""

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


def _records() -> list[LongitudinalRecord]:
    """Create a three-visit synthetic scan history."""

    return [
        LongitudinalRecord(
            subject_id="OAS1_9000",
            session_id="OAS1_9000_MR3",
            visit_order=3,
            scan_timestamp="2003-01-01",
            source_path="scan3.hdr",
            dataset="oasis1",
            volumetric_features={"left_hippocampus_volume_mm3": 2816.0},
            model_probabilities={"ad_like_probability": 0.44},
        ),
        LongitudinalRecord(
            subject_id="OAS1_9000",
            session_id="OAS1_9000_MR1",
            visit_order=1,
            scan_timestamp="2001-01-01",
            source_path="scan1.hdr",
            dataset="oasis1",
            volumetric_features={"left_hippocampus_volume_mm3": 3200.0},
            model_probabilities={"ad_like_probability": 0.20},
        ),
        LongitudinalRecord(
            subject_id="OAS1_9000",
            session_id="OAS1_9000_MR2",
            visit_order=2,
            scan_timestamp="2002-01-01",
            source_path="scan2.hdr",
            dataset="oasis1",
            volumetric_features={"left_hippocampus_volume_mm3": 3008.0},
            model_probabilities={"ad_like_probability": 0.32},
        ),
    ]


def test_sort_records_by_visit_uses_timestamp_then_visit_order() -> None:
    """Records should be ordered correctly even if input order is shuffled."""

    ordered = sort_records_by_visit(_records())

    assert [record.session_id for record in ordered] == ["OAS1_9000_MR1", "OAS1_9000_MR2", "OAS1_9000_MR3"]


def test_build_longitudinal_report_computes_changes_trends_and_alerts() -> None:
    """The report should include interval changes, slopes, and trend alerts."""

    report = build_longitudinal_report(_records(), subject_id="OAS1_9000")

    assert report.timepoint_count == 3
    assert len(report.interval_changes) == 4
    volume_summary = next(item for item in report.trend_summaries if item.feature_name == "left_hippocampus_volume_mm3")
    probability_summary = next(item for item in report.trend_summaries if item.feature_name == "ad_like_probability")
    assert volume_summary.trend_classification == "rapid_decline"
    assert volume_summary.normalized_slope_per_visit == -6.0
    assert probability_summary.trend_classification == "rapid_decline"
    assert probability_summary.normalized_slope_per_visit == 0.12
    assert {alert.feature_name for alert in report.alerts} == {"left_hippocampus_volume_mm3", "ad_like_probability"}
    assert report.timeline[0]["elapsed_days_from_baseline"] == 0.0
    assert report.progression_overview.overall_trend_classification == "rapid_decline"
    assert report.progression_overview.review_recommended is True
    assert set(report.progression_overview.rapid_decline_features) == {
        "left_hippocampus_volume_mm3",
        "ad_like_probability",
    }
    assert report.progression_overview.baseline_session_id == "OAS1_9000_MR1"
    assert report.progression_overview.latest_session_id == "OAS1_9000_MR3"


def test_build_longitudinal_report_uses_configurable_thresholds() -> None:
    """Feature configs should control the stable/mild/rapid classification."""

    config = TrendFeatureConfig(
        feature_name="left_hippocampus_volume_mm3",
        source="volumetric",
        decline_direction="decrease",
        normalization="percent_from_baseline",
        stable_slope_threshold=10.0,
        rapid_slope_threshold=20.0,
    )
    report = build_longitudinal_report(_records(), subject_id="OAS1_9000", feature_configs=[config])

    assert report.trend_summaries[0].trend_classification == "stable"
    assert report.alerts == []


def test_build_longitudinal_report_handles_missing_timestamps() -> None:
    """Missing timestamps should not prevent visit-order trend computation."""

    records = [
        LongitudinalRecord("OAS1_9001", "OAS1_9001_MR2", 2, volumetric_features={"brain_volume": 90.0}),
        LongitudinalRecord("OAS1_9001", "OAS1_9001_MR1", 1, volumetric_features={"brain_volume": 100.0}),
    ]
    report = build_longitudinal_report(records, subject_id="OAS1_9001")

    assert report.timeline[0]["session_id"] == "OAS1_9001_MR1"
    assert any("timestamps" in warning.lower() for warning in report.warnings)
    assert report.trend_summaries[0].trend_classification == "rapid_decline"


def test_records_from_csv_loads_volumetric_and_probability_features(tmp_path: Path) -> None:
    """CSV input should preserve vol__ and prob__ feature namespaces."""

    csv_path = tmp_path / "longitudinal.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "OAS1_9002",
                "session_id": "OAS1_9002_MR1",
                "visit_order": 1,
                "scan_timestamp": "2001-01-01",
                "image": "scan1.hdr",
                "dataset": "oasis1",
                "vol__brain_volume": 100.0,
                "prob__ad_like_probability": 0.2,
                "site": "synthetic",
            }
        ]
    ).to_csv(csv_path, index=False)

    records = records_from_csv(csv_path, subject_id="OAS1_9002")

    assert records[0].source_path == "scan1.hdr"
    assert records[0].volumetric_features["brain_volume"] == 100.0
    assert records[0].model_probabilities["ad_like_probability"] == 0.2
    assert records[0].metadata["site"] == "synthetic"


def test_records_from_structural_summary_payload_maps_metrics() -> None:
    """Existing structural reports should be reusable as generic longitudinal input."""

    records = records_from_structural_summary_payload(
        {
            "subject_id": "OAS1_9003",
            "dataset": "oasis1",
            "dataset_type": "3d_volumes",
            "timepoints": [
                {
                    "session_id": "OAS1_9003_MR1",
                    "visit_order": 1,
                    "scan_timestamp": None,
                    "image": "scan.hdr",
                    "metrics": {"foreground_proxy_brain": 123.0},
                }
            ],
        }
    )

    assert records[0].subject_id == "OAS1_9003"
    assert records[0].volumetric_features["foreground_proxy_brain"] == 123.0


def test_build_and_save_longitudinal_report_writes_json(tmp_path: Path) -> None:
    """Saved reports should be timeline-ready JSON artifacts."""

    settings = _build_settings(tmp_path)
    report, output_path = build_and_save_longitudinal_report(
        _records(),
        subject_id="OAS1_9000",
        settings=settings,
        file_stem="unit_longitudinal",
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["subject_id"] == report.subject_id
    assert payload["timeline"]
    assert payload["trend_summaries"]
    assert payload["progression_overview"]["overall_trend_classification"] == "rapid_decline"
    assert payload["limitations"]
