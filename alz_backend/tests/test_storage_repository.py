"""Tests for lightweight SQLite-backed backend metadata storage."""

from __future__ import annotations

from pathlib import Path

from src.configs.runtime import AppSettings
from src.storage import (
    AuditMetadataRecord,
    BenchmarkMetadataRecord,
    ExperimentMetadataRecord,
    InferenceMetadataRecord,
    LongitudinalMetadataRecord,
    PromotionMetadataRecord,
    ReviewQueueRecord,
    ScanRegistryRecord,
    count_rows,
    list_review_records,
    persist_benchmark_record,
    persist_audit_record,
    persist_experiment_record,
    persist_inference_record,
    persist_longitudinal_record,
    persist_promotion_record,
    persist_review_record,
    persist_scan_record,
)


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for storage tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    storage_root = project_root / "storage"
    data_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)
    storage_root.mkdir(parents=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=project_root.parent,
        oasis_source_root=project_root.parent / "OASIS",
        storage_root=storage_root,
        database_path=storage_root / "test.sqlite3",
    )


def test_storage_repository_persists_all_core_record_types(tmp_path: Path) -> None:
    """Core metadata record types should be writable and countable."""

    settings = _build_settings(tmp_path)
    persist_benchmark_record(
        BenchmarkMetadataRecord(
            benchmark_id="bench",
            benchmark_name="bench",
            manifest_path="manifest.csv",
            manifest_hash_sha256="abc",
            sample_count=10,
            payload={"benchmark_name": "bench"},
        ),
        settings=settings,
    )
    persist_promotion_record(
        PromotionMetadataRecord(
            model_id="oasis_current_baseline",
            run_name="run",
            benchmark_id="bench",
            policy_name="oasis_research_gate_v1",
            approved=True,
            payload={"approved": True},
        ),
        settings=settings,
    )
    persist_experiment_record(ExperimentMetadataRecord(experiment_name="exp", run_name="run"), settings=settings)
    scan = ScanRegistryRecord(subject_id="OAS1_1000", source_path="scan.nii.gz")
    persist_scan_record(scan, settings=settings)
    persist_inference_record(
        InferenceMetadataRecord(
            scan_id=scan.scan_id,
            subject_id="OAS1_1000",
            model_name="densenet121_3d",
            checkpoint_path="checkpoint.pt",
            payload={"predicted_label": 1},
        ),
        settings=settings,
    )
    persist_review_record(
        ReviewQueueRecord(
            inference_id="inf-1",
            trace_id="trace-1",
            scan_id=scan.scan_id,
            subject_id="OAS1_1000",
            source_path="scan.nii.gz",
            model_name="densenet121_3d",
            confidence_level="low",
            probability_score=0.54,
            payload={"review_flag": True},
        ),
        settings=settings,
    )
    persist_longitudinal_record(
        LongitudinalMetadataRecord(subject_id="OAS1_1000", output_path="report.json", payload={"report_type": "longitudinal_tracking"}),
        settings=settings,
    )
    persist_audit_record(AuditMetadataRecord(action="predict_scan", payload={"trace_id": "123"}), settings=settings)

    assert count_rows("benchmarks", settings=settings) == 1
    assert count_rows("promotions", settings=settings) == 1
    assert count_rows("experiments", settings=settings) == 1
    assert count_rows("scans", settings=settings) == 1
    assert count_rows("inference_logs", settings=settings) == 1
    assert count_rows("review_queue", settings=settings) == 1
    assert count_rows("longitudinal_reports", settings=settings) == 1
    assert count_rows("audit_events", settings=settings) == 1
    assert len(list_review_records(settings=settings)) == 1
