"""Lightweight storage foundation for backend metadata persistence."""

from .database import connect_backend_storage, initialize_backend_storage, resolve_database_path
from .repository import (
    count_rows,
    get_review_record,
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
from .schemas import (
    AuditMetadataRecord,
    BenchmarkMetadataRecord,
    ExperimentMetadataRecord,
    InferenceMetadataRecord,
    LongitudinalMetadataRecord,
    PromotionMetadataRecord,
    ReviewQueueRecord,
    ScanRegistryRecord,
)

__all__ = [
    "AuditMetadataRecord",
    "BenchmarkMetadataRecord",
    "ExperimentMetadataRecord",
    "InferenceMetadataRecord",
    "LongitudinalMetadataRecord",
    "PromotionMetadataRecord",
    "ReviewQueueRecord",
    "ScanRegistryRecord",
    "connect_backend_storage",
    "count_rows",
    "get_review_record",
    "initialize_backend_storage",
    "list_review_records",
    "persist_benchmark_record",
    "persist_audit_record",
    "persist_experiment_record",
    "persist_inference_record",
    "persist_longitudinal_record",
    "persist_promotion_record",
    "persist_review_record",
    "persist_scan_record",
    "resolve_database_path",
]
