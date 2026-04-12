"""Schemas for persisted backend metadata records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ExperimentMetadataRecord:
    """Persisted experiment metadata."""

    experiment_id: str = field(default_factory=lambda: str(uuid4()))
    experiment_name: str = ""
    run_name: str = ""
    dataset: str = "oasis1"
    primary_split: str = "val"
    tags: list[str] = field(default_factory=list)
    best_checkpoint_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    summary_path: str | None = None
    created_at: str = field(default_factory=_utc_now)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkMetadataRecord:
    """Persisted benchmark registry entry."""

    benchmark_id: str = ""
    benchmark_name: str = ""
    dataset: str = "oasis1"
    split_name: str = "test"
    manifest_path: str = ""
    manifest_hash_sha256: str = ""
    sample_count: int = 0
    subject_safe: bool = True
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromotionMetadataRecord:
    """Persisted checkpoint-promotion decision."""

    promotion_id: str = field(default_factory=lambda: str(uuid4()))
    model_id: str = ""
    run_name: str = ""
    benchmark_id: str | None = None
    policy_name: str = ""
    approved: bool = False
    output_path: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScanRegistryRecord:
    """Persisted scan registry entry."""

    scan_id: str = field(default_factory=lambda: str(uuid4()))
    subject_id: str | None = None
    session_id: str | None = None
    source_path: str = ""
    file_format: str | None = None
    file_size_bytes: int | None = None
    dataset: str = "oasis1"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferenceMetadataRecord:
    """Persisted inference log entry."""

    inference_id: str = field(default_factory=lambda: str(uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    scan_id: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    model_name: str = ""
    checkpoint_path: str = ""
    output_path: str | None = None
    predicted_label: int | None = None
    label_name: str | None = None
    confidence_level: str | None = None
    review_flag: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewQueueRecord:
    """Persisted human-review queue entry for low-confidence predictions."""

    review_id: str = field(default_factory=lambda: str(uuid4()))
    inference_id: str = ""
    trace_id: str = ""
    scan_id: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    source_path: str | None = None
    model_name: str = ""
    confidence_level: str | None = None
    probability_score: float | None = None
    output_path: str | None = None
    status: str = "pending"
    reason: str = "low_confidence_prediction"
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LongitudinalMetadataRecord:
    """Persisted longitudinal report entry."""

    report_id: str = field(default_factory=lambda: str(uuid4()))
    subject_id: str = ""
    report_type: str = "longitudinal_tracking"
    output_path: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuditMetadataRecord:
    """Persisted audit event entry."""

    audit_id: str = field(default_factory=lambda: str(uuid4()))
    action: str = ""
    actor_id: str | None = None
    subject_id: str | None = None
    patient_id_hash: str | None = None
    outcome: str = "success"
    purpose: str = "clinical_decision_support"
    payload: dict[str, Any] = field(default_factory=dict)
    event_time: str = field(default_factory=_utc_now)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)
