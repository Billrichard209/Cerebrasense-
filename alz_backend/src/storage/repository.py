"""Repository helpers for backend metadata persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from .database import connect_backend_storage
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

LOGGER = logging.getLogger(__name__)


def _execute(
    statement: str,
    values: tuple[Any, ...],
    *,
    settings=None,
) -> None:
    """Execute one write statement against the storage database."""

    connection = connect_backend_storage(settings=settings)
    try:
        with connection:
            connection.execute(statement, values)
    finally:
        connection.close()


def persist_experiment_record(record: ExperimentMetadataRecord, *, settings=None) -> None:
    """Persist one tracked experiment record."""

    _execute(
        """
        INSERT OR REPLACE INTO experiments (
            experiment_id, experiment_name, run_name, dataset, primary_split,
            tags_json, best_checkpoint_path, metrics_json, summary_path, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.experiment_id,
            record.experiment_name,
            record.run_name,
            record.dataset,
            record.primary_split,
            json.dumps(record.tags),
            record.best_checkpoint_path,
            json.dumps(record.metrics),
            record.summary_path,
            record.created_at,
        ),
        settings=settings,
    )


def persist_benchmark_record(record: BenchmarkMetadataRecord, *, settings=None) -> None:
    """Persist one registered benchmark record."""

    _execute(
        """
        INSERT OR REPLACE INTO benchmarks (
            benchmark_id, benchmark_name, dataset, split_name, manifest_path,
            manifest_hash_sha256, sample_count, subject_safe, payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.benchmark_id,
            record.benchmark_name,
            record.dataset,
            record.split_name,
            record.manifest_path,
            record.manifest_hash_sha256,
            record.sample_count,
            int(record.subject_safe),
            json.dumps(record.payload),
            record.created_at,
        ),
        settings=settings,
    )


def persist_promotion_record(record: PromotionMetadataRecord, *, settings=None) -> None:
    """Persist one promotion decision record."""

    _execute(
        """
        INSERT OR REPLACE INTO promotions (
            promotion_id, model_id, run_name, benchmark_id, policy_name,
            approved, output_path, payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.promotion_id,
            record.model_id,
            record.run_name,
            record.benchmark_id,
            record.policy_name,
            int(record.approved),
            record.output_path,
            json.dumps(record.payload),
            record.created_at,
        ),
        settings=settings,
    )


def persist_scan_record(record: ScanRegistryRecord, *, settings=None) -> None:
    """Persist one scan registry record."""

    _execute(
        """
        INSERT OR REPLACE INTO scans (
            scan_id, subject_id, session_id, source_path, file_format,
            file_size_bytes, dataset, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.scan_id,
            record.subject_id,
            record.session_id,
            record.source_path,
            record.file_format,
            record.file_size_bytes,
            record.dataset,
            json.dumps(record.metadata),
            record.created_at,
        ),
        settings=settings,
    )


def persist_inference_record(record: InferenceMetadataRecord, *, settings=None) -> None:
    """Persist one inference log record."""

    _execute(
        """
        INSERT OR REPLACE INTO inference_logs (
            inference_id, trace_id, scan_id, subject_id, session_id, model_name,
            checkpoint_path, output_path, predicted_label, label_name,
            confidence_level, review_flag, payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.inference_id,
            record.trace_id,
            record.scan_id,
            record.subject_id,
            record.session_id,
            record.model_name,
            record.checkpoint_path,
            record.output_path,
            record.predicted_label,
            record.label_name,
            record.confidence_level,
            int(record.review_flag),
            json.dumps(record.payload),
            record.created_at,
        ),
        settings=settings,
    )


def persist_review_record(record: ReviewQueueRecord, *, settings=None) -> None:
    """Persist one human-review queue record."""

    _execute(
        """
        INSERT OR REPLACE INTO review_queue (
            review_id, inference_id, trace_id, scan_id, subject_id, session_id,
            source_path, model_name, confidence_level, probability_score,
            output_path, status, reason, payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.review_id,
            record.inference_id,
            record.trace_id,
            record.scan_id,
            record.subject_id,
            record.session_id,
            record.source_path,
            record.model_name,
            record.confidence_level,
            record.probability_score,
            record.output_path,
            record.status,
            record.reason,
            json.dumps(record.payload),
            record.created_at,
        ),
        settings=settings,
    )


def persist_longitudinal_record(record: LongitudinalMetadataRecord, *, settings=None) -> None:
    """Persist one longitudinal report record."""

    _execute(
        """
        INSERT OR REPLACE INTO longitudinal_reports (
            report_id, subject_id, report_type, output_path, payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            record.report_id,
            record.subject_id,
            record.report_type,
            record.output_path,
            json.dumps(record.payload),
            record.created_at,
        ),
        settings=settings,
    )


def persist_audit_record(record: AuditMetadataRecord, *, settings=None) -> None:
    """Persist one audit event record."""

    try:
        _execute(
            """
            INSERT OR REPLACE INTO audit_events (
                audit_id, action, actor_id, subject_id, patient_id_hash,
                outcome, purpose, payload_json, event_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.audit_id,
                record.action,
                record.actor_id,
                record.subject_id,
                record.patient_id_hash,
                record.outcome,
                record.purpose,
                json.dumps(record.payload),
                record.event_time,
            ),
            settings=settings,
        )
    except sqlite3.Error as error:  # pragma: no cover - defensive storage failure
        LOGGER.warning("Could not persist audit record: %s", error)


def count_rows(table_name: str, *, settings=None) -> int:
    """Return the number of rows in a metadata table."""

    connection = connect_backend_storage(settings=settings)
    try:
        row = connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
        return int(row["count"])
    finally:
        connection.close()


def get_review_record(review_id: str, *, settings=None) -> ReviewQueueRecord | None:
    """Return one review-queue record by id when present."""

    connection = connect_backend_storage(settings=settings)
    try:
        row = connection.execute(
            """
            SELECT review_id, inference_id, trace_id, scan_id, subject_id, session_id,
                   source_path, model_name, confidence_level, probability_score,
                   output_path, status, reason, payload_json, created_at
            FROM review_queue
            WHERE review_id = ?
            """,
            (review_id,),
        ).fetchone()
        if row is None:
            return None
        return ReviewQueueRecord(
            review_id=row["review_id"],
            inference_id=row["inference_id"],
            trace_id=row["trace_id"],
            scan_id=row["scan_id"],
            subject_id=row["subject_id"],
            session_id=row["session_id"],
            source_path=row["source_path"],
            model_name=row["model_name"],
            confidence_level=row["confidence_level"],
            probability_score=row["probability_score"],
            output_path=row["output_path"],
            status=row["status"],
            reason=row["reason"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
        )
    finally:
        connection.close()


def list_review_records(
    *,
    status: str | None = "pending",
    limit: int = 20,
    settings=None,
) -> list[ReviewQueueRecord]:
    """Return recent review-queue records for the requested status."""

    connection = connect_backend_storage(settings=settings)
    try:
        if status is None:
            rows = connection.execute(
                """
                SELECT review_id, inference_id, trace_id, scan_id, subject_id, session_id,
                       source_path, model_name, confidence_level, probability_score,
                       output_path, status, reason, payload_json, created_at
                FROM review_queue
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT review_id, inference_id, trace_id, scan_id, subject_id, session_id,
                       source_path, model_name, confidence_level, probability_score,
                       output_path, status, reason, payload_json, created_at
                FROM review_queue
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (status, int(limit)),
            ).fetchall()
        return [
            ReviewQueueRecord(
                review_id=row["review_id"],
                inference_id=row["inference_id"],
                trace_id=row["trace_id"],
                scan_id=row["scan_id"],
                subject_id=row["subject_id"],
                session_id=row["session_id"],
                source_path=row["source_path"],
                model_name=row["model_name"],
                confidence_level=row["confidence_level"],
                probability_score=row["probability_score"],
                output_path=row["output_path"],
                status=row["status"],
                reason=row["reason"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]
    finally:
        connection.close()
