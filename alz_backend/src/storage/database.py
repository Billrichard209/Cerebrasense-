"""SQLite-backed metadata storage foundation.

This is intentionally a lightweight local foundation. Table and column choices
stay PostgreSQL-friendly so they can later be migrated to JSONB/UUID-backed
production schemas.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

DDL_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS benchmarks (
        benchmark_id TEXT PRIMARY KEY,
        benchmark_name TEXT NOT NULL,
        dataset TEXT NOT NULL,
        split_name TEXT NOT NULL,
        manifest_path TEXT NOT NULL,
        manifest_hash_sha256 TEXT NOT NULL,
        sample_count INTEGER NOT NULL,
        subject_safe INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS promotions (
        promotion_id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        run_name TEXT NOT NULL,
        benchmark_id TEXT,
        policy_name TEXT NOT NULL,
        approved INTEGER NOT NULL,
        output_path TEXT,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        experiment_name TEXT NOT NULL,
        run_name TEXT NOT NULL,
        dataset TEXT NOT NULL,
        primary_split TEXT NOT NULL,
        tags_json TEXT NOT NULL,
        best_checkpoint_path TEXT,
        metrics_json TEXT NOT NULL,
        summary_path TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scans (
        scan_id TEXT PRIMARY KEY,
        subject_id TEXT,
        session_id TEXT,
        source_path TEXT NOT NULL,
        file_format TEXT,
        file_size_bytes INTEGER,
        dataset TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS inference_logs (
        inference_id TEXT PRIMARY KEY,
        trace_id TEXT NOT NULL,
        scan_id TEXT,
        subject_id TEXT,
        session_id TEXT,
        model_name TEXT NOT NULL,
        checkpoint_path TEXT NOT NULL,
        output_path TEXT,
        predicted_label INTEGER,
        label_name TEXT,
        confidence_level TEXT,
        review_flag INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS longitudinal_reports (
        report_id TEXT PRIMARY KEY,
        subject_id TEXT NOT NULL,
        report_type TEXT NOT NULL,
        output_path TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_events (
        audit_id TEXT PRIMARY KEY,
        action TEXT NOT NULL,
        actor_id TEXT,
        subject_id TEXT,
        patient_id_hash TEXT,
        outcome TEXT NOT NULL,
        purpose TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        event_time TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_queue (
        review_id TEXT PRIMARY KEY,
        inference_id TEXT NOT NULL,
        trace_id TEXT NOT NULL,
        scan_id TEXT,
        subject_id TEXT,
        session_id TEXT,
        source_path TEXT,
        model_name TEXT NOT NULL,
        confidence_level TEXT,
        probability_score REAL,
        output_path TEXT,
        status TEXT NOT NULL,
        reason TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
)


def resolve_database_path(*, settings: AppSettings | None = None) -> Path:
    """Resolve the local metadata database path."""

    resolved_settings = settings or get_app_settings()
    if resolved_settings.database_path is not None:
        return Path(resolved_settings.database_path)
    return resolved_settings.project_root / "storage" / "alz_backend.sqlite3"


def connect_backend_storage(*, settings: AppSettings | None = None) -> sqlite3.Connection:
    """Open a SQLite connection and initialize the schema."""

    database_path = resolve_database_path(settings=settings)
    ensure_directory(database_path.parent)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    initialize_backend_storage(connection)
    return connection


def initialize_backend_storage(connection: sqlite3.Connection) -> None:
    """Create tables when they do not already exist."""

    with connection:
        for statement in DDL_STATEMENTS:
            connection.execute(statement)
