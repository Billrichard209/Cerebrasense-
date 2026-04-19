"""Manifest adapter stub for local-first OASIS-2 onboarding."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings
from src.utils.io_utils import ensure_directory

from .base_dataset import (
    DatasetSample,
    canonicalize_optional_string,
    load_manifest_frame,
    parse_manifest_meta,
)
from .oasis2 import resolve_oasis2_source_layout

OASIS2_MANIFEST_COLUMNS = {
    "image",
    "label",
    "label_name",
    "subject_id",
    "session_id",
    "visit_number",
    "scan_timestamp",
    "dataset",
    "dataset_type",
    "meta",
}
OASIS2_DEFAULT_DATASET_TYPE = "3d_volumes"
OASIS2_ALLOWED_USES = (
    "subject_session_indexing",
    "preprocessing_previews",
    "transform_validation",
    "longitudinal_tracking_preparation",
    "structural_volumetric_workflows",
)
OASIS2_BLOCKED_USES = (
    "supervised_classification_training",
    "threshold_calibration",
    "baseline_promotion",
    "production_serving_default",
)


@dataclass(slots=True)
class OASIS2DatasetSpec:
    """Specification for the OASIS-2 onboarding branch."""

    name: str
    priority: str
    source_root: Path
    interim_root: Path
    subject_id_column: str
    visit_id_column: str
    label_policy: str
    dataset_type: str

    def to_snapshot(self) -> dict[str, object]:
        """Return a JSON-safe dictionary for reports and API responses."""

        return {
            "name": self.name,
            "priority": self.priority,
            "source_root": str(self.source_root),
            "source_exists": self.source_root.exists(),
            "interim_root": str(self.interim_root),
            "subject_id_column": self.subject_id_column,
            "visit_id_column": self.visit_id_column,
            "label_policy": self.label_policy,
            "dataset_type": self.dataset_type,
        }


@dataclass(slots=True)
class OASIS2AdapterSummary:
    """Summary for the first dedicated OASIS-2 manifest adapter stub."""

    generated_at: str
    manifest_path: str
    source_root: str
    source_resolution: str
    adapter_mode: str
    record_count: int
    unique_subject_count: int
    unique_session_count: int
    labeled_row_count: int
    unlabeled_row_count: int
    ready_for_supervised_training: bool
    allowed_uses: tuple[str, ...]
    blocked_uses: tuple[str, ...]
    notes: list[str]
    recommendations: list[str]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return asdict(self)


def build_oasis2_dataset_spec(
    settings: AppSettings | None = None,
    *,
    source_root: Path | None = None,
) -> OASIS2DatasetSpec:
    """Build the OASIS-2 onboarding dataset specification."""

    resolved_settings = settings or AppSettings.from_env()
    layout = resolve_oasis2_source_layout(resolved_settings, source_root=source_root)
    return OASIS2DatasetSpec(
        name="oasis2_onboarding",
        priority="future_longitudinal_extension",
        source_root=layout.source_root,
        interim_root=resolved_settings.data_root / "interim",
        subject_id_column="subject_id",
        visit_id_column="session_id",
        label_policy="unlabeled_session_manifest_only",
        dataset_type=OASIS2_DEFAULT_DATASET_TYPE,
    )


def resolve_oasis2_manifest_path(
    settings: AppSettings,
    *,
    manifest_path: Path | None = None,
) -> Path:
    """Resolve the OASIS-2 unlabeled session manifest path."""

    if manifest_path is not None:
        return manifest_path
    return settings.data_root / "interim" / "oasis2_session_manifest.csv"


def load_oasis2_session_manifest(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Load the unlabeled OASIS-2 session manifest."""

    resolved_settings = settings or AppSettings.from_env()
    resolved_path = resolve_oasis2_manifest_path(resolved_settings, manifest_path=manifest_path)
    return load_manifest_frame(
        resolved_path,
        required_columns=OASIS2_MANIFEST_COLUMNS,
        default_dataset_type=OASIS2_DEFAULT_DATASET_TYPE,
    )


def build_oasis2_monai_records(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
    require_labels: bool = False,
) -> list[dict[str, Any]]:
    """Convert the OASIS-2 unlabeled session manifest into MONAI-style records."""

    frame = load_oasis2_session_manifest(settings, manifest_path=manifest_path)
    records: list[dict[str, Any]] = []

    for row in frame.itertuples(index=False):
        image_path = Path(row.image)
        if not image_path.exists():
            raise FileNotFoundError(f"OASIS-2 manifest image path does not exist: {image_path}")

        label_value: int | None = None
        if not pd.isna(row.label):
            label_value = int(float(row.label))
        elif require_labels:
            raise ValueError(
                "OASIS-2 adapter stub only supports unlabeled onboarding right now. "
                "Do not use it for supervised training until a dedicated labeled adapter exists."
            )

        meta = parse_manifest_meta(getattr(row, "meta"))
        visit_number_raw = getattr(row, "visit_number", None)
        visit_number = None if pd.isna(visit_number_raw) else int(float(visit_number_raw))
        if visit_number is not None:
            meta.setdefault("visit_number", visit_number)
        meta.setdefault("oasis2_adapter_mode", "unlabeled_structural_longitudinal_stub")
        meta.setdefault("label_policy", "unlabeled_session_manifest_only")

        sample = DatasetSample(
            sample_id=canonicalize_optional_string(getattr(row, "session_id", None)) or image_path.stem,
            image_path=image_path,
            label=label_value,
            label_name=canonicalize_optional_string(getattr(row, "label_name", None)),
            subject_id=canonicalize_optional_string(getattr(row, "subject_id", None)),
            session_id=canonicalize_optional_string(getattr(row, "session_id", None)),
            dataset=canonicalize_optional_string(getattr(row, "dataset", None)),
            dataset_type=canonicalize_optional_string(getattr(row, "dataset_type", None)) or OASIS2_DEFAULT_DATASET_TYPE,
            scan_timestamp=canonicalize_optional_string(getattr(row, "scan_timestamp", None)),
            meta=meta,
        )
        record = sample.to_monai_record()
        record["visit_number"] = visit_number
        records.append(record)

    return records


def build_oasis2_adapter_summary(
    settings: AppSettings | None = None,
    *,
    source_root: Path | None = None,
    manifest_path: Path | None = None,
) -> OASIS2AdapterSummary:
    """Build a status summary for the dedicated OASIS-2 manifest adapter stub."""

    resolved_settings = settings or AppSettings.from_env()
    layout = resolve_oasis2_source_layout(resolved_settings, source_root=source_root)
    resolved_manifest_path = resolve_oasis2_manifest_path(resolved_settings, manifest_path=manifest_path)
    frame = load_oasis2_session_manifest(resolved_settings, manifest_path=resolved_manifest_path)
    labeled_row_count = int(frame["label"].notna().sum())
    unlabeled_row_count = int(frame["label"].isna().sum())
    ready_for_supervised_training = unlabeled_row_count == 0 and labeled_row_count > 0

    return OASIS2AdapterSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(resolved_manifest_path),
        source_root=str(layout.source_root),
        source_resolution=layout.source_resolution,
        adapter_mode="unlabeled_structural_longitudinal_stub",
        record_count=int(len(frame)),
        unique_subject_count=int(frame["subject_id"].nunique()),
        unique_session_count=int(frame["session_id"].nunique()),
        labeled_row_count=labeled_row_count,
        unlabeled_row_count=unlabeled_row_count,
        ready_for_supervised_training=ready_for_supervised_training,
        allowed_uses=OASIS2_ALLOWED_USES,
        blocked_uses=OASIS2_BLOCKED_USES,
        notes=[
            "This is the first dedicated OASIS-2 adapter path in the repo.",
            "It is intentionally limited to unlabeled onboarding, structural workflows, and longitudinal preparation.",
            "It should not be used for supervised classification until explicit label and split policies exist.",
        ],
        recommendations=[
            "Keep using OASIS-2 locally for onboarding and structural preparation.",
            "Add visit/clinical metadata mapping before attempting supervised evaluation.",
            "Implement a labeled OASIS-2 adapter and subject-safe split policy before any training claims.",
        ],
    )


def save_oasis2_adapter_summary(
    summary: OASIS2AdapterSummary,
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis2_adapter_status",
) -> tuple[Path, Path]:
    """Save the OASIS-2 adapter summary as JSON and Markdown."""

    resolved_settings = settings or AppSettings.from_env()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "onboarding")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    json_path.write_text(json.dumps(summary.to_payload(), indent=2), encoding="utf-8")

    lines = [
        "# OASIS-2 Adapter Status",
        "",
        f"- generated_at: {summary.generated_at}",
        f"- manifest_path: {summary.manifest_path}",
        f"- source_root: {summary.source_root}",
        f"- source_resolution: {summary.source_resolution}",
        f"- adapter_mode: {summary.adapter_mode}",
        f"- record_count: {summary.record_count}",
        f"- unique_subject_count: {summary.unique_subject_count}",
        f"- unique_session_count: {summary.unique_session_count}",
        f"- labeled_row_count: {summary.labeled_row_count}",
        f"- unlabeled_row_count: {summary.unlabeled_row_count}",
        f"- ready_for_supervised_training: {summary.ready_for_supervised_training}",
        "",
        "## Allowed Uses",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.allowed_uses)
    lines.extend(["", "## Blocked Uses", ""])
    lines.extend(f"- {item}" for item in summary.blocked_uses)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {item}" for item in summary.notes)
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in summary.recommendations)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
