"""External-cohort manifest validation for separate non-OASIS evaluation evidence."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .base_dataset import canonicalize_optional_string, load_manifest_frame, parse_manifest_meta

EXTERNAL_COHORT_REQUIRED_COLUMNS = ("image", "dataset", "dataset_type")
RESERVED_INTERNAL_DATASET_NAMES = frozenset({"oasis1", "kaggle", "kaggle_alz"})


class ExternalCohortManifestError(ValueError):
    """Raised when an external evaluation manifest is unsafe or ambiguous."""


@dataclass(slots=True)
class ExternalCohortManifestSummary:
    """Compact external-cohort manifest summary for evaluation reporting."""

    manifest_path: str
    manifest_hash_sha256: str
    dataset_name: str
    dataset_type: str
    sample_count: int
    labeled_sample_count: int
    subject_count: int | None
    subject_id_available: bool
    label_distribution: dict[str, int] = field(default_factory=dict)
    source_label_name_distribution: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_single_value(series: pd.Series, *, field_name: str) -> str:
    """Require one non-empty normalized value for a manifest field."""

    normalized = [value for value in series.map(canonicalize_optional_string).tolist() if value]
    unique_values = sorted(set(normalized))
    if not unique_values:
        raise ExternalCohortManifestError(
            f"External manifest must define at least one non-empty {field_name!r} value."
        )
    if len(unique_values) > 1:
        raise ExternalCohortManifestError(
            f"External manifest mixes multiple {field_name} values: {unique_values}. "
            "Keep one cohort per manifest so external evidence stays traceable."
        )
    return unique_values[0]


def _validate_dataset_name(dataset_name: str) -> None:
    """Reject internal dataset names so external evidence remains separate."""

    if dataset_name.lower() in RESERVED_INTERNAL_DATASET_NAMES:
        raise ExternalCohortManifestError(
            f"External manifest dataset={dataset_name!r} collides with an internal dataset name. "
            "Use the dedicated OASIS or Kaggle evaluation path instead."
        )


def _coerce_binary_label(raw_value: Any) -> int:
    """Normalize one manifest label to the explicit OASIS-binary convention."""

    if raw_value is None or pd.isna(raw_value):
        raise ExternalCohortManifestError(
            "External evaluation manifest is missing labels. "
            "For this binary OASIS model, labels must already be mapped explicitly to 0 or 1."
        )
    try:
        value = int(float(raw_value))
    except (TypeError, ValueError) as error:
        raise ExternalCohortManifestError(
            f"External evaluation label {raw_value!r} is not a valid numeric binary label."
        ) from error
    if value not in {0, 1}:
        raise ExternalCohortManifestError(
            f"External evaluation label {raw_value!r} is outside the supported binary set {{0, 1}}."
        )
    return value


def _count_values(values: list[str | None]) -> dict[str, int]:
    """Count normalized values without extra dependencies."""

    counts: dict[str, int] = {}
    for value in values:
        key = "null" if value in {None, ""} else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_external_cohort_manifest(
    manifest_path: Path,
    *,
    require_labels: bool = True,
    expected_dataset_type: str | None = None,
) -> pd.DataFrame:
    """Load and validate one external-cohort manifest for separate evaluation."""

    required_columns = list(EXTERNAL_COHORT_REQUIRED_COLUMNS)
    if require_labels:
        required_columns.append("label")
    frame = load_manifest_frame(manifest_path, required_columns=required_columns)
    frame = frame.copy().reset_index(drop=True)

    dataset_name = _validate_single_value(frame["dataset"], field_name="dataset")
    _validate_dataset_name(dataset_name)
    dataset_type = _validate_single_value(frame["dataset_type"], field_name="dataset_type")
    if expected_dataset_type is not None and dataset_type != expected_dataset_type:
        raise ExternalCohortManifestError(
            f"External manifest dataset_type={dataset_type!r} does not match the required "
            f"{expected_dataset_type!r} for this 3D evaluation path."
        )

    missing_paths: list[str] = []
    for image_value in frame["image"].tolist():
        image_path = Path(str(image_value))
        if not image_path.exists():
            missing_paths.append(str(image_path))
    if missing_paths:
        raise ExternalCohortManifestError(
            f"External manifest references missing image paths. Examples: {missing_paths[:5]}"
        )

    frame["dataset"] = dataset_name
    frame["dataset_type"] = dataset_type
    if "label" in frame.columns:
        label_values = frame["label"].tolist()
        if any(value is None or pd.isna(value) for value in label_values):
            raise ExternalCohortManifestError(
                "External evaluation manifest contains null labels. "
                "This path requires explicit binary labels for every row."
            )
        frame["label"] = frame["label"].apply(_coerce_binary_label)

    for column_name in ("label_name", "subject_id", "session_id", "scan_timestamp", "meta"):
        if column_name not in frame.columns:
            frame[column_name] = None
    return frame


def summarize_external_cohort_manifest(
    manifest_path: Path,
    *,
    require_labels: bool = True,
    expected_dataset_type: str | None = None,
) -> ExternalCohortManifestSummary:
    """Summarize one external-cohort manifest for reporting and auditability."""

    frame = load_external_cohort_manifest(
        manifest_path,
        require_labels=require_labels,
        expected_dataset_type=expected_dataset_type,
    )
    subject_values = [
        value
        for value in frame["subject_id"].map(canonicalize_optional_string).tolist()
        if value is not None
    ]
    subject_id_available = bool(subject_values)
    warnings: list[str] = []
    notes = [
        "External cohort evidence is kept separate from OASIS internal validation outputs.",
        "No automatic label harmonization was applied. The manifest must already encode 0=control/nondemented and 1=AD-like/demented for this binary model.",
    ]
    if not subject_id_available:
        warnings.append(
            "Manifest does not expose subject_id values, so subject-level leakage review is weaker for this cohort."
        )
    if frame["dataset_type"].iloc[0] != "3d_volumes":
        warnings.append(
            "This external manifest is not marked as 3d_volumes. It is not suitable for the current OASIS 3D classifier."
        )
    source_label_names = frame["label_name"].map(canonicalize_optional_string).tolist()
    if all(value is None for value in source_label_names):
        notes.append(
            "Original source label names were not provided, so only the normalized binary labels are tracked."
        )

    return ExternalCohortManifestSummary(
        manifest_path=str(manifest_path),
        manifest_hash_sha256=_sha256_file(manifest_path),
        dataset_name=str(frame["dataset"].iloc[0]),
        dataset_type=str(frame["dataset_type"].iloc[0]),
        sample_count=int(len(frame)),
        labeled_sample_count=int(frame["label"].notna().sum()) if "label" in frame.columns else 0,
        subject_count=len(set(subject_values)) if subject_id_available else None,
        subject_id_available=subject_id_available,
        label_distribution=_count_values([str(int(value)) for value in frame["label"].tolist()]) if "label" in frame.columns else {},
        source_label_name_distribution=_count_values(source_label_names),
        warnings=warnings,
        notes=notes,
    )


def build_external_cohort_records(
    manifest_path: Path,
    *,
    expected_dataset_type: str = "3d_volumes",
) -> tuple[list[dict[str, Any]], ExternalCohortManifestSummary]:
    """Convert one validated external manifest into MONAI-ready inference records."""

    frame = load_external_cohort_manifest(
        manifest_path,
        require_labels=True,
        expected_dataset_type=expected_dataset_type,
    )
    summary = summarize_external_cohort_manifest(
        manifest_path,
        require_labels=True,
        expected_dataset_type=expected_dataset_type,
    )
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        image_path = Path(str(row.image))
        source_label_name = canonicalize_optional_string(getattr(row, "label_name", None))
        meta = parse_manifest_meta(getattr(row, "meta", None))
        meta.setdefault("source_dataset_name", str(row.dataset))
        meta.setdefault("source_dataset_type", str(row.dataset_type))
        if source_label_name is not None:
            meta.setdefault("source_label_name", source_label_name)
        records.append(
            {
                "image": str(image_path),
                "image_path": str(image_path),
                "label": int(row.label),
                "label_name": source_label_name,
                "subject_id": canonicalize_optional_string(getattr(row, "subject_id", None)) or "",
                "session_id": canonicalize_optional_string(getattr(row, "session_id", None)) or "",
                "scan_timestamp": canonicalize_optional_string(getattr(row, "scan_timestamp", None)) or "",
                "dataset": str(row.dataset),
                "dataset_type": str(row.dataset_type),
                "meta": meta,
            }
        )
    return records, summary
