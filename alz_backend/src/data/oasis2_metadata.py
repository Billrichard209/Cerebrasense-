"""Metadata template and merge helpers for the future OASIS-2 labeled adapter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings
from src.utils.io_utils import ensure_directory

from .base_dataset import canonicalize_optional_string, parse_manifest_meta
from .oasis2_dataset import load_oasis2_session_manifest, resolve_oasis2_manifest_path

OASIS2_METADATA_TEMPLATE_COLUMNS = (
    "subject_id",
    "session_id",
    "visit_number",
    "scan_timestamp",
    "age_at_visit",
    "sex",
    "clinical_status",
    "cdr_global",
    "mmse",
    "diagnosis_label",
    "diagnosis_label_name",
    "metadata_source",
    "metadata_complete",
    "split_group_hint",
    "notes",
)
OASIS2_METADATA_REQUIRED_KEY_COLUMNS = ("subject_id", "session_id", "visit_number")
OASIS2_METADATA_REQUIRED_LABEL_COLUMNS = ("diagnosis_label", "diagnosis_label_name")
OASIS2_OFFICIAL_DEMOGRAPHICS_REQUIRED_COLUMNS = (
    "Subject ID",
    "MRI ID",
    "Group",
    "Visit",
    "Age",
    "M/F",
    "MMSE",
    "CDR",
)
OASIS2_OFFICIAL_DEMOGRAPHICS_URL = (
    "https://sites.wustl.edu/oasisbrains/files/2024/03/"
    "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"
)
OASIS2_BINARY_LABEL_NAME_MAP = {0: "nondemented", 1: "demented"}


@dataclass(slots=True)
class OASIS2MetadataTemplateResult:
    """Artifacts produced by the OASIS-2 metadata template builder."""

    template_path: Path
    summary_path: Path
    row_count: int
    unique_subject_count: int
    unique_session_count: int


@dataclass(slots=True)
class OASIS2MetadataAdapterSummary:
    """Status summary for the OASIS-2 metadata merge adapter."""

    generated_at: str
    manifest_path: str
    metadata_path: str
    merged_manifest_path: str
    record_count: int
    metadata_row_count: int
    matched_metadata_row_count: int
    rows_with_candidate_labels: int
    rows_missing_candidate_labels: int
    metadata_complete_row_count: int
    ready_for_labeled_manifest: bool
    notes: list[str]
    recommendations: list[str]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return asdict(self)


@dataclass(slots=True)
class OASIS2OfficialDemographicsImportSummary:
    """Status summary for importing the official OASIS-2 demographics sheet."""

    generated_at: str
    demographics_path: str
    metadata_path: str
    template_row_count: int
    demographics_row_count: int
    matched_row_count: int
    labeled_row_count: int
    metadata_complete_row_count: int
    converted_group_row_count: int
    group_cdr_disagreement_row_count: int
    unmatched_template_row_count: int
    unmatched_demographics_row_count: int
    notes: list[str]
    recommendations: list[str]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return asdict(self)


def resolve_oasis2_metadata_template_path(
    settings: AppSettings,
    *,
    metadata_path: Path | None = None,
) -> Path:
    """Resolve the OASIS-2 metadata template path."""

    if metadata_path is not None:
        return metadata_path
    return settings.data_root / "interim" / "oasis2_metadata_template.csv"


def resolve_oasis2_labeled_prep_manifest_path(
    settings: AppSettings,
    *,
    output_path: Path | None = None,
) -> Path:
    """Resolve the merged OASIS-2 labeled-prep manifest path."""

    if output_path is not None:
        return output_path
    return settings.data_root / "interim" / "oasis2_labeled_prep_manifest.csv"


def _normalize_optional_bool(value: Any) -> bool:
    """Normalize common truthy values used in CSV-based metadata templates."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "complete", "done"}


def _normalize_optional_float(value: Any) -> float | None:
    """Convert a metadata value to float when possible."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_optional_int(value: Any) -> int | None:
    """Convert a metadata value to int when possible."""

    float_value = _normalize_optional_float(value)
    if float_value is None:
        return None
    return int(float_value)


def _load_tabular_metadata(path: Path) -> pd.DataFrame:
    """Load one OASIS-2 metadata table from CSV or Excel."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, engine="openpyxl")


def map_oasis2_binary_label_from_cdr(raw_value: Any) -> tuple[int, str]:
    """Map one visit-level OASIS-2 CDR score onto the current binary label policy."""

    cdr_global = _normalize_optional_float(raw_value)
    if cdr_global is None:
        raise ValueError("OASIS-2 official demographics row is missing CDR.")
    if cdr_global == 0.0:
        return 0, OASIS2_BINARY_LABEL_NAME_MAP[0]
    if cdr_global > 0.0:
        return 1, OASIS2_BINARY_LABEL_NAME_MAP[1]
    raise ValueError(f"Unsupported negative OASIS-2 CDR value: {raw_value!r}")


def load_oasis2_official_demographics(
    demographics_path: Path,
) -> pd.DataFrame:
    """Load and normalize the official OASIS-2 longitudinal demographics table."""

    resolved_path = demographics_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"OASIS-2 official demographics file not found: {resolved_path}")
    frame = _load_tabular_metadata(resolved_path)
    missing_columns = set(OASIS2_OFFICIAL_DEMOGRAPHICS_REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "OASIS-2 official demographics sheet is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError("OASIS-2 official demographics sheet is empty.")

    working = frame.copy()
    working["subject_id"] = working["Subject ID"].map(canonicalize_optional_string)
    working["session_id"] = working["MRI ID"].map(canonicalize_optional_string)
    working["visit_number"] = working["Visit"].map(_normalize_optional_int)
    if working[list(OASIS2_METADATA_REQUIRED_KEY_COLUMNS)].isna().any().any():
        raise ValueError(
            "OASIS-2 official demographics sheet has empty subject/session/visit keys."
        )
    if working[list(OASIS2_METADATA_REQUIRED_KEY_COLUMNS)].duplicated().any():
        raise ValueError(
            "OASIS-2 official demographics sheet contains duplicate subject/session/visit rows."
        )
    return working


def build_oasis2_metadata_template(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
) -> OASIS2MetadataTemplateResult:
    """Build a CSV template for future OASIS-2 visit/clinical metadata mapping."""

    resolved_settings = settings or AppSettings.from_env()
    frame = load_oasis2_session_manifest(resolved_settings, manifest_path=manifest_path)
    resolved_output_path = resolve_oasis2_metadata_template_path(resolved_settings, metadata_path=output_path)
    ensure_directory(resolved_output_path.parent)
    summary_path = resolved_output_path.with_name("oasis2_metadata_template_summary.json")

    template_rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        template_rows.append(
            {
                "subject_id": canonicalize_optional_string(getattr(row, "subject_id", None)),
                "session_id": canonicalize_optional_string(getattr(row, "session_id", None)),
                "visit_number": None if pd.isna(getattr(row, "visit_number", None)) else int(float(getattr(row, "visit_number"))),
                "scan_timestamp": canonicalize_optional_string(getattr(row, "scan_timestamp", None)),
                "age_at_visit": None,
                "sex": None,
                "clinical_status": None,
                "cdr_global": None,
                "mmse": None,
                "diagnosis_label": None,
                "diagnosis_label_name": None,
                "metadata_source": None,
                "metadata_complete": False,
                "split_group_hint": canonicalize_optional_string(getattr(row, "subject_id", None)),
                "notes": None,
            }
        )

    template_frame = pd.DataFrame(template_rows, columns=OASIS2_METADATA_TEMPLATE_COLUMNS)
    template_frame.to_csv(resolved_output_path, index=False)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "template_path": str(resolved_output_path),
        "row_count": int(len(template_frame)),
        "unique_subject_count": int(template_frame["subject_id"].nunique()),
        "unique_session_count": int(template_frame["session_id"].nunique()),
        "required_key_columns": list(OASIS2_METADATA_REQUIRED_KEY_COLUMNS),
        "required_label_columns": list(OASIS2_METADATA_REQUIRED_LABEL_COLUMNS),
        "notes": [
            "Fill this template with explicit visit- or clinical-level metadata before attempting labeled OASIS-2 work.",
            "Do not infer labels automatically from this template; keep every mapping explicit and auditable.",
            "split_group_hint defaults to subject_id so future subject-safe split logic starts from a safe grouping key.",
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return OASIS2MetadataTemplateResult(
        template_path=resolved_output_path,
        summary_path=summary_path,
        row_count=int(len(template_frame)),
        unique_subject_count=summary_payload["unique_subject_count"],
        unique_session_count=summary_payload["unique_session_count"],
    )


def load_oasis2_metadata_template(
    settings: AppSettings | None = None,
    *,
    metadata_path: Path | None = None,
) -> pd.DataFrame:
    """Load the OASIS-2 metadata template or a filled copy of it."""

    resolved_settings = settings or AppSettings.from_env()
    resolved_metadata_path = resolve_oasis2_metadata_template_path(resolved_settings, metadata_path=metadata_path)
    frame = pd.read_csv(resolved_metadata_path)
    missing = set(OASIS2_METADATA_TEMPLATE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"OASIS-2 metadata template is missing required columns: {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError("OASIS-2 metadata template is empty.")
    return frame


def _load_or_build_oasis2_metadata_template(
    settings: AppSettings,
    *,
    manifest_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load the metadata template or build it automatically when missing."""

    resolved_metadata_path = resolve_oasis2_metadata_template_path(settings, metadata_path=metadata_path)
    if not resolved_metadata_path.exists():
        build_oasis2_metadata_template(
            settings,
            manifest_path=manifest_path,
            output_path=resolved_metadata_path,
        )
    return load_oasis2_metadata_template(settings, metadata_path=resolved_metadata_path), resolved_metadata_path


def merge_oasis2_metadata_template(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
    metadata_path: Path | None = None,
    output_path: Path | None = None,
) -> OASIS2MetadataAdapterSummary:
    """Merge a filled OASIS-2 metadata template into the unlabeled session manifest."""

    resolved_settings = settings or AppSettings.from_env()
    manifest_frame = load_oasis2_session_manifest(resolved_settings, manifest_path=manifest_path)
    resolved_manifest_path = resolve_oasis2_manifest_path(resolved_settings, manifest_path=manifest_path)
    metadata_frame, resolved_metadata_path = _load_or_build_oasis2_metadata_template(
        resolved_settings,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    metadata_frame = metadata_frame.copy()
    merged_output_path = resolve_oasis2_labeled_prep_manifest_path(resolved_settings, output_path=output_path)
    ensure_directory(merged_output_path.parent)

    metadata_frame["subject_id"] = metadata_frame["subject_id"].map(canonicalize_optional_string)
    metadata_frame["session_id"] = metadata_frame["session_id"].map(canonicalize_optional_string)
    metadata_frame["visit_number"] = metadata_frame["visit_number"].map(_normalize_optional_int)
    metadata_frame["metadata_complete"] = metadata_frame["metadata_complete"].map(_normalize_optional_bool)
    if metadata_frame[list(OASIS2_METADATA_REQUIRED_KEY_COLUMNS)].duplicated().any():
        raise ValueError("OASIS-2 metadata template contains duplicate subject/session/visit rows.")

    metadata_lookup = metadata_frame.set_index(list(OASIS2_METADATA_REQUIRED_KEY_COLUMNS), drop=False)

    merged_rows: list[dict[str, Any]] = []
    matched_metadata_row_count = 0
    rows_with_candidate_labels = 0
    metadata_complete_row_count = 0

    for row in manifest_frame.itertuples(index=False):
        key = (
            canonicalize_optional_string(getattr(row, "subject_id", None)),
            canonicalize_optional_string(getattr(row, "session_id", None)),
            None if pd.isna(getattr(row, "visit_number", None)) else int(float(getattr(row, "visit_number"))),
        )
        metadata_row = None
        if key in metadata_lookup.index:
            metadata_row = metadata_lookup.loc[key]
            if isinstance(metadata_row, pd.DataFrame):
                raise ValueError(f"OASIS-2 metadata template produced multiple rows for {key}")
            matched_metadata_row_count += 1

        label_value = None if pd.isna(getattr(row, "label", None)) else int(float(getattr(row, "label")))
        label_name = canonicalize_optional_string(getattr(row, "label_name", None))
        metadata_complete = False
        merged_meta = parse_manifest_meta(getattr(row, "meta"))

        if metadata_row is not None:
            diagnosis_label = _normalize_optional_int(metadata_row["diagnosis_label"])
            diagnosis_label_name = canonicalize_optional_string(metadata_row["diagnosis_label_name"])
            metadata_complete = bool(metadata_row["metadata_complete"])
            metadata_fields = {
                "age_at_visit": _normalize_optional_float(metadata_row["age_at_visit"]),
                "sex": canonicalize_optional_string(metadata_row["sex"]),
                "clinical_status": canonicalize_optional_string(metadata_row["clinical_status"]),
                "cdr_global": _normalize_optional_float(metadata_row["cdr_global"]),
                "mmse": _normalize_optional_float(metadata_row["mmse"]),
                "metadata_source": canonicalize_optional_string(metadata_row["metadata_source"]),
                "metadata_complete": metadata_complete,
                "split_group_hint": canonicalize_optional_string(metadata_row["split_group_hint"]),
                "metadata_notes": canonicalize_optional_string(metadata_row["notes"]),
            }
            if diagnosis_label is not None and diagnosis_label_name is not None:
                label_value = diagnosis_label
                label_name = diagnosis_label_name
                rows_with_candidate_labels += 1
            if metadata_complete:
                metadata_complete_row_count += 1
            merged_meta["oasis2_metadata"] = {key: value for key, value in metadata_fields.items() if value is not None}
            merged_meta["oasis2_metadata_template_path"] = str(resolved_metadata_path)
            merged_meta["oasis2_metadata_match_key"] = {
                "subject_id": key[0],
                "session_id": key[1],
                "visit_number": key[2],
            }

        merged_rows.append(
            {
                "image": str(getattr(row, "image")),
                "label": label_value,
                "label_name": label_name,
                "subject_id": key[0],
                "session_id": key[1],
                "visit_number": key[2],
                "scan_timestamp": canonicalize_optional_string(getattr(row, "scan_timestamp", None)),
                "dataset": canonicalize_optional_string(getattr(row, "dataset", None)),
                "dataset_type": canonicalize_optional_string(getattr(row, "dataset_type", None)),
                "meta": json.dumps(merged_meta, ensure_ascii=True, sort_keys=True),
            }
        )

    merged_frame = pd.DataFrame(merged_rows)
    merged_frame.to_csv(merged_output_path, index=False)
    rows_missing_candidate_labels = int(len(merged_frame) - rows_with_candidate_labels)
    ready_for_labeled_manifest = (
        len(merged_frame) > 0
        and matched_metadata_row_count == len(merged_frame)
        and rows_missing_candidate_labels == 0
    )

    return OASIS2MetadataAdapterSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(resolved_manifest_path),
        metadata_path=str(resolved_metadata_path),
        merged_manifest_path=str(merged_output_path),
        record_count=int(len(merged_frame)),
        metadata_row_count=int(len(metadata_frame)),
        matched_metadata_row_count=matched_metadata_row_count,
        rows_with_candidate_labels=rows_with_candidate_labels,
        rows_missing_candidate_labels=rows_missing_candidate_labels,
        metadata_complete_row_count=metadata_complete_row_count,
        ready_for_labeled_manifest=ready_for_labeled_manifest,
        notes=[
            "This adapter merges explicit metadata rows into the unlabeled OASIS-2 session manifest.",
            "A merged manifest is not automatically training-ready just because it exists; subject-safe split and label-policy checks still matter.",
            "Missing diagnosis labels are allowed during onboarding so the merge path can be validated before real metadata arrives.",
        ],
        recommendations=[
            "Fill diagnosis_label and diagnosis_label_name explicitly before attempting any supervised OASIS-2 study.",
            "Keep split_group_hint aligned to subject-safe grouping so future split logic stays patient-safe.",
            "Treat metadata_complete as a human-reviewed status flag, not as inferred automation output.",
        ],
    )


def import_oasis2_official_demographics_into_metadata_template(
    demographics_path: Path,
    *,
    settings: AppSettings | None = None,
    manifest_path: Path | None = None,
    metadata_path: Path | None = None,
    output_path: Path | None = None,
    overwrite_existing: bool = False,
    metadata_source_name: str | None = None,
) -> OASIS2OfficialDemographicsImportSummary:
    """Fill the OASIS-2 metadata template from the official longitudinal demographics sheet.

    The binary training target is derived from visit-level CDR so converted subjects
    keep their early nondemented visits instead of being forced positive across the
    whole subject history.
    """

    resolved_settings = settings or AppSettings.from_env()
    template_frame, resolved_metadata_path = _load_or_build_oasis2_metadata_template(
        resolved_settings,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    demographics_frame = load_oasis2_official_demographics(demographics_path)
    output_metadata_path = resolve_oasis2_metadata_template_path(
        resolved_settings,
        metadata_path=output_path or resolved_metadata_path,
    )
    ensure_directory(output_metadata_path.parent)

    working = template_frame.copy()
    working["subject_id"] = working["subject_id"].map(canonicalize_optional_string)
    working["session_id"] = working["session_id"].map(canonicalize_optional_string)
    working["visit_number"] = working["visit_number"].map(_normalize_optional_int)
    working["metadata_complete"] = working["metadata_complete"].map(_normalize_optional_bool)

    lookup = demographics_frame.set_index(list(OASIS2_METADATA_REQUIRED_KEY_COLUMNS), drop=False)
    matched_row_count = 0
    labeled_row_count = 0
    metadata_complete_row_count = 0
    converted_group_row_count = 0
    group_cdr_disagreement_row_count = 0
    unmatched_demographics_keys = set(lookup.index.tolist())

    filled_rows: list[dict[str, Any]] = []
    for row in working.to_dict(orient="records"):
        key = (
            canonicalize_optional_string(row.get("subject_id")),
            canonicalize_optional_string(row.get("session_id")),
            _normalize_optional_int(row.get("visit_number")),
        )
        demographics_row = None
        if key in lookup.index:
            demographics_row = lookup.loc[key]
            if isinstance(demographics_row, pd.DataFrame):
                raise ValueError(f"OASIS-2 official demographics produced multiple rows for {key}.")
            matched_row_count += 1
            unmatched_demographics_keys.discard(key)

        updated_row = dict(row)
        if demographics_row is not None:
            clinical_group = canonicalize_optional_string(demographics_row["Group"])
            cdr_global = _normalize_optional_float(demographics_row["CDR"])
            diagnosis_label, diagnosis_label_name = map_oasis2_binary_label_from_cdr(demographics_row["CDR"])

            existing_label = _normalize_optional_int(updated_row.get("diagnosis_label"))
            existing_label_name = canonicalize_optional_string(updated_row.get("diagnosis_label_name"))
            if overwrite_existing or existing_label is None:
                updated_row["diagnosis_label"] = diagnosis_label
            if overwrite_existing or existing_label_name is None:
                updated_row["diagnosis_label_name"] = diagnosis_label_name

            for field_name, field_value in (
                ("age_at_visit", _normalize_optional_float(demographics_row["Age"])),
                ("sex", canonicalize_optional_string(demographics_row["M/F"])),
                ("clinical_status", clinical_group),
                ("cdr_global", cdr_global),
                ("mmse", _normalize_optional_float(demographics_row["MMSE"])),
                (
                    "metadata_source",
                    metadata_source_name
                    or f"official_oasis2_demographics:{Path(demographics_path).name}",
                ),
                ("split_group_hint", canonicalize_optional_string(updated_row.get("split_group_hint")) or key[0]),
            ):
                if overwrite_existing or canonicalize_optional_string(updated_row.get(field_name)) is None:
                    updated_row[field_name] = field_value

            if clinical_group == "Converted":
                converted_group_row_count += 1
            if clinical_group == "Nondemented" and diagnosis_label == 1:
                group_cdr_disagreement_row_count += 1

            note_parts = []
            existing_notes = canonicalize_optional_string(updated_row.get("notes"))
            if existing_notes:
                note_parts.append(existing_notes)
            note_parts.append("binary_label_policy=derived_from_cdr_global")
            if clinical_group:
                note_parts.append(f"official_group={clinical_group}")
            updated_row["notes"] = "; ".join(dict.fromkeys(note_parts))
            updated_row["metadata_complete"] = True

        if (
            _normalize_optional_int(updated_row.get("diagnosis_label")) is not None
            and canonicalize_optional_string(updated_row.get("diagnosis_label_name")) is not None
        ):
            labeled_row_count += 1
        if _normalize_optional_bool(updated_row.get("metadata_complete")):
            metadata_complete_row_count += 1

        filled_rows.append(updated_row)

    filled_frame = pd.DataFrame(filled_rows, columns=OASIS2_METADATA_TEMPLATE_COLUMNS)
    filled_frame.to_csv(output_metadata_path, index=False)

    return OASIS2OfficialDemographicsImportSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        demographics_path=str(Path(demographics_path).expanduser().resolve()),
        metadata_path=str(output_metadata_path),
        template_row_count=int(len(working)),
        demographics_row_count=int(len(demographics_frame)),
        matched_row_count=matched_row_count,
        labeled_row_count=labeled_row_count,
        metadata_complete_row_count=metadata_complete_row_count,
        converted_group_row_count=converted_group_row_count,
        group_cdr_disagreement_row_count=group_cdr_disagreement_row_count,
        unmatched_template_row_count=int(len(working) - matched_row_count),
        unmatched_demographics_row_count=int(len(unmatched_demographics_keys)),
        notes=[
            "The official OASIS-2 longitudinal demographics sheet was merged onto the metadata template by exact subject/session/visit keys.",
            "Binary session labels were derived from visit-level CDR so converted subjects keep their early nondemented visits.",
            "The original OASIS Group field is preserved in clinical_status and notes for auditability.",
        ],
        recommendations=[
            "Review rows where the official Group field disagrees with the visit-level binary CDR mapping before presenting OASIS-2 results clinically.",
            "Keep the current binary OASIS-2 training path scoped to CDR-derived session labels until a richer longitudinal objective is implemented.",
            "Rerun the metadata adapter and supervised readiness gate after this import before starting training.",
        ],
    )


def save_oasis2_metadata_adapter_summary(
    summary: OASIS2MetadataAdapterSummary,
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis2_metadata_adapter_status",
) -> tuple[Path, Path]:
    """Save the OASIS-2 metadata adapter summary as JSON and Markdown."""

    resolved_settings = settings or AppSettings.from_env()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "onboarding")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    json_path.write_text(json.dumps(summary.to_payload(), indent=2), encoding="utf-8")

    lines = [
        "# OASIS-2 Metadata Adapter Status",
        "",
        f"- generated_at: {summary.generated_at}",
        f"- manifest_path: {summary.manifest_path}",
        f"- metadata_path: {summary.metadata_path}",
        f"- merged_manifest_path: {summary.merged_manifest_path}",
        f"- record_count: {summary.record_count}",
        f"- metadata_row_count: {summary.metadata_row_count}",
        f"- matched_metadata_row_count: {summary.matched_metadata_row_count}",
        f"- rows_with_candidate_labels: {summary.rows_with_candidate_labels}",
        f"- rows_missing_candidate_labels: {summary.rows_missing_candidate_labels}",
        f"- metadata_complete_row_count: {summary.metadata_complete_row_count}",
        f"- ready_for_labeled_manifest: {summary.ready_for_labeled_manifest}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.notes)
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in summary.recommendations)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def save_oasis2_official_demographics_import_summary(
    summary: OASIS2OfficialDemographicsImportSummary,
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis2_official_demographics_import",
) -> tuple[Path, Path]:
    """Save the OASIS-2 official demographics import summary as JSON and Markdown."""

    resolved_settings = settings or AppSettings.from_env()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "onboarding")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    json_path.write_text(json.dumps(summary.to_payload(), indent=2), encoding="utf-8")

    lines = [
        "# OASIS-2 Official Demographics Import",
        "",
        f"- generated_at: {summary.generated_at}",
        f"- demographics_path: {summary.demographics_path}",
        f"- metadata_path: {summary.metadata_path}",
        f"- template_row_count: {summary.template_row_count}",
        f"- demographics_row_count: {summary.demographics_row_count}",
        f"- matched_row_count: {summary.matched_row_count}",
        f"- labeled_row_count: {summary.labeled_row_count}",
        f"- metadata_complete_row_count: {summary.metadata_complete_row_count}",
        f"- converted_group_row_count: {summary.converted_group_row_count}",
        f"- group_cdr_disagreement_row_count: {summary.group_cdr_disagreement_row_count}",
        f"- unmatched_template_row_count: {summary.unmatched_template_row_count}",
        f"- unmatched_demographics_row_count: {summary.unmatched_demographics_row_count}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.notes)
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in summary.recommendations)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
