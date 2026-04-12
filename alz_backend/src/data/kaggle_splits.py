"""Leakage-aware split generation for the normalized Kaggle Alzheimer manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory

from .kaggle_alz import build_kaggle_manifest


class KaggleSplitError(ValueError):
    """Raised when the Kaggle manifest cannot be split safely."""


@dataclass(slots=True)
class KaggleSplitResult:
    """Artifacts produced by the Kaggle split builder."""

    split_assignments_path: Path
    train_manifest_path: Path
    val_manifest_path: Path
    test_manifest_path: Path
    summary_path: Path
    train_rows: int
    val_rows: int
    test_rows: int
    warnings: list[str] = field(default_factory=list)


def load_kaggle_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load a previously built Kaggle manifest."""

    if not manifest_path.exists():
        raise KaggleSplitError(f"Kaggle manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    required_columns = {
        "image",
        "label",
        "label_name",
        "subject_id",
        "scan_timestamp",
        "dataset",
        "dataset_type",
        "meta",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise KaggleSplitError(f"Kaggle manifest is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise KaggleSplitError("Kaggle manifest is empty.")
    return frame


def _parse_meta_payload(raw_value: Any) -> dict[str, Any]:
    """Parse a serialized manifest metadata field."""

    if isinstance(raw_value, dict):
        return raw_value
    if raw_value is None or pd.isna(raw_value):
        return {}
    text = str(raw_value).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise KaggleSplitError(f"Could not parse Kaggle manifest metadata JSON: {error}") from error
    if not isinstance(payload, dict):
        raise KaggleSplitError("Kaggle manifest metadata must be a JSON object.")
    return payload


def enrich_kaggle_manifest(manifest_frame: pd.DataFrame) -> pd.DataFrame:
    """Expand the manifest with provenance columns needed for safe splitting."""

    enriched = manifest_frame.copy().reset_index(drop=True)
    meta_payloads = enriched["meta"].apply(_parse_meta_payload)
    enriched["subset_source"] = meta_payloads.apply(lambda payload: payload.get("subset"))
    enriched["original_class_name"] = meta_payloads.apply(lambda payload: payload.get("original_class_name"))
    enriched["organization"] = meta_payloads.apply(lambda payload: payload.get("organization"))
    enriched["row_id"] = enriched.index.map(lambda value: f"row::{value}")
    return enriched


def _is_augmented_subset(subset_name: Any) -> bool:
    """Return whether a subset name suggests augmented imagery."""

    if subset_name is None or pd.isna(subset_name):
        return False
    normalized = str(subset_name).strip().lower()
    return "augment" in normalized


def _validate_split_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    """Validate the requested split fractions."""

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise KaggleSplitError(
            f"Split fractions must sum to 1.0, got train={train_fraction}, val={val_fraction}, test={test_fraction}"
        )
    for name, value in (("train", train_fraction), ("val", val_fraction), ("test", test_fraction)):
        if value <= 0 or value >= 1:
            raise KaggleSplitError(f"{name} fraction must be between 0 and 1, got {value}")


def _resolve_stratify_series(frame: pd.DataFrame) -> pd.Series:
    """Choose the safest available label column for stratified splitting."""

    if frame["label_name"].notna().all():
        return frame["label_name"].astype(str)
    if frame["label"].notna().all():
        return frame["label"].astype(str)
    missing_examples = frame.loc[frame["label_name"].isna() & frame["label"].isna(), "image"].head(5).tolist()
    raise KaggleSplitError(
        "Kaggle split generation needs either `label_name` or `label` populated for every base row. "
        f"Examples missing labels: {missing_examples}"
    )


def assign_group_ids(manifest_frame: pd.DataFrame) -> pd.DataFrame:
    """Assign leakage-aware grouping identifiers to manifest rows."""

    enriched = manifest_frame.copy()
    subject_series = enriched["subject_id"].fillna("").astype(str).str.strip()
    has_subject = subject_series != ""

    enriched["group_id"] = enriched["row_id"]
    enriched["group_kind"] = "row"
    enriched.loc[has_subject, "group_id"] = subject_series.loc[has_subject].map(lambda value: f"subject::{value}")
    enriched.loc[has_subject, "group_kind"] = "subject"
    enriched["stratify_label"] = _resolve_stratify_series(enriched)

    grouped = enriched.groupby("group_id")["stratify_label"].nunique()
    inconsistent_groups = grouped[grouped > 1].index.tolist()
    if inconsistent_groups:
        raise KaggleSplitError(
            "Some Kaggle groups have multiple labels. "
            f"Examples: {inconsistent_groups[:10]}"
        )
    return enriched


def _safe_train_test_split(
    frame: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    warnings: list[str],
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a stratified split when possible, otherwise fall back with a warning."""

    stratify_series = frame["stratify_label"]
    can_stratify = not frame.empty and stratify_series.value_counts().min() >= 2

    try:
        return train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_series if can_stratify else None,
        )
    except ValueError as error:
        if can_stratify:
            warnings.append(
                f"Falling back to unstratified splitting for {split_name} because stratification failed: {error}"
            )
            return train_test_split(
                frame,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
        raise KaggleSplitError(f"Could not split Kaggle groups for {split_name}: {error}") from error


def build_group_assignments(
    grouped_frame: pd.DataFrame,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_state: int,
    warnings: list[str],
) -> pd.DataFrame:
    """Assign train/val/test labels to grouped Kaggle rows."""

    _validate_split_fractions(train_fraction, val_fraction, test_fraction)

    if len(grouped_frame) < 3:
        raise KaggleSplitError("Need at least 3 Kaggle groups to create train/val/test splits.")

    train_groups, temp_groups = _safe_train_test_split(
        grouped_frame,
        test_size=(val_fraction + test_fraction),
        random_state=random_state,
        warnings=warnings,
        split_name="train-vs-temp",
    )

    temp_test_fraction = test_fraction / (val_fraction + test_fraction)
    val_groups, test_groups = _safe_train_test_split(
        temp_groups,
        test_size=temp_test_fraction,
        random_state=random_state,
        warnings=warnings,
        split_name="val-vs-test",
    )

    train_groups = train_groups.assign(split="train", assignment_reason="base_split")
    val_groups = val_groups.assign(split="val", assignment_reason="base_split")
    test_groups = test_groups.assign(split="test", assignment_reason="base_split")

    assignments = pd.concat([train_groups, val_groups, test_groups], ignore_index=True)
    return assignments.sort_values("group_id").reset_index(drop=True)


def apply_group_splits(manifest_frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Join group split assignments back onto the row-level Kaggle manifest."""

    merged = manifest_frame.merge(
        assignments[["group_id", "split", "assignment_reason"]],
        on="group_id",
        how="left",
    )
    if merged["split"].isna().any():
        missing_examples = merged.loc[merged["split"].isna(), "image"].head(5).tolist()
        raise KaggleSplitError(f"Some Kaggle manifest rows were not assigned a split. Examples: {missing_examples}")
    return merged


def save_kaggle_split_outputs(merged_frame: pd.DataFrame, output_root: Path, warnings: list[str]) -> KaggleSplitResult:
    """Save split assignments and split-specific manifest files."""

    ensure_directory(output_root)
    assignments_path = output_root / "kaggle_alz_split_assignments.csv"
    train_path = output_root / "kaggle_alz_train_manifest.csv"
    val_path = output_root / "kaggle_alz_val_manifest.csv"
    test_path = output_root / "kaggle_alz_test_manifest.csv"
    summary_path = output_root / "kaggle_alz_split_summary.json"

    assignment_columns = [
        "image",
        "label",
        "label_name",
        "subject_id",
        "dataset_type",
        "subset_source",
        "group_id",
        "group_kind",
        "split",
        "assignment_reason",
    ]
    merged_frame[assignment_columns].to_csv(assignments_path, index=False)
    merged_frame.loc[merged_frame["split"] == "train"].to_csv(train_path, index=False)
    merged_frame.loc[merged_frame["split"] == "val"].to_csv(val_path, index=False)
    merged_frame.loc[merged_frame["split"] == "test"].to_csv(test_path, index=False)

    label_key = "label_name" if merged_frame["label_name"].notna().all() else "label"
    summary_payload = {
        "train_rows": int((merged_frame["split"] == "train").sum()),
        "val_rows": int((merged_frame["split"] == "val").sum()),
        "test_rows": int((merged_frame["split"] == "test").sum()),
        "train_groups": int(merged_frame.loc[merged_frame["split"] == "train", "group_id"].nunique()),
        "val_groups": int(merged_frame.loc[merged_frame["split"] == "val", "group_id"].nunique()),
        "test_groups": int(merged_frame.loc[merged_frame["split"] == "test", "group_id"].nunique()),
        "dataset_type": str(merged_frame["dataset_type"].mode().iloc[0]),
        "split_strategy": {
            "augmented_rows_forced_to_train": bool(
                merged_frame["assignment_reason"].eq("augmented_train_only").any()
            ),
            "grouping_priority": ["subject_id", "row_id"],
            "stratify_key": label_key,
        },
        "label_distribution_by_split": {
            split_name: {str(label): int(count) for label, count in subset[label_key].value_counts().to_dict().items()}
            for split_name, subset in merged_frame.groupby("split")
        },
        "subset_distribution_by_split": {
            split_name: {
                str(label): int(count)
                for label, count in subset["subset_source"].fillna("unspecified").value_counts().to_dict().items()
            }
            for split_name, subset in merged_frame.groupby("split")
        },
        "warnings": warnings,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return KaggleSplitResult(
        split_assignments_path=assignments_path,
        train_manifest_path=train_path,
        val_manifest_path=val_path,
        test_manifest_path=test_path,
        summary_path=summary_path,
        train_rows=summary_payload["train_rows"],
        val_rows=summary_payload["val_rows"],
        test_rows=summary_payload["test_rows"],
        warnings=warnings,
    )


def build_kaggle_splits(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> KaggleSplitResult:
    """Create leakage-aware Kaggle train/val/test splits from the normalized manifest."""

    resolved_settings = settings or get_app_settings()
    output_root = resolved_settings.data_root / "interim"
    resolved_manifest_path = manifest_path or (output_root / "kaggle_alz_manifest.csv")

    if not resolved_manifest_path.exists():
        build_kaggle_manifest(settings=resolved_settings, output_format="csv")

    manifest_frame = load_kaggle_manifest(resolved_manifest_path)
    manifest_frame = assign_group_ids(enrich_kaggle_manifest(manifest_frame))
    warnings: list[str] = []

    augmented_mask = manifest_frame["subset_source"].apply(_is_augmented_subset)
    augmented_rows = manifest_frame.loc[augmented_mask].copy()
    base_rows = manifest_frame.loc[~augmented_mask].copy()

    if not augmented_rows.empty and not base_rows.empty:
        warnings.append(
            "Rows from augmented Kaggle subsets were forced into the training split to reduce leakage risk."
        )
    elif not augmented_rows.empty and base_rows.empty:
        warnings.append(
            "Only augmented Kaggle rows were detected, so all rows were split together. Leakage guarantees are weaker."
        )
        base_rows = augmented_rows
        augmented_rows = augmented_rows.iloc[0:0].copy()

    group_columns = ["group_id", "group_kind", "stratify_label"]
    grouped_base = base_rows[group_columns].drop_duplicates().reset_index(drop=True)
    base_assignments = build_group_assignments(
        grouped_base,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_state=random_state,
        warnings=warnings,
    )

    if not augmented_rows.empty:
        augmented_assignments = (
            augmented_rows[group_columns]
            .drop_duplicates()
            .assign(split="train", assignment_reason="augmented_train_only")
            .reset_index(drop=True)
        )
        assignments = pd.concat([base_assignments, augmented_assignments], ignore_index=True)
    else:
        assignments = base_assignments

    merged_frame = apply_group_splits(manifest_frame, assignments)
    return save_kaggle_split_outputs(merged_frame, output_root, warnings)
