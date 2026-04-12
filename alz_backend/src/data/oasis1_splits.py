"""Subject-safe split generation for the normalized OASIS-1 manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory


class OASIS1SplitError(ValueError):
    """Raised when the OASIS-1 manifest cannot be split safely."""


@dataclass(slots=True)
class OASIS1SplitResult:
    """Artifacts produced by the OASIS-1 split builder."""

    split_assignments_path: Path
    train_manifest_path: Path
    val_manifest_path: Path
    test_manifest_path: Path
    summary_path: Path
    train_rows: int
    val_rows: int
    test_rows: int


def load_oasis1_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load a previously built OASIS-1 manifest."""

    if not manifest_path.exists():
        raise OASIS1SplitError(f"OASIS-1 manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    required_columns = {"image", "label", "label_name", "subject_id", "dataset", "meta"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise OASIS1SplitError(f"OASIS-1 manifest is missing required columns: {sorted(missing)}")
    return frame


def build_subject_table(manifest_frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the manifest to one row per subject for leakage-safe splitting."""

    if manifest_frame.empty:
        raise OASIS1SplitError("OASIS-1 manifest is empty.")

    label_counts = manifest_frame.groupby("subject_id")["label"].nunique()
    inconsistent_subjects = label_counts[label_counts > 1].index.tolist()
    if inconsistent_subjects:
        raise OASIS1SplitError(
            "Some subjects have multiple labels in the manifest. "
            f"Examples: {inconsistent_subjects[:10]}"
        )

    subject_frame = (
        manifest_frame.groupby("subject_id", as_index=False)
        .agg(label=("label", "first"), label_name=("label_name", "first"))
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    return subject_frame


def _validate_split_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    """Validate the requested split fractions."""

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise OASIS1SplitError(
            f"Split fractions must sum to 1.0, got train={train_fraction}, val={val_fraction}, test={test_fraction}"
        )
    for name, value in (
        ("train", train_fraction),
        ("val", val_fraction),
        ("test", test_fraction),
    ):
        if value <= 0 or value >= 1:
            raise OASIS1SplitError(f"{name} fraction must be between 0 and 1, got {value}")


def _stratified_split(
    subject_frame: pd.DataFrame,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    """Assign train/val/test labels to subject rows with stratification."""

    _validate_split_fractions(train_fraction, val_fraction, test_fraction)

    if subject_frame["label"].value_counts().min() < 2:
        raise OASIS1SplitError("Each class needs at least 2 subjects for stratified splitting.")

    train_subjects, temp_subjects = train_test_split(
        subject_frame,
        test_size=(val_fraction + test_fraction),
        random_state=random_state,
        stratify=subject_frame["label"],
    )

    temp_test_fraction = test_fraction / (val_fraction + test_fraction)
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=temp_test_fraction,
        random_state=random_state,
        stratify=temp_subjects["label"],
    )

    train_subjects = train_subjects.assign(split="train")
    val_subjects = val_subjects.assign(split="val")
    test_subjects = test_subjects.assign(split="test")

    assignments = pd.concat([train_subjects, val_subjects, test_subjects], ignore_index=True)
    return assignments.sort_values("subject_id").reset_index(drop=True)


def apply_subject_splits(manifest_frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Join subject split assignments back onto the session-level manifest."""

    merged = manifest_frame.merge(assignments[["subject_id", "split"]], on="subject_id", how="left")
    if merged["split"].isna().any():
        missing_subjects = merged.loc[merged["split"].isna(), "subject_id"].unique().tolist()
        raise OASIS1SplitError(f"Some manifest rows were not assigned a split. Examples: {missing_subjects[:10]}")
    return merged


def save_split_outputs(
    merged_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    output_root: Path,
) -> OASIS1SplitResult:
    """Save split assignments and split-specific manifest files."""

    ensure_directory(output_root)
    assignments_path = output_root / "oasis1_subject_splits.csv"
    train_path = output_root / "oasis1_train_manifest.csv"
    val_path = output_root / "oasis1_val_manifest.csv"
    test_path = output_root / "oasis1_test_manifest.csv"
    summary_path = output_root / "oasis1_split_summary.json"

    assignments.to_csv(assignments_path, index=False)
    merged_frame.loc[merged_frame["split"] == "train"].to_csv(train_path, index=False)
    merged_frame.loc[merged_frame["split"] == "val"].to_csv(val_path, index=False)
    merged_frame.loc[merged_frame["split"] == "test"].to_csv(test_path, index=False)

    split_summary = {
        "train_rows": int((merged_frame["split"] == "train").sum()),
        "val_rows": int((merged_frame["split"] == "val").sum()),
        "test_rows": int((merged_frame["split"] == "test").sum()),
        "train_subjects": int((assignments["split"] == "train").sum()),
        "val_subjects": int((assignments["split"] == "val").sum()),
        "test_subjects": int((assignments["split"] == "test").sum()),
        "label_distribution_by_split": {
            split_name: {
                str(label): int(count)
                for label, count in subset["label"].value_counts().sort_index().to_dict().items()
            }
            for split_name, subset in assignments.groupby("split")
        },
    }
    summary_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    return OASIS1SplitResult(
        split_assignments_path=assignments_path,
        train_manifest_path=train_path,
        val_manifest_path=val_path,
        test_manifest_path=test_path,
        summary_path=summary_path,
        train_rows=split_summary["train_rows"],
        val_rows=split_summary["val_rows"],
        test_rows=split_summary["test_rows"],
    )


def build_oasis1_splits(
    settings: AppSettings | None = None,
    *,
    manifest_path: Path | None = None,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> OASIS1SplitResult:
    """Create subject-safe OASIS-1 train/val/test splits from the normalized manifest."""

    resolved_settings = settings or get_app_settings()
    resolved_manifest_path = manifest_path or (resolved_settings.data_root / "interim" / "oasis1_manifest.csv")
    output_root = resolved_settings.data_root / "interim"

    manifest_frame = load_oasis1_manifest(resolved_manifest_path)
    subject_frame = build_subject_table(manifest_frame)
    assignments = _stratified_split(
        subject_frame,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    merged_frame = apply_subject_splits(manifest_frame, assignments)
    return save_split_outputs(merged_frame, assignments, output_root)
