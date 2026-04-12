"""Reproducible MONAI-compatible OASIS dataset and dataloader builders."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.runtime import AppSettings, get_app_settings
from src.transforms.oasis_transforms import (
    OASISTransformConfig,
    build_oasis_infer_transforms,
    build_oasis_train_transforms,
    build_oasis_val_transforms,
    load_oasis_transform_config,
)
from src.utils.io_utils import ensure_directory
from src.utils.monai_utils import load_monai_data_symbols, load_torch_symbols

from .base_dataset import build_monai_dataset, canonicalize_optional_string, parse_manifest_meta
from .oasis_dataset import OASIS_DEFAULT_DATASET_TYPE, load_oasis_manifest

_load_monai_data_symbols = load_monai_data_symbols
_load_torch_symbols = load_torch_symbols
_SESSION_VISIT_PATTERN = re.compile(r"_MR(\d+)\b", re.IGNORECASE)


class OASISLoaderError(ValueError):
    """Raised when OASIS dataset loading or split generation cannot proceed safely."""


@dataclass(slots=True, frozen=True)
class OASISLoaderConfig:
    """Configuration for reproducible OASIS MONAI dataset and dataloader construction."""

    settings: AppSettings | None = None
    manifest_path: Path | None = None
    reports_root: Path | None = None
    seed: int = 42
    split_seed: int | None = None
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 1
    num_workers: int = 0
    cache_rate: float = 0.0
    weighted_sampling: bool = False
    weighted_sampling_replacement: bool = True
    transform_config: OASISTransformConfig = field(default_factory=load_oasis_transform_config)


@dataclass(slots=True)
class OASISSplitArtifacts:
    """Saved split reports and in-memory split frames for one reproducible run."""

    report_root: Path
    split_assignments_path: Path
    longitudinal_index_path: Path
    longitudinal_subject_summary_path: Path
    train_manifest_path: Path
    val_manifest_path: Path
    test_manifest_path: Path
    summary_path: Path
    assignments: pd.DataFrame
    longitudinal_frame: pd.DataFrame
    longitudinal_subject_summary: pd.DataFrame
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame


@dataclass(slots=True)
class OASISDatasetBundle:
    """MONAI datasets and supporting split artifacts for OASIS."""

    train_dataset: object
    val_dataset: object
    test_dataset: object
    split_artifacts: OASISSplitArtifacts
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    train_class_weights: dict[int, float]


@dataclass(slots=True)
class OASISDataloaderBundle:
    """MONAI dataloaders and the datasets they were built from."""

    train_loader: object
    val_loader: object
    test_loader: object
    dataset_bundle: OASISDatasetBundle
    train_sampler: object | None = None


def _validate_split_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    """Validate that split fractions form a proper partition."""

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise OASISLoaderError(
            f"Split fractions must sum to 1.0, got train={train_fraction}, val={val_fraction}, test={test_fraction}"
        )
    for name, value in (("train", train_fraction), ("val", val_fraction), ("test", test_fraction)):
        if value <= 0 or value >= 1:
            raise OASISLoaderError(f"{name} fraction must be between 0 and 1, got {value}")


def _build_subject_table(manifest_frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the manifest to one row per subject for leakage-safe splitting."""

    if manifest_frame.empty:
        raise OASISLoaderError("OASIS manifest is empty.")
    subject_series = manifest_frame["subject_id"].fillna("").astype(str).str.strip()
    if (subject_series == "").any():
        raise OASISLoaderError("OASIS manifest is missing subject_id values required for subject-safe splitting.")
    if manifest_frame["label"].isna().any():
        raise OASISLoaderError("OASIS manifest is missing numeric labels required for reproducible splitting.")

    label_counts = manifest_frame.groupby("subject_id")["label"].nunique()
    inconsistent_subjects = label_counts[label_counts > 1].index.tolist()
    if inconsistent_subjects:
        raise OASISLoaderError(
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


def _assign_subject_splits(
    subject_frame: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    """Create deterministic train/val/test subject assignments with stratification."""

    _validate_split_fractions(train_fraction, val_fraction, test_fraction)
    class_counts = subject_frame["label"].value_counts()
    if class_counts.min() < 2:
        raise OASISLoaderError("Each class needs at least 2 subjects for stratified OASIS splitting.")

    try:
        train_subjects, temp_subjects = train_test_split(
            subject_frame,
            test_size=(val_fraction + test_fraction),
            random_state=seed,
            stratify=subject_frame["label"],
        )
        temp_test_fraction = test_fraction / (val_fraction + test_fraction)
        val_subjects, test_subjects = train_test_split(
            temp_subjects,
            test_size=temp_test_fraction,
            random_state=seed,
            stratify=temp_subjects["label"],
        )
    except ValueError as error:
        raise OASISLoaderError(
            "Could not create a safe stratified OASIS split. "
            "This usually means there are too few subjects in one class for the requested fractions."
        ) from error

    assignments = pd.concat(
        [
            train_subjects.assign(split="train"),
            val_subjects.assign(split="val"),
            test_subjects.assign(split="test"),
        ],
        ignore_index=True,
    )
    return assignments.sort_values("subject_id").reset_index(drop=True)


def _apply_assignments(manifest_frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Join subject split assignments onto the row-level manifest."""

    merged = manifest_frame.merge(assignments[["subject_id", "split"]], on="subject_id", how="left")
    if merged["split"].isna().any():
        missing_examples = merged.loc[merged["split"].isna(), "subject_id"].unique().tolist()
        raise OASISLoaderError(f"Some manifest rows were not assigned a split. Examples: {missing_examples[:10]}")
    return merged


def _extract_session_id(row: pd.Series) -> tuple[str, str]:
    """Resolve a session identifier for one manifest row."""

    direct_session_id = canonicalize_optional_string(row.get("session_id"))
    if direct_session_id:
        return direct_session_id, "session_id_column"

    meta_payload = parse_manifest_meta(row.get("meta"))
    meta_session_id = canonicalize_optional_string(meta_payload.get("session_id"))
    if meta_session_id:
        return meta_session_id, "meta.session_id"

    image_value = canonicalize_optional_string(row.get("image"))
    if image_value:
        return Path(image_value).stem, "image_stem_fallback"

    raise OASISLoaderError("Could not resolve a session_id for one OASIS manifest row.")


def _extract_session_visit_number(session_id: str) -> int | None:
    """Infer an MR visit number from an OASIS-like session identifier."""

    match = _SESSION_VISIT_PATTERN.search(session_id)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_scan_timestamp(raw_value: Any) -> str | None:
    """Normalize a timestamp-like value to string when possible."""

    normalized = canonicalize_optional_string(raw_value)
    if normalized is None:
        return None
    return normalized


def _build_longitudinal_frame(manifest_frame: pd.DataFrame) -> pd.DataFrame:
    """Augment a split-ready manifest with visit-order metadata for future longitudinal use."""

    enriched = manifest_frame.copy().reset_index(drop=True)
    session_pairs = enriched.apply(_extract_session_id, axis=1, result_type="expand")
    enriched["session_id"] = session_pairs[0].astype(str)
    enriched["session_id_source"] = session_pairs[1].astype(str)
    if enriched["session_id"].duplicated().any():
        duplicate_examples = (
            enriched.loc[enriched["session_id"].duplicated(keep=False), "session_id"].drop_duplicates().tolist()
        )
        raise OASISLoaderError(
            "OASIS longitudinal indexing requires unique session_id values. "
            f"Examples: {duplicate_examples[:10]}"
        )

    enriched["scan_timestamp"] = enriched["scan_timestamp"].apply(_normalize_scan_timestamp)
    enriched["scan_timestamp_parsed"] = pd.to_datetime(enriched["scan_timestamp"], errors="coerce")
    enriched["session_visit_number"] = enriched["session_id"].apply(_extract_session_visit_number)
    enriched["visit_order_source"] = "stable_row_order"
    enriched.loc[enriched["session_visit_number"].notna(), "visit_order_source"] = "session_id_pattern"
    enriched.loc[enriched["scan_timestamp_parsed"].notna(), "visit_order_source"] = "scan_timestamp"

    sort_frame = enriched.assign(
        _timestamp_missing=enriched["scan_timestamp_parsed"].isna().astype(int),
        _timestamp_value=enriched["scan_timestamp_parsed"].fillna(pd.Timestamp.max),
        _visit_missing=enriched["session_visit_number"].isna().astype(int),
        _visit_value=enriched["session_visit_number"].fillna(10**9).astype(int),
        _stable_index=enriched.index,
    )
    sort_frame = sort_frame.sort_values(
        [
            "subject_id",
            "_timestamp_missing",
            "_timestamp_value",
            "_visit_missing",
            "_visit_value",
            "session_id",
            "_stable_index",
        ]
    ).reset_index(drop=True)
    sort_frame["visit_order"] = sort_frame.groupby("subject_id").cumcount() + 1
    subject_session_counts = sort_frame.groupby("subject_id")["session_id"].size().to_dict()
    sort_frame["subject_session_count"] = sort_frame["subject_id"].map(subject_session_counts).astype(int)
    sort_frame["is_longitudinal_subject"] = sort_frame["subject_session_count"] > 1
    sort_frame["longitudinal_group_id"] = sort_frame["subject_id"].map(lambda value: f"subject::{value}")
    return sort_frame.drop(
        columns=["_timestamp_missing", "_timestamp_value", "_visit_missing", "_visit_value", "_stable_index"]
    )


def _build_longitudinal_subject_summary(longitudinal_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-subject longitudinal coverage for reporting and future reuse."""

    rows: list[dict[str, Any]] = []
    for subject_id, subject_frame in longitudinal_frame.groupby("subject_id", sort=True):
        ordered_frame = subject_frame.sort_values("visit_order").reset_index(drop=True)
        ordered_timestamps = [value for value in ordered_frame["scan_timestamp"].tolist() if value]
        rows.append(
            {
                "subject_id": subject_id,
                "split": ordered_frame["split"].iloc[0],
                "label": int(float(ordered_frame["label"].iloc[0])),
                "label_name": str(ordered_frame["label_name"].iloc[0]),
                "subject_session_count": int(len(ordered_frame)),
                "is_longitudinal_subject": bool(len(ordered_frame) > 1),
                "first_session_id": str(ordered_frame["session_id"].iloc[0]),
                "last_session_id": str(ordered_frame["session_id"].iloc[-1]),
                "first_scan_timestamp": ordered_timestamps[0] if ordered_timestamps else None,
                "last_scan_timestamp": ordered_timestamps[-1] if ordered_timestamps else None,
                "visit_order_source": str(ordered_frame["visit_order_source"].mode().iloc[0]),
                "session_ids_json": json.dumps(ordered_frame["session_id"].tolist()),
                "visit_orders_json": json.dumps([int(value) for value in ordered_frame["visit_order"].tolist()]),
            }
        )
    return pd.DataFrame(rows)


def _report_folder_name(cfg: OASISLoaderConfig) -> str:
    """Build a stable report folder name for one split configuration."""

    resolved_split_seed = cfg.seed if cfg.split_seed is None else cfg.split_seed
    return (
        f"oasis_loaders_seed{cfg.seed}"
        f"_split{resolved_split_seed}"
        f"_train{int(round(cfg.train_fraction * 100))}"
        f"_val{int(round(cfg.val_fraction * 100))}"
        f"_test{int(round(cfg.test_fraction * 100))}"
    )


def _resolve_reports_root(cfg: OASISLoaderConfig, settings: AppSettings) -> Path:
    """Resolve the output report folder for split artifacts."""

    base_root = cfg.reports_root or (settings.outputs_root / "reports")
    return ensure_directory(Path(base_root) / _report_folder_name(cfg))


def _save_split_reports(
    merged_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    cfg: OASISLoaderConfig,
    settings: AppSettings,
) -> OASISSplitArtifacts:
    """Save split manifests and summary reports under outputs/reports."""

    report_root = _resolve_reports_root(cfg, settings)
    assignments_path = report_root / "oasis_split_assignments.csv"
    longitudinal_index_path = report_root / "oasis_longitudinal_index.csv"
    longitudinal_subject_summary_path = report_root / "oasis_longitudinal_subject_summary.csv"
    train_manifest_path = report_root / "oasis_train_manifest.csv"
    val_manifest_path = report_root / "oasis_val_manifest.csv"
    test_manifest_path = report_root / "oasis_test_manifest.csv"
    summary_path = report_root / "oasis_split_summary.json"

    longitudinal_frame = _build_longitudinal_frame(merged_frame)
    longitudinal_subject_summary = _build_longitudinal_subject_summary(longitudinal_frame)
    train_frame = longitudinal_frame.loc[longitudinal_frame["split"] == "train"].copy().reset_index(drop=True)
    val_frame = longitudinal_frame.loc[longitudinal_frame["split"] == "val"].copy().reset_index(drop=True)
    test_frame = longitudinal_frame.loc[longitudinal_frame["split"] == "test"].copy().reset_index(drop=True)

    assignments.to_csv(assignments_path, index=False)
    longitudinal_frame.to_csv(longitudinal_index_path, index=False)
    longitudinal_subject_summary.to_csv(longitudinal_subject_summary_path, index=False)
    train_frame.to_csv(train_manifest_path, index=False)
    val_frame.to_csv(val_manifest_path, index=False)
    test_frame.to_csv(test_manifest_path, index=False)

    subject_sets = {
        "train": set(train_frame["subject_id"].dropna().tolist()),
        "val": set(val_frame["subject_id"].dropna().tolist()),
        "test": set(test_frame["subject_id"].dropna().tolist()),
    }
    summary_payload = {
        "seed": cfg.seed,
        "split_seed": cfg.seed if cfg.split_seed is None else cfg.split_seed,
        "fractions": {
            "train": cfg.train_fraction,
            "val": cfg.val_fraction,
            "test": cfg.test_fraction,
        },
        "weighted_sampling": cfg.weighted_sampling,
        "dataset": "oasis1",
        "dataset_type": OASIS_DEFAULT_DATASET_TYPE,
        "subject_counts": {split_name: len(subject_ids) for split_name, subject_ids in subject_sets.items()},
        "row_counts": {
            "train": int(len(train_frame)),
            "val": int(len(val_frame)),
            "test": int(len(test_frame)),
        },
        "label_distribution_by_split": {
            split_name: {
                str(label): int(count)
                for label, count in frame["label"].value_counts().sort_index().to_dict().items()
            }
            for split_name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame))
        },
        "subject_overlap": {
            "train_val": sorted(subject_sets["train"].intersection(subject_sets["val"])),
            "train_test": sorted(subject_sets["train"].intersection(subject_sets["test"])),
            "val_test": sorted(subject_sets["val"].intersection(subject_sets["test"])),
        },
        "longitudinal": {
            "subjects_with_multiple_sessions": int(longitudinal_subject_summary["is_longitudinal_subject"].sum()),
            "subjects_with_single_session": int((~longitudinal_subject_summary["is_longitudinal_subject"]).sum()),
            "max_sessions_per_subject": int(longitudinal_subject_summary["subject_session_count"].max()),
            "split_subjects_with_multiple_sessions": {
                split_name: int(group["is_longitudinal_subject"].sum())
                for split_name, group in longitudinal_subject_summary.groupby("split")
            },
            "session_id_sources": {
                str(source): int(count)
                for source, count in longitudinal_frame["session_id_source"].value_counts().to_dict().items()
            },
            "visit_order_sources": {
                str(source): int(count)
                for source, count in longitudinal_frame["visit_order_source"].value_counts().to_dict().items()
            },
        },
        "config": {
            **{key: value for key, value in asdict(cfg).items() if key != "settings"},
            "manifest_path": str(cfg.manifest_path) if cfg.manifest_path is not None else None,
            "reports_root": str(report_root),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return OASISSplitArtifacts(
        report_root=report_root,
        split_assignments_path=assignments_path,
        longitudinal_index_path=longitudinal_index_path,
        longitudinal_subject_summary_path=longitudinal_subject_summary_path,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        test_manifest_path=test_manifest_path,
        summary_path=summary_path,
        assignments=assignments,
        longitudinal_frame=longitudinal_frame,
        longitudinal_subject_summary=longitudinal_subject_summary,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
    )


def _records_from_split_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert one split frame into MONAI-friendly supervised records."""

    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        image_path = Path(row.image)
        if not image_path.exists():
            raise FileNotFoundError(f"OASIS split references a missing image path: {image_path}")
        if pd.isna(row.label):
            raise OASISLoaderError(f"OASIS split record is missing a numeric label: {image_path}")
        subject_id = str(row.subject_id).strip() if not pd.isna(row.subject_id) else ""
        if not subject_id:
            raise OASISLoaderError(f"OASIS split record is missing a subject_id: {image_path}")
        records.append(
            {
                "image": str(image_path),
                "image_path": str(image_path),
                "label": int(float(row.label)),
                "subject_id": subject_id,
                "session_id": str(getattr(row, "session_id", "")),
                "visit_order": int(getattr(row, "visit_order", 1)),
                "subject_session_count": int(getattr(row, "subject_session_count", 1)),
                "is_longitudinal_subject": bool(getattr(row, "is_longitudinal_subject", False)),
                "longitudinal_group_id": str(getattr(row, "longitudinal_group_id", f"subject::{subject_id}")),
                "scan_timestamp": "" if pd.isna(getattr(row, "scan_timestamp", None)) else str(row.scan_timestamp),
            }
        )
    return records


def _compute_class_weights(records: list[dict[str, Any]]) -> dict[int, float]:
    """Compute inverse-frequency weights for the training split."""

    labels = [int(record["label"]) for record in records]
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    return {int(label): float(1.0 / count) for label, count in counts.items()}


def _build_weighted_sampler(records: list[dict[str, Any]], *, seed: int, replacement: bool) -> object:
    """Create a seed-controlled weighted sampler for training rows."""

    torch = _load_torch_symbols()["torch"]
    class_weights = _compute_class_weights(records)
    sample_weights = [class_weights[int(record["label"])] for record in records]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=replacement,
        generator=generator,
    )


def _build_torch_generator(seed: int) -> object:
    """Build a torch generator for reproducible loader shuffling."""

    torch = _load_torch_symbols()["torch"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_oasis_datasets(cfg: OASISLoaderConfig) -> OASISDatasetBundle:
    """Build reproducible OASIS train/val/test MONAI datasets from the manifest."""

    settings = cfg.settings or get_app_settings()
    manifest_frame = load_oasis_manifest(
        settings,
        manifest_path=cfg.manifest_path,
    )
    subject_frame = _build_subject_table(manifest_frame)
    assignments = _assign_subject_splits(
        subject_frame,
        seed=cfg.seed if cfg.split_seed is None else cfg.split_seed,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
    )
    merged_frame = _apply_assignments(manifest_frame, assignments)
    split_artifacts = _save_split_reports(
        merged_frame,
        assignments,
        cfg=cfg,
        settings=settings,
    )

    train_records = _records_from_split_frame(split_artifacts.train_frame)
    val_records = _records_from_split_frame(split_artifacts.val_frame)
    test_records = _records_from_split_frame(split_artifacts.test_frame)
    train_class_weights = _compute_class_weights(train_records)

    train_dataset = build_monai_dataset(
        train_records,
        build_oasis_train_transforms(cfg.transform_config),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    val_dataset = build_monai_dataset(
        val_records,
        build_oasis_val_transforms(cfg.transform_config),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    test_dataset = build_monai_dataset(
        test_records,
        build_oasis_infer_transforms(cfg.transform_config),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )

    return OASISDatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        split_artifacts=split_artifacts,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        train_class_weights=train_class_weights,
    )


def build_oasis_dataloaders(cfg: OASISLoaderConfig) -> OASISDataloaderBundle:
    """Build reproducible OASIS train/val/test MONAI dataloaders."""

    dataset_bundle = build_oasis_datasets(cfg)
    data_loader_cls = _load_monai_data_symbols()["DataLoader"]

    train_sampler = None
    train_shuffle = True
    train_generator = _build_torch_generator(cfg.seed)
    if cfg.weighted_sampling:
        train_sampler = _build_weighted_sampler(
            dataset_bundle.train_records,
            seed=cfg.seed,
            replacement=cfg.weighted_sampling_replacement,
        )
        train_shuffle = False

    train_loader = data_loader_cls(
        dataset_bundle.train_dataset,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        generator=train_generator,
    )
    val_loader = data_loader_cls(
        dataset_bundle.val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = data_loader_cls(
        dataset_bundle.test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return OASISDataloaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_bundle=dataset_bundle,
        train_sampler=train_sampler,
    )
