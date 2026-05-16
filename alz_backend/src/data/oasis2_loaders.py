"""Reproducible MONAI-compatible OASIS-2 dataset and dataloader builders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.configs.runtime import AppSettings, get_app_settings
from src.transforms.oasis_transforms import (
    OASISTransformConfig,
    build_oasis_infer_transforms,
    build_oasis_train_transforms,
    build_oasis_val_transforms,
    load_oasis_transform_config,
)
from src.utils.monai_utils import load_monai_data_symbols, load_torch_symbols

from .base_dataset import build_monai_dataset, canonicalize_optional_string, parse_manifest_meta
from .oasis2_supervised import (
    OASIS2SupervisedSplitArtifacts,
    OASIS2SupervisedSplitConfig,
    build_oasis2_supervised_split_artifacts,
)

_load_monai_data_symbols = load_monai_data_symbols
_load_torch_symbols = load_torch_symbols

_MISSING_IMAGE_RESOLUTION_CACHE: dict[tuple[str, tuple[str, ...]], Path] = {}


def _safe_optional_text(value: Any) -> str:
    """Return one MONAI-collate-safe optional text field."""

    normalized = canonicalize_optional_string(value)
    return "" if normalized is None else normalized


@dataclass(slots=True, frozen=True)
class OASIS2LoaderConfig:
    """Configuration for reproducible OASIS-2 datasets and dataloaders."""

    settings: AppSettings | None = None
    manifest_path: Path | None = None
    split_plan_path: Path | None = None
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
class OASIS2DatasetBundle:
    """MONAI datasets and supporting split artifacts for OASIS-2."""

    train_dataset: object
    val_dataset: object
    test_dataset: object
    split_artifacts: OASIS2SupervisedSplitArtifacts
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    train_class_weights: dict[int, float]


@dataclass(slots=True)
class OASIS2DataloaderBundle:
    """MONAI dataloaders and datasets for OASIS-2."""

    train_loader: object
    val_loader: object
    test_loader: object
    dataset_bundle: OASIS2DatasetBundle
    train_sampler: object | None = None


def _oasis2_relative_suffix(image_path: Path) -> Path | None:
    parts = image_path.parts
    for index, part in enumerate(parts):
        if part.startswith("OAS2_RAW_PART"):
            return Path(*parts[index:])
    return None


def _candidate_oasis2_bundle_roots(settings: AppSettings) -> list[Path]:
    exports_root = settings.outputs_root / "exports"
    roots = [
        settings.collection_root,
        settings.project_root,
        exports_root / "oasis2_upload_bundle_ready",
        exports_root / "oasis2_upload_bundle",
    ]
    if exports_root.exists():
        roots.extend(path for path in exports_root.glob("oasis2_upload_bundle*") if path.is_dir())
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _resolve_oasis2_image_path(image_path: Path, *, search_roots: tuple[Path, ...]) -> Path:
    if image_path.exists():
        return image_path
    suffix = _oasis2_relative_suffix(image_path)
    if suffix is None:
        raise FileNotFoundError(f"OASIS-2 split references a missing image path: {image_path}")

    cache_key = (str(suffix), tuple(str(root) for root in search_roots))
    cached = _MISSING_IMAGE_RESOLUTION_CACHE.get(cache_key)
    if cached is not None and cached.exists():
        return cached

    for root in search_roots:
        candidate = root / suffix
        if candidate.exists():
            _MISSING_IMAGE_RESOLUTION_CACHE[cache_key] = candidate
            return candidate
    for root in search_roots:
        if not root.exists():
            continue
        matches = list(root.glob(f"**/{suffix.as_posix()}"))
        if matches:
            _MISSING_IMAGE_RESOLUTION_CACHE[cache_key] = matches[0]
            return matches[0]
    raise FileNotFoundError(
        f"OASIS-2 split references a missing image path: {image_path}. "
        f"Also checked bundle-relative suffix {suffix} under: {[str(root) for root in search_roots]}"
    )


def _records_from_split_frame(
    frame: pd.DataFrame,
    *,
    search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    """Convert one OASIS-2 split frame into MONAI-friendly supervised records."""

    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        image_path = _resolve_oasis2_image_path(Path(row.image), search_roots=search_roots)
        label_value = int(float(row.label))
        subject_id = canonicalize_optional_string(getattr(row, "subject_id", None))
        session_id = canonicalize_optional_string(getattr(row, "session_id", None))
        if subject_id is None or session_id is None:
            raise ValueError(f"OASIS-2 split record is missing subject or session identity: {image_path}")
        meta = parse_manifest_meta(getattr(row, "meta", None))
        meta.setdefault("split_group_key", canonicalize_optional_string(getattr(row, "split_group_key", None)) or subject_id)
        meta.setdefault("group_binary_label", int(float(getattr(row, "group_binary_label", label_value))))
        meta.setdefault("mixed_label_group", bool(getattr(row, "mixed_label_group", False)))
        meta.setdefault("subject_safe_bucket", int(float(getattr(row, "subject_safe_bucket", 0))))
        meta.setdefault("future_role_hint", _safe_optional_text(getattr(row, "future_role_hint", None)))
        visit_number_raw = getattr(row, "visit_number", None)
        visit_number = None if pd.isna(visit_number_raw) else int(float(visit_number_raw))
        if visit_number is not None:
            meta.setdefault("visit_number", visit_number)

        records.append(
            {
                "image": str(image_path),
                "image_path": str(image_path),
                "label": label_value,
                "label_name": _safe_optional_text(getattr(row, "label_name", None)),
                "subject_id": subject_id,
                "session_id": session_id,
                "scan_timestamp": _safe_optional_text(getattr(row, "scan_timestamp", None)),
                "dataset": _safe_optional_text(getattr(row, "dataset", None)),
                "dataset_type": _safe_optional_text(getattr(row, "dataset_type", None)),
                "visit_number": visit_number,
                "split_group_key": meta["split_group_key"],
                "group_binary_label": meta["group_binary_label"],
                "mixed_label_group": meta["mixed_label_group"],
                "meta": meta,
            }
        )
    return records


def _compute_class_weights(records: list[dict[str, Any]]) -> dict[int, float]:
    labels = [int(record["label"]) for record in records]
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    return {int(label): float(1.0 / count) for label, count in counts.items()}


def _build_weighted_sampler(records: list[dict[str, Any]], *, seed: int, replacement: bool) -> object:
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
    torch = _load_torch_symbols()["torch"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_oasis2_datasets(cfg: OASIS2LoaderConfig) -> OASIS2DatasetBundle:
    """Build reproducible OASIS-2 train/val/test MONAI datasets from supervised split manifests."""

    settings = cfg.settings or get_app_settings()
    split_artifacts = build_oasis2_supervised_split_artifacts(
        OASIS2SupervisedSplitConfig(
            settings=settings,
            manifest_path=cfg.manifest_path,
            split_plan_path=cfg.split_plan_path,
            reports_root=cfg.reports_root,
            seed=cfg.seed,
            split_seed=cfg.split_seed,
            train_fraction=cfg.train_fraction,
            val_fraction=cfg.val_fraction,
            test_fraction=cfg.test_fraction,
        )
    )

    search_roots = tuple(_candidate_oasis2_bundle_roots(settings))
    train_records = _records_from_split_frame(split_artifacts.train_frame, search_roots=search_roots)
    val_records = _records_from_split_frame(split_artifacts.val_frame, search_roots=search_roots)
    test_records = _records_from_split_frame(split_artifacts.test_frame, search_roots=search_roots)
    train_class_weights = _compute_class_weights(train_records)

    # Strip variable metadata for MONAI datasets — the training loop only uses
    # ``image`` and ``label``, and variable-length meta dicts (e.g. mmse present
    # in some sessions but not others) cause KeyError during batch collation
    # when batch_size > 1.  Full records are still kept in the bundle for
    # analysis and reporting.
    def _collation_safe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "image": r["image"],
                "image_path": r["image_path"],
                "label": r["label"],
                "label_name": r["label_name"],
                "subject_id": r["subject_id"],
                "session_id": r["session_id"],
                "split_group_key": r["split_group_key"],
                "meta": r["meta"],
            }
            for r in records
        ]

    train_dataset = build_monai_dataset(
        _collation_safe(train_records),
        build_oasis_train_transforms(cfg.transform_config),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    val_dataset = build_monai_dataset(
        _collation_safe(val_records),
        build_oasis_val_transforms(cfg.transform_config),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    test_dataset = build_monai_dataset(
        _collation_safe(test_records),
        build_oasis_infer_transforms(cfg.transform_config),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )

    return OASIS2DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        split_artifacts=split_artifacts,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        train_class_weights=train_class_weights,
    )


def build_oasis2_dataloaders(cfg: OASIS2LoaderConfig) -> OASIS2DataloaderBundle:
    """Build reproducible OASIS-2 train/val/test MONAI dataloaders."""

    dataset_bundle = build_oasis2_datasets(cfg)
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

    return OASIS2DataloaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_bundle=dataset_bundle,
        train_sampler=train_sampler,
    )
