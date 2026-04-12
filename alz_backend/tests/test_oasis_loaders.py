"""Tests for reproducible OASIS MONAI dataset and dataloader builders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.data.loaders import OASISLoaderConfig, OASISLoaderError, build_oasis_dataloaders, build_oasis_datasets


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for loader tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    data_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=project_root.parent,
        oasis_source_root=project_root.parent / "OASIS",
    )


def _write_manifest(
    settings: AppSettings,
    *,
    subjects_per_class: tuple[int, int] = (6, 6),
    sessions_per_subject: int = 2,
    reverse_timestamps_for_first_subject: bool = False,
) -> Path:
    """Create a synthetic OASIS manifest with real placeholder image files."""

    interim_root = settings.data_root / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)
    image_root = interim_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    subject_index = 0
    for label, subject_count in enumerate(subjects_per_class):
        for _ in range(subject_count):
            subject_id = f"OAS1_{subject_index:04d}"
            label_name = "nondemented" if label == 0 else "demented"
            for session_index in range(sessions_per_subject):
                image_path = image_root / f"{subject_id}_MR{session_index + 1}.hdr"
                image_path.write_text("hdr", encoding="utf-8")
                timestamp_value = f"200{session_index}-01-01"
                if reverse_timestamps_for_first_subject and subject_index == 0:
                    timestamp_value = f"200{sessions_per_subject - session_index - 1}-01-01"
                rows.append(
                    {
                        "image": str(image_path),
                        "label": label,
                        "label_name": label_name,
                        "subject_id": subject_id,
                        "scan_timestamp": timestamp_value,
                        "dataset": "oasis1",
                        "meta": json.dumps({"session_id": f"{subject_id}_MR{session_index + 1}"}),
                    }
                )
            subject_index += 1

    manifest_path = interim_root / "oasis1_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


class _FakeDataset:
    def __init__(self, records: list[dict[str, object]], transform: object, cache_rate: float, num_workers: int) -> None:
        self.records = records
        self.transform = transform
        self.cache_rate = cache_rate
        self.num_workers = num_workers


class _FakeDataLoader:
    def __init__(
        self,
        dataset: object,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        sampler: object | None = None,
        generator: object | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sampler = sampler
        self.generator = generator


def _patch_loader_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace MONAI-heavy dependencies with lightweight fakes."""

    monkeypatch.setattr(
        "src.data.loaders.build_monai_dataset",
        lambda records, transform, cache_rate=0.0, num_workers=0: _FakeDataset(
            records=records,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        ),
    )
    monkeypatch.setattr("src.data.loaders.build_oasis_train_transforms", lambda *_: "train_transform")
    monkeypatch.setattr("src.data.loaders.build_oasis_val_transforms", lambda *_: "val_transform")
    monkeypatch.setattr("src.data.loaders.build_oasis_infer_transforms", lambda *_: "infer_transform")
    monkeypatch.setattr("src.data.loaders._load_monai_data_symbols", lambda: {"DataLoader": _FakeDataLoader})


def test_build_oasis_datasets_is_reproducible_and_subject_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader builder should create deterministic, leakage-safe subject partitions."""

    settings = _build_settings(tmp_path)
    _write_manifest(settings)
    _patch_loader_dependencies(monkeypatch)

    cfg = OASISLoaderConfig(
        settings=settings,
        seed=17,
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
    )

    first_bundle = build_oasis_datasets(cfg)
    second_bundle = build_oasis_datasets(cfg)

    pd.testing.assert_frame_equal(
        first_bundle.split_artifacts.assignments.reset_index(drop=True),
        second_bundle.split_artifacts.assignments.reset_index(drop=True),
    )

    train_subjects = set(first_bundle.split_artifacts.train_frame["subject_id"])
    val_subjects = set(first_bundle.split_artifacts.val_frame["subject_id"])
    test_subjects = set(first_bundle.split_artifacts.test_frame["subject_id"])

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)
    assert first_bundle.split_artifacts.summary_path.exists()
    assert first_bundle.split_artifacts.longitudinal_index_path.exists()
    assert first_bundle.split_artifacts.longitudinal_subject_summary_path.exists()
    assert first_bundle.train_dataset.transform == "train_transform"
    assert first_bundle.val_dataset.transform == "val_transform"
    assert first_bundle.test_dataset.transform == "infer_transform"
    assert {"session_id", "visit_order", "subject_session_count", "is_longitudinal_subject"} <= set(
        first_bundle.train_records[0]
    )


def test_build_oasis_datasets_changes_assignments_when_seed_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the seed should change at least part of the subject assignment table."""

    settings = _build_settings(tmp_path)
    _write_manifest(settings, subjects_per_class=(8, 8))
    _patch_loader_dependencies(monkeypatch)

    first_bundle = build_oasis_datasets(
        OASISLoaderConfig(settings=settings, seed=11, train_fraction=0.5, val_fraction=0.25, test_fraction=0.25)
    )
    second_bundle = build_oasis_datasets(
        OASISLoaderConfig(settings=settings, seed=23, train_fraction=0.5, val_fraction=0.25, test_fraction=0.25)
    )

    merged = first_bundle.split_artifacts.assignments.merge(
        second_bundle.split_artifacts.assignments,
        on="subject_id",
        suffixes=("_first", "_second"),
    )
    assert (merged["split_first"] != merged["split_second"]).any()


def test_build_oasis_datasets_changes_assignments_when_split_seed_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing split_seed should change subject partitions without changing training seed."""

    settings = _build_settings(tmp_path)
    _write_manifest(settings, subjects_per_class=(8, 8))
    _patch_loader_dependencies(monkeypatch)

    first_bundle = build_oasis_datasets(
        OASISLoaderConfig(
            settings=settings,
            seed=11,
            split_seed=101,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
        )
    )
    second_bundle = build_oasis_datasets(
        OASISLoaderConfig(
            settings=settings,
            seed=11,
            split_seed=202,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
        )
    )

    merged = first_bundle.split_artifacts.assignments.merge(
        second_bundle.split_artifacts.assignments,
        on="subject_id",
        suffixes=("_first", "_second"),
    )
    assert (merged["split_first"] != merged["split_second"]).any()
    first_summary = json.loads(first_bundle.split_artifacts.summary_path.read_text(encoding="utf-8"))
    second_summary = json.loads(second_bundle.split_artifacts.summary_path.read_text(encoding="utf-8"))
    assert first_summary["seed"] == 11
    assert first_summary["split_seed"] == 101
    assert second_summary["seed"] == 11
    assert second_summary["split_seed"] == 202


def test_build_oasis_dataloaders_support_weighted_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weighted sampling should attach a sampler and keep the saved split artifacts intact."""

    settings = _build_settings(tmp_path)
    _write_manifest(settings, subjects_per_class=(8, 4))
    _patch_loader_dependencies(monkeypatch)
    monkeypatch.setattr("src.data.loaders._build_torch_generator", lambda seed: f"generator::{seed}")
    monkeypatch.setattr(
        "src.data.loaders._build_weighted_sampler",
        lambda records, seed, replacement: {
            "kind": "weighted_sampler",
            "seed": seed,
            "replacement": replacement,
            "count": len(records),
        },
    )

    bundle = build_oasis_dataloaders(
        OASISLoaderConfig(
            settings=settings,
            seed=5,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            batch_size=2,
            weighted_sampling=True,
        )
    )

    assert bundle.train_sampler == {
        "kind": "weighted_sampler",
        "seed": 5,
        "replacement": True,
        "count": len(bundle.dataset_bundle.train_records),
    }
    assert bundle.train_loader.sampler == bundle.train_sampler
    assert bundle.train_loader.shuffle is False
    assert bundle.train_loader.generator == "generator::5"
    assert bundle.dataset_bundle.train_class_weights[1] > bundle.dataset_bundle.train_class_weights[0]


def test_build_oasis_datasets_builds_longitudinal_visit_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated scans should keep subject grouping and produce ordered visit metadata."""

    settings = _build_settings(tmp_path)
    _write_manifest(
        settings,
        subjects_per_class=(6, 6),
        sessions_per_subject=3,
        reverse_timestamps_for_first_subject=True,
    )
    _patch_loader_dependencies(monkeypatch)

    bundle = build_oasis_datasets(
        OASISLoaderConfig(settings=settings, seed=13, train_fraction=0.5, val_fraction=0.25, test_fraction=0.25)
    )

    first_subject = bundle.split_artifacts.longitudinal_frame.loc[
        bundle.split_artifacts.longitudinal_frame["subject_id"] == "OAS1_0000"
    ].sort_values("visit_order")
    assert first_subject["visit_order"].tolist() == [1, 2, 3]
    assert first_subject["session_id"].tolist() == ["OAS1_0000_MR3", "OAS1_0000_MR2", "OAS1_0000_MR1"]
    assert first_subject["is_longitudinal_subject"].tolist() == [True, True, True]
    assert first_subject["subject_session_count"].tolist() == [3, 3, 3]

    subject_summary = bundle.split_artifacts.longitudinal_subject_summary.set_index("subject_id")
    assert bool(subject_summary.loc["OAS1_0000", "is_longitudinal_subject"]) is True
    assert int(subject_summary.loc["OAS1_0000", "subject_session_count"]) == 3


def test_build_oasis_datasets_requires_subject_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank subject identifiers should fail before split generation."""

    settings = _build_settings(tmp_path)
    manifest_path = _write_manifest(settings)
    frame = pd.read_csv(manifest_path)
    frame.loc[0, "subject_id"] = "   "
    frame.to_csv(manifest_path, index=False)
    _patch_loader_dependencies(monkeypatch)

    with pytest.raises(OASISLoaderError):
        build_oasis_datasets(OASISLoaderConfig(settings=settings))
