"""Tests for MONAI-oriented dataset, transform, and model integration layers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.configs.runtime import AppSettings
from src.data.kaggle_dataset import (
    build_kaggle_monai_dataset,
    build_kaggle_monai_records,
    infer_kaggle_dataset_type,
    load_kaggle_manifest,
)
from src.data.oasis_dataset import build_oasis_monai_dataloader, build_oasis_monai_dataset, build_oasis_monai_records
from src.models.kaggle_model import KaggleMonaiModelConfig, build_kaggle_monai_network
from src.models.oasis_model import build_oasis_monai_network
from src.transforms.kaggle_transforms import (
    build_kaggle_infer_transforms,
    build_kaggle_monai_transforms,
    build_kaggle_train_transforms,
    load_kaggle_transform_config,
)
from src.transforms.oasis_transforms import (
    build_oasis_infer_transforms,
    build_oasis_monai_transforms,
    build_oasis_train_transforms,
    load_oasis_transform_config,
)


def _build_settings(tmp_path: Path, *, kaggle_source_root: Path | None = None) -> AppSettings:
    """Create isolated app settings for MONAI-facing tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    data_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)
    resolved_kaggle_root = kaggle_source_root or (tmp_path / "kaggle_source")
    resolved_kaggle_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=resolved_kaggle_root,
        oasis_source_root=project_root.parent / "OASIS",
    )


class _FakeDataset:
    def __init__(self, *, data: list[dict], transform: object) -> None:
        self.data = data
        self.transform = transform


class _FakeCacheDataset(_FakeDataset):
    def __init__(self, *, data: list[dict], transform: object, cache_rate: float, num_workers: int) -> None:
        super().__init__(data=data, transform=transform)
        self.cache_rate = cache_rate
        self.num_workers = num_workers


class _FakeDataLoader:
    def __init__(self, dataset: object, *, batch_size: int, shuffle: bool, num_workers: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


class _FakeDenseNet121:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeTransformFactory:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, **kwargs: object) -> dict[str, object]:
        return {"name": self.name, "kwargs": kwargs}


def test_oasis_monai_dataset_defaults_to_3d_and_builds_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OASIS manifest-driven dataset builders should default to 3D MONAI records."""

    settings = _build_settings(tmp_path)
    interim_root = settings.data_root / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)
    image_path = interim_root / "scan_001.hdr"
    image_path.write_text("hdr", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "image": str(image_path),
                "label": 1,
                "label_name": "demented",
                "subject_id": "OAS1_0001",
                "scan_timestamp": None,
                "dataset": "oasis1",
                "meta": json.dumps({"session_id": "OAS1_0001_MR1"}),
            }
        ]
    ).to_csv(interim_root / "oasis1_train_manifest.csv", index=False)

    monkeypatch.setattr(
        "src.data.base_dataset._load_monai_data_symbols",
        lambda: {"Dataset": _FakeDataset, "CacheDataset": _FakeCacheDataset, "DataLoader": _FakeDataLoader},
    )
    monkeypatch.setattr("src.data.oasis_dataset.build_oasis_train_transforms", lambda *_: "oasis_train_transform")
    monkeypatch.setattr("src.data.oasis_dataset.build_oasis_val_transforms", lambda *_: "oasis_val_transform")

    records = build_oasis_monai_records(settings=settings, split="train")
    assert records[0]["dataset_type"] == "3d_volumes"
    assert records[0]["label"] == 1

    dataset = build_oasis_monai_dataset(settings=settings, split="train", training=True)
    assert isinstance(dataset, _FakeDataset)
    assert dataset.transform == "oasis_train_transform"

    loader = build_oasis_monai_dataloader(settings=settings, split="train", training=True, batch_size=2)
    assert isinstance(loader, _FakeDataLoader)
    assert loader.batch_size == 2
    assert loader.shuffle is True


def test_kaggle_monai_records_require_explicit_label_mapping_for_training(tmp_path: Path) -> None:
    """Kaggle MONAI records should not fabricate numeric labels without an explicit mapping."""

    kaggle_root = tmp_path / "kaggle_source"
    settings = _build_settings(tmp_path, kaggle_source_root=kaggle_root)
    interim_root = settings.data_root / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)
    image_path = kaggle_root / "OriginalDataset" / "NonDemented" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_text("jpg", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "image": str(image_path),
                "label": None,
                "label_name": "NonDemented",
                "subject_id": None,
                "scan_timestamp": None,
                "dataset": "kaggle_alz",
                "dataset_type": "2d_slices",
                "meta": json.dumps({"subset": "OriginalDataset"}),
            }
        ]
    ).to_csv(interim_root / "kaggle_alz_train_manifest.csv", index=False)

    with pytest.raises(ValueError):
        build_kaggle_monai_records(settings=settings, split="train", require_labels=True)


def test_kaggle_monai_dataset_applies_explicit_runtime_label_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kaggle MONAI dataset builders should honor explicit runtime label maps."""

    kaggle_root = tmp_path / "kaggle_source"
    settings = _build_settings(tmp_path, kaggle_source_root=kaggle_root)
    interim_root = settings.data_root / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)
    first_image = kaggle_root / "OriginalDataset" / "NonDemented" / "sample_001.jpg"
    second_image = kaggle_root / "OriginalDataset" / "MildDemented" / "sample_002.jpg"
    first_image.parent.mkdir(parents=True, exist_ok=True)
    second_image.parent.mkdir(parents=True, exist_ok=True)
    first_image.write_text("jpg", encoding="utf-8")
    second_image.write_text("jpg", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "image": str(first_image),
                "label": None,
                "label_name": "NonDemented",
                "subject_id": None,
                "scan_timestamp": None,
                "dataset": "kaggle_alz",
                "dataset_type": "2d_slices",
                "meta": json.dumps({"subset": "OriginalDataset"}),
            },
            {
                "image": str(second_image),
                "label": None,
                "label_name": "MildDemented",
                "subject_id": None,
                "scan_timestamp": None,
                "dataset": "kaggle_alz",
                "dataset_type": "2d_slices",
                "meta": json.dumps({"subset": "OriginalDataset"}),
            },
        ]
    ).to_csv(interim_root / "kaggle_alz_train_manifest.csv", index=False)

    monkeypatch.setattr(
        "src.data.base_dataset._load_monai_data_symbols",
        lambda: {"Dataset": _FakeDataset, "CacheDataset": _FakeCacheDataset, "DataLoader": _FakeDataLoader},
    )
    monkeypatch.setattr("src.data.kaggle_dataset.build_kaggle_train_transforms", lambda *_, **__: "kaggle_train_transform")
    monkeypatch.setattr("src.data.kaggle_dataset.build_kaggle_val_transforms", lambda *_, **__: "kaggle_val_transform")

    records = build_kaggle_monai_records(
        settings=settings,
        split="train",
        require_labels=True,
        label_map={"NonDemented": 0, "MildDemented": 1},
    )
    assert [record["label"] for record in records] == [0, 1]
    assert records[0]["meta"]["explicit_runtime_label_map_applied"] is True

    dataset = build_kaggle_monai_dataset(
        settings=settings,
        split="train",
        training=True,
        label_map={"NonDemented": 0, "MildDemented": 1},
    )
    assert isinstance(dataset, _FakeDataset)
    assert dataset.transform == "kaggle_train_transform"
    assert infer_kaggle_dataset_type(load_kaggle_manifest(settings=settings, split="train")) == "2d_slices"


def test_monai_transform_builders_use_framework_specific_stage_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transform builders should assemble MONAI-style Compose pipelines."""

    fake_symbols = {
        "Compose": lambda transforms: {"compose": transforms},
        "CropForegroundd": _FakeTransformFactory("CropForegroundd"),
        "EnsureChannelFirstd": _FakeTransformFactory("EnsureChannelFirstd"),
        "EnsureTyped": _FakeTransformFactory("EnsureTyped"),
        "Lambdad": _FakeTransformFactory("Lambdad"),
        "LoadImaged": _FakeTransformFactory("LoadImaged"),
        "NormalizeIntensityd": _FakeTransformFactory("NormalizeIntensityd"),
        "Orientationd": _FakeTransformFactory("Orientationd"),
        "RandAffined": _FakeTransformFactory("RandAffined"),
        "RandFlipd": _FakeTransformFactory("RandFlipd"),
        "RandGaussianNoised": _FakeTransformFactory("RandGaussianNoised"),
        "RandRotate90d": _FakeTransformFactory("RandRotate90d"),
        "ResizeWithPadOrCropd": _FakeTransformFactory("ResizeWithPadOrCropd"),
        "Resized": _FakeTransformFactory("Resized"),
        "ScaleIntensityRangePercentilesd": _FakeTransformFactory("ScaleIntensityRangePercentilesd"),
        "Spacingd": _FakeTransformFactory("Spacingd"),
    }
    monkeypatch.setattr("src.transforms.oasis_transforms._load_monai_transform_symbols", lambda: fake_symbols)
    monkeypatch.setattr("src.transforms.kaggle_transforms._load_monai_transform_symbols", lambda: fake_symbols)

    oasis_pipeline = build_oasis_monai_transforms(training=True)
    kaggle_pipeline = build_kaggle_monai_transforms(dataset_type="2d_slices", training=False)

    oasis_names = [step["name"] for step in oasis_pipeline["compose"]]
    kaggle_names = [step["name"] for step in kaggle_pipeline["compose"]]

    assert "LoadImaged" in oasis_names
    assert "Spacingd" in oasis_names
    assert "RandGaussianNoised" in oasis_names
    assert "Lambdad" in kaggle_names
    assert "Resized" in kaggle_names
    assert "Orientationd" not in kaggle_names


def test_oasis_yaml_config_loader_and_explicit_builders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The OASIS YAML config should drive the explicit train/infer builders."""

    config_path = tmp_path / "oasis_transforms.yaml"
    config_path.write_text(
        """
spacing:
  enabled: false
skull_strip:
  enabled: false
augmentation:
  enabled: true
  affine_probability: 0.3
  rotate_range_degrees: [3.0, 3.0, 3.0]
  scale_range: [0.01, 0.01, 0.01]
""",
        encoding="utf-8",
    )

    fake_symbols = {
        "Compose": lambda transforms: {"compose": transforms},
        "CropForegroundd": _FakeTransformFactory("CropForegroundd"),
        "EnsureChannelFirstd": _FakeTransformFactory("EnsureChannelFirstd"),
        "EnsureTyped": _FakeTransformFactory("EnsureTyped"),
        "Lambdad": _FakeTransformFactory("Lambdad"),
        "LoadImaged": _FakeTransformFactory("LoadImaged"),
        "NormalizeIntensityd": _FakeTransformFactory("NormalizeIntensityd"),
        "Orientationd": _FakeTransformFactory("Orientationd"),
        "RandAffined": _FakeTransformFactory("RandAffined"),
        "RandFlipd": _FakeTransformFactory("RandFlipd"),
        "RandGaussianNoised": _FakeTransformFactory("RandGaussianNoised"),
        "RandRotate90d": _FakeTransformFactory("RandRotate90d"),
        "ResizeWithPadOrCropd": _FakeTransformFactory("ResizeWithPadOrCropd"),
        "Resized": _FakeTransformFactory("Resized"),
        "ScaleIntensityRangePercentilesd": _FakeTransformFactory("ScaleIntensityRangePercentilesd"),
        "Spacingd": _FakeTransformFactory("Spacingd"),
    }
    monkeypatch.setattr("src.transforms.oasis_transforms._load_monai_transform_symbols", lambda: fake_symbols)

    cfg = load_oasis_transform_config(config_path)
    train_pipeline = build_oasis_train_transforms(cfg)
    infer_pipeline = build_oasis_infer_transforms(cfg)
    train_names = [step["name"] for step in train_pipeline["compose"]]
    infer_names = [step["name"] for step in infer_pipeline["compose"]]

    assert cfg.spacing.enabled is False
    assert cfg.skull_strip.enabled is False
    assert "Spacingd" not in train_names
    assert "CropForegroundd" not in train_names
    assert "RandAffined" in train_names
    assert "RandAffined" not in infer_names


def test_kaggle_yaml_config_loader_and_explicit_builders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Kaggle YAML config should drive the explicit train/infer builders."""

    config_path = tmp_path / "kaggle_transforms.yaml"
    config_path.write_text(
        """
foreground:
  enabled: true
augmentation:
  enabled: true
  affine_probability_2d: 0.3
  horizontal_flip_probability_2d: 0.1
spatial:
  image_size_2d: [192, 192]
""",
        encoding="utf-8",
    )

    fake_symbols = {
        "Compose": lambda transforms: {"compose": transforms},
        "CropForegroundd": _FakeTransformFactory("CropForegroundd"),
        "EnsureChannelFirstd": _FakeTransformFactory("EnsureChannelFirstd"),
        "EnsureTyped": _FakeTransformFactory("EnsureTyped"),
        "Lambdad": _FakeTransformFactory("Lambdad"),
        "LoadImaged": _FakeTransformFactory("LoadImaged"),
        "NormalizeIntensityd": _FakeTransformFactory("NormalizeIntensityd"),
        "Orientationd": _FakeTransformFactory("Orientationd"),
        "RandAffined": _FakeTransformFactory("RandAffined"),
        "RandFlipd": _FakeTransformFactory("RandFlipd"),
        "RandGaussianNoised": _FakeTransformFactory("RandGaussianNoised"),
        "RandRotate90d": _FakeTransformFactory("RandRotate90d"),
        "ResizeWithPadOrCropd": _FakeTransformFactory("ResizeWithPadOrCropd"),
        "Resized": _FakeTransformFactory("Resized"),
        "ScaleIntensityRangePercentilesd": _FakeTransformFactory("ScaleIntensityRangePercentilesd"),
        "Spacingd": _FakeTransformFactory("Spacingd"),
    }
    monkeypatch.setattr("src.transforms.kaggle_transforms._load_monai_transform_symbols", lambda: fake_symbols)

    cfg = load_kaggle_transform_config(config_path)
    train_pipeline = build_kaggle_train_transforms(cfg, dataset_type="2d_slices")
    infer_pipeline = build_kaggle_infer_transforms(cfg, dataset_type="2d_slices")
    train_names = [step["name"] for step in train_pipeline["compose"]]
    infer_names = [step["name"] for step in infer_pipeline["compose"]]

    assert cfg.foreground.enabled is True
    assert cfg.spatial.image_size_2d == (192, 192)
    assert "CropForegroundd" in train_names
    assert "RandAffined" in train_names
    assert "RandFlipd" in train_names
    assert "RandAffined" not in infer_names


def test_monai_model_builders_choose_expected_spatial_dims(monkeypatch: pytest.MonkeyPatch) -> None:
    """OASIS should default to 3D MONAI models and Kaggle slices to 2D ones."""

    monkeypatch.setattr(
        "src.models.base_model._load_monai_network_symbols",
        lambda: {"DenseNet121": _FakeDenseNet121},
    )
    monkeypatch.setattr(
        "src.models.factory._load_monai_network_symbols",
        lambda: {"DenseNet121": _FakeDenseNet121},
    )

    oasis_model = build_oasis_monai_network()
    kaggle_model = build_kaggle_monai_network(KaggleMonaiModelConfig(dataset_type="2d_slices", out_channels=4))

    assert oasis_model.kwargs["spatial_dims"] == 3
    assert oasis_model.kwargs["out_channels"] == 2
    assert kaggle_model.kwargs["spatial_dims"] == 2
    assert kaggle_model.kwargs["out_channels"] == 4
