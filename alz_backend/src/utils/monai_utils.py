"""Lazy MONAI and torch import helpers for medical imaging workflows."""

from __future__ import annotations

from typing import Any


def load_monai_transform_symbols() -> dict[str, Any]:
    """Load MONAI transform classes only when a pipeline is actually built."""

    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        Lambdad,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        RandAffined,
        RandFlipd,
        RandGaussianNoised,
        RandRotate90d,
        ResizeWithPadOrCropd,
        Resized,
        ScaleIntensityRangePercentilesd,
        Spacingd,
    )

    return {
        "Compose": Compose,
        "CropForegroundd": CropForegroundd,
        "EnsureChannelFirstd": EnsureChannelFirstd,
        "EnsureTyped": EnsureTyped,
        "Lambdad": Lambdad,
        "LoadImaged": LoadImaged,
        "NormalizeIntensityd": NormalizeIntensityd,
        "Orientationd": Orientationd,
        "RandAffined": RandAffined,
        "RandFlipd": RandFlipd,
        "RandGaussianNoised": RandGaussianNoised,
        "RandRotate90d": RandRotate90d,
        "ResizeWithPadOrCropd": ResizeWithPadOrCropd,
        "Resized": Resized,
        "ScaleIntensityRangePercentilesd": ScaleIntensityRangePercentilesd,
        "Spacingd": Spacingd,
    }


def load_monai_data_symbols() -> dict[str, Any]:
    """Load MONAI dataset and dataloader classes lazily."""

    from monai.data import CacheDataset, DataLoader, Dataset

    return {
        "CacheDataset": CacheDataset,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
    }


def load_monai_network_symbols() -> dict[str, Any]:
    """Load MONAI network implementations lazily."""

    from monai.networks.nets import DenseNet121

    return {"DenseNet121": DenseNet121}


def load_monai_inferer_symbols() -> dict[str, Any]:
    """Load MONAI inferer helpers lazily."""

    from monai.inferers import SimpleInferer

    return {"SimpleInferer": SimpleInferer}


def load_torch_symbols() -> dict[str, Any]:
    """Load torch modules lazily so non-training code avoids startup cost."""

    import torch
    from torch import nn, optim

    return {
        "torch": torch,
        "nn": nn,
        "optim": optim,
    }
