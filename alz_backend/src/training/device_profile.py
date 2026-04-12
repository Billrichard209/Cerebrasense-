"""Training-device profiling helpers for laptop-safe 3D MRI experiments."""

from __future__ import annotations

import ctypes
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.configs.runtime import AppSettings, get_app_settings
from src.utils.io_utils import ensure_directory


@dataclass(slots=True)
class TrainingDeviceProfile:
    """A JSON-safe snapshot of local hardware readiness for 3D MONAI training."""

    python_version: str
    platform: str
    cpu_count: int | None
    total_memory_gb: float | None
    available_memory_gb: float | None
    torch_version: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    recommended_device: str
    can_run_full_3d_densenet: bool
    recommended_config_path: str
    warnings: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe profile payload."""

        return asdict(self)


class _MemoryStatusEx(ctypes.Structure):
    """Windows MEMORYSTATUSEX structure used without adding psutil."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _read_memory_gb() -> tuple[float | None, float | None]:
    """Return total and available memory in GB when the OS exposes it."""

    if os.name == "nt":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            gb = 1024**3
            return round(status.ullTotalPhys / gb, 2), round(status.ullAvailPhys / gb, 2)
    return None, None


def _read_torch_profile() -> tuple[str | None, bool, int, str | None]:
    """Return torch/CUDA capability without making torch a hard import."""

    try:
        import torch
    except ImportError:
        return None, False, 0, None

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return str(torch.__version__), cuda_available, cuda_device_count, cuda_device_name


def collect_training_device_profile(
    *,
    settings: AppSettings | None = None,
    minimum_cpu_memory_gb: float = 12.0,
    low_available_memory_gb: float = 2.0,
) -> TrainingDeviceProfile:
    """Collect a conservative readiness profile for local OASIS 3D training."""

    resolved_settings = settings or get_app_settings()
    total_memory_gb, available_memory_gb = _read_memory_gb()
    torch_version, cuda_available, cuda_device_count, cuda_device_name = _read_torch_profile()
    low_memory_mode = bool(cuda_available and available_memory_gb is not None and available_memory_gb < low_available_memory_gb)
    recommended_config_name = (
        "oasis_train_rtx2050_lowram.yaml"
        if low_memory_mode
        else ("oasis_train_rtx2050_gpu.yaml" if cuda_available else "oasis_train_laptop_cpu.yaml")
    )
    recommended_config_path = resolved_settings.project_root / "configs" / recommended_config_name

    warnings: list[str] = []
    recommendations: list[str] = []
    recommended_device = "cuda" if cuda_available else "cpu"

    if not cuda_available:
        warnings.append("CUDA is not available; 3D DenseNet training will run on CPU and may be slow.")
        recommendations.append("Use the laptop CPU config with batch_size=1, cache_rate=0.0, and image_size=[64,64,64].")
    else:
        if low_memory_mode:
            recommendations.append(
                "Use the low-RAM RTX GPU config with batch_size=1, gradient_accumulation_steps=2, mixed_precision=true, and image_size=[64,64,64]."
            )
        else:
            recommendations.append(
                "Use the RTX GPU config with device=cuda, mixed_precision=true, batch_size=2, and image_size=[64,64,64]."
            )

    if total_memory_gb is not None and total_memory_gb < minimum_cpu_memory_gb:
        warnings.append(
            f"Total RAM is about {total_memory_gb} GB; full 3D training is constrained on this machine."
        )
        recommendations.append("Avoid larger 3D inputs such as 128x128x128 unless a GPU or more RAM is available.")

    if available_memory_gb is not None and available_memory_gb < low_available_memory_gb:
        warnings.append(
            f"Available RAM is about {available_memory_gb} GB; defer heavy training until memory is freed."
        )
        recommendations.append("Close unrelated apps or switch to the low-RAM RTX config before starting longer training.")

    can_run_full_3d_densenet = bool(
        cuda_available or (total_memory_gb is not None and total_memory_gb >= minimum_cpu_memory_gb)
    )

    if can_run_full_3d_densenet:
        recommendations.append("Full 3D DenseNet experiments are feasible on the GPU, but still monitor VRAM and RAM.")
    else:
        recommendations.append("Prefer the saved best checkpoint or a lighter future model over long CPU-only runs.")

    return TrainingDeviceProfile(
        python_version=platform.python_version(),
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        total_memory_gb=total_memory_gb,
        available_memory_gb=available_memory_gb,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        recommended_device=recommended_device,
        can_run_full_3d_densenet=can_run_full_3d_densenet,
        recommended_config_path=str(recommended_config_path),
        warnings=warnings,
        recommendations=recommendations,
    )


def save_training_device_profile(
    profile: TrainingDeviceProfile,
    output_path: str | Path | None = None,
    *,
    settings: AppSettings | None = None,
) -> Path:
    """Save a training-device profile JSON report."""

    resolved_settings = settings or get_app_settings()
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else resolved_settings.outputs_root / "reports" / "readiness" / "training_device_profile.json"
    )
    ensure_directory(resolved_output_path.parent)
    resolved_output_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return resolved_output_path
