"""Tests for local training-device readiness profiling."""

from __future__ import annotations

from pathlib import Path

from src.configs.runtime import AppSettings
from src.training.device_profile import TrainingDeviceProfile, collect_training_device_profile, save_training_device_profile


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=project_root / "data",
        outputs_root=project_root / "outputs",
        oasis_source_root=project_root.parent / "OASIS",
        kaggle_source_root=project_root.parent / "archive (1)",
    )


def test_collect_training_device_profile_is_json_safe(tmp_path: Path) -> None:
    """The profile should be serializable and include the laptop-safe config path."""

    profile = collect_training_device_profile(settings=_settings(tmp_path))

    payload = profile.to_dict()
    assert payload["recommended_config_path"].endswith(".yaml")
    assert "oasis_train_" in payload["recommended_config_path"]
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["recommendations"], list)


def test_save_training_device_profile_writes_json(tmp_path: Path) -> None:
    """The training-device profile should write to the requested report path."""

    profile = TrainingDeviceProfile(
        python_version="3.11",
        platform="unit-test",
        cpu_count=8,
        total_memory_gb=8.0,
        available_memory_gb=1.0,
        torch_version="2.x",
        cuda_available=False,
        cuda_device_count=0,
        cuda_device_name=None,
        recommended_device="cpu",
        can_run_full_3d_densenet=False,
        recommended_config_path="configs/oasis_train_laptop_cpu.yaml",
        warnings=["low memory"],
        recommendations=["use laptop config"],
    )
    output_path = save_training_device_profile(profile, settings=_settings(tmp_path))

    assert output_path.exists()
    assert "low memory" in output_path.read_text(encoding="utf-8")


def test_collect_training_device_profile_prefers_lowram_gpu_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Low available RAM on a CUDA laptop should recommend the low-RAM GPU config."""

    monkeypatch.setattr("src.training.device_profile._read_memory_gb", lambda: (8.0, 1.0))
    monkeypatch.setattr(
        "src.training.device_profile._read_torch_profile",
        lambda: ("2.x", True, 1, "RTX 2050"),
    )

    profile = collect_training_device_profile(settings=_settings(tmp_path))

    assert profile.recommended_device == "cuda"
    assert profile.recommended_config_path.endswith("oasis_train_rtx2050_lowram.yaml")
    assert any("gradient_accumulation_steps=2" in item for item in profile.recommendations)
