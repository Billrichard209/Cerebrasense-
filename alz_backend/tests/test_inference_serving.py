"""Tests for serving-time inference configuration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.services import save_raw_scan_upload
from src.configs.runtime import AppSettings
from src.inference.serving import (
    BackendServingConfig,
    ConfidenceBandConfig,
    OASISDecisionPolicy,
    ThresholdPolicyConfig,
    load_backend_serving_config,
    resolve_oasis_decision_policy,
    resolve_inference_threshold,
)
from src.models.registry import ModelRegistryEntry, save_oasis_model_entry


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for serving tests."""

    project_root = tmp_path / "alz_backend"
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    storage_root = project_root / "storage"
    config_root = project_root / "configs"
    data_root.mkdir(parents=True)
    outputs_root.mkdir(parents=True)
    storage_root.mkdir(parents=True)
    config_root.mkdir(parents=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=project_root.parent,
        collection_root=project_root.parent,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=project_root.parent,
        oasis_source_root=project_root.parent / "OASIS",
        storage_root=storage_root,
        database_path=storage_root / "backend.sqlite3",
        serving_config_path=config_root / "backend_serving.yaml",
    )


def test_resolve_inference_threshold_uses_registry_when_explicit_threshold_is_missing() -> None:
    """Serving should be able to defer to the promoted registry threshold."""

    registry_entry = ModelRegistryEntry(
        registry_version="1.0",
        model_id="oasis_test",
        dataset="oasis1",
        run_name="run",
        checkpoint_path="checkpoint.pt",
        model_config_path=None,
        preprocessing_config_path=None,
        image_size=[64, 64, 64],
        promoted_at_utc="2026-01-01T00:00:00+00:00",
        decision_support_only=True,
        clinical_disclaimer="Decision-support only.",
        default_threshold=0.5,
        recommended_threshold=0.73,
    )
    serving_config = BackendServingConfig(
        active_oasis_model_registry=Path("registry.json"),
        threshold_policy=ThresholdPolicyConfig(use_registry_recommended_threshold=True, fallback_threshold=0.5),
    )

    threshold = resolve_inference_threshold(
        explicit_threshold=None,
        registry_entry=registry_entry,
        serving_config=serving_config,
    )

    assert threshold == pytest.approx(0.73)


def test_load_backend_serving_config_reads_yaml_sections(tmp_path: Path) -> None:
    """The serving config loader should parse threshold and upload policies."""

    settings = _build_settings(tmp_path)
    settings.serving_config_path.write_text(
        "\n".join(
            [
                "default_device: cuda",
                "threshold_policy:",
                "  use_registry_recommended_threshold: true",
                "  fallback_threshold: 0.61",
                "scan_validation:",
                "  max_file_size_mb: 8",
                "  allowed_suffixes: ['.nii', '.nii.gz']",
            ]
        ),
        encoding="utf-8",
    )

    serving_config = load_backend_serving_config(settings=settings)

    assert serving_config.default_device == "cuda"
    assert serving_config.threshold_policy.fallback_threshold == pytest.approx(0.61)
    assert serving_config.scan_validation.max_file_size_mb == 8
    assert serving_config.scan_validation.allowed_suffixes == (".nii", ".nii.gz")


def test_load_backend_serving_config_resolves_relative_registry_path_from_project_root(tmp_path: Path) -> None:
    """Relative registry paths in backend_serving.yaml should resolve from the backend project root."""

    settings = _build_settings(tmp_path)
    settings.serving_config_path.write_text(
        "active_oasis_model_registry: outputs/model_registry/oasis_current_baseline.json\n",
        encoding="utf-8",
    )

    serving_config = load_backend_serving_config(settings=settings)

    assert serving_config.active_oasis_model_registry == (
        settings.project_root / "outputs" / "model_registry" / "oasis_current_baseline.json"
    )


def test_save_raw_scan_upload_enforces_serving_upload_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw upload handling should reject files above the serving-size limit."""

    settings = _build_settings(tmp_path)
    settings.serving_config_path.write_text(
        "\n".join(
            [
                "scan_validation:",
                "  max_file_size_mb: 1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.api.services.get_app_settings", lambda: settings)

    with pytest.raises(ValueError, match="serving limit"):
        save_raw_scan_upload(
            file_name="scan.nii.gz",
            content=b"x" * (2 * 1024 * 1024),
            output_name="oversized_upload",
        )


def test_resolve_oasis_decision_policy_uses_registry_threshold_and_confidence(tmp_path: Path) -> None:
    """Shared OASIS decision policy should honor promoted registry threshold and confidence bands."""

    settings = _build_settings(tmp_path)
    registry_entry = ModelRegistryEntry(
        registry_version="1.0",
        model_id="oasis_current_baseline",
        dataset="oasis1",
        run_name="run",
        checkpoint_path="checkpoint.pt",
        model_config_path=None,
        preprocessing_config_path=None,
        image_size=[64, 64, 64],
        promoted_at_utc="2026-01-01T00:00:00+00:00",
        decision_support_only=True,
        clinical_disclaimer="Decision-support only.",
        default_threshold=0.5,
        recommended_threshold=0.45,
        temperature_scaling={"enabled": False, "temperature": 1.2},
        confidence_policy={
            "high_confidence_min": 0.9,
            "medium_confidence_min": 0.7,
            "high_entropy_max": 0.3,
            "medium_entropy_max": 0.8,
        },
    )
    save_oasis_model_entry(registry_entry, settings=settings)

    policy = resolve_oasis_decision_policy(settings=settings)

    assert isinstance(policy, OASISDecisionPolicy)
    assert isinstance(policy.confidence_config, ConfidenceBandConfig)
    assert policy.threshold == pytest.approx(0.45)
    assert policy.confidence_config.temperature == pytest.approx(1.2)
    assert policy.confidence_config.medium_confidence_min == pytest.approx(0.7)
