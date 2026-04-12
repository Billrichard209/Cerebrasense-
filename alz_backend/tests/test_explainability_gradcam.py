"""Tests for Grad-CAM-style scan explanations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.configs.runtime import AppSettings
from src.explainability.gradcam import ExplainScanConfig, explain_scan
from src.models.factory import OASISModelConfig
from src.models.registry import ModelRegistryEntry, save_oasis_model_entry
from src.transforms.oasis_transforms import OASISTransformConfig


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for explainability tests."""

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


def _build_tiny_3d_model(torch: object) -> object:
    """Build a tiny 3D model with an explicit target layer."""

    class _Tiny3DModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool3d(1)
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, image: object) -> object:
            features = torch.relu(self.conv(image))
            pooled = self.pool(features).flatten(1)
            return self.fc(pooled)

    return _Tiny3DModel()


class _FakeTransform:
    def __init__(self, torch: object) -> None:
        self.torch = torch

    def __call__(self, sample: dict[str, object]) -> dict[str, object]:
        image = self.torch.zeros((1, 8, 8, 8), dtype=self.torch.float32)
        image[:, 2:6, 2:6, 2:6] = 1.0
        return {**sample, "image": image}


def test_explain_scan_generates_gradcam_and_saliency_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grad-CAM explanation should save report and selected overlay images."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    settings = _build_settings(tmp_path)
    scan_path = tmp_path / "scan.hdr"
    scan_path.write_text("placeholder", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = _build_tiny_3d_model(torch)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, checkpoint_path)

    monkeypatch.setattr("src.explainability.gradcam.build_model", lambda _cfg: _build_tiny_3d_model(torch))
    monkeypatch.setattr(
        "src.explainability.gradcam.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 8, 8, 8)),
    )
    monkeypatch.setattr("src.explainability.gradcam.load_oasis_transform_config", lambda _path=None: OASISTransformConfig())
    monkeypatch.setattr("src.explainability.gradcam.build_oasis_infer_transforms", lambda _cfg: _FakeTransform(torch))

    result = explain_scan(
        ExplainScanConfig(
            scan_path=scan_path,
            checkpoint_path=checkpoint_path,
            output_name="unit_explanation",
            target_layer="conv",
            target_class=1,
            image_size=(8, 8, 8),
            slice_indices=(4,),
        ),
        settings=settings,
    )

    assert result.report_path.exists()
    assert len(result.overlay_paths) == 1
    assert len(result.saliency_paths) == 1
    assert result.overlay_paths[0].exists()
    assert result.saliency_paths[0].exists()
    assert result.payload["target_layer"] == "conv"
    assert result.payload["target_class"] == 1
    assert "region_importance_proxy" in result.payload
    assert "uncertainty" in result.payload
    assert "heatmap_intensity_summary" in result.payload
    assert result.payload["highlighted_regions"]
    assert result.payload["explanation_quality"] in {"clear", "uncertain"}
    assert "not a diagnosis" in " ".join(result.payload["limitations"])

    saved_payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved_payload["method"] == "grad_cam_style_3d"
    assert saved_payload["artifacts"]["gradcam_overlays"]
    assert "confidence_interpretation" in saved_payload


def test_explain_scan_uses_active_registry_image_size_override_when_not_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explainability should align with the active registry image_size when the checkpoint matches."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    settings = _build_settings(tmp_path)
    scan_path = tmp_path / "scan.hdr"
    scan_path.write_text("placeholder", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = _build_tiny_3d_model(torch)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, checkpoint_path)

    registry_entry = ModelRegistryEntry(
        registry_version="1.0",
        model_id="oasis_current_baseline",
        dataset="oasis1",
        run_name="unit_run",
        checkpoint_path=str(checkpoint_path),
        model_config_path="configs/oasis_model.yaml",
        preprocessing_config_path="configs/oasis_transforms.yaml",
        image_size=[64, 64, 64],
        promoted_at_utc="2026-01-01T00:00:00+00:00",
        decision_support_only=True,
        clinical_disclaimer="Decision-support only.",
    )
    save_oasis_model_entry(registry_entry, settings=settings)

    captured_spatial_size: dict[str, tuple[int, int, int]] = {}

    def _capture_transform(cfg: OASISTransformConfig):
        captured_spatial_size["value"] = tuple(cfg.spatial.spatial_size)
        return _FakeTransform(torch)

    monkeypatch.setattr("src.explainability.gradcam.build_model", lambda _cfg: _build_tiny_3d_model(torch))
    monkeypatch.setattr(
        "src.explainability.gradcam.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 8, 8, 8)),
    )
    monkeypatch.setattr("src.explainability.gradcam.load_oasis_transform_config", lambda _path=None: OASISTransformConfig())
    monkeypatch.setattr("src.explainability.gradcam.build_oasis_infer_transforms", _capture_transform)

    result = explain_scan(
        ExplainScanConfig(
            scan_path=scan_path,
            checkpoint_path=checkpoint_path,
            output_name="unit_explanation_registry_size",
            target_layer="conv",
            target_class=1,
            slice_indices=(4,),
        ),
        settings=settings,
    )

    assert captured_spatial_size["value"] == (64, 64, 64)
    assert result.payload["preprocessing_overrides"]["spatial_size"] == [64, 64, 64]
