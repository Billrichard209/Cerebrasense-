"""Tests for the clean scan-level inference facade."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.configs.runtime import AppSettings
from src.inference.serving import BackendServingConfig
from src.inference.pipeline import PredictScanError, PredictScanOptions, predict_scan, validate_scan_path
from src.models.factory import OASISModelConfig
from src.models.registry import ModelRegistryEntry, save_oasis_model_entry
from src.storage import count_rows
from src.transforms.oasis_transforms import OASISTransformConfig


def _build_settings(tmp_path: Path) -> AppSettings:
    """Create isolated app settings for prediction tests."""

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


def _build_tiny_classifier(torch: object) -> object:
    """Build a tiny model that emits deterministic logits."""

    class _TinyClassifier(torch.nn.Module):
        def forward(self, image: object) -> object:
            batch_size = image.shape[0]
            return torch.tensor([[0.2, 1.2]] * batch_size, dtype=torch.float32, device=image.device)

    return _TinyClassifier()


class _FakeTransform:
    def __init__(self, torch: object) -> None:
        self.torch = torch

    def __call__(self, sample: dict[str, object]) -> dict[str, object]:
        image = self.torch.zeros((1, 6, 6, 6), dtype=self.torch.float32)
        image[:, 2:4, 2:4, 2:4] = 1.0
        return {**sample, "image": image}


def test_validate_scan_path_rejects_unsupported_formats(tmp_path: Path) -> None:
    """Scan validation should fail clearly before inference starts."""

    scan_path = tmp_path / "scan.txt"
    scan_path.write_text("not a scan", encoding="utf-8")

    with pytest.raises(PredictScanError):
        validate_scan_path(scan_path)


def test_predict_scan_returns_schema_and_saves_debug_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """predict_scan should return the backend inference schema and save reports."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    settings = _build_settings(tmp_path)
    scan_path = tmp_path / "scan.hdr"
    scan_path.write_text("placeholder", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": _build_tiny_classifier(torch).state_dict(), "epoch": 1}, checkpoint_path)

    monkeypatch.setattr("src.inference.pipeline.build_model", lambda _cfg: _build_tiny_classifier(torch))
    monkeypatch.setattr(
        "src.inference.pipeline.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 6, 6, 6)),
    )
    monkeypatch.setattr("src.inference.pipeline.load_oasis_transform_config", lambda _path=None: OASISTransformConfig())
    monkeypatch.setattr("src.inference.pipeline.build_oasis_infer_transforms", lambda _cfg: _FakeTransform(torch))

    payload = predict_scan(
        str(scan_path),
        str(checkpoint_path),
        None,
        options=PredictScanOptions(
            output_name="unit_scan_prediction",
            threshold=0.5,
            use_cached_model=False,
            save_debug_slices=True,
            subject_id="OAS1_0001",
            session_id="OAS1_0001_MR1",
        ),
        settings=settings,
    )

    assert payload["predicted_label"] == 1
    assert payload["label_name"] == "demented"
    assert payload["probability_score"] > 0.5
    assert payload["calibrated_probability_score"] > 0.5
    assert payload["confidence_score"] == pytest.approx(payload["probability_score"])
    assert payload["confidence_level"] in {"medium", "high"}
    assert payload["review_flag"] is False
    assert payload["prediction_id"] is not None
    assert payload["trace_id"] is not None
    assert payload["model_name"] == "densenet121_3d"
    assert payload["input_metadata"]["subject_id"] == "OAS1_0001"
    assert payload["decision_support_only"] is True
    assert "not a diagnosis" in payload["ai_summary"]
    assert payload["outputs"]["prediction_json"] is not None
    assert len(payload["outputs"]["processed_slices"]) == 3
    assert all(Path(path).exists() for path in payload["outputs"]["processed_slices"])

    saved_payload = json.loads(Path(payload["outputs"]["prediction_json"]).read_text(encoding="utf-8"))
    assert saved_payload["predicted_label"] == payload["predicted_label"]
    assert saved_payload["input_metadata"]["source_path"] == str(scan_path)
    assert saved_payload["confidence_level"] == payload["confidence_level"]
    assert count_rows("scans", settings=settings) == 1
    assert count_rows("inference_logs", settings=settings) == 1
    assert count_rows("review_queue", settings=settings) == 0


def test_predict_scan_queues_low_confidence_cases_for_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-confidence predictions should be written to the review queue."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    scan_path = tmp_path / "scan.hdr"
    scan_path.write_text("placeholder", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": _build_tiny_classifier(torch).state_dict(), "epoch": 1}, checkpoint_path)

    class _LowConfidenceClassifier(torch.nn.Module):
        def forward(self, image: object) -> object:
            batch_size = image.shape[0]
            return torch.tensor([[0.0, 0.02]] * batch_size, dtype=torch.float32, device=image.device)

    monkeypatch.setattr("src.inference.pipeline.build_model", lambda _cfg: _LowConfidenceClassifier())
    monkeypatch.setattr(
        "src.inference.pipeline.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 6, 6, 6)),
    )
    monkeypatch.setattr("src.inference.pipeline.load_oasis_transform_config", lambda _path=None: OASISTransformConfig())
    monkeypatch.setattr("src.inference.pipeline.build_oasis_infer_transforms", lambda _cfg: _FakeTransform(torch))

    payload = predict_scan(
        str(scan_path),
        str(checkpoint_path),
        None,
        options=PredictScanOptions(
            output_name="unit_scan_prediction_low_confidence",
            threshold=0.5,
            use_cached_model=False,
            subject_id="OAS1_0002",
        ),
        settings=settings,
    )

    assert payload["review_flag"] is True
    assert payload["confidence_level"] == "low"
    assert count_rows("review_queue", settings=settings) == 1


def test_predict_scan_forces_manual_review_when_active_model_is_on_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operational hold should force review even when confidence is otherwise acceptable."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    scan_path = tmp_path / "scan.hdr"
    scan_path.write_text("placeholder", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": _build_tiny_classifier(torch).state_dict(), "epoch": 1}, checkpoint_path)

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
        operational_status="hold",
        serving_restrictions={
            "force_manual_review": True,
            "allow_prediction_output": True,
            "block_as_operational_default": True,
        },
    )
    save_oasis_model_entry(registry_entry, settings=settings)
    monkeypatch.setattr(
        "src.inference.pipeline.load_backend_serving_config",
        lambda *_args, **_kwargs: BackendServingConfig(
            active_oasis_model_registry=settings.outputs_root / "model_registry" / "oasis_current_baseline.json",
        ),
    )
    monkeypatch.setattr("src.inference.pipeline.build_model", lambda _cfg: _build_tiny_classifier(torch))
    monkeypatch.setattr(
        "src.inference.pipeline.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 6, 6, 6)),
    )
    monkeypatch.setattr("src.inference.pipeline.load_oasis_transform_config", lambda _path=None: OASISTransformConfig())
    monkeypatch.setattr("src.inference.pipeline.build_oasis_infer_transforms", lambda _cfg: _FakeTransform(torch))

    payload = predict_scan(
        str(scan_path),
        str(checkpoint_path),
        None,
        options=PredictScanOptions(
            output_name="unit_scan_prediction_hold",
            threshold=0.5,
            use_cached_model=False,
        ),
        settings=settings,
    )

    assert payload["confidence_level"] in {"medium", "high"}
    assert payload["review_flag"] is True
    assert payload["operational_status"] == "hold"
    assert payload["serving_restrictions"]["force_manual_review"] is True
    assert "governance hold" in payload["ai_summary"].lower()
    assert count_rows("review_queue", settings=settings) == 1


def test_predict_scan_uses_active_registry_image_size_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active registry image_size should override the generic transform YAML spatial size."""

    torch = pytest.importorskip("torch")
    settings = _build_settings(tmp_path)
    scan_path = tmp_path / "scan.hdr"
    scan_path.write_text("placeholder", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": _build_tiny_classifier(torch).state_dict(), "epoch": 1}, checkpoint_path)

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

    monkeypatch.setattr("src.inference.pipeline.build_model", lambda _cfg: _build_tiny_classifier(torch))
    monkeypatch.setattr(
        "src.inference.pipeline.load_oasis_model_config",
        lambda _path=None: OASISModelConfig(expected_input_shape=(1, 1, 6, 6, 6)),
    )
    monkeypatch.setattr("src.inference.pipeline.load_oasis_transform_config", lambda _path=None: OASISTransformConfig())
    monkeypatch.setattr("src.inference.pipeline.build_oasis_infer_transforms", _capture_transform)

    payload = predict_scan(
        str(scan_path),
        str(checkpoint_path),
        None,
        options=PredictScanOptions(
            output_name="unit_scan_prediction_registry_size",
            use_cached_model=False,
        ),
        settings=settings,
    )

    assert captured_spatial_size["value"] == (64, 64, 64)
    assert payload["preprocessing_overrides"]["spatial_size"] == [64, 64, 64]
