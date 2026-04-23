"""Tests for the OASIS batch inference helper."""

from __future__ import annotations

from pathlib import Path

from src.configs.runtime import AppSettings

from scripts import batch_predict_oasis_scans as batch_module


class _RegistryEntry:
    run_name = "oasis_active"
    checkpoint_path = "outputs/runs/oasis/oasis_active/checkpoints/best_model.pt"
    model_config_path = "configs/oasis_model.yaml"
    preprocessing_config_path = "configs/oasis_transforms.yaml"
    recommended_threshold = 0.45
    confidence_policy = {}
    temperature_scaling = {}


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    config_root = project_root / "configs"
    outputs_root = project_root / "outputs"
    data_root = project_root / "data"
    config_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path,
        oasis_source_root=tmp_path / "OASIS",
    )


def test_build_batch_oasis_predictions_writes_summary(tmp_path: Path, monkeypatch) -> None:
    """Batch inference should save a CSV and JSON/Markdown summary for one folder."""

    settings = _settings(tmp_path)
    scan_root = tmp_path / "scans"
    scan_a = scan_root / "disc1" / "OAS1_0001_MR1_scan.hdr"
    scan_b = scan_root / "disc1" / "OAS1_0002_MR1_scan.hdr"
    scan_a.parent.mkdir(parents=True, exist_ok=True)
    scan_a.write_text("hdr", encoding="utf-8")
    scan_b.write_text("hdr", encoding="utf-8")

    monkeypatch.setattr(batch_module, "load_current_oasis_model_entry", lambda *_args, **_kwargs: _RegistryEntry())

    def _fake_predict_scan(scan_path: str, checkpoint_path: str, config_path: str, *, options, settings=None):  # noqa: ANN001
        return {
            "predicted_label": 0,
            "label_name": "nondemented",
            "probability_score": 0.12,
            "confidence_score": 0.88,
            "confidence_level": "high",
            "review_flag": False,
            "outputs": {
                "prediction_json": str(settings.outputs_root / "predictions" / f"{Path(scan_path).stem}.json"),
            },
        }

    monkeypatch.setattr(batch_module, "predict_scan", _fake_predict_scan)

    summary = batch_module.build_batch_oasis_predictions(
        settings=settings,
        scan_root=scan_root,
        output_name="unit_batch",
    )

    assert summary["scan_count"] == 2
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert Path(summary["batch_predictions_csv"]).exists()
    assert Path(summary["summary_json_path"]).exists()
    assert Path(summary["summary_md_path"]).exists()


def test_build_batch_oasis_predictions_records_failures(tmp_path: Path, monkeypatch) -> None:
    """Batch inference should keep going and record per-scan failures by default."""

    settings = _settings(tmp_path)
    scan_root = tmp_path / "scans"
    scan_a = scan_root / "disc1" / "OAS1_0001_MR1_scan.hdr"
    scan_b = scan_root / "disc1" / "OAS1_0002_MR1_scan.hdr"
    scan_a.parent.mkdir(parents=True, exist_ok=True)
    scan_a.write_text("hdr", encoding="utf-8")
    scan_b.write_text("hdr", encoding="utf-8")

    monkeypatch.setattr(batch_module, "load_current_oasis_model_entry", lambda *_args, **_kwargs: _RegistryEntry())

    def _fake_predict_scan(scan_path: str, checkpoint_path: str, config_path: str, *, options, settings=None):  # noqa: ANN001
        if "0002" in scan_path:
            raise RuntimeError("bad scan")
        return {
            "predicted_label": 1,
            "label_name": "demented",
            "probability_score": 0.76,
            "confidence_score": 0.76,
            "confidence_level": "medium",
            "review_flag": True,
            "outputs": {
                "prediction_json": str(settings.outputs_root / "predictions" / f"{Path(scan_path).stem}.json"),
            },
        }

    monkeypatch.setattr(batch_module, "predict_scan", _fake_predict_scan)

    summary = batch_module.build_batch_oasis_predictions(
        settings=settings,
        scan_root=scan_root,
        output_name="unit_batch_failures",
    )

    assert summary["scan_count"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["review_required"] == 1
