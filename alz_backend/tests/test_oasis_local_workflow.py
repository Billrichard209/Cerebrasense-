"""Tests for the combined local OASIS workflow runner."""

from __future__ import annotations

from pathlib import Path

from src.configs.runtime import AppSettings

from scripts import build_oasis_local_workflow as workflow_module


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


def test_build_oasis_local_workflow_writes_combined_summary(tmp_path: Path, monkeypatch) -> None:
    """The combined workflow should save a workflow summary that links both phases."""

    settings = _settings(tmp_path)
    scan_root = tmp_path / "scans"
    demo_scan = scan_root / "disc1" / "OAS1_0001_MR1_scan.hdr"
    demo_scan.parent.mkdir(parents=True, exist_ok=True)
    demo_scan.write_text("hdr", encoding="utf-8")

    monkeypatch.setattr(
        workflow_module,
        "build_oasis_demo_bundle",
        lambda **kwargs: {
            "bundle_root": str(settings.outputs_root / "reports" / "demo" / "unit_demo"),
            "sample_scan_path": str(demo_scan),
            "requested_run_name": "oasis_active",
        },
    )
    monkeypatch.setattr(
        workflow_module,
        "build_batch_oasis_predictions",
        lambda **kwargs: {
            "report_root": str(settings.outputs_root / "reports" / "batch_inference" / "unit_batch"),
            "batch_predictions_csv": str(settings.outputs_root / "reports" / "batch_inference" / "unit_batch" / "batch_predictions.csv"),
        },
    )

    summary = workflow_module.build_oasis_local_workflow(
        settings=settings,
        scan_root=scan_root,
        output_name="unit_workflow",
    )

    assert summary["batch_enabled"] is True
    assert summary["demo_scan_path"] == str(demo_scan)
    assert Path(summary["summary_json_path"]).exists()
    assert Path(summary["summary_md_path"]).exists()
