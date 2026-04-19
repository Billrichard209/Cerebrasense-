"""Tests for the OASIS presentation summary builder."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings

from scripts import build_oasis_presentation_summary as presentation_module


def _settings(tmp_path: Path) -> AppSettings:
    project_root = tmp_path / "alz_backend"
    config_root = project_root / "configs"
    outputs_root = project_root / "outputs"
    data_root = project_root / "data"
    config_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    (config_root / "backend_serving.yaml").write_text(
        "active_oasis_model_registry: outputs/model_registry/oasis_current_baseline.json\n",
        encoding="utf-8",
    )
    return AppSettings(
        project_root=project_root,
        workspace_root=tmp_path,
        collection_root=tmp_path,
        data_root=data_root,
        outputs_root=outputs_root,
        kaggle_source_root=tmp_path,
        oasis_source_root=tmp_path / "OASIS",
        serving_config_path=config_root / "backend_serving.yaml",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_oasis_presentation_summary_uses_existing_reports(tmp_path: Path) -> None:
    """The presentation summary should condense comparison, evidence, and productization artifacts."""

    settings = _settings(tmp_path)
    comparison_path = settings.outputs_root / "reports" / "comparison" / "oasis_baseline_comparison.json"
    evidence_path = settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.json"
    productization_path = settings.outputs_root / "reports" / "productization" / "oasis_productization_status.json"
    _write_json(
        comparison_path,
        {
            "active": {
                "run_name": "oasis_baseline_rtx2050_gpu_seed42_split42",
                "test_metrics": {"auroc": 0.8793650793650795, "review_required_count": 10},
                "threshold_calibration": {"test_metrics": {"f1": 0.8484848484848485}},
            },
            "candidate": {
                "run_name": "oasis_colab_full_v3_auroc_monitor",
                "test_metrics": {"auroc": 0.8634920634920635, "review_required_count": 23},
                "threshold_calibration": {"test_metrics": {"f1": 0.8108108108108109}},
            },
            "delta": {
                "validation_auroc": 0.05,
                "test_auroc": -0.015873015873015928,
                "threshold_test_f1": -0.03767403767403765,
                "test_review_required_count": 13.0,
            },
            "recommendation": {"action": "keep_active"},
            "demo_bundles": {
                "active": {"bundle_root": "demo_active", "prediction_output": {"prediction_json": "active_prediction.json"}},
                "candidate": {"bundle_root": "demo_candidate", "prediction_output": {"prediction_json": "candidate_prediction.json"}},
            },
        },
    )
    _write_json(
        evidence_path,
        {
            "oasis_repeated_splits": {
                "test_aggregate": {
                    "auroc": {"mean": 0.8793650793650793},
                }
            }
        },
    )
    _write_json(
        productization_path,
        {
            "overall_status": "fail",
            "summary": {"pass": 6, "fail": 1},
        },
    )

    summary = presentation_module.build_oasis_presentation_summary(
        settings=settings,
        comparison_report_path=comparison_path,
        evidence_report_path=evidence_path,
        productization_report_path=productization_path,
    )

    assert summary.recommendation == "keep_active"
    assert summary.project_state["productization_overall_status"] == "fail"
    assert summary.key_metrics["active_test_auroc"] == 0.8793650793650795
    assert summary.demo_assets["candidate_bundle_root"] == "demo_candidate"

    json_path, md_path = presentation_module.save_oasis_presentation_summary(summary, settings)

    assert json_path.exists()
    assert md_path.exists()
    assert "OASIS Presentation Summary" in md_path.read_text(encoding="utf-8")
