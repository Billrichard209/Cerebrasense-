"""Tests for the local OASIS demo bundle builder."""

from __future__ import annotations

import json
from pathlib import Path

from src.configs.runtime import AppSettings

from scripts import build_oasis_demo_bundle as demo_module


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
    (config_root / "oasis_model.yaml").write_text("architecture: densenet121_3d\n", encoding="utf-8")
    (config_root / "oasis_transforms.yaml").write_text("spatial:\n  spatial_size: [64, 64, 64]\n", encoding="utf-8")
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


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeClient:
    def __init__(self, _app) -> None:
        self._responses = {
            ("get", "/"): {"message": "Structural MRI backend core is running.", "mode": "decision_support", "primary_dataset": "oasis"},
            ("get", "/health"): {"status": "ok", "primary_dataset": "oasis", "decision_support_only": True},
            ("get", "/models/oasis/metadata"): {"dataset": "oasis1", "framework": "monai", "decision_support_only": True},
            ("get", "/models/oasis/active"): {"run_name": "oasis_colab_full_v3_auroc_monitor", "recommended_threshold": 0.45},
            ("get", "/reviews/dashboard"): {"summary": {"recommended_action": "Continue routine monitoring."}},
            ("post", "/predict/scan"): {
                "probability_score": 0.24,
                "outputs": {"prediction_json": "prediction.json"},
            },
            ("post", "/explain/scan"): {
                "artifacts": {"report_json": "explanation.json"},
            },
            ("post", "/longitudinal/report"): {
                "output_path": "outputs/reports/longitudinal/demo.json",
            },
        }

    def get(self, path: str, json=None, headers=None):  # noqa: ANN001
        return _FakeResponse(self._responses[("get", path)])

    def post(self, path: str, json=None, headers=None):  # noqa: ANN001
        return _FakeResponse(self._responses[("post", path)])


def _seed_registry_and_scan(settings: AppSettings) -> Path:
    scan_path = settings.outputs_root / "exports" / "oasis1_upload_bundle" / "OASIS" / "disc1" / "OAS1_0001_MR1.hdr"
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    scan_path.write_text("placeholder", encoding="utf-8")
    scan_path.with_suffix(".img").write_text("placeholder", encoding="utf-8")

    checkpoint_path = settings.outputs_root / "runs" / "oasis" / "oasis_colab_full_v3_auroc_monitor" / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")

    _write_json(
        settings.outputs_root / "model_registry" / "oasis_current_baseline.json",
        {
            "registry_version": "1.0",
            "model_id": "oasis_current_baseline",
            "dataset": "oasis1",
            "run_name": "oasis_colab_full_v3_auroc_monitor",
            "checkpoint_path": str(checkpoint_path),
            "model_config_path": str(settings.project_root / "configs" / "oasis_model.yaml"),
            "preprocessing_config_path": str(settings.project_root / "configs" / "oasis_transforms.yaml"),
            "image_size": [64, 64, 64],
            "promoted_at_utc": "2026-01-01T00:00:00+00:00",
            "decision_support_only": True,
            "clinical_disclaimer": "Decision-support only, not a diagnosis. Use clinical judgment.",
            "recommended_threshold": 0.45,
            "default_threshold": 0.5,
        },
    )
    return scan_path


def test_resolve_sample_scan_path_prefers_bundle_export(tmp_path: Path) -> None:
    """The demo bundle should auto-resolve a scan from the exported OASIS bundle first."""

    settings = _settings(tmp_path)
    scan_path = _seed_registry_and_scan(settings)

    resolved = demo_module._resolve_sample_scan_path(None, settings)

    assert resolved == scan_path.resolve()


def test_build_demo_bundle_writes_summary_and_payloads(tmp_path: Path, monkeypatch) -> None:
    """The demo bundle builder should save bundled JSON outputs from the API surface."""

    settings = _settings(tmp_path)
    scan_path = _seed_registry_and_scan(settings)

    readiness_json = settings.outputs_root / "reports" / "readiness" / "backend_readiness.json"
    readiness_md = settings.outputs_root / "reports" / "readiness" / "backend_readiness.md"
    evidence_json = settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.json"
    evidence_md = settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.md"

    monkeypatch.setattr(demo_module, "create_app", lambda: object())
    monkeypatch.setattr(demo_module, "TestClient", _FakeClient)
    monkeypatch.setattr(demo_module, "build_backend_readiness_report", lambda settings=None: {"overall_status": "pass"})
    monkeypatch.setattr(demo_module, "save_backend_readiness_report", lambda report, settings=None: (readiness_json, readiness_md))
    monkeypatch.setattr(demo_module, "build_scope_aligned_evidence_report", lambda settings=None: {"goal_statement": "scope"})
    monkeypatch.setattr(
        demo_module,
        "save_scope_aligned_evidence_report",
        lambda report, settings=None, file_stem="scope_aligned_evidence_report": (evidence_json, evidence_md),
    )

    readiness_json.parent.mkdir(parents=True, exist_ok=True)
    readiness_json.write_text("{}", encoding="utf-8")
    readiness_md.write_text("readiness", encoding="utf-8")
    evidence_json.parent.mkdir(parents=True, exist_ok=True)
    evidence_json.write_text("{}", encoding="utf-8")
    evidence_md.write_text("evidence", encoding="utf-8")

    summary = demo_module.build_oasis_demo_bundle(
        settings=settings,
        scan_path=scan_path,
        output_name="unit_demo_bundle",
    )

    bundle_root = Path(summary["bundle_root"])
    assert bundle_root.exists()
    assert (bundle_root / "demo_summary.json").exists()
    assert (bundle_root / "prediction.json").exists()
    assert (bundle_root / "explanation.json").exists()
    assert (bundle_root / "review_dashboard.json").exists()


def test_build_demo_bundle_supports_requested_candidate_registry(tmp_path: Path, monkeypatch) -> None:
    """A custom registry path should drive prediction artifacts without changing active metadata."""

    settings = _settings(tmp_path)
    scan_path = _seed_registry_and_scan(settings)
    candidate_checkpoint_path = settings.outputs_root / "runs" / "oasis" / "oasis_candidate_v3" / "checkpoints" / "best_model.pt"
    candidate_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint_path.write_bytes(b"candidate-checkpoint")
    candidate_registry_path = settings.outputs_root / "model_registry" / "oasis_candidate_v3.json"
    _write_json(
        candidate_registry_path,
        {
            "registry_version": "1.0",
            "model_id": "oasis_current_baseline",
            "dataset": "oasis1",
            "run_name": "oasis_candidate_v3",
            "checkpoint_path": str(candidate_checkpoint_path),
            "model_config_path": str(settings.project_root / "configs" / "oasis_model.yaml"),
            "preprocessing_config_path": str(settings.project_root / "configs" / "oasis_transforms.yaml"),
            "image_size": [64, 64, 64],
            "promoted_at_utc": "2026-01-01T00:00:00+00:00",
            "decision_support_only": True,
            "clinical_disclaimer": "Decision-support only, not a diagnosis. Use clinical judgment.",
            "recommended_threshold": 0.41,
            "default_threshold": 0.5,
        },
    )

    readiness_json = settings.outputs_root / "reports" / "readiness" / "backend_readiness.json"
    readiness_md = settings.outputs_root / "reports" / "readiness" / "backend_readiness.md"
    evidence_json = settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.json"
    evidence_md = settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.md"

    monkeypatch.setattr(demo_module, "create_app", lambda: object())
    monkeypatch.setattr(demo_module, "TestClient", _FakeClient)
    monkeypatch.setattr(demo_module, "build_backend_readiness_report", lambda settings=None: {"overall_status": "pass"})
    monkeypatch.setattr(demo_module, "save_backend_readiness_report", lambda report, settings=None: (readiness_json, readiness_md))
    monkeypatch.setattr(demo_module, "build_scope_aligned_evidence_report", lambda settings=None: {"goal_statement": "scope"})
    monkeypatch.setattr(
        demo_module,
        "save_scope_aligned_evidence_report",
        lambda report, settings=None, file_stem="scope_aligned_evidence_report": (evidence_json, evidence_md),
    )

    readiness_json.parent.mkdir(parents=True, exist_ok=True)
    readiness_json.write_text("{}", encoding="utf-8")
    readiness_md.write_text("readiness", encoding="utf-8")
    evidence_json.parent.mkdir(parents=True, exist_ok=True)
    evidence_json.write_text("{}", encoding="utf-8")
    evidence_md.write_text("evidence", encoding="utf-8")

    summary = demo_module.build_oasis_demo_bundle(
        settings=settings,
        scan_path=scan_path,
        registry_path=candidate_registry_path,
        output_name="candidate_demo_bundle",
    )

    assert summary["requested_run_name"] == "oasis_candidate_v3"
    assert summary["active_run_name"] == "oasis_colab_full_v3_auroc_monitor"
    assert summary["using_active_registry"] is False
    bundle_root = Path(summary["bundle_root"])
    assert (bundle_root / "requested_registry.json").exists()
    assert (bundle_root / "active_registry.json").exists()
