"""Build a single OASIS demo bundle from the active local registry and API surface."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import create_app  # noqa: E402
from src.configs.runtime import AppSettings, get_app_settings  # noqa: E402
from src.models.registry import load_current_oasis_model_entry  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402
from scripts.build_scope_evidence_report import (  # noqa: E402
    build_scope_aligned_evidence_report,
    save_scope_aligned_evidence_report,
)
from scripts.check_backend_readiness import (  # noqa: E402
    build_backend_readiness_report,
    save_backend_readiness_report,
)


def _safe_name(value: str) -> str:
    """Return a path-safe bundle name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _resolve_sample_scan_path(scan_path: Path | None, settings: AppSettings) -> Path:
    """Resolve a stable sample scan for local demos."""

    if scan_path is not None:
        resolved = scan_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Demo scan path not found: {resolved}")
        return resolved

    bundle_root = settings.outputs_root / "exports" / "oasis1_upload_bundle" / "OASIS"
    bundle_candidates = sorted(bundle_root.rglob("*.hdr"))
    if bundle_candidates:
        return bundle_candidates[0].resolve()

    manifest_path = settings.data_root / "interim" / "oasis1_manifest.csv"
    if manifest_path.exists():
        frame = pd.read_csv(manifest_path)
        for image_value in frame.get("image", []).tolist():
            candidate = Path(str(image_value)).expanduser().resolve()
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not auto-resolve a sample OASIS scan. Provide --scan-path explicitly."
    )


def _match_manifest_metadata(scan_path: Path, settings: AppSettings) -> tuple[str | None, str | None]:
    """Look up subject/session metadata for the chosen scan from the local manifest when possible."""

    manifest_path = settings.data_root / "interim" / "oasis1_manifest.csv"
    if not manifest_path.exists():
        return None, None
    frame = pd.read_csv(manifest_path)
    resolved_scan = str(scan_path.expanduser().resolve())
    for row in frame.to_dict(orient="records"):
        image_path = Path(str(row.get("image", ""))).expanduser().resolve()
        if str(image_path) == resolved_scan:
            subject_id = row.get("subject_id")
            session_id = row.get("session_id")
            return (
                None if subject_id in {None, ""} else str(subject_id),
                None if session_id in {None, ""} else str(session_id),
            )
    return None, None


def _parse_subject_session_from_scan_path(scan_path: Path) -> tuple[str | None, str | None]:
    """Infer OASIS-style subject/session ids from a scan path when possible."""

    match = re.search(r"(OAS\d?_\d{4})_(MR\d+)", str(scan_path))
    if not match:
        return None, None
    subject_id = match.group(1)
    session_id = f"{subject_id}_{match.group(2)}"
    return subject_id, session_id


def _resolve_subject_session(scan_path: Path, settings: AppSettings) -> tuple[str, str]:
    """Resolve subject/session ids for demo requests."""

    manifest_subject_id, manifest_session_id = _match_manifest_metadata(scan_path, settings)
    if manifest_subject_id and manifest_session_id:
        return manifest_subject_id, manifest_session_id

    parsed_subject_id, parsed_session_id = _parse_subject_session_from_scan_path(scan_path)
    if parsed_subject_id and parsed_session_id:
        return parsed_subject_id, parsed_session_id

    return "OAS1_DEMO", "OAS1_DEMO_MR1"


def _build_demo_longitudinal_request(
    *,
    subject_id: str,
    scan_path: Path,
    probability_score: float,
    output_name: str,
) -> dict[str, Any]:
    """Build a stable explicit longitudinal request that does not depend on repeated OASIS visits."""

    baseline_probability = max(0.0, min(1.0, probability_score - 0.10))
    followup_probability = max(0.0, min(1.0, probability_score))
    return {
        "subject_id": subject_id,
        "output_name": output_name,
        "records": [
            {
                "subject_id": subject_id,
                "session_id": f"{subject_id}_DEMO_VISIT1",
                "visit_order": 1,
                "summary_label": "baseline",
                "scan_timestamp": "2001-01-01",
                "source_path": str(scan_path),
                "dataset": "oasis1",
                "volumetric_features": {
                    "left_hippocampus_volume_mm3": 3420.0,
                    "right_hippocampus_volume_mm3": 3380.0,
                    "ventricular_proxy_volume_mm3": 14900.0,
                },
                "model_probabilities": {
                    "ad_like_probability": baseline_probability,
                },
            },
            {
                "subject_id": subject_id,
                "session_id": f"{subject_id}_DEMO_VISIT2",
                "visit_order": 2,
                "summary_label": "follow_up",
                "scan_timestamp": "2002-01-01",
                "source_path": str(scan_path),
                "dataset": "oasis1",
                "volumetric_features": {
                    "left_hippocampus_volume_mm3": 3250.0,
                    "right_hippocampus_volume_mm3": 3205.0,
                    "ventricular_proxy_volume_mm3": 15720.0,
                },
                "model_probabilities": {
                    "ad_like_probability": followup_probability,
                },
            },
        ],
    }


def _copy_if_exists(source_path: Path, destination_path: Path) -> None:
    """Copy one file when it exists."""

    if source_path.exists():
        shutil.copy2(source_path, destination_path)


def _write_json(path: Path, payload: Any) -> None:
    """Save a JSON payload with indentation."""

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _request_json(
    client: TestClient,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call one API route and return the JSON response or raise a helpful error."""

    request_kwargs: dict[str, Any] = {"headers": headers}
    if json_payload is not None:
        request_kwargs["json"] = json_payload
    response = getattr(client, method.lower())(path, **request_kwargs)
    if response.status_code != 200:
        raise RuntimeError(f"{method.upper()} {path} failed: status={response.status_code} body={response.text}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"{method.upper()} {path} did not return a JSON object.")
    return payload


def build_oasis_demo_bundle(
    *,
    settings: AppSettings | None = None,
    scan_path: Path | None = None,
    registry_path: Path | None = None,
    device: str = "cpu",
    output_name: str = "oasis_demo_bundle",
    api_key: str | None = None,
    skip_explanation: bool = False,
) -> dict[str, Any]:
    """Build the full local OASIS demo bundle from the active local backend."""

    resolved_settings = settings or get_app_settings()
    safe_output_name = _safe_name(output_name)
    bundle_root = ensure_directory(resolved_settings.outputs_root / "reports" / "demo" / safe_output_name)
    sample_scan_path = _resolve_sample_scan_path(scan_path, resolved_settings)
    subject_id, session_id = _resolve_subject_session(sample_scan_path, resolved_settings)

    active_registry_path = resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    resolved_registry_path = (
        registry_path.expanduser().resolve()
        if registry_path is not None
        else active_registry_path
    )
    registry_entry = load_current_oasis_model_entry(path=resolved_registry_path, settings=resolved_settings)
    if not registry_entry.model_config_path or not registry_entry.preprocessing_config_path:
        raise FileNotFoundError("Active registry is missing model or preprocessing config paths.")

    headers = {"X-API-Key": api_key} if api_key else None
    client = TestClient(create_app())

    root_payload = _request_json(client, "get", "/", headers=headers)
    health_payload = _request_json(client, "get", "/health", headers=headers)
    metadata_payload = _request_json(client, "get", "/models/oasis/metadata", headers=headers)
    active_model_payload = _request_json(client, "get", "/models/oasis/active", headers=headers)
    review_dashboard_payload = _request_json(client, "get", "/reviews/dashboard", headers=headers)

    prediction_request = {
        "scan_path": str(sample_scan_path),
        "checkpoint_path": registry_entry.checkpoint_path,
        "config_path": registry_entry.preprocessing_config_path,
        "model_config_path": registry_entry.model_config_path,
        "output_name": f"{safe_output_name}_prediction",
        "threshold": float(registry_entry.recommended_threshold),
        "device": device,
        "subject_id": subject_id,
        "session_id": session_id,
    }
    prediction_payload = _request_json(client, "post", "/predict/scan", headers=headers, json_payload=prediction_request)

    explanation_payload: dict[str, Any] | None = None
    if not skip_explanation:
        explanation_request = {
            "scan_path": str(sample_scan_path),
            "checkpoint_path": registry_entry.checkpoint_path,
            "preprocessing_config_path": registry_entry.preprocessing_config_path,
            "model_config_path": registry_entry.model_config_path,
            "output_name": f"{safe_output_name}_explanation",
            "target_layer": "auto",
            "device": device,
            "save_saliency": True,
        }
        explanation_payload = _request_json(
            client,
            "post",
            "/explain/scan",
            headers=headers,
            json_payload=explanation_request,
        )

    longitudinal_request = _build_demo_longitudinal_request(
        subject_id=subject_id,
        scan_path=sample_scan_path,
        probability_score=float(prediction_payload.get("probability_score", 0.5)),
        output_name=f"{safe_output_name}_longitudinal",
    )
    longitudinal_payload = _request_json(
        client,
        "post",
        "/longitudinal/report",
        headers=headers,
        json_payload=longitudinal_request,
    )

    readiness_report = build_backend_readiness_report(resolved_settings)
    readiness_json_path, readiness_md_path = save_backend_readiness_report(readiness_report, resolved_settings)
    evidence_report = build_scope_aligned_evidence_report(resolved_settings)
    evidence_json_path, evidence_md_path = save_scope_aligned_evidence_report(
        evidence_report,
        resolved_settings,
        file_stem="scope_aligned_evidence_report",
    )

    registry_copy_path = bundle_root / "requested_registry.json"
    shutil.copy2(resolved_registry_path, registry_copy_path)
    if resolved_registry_path != active_registry_path and active_registry_path.exists():
        shutil.copy2(active_registry_path, bundle_root / "active_registry.json")
    _copy_if_exists(readiness_json_path, bundle_root / readiness_json_path.name)
    _copy_if_exists(readiness_md_path, bundle_root / readiness_md_path.name)
    _copy_if_exists(evidence_json_path, bundle_root / evidence_json_path.name)
    _copy_if_exists(evidence_md_path, bundle_root / evidence_md_path.name)

    _write_json(bundle_root / "root.json", root_payload)
    _write_json(bundle_root / "health.json", health_payload)
    _write_json(bundle_root / "model_metadata.json", metadata_payload)
    _write_json(bundle_root / "active_model.json", active_model_payload)
    _write_json(bundle_root / "review_dashboard.json", review_dashboard_payload)
    _write_json(bundle_root / "prediction.json", prediction_payload)
    if explanation_payload is not None:
        _write_json(bundle_root / "explanation.json", explanation_payload)
    _write_json(bundle_root / "longitudinal.json", longitudinal_payload)

    summary = {
        "output_name": safe_output_name,
        "bundle_root": str(bundle_root),
        "sample_scan_path": str(sample_scan_path),
        "subject_id": subject_id,
        "session_id": session_id,
        "requested_registry_path": str(resolved_registry_path),
        "active_registry_path": str(active_registry_path),
        "requested_run_name": registry_entry.run_name,
        "active_run_name": active_model_payload.get("run_name"),
        "recommended_threshold": active_model_payload.get("recommended_threshold"),
        "requested_recommended_threshold": registry_entry.recommended_threshold,
        "using_active_registry": resolved_registry_path == active_registry_path,
        "readiness_report_json": str(readiness_json_path),
        "evidence_report_json": str(evidence_json_path),
        "saved_bundle_files": sorted(path.name for path in bundle_root.glob("*.json")) + sorted(
            path.name for path in bundle_root.glob("*.md")
        ),
        "prediction_output": prediction_payload.get("outputs", {}),
        "explanation_artifacts": None if explanation_payload is None else explanation_payload.get("artifacts", {}),
        "review_summary": review_dashboard_payload.get("summary", {}),
        "longitudinal_output_path": longitudinal_payload.get("output_path"),
        "notes": [
            "This bundle exercises the local FastAPI surface through an in-process TestClient.",
            "Prediction/explanation use the requested OASIS registry checkpoint and threshold.",
            "Longitudinal payload uses explicit demo records for reproducible timeline output.",
        ],
    }
    if resolved_registry_path != active_registry_path:
        summary["notes"].append(
            "The requested registry differs from the backend active registry, so prediction/explanation artifacts reflect the requested model while review/dashboard metadata still reflects the active backend state."
        )
    _write_json(bundle_root / "demo_summary.json", summary)

    summary_md_lines = [
        "# OASIS Demo Bundle",
        "",
        f"- output_name: {safe_output_name}",
        f"- requested_run_name: {summary.get('requested_run_name')}",
        f"- active_run_name: {summary.get('active_run_name')}",
        f"- sample_scan_path: {sample_scan_path}",
        f"- subject_id: {subject_id}",
        f"- session_id: {session_id}",
        f"- readiness_report_json: {readiness_json_path}",
        f"- evidence_report_json: {evidence_json_path}",
        f"- longitudinal_output_path: {summary.get('longitudinal_output_path')}",
        "",
        "## Notes",
        "",
    ]
    summary_md_lines.extend(f"- {note}" for note in summary["notes"])
    (bundle_root / "demo_summary.md").write_text("\n".join(summary_md_lines), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build a local OASIS demo bundle from the active registry and API surface.")
    parser.add_argument("--scan-path", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-name", type=str, default="oasis_demo_bundle")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--skip-explanation", action="store_true")
    return parser


def main() -> None:
    """Build the demo bundle and print a compact summary."""

    args = build_parser().parse_args()
    summary = build_oasis_demo_bundle(
        scan_path=args.scan_path,
        registry_path=args.registry_path,
        device=args.device,
        output_name=args.output_name,
        api_key=args.api_key,
        skip_explanation=args.skip_explanation,
    )
    print(f"bundle_root={summary['bundle_root']}")
    print(f"sample_scan_path={summary['sample_scan_path']}")
    print(f"active_run_name={summary['active_run_name']}")
    print(f"readiness_report_json={summary['readiness_report_json']}")
    print(f"evidence_report_json={summary['evidence_report_json']}")
    print(f"longitudinal_output_path={summary['longitudinal_output_path']}")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
