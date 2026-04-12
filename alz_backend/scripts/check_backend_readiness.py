"""Operational readiness checks for the structural MRI backend core.

This script is intentionally lightweight. It verifies project structure,
configuration files, important data artifacts, optional trained artifacts,
package availability, API importability, and security disclaimer wiring without
running expensive model training or inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import AppSettings, get_app_settings  # noqa: E402
from src.security.disclaimers import STANDARD_DECISION_SUPPORT_DISCLAIMER  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402


@dataclass(slots=True, frozen=True)
class ReadinessCheck:
    """One readiness check result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendReadinessReport:
    """Readiness report for local development and handoff checks."""

    generated_at: str
    overall_status: str
    checks: list[ReadinessCheck]
    recommendations: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the report to JSON-safe data."""

        status_counts = {"pass": 0, "warn": 0, "fail": 0}
        for check in self.checks:
            status_counts[check.status] = status_counts.get(check.status, 0) + 1
        return {
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
            "summary": status_counts,
            "checks": [asdict(check) for check in self.checks],
            "recommendations": list(self.recommendations),
        }


def _check_path(name: str, path: Path, *, required: bool = True, expected_type: str = "any") -> ReadinessCheck:
    """Check one file or directory path."""

    exists = path.exists()
    type_ok = exists and (
        expected_type == "any"
        or (expected_type == "file" and path.is_file())
        or (expected_type == "directory" and path.is_dir())
    )
    status = "pass" if type_ok else ("fail" if required else "warn")
    expected = f"required {expected_type}" if required else f"optional {expected_type}"
    message = f"{name} is available." if type_ok else f"{name} is missing or not a {expected_type}."
    return ReadinessCheck(
        name=name,
        status=status,
        message=message,
        details={"path": str(path), "expected": expected, "exists": exists},
    )


def _check_import(name: str, package_name: str, *, required: bool = True) -> ReadinessCheck:
    """Check if a Python package can be discovered without importing it."""

    found = importlib.util.find_spec(package_name) is not None
    return ReadinessCheck(
        name=name,
        status="pass" if found else ("fail" if required else "warn"),
        message=f"{package_name} is installed." if found else f"{package_name} was not found.",
        details={"package": package_name, "required": required},
    )


def _check_api_importable() -> ReadinessCheck:
    """Check that the FastAPI app factory imports successfully."""

    try:
        from src.api.main import create_app

        app = create_app()
    except Exception as exc:  # pragma: no cover - failure path is intentionally diagnostic.
        return ReadinessCheck(
            name="api_import",
            status="fail",
            message="FastAPI app factory could not be imported.",
            details={"error": repr(exc)},
        )
    return ReadinessCheck(
        name="api_import",
        status="pass",
        message="FastAPI app factory imports successfully.",
        details={"title": app.title, "version": app.version},
    )


def _check_disclaimer() -> ReadinessCheck:
    """Check that the standard clinical disclaimer has the required posture."""

    lowered = STANDARD_DECISION_SUPPORT_DISCLAIMER.lower()
    ok = "not a diagnosis" in lowered and "clinical judgment" in lowered
    return ReadinessCheck(
        name="decision_support_disclaimer",
        status="pass" if ok else "fail",
        message="Standard disclaimer is decision-support aligned." if ok else "Standard disclaimer is missing required wording.",
        details={"disclaimer": STANDARD_DECISION_SUPPORT_DISCLAIMER},
    )


def _required_project_checks(settings: AppSettings) -> list[ReadinessCheck]:
    """Check required project folders and source modules."""

    project_root = settings.project_root
    paths = [
        ("project_root", project_root, True, "directory"),
        ("src_root", project_root / "src", True, "directory"),
        ("api_main", project_root / "src" / "api" / "main.py", True, "file"),
        ("inference_pipeline", project_root / "src" / "inference" / "pipeline.py", True, "file"),
        ("explainability_gradcam", project_root / "src" / "explainability" / "gradcam.py", True, "file"),
        ("longitudinal_tracker", project_root / "src" / "longitudinal" / "tracker.py", True, "file"),
        ("volumetrics_structural", project_root / "src" / "volumetrics" / "structural.py", True, "file"),
        ("security_guardrails", project_root / "src" / "security" / "disclaimers.py", True, "file"),
        ("data_root", settings.data_root, True, "directory"),
        ("outputs_root", settings.outputs_root, True, "directory"),
    ]
    return [_check_path(name, path, required=required, expected_type=expected) for name, path, required, expected in paths]


def _config_checks(settings: AppSettings) -> list[ReadinessCheck]:
    """Check model, training, and transform config files."""

    config_root = settings.project_root / "configs"
    config_names = (
        "oasis_transforms.yaml",
        "kaggle_transforms.yaml",
        "oasis_model.yaml",
        "oasis_train.yaml",
        "longitudinal_report_schema.example.json",
        "structural_metrics_schema.example.json",
    )
    return [_check_path(f"config:{name}", config_root / name, required=True, expected_type="file") for name in config_names]


def _artifact_checks(settings: AppSettings) -> list[ReadinessCheck]:
    """Check generated artifacts that strengthen the backend but may be missing on fresh installs."""

    data_interim = settings.data_root / "interim"
    outputs_root = settings.outputs_root
    artifacts = [
        ("oasis_manifest", data_interim / "oasis1_manifest.csv"),
        ("oasis_train_manifest", data_interim / "oasis1_train_manifest.csv"),
        ("oasis_val_manifest", data_interim / "oasis1_val_manifest.csv"),
        ("oasis_test_manifest", data_interim / "oasis1_test_manifest.csv"),
        ("kaggle_manifest", data_interim / "kaggle_alz_manifest.csv"),
        ("dataset_inventory_report", outputs_root / "reports" / "dataset_inventory.json"),
        ("sample_prediction_dir", outputs_root / "predictions"),
        ("sample_explanation_dir", outputs_root / "visualizations" / "explanations"),
        ("run_outputs_dir", outputs_root / "runs" / "oasis"),
    ]
    checks = [_check_path(name, path, required=False) for name, path in artifacts]
    best_model_paths = list((outputs_root / "runs" / "oasis").glob("*/checkpoints/best_model.pt"))
    checks.append(
        ReadinessCheck(
            name="trained_checkpoint",
            status="pass" if best_model_paths else "warn",
            message="At least one best_model.pt checkpoint was found." if best_model_paths else "No trained checkpoint was found yet.",
            details={"count": len(best_model_paths), "examples": [str(path) for path in best_model_paths[:5]]},
        )
    )
    return checks


def _package_checks() -> list[ReadinessCheck]:
    """Check core package availability."""

    return [
        _check_import("package:fastapi", "fastapi"),
        _check_import("package:pydantic", "pydantic"),
        _check_import("package:nibabel", "nibabel"),
        _check_import("package:monai", "monai"),
        _check_import("package:torch", "torch"),
        _check_import("package:pandas", "pandas"),
        _check_import("package:sklearn", "sklearn"),
        _check_import("package:matplotlib", "matplotlib"),
    ]


def _overall_status(checks: list[ReadinessCheck]) -> str:
    """Compute the report status."""

    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _recommendations(checks: list[ReadinessCheck]) -> list[str]:
    """Create short remediation guidance."""

    recommendations: list[str] = []
    if any(check.status == "fail" for check in checks):
        recommendations.append("Fix failed checks before relying on the backend for experiments or API demos.")
    if any(check.name == "trained_checkpoint" and check.status == "warn" for check in checks):
        recommendations.append("Run a real OASIS training experiment before interpreting model predictions.")
    if any(check.name == "kaggle_manifest" and check.status == "warn" for check in checks):
        recommendations.append("Build the Kaggle manifest only if you plan to use the separate Kaggle workflow.")
    if not recommendations:
        recommendations.append("Backend readiness checks passed without required remediation.")
    recommendations.append("This readiness check is a development aid, not a production compliance audit.")
    return recommendations


def build_backend_readiness_report(settings: AppSettings | None = None) -> BackendReadinessReport:
    """Build the full backend readiness report."""

    resolved_settings = settings or get_app_settings()
    checks = [
        *_required_project_checks(resolved_settings),
        *_config_checks(resolved_settings),
        *_artifact_checks(resolved_settings),
        *_package_checks(),
        _check_api_importable(),
        _check_disclaimer(),
    ]
    return BackendReadinessReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        overall_status=_overall_status(checks),
        checks=checks,
        recommendations=_recommendations(checks),
    )


def save_backend_readiness_report(report: BackendReadinessReport, settings: AppSettings | None = None) -> tuple[Path, Path]:
    """Save JSON and Markdown readiness reports."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "readiness")
    json_path = output_root / "backend_readiness.json"
    md_path = output_root / "backend_readiness.md"
    payload = report.to_payload()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Backend Readiness Report",
        "",
        f"- overall_status: {report.overall_status}",
        f"- generated_at: {report.generated_at}",
        f"- pass: {payload['summary'].get('pass', 0)}",
        f"- warn: {payload['summary'].get('warn', 0)}",
        f"- fail: {payload['summary'].get('fail', 0)}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {check.status.upper()}: {check.name} - {check.message}" for check in report.checks)
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {recommendation}" for recommendation in report.recommendations)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Check structural MRI backend readiness.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on warnings as well as failures.")
    parser.add_argument("--no-save", action="store_true", help="Print report summary without saving artifacts.")
    return parser


def main() -> None:
    """Run the readiness check."""

    args = build_parser().parse_args()
    report = build_backend_readiness_report()
    if not args.no_save:
        json_path, md_path = save_backend_readiness_report(report)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")
    print(f"overall_status={report.overall_status}")
    print(f"check_count={len(report.checks)}")
    print(f"recommendations={report.recommendations}")
    if report.overall_status == "fail" or (args.strict and report.overall_status != "pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
