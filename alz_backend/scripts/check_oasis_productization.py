"""Check whether the OASIS productization path is aligned across local and synced artifacts."""

from __future__ import annotations

import argparse
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
from src.evaluation.evidence_reporting import resolve_scope_evidence_paths  # noqa: E402
from src.models.registry import load_current_oasis_model_entry  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402

DEFAULT_EXPECTED_RUN_NAME = "oasis_colab_full_v3_auroc_monitor"


@dataclass(slots=True, frozen=True)
class ProductizationCheck:
    """One alignment/productization check result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OASISProductizationReport:
    """Structured productization/alignment report."""

    generated_at: str
    expected_run_name: str | None
    overall_status: str
    checks: list[ProductizationCheck]
    recommendations: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the report to JSON-safe data."""

        summary = {"pass": 0, "warn": 0, "fail": 0}
        for check in self.checks:
            summary[check.status] = summary.get(check.status, 0) + 1
        return {
            "generated_at": self.generated_at,
            "expected_run_name": self.expected_run_name,
            "overall_status": self.overall_status,
            "summary": summary,
            "checks": [asdict(check) for check in self.checks],
            "recommendations": list(self.recommendations),
        }


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _overall_status(checks: list[ProductizationCheck]) -> str:
    """Compute the overall report status."""

    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _check_local_active_registry(
    settings: AppSettings,
    *,
    expected_run_name: str | None,
) -> tuple[list[ProductizationCheck], dict[str, Any] | None]:
    """Validate the active local registry and its filesystem targets."""

    checks: list[ProductizationCheck] = []
    registry_path = settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    if not registry_path.exists():
        checks.append(
            ProductizationCheck(
                name="local_active_registry",
                status="fail",
                message="Active local OASIS registry is missing.",
                details={"registry_path": str(registry_path)},
            )
        )
        return checks, None

    try:
        entry = load_current_oasis_model_entry(path=registry_path, settings=settings)
    except Exception as exc:  # pragma: no cover - diagnostic path.
        checks.append(
            ProductizationCheck(
                name="local_active_registry",
                status="fail",
                message="Active local OASIS registry could not be loaded.",
                details={"registry_path": str(registry_path), "error": repr(exc)},
            )
        )
        return checks, None

    checkpoint_path = Path(entry.checkpoint_path)
    checks.append(
        ProductizationCheck(
            name="local_active_registry",
            status="pass",
            message="Active local OASIS registry is loadable.",
            details={
                "registry_path": str(registry_path),
                "run_name": entry.run_name,
                "recommended_threshold": entry.recommended_threshold,
            },
        )
    )
    checks.append(
        ProductizationCheck(
            name="local_active_checkpoint",
            status="pass" if checkpoint_path.exists() else "fail",
            message="Active local checkpoint exists." if checkpoint_path.exists() else "Active local checkpoint is missing.",
            details={"checkpoint_path": str(checkpoint_path)},
        )
    )

    model_config_path = Path(entry.model_config_path) if entry.model_config_path else None
    preprocessing_config_path = Path(entry.preprocessing_config_path) if entry.preprocessing_config_path else None
    config_ok = bool(model_config_path and model_config_path.exists() and preprocessing_config_path and preprocessing_config_path.exists())
    checks.append(
        ProductizationCheck(
            name="local_active_configs",
            status="pass" if config_ok else "fail",
            message="Active local model/config paths resolve cleanly." if config_ok else "Active local model/config path is missing.",
            details={
                "model_config_path": None if model_config_path is None else str(model_config_path),
                "preprocessing_config_path": None if preprocessing_config_path is None else str(preprocessing_config_path),
            },
        )
    )

    expected_status = "pass" if not expected_run_name or entry.run_name == expected_run_name else "fail"
    checks.append(
        ProductizationCheck(
            name="local_expected_run",
            status=expected_status,
            message=(
                "Active local run matches the expected canonical run."
                if expected_status == "pass"
                else "Active local run does not match the expected canonical run."
            ),
            details={
                "expected_run_name": expected_run_name,
                "active_run_name": entry.run_name,
            },
        )
    )

    return checks, entry.to_dict()


def _check_scope_evidence_alignment(
    settings: AppSettings,
    *,
    active_entry_payload: dict[str, Any] | None,
) -> list[ProductizationCheck]:
    """Validate that the saved scope evidence report agrees with the active registry."""

    checks: list[ProductizationCheck] = []
    evidence_path = settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.json"
    if not evidence_path.exists():
        checks.append(
            ProductizationCheck(
                name="scope_evidence_report",
                status="warn",
                message="Scope-aligned evidence report is missing.",
                details={"evidence_path": str(evidence_path)},
            )
        )
        return checks

    try:
        evidence_payload = _load_json(evidence_path)
    except Exception as exc:  # pragma: no cover - diagnostic path.
        checks.append(
            ProductizationCheck(
                name="scope_evidence_report",
                status="fail",
                message="Scope-aligned evidence report could not be loaded.",
                details={"evidence_path": str(evidence_path), "error": repr(exc)},
            )
        )
        return checks

    checks.append(
        ProductizationCheck(
            name="scope_evidence_report",
            status="pass",
            message="Scope-aligned evidence report exists.",
            details={"evidence_path": str(evidence_path)},
        )
    )
    if active_entry_payload is None:
        return checks

    evidence_oasis = dict(evidence_payload.get("oasis_primary", {}))
    same_run = evidence_oasis.get("run_name") == active_entry_payload.get("run_name")
    same_threshold = evidence_oasis.get("recommended_threshold") == active_entry_payload.get("recommended_threshold")
    checks.append(
        ProductizationCheck(
            name="scope_evidence_alignment",
            status="pass" if same_run and same_threshold else "fail",
            message=(
                "Scope evidence matches the active registry baseline."
                if same_run and same_threshold
                else "Scope evidence is drifting from the active registry baseline."
            ),
            details={
                "evidence_run_name": evidence_oasis.get("run_name"),
                "active_run_name": active_entry_payload.get("run_name"),
                "evidence_recommended_threshold": evidence_oasis.get("recommended_threshold"),
                "active_recommended_threshold": active_entry_payload.get("recommended_threshold"),
            },
        )
    )
    return checks


def _check_repeated_split_support(settings: AppSettings) -> list[ProductizationCheck]:
    """Verify repeated-split OASIS support remains visible."""

    checks: list[ProductizationCheck] = []
    paths = resolve_scope_evidence_paths(settings)
    study_path = paths.oasis_repeated_split_study_path
    checks.append(
        ProductizationCheck(
            name="oasis_repeated_split_support",
            status="pass" if study_path is not None and study_path.exists() else "warn",
            message=(
                "Repeated-split OASIS study summary is available."
                if study_path is not None and study_path.exists()
                else "Repeated-split OASIS study summary is not available."
            ),
            details={"study_path": None if study_path is None else str(study_path)},
        )
    )
    return checks


def _check_synced_runtime_alignment(
    *,
    source_runtime_root: Path | None,
    expected_run_name: str | None,
    active_entry_payload: dict[str, Any] | None,
) -> list[ProductizationCheck]:
    """Compare the synced backend_runtime payload with the local active baseline."""

    checks: list[ProductizationCheck] = []
    if source_runtime_root is None:
        return checks

    runtime_root = source_runtime_root.expanduser().resolve()
    runtime_registry_path = runtime_root / "outputs" / "model_registry" / "oasis_current_baseline.json"
    if not runtime_registry_path.exists():
        checks.append(
            ProductizationCheck(
                name="synced_runtime_registry",
                status="fail",
                message="Synced backend_runtime registry is missing.",
                details={"runtime_registry_path": str(runtime_registry_path)},
            )
        )
        return checks

    runtime_registry = _load_json(runtime_registry_path)
    runtime_run_name = runtime_registry.get("run_name")
    checks.append(
        ProductizationCheck(
            name="synced_runtime_registry",
            status="pass",
            message="Synced backend_runtime registry exists.",
            details={
                "runtime_registry_path": str(runtime_registry_path),
                "run_name": runtime_run_name,
                "recommended_threshold": runtime_registry.get("recommended_threshold"),
            },
        )
    )

    runtime_run_root = runtime_root / "outputs" / "runs" / "oasis" / str(runtime_run_name)
    checks.append(
        ProductizationCheck(
            name="synced_runtime_run",
            status="pass" if runtime_run_root.exists() else "fail",
            message="Synced backend_runtime run root exists." if runtime_run_root.exists() else "Synced backend_runtime run root is missing.",
            details={"runtime_run_root": str(runtime_run_root)},
        )
    )

    if expected_run_name:
        checks.append(
            ProductizationCheck(
                name="synced_runtime_expected_run",
                status="pass" if runtime_run_name == expected_run_name else "fail",
                message=(
                    "Synced backend_runtime run matches the expected canonical run."
                    if runtime_run_name == expected_run_name
                    else "Synced backend_runtime run does not match the expected canonical run."
                ),
                details={"expected_run_name": expected_run_name, "runtime_run_name": runtime_run_name},
            )
        )

    if active_entry_payload is not None:
        local_run_name = active_entry_payload.get("run_name")
        local_threshold = active_entry_payload.get("recommended_threshold")
        runtime_threshold = runtime_registry.get("recommended_threshold")
        same_local = runtime_run_name == local_run_name and runtime_threshold == local_threshold
        checks.append(
            ProductizationCheck(
                name="synced_runtime_local_alignment",
                status="pass" if same_local else "fail",
                message=(
                    "Synced backend_runtime baseline matches the active local registry."
                    if same_local
                    else "Synced backend_runtime baseline differs from the active local registry."
                ),
                details={
                    "runtime_run_name": runtime_run_name,
                    "local_run_name": local_run_name,
                    "runtime_recommended_threshold": runtime_threshold,
                    "local_recommended_threshold": local_threshold,
                },
            )
        )
    return checks


def _recommendations(checks: list[ProductizationCheck]) -> list[str]:
    """Build brief remediation guidance."""

    recommendations: list[str] = []
    if any(check.name == "local_expected_run" and check.status == "fail" for check in checks):
        recommendations.append(
            "Import the canonical promoted Colab run locally before relying on demos or reports."
        )
    if any(check.name == "scope_evidence_alignment" and check.status == "fail" for check in checks):
        recommendations.append(
            "Regenerate the scope evidence report so it matches the active local registry baseline."
        )
    if any(check.name == "synced_runtime_local_alignment" and check.status == "fail" for check in checks):
        recommendations.append(
            "Re-import the synced backend_runtime baseline or update the local active registry so cloud and local stay aligned."
        )
    if any(check.name == "oasis_repeated_split_support" and check.status == "warn" for check in checks):
        recommendations.append(
            "Keep the repeated-split OASIS study visible in outputs/model_selection before making stronger model claims."
        )
    if not recommendations:
        recommendations.append("OASIS productization checks passed without required remediation.")
    recommendations.append("Treat backend_runtime as the cloud source of truth for promoted Colab OASIS artifacts.")
    return recommendations


def build_oasis_productization_report(
    settings: AppSettings | None = None,
    *,
    expected_run_name: str | None = DEFAULT_EXPECTED_RUN_NAME,
    source_runtime_root: Path | None = None,
) -> OASISProductizationReport:
    """Build the OASIS productization/alignment report."""

    resolved_settings = settings or get_app_settings()
    checks: list[ProductizationCheck] = []
    local_checks, active_entry_payload = _check_local_active_registry(
        resolved_settings,
        expected_run_name=expected_run_name,
    )
    checks.extend(local_checks)
    checks.extend(_check_scope_evidence_alignment(resolved_settings, active_entry_payload=active_entry_payload))
    checks.extend(_check_repeated_split_support(resolved_settings))
    checks.extend(
        _check_synced_runtime_alignment(
            source_runtime_root=source_runtime_root,
            expected_run_name=expected_run_name,
            active_entry_payload=active_entry_payload,
        )
    )
    return OASISProductizationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        expected_run_name=expected_run_name,
        overall_status=_overall_status(checks),
        checks=checks,
        recommendations=_recommendations(checks),
    )


def save_oasis_productization_report(
    report: OASISProductizationReport,
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis_productization_status",
) -> tuple[Path, Path]:
    """Save JSON and Markdown productization reports."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "productization")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    payload = report.to_payload()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# OASIS Productization Status",
        "",
        f"- overall_status: {report.overall_status}",
        f"- generated_at: {report.generated_at}",
        f"- expected_run_name: {report.expected_run_name}",
        f"- pass: {payload['summary'].get('pass', 0)}",
        f"- warn: {payload['summary'].get('warn', 0)}",
        f"- fail: {payload['summary'].get('fail', 0)}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {check.status.upper()}: {check.name} - {check.message}" for check in report.checks)
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report.recommendations)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Check OASIS productization/alignment across local and synced artifacts.")
    parser.add_argument("--expected-run-name", type=str, default=DEFAULT_EXPECTED_RUN_NAME)
    parser.add_argument(
        "--source-runtime-root",
        type=Path,
        default=None,
        help="Optional local Google Drive sync path to Cerebrasensecloud/backend_runtime.",
    )
    parser.add_argument("--no-save", action="store_true", help="Print a summary without writing report files.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on warnings as well as failures.")
    return parser


def main() -> None:
    """Run the productization check."""

    args = build_parser().parse_args()
    report = build_oasis_productization_report(
        expected_run_name=args.expected_run_name or None,
        source_runtime_root=args.source_runtime_root,
    )
    if not args.no_save:
        json_path, md_path = save_oasis_productization_report(report)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")
    print(f"overall_status={report.overall_status}")
    print(f"expected_run_name={report.expected_run_name}")
    print(f"check_count={len(report.checks)}")
    print(f"recommendations={report.recommendations}")
    if report.overall_status == "fail" or (args.strict and report.overall_status != "pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
