"""Build a single current-project-status artifact bundle for OASIS-first reviews and demos."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import AppSettings, get_app_settings  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _safe_name(value: str) -> str:
    """Return a path-safe artifact bundle name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _copy_file(source_path: Path, destination_path: Path) -> None:
    """Copy one file and create parent directories if needed."""

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _copy_directory(source_path: Path, destination_path: Path) -> None:
    """Copy one directory tree, replacing any prior copy."""

    if destination_path.exists():
        shutil.rmtree(destination_path)
    shutil.copytree(source_path, destination_path)


@dataclass(slots=True)
class ProjectStatusBundle:
    """Structured result for one current project status bundle."""

    generated_at: str
    output_name: str
    bundle_root: str
    recommendation: str
    headline: str
    active_run_name: str
    candidate_run_name: str | None
    included_artifacts: dict[str, str]
    notes: list[str]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return asdict(self)


def build_project_status_bundle(
    *,
    settings: AppSettings | None = None,
    output_name: str = "current_project_status",
) -> ProjectStatusBundle:
    """Build the full current project status bundle from existing reports and docs."""

    resolved_settings = settings or get_app_settings()
    safe_output_name = _safe_name(output_name)
    bundle_root = ensure_directory(resolved_settings.outputs_root / "reports" / "status" / safe_output_name)

    comparison_json = resolved_settings.outputs_root / "reports" / "comparison" / "oasis_baseline_comparison.json"
    comparison_md = resolved_settings.outputs_root / "reports" / "comparison" / "oasis_baseline_comparison.md"
    presentation_json = resolved_settings.outputs_root / "reports" / "presentation" / "oasis_presentation_summary.json"
    presentation_md = resolved_settings.outputs_root / "reports" / "presentation" / "oasis_presentation_summary.md"
    evidence_json = resolved_settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.json"
    evidence_md = resolved_settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.md"
    readiness_json = resolved_settings.outputs_root / "reports" / "readiness" / "backend_readiness.json"
    readiness_md = resolved_settings.outputs_root / "reports" / "readiness" / "backend_readiness.md"
    productization_json = resolved_settings.outputs_root / "reports" / "productization" / "oasis_productization_status.json"
    productization_md = resolved_settings.outputs_root / "reports" / "productization" / "oasis_productization_status.md"
    active_registry = resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json"
    candidate_registry = resolved_settings.outputs_root / "model_registry" / "oasis_candidate_v3.json"

    backbone_doc = resolved_settings.project_root / "docs" / "PROJECT_BACKBONE.md"
    scope_doc = resolved_settings.project_root / "docs" / "project_scope.md"
    productization_doc = resolved_settings.project_root / "docs" / "oasis_productization_workflow.md"
    github_drive_doc = resolved_settings.project_root / "docs" / "github_drive_workflow.md"

    comparison_payload = _load_json(comparison_json)
    presentation_payload = _load_json(presentation_json)

    active_bundle_root = Path(str(dict(comparison_payload.get("demo_bundles", {})).get("active", {}).get("bundle_root", "")))
    candidate_bundle_root = Path(str(dict(comparison_payload.get("demo_bundles", {})).get("candidate", {}).get("bundle_root", "")))

    included_artifacts: dict[str, str] = {}

    files_to_copy = {
        "comparison_json": comparison_json,
        "comparison_md": comparison_md,
        "presentation_json": presentation_json,
        "presentation_md": presentation_md,
        "evidence_json": evidence_json,
        "evidence_md": evidence_md,
        "readiness_json": readiness_json,
        "readiness_md": readiness_md,
        "productization_json": productization_json,
        "productization_md": productization_md,
        "active_registry": active_registry,
        "candidate_registry": candidate_registry,
        "project_backbone_doc": backbone_doc,
        "project_scope_doc": scope_doc,
        "oasis_productization_doc": productization_doc,
        "github_drive_workflow_doc": github_drive_doc,
    }
    for key, source_path in files_to_copy.items():
        if source_path.exists():
            destination_path = bundle_root / "files" / source_path.name
            _copy_file(source_path, destination_path)
            included_artifacts[key] = str(destination_path)

    if active_bundle_root.exists():
        destination_path = bundle_root / "demo" / "active_bundle"
        _copy_directory(active_bundle_root, destination_path)
        included_artifacts["active_demo_bundle"] = str(destination_path)
    if candidate_bundle_root.exists():
        destination_path = bundle_root / "demo" / "candidate_bundle"
        _copy_directory(candidate_bundle_root, destination_path)
        included_artifacts["candidate_demo_bundle"] = str(destination_path)

    summary = ProjectStatusBundle(
        generated_at=datetime.now(timezone.utc).isoformat(),
        output_name=safe_output_name,
        bundle_root=str(bundle_root),
        recommendation=str(presentation_payload.get("recommendation", "unknown")),
        headline=str(presentation_payload.get("headline", "")),
        active_run_name=str(presentation_payload.get("project_state", {}).get("active_run_name", "")),
        candidate_run_name=(
            None
            if not presentation_payload.get("project_state", {}).get("candidate_run_name")
            else str(presentation_payload.get("project_state", {}).get("candidate_run_name"))
        ),
        included_artifacts=included_artifacts,
        notes=[
            "This bundle is the current single-folder project review pack for OASIS-first status updates.",
            "It combines the active-vs-candidate comparison, presentation summary, evidence, readiness, and productization outputs.",
            "Active and candidate demo bundles are copied side by side when they exist.",
        ],
    )

    summary_json_path = bundle_root / "project_status_bundle.json"
    summary_md_path = bundle_root / "project_status_bundle.md"
    summary_json_path.write_text(json.dumps(summary.to_payload(), indent=2), encoding="utf-8")

    lines = [
        "# Current Project Status Bundle",
        "",
        f"- generated_at: {summary.generated_at}",
        f"- recommendation: {summary.recommendation}",
        f"- active_run_name: {summary.active_run_name}",
        f"- candidate_run_name: {summary.candidate_run_name}",
        "",
        "## Headline",
        "",
        summary.headline,
        "",
        "## Included Artifacts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary.included_artifacts.items())
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {item}" for item in summary.notes)
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    included_artifacts["bundle_summary_json"] = str(summary_json_path)
    included_artifacts["bundle_summary_md"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build one current project status artifact bundle from the latest OASIS productization outputs.")
    parser.add_argument("--output-name", type=str, default="current_project_status")
    return parser


def main() -> None:
    """Build the bundle and print a compact summary."""

    args = build_parser().parse_args()
    result = build_project_status_bundle(output_name=args.output_name)
    print(f"bundle_root={result.bundle_root}")
    print(f"recommendation={result.recommendation}")
    print(f"active_run_name={result.active_run_name}")
    print(f"candidate_run_name={result.candidate_run_name}")
    print("summary=" + json.dumps(result.to_payload(), indent=2))


if __name__ == "__main__":
    main()
