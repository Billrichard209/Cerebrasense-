"""Build a single OASIS-2 onboarding bundle from local readiness artifacts."""

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
from src.data.oasis2 import build_oasis2_raw_inventory, build_oasis2_session_manifest  # noqa: E402
from src.data.oasis2_readiness import build_oasis2_readiness_report, save_oasis2_readiness_report  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _safe_name(value: str) -> str:
    """Return a path-safe bundle name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _copy_file(source_path: Path, destination_path: Path) -> None:
    """Copy one file and create parent directories if needed."""

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


@dataclass(slots=True)
class OASIS2OnboardingBundle:
    """Structured result for one OASIS-2 onboarding bundle."""

    generated_at: str
    output_name: str
    bundle_root: str
    source_root: str
    readiness_status: str
    unique_subject_count: int
    unique_session_count: int
    longitudinal_subject_count: int
    upload_to_drive_now: bool
    included_artifacts: dict[str, str]
    notes: list[str]
    next_steps: list[str]
    drive_guidance: str

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return asdict(self)


def build_oasis2_onboarding_bundle(
    *,
    settings: AppSettings | None = None,
    source_root: Path | None = None,
    output_name: str = "current_oasis2_onboarding",
    include_longitudinal_focus: bool = True,
) -> OASIS2OnboardingBundle:
    """Build the current OASIS-2 onboarding bundle from local data only."""

    resolved_settings = settings or get_app_settings()
    safe_output_name = _safe_name(output_name)
    bundle_root = ensure_directory(resolved_settings.outputs_root / "reports" / "onboarding" / safe_output_name)

    readiness_report = build_oasis2_readiness_report(resolved_settings, source_root=source_root)
    readiness_json, readiness_md = save_oasis2_readiness_report(
        readiness_report,
        resolved_settings,
        file_stem="oasis2_readiness",
    )
    inventory_result = build_oasis2_raw_inventory(resolved_settings, source_root=source_root)
    manifest_result = build_oasis2_session_manifest(
        resolved_settings,
        source_root=source_root,
        inventory_path=inventory_result.inventory_path,
    )

    readiness_payload = _load_json(readiness_json)
    inventory_summary = _load_json(inventory_result.summary_path)
    manifest_summary = _load_json(manifest_result.summary_path)

    included_artifacts: dict[str, str] = {}
    files_to_copy: dict[str, tuple[str, Path]] = {
        "oasis2_readiness_json": ("generated", readiness_json),
        "oasis2_readiness_md": ("generated", readiness_md),
        "oasis2_raw_inventory_csv": ("generated", inventory_result.inventory_path),
        "oasis2_raw_inventory_dropped_rows_csv": ("generated", inventory_result.dropped_rows_path),
        "oasis2_raw_inventory_summary_json": ("generated", inventory_result.summary_path),
        "oasis2_session_manifest_csv": ("generated", manifest_result.manifest_path),
        "oasis2_longitudinal_records_csv": ("generated", manifest_result.longitudinal_records_path),
        "oasis2_subject_summary_csv": ("generated", manifest_result.subject_summary_path),
        "oasis2_session_manifest_summary_json": ("generated", manifest_result.summary_path),
        "oasis2_readiness_doc": ("docs", resolved_settings.project_root / "docs" / "oasis2_readiness.md"),
        "project_backbone_doc": ("docs", resolved_settings.project_root / "docs" / "PROJECT_BACKBONE.md"),
        "project_scope_doc": ("docs", resolved_settings.project_root / "docs" / "project_scope.md"),
    }

    if include_longitudinal_focus:
        longitudinal_focus_json = resolved_settings.outputs_root / "reports" / "evidence" / "oasis_longitudinal_focus_report.json"
        longitudinal_focus_md = resolved_settings.outputs_root / "reports" / "evidence" / "oasis_longitudinal_focus_report.md"
        files_to_copy["oasis_longitudinal_focus_json"] = ("generated", longitudinal_focus_json)
        files_to_copy["oasis_longitudinal_focus_md"] = ("generated", longitudinal_focus_md)

    for key, (subdirectory, source_path) in files_to_copy.items():
        if source_path.exists():
            destination_path = bundle_root / "files" / subdirectory / source_path.name
            _copy_file(source_path, destination_path)
            included_artifacts[key] = str(destination_path)

    source_root_value = str(readiness_payload.get("source_root", ""))
    drive_guidance = (
        "Do not upload OASIS-2 to Google Drive yet. Keep it local while this project phase is limited to readiness, "
        "raw inventory, unlabeled session manifests, and onboarding planning. Upload it to Drive only when we open a "
        "real Colab or remote preprocessing/training/evaluation workflow that needs OASIS-2 outside this machine."
    )
    next_steps = [
        "Keep OASIS-2 local for now and use this bundle as the onboarding review pack.",
        "Collect or map the missing visit/label metadata before any supervised OASIS-2 evaluation work.",
        "Implement a dedicated OASIS-2 manifest adapter and subject-safe split policy before any training claims.",
        "Only upload OASIS-2 to Google Drive when an actual Colab or remote runtime path is ready for it.",
    ]
    notes = [
        "This bundle is intentionally local-first and does not assume Google Drive sync.",
        "The OASIS-2 manifest here is unlabeled and should not be treated as a supervised training manifest.",
        drive_guidance,
    ]

    summary = OASIS2OnboardingBundle(
        generated_at=datetime.now(timezone.utc).isoformat(),
        output_name=safe_output_name,
        bundle_root=str(bundle_root),
        source_root=source_root_value,
        readiness_status=str(readiness_payload.get("overall_status", "warn")),
        unique_subject_count=int(manifest_summary.get("unique_subject_count", 0)),
        unique_session_count=int(manifest_summary.get("session_row_count", 0)),
        longitudinal_subject_count=int(manifest_summary.get("longitudinal_subject_count", 0)),
        upload_to_drive_now=False,
        included_artifacts=included_artifacts,
        notes=notes,
        next_steps=next_steps,
        drive_guidance=drive_guidance,
    )

    summary_json_path = bundle_root / "oasis2_onboarding_bundle.json"
    summary_md_path = bundle_root / "oasis2_onboarding_bundle.md"
    summary_json_path.write_text(json.dumps(summary.to_payload(), indent=2), encoding="utf-8")

    lines = [
        "# OASIS-2 Onboarding Bundle",
        "",
        f"- generated_at: {summary.generated_at}",
        f"- readiness_status: {summary.readiness_status}",
        f"- source_root: {summary.source_root}",
        f"- unique_subject_count: {summary.unique_subject_count}",
        f"- unique_session_count: {summary.unique_session_count}",
        f"- longitudinal_subject_count: {summary.longitudinal_subject_count}",
        f"- upload_to_drive_now: {summary.upload_to_drive_now}",
        "",
        "## Drive Guidance",
        "",
        summary.drive_guidance,
        "",
        "## Included Artifacts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary.included_artifacts.items())
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {item}" for item in summary.notes)
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in summary.next_steps)
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    included_artifacts["bundle_summary_json"] = str(summary_json_path)
    included_artifacts["bundle_summary_md"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build one local-first OASIS-2 onboarding bundle from readiness, inventory, and session-manifest artifacts."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional OASIS-2 source root override. Point this at the parent folder containing the OAS2 raw parts or a resolved OASIS-2 root.",
    )
    parser.add_argument("--output-name", type=str, default="current_oasis2_onboarding")
    parser.add_argument(
        "--skip-longitudinal-focus",
        action="store_true",
        help="Do not copy the latest OASIS longitudinal focus report into the onboarding bundle.",
    )
    return parser


def main() -> None:
    """Build the bundle and print a compact summary."""

    args = build_parser().parse_args()
    result = build_oasis2_onboarding_bundle(
        source_root=args.source_root,
        output_name=args.output_name,
        include_longitudinal_focus=not args.skip_longitudinal_focus,
    )
    print(f"bundle_root={result.bundle_root}")
    print(f"readiness_status={result.readiness_status}")
    print(f"source_root={result.source_root}")
    print(f"upload_to_drive_now={result.upload_to_drive_now}")
    print("summary=" + json.dumps(result.to_payload(), indent=2))


if __name__ == "__main__":
    main()
