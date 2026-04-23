"""Build the local OASIS demo bundle and batch inference pack together."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_predict_oasis_scans import build_batch_oasis_predictions  # noqa: E402
from scripts.build_oasis_demo_bundle import build_oasis_demo_bundle  # noqa: E402
from src.configs.runtime import AppSettings, get_app_settings  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402


def _safe_name(value: str) -> str:
    """Return a path-safe workflow name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _resolve_demo_scan_path(scan_path: Path | None, scan_root: Path | None, pattern: str) -> Path | None:
    """Resolve a scan for the demo phase from explicit input or the batch folder."""

    if scan_path is not None:
        return scan_path.expanduser().resolve()
    if scan_root is None:
        return None
    candidates = sorted(path.resolve() for path in scan_root.expanduser().resolve().rglob(pattern) if path.is_file())
    return candidates[0] if candidates else None


def build_oasis_local_workflow(
    *,
    settings: AppSettings | None = None,
    scan_root: Path | None = None,
    demo_scan_path: Path | None = None,
    registry_path: Path | None = None,
    output_name: str = "oasis_local_workflow",
    device: str = "cpu",
    pattern: str = "*.hdr",
    skip_explanation: bool = False,
) -> dict[str, Any]:
    """Run the local OASIS demo bundle plus optional batch inference in one workflow."""

    resolved_settings = settings or get_app_settings()
    safe_output_name = _safe_name(output_name)
    workflow_root = ensure_directory(resolved_settings.outputs_root / "reports" / "workflows" / safe_output_name)
    resolved_demo_scan_path = _resolve_demo_scan_path(demo_scan_path, scan_root, pattern)

    demo_summary = build_oasis_demo_bundle(
        settings=resolved_settings,
        scan_path=resolved_demo_scan_path,
        registry_path=registry_path,
        device=device,
        output_name=f"{safe_output_name}_demo",
        skip_explanation=skip_explanation,
    )

    batch_summary: dict[str, Any] | None = None
    if scan_root is not None:
        batch_summary = build_batch_oasis_predictions(
            settings=resolved_settings,
            scan_root=scan_root,
            registry_path=registry_path,
            output_name=f"{safe_output_name}_batch",
            device=device,
            pattern=pattern,
        )

    summary = {
        "output_name": safe_output_name,
        "workflow_root": str(workflow_root),
        "demo_bundle_root": demo_summary["bundle_root"],
        "demo_scan_path": demo_summary["sample_scan_path"],
        "batch_enabled": batch_summary is not None,
        "batch_report_root": None if batch_summary is None else batch_summary["report_root"],
        "batch_predictions_csv": None if batch_summary is None else batch_summary["batch_predictions_csv"],
        "run_name": demo_summary["requested_run_name"],
        "registry_path": str(registry_path.expanduser().resolve()) if registry_path is not None else None,
        "notes": [
            "This local workflow keeps OASIS-only productization work moving without any Colab or Drive dependency.",
            "The demo bundle exercises the API-facing path while the batch report gives a folder-level inference snapshot.",
        ],
    }
    summary_json_path = workflow_root / "workflow_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# OASIS Local Workflow",
        "",
        f"- output_name: {safe_output_name}",
        f"- run_name: {summary['run_name']}",
        f"- demo_bundle_root: {summary['demo_bundle_root']}",
        f"- demo_scan_path: {summary['demo_scan_path']}",
        f"- batch_enabled: {summary['batch_enabled']}",
        f"- batch_report_root: {summary['batch_report_root']}",
        f"- batch_predictions_csv: {summary['batch_predictions_csv']}",
        "",
        "## Notes",
        "",
    ]
    md_lines.extend(f"- {note}" for note in summary["notes"])
    summary_md_path = workflow_root / "workflow_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_md_path"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build the local OASIS demo bundle and batch inference pack together.")
    parser.add_argument("--scan-root", type=Path, default=None)
    parser.add_argument("--demo-scan-path", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--output-name", type=str, default="oasis_local_workflow")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pattern", type=str, default="*.hdr")
    parser.add_argument("--skip-explanation", action="store_true")
    return parser


def main() -> None:
    """Run the combined local OASIS workflow and print a compact summary."""

    args = build_parser().parse_args()
    summary = build_oasis_local_workflow(
        scan_root=args.scan_root,
        demo_scan_path=args.demo_scan_path,
        registry_path=args.registry_path,
        output_name=args.output_name,
        device=args.device,
        pattern=args.pattern,
        skip_explanation=args.skip_explanation,
    )
    print(f"workflow_root={summary['workflow_root']}")
    print(f"demo_bundle_root={summary['demo_bundle_root']}")
    print(f"demo_scan_path={summary['demo_scan_path']}")
    print(f"batch_enabled={summary['batch_enabled']}")
    if summary["batch_report_root"] is not None:
        print(f"batch_report_root={summary['batch_report_root']}")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
