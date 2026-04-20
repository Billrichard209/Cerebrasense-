"""Colab-friendly OASIS-2 bundle runner with explicit training-readiness gates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OASIS2_REPO_DEMOGRAPHICS_RELATIVE_PATH = (
    Path("data") / "metadata" / "oasis2" / "oasis2_official_demographics.csv"
)

from scripts.train_oasis2 import apply_cli_overrides, build_parser as build_train_parser  # noqa: E402
from src.configs.runtime import get_app_settings  # noqa: E402
from src.data.oasis2 import build_oasis2_session_manifest  # noqa: E402
from src.data.oasis2_metadata import (  # noqa: E402
    OASIS2_OFFICIAL_DEMOGRAPHICS_URL,
    import_oasis2_official_demographics_into_metadata_template,
    load_oasis2_metadata_template,
    merge_oasis2_metadata_template,
    save_oasis2_metadata_adapter_summary,
    save_oasis2_official_demographics_import_summary,
)
from src.data.oasis2_split_policy import build_oasis2_subject_safe_split_plan  # noqa: E402
from src.data.oasis2_supervised import (  # noqa: E402
    build_oasis2_training_readiness_report,
    save_oasis2_training_readiness_report,
)
from src.data.oasis2_upload_bundle import (  # noqa: E402
    inspect_oasis2_upload_bundle,
    save_oasis2_upload_bundle_report,
)
from src.training.oasis2_research import default_oasis2_train_config_path, run_research_oasis2_training  # noqa: E402
from src.training.oasis_research import ResearchTrainingError, load_research_oasis_training_config  # noqa: E402


@dataclass(slots=True)
class OASIS2RuntimeRefreshResult:
    """Artifacts produced when rebuilding runtime data from a bundle source root."""

    source_root: Path
    manifest_source: str
    manifest_path: Path
    longitudinal_records_path: Path
    subject_summary_path: Path
    manifest_summary_path: Path
    metadata_template_path: Path
    official_demographics_path: Path | None
    official_demographics_import_json_path: Path | None
    official_demographics_import_md_path: Path | None
    official_demographics_import_payload: dict[str, Any] | None
    metadata_adapter_json_path: Path
    metadata_adapter_md_path: Path
    split_plan_path: Path
    training_readiness_json_path: Path
    training_readiness_md_path: Path
    readiness_payload: dict[str, Any]


def _running_in_colab() -> bool:
    """Return whether the current interpreter looks like Google Colab."""

    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _mount_drive_if_requested(should_mount: bool) -> None:
    """Mount Google Drive when running inside Colab and requested."""

    if not should_mount or not _running_in_colab():
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def _resolve_training_device(requested_device: str | None) -> tuple[str, bool]:
    """Resolve a safe device and mixed-precision setting for the current runtime."""

    import torch

    normalized = "auto" if requested_device is None else str(requested_device).strip().lower()
    if normalized == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = normalized

    if resolved == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA, but this runtime does not expose a GPU. Falling back to CPU.")
        resolved = "cpu"

    return resolved, resolved == "cuda"


def _configure_runtime_roots(*, project_root: Path, runtime_root: Path | None) -> tuple[Path, Path]:
    """Resolve and export the runtime `data/` and `outputs/` roots."""

    if runtime_root is None:
        data_root = project_root / "data"
        outputs_root = project_root / "outputs"
    else:
        resolved_runtime_root = runtime_root.expanduser().resolve()
        data_root = resolved_runtime_root / "data"
        outputs_root = resolved_runtime_root / "outputs"

    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    os.environ["ALZ_DATA_ROOT"] = str(data_root)
    os.environ["ALZ_OUTPUTS_ROOT"] = str(outputs_root)
    return data_root, outputs_root


def _resolve_bundle_metadata_source(bundle_root: Path, override_metadata_path: Path | None) -> Path:
    """Resolve the metadata template source to copy into the runtime."""

    if override_metadata_path is not None:
        resolved = override_metadata_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"OASIS-2 metadata template override not found: {resolved}")
        return resolved

    candidate = bundle_root / "backend_reference" / "oasis2_metadata_template.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            "OASIS-2 bundle metadata template not found. "
            f"Expected {candidate}. Fill or upload backend_reference/oasis2_metadata_template.csv first."
        )
    return candidate


def _copy_metadata_template_to_runtime(*, source_path: Path, data_root: Path) -> Path:
    """Copy the current OASIS-2 metadata template into the runtime data/interim root."""

    destination = data_root / "interim" / "oasis2_metadata_template.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination


def _runtime_template_has_candidate_labels(*, metadata_template_path: Path, settings: Any) -> bool:
    """Return whether the runtime metadata template already contains session labels."""

    frame = load_oasis2_metadata_template(settings=settings, metadata_path=metadata_template_path)
    return bool((frame["diagnosis_label"].notna() & frame["diagnosis_label_name"].notna()).any())


def _download_official_demographics(*, outputs_root: Path, demographics_url: str) -> Path:
    """Download the official OASIS-2 demographics sheet into the runtime imports folder."""

    imports_root = outputs_root / "imports"
    imports_root.mkdir(parents=True, exist_ok=True)
    destination = imports_root / "oasis2_official_demographics.xlsx"
    if destination.exists():
        return destination
    urllib.request.urlretrieve(demographics_url, destination)
    return destination


def _resolve_official_demographics_source(
    *,
    project_root: Path,
    bundle_root: Path,
    outputs_root: Path,
    override_path: Path | None,
    demographics_url: str,
) -> Path:
    """Resolve the best available OASIS-2 official demographics source.

    Preference order:
    1. explicit override path
    2. bundled backend_reference copy
    3. repo-bundled reference CSV
    4. cached runtime import
    5. public website download
    """

    if override_path is not None:
        resolved = override_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"OASIS-2 demographics override not found: {resolved}")
        return resolved

    candidates = [
        bundle_root / "backend_reference" / "oasis2_official_demographics.csv",
        bundle_root / "backend_reference" / "oasis2_demographics.csv",
        bundle_root / "backend_reference" / "oasis2_official_demographics.xlsx",
        bundle_root / "backend_reference" / "oasis2_demographics.xlsx",
        project_root / OASIS2_REPO_DEMOGRAPHICS_RELATIVE_PATH,
        outputs_root / "imports" / "oasis2_official_demographics.csv",
        outputs_root / "imports" / "oasis2_official_demographics.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return _download_official_demographics(
        outputs_root=outputs_root,
        demographics_url=demographics_url,
    )


def _stage_bundle_to_local(*, bundle_root: Path, stage_root: Path, force_restage: bool) -> Path:
    """Copy the uploaded OASIS-2 bundle into local runtime storage for faster training I/O."""

    resolved_bundle_root = bundle_root.expanduser().resolve()
    resolved_stage_root = stage_root.expanduser().resolve()
    if resolved_stage_root.exists() and force_restage:
        shutil.rmtree(resolved_stage_root)
    if resolved_stage_root.exists():
        print(f"Reusing staged OASIS-2 bundle: {resolved_stage_root}")
        return resolved_stage_root

    resolved_stage_root.parent.mkdir(parents=True, exist_ok=True)
    print(f"Staging OASIS-2 bundle to local disk: {resolved_bundle_root} -> {resolved_stage_root}")
    shutil.copytree(resolved_bundle_root, resolved_stage_root)
    return resolved_stage_root


def _rewrite_bundle_relative_meta(meta_payload: dict[str, Any], *, bundle_root: Path, image_path: Path) -> dict[str, Any]:
    """Rewrite portable bundle metadata so the runtime points at the current bundle root."""

    rewritten = dict(meta_payload)
    paired_image_value = rewritten.get("paired_image")
    if isinstance(paired_image_value, str) and paired_image_value.strip():
        paired_candidate = Path(paired_image_value)
        if not paired_candidate.is_absolute():
            rewritten["paired_image"] = str((bundle_root / paired_candidate).resolve())
    rewritten["source_root"] = str(bundle_root)
    rewritten["session_dir"] = str(image_path.parent.parent)
    rewritten["raw_dir"] = str(image_path.parent)
    return rewritten


def _materialize_runtime_manifest_from_bundle_reference(
    *,
    bundle_root: Path,
    data_root: Path,
) -> tuple[Path, Path, Path, Path]:
    """Build runtime manifest artifacts from the bundle's backend_reference manifest."""

    backend_reference_root = bundle_root / "backend_reference"
    relative_manifest_path = backend_reference_root / "oasis2_session_manifest_relative.csv"
    if not relative_manifest_path.exists():
        raise FileNotFoundError(f"Bundle reference manifest not found: {relative_manifest_path}")

    destination_root = data_root / "interim"
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_root / "oasis2_session_manifest.csv"
    longitudinal_records_path = destination_root / "oasis2_longitudinal_records.csv"
    subject_summary_path = destination_root / "oasis2_subject_summary.csv"
    summary_path = destination_root / "oasis2_session_manifest_summary.json"

    relative_manifest = pd.read_csv(relative_manifest_path)
    rewritten_rows: list[dict[str, Any]] = []
    for row in relative_manifest.to_dict(orient="records"):
        relative_image = str(row.get("image", "")).strip()
        if not relative_image:
            raise ValueError("OASIS-2 bundle reference manifest contains an empty image path.")
        image_path = (bundle_root / relative_image).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"OASIS-2 bundle reference image is missing: {image_path}")

        meta_payload = json.loads(row.get("meta") or "{}")
        rewritten_meta = _rewrite_bundle_relative_meta(meta_payload, bundle_root=bundle_root, image_path=image_path)
        rewritten_row = dict(row)
        rewritten_row["image"] = str(image_path)
        rewritten_row["meta"] = json.dumps(rewritten_meta, ensure_ascii=True, sort_keys=True)
        rewritten_rows.append(rewritten_row)

    manifest_frame = pd.DataFrame(rewritten_rows).sort_values(
        by=["subject_id", "visit_number", "session_id"],
        kind="stable",
    )
    manifest_frame.to_csv(manifest_path, index=False)

    longitudinal_records = manifest_frame.rename(columns={"image": "source_path", "visit_number": "visit_order"})
    longitudinal_records.insert(0, "record_type", "oasis2_session")
    longitudinal_records.to_csv(longitudinal_records_path, index=False)

    subject_summary = (
        manifest_frame.groupby("subject_id", sort=True)
        .agg(
            session_count=("session_id", "nunique"),
            first_visit=("visit_number", "min"),
            last_visit=("visit_number", "max"),
        )
        .reset_index()
    )
    session_lists = (
        manifest_frame.groupby("subject_id", sort=True)["session_id"]
        .apply(lambda values: "|".join(str(value) for value in values))
        .reset_index(name="session_ids")
    )
    subject_summary = subject_summary.merge(session_lists, on="subject_id", how="left")
    subject_summary.to_csv(subject_summary_path, index=False)

    summary_payload = {
        "source_root": str(bundle_root),
        "manifest_reference_path": str(relative_manifest_path),
        "manifest_path": str(manifest_path),
        "longitudinal_records_path": str(longitudinal_records_path),
        "subject_summary_path": str(subject_summary_path),
        "session_row_count": int(len(manifest_frame)),
        "unique_subject_count": int(manifest_frame["subject_id"].nunique()),
        "longitudinal_subject_count": int((subject_summary["session_count"] > 1).sum()),
        "selection_strategy": "bundle_reference_relative_manifest",
        "notes": [
            "This runtime manifest was materialized directly from backend_reference/oasis2_session_manifest_relative.csv.",
            "Using the bundle reference manifest avoids Drive folder-rescan drift during remote OASIS-2 runs.",
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return manifest_path, longitudinal_records_path, subject_summary_path, summary_path


def _build_blocked_reason(readiness_payload: dict[str, Any]) -> str:
    """Collapse failing readiness checks into one compact explanation string."""

    failed_messages = [
        f"{check.get('name')}: {check.get('message')}"
        for check in readiness_payload.get("checks", [])
        if check.get("status") == "fail"
    ]
    if failed_messages:
        return " | ".join(failed_messages)
    return f"OASIS-2 training readiness did not pass: {readiness_payload.get('overall_status')}"


def _write_summary_files(*, outputs_root: Path, summary: dict[str, Any], run_root: Path | None = None) -> tuple[Path, Path]:
    """Persist a compact JSON/Markdown summary for the Colab OASIS-2 pipeline."""

    if run_root is None:
        summary_root = outputs_root / "reports" / "onboarding"
        file_stem = "oasis2_colab_bundle_summary"
    else:
        summary_root = run_root / "reports"
        file_stem = "colab_run_summary"

    summary_root.mkdir(parents=True, exist_ok=True)
    json_path = summary_root / f"{file_stem}.json"
    md_path = summary_root / f"{file_stem}.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# OASIS-2 Colab Summary",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- bundle_root: {summary.get('bundle_root')}",
        f"- runtime_data_root: {summary.get('runtime_data_root')}",
        f"- runtime_outputs_root: {summary.get('runtime_outputs_root')}",
        f"- training_ready: {summary.get('training_ready')}",
        f"- training_started: {summary.get('training_started')}",
        f"- readiness_status: {summary.get('readiness_status')}",
    ]
    if summary.get("run_name"):
        lines.append(f"- run_name: {summary.get('run_name')}")
    if summary.get("run_root"):
        lines.append(f"- run_root: {summary.get('run_root')}")
    if summary.get("blocked_reason"):
        lines.extend(["", "## Blocked Reason", "", f"- {summary['blocked_reason']}"])
    recommendations = summary.get("recommendations") or []
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"- {item}" for item in recommendations)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _refresh_runtime_from_source(
    *,
    source_root: Path,
    metadata_source_path: Path,
    file_stem_suffix: str,
    demographics_path: Path | None = None,
    auto_fill_from_official_demographics: bool = True,
    official_demographics_url: str = OASIS2_OFFICIAL_DEMOGRAPHICS_URL,
) -> OASIS2RuntimeRefreshResult:
    """Rebuild OASIS-2 runtime artifacts from the selected source root."""

    resolved_source_root = source_root.expanduser().resolve()
    os.environ["ALZ_OASIS2_SOURCE_DIR"] = str(resolved_source_root)
    get_app_settings.cache_clear()
    settings = get_app_settings()

    runtime_metadata_template_path = _copy_metadata_template_to_runtime(
        source_path=metadata_source_path,
        data_root=settings.data_root,
    )
    official_demographics_path: Path | None = None
    official_demographics_import_json_path: Path | None = None
    official_demographics_import_md_path: Path | None = None
    official_demographics_import_payload: dict[str, Any] | None = None

    if auto_fill_from_official_demographics and not _runtime_template_has_candidate_labels(
        metadata_template_path=runtime_metadata_template_path,
        settings=settings,
    ):
        official_demographics_path = _resolve_official_demographics_source(
            project_root=settings.project_root,
            bundle_root=resolved_source_root,
            outputs_root=settings.outputs_root,
            override_path=demographics_path,
            demographics_url=official_demographics_url,
        )
        demographics_summary = import_oasis2_official_demographics_into_metadata_template(
            official_demographics_path,
            settings=settings,
            metadata_path=runtime_metadata_template_path,
            output_path=runtime_metadata_template_path,
            metadata_source_name=f"official_oasis2_demographics:{official_demographics_path.name}",
        )
        (
            official_demographics_import_json_path,
            official_demographics_import_md_path,
        ) = save_oasis2_official_demographics_import_summary(
            demographics_summary,
            settings=settings,
            file_stem=f"oasis2_official_demographics_import_{file_stem_suffix}",
        )
        official_demographics_import_payload = demographics_summary.to_payload()

    relative_manifest_path = resolved_source_root / "backend_reference" / "oasis2_session_manifest_relative.csv"
    if relative_manifest_path.exists():
        (
            manifest_path,
            longitudinal_records_path,
            subject_summary_path,
            manifest_summary_path,
        ) = _materialize_runtime_manifest_from_bundle_reference(
            bundle_root=resolved_source_root,
            data_root=settings.data_root,
        )
        manifest_source = "bundle_reference_relative_manifest"
    else:
        manifest_result = build_oasis2_session_manifest(
            settings=settings,
            source_root=resolved_source_root,
        )
        manifest_path = manifest_result.manifest_path
        longitudinal_records_path = manifest_result.longitudinal_records_path
        subject_summary_path = manifest_result.subject_summary_path
        manifest_summary_path = manifest_result.summary_path
        manifest_source = "raw_bundle_rescan"

    metadata_summary = merge_oasis2_metadata_template(
        settings=settings,
        metadata_path=runtime_metadata_template_path,
    )
    metadata_json_path, metadata_md_path = save_oasis2_metadata_adapter_summary(
        metadata_summary,
        settings=settings,
        file_stem=f"oasis2_metadata_adapter_status_{file_stem_suffix}",
    )
    split_summary = build_oasis2_subject_safe_split_plan(
        settings=settings,
        metadata_path=runtime_metadata_template_path,
    )
    readiness_report = build_oasis2_training_readiness_report(settings=settings)
    readiness_json_path, readiness_md_path = save_oasis2_training_readiness_report(
        readiness_report,
        settings=settings,
        file_stem=f"oasis2_training_readiness_{file_stem_suffix}",
    )

    return OASIS2RuntimeRefreshResult(
        source_root=resolved_source_root,
        manifest_source=manifest_source,
        manifest_path=manifest_path,
        longitudinal_records_path=longitudinal_records_path,
        subject_summary_path=subject_summary_path,
        manifest_summary_path=manifest_summary_path,
        metadata_template_path=runtime_metadata_template_path,
        official_demographics_path=official_demographics_path,
        official_demographics_import_json_path=official_demographics_import_json_path,
        official_demographics_import_md_path=official_demographics_import_md_path,
        official_demographics_import_payload=official_demographics_import_payload,
        metadata_adapter_json_path=metadata_json_path,
        metadata_adapter_md_path=metadata_md_path,
        split_plan_path=Path(split_summary.plan_csv_path),
        training_readiness_json_path=readiness_json_path,
        training_readiness_md_path=readiness_md_path,
        readiness_payload=readiness_report.to_payload(),
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the OASIS-2 Colab parser by extending the local trainer parser."""

    parser = build_train_parser()
    parser.description = (
        "Run the OASIS-2 training gate directly from an uploaded Drive bundle. "
        "This validates the bundle, copies the metadata template into the runtime, "
        "rebuilds manifests from the bundle, and starts supervised training only when "
        "label coverage and subject-safe split checks truly pass."
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--demographics-path", type=Path, default=None)
    parser.add_argument(
        "--official-demographics-url",
        type=str,
        default=OASIS2_OFFICIAL_DEMOGRAPHICS_URL,
        help="Official OASIS-2 demographics spreadsheet URL used to auto-fill blank runtime metadata templates.",
    )
    parser.add_argument(
        "--auto-fill-from-official-demographics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the uploaded metadata template is blank, auto-fill the runtime copy from the official OASIS-2 demographics sheet.",
    )
    parser.add_argument("--stage-bundle-to-local", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-bundle-root", type=Path, default=Path("/content/oasis2_stage/OASIS-2"))
    parser.add_argument("--force-restage", action="store_true")
    parser.add_argument(
        "--require-training-ready",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit nonzero when the bundle is valid but labels or split readiness are still incomplete.",
    )
    return parser


def run_oasis2_colab_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run the OASIS-2 bundle pipeline and return a JSON-safe summary payload."""

    _mount_drive_if_requested(bool(getattr(args, "mount_drive", False)))

    project_root = args.project_root.expanduser().resolve() if args.project_root is not None else PROJECT_ROOT
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    bundle_root = args.bundle_root.expanduser().resolve()
    if not bundle_root.exists():
        raise FileNotFoundError(f"OASIS-2 bundle root not found: {bundle_root}")

    data_root, outputs_root = _configure_runtime_roots(project_root=project_root, runtime_root=args.runtime_root)
    get_app_settings.cache_clear()
    settings = get_app_settings()

    upload_report = inspect_oasis2_upload_bundle(settings=settings, bundle_root=bundle_root)
    upload_json_path, upload_md_path = save_oasis2_upload_bundle_report(
        upload_report,
        settings=settings,
        file_stem="oasis2_upload_bundle_status_from_runtime",
    )
    if upload_report.overall_status == "fail":
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bundle_root": str(bundle_root),
            "runtime_data_root": str(data_root),
            "runtime_outputs_root": str(outputs_root),
            "upload_bundle_status": upload_report.overall_status,
            "upload_bundle_report_json_path": str(upload_json_path),
            "upload_bundle_report_md_path": str(upload_md_path),
            "training_ready": False,
            "training_started": False,
            "readiness_status": None,
            "blocked_reason": "Uploaded OASIS-2 bundle validation failed.",
            "recommendations": upload_report.recommendations,
        }
        summary_json_path, summary_md_path = _write_summary_files(outputs_root=outputs_root, summary=summary)
        summary["summary_json_path"] = str(summary_json_path)
        summary["summary_md_path"] = str(summary_md_path)
        return summary

    metadata_source_path = _resolve_bundle_metadata_source(bundle_root, args.metadata_path)
    try:
        preflight_refresh = _refresh_runtime_from_source(
            source_root=bundle_root,
            metadata_source_path=metadata_source_path,
            file_stem_suffix="from_bundle",
            demographics_path=args.demographics_path,
            auto_fill_from_official_demographics=args.auto_fill_from_official_demographics,
            official_demographics_url=args.official_demographics_url,
        )
        active_refresh = preflight_refresh

        if preflight_refresh.readiness_payload.get("overall_status") == "pass" and args.stage_bundle_to_local:
            staged_root = _stage_bundle_to_local(
                bundle_root=bundle_root,
                stage_root=args.local_bundle_root,
                force_restage=args.force_restage,
            )
            active_refresh = _refresh_runtime_from_source(
                source_root=staged_root,
                metadata_source_path=metadata_source_path,
                file_stem_suffix="from_staged_bundle",
                demographics_path=args.demographics_path,
                auto_fill_from_official_demographics=args.auto_fill_from_official_demographics,
                official_demographics_url=args.official_demographics_url,
            )
    except Exception as error:
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bundle_root": str(bundle_root),
            "runtime_data_root": str(data_root),
            "runtime_outputs_root": str(outputs_root),
            "metadata_template_source_path": str(metadata_source_path),
            "training_ready": False,
            "training_started": False,
            "readiness_status": "fail",
            "blocked_reason": str(error),
            "recommendations": [
                "Inspect the official demographics import step and the saved runtime reports before retrying OASIS-2 training.",
                "If the runtime cannot reach the official demographics URL, pass --demographics-path with a local copy of the spreadsheet.",
            ],
        }
        summary_json_path, summary_md_path = _write_summary_files(outputs_root=outputs_root, summary=summary)
        summary["summary_json_path"] = str(summary_json_path)
        summary["summary_md_path"] = str(summary_md_path)
        if args.require_training_ready:
            raise SystemExit(str(error)) from error
        return summary

    readiness_status = active_refresh.readiness_payload.get("overall_status")
    recommendations = list(active_refresh.readiness_payload.get("recommendations", []))
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle_root),
        "bundle_source_for_checks": str(preflight_refresh.source_root),
        "source_root_for_training": str(active_refresh.source_root),
        "runtime_data_root": str(data_root),
        "runtime_outputs_root": str(outputs_root),
        "metadata_template_source_path": str(metadata_source_path),
        "runtime_metadata_template_path": str(active_refresh.metadata_template_path),
        "official_demographics_url": args.official_demographics_url if args.auto_fill_from_official_demographics else None,
        "official_demographics_path": None
        if active_refresh.official_demographics_path is None
        else str(active_refresh.official_demographics_path),
        "official_demographics_import_json_path": None
        if active_refresh.official_demographics_import_json_path is None
        else str(active_refresh.official_demographics_import_json_path),
        "official_demographics_import_md_path": None
        if active_refresh.official_demographics_import_md_path is None
        else str(active_refresh.official_demographics_import_md_path),
        "official_demographics_import_summary": active_refresh.official_demographics_import_payload,
        "upload_bundle_status": upload_report.overall_status,
        "upload_bundle_report_json_path": str(upload_json_path),
        "upload_bundle_report_md_path": str(upload_md_path),
        "manifest_path": str(active_refresh.manifest_path),
        "manifest_source": active_refresh.manifest_source,
        "manifest_summary_path": str(active_refresh.manifest_summary_path),
        "metadata_adapter_json_path": str(active_refresh.metadata_adapter_json_path),
        "metadata_adapter_md_path": str(active_refresh.metadata_adapter_md_path),
        "split_plan_path": str(active_refresh.split_plan_path),
        "training_readiness_json_path": str(active_refresh.training_readiness_json_path),
        "training_readiness_md_path": str(active_refresh.training_readiness_md_path),
        "training_ready": readiness_status == "pass",
        "training_started": False,
        "readiness_status": readiness_status,
        "blocked_reason": None,
        "recommendations": recommendations,
        "training_readiness_summary": {
            "labeled_row_count": active_refresh.readiness_payload.get("dataset_summary", {}).get("labeled_row_count", 0),
            "unlabeled_row_count": active_refresh.readiness_payload.get("dataset_summary", {}).get("unlabeled_row_count", 0),
            "unique_subject_count": active_refresh.readiness_payload.get("dataset_summary", {}).get("unique_subject_count", 0),
            "unique_session_count": active_refresh.readiness_payload.get("dataset_summary", {}).get("unique_session_count", 0),
            "split_group_count": active_refresh.readiness_payload.get("dataset_summary", {}).get("split_group_count", 0),
        },
    }

    if readiness_status != "pass":
        summary["blocked_reason"] = _build_blocked_reason(active_refresh.readiness_payload)
        summary_json_path, summary_md_path = _write_summary_files(outputs_root=outputs_root, summary=summary)
        summary["summary_json_path"] = str(summary_json_path)
        summary["summary_md_path"] = str(summary_md_path)
        if args.require_training_ready:
            raise SystemExit(summary["blocked_reason"])
        return summary

    config_path = args.config or default_oasis2_train_config_path()
    cfg = load_research_oasis_training_config(config_path)
    resolved_device, default_mixed_precision = _resolve_training_device(args.device)
    args.device = resolved_device
    if args.mixed_precision is None:
        args.mixed_precision = default_mixed_precision
    cfg = apply_cli_overrides(cfg, args)

    try:
        result = run_research_oasis2_training(cfg, settings=get_app_settings())
    except ResearchTrainingError as error:
        summary["blocked_reason"] = str(error)
        summary_json_path, summary_md_path = _write_summary_files(outputs_root=outputs_root, summary=summary)
        summary["summary_json_path"] = str(summary_json_path)
        summary["summary_md_path"] = str(summary_md_path)
        raise SystemExit(str(error)) from error

    summary.update(
        {
            "training_started": True,
            "run_name": result.run_name,
            "run_root": str(result.run_root),
            "best_checkpoint": None if result.best_checkpoint_path is None else str(result.best_checkpoint_path),
            "last_checkpoint": None if result.last_checkpoint_path is None else str(result.last_checkpoint_path),
            "epoch_metrics_csv_path": str(result.epoch_metrics_csv_path),
            "epoch_metrics_json_path": str(result.epoch_metrics_json_path),
            "confusion_matrix_path": str(result.confusion_matrix_path),
            "summary_report_path": str(result.summary_report_path),
            "resolved_config_path": str(result.resolved_config_path),
            "best_epoch": result.best_epoch,
            "best_monitor_value": result.best_monitor_value,
            "stopped_early": result.stopped_early,
            "final_metrics": result.final_metrics,
        }
    )
    summary_json_path, summary_md_path = _write_summary_files(
        outputs_root=outputs_root,
        summary=summary,
        run_root=Path(result.run_root),
    )
    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_md_path"] = str(summary_md_path)
    return summary


def main() -> None:
    """Run the OASIS-2 Colab bundle pipeline."""

    args = build_parser().parse_args()
    summary = run_oasis2_colab_pipeline(args)
    print(f"runtime_data_root={summary['runtime_data_root']}")
    print(f"runtime_outputs_root={summary['runtime_outputs_root']}")
    print(f"bundle_root={summary['bundle_root']}")
    print(f"manifest_path={summary.get('manifest_path')}")
    print(f"metadata_template_source_path={summary.get('metadata_template_source_path')}")
    print(f"runtime_metadata_template_path={summary.get('runtime_metadata_template_path')}")
    if summary.get("official_demographics_path"):
        print(f"official_demographics_path={summary['official_demographics_path']}")
    print(f"training_ready={summary.get('training_ready')}")
    print(f"training_started={summary.get('training_started')}")
    print(f"readiness_status={summary.get('readiness_status')}")
    if summary.get("run_name"):
        print(f"run_name={summary['run_name']}")
    if summary.get("run_root"):
        print(f"run_root={summary['run_root']}")
    if summary.get("best_checkpoint"):
        print(f"best_checkpoint={summary['best_checkpoint']}")
    if summary.get("blocked_reason"):
        print(f"blocked_reason={summary['blocked_reason']}")
    print(f"summary_json_path={summary['summary_json_path']}")
    print(f"summary_md_path={summary['summary_md_path']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
