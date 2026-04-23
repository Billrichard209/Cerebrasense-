"""Run registry-backed OASIS batch inference over a folder of scans."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import AppSettings, get_app_settings  # noqa: E402
from src.evaluation.calibration import ConfidenceBandConfig  # noqa: E402
from src.inference.pipeline import PredictScanOptions, predict_scan  # noqa: E402
from src.models.registry import load_current_oasis_model_entry  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402


def _safe_name(value: str) -> str:
    """Return a path-safe run name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _discover_scans(scan_root: Path, *, recursive: bool, pattern: str) -> list[Path]:
    """Find supported OASIS scan files under one root."""

    resolved_root = scan_root.expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Batch scan root not found: {resolved_root}")

    iterator = resolved_root.rglob(pattern) if recursive else resolved_root.glob(pattern)
    scans = sorted(path.resolve() for path in iterator if path.is_file())
    if not scans:
        raise FileNotFoundError(
            f"No scans matched pattern {pattern!r} under {resolved_root}. "
            "Provide a different --scan-root or --pattern."
        )
    return scans


def _infer_subject_session(scan_path: Path) -> tuple[str | None, str | None]:
    """Infer OASIS subject/session identifiers from a scan path when possible."""

    match = re.search(r"(OAS\d?_\d{4})_(MR\d+)", str(scan_path))
    if not match:
        return None, None
    subject_id = match.group(1)
    session_id = f"{subject_id}_{match.group(2)}"
    return subject_id, session_id


def _build_confidence_config(registry_entry: Any) -> ConfidenceBandConfig | None:
    """Build an optional confidence configuration from one registry entry."""

    if registry_entry is None:
        return None
    policy = dict(registry_entry.confidence_policy)
    scaling = dict(registry_entry.temperature_scaling)
    return ConfidenceBandConfig(
        temperature=float(scaling.get("temperature", 1.0)),
        high_confidence_min=float(policy.get("high_confidence_min", 0.85)),
        medium_confidence_min=float(policy.get("medium_confidence_min", 0.65)),
        high_entropy_max=float(policy.get("high_entropy_max", 0.35)),
        medium_entropy_max=float(policy.get("medium_entropy_max", 0.70)),
    )


def build_batch_oasis_predictions(
    *,
    scan_root: Path,
    settings: AppSettings | None = None,
    registry_path: Path | None = None,
    output_name: str = "oasis_batch_inference",
    device: str = "cpu",
    pattern: str = "*.hdr",
    recursive: bool = True,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    """Run batch OASIS inference and save a compact batch report."""

    resolved_settings = settings or get_app_settings()
    safe_output_name = _safe_name(output_name)
    report_root = ensure_directory(resolved_settings.outputs_root / "reports" / "batch_inference" / safe_output_name)
    scans = _discover_scans(scan_root, recursive=recursive, pattern=pattern)

    registry_entry = load_current_oasis_model_entry(registry_path, settings=resolved_settings)
    if not registry_entry.model_config_path or not registry_entry.preprocessing_config_path:
        raise FileNotFoundError("Active OASIS registry entry is missing model or preprocessing config paths.")

    confidence_config = _build_confidence_config(registry_entry)
    rows: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    review_required = 0

    for index, scan_path in enumerate(scans, start=1):
        subject_id, session_id = _infer_subject_session(scan_path)
        prediction_output_name = f"{safe_output_name}_{index:04d}_{scan_path.stem}"
        try:
            payload = predict_scan(
                str(scan_path),
                registry_entry.checkpoint_path,
                registry_entry.preprocessing_config_path,
                options=PredictScanOptions(
                    output_name=prediction_output_name,
                    threshold=float(registry_entry.recommended_threshold),
                    device=device,
                    model_config_path=Path(registry_entry.model_config_path),
                    subject_id=subject_id,
                    session_id=session_id,
                    confidence_config=confidence_config,
                ),
                settings=resolved_settings,
            )
            rows.append(
                {
                    "scan_path": str(scan_path),
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "status": "ok",
                    "predicted_label": payload.get("predicted_label"),
                    "label_name": payload.get("label_name"),
                    "probability_score": payload.get("probability_score"),
                    "confidence_score": payload.get("confidence_score"),
                    "confidence_level": payload.get("confidence_level"),
                    "review_flag": payload.get("review_flag"),
                    "prediction_json": payload.get("outputs", {}).get("prediction_json"),
                    "error": None,
                }
            )
            succeeded += 1
            if payload.get("review_flag"):
                review_required += 1
        except Exception as error:  # noqa: BLE001
            failed += 1
            rows.append(
                {
                    "scan_path": str(scan_path),
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "status": "error",
                    "predicted_label": None,
                    "label_name": None,
                    "probability_score": None,
                    "confidence_score": None,
                    "confidence_level": None,
                    "review_flag": None,
                    "prediction_json": None,
                    "error": str(error),
                }
            )
            if stop_on_error:
                raise

    predictions_frame = pd.DataFrame(rows)
    csv_path = report_root / "batch_predictions.csv"
    predictions_frame.to_csv(csv_path, index=False)

    label_counts = {
        str(key): int(value)
        for key, value in predictions_frame.loc[predictions_frame["status"] == "ok", "label_name"].value_counts(dropna=True).items()
    }
    summary = {
        "output_name": safe_output_name,
        "report_root": str(report_root),
        "scan_root": str(scan_root.expanduser().resolve()),
        "pattern": pattern,
        "recursive": recursive,
        "registry_path": str(registry_path.expanduser().resolve()) if registry_path is not None else None,
        "run_name": registry_entry.run_name,
        "checkpoint_path": registry_entry.checkpoint_path,
        "scan_count": len(scans),
        "succeeded": succeeded,
        "failed": failed,
        "review_required": review_required,
        "label_counts": label_counts,
        "batch_predictions_csv": str(csv_path),
        "notes": [
            "This batch run uses the requested or active OASIS registry entry for checkpoint, preprocessing, and threshold selection.",
            "Failures are recorded per scan so one bad file does not invalidate the whole batch unless --stop-on-error is used.",
        ],
    }
    summary_json_path = report_root / "batch_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# OASIS Batch Inference",
        "",
        f"- output_name: {safe_output_name}",
        f"- run_name: {registry_entry.run_name}",
        f"- scan_root: {summary['scan_root']}",
        f"- scan_count: {summary['scan_count']}",
        f"- succeeded: {succeeded}",
        f"- failed: {failed}",
        f"- review_required: {review_required}",
        f"- batch_predictions_csv: {csv_path}",
        "",
        "## Label Counts",
        "",
    ]
    if label_counts:
        md_lines.extend(f"- {label}: {count}" for label, count in sorted(label_counts.items()))
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Notes", ""])
    md_lines.extend(f"- {note}" for note in summary["notes"])
    summary_md_path = report_root / "batch_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_md_path"] = str(summary_md_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Run registry-backed OASIS batch inference over a folder of scans.")
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--output-name", type=str, default="oasis_batch_inference")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pattern", type=str, default="*.hdr")
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> None:
    """Run one batch OASIS prediction pass and print a compact summary."""

    args = build_parser().parse_args()
    summary = build_batch_oasis_predictions(
        scan_root=args.scan_root,
        registry_path=args.registry_path,
        output_name=args.output_name,
        device=args.device,
        pattern=args.pattern,
        recursive=not args.non_recursive,
        stop_on_error=args.stop_on_error,
    )
    print(f"report_root={summary['report_root']}")
    print(f"run_name={summary['run_name']}")
    print(f"scan_count={summary['scan_count']}")
    print(f"succeeded={summary['succeeded']}")
    print(f"failed={summary['failed']}")
    print(f"review_required={summary['review_required']}")
    print(f"batch_predictions_csv={summary['batch_predictions_csv']}")
    print("summary=" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
