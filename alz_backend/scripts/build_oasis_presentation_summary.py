"""Build a presentation-ready OASIS status summary from compare-first productization artifacts."""

from __future__ import annotations

import argparse
import json
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


def _default_path(raw_path: Path | None, *, fallback: Path) -> Path:
    """Resolve one optional path with a fallback."""

    return (raw_path or fallback).expanduser().resolve()


@dataclass(slots=True)
class OASISPresentationSummary:
    """Presentation-ready project snapshot for the current OASIS productization state."""

    generated_at: str
    headline: str
    recommendation: str
    executive_summary: list[str]
    key_metrics: dict[str, Any]
    project_state: dict[str, Any]
    demo_assets: dict[str, Any]
    talking_points: list[str]
    risks_and_caveats: list[str]
    next_steps: list[str]
    source_artifacts: dict[str, str]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return asdict(self)


def build_oasis_presentation_summary(
    *,
    settings: AppSettings | None = None,
    comparison_report_path: Path | None = None,
    evidence_report_path: Path | None = None,
    productization_report_path: Path | None = None,
) -> OASISPresentationSummary:
    """Build a polished OASIS presentation summary from saved report artifacts."""

    resolved_settings = settings or get_app_settings()
    resolved_comparison_path = _default_path(
        comparison_report_path,
        fallback=resolved_settings.outputs_root / "reports" / "comparison" / "oasis_baseline_comparison.json",
    )
    resolved_evidence_path = _default_path(
        evidence_report_path,
        fallback=resolved_settings.outputs_root / "reports" / "evidence" / "scope_aligned_evidence_report.json",
    )
    resolved_productization_path = _default_path(
        productization_report_path,
        fallback=resolved_settings.outputs_root / "reports" / "productization" / "oasis_productization_status.json",
    )

    comparison_payload = _load_json(resolved_comparison_path)
    evidence_payload = _load_json(resolved_evidence_path)
    productization_payload = _load_json(resolved_productization_path)

    active = dict(comparison_payload.get("active", {}))
    candidate = dict(comparison_payload.get("candidate", {}))
    delta = dict(comparison_payload.get("delta", {}))
    recommendation = dict(comparison_payload.get("recommendation", {}))
    demo_bundles = dict(comparison_payload.get("demo_bundles", {}))
    repeated_splits = dict(evidence_payload.get("oasis_repeated_splits", {}))
    productization_summary = dict(productization_payload.get("summary", {}))

    active_test = dict(active.get("test_metrics", {}))
    candidate_test = dict(candidate.get("test_metrics", {}))
    active_threshold_test = dict(dict(active.get("threshold_calibration", {})).get("test_metrics", {}))
    candidate_threshold_test = dict(dict(candidate.get("threshold_calibration", {})).get("test_metrics", {}))

    headline = (
        "Keep the stronger local OASIS baseline active while retaining the imported Colab v3 as a validated candidate."
    )
    executive_summary = [
        (
            f"The current active local OASIS baseline `{active.get('run_name')}` remains the safer default because its held-out "
            f"test AUROC ({active_test.get('auroc')}) is stronger than the imported Colab candidate `{candidate.get('run_name')}` "
            f"({candidate_test.get('auroc')})."
        ),
        (
            f"The imported Colab candidate improved validation AUROC by {delta.get('validation_auroc')}, but it did not carry that gain "
            f"through to held-out test performance or review burden."
        ),
        (
            "The candidate was still imported locally and validated end to end, so cloud and local evidence are now connected even though "
            "the serving default was not changed."
        ),
    ]

    talking_points = [
        "OASIS remains the primary 3D structural MRI evidence track; Kaggle stays a separate 2D comparison branch.",
        "The compare-first workflow now lets us import a cloud-promoted candidate, validate it locally, and decide promotion without disrupting the active backend model.",
        "We now have side-by-side demo bundles for both the active baseline and the imported v3 candidate on the same local OASIS scan.",
        "The productization tooling is working: import, alignment check, comparison reporting, and demo bundle generation all run locally.",
    ]
    risks_and_caveats = [
        "Cloud and local are not using the same default model right now because the imported Colab v3 underperformed the active local baseline on held-out test AUROC.",
        "Review-monitoring evidence is still limited, so the governance story is stronger than before but not yet mature.",
        "These outputs are for research and decision support only, not diagnosis or clinical deployment claims.",
    ]
    next_steps = [
        "Keep the current local OASIS baseline active unless a future candidate beats it on held-out test performance without increasing review burden.",
        "Use the new comparison and presentation summary artifacts for demos, status updates, and portfolio-quality project communication.",
        "If more training happens before new data arrives, prioritize stability-focused work like repeated-split refreshes or error analysis rather than random hyperparameter sweeps.",
        "When ready, refresh the Kaggle branch as a secondary benchmark, then move to OASIS-2 onboarding as the next major expansion.",
    ]

    return OASISPresentationSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        headline=headline,
        recommendation=str(recommendation.get("action")),
        executive_summary=executive_summary,
        key_metrics={
            "active_test_auroc": active_test.get("auroc"),
            "candidate_test_auroc": candidate_test.get("auroc"),
            "delta_test_auroc": delta.get("test_auroc"),
            "active_threshold_test_f1": active_threshold_test.get("f1"),
            "candidate_threshold_test_f1": candidate_threshold_test.get("f1"),
            "delta_threshold_test_f1": delta.get("threshold_test_f1"),
            "active_review_required_count": active_test.get("review_required_count"),
            "candidate_review_required_count": candidate_test.get("review_required_count"),
            "delta_review_required_count": delta.get("test_review_required_count"),
            "repeated_split_test_auroc_mean": dict(repeated_splits.get("test_aggregate", {})).get("auroc", {}).get("mean"),
        },
        project_state={
            "active_run_name": active.get("run_name"),
            "candidate_run_name": candidate.get("run_name"),
            "productization_overall_status": productization_payload.get("overall_status"),
            "productization_pass_count": productization_summary.get("pass"),
            "productization_fail_count": productization_summary.get("fail"),
            "cloud_local_alignment_note": "Imported candidate is available locally, but the active local baseline intentionally remains the older stronger run.",
        },
        demo_assets={
            "active_bundle_root": dict(demo_bundles.get("active", {})).get("bundle_root"),
            "candidate_bundle_root": dict(demo_bundles.get("candidate", {})).get("bundle_root"),
            "active_prediction_json": dict(dict(demo_bundles.get("active", {})).get("prediction_output", {})).get("prediction_json"),
            "candidate_prediction_json": dict(dict(demo_bundles.get("candidate", {})).get("prediction_output", {})).get("prediction_json"),
        },
        talking_points=talking_points,
        risks_and_caveats=risks_and_caveats,
        next_steps=next_steps,
        source_artifacts={
            "comparison_report_path": str(resolved_comparison_path),
            "evidence_report_path": str(resolved_evidence_path),
            "productization_report_path": str(resolved_productization_path),
        },
    )


def save_oasis_presentation_summary(
    summary: OASISPresentationSummary,
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis_presentation_summary",
) -> tuple[Path, Path]:
    """Save JSON and Markdown presentation summary artifacts."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "presentation")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    payload = summary.to_payload()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# OASIS Presentation Summary",
        "",
        f"## Headline",
        "",
        summary.headline,
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.executive_summary)
    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            f"- active_test_auroc: {summary.key_metrics.get('active_test_auroc')}",
            f"- candidate_test_auroc: {summary.key_metrics.get('candidate_test_auroc')}",
            f"- delta_test_auroc: {summary.key_metrics.get('delta_test_auroc')}",
            f"- active_threshold_test_f1: {summary.key_metrics.get('active_threshold_test_f1')}",
            f"- candidate_threshold_test_f1: {summary.key_metrics.get('candidate_threshold_test_f1')}",
            f"- delta_threshold_test_f1: {summary.key_metrics.get('delta_threshold_test_f1')}",
            f"- active_review_required_count: {summary.key_metrics.get('active_review_required_count')}",
            f"- candidate_review_required_count: {summary.key_metrics.get('candidate_review_required_count')}",
            f"- delta_review_required_count: {summary.key_metrics.get('delta_review_required_count')}",
            f"- repeated_split_test_auroc_mean: {summary.key_metrics.get('repeated_split_test_auroc_mean')}",
            "",
            "## Talking Points",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary.talking_points)
    lines.extend(["", "## Risks And Caveats", ""])
    lines.extend(f"- {item}" for item in summary.risks_and_caveats)
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in summary.next_steps)
    lines.extend(["", "## Demo Assets", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary.demo_assets.items())
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build a presentation-ready OASIS summary from comparison and productization artifacts.")
    parser.add_argument("--comparison-report-path", type=Path, default=None)
    parser.add_argument("--evidence-report-path", type=Path, default=None)
    parser.add_argument("--productization-report-path", type=Path, default=None)
    return parser


def main() -> None:
    """Build the summary and print a compact result."""

    args = build_parser().parse_args()
    summary = build_oasis_presentation_summary(
        comparison_report_path=args.comparison_report_path,
        evidence_report_path=args.evidence_report_path,
        productization_report_path=args.productization_report_path,
    )
    json_path, md_path = save_oasis_presentation_summary(summary)
    print(f"json_report={json_path}")
    print(f"markdown_report={md_path}")
    print(f"recommendation={summary.recommendation}")
    print("summary=" + json.dumps(summary.to_payload(), indent=2))


if __name__ == "__main__":
    main()
