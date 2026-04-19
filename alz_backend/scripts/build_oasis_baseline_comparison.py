"""Build a compare-first report for the active local OASIS baseline vs an imported candidate."""

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
from src.models.registry import load_current_oasis_model_entry  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402
from scripts.build_oasis_demo_bundle import build_oasis_demo_bundle  # noqa: E402

DEFAULT_CANDIDATE_REGISTRY = "oasis_candidate_v3.json"


def _safe_name(value: str) -> str:
    """Return a path-safe report name."""

    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the comparison payload focused on the main held-out metrics."""

    if not metrics:
        return {}
    return {
        "sample_count": metrics.get("sample_count"),
        "accuracy": metrics.get("accuracy"),
        "auroc": metrics.get("auroc"),
        "sensitivity": metrics.get("sensitivity", metrics.get("recall_sensitivity")),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
        "review_required_count": metrics.get("review_required_count"),
    }


def _delta(candidate_value: Any, active_value: Any) -> float | None:
    """Compute a simple numeric delta when both values are numbers."""

    if not isinstance(candidate_value, (int, float)) or not isinstance(active_value, (int, float)):
        return None
    return float(candidate_value) - float(active_value)


def _resolve_registry_path(
    raw_path: Path | None,
    *,
    default_path: Path,
) -> Path:
    """Resolve one registry path with a default."""

    return (raw_path or default_path).expanduser().resolve()


def _build_model_summary(path: Path, settings: AppSettings) -> dict[str, Any]:
    """Load one registry and extract comparison-friendly summary fields."""

    entry = load_current_oasis_model_entry(path=path, settings=settings)
    payload = _load_json(path)
    return {
        "registry_path": str(path),
        "run_name": entry.run_name,
        "checkpoint_path": entry.checkpoint_path,
        "recommended_threshold": entry.recommended_threshold,
        "operational_status": payload.get("operational_status"),
        "validation_metrics": _metric_summary(dict(payload.get("validation_metrics", {}))),
        "test_metrics": _metric_summary(dict(payload.get("test_metrics", {}))),
        "threshold_calibration": {
            "selection_metric": dict(payload.get("threshold_calibration", {})).get("selection_metric"),
            "threshold": dict(payload.get("threshold_calibration", {})).get("threshold"),
            "test_metrics": _metric_summary(
                dict(dict(payload.get("threshold_calibration", {})).get("test_metrics", {}))
            ),
        },
        "review_monitoring": {
            "total_reviews": dict(payload.get("review_monitoring", {})).get("total_reviews"),
            "override_rate": dict(payload.get("review_monitoring", {})).get("override_rate"),
            "high_risk": dict(payload.get("review_monitoring", {})).get("high_risk"),
        },
    }


def _recommendation(active: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Generate a compare-first recommendation."""

    active_test = dict(active.get("test_metrics", {}))
    candidate_test = dict(candidate.get("test_metrics", {}))
    active_threshold_test = dict(dict(active.get("threshold_calibration", {})).get("test_metrics", {}))
    candidate_threshold_test = dict(dict(candidate.get("threshold_calibration", {})).get("test_metrics", {}))

    reasons: list[str] = []
    action = "keep_active"

    active_test_auroc = active_test.get("auroc")
    candidate_test_auroc = candidate_test.get("auroc")
    if isinstance(active_test_auroc, (int, float)) and isinstance(candidate_test_auroc, (int, float)):
        if candidate_test_auroc > active_test_auroc:
            reasons.append("Candidate beats the active baseline on raw held-out test AUROC.")
        else:
            reasons.append("Candidate trails the active baseline on raw held-out test AUROC.")

    active_threshold_f1 = active_threshold_test.get("f1")
    candidate_threshold_f1 = candidate_threshold_test.get("f1")
    if isinstance(active_threshold_f1, (int, float)) and isinstance(candidate_threshold_f1, (int, float)):
        if candidate_threshold_f1 > active_threshold_f1:
            reasons.append("Candidate improves threshold-calibrated test F1.")
        else:
            reasons.append("Candidate does not improve threshold-calibrated test F1.")

    active_reviews = active_test.get("review_required_count")
    candidate_reviews = candidate_test.get("review_required_count")
    if isinstance(active_reviews, (int, float)) and isinstance(candidate_reviews, (int, float)):
        if candidate_reviews <= active_reviews:
            reasons.append("Candidate does not increase review burden.")
        else:
            reasons.append("Candidate increases the number of review-required test cases.")

    candidate_better = (
        isinstance(active_test_auroc, (int, float))
        and isinstance(candidate_test_auroc, (int, float))
        and candidate_test_auroc > active_test_auroc
        and (
            not isinstance(active_reviews, (int, float))
            or not isinstance(candidate_reviews, (int, float))
            or candidate_reviews <= active_reviews
        )
    )
    if candidate_better:
        action = "promote_candidate"

    return {
        "action": action,
        "active_run_name": active.get("run_name"),
        "candidate_run_name": candidate.get("run_name"),
        "reasons": reasons,
    }


@dataclass(slots=True)
class OASISBaselineComparisonReport:
    """Structured compare-first report for active vs candidate OASIS baselines."""

    generated_at: str
    active: dict[str, Any]
    candidate: dict[str, Any]
    delta: dict[str, Any]
    recommendation: dict[str, Any]
    demo_bundles: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return asdict(self)


def build_oasis_baseline_comparison_report(
    *,
    settings: AppSettings | None = None,
    active_registry_path: Path | None = None,
    candidate_registry_path: Path | None = None,
    scan_path: Path | None = None,
    device: str = "cpu",
    output_name: str = "oasis_baseline_compare",
    build_demo_bundles: bool = False,
    skip_explanation: bool = False,
) -> OASISBaselineComparisonReport:
    """Build the active-vs-candidate OASIS baseline comparison report."""

    resolved_settings = settings or get_app_settings()
    resolved_active_registry_path = _resolve_registry_path(
        active_registry_path,
        default_path=resolved_settings.outputs_root / "model_registry" / "oasis_current_baseline.json",
    )
    resolved_candidate_registry_path = _resolve_registry_path(
        candidate_registry_path,
        default_path=resolved_settings.outputs_root / "model_registry" / DEFAULT_CANDIDATE_REGISTRY,
    )
    if not resolved_active_registry_path.exists():
        raise FileNotFoundError(f"Active registry not found: {resolved_active_registry_path}")
    if not resolved_candidate_registry_path.exists():
        raise FileNotFoundError(f"Candidate registry not found: {resolved_candidate_registry_path}")

    active_summary = _build_model_summary(resolved_active_registry_path, resolved_settings)
    candidate_summary = _build_model_summary(resolved_candidate_registry_path, resolved_settings)
    delta = {
        "validation_auroc": _delta(
            candidate_summary["validation_metrics"].get("auroc"),
            active_summary["validation_metrics"].get("auroc"),
        ),
        "test_auroc": _delta(
            candidate_summary["test_metrics"].get("auroc"),
            active_summary["test_metrics"].get("auroc"),
        ),
        "test_accuracy": _delta(
            candidate_summary["test_metrics"].get("accuracy"),
            active_summary["test_metrics"].get("accuracy"),
        ),
        "threshold_test_f1": _delta(
            candidate_summary["threshold_calibration"]["test_metrics"].get("f1"),
            active_summary["threshold_calibration"]["test_metrics"].get("f1"),
        ),
        "test_review_required_count": _delta(
            candidate_summary["test_metrics"].get("review_required_count"),
            active_summary["test_metrics"].get("review_required_count"),
        ),
    }

    demo_bundles: dict[str, Any] = {}
    if build_demo_bundles:
        safe_output_name = _safe_name(output_name)
        demo_bundles = {
            "active": build_oasis_demo_bundle(
                settings=resolved_settings,
                scan_path=scan_path,
                registry_path=resolved_active_registry_path,
                device=device,
                output_name=f"{safe_output_name}_active",
                skip_explanation=skip_explanation,
            ),
            "candidate": build_oasis_demo_bundle(
                settings=resolved_settings,
                scan_path=scan_path,
                registry_path=resolved_candidate_registry_path,
                device=device,
                output_name=f"{safe_output_name}_candidate",
                skip_explanation=skip_explanation,
            ),
        }

    return OASISBaselineComparisonReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        active=active_summary,
        candidate=candidate_summary,
        delta=delta,
        recommendation=_recommendation(active_summary, candidate_summary),
        demo_bundles=demo_bundles,
    )


def save_oasis_baseline_comparison_report(
    report: OASISBaselineComparisonReport,
    settings: AppSettings | None = None,
    *,
    file_stem: str = "oasis_baseline_comparison",
) -> tuple[Path, Path]:
    """Save JSON and Markdown comparison reports."""

    resolved_settings = settings or get_app_settings()
    output_root = ensure_directory(resolved_settings.outputs_root / "reports" / "comparison")
    json_path = output_root / f"{file_stem}.json"
    md_path = output_root / f"{file_stem}.md"
    payload = report.to_payload()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    active = report.active
    candidate = report.candidate
    lines = [
        "# OASIS Baseline Comparison",
        "",
        f"- generated_at: {report.generated_at}",
        f"- active_run_name: {active.get('run_name')}",
        f"- candidate_run_name: {candidate.get('run_name')}",
        f"- recommendation: {report.recommendation.get('action')}",
        "",
        "## Held-Out Comparison",
        "",
        f"- active_test_auroc: {active.get('test_metrics', {}).get('auroc')}",
        f"- candidate_test_auroc: {candidate.get('test_metrics', {}).get('auroc')}",
        f"- delta_test_auroc: {report.delta.get('test_auroc')}",
        f"- active_test_accuracy: {active.get('test_metrics', {}).get('accuracy')}",
        f"- candidate_test_accuracy: {candidate.get('test_metrics', {}).get('accuracy')}",
        f"- delta_test_accuracy: {report.delta.get('test_accuracy')}",
        f"- active_threshold_test_f1: {active.get('threshold_calibration', {}).get('test_metrics', {}).get('f1')}",
        f"- candidate_threshold_test_f1: {candidate.get('threshold_calibration', {}).get('test_metrics', {}).get('f1')}",
        f"- delta_threshold_test_f1: {report.delta.get('threshold_test_f1')}",
        f"- active_review_required_count: {active.get('test_metrics', {}).get('review_required_count')}",
        f"- candidate_review_required_count: {candidate.get('test_metrics', {}).get('review_required_count')}",
        f"- delta_review_required_count: {report.delta.get('test_review_required_count')}",
        "",
        "## Recommendation",
        "",
    ]
    lines.extend(f"- {item}" for item in report.recommendation.get("reasons", []))
    if report.demo_bundles:
        lines.extend(["", "## Demo Bundles", ""])
        active_bundle = dict(report.demo_bundles.get("active", {}))
        candidate_bundle = dict(report.demo_bundles.get("candidate", {}))
        lines.append(f"- active_bundle_root: {active_bundle.get('bundle_root')}")
        lines.append(f"- candidate_bundle_root: {candidate_bundle.get('bundle_root')}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build a compare-first OASIS baseline report for the active model vs an imported candidate.")
    parser.add_argument("--active-registry-path", type=Path, default=None)
    parser.add_argument("--candidate-registry-path", type=Path, default=None)
    parser.add_argument("--scan-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-name", type=str, default="oasis_baseline_compare")
    parser.add_argument("--build-demo-bundles", action="store_true")
    parser.add_argument("--skip-explanation", action="store_true")
    return parser


def main() -> None:
    """Build the report and print a compact summary."""

    args = build_parser().parse_args()
    report = build_oasis_baseline_comparison_report(
        active_registry_path=args.active_registry_path,
        candidate_registry_path=args.candidate_registry_path,
        scan_path=args.scan_path,
        device=args.device,
        output_name=args.output_name,
        build_demo_bundles=args.build_demo_bundles,
        skip_explanation=args.skip_explanation,
    )
    json_path, md_path = save_oasis_baseline_comparison_report(report)
    print(f"json_report={json_path}")
    print(f"markdown_report={md_path}")
    print(f"recommendation={report.recommendation.get('action')}")
    print("summary=" + json.dumps(report.to_payload(), indent=2))


if __name__ == "__main__":
    main()
