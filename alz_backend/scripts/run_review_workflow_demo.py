"""Run a safe benchmark-proxy review workflow rehearsal on an isolated SQLite DB."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import ReviewResolutionRequest  # noqa: E402
from src.api.services import (  # noqa: E402
    build_pending_review_queue_payload,
    build_resolved_review_queue_payload,
    build_review_analytics_payload,
    build_review_dashboard_payload,
    build_review_learning_payload,
    resolve_review_queue_item_payload,
)
from src.configs.runtime import get_app_settings  # noqa: E402
from src.configs.runtime_overrides import temporary_runtime_database_override  # noqa: E402
from src.inference.pipeline import PredictScanOptions, predict_scan  # noqa: E402
from src.models.registry import load_current_oasis_model_entry  # noqa: E402
from src.utils.io_utils import ensure_directory  # noqa: E402
from src.utils.monai_utils import load_torch_symbols  # noqa: E402

PROXY_REVIEWER_ID = "benchmark_proxy_reviewer"
PROXY_NOTE = (
    "Benchmark-label proxy resolution for workflow rehearsal only. "
    "This is not a human clinical review and should not be mixed with live reviewer evidence."
)


@dataclass(slots=True, frozen=True)
class DemoCase:
    """One benchmark case selected for review-workflow rehearsal."""

    sample_id: str
    image_path: str
    true_label: int
    true_label_name: str | None
    predicted_label: int | None
    predicted_label_name: str | None
    confidence: float | None
    confidence_level: str | None
    review_flag: bool
    subject_id: str | None
    session_id: str | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay low-confidence benchmark cases through the active review workflow "
            "using an isolated database so the live governance evidence stays clean."
        )
    )
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=24,
        help="Maximum number of benchmark candidates to replay while searching for queued cases.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prefer-misclassified", action="store_true")
    parser.add_argument("--keep-pending", action="store_true")
    parser.add_argument("--save-debug-slices", action="store_true")
    parser.add_argument("--report-name", type=str, default="review_workflow_demo")
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help="Optional isolated SQLite path. Defaults to outputs/review_demo/<report_name>.sqlite3",
    )
    return parser


def _resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    torch = load_torch_symbols()["torch"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_predictions_csv(registry_entry: object) -> Path:
    evidence = dict(getattr(registry_entry, "evidence", {}))
    test_metrics_path = evidence.get("test_metrics_path")
    if test_metrics_path:
        candidate = Path(test_metrics_path).with_name("predictions.csv")
        if candidate.exists():
            return candidate
    run_name = getattr(registry_entry, "run_name", "")
    settings = get_app_settings()
    fallback = (
        settings.outputs_root
        / "runs"
        / "oasis"
        / run_name
        / "evaluation"
        / "post_train_test_best_model"
        / "predictions.csv"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not resolve the active model test predictions CSV for the review demo.")


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _select_demo_cases(
    predictions_frame: pd.DataFrame,
    *,
    limit: int,
    prefer_misclassified: bool,
) -> list[DemoCase]:
    required_columns = {"sample_id", "true_label", "meta_image_path"}
    missing_columns = required_columns - set(predictions_frame.columns)
    if missing_columns:
        raise ValueError(
            "Predictions CSV is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    frame = predictions_frame.copy()
    frame["review_flag"] = frame.get("review_flag", False).map(_coerce_bool)
    if "confidence" in frame.columns:
        frame["confidence_sort"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(1.0)
    else:
        frame["confidence_sort"] = 1.0

    review_frame = frame[frame["review_flag"]].copy()
    if review_frame.empty:
        review_frame = frame.copy()

    review_frame = review_frame.sort_values(by=["confidence_sort", "sample_id"], ascending=[True, True])
    review_frame = review_frame.drop_duplicates(subset=["sample_id"], keep="first")

    if prefer_misclassified and "predicted_label" in review_frame.columns:
        misclassified = review_frame[
            pd.to_numeric(review_frame["predicted_label"], errors="coerce")
            != pd.to_numeric(review_frame["true_label"], errors="coerce")
        ]
        correctly_classified = review_frame.drop(misclassified.index, errors="ignore")
        review_frame = pd.concat([misclassified, correctly_classified], ignore_index=True)

    selected_cases: list[DemoCase] = []
    for row in review_frame.head(limit).to_dict(orient="records"):
        selected_cases.append(
            DemoCase(
                sample_id=str(row.get("sample_id")),
                image_path=str(row.get("meta_image_path")),
                true_label=int(row.get("true_label")),
                true_label_name=None if pd.isna(row.get("true_label_name")) else str(row.get("true_label_name")),
                predicted_label=(
                    None if pd.isna(row.get("predicted_label")) else int(row.get("predicted_label"))
                ),
                predicted_label_name=(
                    None
                    if pd.isna(row.get("predicted_label_name"))
                    else str(row.get("predicted_label_name"))
                ),
                confidence=(
                    None if pd.isna(row.get("confidence")) else float(row.get("confidence"))
                ),
                confidence_level=(
                    None if pd.isna(row.get("confidence_level")) else str(row.get("confidence_level"))
                ),
                review_flag=_coerce_bool(row.get("review_flag")),
                subject_id=None if pd.isna(row.get("meta_subject_id")) else str(row.get("meta_subject_id")),
                session_id=None if pd.isna(row.get("meta_session_id")) else str(row.get("meta_session_id")),
            )
        )
    if not selected_cases:
        raise ValueError("No benchmark cases were available for the review-workflow rehearsal.")
    return selected_cases


def _build_demo_database_path(report_name: str) -> Path:
    settings = get_app_settings()
    return settings.outputs_root / "review_demo" / f"{report_name}.sqlite3"


def _build_report_root(report_name: str) -> Path:
    settings = get_app_settings()
    return ensure_directory(settings.outputs_root / "reports" / "review_workflow_demo" / report_name)


def _prediction_payload_for_case(
    case: DemoCase,
    *,
    registry_entry: object,
    device: str,
    save_debug_slices: bool,
    report_name: str,
) -> dict[str, Any]:
    return predict_scan(
        case.image_path,
        getattr(registry_entry, "checkpoint_path"),
        getattr(registry_entry, "preprocessing_config_path", None),
        options=PredictScanOptions(
            output_name=f"{report_name}_{case.sample_id}",
            device=device,
            model_config_path=(
                None
                if getattr(registry_entry, "model_config_path", None) is None
                else Path(getattr(registry_entry, "model_config_path"))
            ),
            save_debug_slices=save_debug_slices,
            subject_id=case.subject_id,
            session_id=case.session_id,
        ),
    )


def _proxy_resolution_request(case: DemoCase, prediction_payload: dict[str, Any]) -> ReviewResolutionRequest:
    predicted_label = int(prediction_payload["predicted_label"])
    if predicted_label == case.true_label:
        return ReviewResolutionRequest(
            reviewer_id=PROXY_REVIEWER_ID,
            action="confirm_model_output",
            resolution_note=PROXY_NOTE,
        )
    return ReviewResolutionRequest(
        reviewer_id=PROXY_REVIEWER_ID,
        action="override_prediction",
        resolved_label=case.true_label,
        resolved_label_name=case.true_label_name,
        resolution_note=PROXY_NOTE,
    )


def _save_demo_artifacts(
    *,
    report_root: Path,
    selected_cases: list[DemoCase],
    summary_payload: dict[str, Any],
) -> tuple[Path, Path]:
    selected_cases_path = report_root / "selected_cases.csv"
    pd.DataFrame([asdict(case) for case in selected_cases]).to_csv(selected_cases_path, index=False)
    summary_path = report_root / "workflow_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return selected_cases_path, summary_path


def main() -> None:
    args = _build_parser().parse_args()
    report_name = args.report_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    resolved_device = _resolve_device(args.device)
    report_root = _build_report_root(report_name)
    demo_database_path = (args.database_path or _build_demo_database_path(report_name)).expanduser().resolve()
    ensure_directory(demo_database_path.parent)

    with temporary_runtime_database_override(demo_database_path):
        registry_entry = load_current_oasis_model_entry(args.registry_path)
        predictions_csv = (args.predictions_csv or _default_predictions_csv(registry_entry)).expanduser().resolve()
        predictions_frame = pd.read_csv(predictions_csv)
        candidate_cases = _select_demo_cases(
            predictions_frame,
            limit=max(args.limit, args.max_candidates),
            prefer_misclassified=args.prefer_misclassified,
        )

        selected_cases: list[DemoCase] = []
        prediction_results: list[dict[str, Any]] = []
        queued_prediction_ids: list[str] = []
        for case in candidate_cases:
            payload = _prediction_payload_for_case(
                case,
                registry_entry=registry_entry,
                device=resolved_device,
                save_debug_slices=args.save_debug_slices,
                report_name=report_name,
            )
            selected_cases.append(case)
            prediction_results.append(
                {
                    "sample_id": case.sample_id,
                    "prediction_id": payload["prediction_id"],
                    "trace_id": payload["trace_id"],
                    "predicted_label": payload["predicted_label"],
                    "label_name": payload["label_name"],
                    "probability_score": payload["probability_score"],
                    "confidence_level": payload["confidence_level"],
                    "review_flag": payload["review_flag"],
                    "prediction_json": payload["outputs"]["prediction_json"],
                }
            )
            if payload["review_flag"]:
                queued_prediction_ids.append(payload["prediction_id"])
            if len(queued_prediction_ids) >= args.limit:
                break

        pending_payload = build_pending_review_queue_payload(limit=max(args.limit * 2, 20))
        pending_by_inference_id = {
            item["inference_id"]: item
            for item in pending_payload["items"]
        }

        resolution_results: list[dict[str, Any]] = []
        if not args.keep_pending:
            for case, prediction in zip(selected_cases, prediction_results, strict=True):
                if not prediction["review_flag"]:
                    resolution_results.append(
                        {
                            "sample_id": case.sample_id,
                            "status": "skipped_not_queued",
                            "note": "Prediction did not enter the review queue during rehearsal.",
                        }
                    )
                    continue
                pending_item = pending_by_inference_id.get(prediction["prediction_id"])
                if pending_item is None:
                    resolution_results.append(
                        {
                            "sample_id": case.sample_id,
                            "status": "skipped_missing_queue_item",
                            "note": "Expected queued review item was not found in the isolated demo store.",
                        }
                    )
                    continue
                resolution_request = _proxy_resolution_request(case, prediction)
                resolved_payload = resolve_review_queue_item_payload(
                    pending_item["review_id"],
                    resolution_request,
                )
                resolution_results.append(
                    {
                        "sample_id": case.sample_id,
                        "review_id": resolved_payload["review_id"],
                        "status": resolved_payload["status"],
                        "action": resolution_request.action,
                    }
                )

        final_pending = build_pending_review_queue_payload(limit=max(args.limit * 2, 20))
        final_resolved = build_resolved_review_queue_payload(limit=max(args.limit * 2, 20))
        analytics = build_review_analytics_payload(limit=200, active_model_only=True)
        learning = build_review_learning_payload(limit=200, active_model_only=True)
        dashboard = build_review_dashboard_payload(
            pending_limit=max(args.limit * 2, 20),
            resolved_limit=max(args.limit * 2, 20),
            history_limit=10,
        )

        summary_payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_name": report_name,
            "demo_database_path": str(demo_database_path),
            "predictions_csv": str(predictions_csv),
            "device": resolved_device,
            "candidate_budget": args.max_candidates,
            "attempted_cases": len(selected_cases),
            "queued_case_target": args.limit,
            "queued_case_hits": len(queued_prediction_ids),
            "proxy_resolution": not args.keep_pending,
            "notes": [
                PROXY_NOTE,
                "This rehearsal uses an isolated SQLite database so the live backend review evidence remains unchanged.",
                "Benchmark cases were replayed through the live inference pathway until the requested number of actually queued cases was reached or the candidate budget was exhausted.",
            ],
            "selected_cases": [asdict(case) for case in selected_cases],
            "prediction_results": prediction_results,
            "resolution_results": resolution_results,
            "pending_reviews": final_pending,
            "resolved_reviews": final_resolved,
            "analytics": analytics,
            "learning": learning,
            "dashboard": dashboard,
        }
        selected_cases_path, summary_path = _save_demo_artifacts(
            report_root=report_root,
            selected_cases=selected_cases,
            summary_payload=summary_payload,
        )

    print(f"demo_database={demo_database_path}")
    print(f"selected_cases_csv={selected_cases_path}")
    print(f"summary_json={summary_path}")
    print(f"queued_cases={summary_payload['pending_reviews']['total']}")
    print(f"resolved_cases={summary_payload['resolved_reviews']['total']}")
    print(f"recommended_action={summary_payload['learning']['recommended_action']}")


if __name__ == "__main__":
    main()
