"""Terminal reviewer workflow for queued and resolved backend review cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import ReviewResolutionRequest  # noqa: E402
from src.api.services import (  # noqa: E402
    build_pending_review_queue_payload,
    build_resolved_review_queue_payload,
    build_review_analytics_payload,
    build_review_dashboard_payload,
    build_review_detail_payload,
    build_review_learning_payload,
    resolve_review_queue_item_payload,
)
from src.configs.runtime_overrides import apply_runtime_database_override  # noqa: E402


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Operate the backend review queue from the terminal. "
            "This is intended for decision-support review workflows, not diagnosis."
        )
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help="Optional SQLite database path so review workflows can target an isolated demo store.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pending_parser = subparsers.add_parser("pending", help="List pending review cases.")
    pending_parser.add_argument("--limit", type=int, default=20)

    resolved_parser = subparsers.add_parser("resolved", help="List resolved review cases.")
    resolved_parser.add_argument("--limit", type=int, default=20)
    resolved_parser.add_argument("--status", type=str, default=None)

    show_parser = subparsers.add_parser("show", help="Show one review case by id.")
    show_parser.add_argument("--review-id", required=True)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve one review case.")
    resolve_parser.add_argument("--review-id", required=True)
    resolve_parser.add_argument("--reviewer-id", required=True)
    resolve_parser.add_argument(
        "--action",
        required=True,
        choices=["confirm_model_output", "override_prediction", "dismiss"],
    )
    resolve_parser.add_argument("--resolved-label", type=int, default=None)
    resolve_parser.add_argument("--resolved-label-name", type=str, default=None)
    resolve_parser.add_argument("--resolution-note", type=str, default=None)

    analytics_parser = subparsers.add_parser("analytics", help="Show review analytics.")
    analytics_parser.add_argument("--limit", type=int, default=200)
    analytics_parser.add_argument("--model-name", type=str, default=None)
    analytics_parser.add_argument("--active-model-only", action="store_true")

    learning_parser = subparsers.add_parser("learning", help="Show reviewer learning report.")
    learning_parser.add_argument("--limit", type=int, default=200)
    learning_parser.add_argument("--model-name", type=str, default=None)
    learning_parser.add_argument("--active-model-only", action="store_true")
    learning_parser.add_argument("--selection-metric", type=str, default="balanced_accuracy")
    learning_parser.add_argument("--threshold-step", type=float, default=0.05)

    dashboard_parser = subparsers.add_parser("dashboard", help="Show compact reviewer dashboard.")
    dashboard_parser.add_argument("--pending-limit", type=int, default=10)
    dashboard_parser.add_argument("--resolved-limit", type=int, default=10)
    dashboard_parser.add_argument("--history-limit", type=int, default=10)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    apply_runtime_database_override(args.database_path)

    if args.command == "pending":
        _print_json(build_pending_review_queue_payload(limit=args.limit))
        return
    if args.command == "resolved":
        _print_json(build_resolved_review_queue_payload(limit=args.limit, status=args.status))
        return
    if args.command == "show":
        _print_json(build_review_detail_payload(args.review_id))
        return
    if args.command == "resolve":
        payload = resolve_review_queue_item_payload(
            args.review_id,
            ReviewResolutionRequest(
                reviewer_id=args.reviewer_id,
                action=args.action,
                resolved_label=args.resolved_label,
                resolved_label_name=args.resolved_label_name,
                resolution_note=args.resolution_note,
            ),
        )
        _print_json(payload)
        return
    if args.command == "analytics":
        _print_json(
            build_review_analytics_payload(
                limit=args.limit,
                model_name=args.model_name,
                active_model_only=args.active_model_only,
            )
        )
        return
    if args.command == "learning":
        _print_json(
            build_review_learning_payload(
                limit=args.limit,
                model_name=args.model_name,
                active_model_only=args.active_model_only,
                selection_metric=args.selection_metric,
                threshold_step=args.threshold_step,
            )
        )
        return
    if args.command == "dashboard":
        _print_json(
            build_review_dashboard_payload(
                pending_limit=args.pending_limit,
                resolved_limit=args.resolved_limit,
                history_limit=args.history_limit,
            )
        )
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
