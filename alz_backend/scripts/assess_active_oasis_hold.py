"""Assess the active OASIS model for post-promotion operational hold status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.review_monitoring import assess_active_oasis_model_hold


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the active OASIS registry entry against post-promotion review-monitoring "
            "thresholds and update its operational status."
        )
    )
    parser.add_argument("--registry-path", type=Path, default=None, help="Optional path to the active model registry JSON.")
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=None,
        help="Optional YAML policy overriding configs/oasis_operational_hold.yaml.",
    )
    parser.add_argument(
        "--actor-id",
        default="manual_cli",
        help="Audit-friendly actor id recorded when the hold status changes.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = assess_active_oasis_model_hold(
        registry_path=args.registry_path,
        policy_config_path=args.policy_config,
        actor_id=args.actor_id,
    )
    print(json.dumps(
        {
            "operational_status": result.registry_entry.operational_status,
            "hold_applied": result.decision.hold_applied,
            "trigger_codes": result.decision.trigger_codes,
            "history_path": str(result.history_path),
            "registry_path": str(result.registry_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
