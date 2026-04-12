"""Print a compact OASIS promotion workflow dashboard from saved artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services import build_promotion_dashboard_payload  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review the active OASIS model against tracked promotion candidates and studies."
    )
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--study-limit", type=int, default=5)
    parser.add_argument("--history-limit", type=int, default=5)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = build_promotion_dashboard_payload(
        candidate_limit=args.candidate_limit,
        study_limit=args.study_limit,
        history_limit=args.history_limit,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
