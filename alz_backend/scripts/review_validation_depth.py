"""Print a compact OASIS validation-depth dashboard from saved study artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services import build_validation_depth_payload  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Review saved OASIS multi-seed and repeated subject-safe studies to "
            "estimate current validation depth for the active model family."
        )
    )
    parser.add_argument("--limit", type=int, default=10)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = build_validation_depth_payload(limit=args.limit)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
