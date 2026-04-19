"""Build the first subject-safe OASIS-2 split-plan preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2_split_policy import build_oasis2_subject_safe_split_plan  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a planning-only subject-safe split preview for OASIS-2. "
            "This is deterministic and patient-safe, but it is not yet a supervised training split."
        )
    )
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional OASIS-2 metadata template CSV override.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional subject-safe split-plan CSV output override.")
    parser.add_argument("--bucket-count", type=int, default=5)
    return parser


def main() -> None:
    """Build the split plan and print a compact summary."""

    args = build_parser().parse_args()
    summary = build_oasis2_subject_safe_split_plan(
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        bucket_count=args.bucket_count,
    )
    print(f"metadata_path={summary.metadata_path}")
    print(f"plan_csv_path={summary.plan_csv_path}")
    print(f"subject_count={summary.subject_count}")
    print(f"bucket_count={summary.bucket_count}")
    print("summary=" + json.dumps(summary.to_payload(), indent=2))


if __name__ == "__main__":
    main()
