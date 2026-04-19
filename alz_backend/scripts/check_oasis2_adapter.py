"""Build and save the first dedicated OASIS-2 adapter status report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2_dataset import build_oasis2_adapter_summary, save_oasis2_adapter_summary  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Check the current OASIS-2 unlabeled manifest adapter status. "
            "This is the first dedicated adapter path and is intentionally limited "
            "to onboarding, structural workflows, and longitudinal preparation."
        )
    )
    parser.add_argument("--source-root", type=Path, default=None, help="Optional OASIS-2 source directory override.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional OASIS-2 session manifest CSV override.")
    parser.add_argument("--file-stem", type=str, default="oasis2_adapter_status")
    parser.add_argument("--no-save", action="store_true", help="Print a summary without writing JSON/Markdown artifacts.")
    return parser


def main() -> None:
    """Build the adapter summary and optionally save it."""

    args = build_parser().parse_args()
    summary = build_oasis2_adapter_summary(
        source_root=args.source_root,
        manifest_path=args.manifest_path,
    )
    if not args.no_save:
        json_path, md_path = save_oasis2_adapter_summary(summary, file_stem=args.file_stem)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")

    print(f"manifest_path={summary.manifest_path}")
    print(f"adapter_mode={summary.adapter_mode}")
    print(f"record_count={summary.record_count}")
    print(f"unique_subject_count={summary.unique_subject_count}")
    print(f"unique_session_count={summary.unique_session_count}")
    print(f"ready_for_supervised_training={summary.ready_for_supervised_training}")
    print("summary=" + json.dumps(summary.to_payload(), indent=2))


if __name__ == "__main__":
    main()
