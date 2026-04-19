"""Check the current OASIS-2 metadata merge adapter status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2_metadata import merge_oasis2_metadata_template, save_oasis2_metadata_adapter_summary  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Merge the OASIS-2 metadata template with the unlabeled session manifest and report "
            "whether labeled-manifest preparation is actually ready."
        )
    )
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional OASIS-2 session manifest CSV override.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional OASIS-2 metadata template CSV override.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional merged manifest CSV output path override.")
    parser.add_argument("--file-stem", type=str, default="oasis2_metadata_adapter_status")
    parser.add_argument("--no-save", action="store_true", help="Print a summary without writing JSON/Markdown artifacts.")
    return parser


def main() -> None:
    """Build the merge summary and optionally save it."""

    args = build_parser().parse_args()
    summary = merge_oasis2_metadata_template(
        manifest_path=args.manifest_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
    )
    if not args.no_save:
        json_path, md_path = save_oasis2_metadata_adapter_summary(summary, file_stem=args.file_stem)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")

    print(f"metadata_path={summary.metadata_path}")
    print(f"merged_manifest_path={summary.merged_manifest_path}")
    print(f"record_count={summary.record_count}")
    print(f"matched_metadata_row_count={summary.matched_metadata_row_count}")
    print(f"rows_with_candidate_labels={summary.rows_with_candidate_labels}")
    print(f"ready_for_labeled_manifest={summary.ready_for_labeled_manifest}")
    print("summary=" + json.dumps(summary.to_payload(), indent=2))


if __name__ == "__main__":
    main()
