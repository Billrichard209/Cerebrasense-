"""Fill the OASIS-2 metadata template from the official longitudinal demographics sheet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2_metadata import (  # noqa: E402
    OASIS2_OFFICIAL_DEMOGRAPHICS_URL,
    import_oasis2_official_demographics_into_metadata_template,
    save_oasis2_official_demographics_import_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Fill the OASIS-2 metadata template from the official longitudinal demographics sheet. "
            "Binary session labels are derived from visit-level CDR."
        )
    )
    parser.add_argument(
        "--demographics-path",
        type=Path,
        required=True,
        help="Path to the official OASIS-2 longitudinal demographics spreadsheet.",
    )
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional OASIS-2 session manifest CSV override.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional OASIS-2 metadata template CSV override.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional filled metadata template CSV output path override.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Replace existing filled values instead of only filling blanks.")
    parser.add_argument(
        "--metadata-source-name",
        type=str,
        default=f"official_oasis2_demographics:{Path(OASIS2_OFFICIAL_DEMOGRAPHICS_URL).name}",
        help="Explicit metadata_source value to write into the filled template rows.",
    )
    parser.add_argument("--file-stem", type=str, default="oasis2_official_demographics_import")
    parser.add_argument("--no-save", action="store_true", help="Print the summary without writing JSON/Markdown reports.")
    return parser


def main() -> None:
    """Fill the template and print the saved artifact paths."""

    args = build_parser().parse_args()
    summary = import_oasis2_official_demographics_into_metadata_template(
        args.demographics_path,
        manifest_path=args.manifest_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        overwrite_existing=args.overwrite_existing,
        metadata_source_name=args.metadata_source_name,
    )
    if not args.no_save:
        json_path, md_path = save_oasis2_official_demographics_import_summary(summary, file_stem=args.file_stem)
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")

    print(f"demographics_path={summary.demographics_path}")
    print(f"metadata_path={summary.metadata_path}")
    print(f"matched_row_count={summary.matched_row_count}")
    print(f"labeled_row_count={summary.labeled_row_count}")
    print(f"converted_group_row_count={summary.converted_group_row_count}")
    print(f"group_cdr_disagreement_row_count={summary.group_cdr_disagreement_row_count}")
    print("summary=" + json.dumps(summary.to_payload(), indent=2))


if __name__ == "__main__":
    main()
