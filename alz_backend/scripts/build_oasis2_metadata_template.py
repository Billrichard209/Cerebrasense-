"""Build the OASIS-2 metadata mapping template from the current session manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2_metadata import build_oasis2_metadata_template  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the OASIS-2 metadata mapping template from the current unlabeled session manifest. "
            "Use this before any labeled OASIS-2 integration work."
        )
    )
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional OASIS-2 session manifest CSV override.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional metadata template CSV output path override.")
    return parser


def main() -> None:
    """Build the template and print the saved artifact paths."""

    args = build_parser().parse_args()
    result = build_oasis2_metadata_template(
        manifest_path=args.manifest_path,
        output_path=args.output_path,
    )
    print(f"template_csv={result.template_path}")
    print(f"summary_json={result.summary_path}")
    print(f"row_count={result.row_count}")
    print(f"unique_subject_count={result.unique_subject_count}")
    print(f"unique_session_count={result.unique_session_count}")


if __name__ == "__main__":
    main()
