"""Build a normalized OASIS-1 manifest for the Alzheimer backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis1 import build_oasis1_manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build a normalized OASIS-1 manifest.")
    parser.add_argument(
        "--output-format",
        choices=("csv", "jsonl", "both"),
        default="csv",
        help="Manifest serialization format.",
    )
    return parser


def main() -> None:
    """Run the OASIS-1 manifest builder and print the saved artifacts."""

    parser = build_parser()
    args = parser.parse_args()
    result = build_oasis1_manifest(output_format=args.output_format)
    if result.manifest_csv_path:
        print(f"manifest_csv={result.manifest_csv_path}")
    if result.manifest_jsonl_path:
        print(f"manifest_jsonl={result.manifest_jsonl_path}")
    print(f"dropped_rows={result.dropped_rows_path}")
    print(f"manifest_rows={result.manifest_row_count}")
    print(f"dropped_rows_count={result.dropped_row_count}")


if __name__ == "__main__":
    main()
