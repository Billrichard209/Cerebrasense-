"""Build a raw OASIS-2 inventory from local split-part session folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2 import build_oasis2_raw_inventory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for OASIS-2 raw inventory generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a raw OASIS-2 inventory from local OAS2 split-part folders. "
            "This indexes sessions and structural volumes without inventing labels."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Optional OASIS-2 root override. You can point this at the parent folder that contains "
            "OAS2_RAW_PART1/OAS2_RAW_PART2 or at a specific OASIS-2 root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output folder. Defaults to alz_backend/data/interim.",
    )
    return parser


def main() -> None:
    """Build the inventory and print the saved artifact paths."""

    args = build_parser().parse_args()
    result = build_oasis2_raw_inventory(
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(f"inventory_csv={result.inventory_path}")
    print(f"dropped_rows_csv={result.dropped_rows_path}")
    print(f"summary_json={result.summary_path}")
    print(f"session_row_count={result.session_row_count}")
    print(f"unique_subject_count={result.unique_subject_count}")
    print(f"unique_session_count={result.unique_session_count}")
    print(f"part_roots={','.join(str(path) for path in result.part_roots)}")


if __name__ == "__main__":
    main()
