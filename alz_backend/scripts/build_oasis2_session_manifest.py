"""Build an unlabeled OASIS-2 session manifest and longitudinal records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2 import build_oasis2_session_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for OASIS-2 session manifest generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Build an unlabeled OASIS-2 session manifest from the raw split-part dataset. "
            "This selects one representative acquisition per session for preprocessing and longitudinal preparation."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional OASIS-2 root override. You can point this at the parent folder that contains OAS2 raw parts.",
    )
    parser.add_argument(
        "--inventory-path",
        type=Path,
        default=None,
        help="Optional existing oasis2_raw_inventory.csv path to reuse.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output folder. Defaults to alz_backend/data/interim.",
    )
    return parser


def main() -> None:
    """Build the manifest and print the generated artifact locations."""

    args = build_parser().parse_args()
    result = build_oasis2_session_manifest(
        inventory_path=args.inventory_path,
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(f"manifest_csv={result.manifest_path}")
    print(f"longitudinal_records_csv={result.longitudinal_records_path}")
    print(f"subject_summary_csv={result.subject_summary_path}")
    print(f"summary_json={result.summary_path}")
    print(f"session_row_count={result.session_row_count}")
    print(f"unique_subject_count={result.unique_subject_count}")
    print(f"part_roots={','.join(str(path) for path in result.source_part_roots)}")


if __name__ == "__main__":
    main()
