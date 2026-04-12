"""Build an OASIS-2 upload bundle for Drive or Colab use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis2_upload_bundle import build_oasis2_upload_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a portable OASIS-2 upload bundle from the unlabeled session manifest. "
            "This is intended for preprocessing and longitudinal preparation."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional OASIS-2 session manifest override.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional OASIS-2 root override. Point it at the parent folder that contains the OAS2 raw parts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output bundle directory.",
    )
    parser.add_argument(
        "--materialize-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize files in the bundle. Hardlink saves local storage when possible.",
    )
    return parser


def main() -> None:
    """Build the OASIS-2 upload bundle and print the saved artifact paths."""

    args = build_parser().parse_args()
    result = build_oasis2_upload_bundle(
        manifest_path=args.manifest_path,
        source_root=args.source_root,
        output_root=args.output_root,
        materialize_mode=args.materialize_mode,
    )
    print(f"bundle_root={result.bundle_root}")
    print(f"relative_manifest={result.relative_manifest_path}")
    print(f"session_index={result.session_index_path}")
    print(f"summary_json={result.summary_path}")
    print(f"included_session_count={result.included_session_count}")
    print(f"materialized_file_count={result.materialized_file_count}")
    print(f"missing_reference_count={result.missing_reference_count}")
    print(f"materialize_mode={result.materialize_mode}")


if __name__ == "__main__":
    main()
