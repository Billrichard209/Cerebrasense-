        """Build a smaller labeled OASIS upload bundle for Drive or Colab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis_upload_bundle import build_oasis_upload_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a smaller OASIS upload bundle that contains only the labeled "
            "sessions currently used by the backend manifest."
        )
    )
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional OASIS manifest override.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional output bundle directory.")
    parser.add_argument(
        "--materialize-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize files in the bundle. Hardlink saves space on the local machine.",
    )
    return parser


def main() -> None:
    """Build the OASIS upload bundle and print the saved artifact paths."""

    args = build_parser().parse_args()
    result = build_oasis_upload_bundle(
        manifest_path=args.manifest_path,
        output_root=args.output_root,
        materialize_mode=args.materialize_mode,
    )
    print(f"bundle_root={result.bundle_root}")
    print(f"oasis_subset_root={result.oasis_subset_root}")
    print(f"relative_manifest={result.relative_manifest_path}")
    print(f"session_index={result.session_index_path}")
    print(f"summary_json={result.summary_path}")
    print(f"included_session_count={result.included_session_count}")
    print(f"materialized_file_count={result.materialized_file_count}")
    print(f"missing_reference_count={result.missing_reference_count}")
    print(f"materialize_mode={result.materialize_mode}")


if __name__ == "__main__":
    main()
