"""Build a portable Kaggle upload bundle for Drive or Colab use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kaggle_upload_bundle import build_kaggle_upload_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Kaggle upload bundle generation."""

    parser = argparse.ArgumentParser(
        description="Build a portable Kaggle Alzheimer upload bundle for Google Drive or Colab."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional path to a Kaggle manifest CSV. Defaults to data/interim/kaggle_alz_manifest.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional destination folder. Defaults to outputs/exports/kaggle_alz_upload_bundle.",
    )
    parser.add_argument(
        "--materialize-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hardlinks when possible to avoid duplicating local storage.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the bundle and print the generated artifact locations."""

    args = parse_args()
    result = build_kaggle_upload_bundle(
        manifest_path=args.manifest_path,
        output_root=args.output_root,
        materialize_mode=args.materialize_mode,
    )
    print(f"bundle_root={result.bundle_root}")
    print(f"relative_manifest={result.relative_manifest_path}")
    print(f"file_index={result.file_index_path}")
    print(f"summary_json={result.summary_path}")
    print(f"included_subset_names={','.join(result.included_subset_names)}")
    print(f"materialized_file_count={result.materialized_file_count}")
    print(f"missing_reference_count={result.missing_reference_count}")
    print(f"materialize_mode={result.materialize_mode}")


if __name__ == "__main__":
    main()
