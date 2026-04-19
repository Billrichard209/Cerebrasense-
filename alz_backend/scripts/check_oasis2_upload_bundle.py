"""Validate a built OASIS-2 upload bundle for remote review or Colab use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import get_app_settings  # noqa: E402
from src.data.oasis2_upload_bundle import (  # noqa: E402
    inspect_oasis2_upload_bundle,
    save_oasis2_upload_bundle_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Validate an extracted OASIS-2 upload bundle and save a status report.")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Optional OASIS-2 upload bundle root override. Defaults to alz_backend/outputs/exports/oasis2_upload_bundle.",
    )
    parser.add_argument("--file-stem", type=str, default="oasis2_upload_bundle_status")
    return parser


def main() -> None:
    """Run the validator and print a compact summary."""

    args = build_parser().parse_args()
    settings = get_app_settings()
    report = inspect_oasis2_upload_bundle(settings=settings, bundle_root=args.bundle_root)
    json_path, md_path = save_oasis2_upload_bundle_report(report, settings=settings, file_stem=args.file_stem)
    print(f"json_report={json_path}")
    print(f"markdown_report={md_path}")
    print(f"overall_status={report.overall_status}")
    print("summary=" + json.dumps(report.to_payload(), indent=2))


if __name__ == "__main__":
    main()
