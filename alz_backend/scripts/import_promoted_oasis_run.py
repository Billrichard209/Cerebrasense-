"""Import a Drive/Colab-exported OASIS run into the local backend workspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import import_promoted_oasis_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Import a promoted OASIS run into local outputs and registry.")
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--source-registry-path", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--registry-output-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Run the import and print a compact summary."""

    args = build_parser().parse_args()
    result = import_promoted_oasis_run(
        source_run_root=args.source_run_root,
        source_registry_path=args.source_registry_path,
        run_name=args.run_name,
        registry_output_path=args.registry_output_path,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
