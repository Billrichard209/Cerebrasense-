"""Register a frozen OASIS benchmark manifest with a hash and summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.governance import register_benchmark  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Register a frozen OASIS benchmark manifest.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--benchmark-name", type=str, required=True)
    parser.add_argument("--split-name", type=str, default="test")
    return parser


def main() -> None:
    """Register one benchmark and print key artifact paths."""

    args = build_parser().parse_args()
    entry, output_path = register_benchmark(
        manifest_path=args.manifest_path,
        benchmark_name=args.benchmark_name,
        split_name=args.split_name,
    )
    print(f"benchmark_id={entry.benchmark_id}")
    print(f"sample_count={entry.sample_count}")
    print(f"subject_safe={entry.subject_safe}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
