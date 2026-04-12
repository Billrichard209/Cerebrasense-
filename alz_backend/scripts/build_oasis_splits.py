"""Build subject-safe OASIS-1 train/val/test splits from the normalized manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.oasis1_splits import build_oasis1_splits


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build subject-safe OASIS-1 train/val/test splits.")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Run the split builder and print saved artifact paths."""

    parser = build_parser()
    args = parser.parse_args()
    result = build_oasis1_splits(
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        random_state=args.seed,
    )
    print(f"subject_splits={result.split_assignments_path}")
    print(f"train_manifest={result.train_manifest_path}")
    print(f"val_manifest={result.val_manifest_path}")
    print(f"test_manifest={result.test_manifest_path}")
    print(f"summary={result.summary_path}")
    print(f"train_rows={result.train_rows}")
    print(f"val_rows={result.val_rows}")
    print(f"test_rows={result.test_rows}")


if __name__ == "__main__":
    main()
