"""Compare tracked OASIS experiments saved under outputs/experiments/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.runtime import get_app_settings  # noqa: E402
from src.training.experiment_tracking import load_experiment_rows  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Compare tracked OASIS experiments.")
    parser.add_argument("--experiments-root", type=Path, default=None)
    parser.add_argument("--sort-by", choices=["accuracy", "auroc", "sensitivity", "specificity"], default="auroc")
    return parser


def main() -> None:
    """Print a comparison table and highlight the best experiment."""

    args = build_parser().parse_args()
    settings = get_app_settings()
    experiments_root = args.experiments_root or (settings.outputs_root / "experiments")
    if not experiments_root.exists():
        raise FileNotFoundError(f"Experiments root not found: {experiments_root}")

    frame = load_experiment_rows(experiments_root)
    if frame.empty:
        raise ValueError(f"No experiment folders with comparison artifacts were found under {experiments_root}.")

    sorted_frame = frame.sort_values(by=[args.sort_by, "accuracy"], ascending=[False, False]).reset_index(drop=True)
    print(sorted_frame[["experiment_name", "accuracy", "auroc", "sensitivity", "specificity"]].to_string(index=False))
    best_row = sorted_frame.iloc[0]
    print(f"best_experiment={best_row['experiment_name']}")
    print(f"best_metric={args.sort_by}")
    print(f"best_value={float(best_row[args.sort_by]):.6f}")


if __name__ == "__main__":
    main()
